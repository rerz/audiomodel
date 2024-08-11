use std::marker::PhantomData;

use burn::backend::autodiff::ops::Backward;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::Dataset;
use burn::data::dataset::InMemDataset;
use burn::module::{AutodiffModule, Module, ModuleDisplay};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::optim::AdamWConfig;
use burn::prelude::{Backend, Int};
use burn::tensor::{Bool, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use burn::train::metric::{Adaptor, LossInput, LossMetric};
use itertools::Itertools;
use num_traits::real::Real;
use parquet::data_type::AsBytes;
use parquet::record::{Row, RowAccessor};
use rand::Rng;

use crate::data::{AudioBatch, AudioBatcher, AudioItem, AudioItemVec};
use crate::mine::get_negative_samples;
use crate::model::{AudioModel, AudioModelConfig, AudioModelInput};
use crate::model::encoder::{Encoder, EncoderConfig};
use crate::model::encoder::transformer::TransformerEncoderConfig;
use crate::model::extractor::FeatureExtractorConfig;
use crate::model::projection::FeatureProjectionConfig;
use crate::model::quantizer::{Quantizer, QuantizerConfig};
use crate::model::quantizer::gumbel::GumbelQuantizerConfig;
use crate::util::download_hf_dataset;

#[derive(Config)]
pub struct PretrainConfig {
    pub model_config: AudioModelConfig,
    pub feature_dropout: f32,
    pub projected_size: usize,
}

impl PretrainConfig {
    pub fn init<
        B: Backend,
        EC: EncoderConfig,
        QC: QuantizerConfig,
    >(
        self,
        input_len: usize,
        encoder_config: EC,
        quantizer_config: QC,
        device: &B::Device,
    ) -> Pretrain<B, EC::Model<B>, QC::Model<B>>
    {
        let hidden_size = self.model_config.hidden_size;
        let vector_dim = quantizer_config.quantized_dim();
        let last_conv_dim = self.model_config.feature_extractor_config.last_conv_dim();

        Pretrain {
            model: self.model_config.init(input_len, encoder_config, device),
            quantizer: <QC::Model<B> as Quantizer<B>>::new(last_conv_dim, quantizer_config, device),
            feature_dropout: DropoutConfig::new(self.feature_dropout as f64).init(),
            project_hidden: LinearConfig::new(hidden_size, self.projected_size)
                .init(device),
            project_quantized: LinearConfig::new(vector_dim, self.projected_size)
                .init(device),
            loss: CrossEntropyLossConfig::new().with_pad_tokens(Some(vec![5])).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Pretrain<B: Backend, E, Q> {
    model: AudioModel<B, E>,
    feature_dropout: Dropout,
    quantizer: Q,
    project_hidden: Linear<B>,
    project_quantized: Linear<B>,
    loss: CrossEntropyLoss<B>,
}


fn l2<B: Backend, const D: usize>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let squared = tensor.powi_scalar(2);
    let summed = squared.sum_dim(dim);
    let norm = summed.sqrt();
    norm
}

fn cosine_similarity<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let dot = Tensor::sum_dim(a.clone() * b.clone(), dim);
    let norm_a = l2(a, dim);
    let norm_b = l2(b, dim);

    let sim = dot / (norm_a * norm_b);

    sim
}

fn contrastive_logits<B: Backend>(
    target_features: Tensor<B, 4>,
    negative_features: Tensor<B, 4>,
    predicted_features: Tensor<B, 3>,
    temperature: f32,
) -> Tensor<B, 4> {
    let target_features = Tensor::cat(vec![target_features, negative_features], 0);

    let logits = cosine_similarity(predicted_features.unsqueeze_dim(0), target_features, 2);

    logits / temperature
}

pub struct PretrainInput<B: Backend> {
    input_values: Tensor<B, 2>,
    attention_mask: Tensor<B, 2, Int>,
    masked_time_indices: Tensor<B, 2, Int>,
    sampled_negatives: Tensor<B, 3, Int>,
}

pub struct PretrainOutput<B: Backend> {
    _phantom: PhantomData<B>,
}

impl<B: Backend, Q: Quantizer<B>, E: Encoder<B>> Pretrain<B, E, Q> {
    pub fn forward(
        &self,
        inputs: Tensor<B, 2>,
        attention_mask: Tensor<B, 2, Bool>,
        seq_lens: Vec<usize>,
        masked_time_steps: Tensor<B, 2, Bool>,
        sampled_negatives: Tensor<B, 3, Int>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [batch, seq] = inputs.dims();

        let (last_hidden, features) = self.model.forward(
            AudioModelInput {
                inputs,
                seq_lens: seq_lens.clone(),
                masked_time_steps: masked_time_steps.clone(),
            },
            device
        );

        let predicted_features = self.project_hidden.forward(last_hidden.clone());
        let predicted_features = predicted_features.unsqueeze_dim(0);

        let extracted_features = self.feature_dropout.forward(features);
        let (quantized_features, perplexity) = self
            .quantizer
            .forward::<true>(extracted_features, masked_time_steps.clone(), device);

        let quantized_features = self.project_quantized.forward(quantized_features);
        let quantized_features = quantized_features.unsqueeze_dim::<4>(0);

        let negative_features = get_negative_samples(sampled_negatives, quantized_features.clone());

        // quantized features: 1 x batch x ?? x mask_len
        // negative_features: num_negatives x batch x ?? x mask_len
        // projected_features: 1 x batch x ?? x mask_len

        let contrastive_logits = self.contrastive_logits(
            quantized_features,
            negative_features,
            predicted_features,
            2.0,
        );

        // constrastive_logits: (num_negatives + 1) x batch x seq

        // TODO: neg is pos thing

        let mask_dims = masked_time_steps.dims();
        let logit_dims = contrastive_logits.dims();

        let logits = contrastive_logits
            .clone()
            .swap_dims(0, 2)
            .reshape([-1, contrastive_logits.dims()[0] as i32]); // (batch x hidden) x (num_negatives + 1)

        // batch_x_time x num_negatives + 1
        let logit_dims = logits.dims();


        let extracted_seq_len = self.model.feature_extractor.output_len(seq);

        // TODO: pytorch cross entropy loss ignores -100 target values, needs a workaround
        let target = (masked_time_steps.clone().bool_not().float() * 5)
            .int()
            .swap_dims(0, 1)
            .flatten(0, 1);

        let target_dims = target.dims();

        let num_code_vectors = self.quantizer.num_groups() * self.quantizer.num_vectors_per_group();

        let contrastive_loss = self.loss.forward(logits, target);//.sum();

        let perplexity = perplexity.to_data().to_vec::<f32>().unwrap()[0];
        let diversity_loss =
            masked_time_steps.clone().float().sum() * ((num_code_vectors as f32 - perplexity) / num_code_vectors as f32);

        let diversity_loss_weight = 0.1;
        let loss = contrastive_loss + diversity_loss * diversity_loss_weight;

        (last_hidden, loss)
    }

    fn contrastive_logits(
        &self,
        target_features: Tensor<B, 4>,
        negative_features: Tensor<B, 4>,
        predicted_features: Tensor<B, 4>,
        temperature: f32,
    ) -> Tensor<B, 3> {
        let target_features = Tensor::cat(vec![target_features, negative_features], 0);

        let logits = cosine_similarity(predicted_features, target_features, 3);
        let logits = logits / temperature;

        logits.squeeze(3)
    }
}

fn decode_row(row: &Row) -> AudioItem {
    let group = row.get_group(0).unwrap();
    let mut audio = group.get_bytes(0).unwrap().as_bytes().to_vec();

    let audio = audio.chunks_mut(4).map(|chunk| {
        if chunk.len() != 4 {
            return 0.0;
        }

        f32::from_ne_bytes(chunk.try_into().unwrap())
    }).collect_vec();

    let audio = rand::thread_rng().sample_iter(rand::distributions::Uniform::new(-1.0, 1.0)).take(3000).collect_vec();

    //let (audio, sr) = crate::io::read_bytes::<Wav>(&audio);

    let song_id = row.get_long(1).unwrap() as usize;
    let genre_id = row.get_long(2).unwrap() as usize;

    let genre = row.get_string(3).unwrap().clone();
    let seq_len = audio.len();

    AudioItem {
        audio,
        song_id,
        genre_id,
        genre,
        sr: 16_000,
        seq_len,
    }
}

pub struct PretrainStepOutput<B: Backend> {
    hidden: Tensor<B, 3>,
    loss: Tensor<B, 1>
}

impl<B: Backend> Adaptor<LossInput<B>> for PretrainStepOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: AutodiffBackend, E: Encoder<B> + AutodiffModule<B> + ModuleDisplay, Q: Quantizer<B> + AutodiffModule<B> + ModuleDisplay> TrainStep<AudioBatch<B>, PretrainStepOutput<B>> for Pretrain<B, E, Q> where <E as AutodiffModule<B>>::InnerModule: ModuleDisplay, <Q as AutodiffModule<B>>::InnerModule: ModuleDisplay {
    fn step(&self, item: AudioBatch<B>) -> TrainOutput<PretrainStepOutput<B>> {
        let (hidden, loss) = self.forward(
            item.sequences,
            item.attention_mask,
            item.sequence_lens,
            item.masked_time_indices,
            item.sampled_negative_indices,
            &item.device,
        );

        TrainOutput::new(self, loss.backward(), PretrainStepOutput {
            loss,
            hidden,
        })
    }
}

impl<B: Backend, E: Encoder<B>, Q: Quantizer<B>> ValidStep<AudioBatch<B>, ()> for Pretrain<B, E, Q> {
    fn step(&self, item: AudioBatch<B>) -> () {
        ()
    }
}

pub fn pretrain<B: AutodiffBackend>(device: B::Device) {
    let hidden_size = 80;

    let feature_extractor_config = FeatureExtractorConfig {
        conv_dims: vec![1, 8, 16, 32, 64],
        conv_kernels: vec![3, 3, 2, 2],
        conv_strides: vec![2, 2, 2, 2],
    };

    let feature_projection_config = FeatureProjectionConfig {
        hidden_size,
        dropout: 0.0,
        last_conv_dim: feature_extractor_config.last_conv_dim(),
        layer_norm_eps: 1e-8,
    };

    let model_config = AudioModelConfig {
        hidden_size,
        feature_extractor_config,
        feature_projection_config,
    };

    let encoder_config = TransformerEncoderConfig {
        num_heads: 4,
        num_layers: 3,
        ff_intermediate_size: 100,
        ff_dropout: 0.0,
        attention_dropout: 0.0,
        num_posconv_groups: 20,
        num_posconv_embeddings: 50,
    };

    let quantizer_config = GumbelQuantizerConfig {
        num_groups: 4,
        vectors_per_group: 10,
        vector_dim: 100,
    };


    let pretraining_config = PretrainConfig {
        model_config: model_config.clone(),
        feature_dropout: 0.0,
        projected_size: 100,
    };


    let mut dataset = download_hf_dataset(&hf_hub::api::sync::Api::new().unwrap(), "lewtun/music_genres_small".into()).unwrap();
    let dataset = dataset.remove(0);

    let items = AudioItemVec::from_iter(dataset.into_iter().map(|row| decode_row(&row.unwrap())));

    let dataset = InMemDataset::new(items.into_iter().map(|i| i.to_owned()).collect_vec());
    let batcher = AudioBatcher::new(model_config.clone(), device.clone());
    let data_loader_train = DataLoaderBuilder::new(batcher)
        .batch_size(4)
        .num_workers(1)
        .build(dataset);


    let dataset = InMemDataset::new(items.into_iter().map(|i| i.to_owned()).collect_vec());
    let batcher = AudioBatcher::new(model_config, device.clone());
    let data_loader_test = DataLoaderBuilder::new(batcher)
        .batch_size(4)
        .num_workers(1)
        .build(dataset);

    let model = pretraining_config.init::<B, _, GumbelQuantizerConfig>(100_000, (), quantizer_config, &device);

    //let batch = data_loader_train.iter().next().unwrap();
    //model.forward(batch.sequences, batch.attention_mask, batch.sequence_lens, batch.masked_time_indices, batch.sampled_negative_indices, &device);

    let optimizer = AdamWConfig::new().init::<B, Pretrain<_, _, _>>();

    let learner = LearnerBuilder::<B, _, _, Pretrain<_, _, _>, _, _>::new(".out")
        .devices(vec![device])
        .metric_train(LossMetric::new())
        .num_epochs(500)
        .summary()
        .build(model, optimizer, 0.001);
    let trained = learner.fit(data_loader_train, data_loader_test);
}