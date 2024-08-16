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
use burn::prelude::{Backend, ElementConversion, Int};
use burn::tensor::{Bool, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use burn::train::metric::{Adaptor, LossInput, LossMetric, Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use itertools::Itertools;
use num_traits::real::Real;
use parquet::data_type::AsBytes;
use parquet::record::{Row, RowAccessor};
use rand::Rng;

use crate::data::{AudioBatch, AudioBatcher, AudioItem, AudioItemVec};
use crate::mine::get_negative_samples;
use crate::model::{AudioModel, AudioModelConfig, AudioModelInput};
use crate::model::encoder::{Encoder, EncoderConfig};
use crate::model::encoder::burn_transformer::BurnTransformer;
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


fn l2<B: Backend, const D: usize>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, 1> {
    let squared = tensor.powi_scalar(2);
    let summed = squared.sum();
    let norm = summed.sqrt();
    norm
}

fn cosine_similarity<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let dot = Tensor::sum_dim(a.clone() * b.clone(), dim);
    let norm_a = l2(a, dim).to_data().to_vec::<B::FloatElem>().unwrap()[0].elem::<f32>();
    let norm_b = l2(b, dim).to_data().to_vec::<B::FloatElem>().unwrap()[0].elem::<f32>();

    let norm_a = f32::max(norm_a, 1e-8);
    let norm_b = f32::max(norm_b, 1e-8);

    let sim = dot / (norm_a * norm_b);

    sim
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
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 1>) {
        let [batch, seq] = inputs.dims();

        let (last_hidden, features) = self.model.forward(
            AudioModelInput {
                inputs,
                seq_lens: seq_lens.clone(),
                masked_time_steps: masked_time_steps.clone(),
            },
            device,
        );


        let predicted_features = self.project_hidden.forward(last_hidden.clone());
        let predicted_features = predicted_features.unsqueeze_dim(0);

        let extracted_features = self.feature_dropout.forward(features);
        let (quantized_features, perplexity) = self
            .quantizer
            .forward::<true>(extracted_features, masked_time_steps.clone(), device);


        let quantized_features = self.project_quantized.forward(quantized_features);

        let negative_features = get_negative_samples(sampled_negatives, quantized_features.clone());

        let quantized_features = quantized_features.unsqueeze_dim::<4>(0);
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

        let logits = contrastive_logits
            .clone()
            .swap_dims(0, 2)
            .reshape([-1, contrastive_logits.dims()[0] as i32]); // (batch x hidden) x (num_negatives + 1)

        let unmasked_indices = masked_time_steps.clone().bool_not().nonzero()[1].clone();

        // ignoring indices in cross entropy loss seems broken so we just remove them from the tensor
        let unmasked_logits = Tensor::select(logits, 0, unmasked_indices.clone());

        // let target = masked_time_steps.clone().bool_not()
        //     .int()
        //     .swap_dims(0, 1)
        //     .flatten(0, 1);

        let target = Tensor::zeros([unmasked_logits.dims()[0]], device);

        //let unmasked_targets = Tensor::select(target, 0, unmasked_indices);

        //println!("{}", unmasked_targets);

        let num_code_vectors = self.quantizer.num_groups() * self.quantizer.num_vectors_per_group();

        let contrastive_loss = self.loss.forward(unmasked_logits, target);//.sum();

        let diversity_loss =
            masked_time_steps.clone().float().sum() * ((num_code_vectors as f32 - perplexity.clone().to_data().to_vec::<f32>().unwrap()[0]) / num_code_vectors as f32);

        let diversity_loss_weight = 0.1;
        let loss = contrastive_loss + diversity_loss * diversity_loss_weight;

        (last_hidden, loss, perplexity)
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

fn decode_row(row: Row) -> AudioItem {
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
    loss: Tensor<B, 1>,
    perplexity: Tensor<B, 1>,
}

impl<B: Backend> Adaptor<LossInput<B>> for PretrainStepOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

#[derive(Default)]
pub struct PerplexityMetric<B: Backend> {
    state: NumericMetricState,
    _phantom: PhantomData<B>
}

impl<B: Backend> Metric for PerplexityMetric<B> {
    const NAME: &'static str = "Perplexity";
    type Input = Tensor<B, 1>;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        let perplexity = item.to_data().to_vec::<f32>().unwrap()[0];
        self.state.update(perplexity as f64, 1, FormatOptions::new("Perplexity").precision(3))
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Adaptor<Tensor<B, 1>> for PretrainStepOutput<B> {
    fn adapt(&self) -> Tensor<B, 1> {
        self.perplexity.clone()
    }
}

impl<B: AutodiffBackend, E: Encoder<B> + AutodiffModule<B, InnerModule: ModuleDisplay> + ModuleDisplay, Q: Quantizer<B> + AutodiffModule<B, InnerModule: ModuleDisplay> + ModuleDisplay> TrainStep<AudioBatch<B>, PretrainStepOutput<B>> for Pretrain<B, E, Q> {
    fn step(&self, item: AudioBatch<B>) -> TrainOutput<PretrainStepOutput<B>> {
        let (hidden, loss, perplexity) = self.forward(
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
            perplexity
        })
    }
}

impl<B: Backend, E: Encoder<B>, Q: Quantizer<B>> ValidStep<AudioBatch<B>, ()> for Pretrain<B, E, Q> {
    fn step(&self, item: AudioBatch<B>) -> () {
        ()
    }
}

impl<B: Backend> Numeric for PerplexityMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

pub fn train_with_learner_builder() {}

pub fn train_with_custom_loop() {}

pub fn pretrain<B: AutodiffBackend>(device: B::Device) {
    let (pretrain_config, encoder_config, quantizer_config) = crate::config::small_music::small_music_config::<B>();

    let model = pretrain_config.clone().init(100_000, encoder_config, quantizer_config, &device);

    let (train_rows, test_rows) = download_hf_dataset(&hf_hub::api::sync::Api::new().unwrap(), "lewtun/music_genres".into()).unwrap();

    let train_items = AudioItemVec::from_iter(train_rows.map(decode_row).take(10_000));
    let test_items = AudioItemVec::from_iter(test_rows.map(decode_row).take(10_000));

    let dataset = InMemDataset::new(train_items.into_iter().map(|i| i.to_owned()).collect_vec());
    let batcher = AudioBatcher::<B>::new(pretrain_config.model_config.clone(), device.clone());
    let data_loader_train = DataLoaderBuilder::new(batcher)
        .batch_size(12)
        .num_workers(1)
        .build(dataset);


    let dataset = InMemDataset::new(test_items.into_iter().map(|i| i.to_owned()).collect_vec());
    let batcher = AudioBatcher::<B::InnerBackend>::new(pretrain_config.model_config, device.clone());
    let data_loader_test = DataLoaderBuilder::new(batcher)
        .batch_size(4)
        .num_workers(1)
        .build(dataset);

    TrainStep::step(&model, data_loader_train.iter().next().unwrap());

    let optimizer = AdamWConfig::new().init();

    let learner = LearnerBuilder::new(".out")
        .devices(vec![device])
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(PerplexityMetric::default())
        .num_epochs(500)
        .summary()
        .build(model, optimizer, 0.00001);
    let trained = learner.fit(data_loader_train, data_loader_test);
}