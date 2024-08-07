use std::marker::PhantomData;

use burn::backend::{Autodiff, LibTorch};
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::NodeID;
use burn::backend::autodiff::ops::{Backward, broadcast_shape, Ops, OpsKind};
use burn::backend::libtorch::TchTensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::optim::AdamWConfig;
use burn::prelude::{Backend, ElementConversion, Int};
use burn::tensor::{Bool, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::loss::cross_entropy_with_logits;
use burn::tensor::ops::{ActivationOps, FloatTensorOps};
use burn::train::LearnerBuilder;
use burn::train::metric::{LossInput, LossMetric};
use num_traits::real::Real;

use crate::mine::{get_negative_samples, sample_negative_indices};
use crate::model::{AudioModel, AudioModelConfig, AudioModelInput};
use crate::model::encoder::{Encoder, EncoderConfig};
use crate::model::encoder::transformer::{TransformerEncoder, TransformerEncoderConfig};
use crate::model::extractor::FeatureExtractorConfig;
use crate::model::projection::FeatureProjectionConfig;
use crate::model::quantizer::{Quantizer, QuantizerConfig};
use crate::model::quantizer::gumbel::{GumbelQuantizer, GumbelQuantizerConfig};
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
            model: self.model_config.init(input_len, encoder_config),
            quantizer: <QC::Model<B> as Quantizer<B>>::new(last_conv_dim, quantizer_config, device),
            feature_dropout: DropoutConfig::new(self.feature_dropout as f64).init(),
            project_hidden: LinearConfig::new(hidden_size, self.projected_size)
                .init(device),
            project_quantized: LinearConfig::new(vector_dim, self.projected_size)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Pretrain<B: Backend, E: Module<B>, Q> {
    model: AudioModel<B, E>,
    feature_dropout: Dropout,
    quantizer: Q,
    project_hidden: Linear<B>,
    project_quantized: Linear<B>,
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

impl<B: Backend, E: Encoder<B>, Q: Quantizer<B>> Pretrain<B, E, Q> {
    pub fn forward(
        &self,
        inputs: Tensor<B, 2>,
        attention_mask: Tensor<B, 2, Int>,
        seq_lens: Vec<usize>,
        masked_time_steps: Tensor<B, 2, Bool>,
        sampled_negatives: Tensor<B, 3, Int>,
    ) -> (Tensor<B, 3>, f32) {
        let [batch, seq] = inputs.dims();

        let (last_hidden, features) = self.model.forward(
            AudioModelInput {
                inputs,
                seq_lens: seq_lens.clone(),
                masked_time_steps: masked_time_steps.clone(),
            }
        );

        let projected_features = self.project_hidden.forward(last_hidden.clone());
        let extracted_features = self.feature_dropout.forward(features);

        let (quantized_features, perplexity) = self
            .quantizer
            .forward::<true>(extracted_features, masked_time_steps.clone());

        let quantized_features = self.project_quantized.forward(quantized_features);

        let negative_indices =
            sample_negative_indices([batch, seq], 10, masked_time_steps.clone(), seq_lens);
        let negative_features = get_negative_samples(negative_indices, quantized_features.clone());

        let contrastive_logits = self.contrastive_logits(
            quantized_features.unsqueeze_dim(0),
            negative_features,
            projected_features,
            2.0,
        );

        // TODO: neg is pos thing

        let logits = contrastive_logits
            .clone()
            .swap_dims(0, 2)
            .reshape([-1, contrastive_logits.dims()[0] as i32]);

        let masked_logits = masked_time_steps.clone().bool_not().nonzero();

        dbg!(&logits);
        dbg!(&masked_logits);

        let logits = Tensor::gather(logits, 0, masked_logits[1].clone().unsqueeze_dim::<2>(0));

        // TODO: pytorch cross entropy loss ignores -100 target values, needs a workaround
        let target = ((Tensor::ones_like(&masked_time_steps.clone().float()) - masked_time_steps.clone().float()) * -100.0)
            .int()
            .swap_dims(0, 1)
            .flatten(0, 2);

        let num_code_vectors = self.quantizer.num_groups() * self.quantizer.num_vectors_per_group();

        let contrastive_loss = cross_entropy_with_logits(logits, target.float())
            .sum()
            .to_data()
            .value[0]
            .elem::<f32>();

        let diversity_loss =
            masked_time_steps.float().sum() * ((num_code_vectors as f32 - perplexity) / num_code_vectors as f32);
        let diversity_loss = diversity_loss.to_data().value[0].elem::<f32>();

        let diversity_loss_weight = 0.1;
        let loss = contrastive_loss + diversity_loss * diversity_loss_weight;

        (last_hidden, loss)
    }

    fn contrastive_logits(
        &self,
        target_features: Tensor<B, 4>,
        negative_features: Tensor<B, 4>,
        predicted_features: Tensor<B, 3>,
        temperature: f32,
    ) -> Tensor<B, 4> {
        let target_features = Tensor::cat(vec![target_features, negative_features], 0);

        let logits = cosine_similarity(predicted_features.unsqueeze_dim(0), target_features, 2);
        let logits = logits / temperature;

        logits
    }
}

pub fn pretrain<B: AutodiffBackend>(device: B::Device) {
        let batch_size = 10;
    let data = crate::util::sample_test_batch::<B>(batch_size, 1000, 2000);

    let hidden_size = 100;

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
        model_config,
        feature_dropout: 0.0,
        projected_size: 100,
    };
    let dataset = download_hf_dataset(&hf_hub::api::sync::Api::new().unwrap(), "illuin/small_commonvoice_test_set".into()).unwrap();

   // let data_loader_train = DataLoaderBuilder::new()

    let model = pretraining_config.init::<B, _, _>(data.input_len(), encoder_config, quantizer_config, &device);
    let optimizer = AdamWConfig::new().init::<B, Pretrain<_, _, _>>();

    let learner = LearnerBuilder::<B, (), (), Pretrain<_, _, _>, _, _>::new(".out")
        .devices(vec![device])
        .num_epochs(500)
        .summary()
        .build(model, optimizer, 0.0001);

//    let trained = learner.fit(data_loader_train, data_loader_test);
}