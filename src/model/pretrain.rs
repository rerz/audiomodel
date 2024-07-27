use std::marker::PhantomData;

use burn::backend::LibTorch;
use burn::backend::libtorch::TchTensor;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::{Backend, ElementConversion, Int};
use burn::tensor::loss::cross_entropy_with_logits;
use burn::tensor::{Bool, Tensor};
use itertools::Itertools;

use crate::mine;
use crate::mine::{get_negative_samples, sample_negative_indices};
use crate::model::{AudioModel, AudioModelConfig, AudioModelInput};
use crate::model::encoder::{Encoder, EncoderConfig};
use crate::model::extractor::FeatureExtractorConfig;
use crate::model::quantizer::{Quantizer, QuantizerConfig};
use crate::util::sample_test_batch;

#[derive(Config)]
pub struct PretrainConfig {
    model_config: AudioModelConfig,
    feature_dropout: f32,
    projected_size: usize,
}

impl PretrainConfig {
    pub fn init<
        B: Backend,
        E: Encoder<B, Config = EC>,
        EC: EncoderConfig,
        Q: Quantizer<B, Config = QC>,
        QC: QuantizerConfig,
    >(
        self,
        input_len: usize,
        encoder_config: EC,
        quantizer_config: QC,
    ) -> Pretrain<B, E, Q> {
        let hidden_size = self.model_config.hidden_size;
        let vector_dim = quantizer_config.quantized_dim();
        let last_conv_dim = self.model_config.feature_extractor_config.last_conv_dim();

        Pretrain {
            model: self.model_config.init(input_len, encoder_config),
            quantizer: Q::new(last_conv_dim, quantizer_config),
            feature_dropout: DropoutConfig::new(self.feature_dropout as f64).init(),
            project_hidden: LinearConfig::new(hidden_size, self.projected_size)
                .init(&B::Device::default()),
            project_quantized: LinearConfig::new(vector_dim, self.projected_size)
                .init(&B::Device::default()),
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
}

pub trait AudioStuffBackend: Backend {
    fn cosine_similarity(a: Tensor<Self, 4>, b: Tensor<Self, 4>, dim: usize) -> Tensor<Self, 3>;
}

impl AudioStuffBackend for LibTorch {
    fn cosine_similarity(a: Tensor<Self, 4>, b: Tensor<Self, 4>, dim: usize) -> Tensor<Self, 3> {
        let a = a.into_primitive().tensor;
        let b = b.into_primitive().tensor;
        let sim = tch::Tensor::cosine_similarity(&a, &b, dim as i64, 1e-08);

        Tensor::from_primitive(TchTensor::new(sim))
    }
}

fn contrastive_logits<B: AudioStuffBackend>(
    target_features: Tensor<B, 4>,
    negative_features: Tensor<B, 4>,
    predicted_features: Tensor<B, 3>,
    temperature: f32,
) -> Tensor<B, 3> {
    let target_features = Tensor::cat(vec![target_features, negative_features], 0);

    let logits = B::cosine_similarity(predicted_features.unsqueeze_dim(0), target_features, 2);

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

impl<B: AudioStuffBackend, E: Encoder<B>, Q: Quantizer<B>> Pretrain<B, E, Q> {
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
    ) -> Tensor<B, 3> {
        let target_features = Tensor::cat(vec![target_features, negative_features], 0);

        let logits = B::cosine_similarity(predicted_features.unsqueeze_dim(0), target_features, 2);
        let logits = logits / temperature;

        logits
    }
}