use std::iter;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicI64, AtomicU64};

use burn::backend::autodiff::ops::Backward;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::Dataset;
use burn::data::dataset::InMemDataset;
use burn::module::{AutodiffModule, Ignored, Module, ModuleDisplay, ModuleVisitor, Param, ParamId};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::{Backend, ElementConversion, Int};
use burn::tensor::{Bool, Tensor, TensorData};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::loss::cross_entropy_with_logits;
use burn::tensor::ops::IntElem;
use burn::train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use burn::train::metric::{Adaptor, LossInput, LossMetric, Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use itertools::Itertools;
use num_traits::real::Real;
use parquet::data_type::AsBytes;
use parquet::record::{Row, RowAccessor};
use rand::Rng;

use crate::data::{AudioBatch, AudioBatcher};
use crate::metric::correct::{CorrectInput, CorrectMetric};
use crate::metric::gradnorm::{GradientNormIntput, GradientNormMetric};
use crate::metric::perplexity::{PerplexityInput, PerplexityMetric};
use crate::metric::temperature::TemperatureInput;
use crate::mine::get_negative_samples;
use crate::model::{AudioModel, AudioModelConfig, AudioModelInput};
use crate::model::encoder::{Encoder, EncoderConfig};
use crate::model::encoder::burn_transformer::BurnTransformer;
use crate::model::encoder::transformer::TransformerEncoderConfig;
use crate::model::extractor::FeatureExtractorConfig;
use crate::model::projection::FeatureProjectionConfig;
use crate::model::quantizer::{Quantizer, QuantizerConfig};
use crate::model::quantizer::gumbel::GumbelQuantizerConfig;
use crate::ops::cosine_similarity;
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
            loss: CrossEntropyLossConfig::new().init(device),
            steps: Ignored(Arc::new(AtomicU64::new(0))),
        }
    }
}

#[derive(Module, Debug)]
pub struct Pretrain<B: Backend, E, Q> {
    pub model: AudioModel<B, E>,
    pub feature_dropout: Dropout,
    pub quantizer: Q,
    pub project_hidden: Linear<B>,
    pub project_quantized: Linear<B>,
    pub loss: CrossEntropyLoss<B>,
    pub steps: Ignored<Arc<AtomicU64>>
}



pub struct PretrainInput<B: Backend> {
    input_values: Tensor<B, 2>,
    attention_mask: Tensor<B, 2, Int>,
    masked_time_indices: Tensor<B, 2, Int>,
    sampled_negatives: Tensor<B, 3, Int>,
}

pub struct PretrainOutput<B: Backend> {
    loss: Tensor<B, 1>,
    contrastive_loss: Tensor<B, 1>,
    diversity_loss: Tensor<B, 1>,
    perplexity: Tensor<B, 1>,
    gradient_norm: Tensor<B, 1>,
}

pub trait ScalarExt<B: Backend> {
    type Elem;

    fn scalar(&self) -> Self::Elem;
}

impl<B: Backend> ScalarExt<B> for Tensor<B, 1> {
    type Elem = B::FloatElem;

    fn scalar(&self) -> B::FloatElem {
        let data = self.to_data();

        assert_eq!(data.num_elements(), 1);

        data.to_vec().unwrap()[0]
    }
}

impl<B: Backend> ScalarExt<B> for Tensor<B, 1, Int> {
    type Elem = B::IntElem;

    fn scalar(&self) -> Self::Elem {
        let data = self.to_data();

        assert_eq!(data.num_elements(), 1);

        data.to_vec().unwrap()[0]
    }
}

impl<B: Backend, Q: Quantizer<B>, E: Encoder<B>> Pretrain<B, E, Q> {

    fn quantize(&self, y: Tensor<B, 3,>, masked_time_steps: Tensor<B, 2, Bool>, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 1>) {
        let extracted_features = self.feature_dropout.forward(y);
        let (quantized_features, prob_perplexity) = self
            .quantizer
            .forward::<true>(extracted_features, masked_time_steps.clone(), device);


        let quantized_features = self.project_quantized.forward(quantized_features);
        let quantized_features = quantized_features.unsqueeze_dim::<4>(0);

        (quantized_features, prob_perplexity)
    }

    fn predict(&self, inputs: Tensor<B, 2>, seq_lens: Vec<usize>, masked_time_steps: Tensor<B, 2, Bool>, device: &B::Device,) -> (Tensor<B, 4>, Tensor<B, 3>, Tensor<B, 1>) {
        // y are the unmasked extracted features
        let (x, y, features_penalty) = self.model.forward(
            AudioModelInput {
                sequences: inputs,
                sequence_lens: seq_lens.clone(),
                masked_time_steps: masked_time_steps.clone(),
            },
            device,
        );

        let predicted_features = self.project_hidden.forward(x);
        let predicted_features = predicted_features.unsqueeze_dim(0);

        (predicted_features, y, features_penalty)
    }

    pub fn forward(
        &self,
        inputs: Tensor<B, 2>,
        attention_mask: Tensor<B, 2, Bool>,
        seq_lens: Vec<usize>,
        masked_time_steps: Tensor<B, 2, Bool>,
        negative_indices: Tensor<B, 3, Int>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 1>, u32) {
        let [batch, seq] = inputs.dims();

        let num_samples = attention_mask
            .clone()
            .int()
            .sum()
            .scalar();

        let (x, y, features_penalty) = self.predict(
            inputs,
            seq_lens,
            masked_time_steps.clone(),
            device
        );

        let (y, prob_perplexity) = self.quantize(
            y,
            masked_time_steps.clone(),
            device
        );


        let negative_features = get_negative_samples(
            negative_indices,
            y.clone()
        );


        // quantized features: 1 x batch x ?? x mask_len
        // negative_features: num_negatives x batch x ?? x mask_len
        // projected_features: 1 x batch x ?? x mask_len

        let logits = self.contrastive_logits(
            x.clone(),
            y,
            negative_features,
            0.1,
        );

        // constrastive_logits: (num_negatives + 1) x batch x seq

        // TODO: neg is pos thing

        let logits = logits
            .clone()
            .swap_dims(0, 2)
            .reshape([-1, logits.dims()[0] as i32]); // (batch x hidden) x (num_negatives + 1)

        let (correct, count) = if logits.to_data().num_elements() == 0 {
            (0, 0)
        } else {
            let max = logits.clone().argmax(1).equal_elem(0);
            let min = logits.clone().argmin(1).equal_elem(0);

            let max_iter = max.clone().to_data().to_vec::<bool>().unwrap();
            let min_iter = min.clone().to_data().to_vec::<bool>().unwrap();

            let both = iter::zip(max_iter, min_iter).map(|(a, b)| a & b);
            let both = Tensor::<B, 2, Bool>::from_bool(TensorData::new(both.collect_vec(), max.shape()), device);

            let correct = max.clone().int().sum().scalar().elem::<u32>() - both.int().sum().scalar().elem::<u32>();
            let count = max.to_data().num_elements();

            (correct, count)
        };

        let masked_indices = masked_time_steps.clone().nonzero()[1].clone();//.clone().to_data().to_vec::<i64>().unwrap();

        // ignoring indices in cross entropy loss seems broken so we just remove them from the tensor
        let unmasked_logits = Tensor::select(logits, 0, masked_indices);
        //let unmasked_logits = logits;

        let target = Tensor::zeros([unmasked_logits.dims()[0]], device);

        let num_code_vectors = self.quantizer.num_groups() * self.quantizer.num_vectors_per_group();
        //let contrastive_loss = cross_entropy_with_logits(unmasked_logits, target.unsqueeze_dim(1));
        let contrastive_loss = self.loss.forward(unmasked_logits, target);//.sum();

        let diversity_loss = (num_code_vectors as f32 - prob_perplexity.clone().scalar().elem::<f32>()) / num_code_vectors as f32;

        let diversity_loss_weight = 0.1;
        let features_penalty_weight = 10.0;
        // TODO: weight extra losses by sample size?
        //         if "sample_size" in sample:
        //             sample_size = sample["sample_size"]
        //         elif "mask_indices" in sample["net_input"]:
        //             sample_size = sample["net_input"]["mask_indices"].sum()
        //         else:
        //             sample_size = target.numel() if self.infonce else target.long().sum().item()


        let loss = contrastive_loss + diversity_loss * diversity_loss_weight + features_penalty * features_penalty_weight;



        (x.squeeze(0), loss, prob_perplexity, correct)
    }

    fn contrastive_logits(
        &self,
        predicted_features: Tensor<B, 4>,
        target_features: Tensor<B, 4>,
        negative_features: Tensor<B, 4>,
        logit_temperature: f32,
    ) -> Tensor<B, 3> {
        let target_and_negatives = Tensor::cat(vec![target_features.clone(), negative_features.clone()], 0);

        let neg_is_pos = (target_features.clone().equal(negative_features.clone()).all_dim(3));

        let logits = cosine_similarity(predicted_features, target_and_negatives, 3);
        let logits = logits / logit_temperature;

        let [num_samples, batch, seq, _sim] = logits.dims();

        // TODO: maybe skip first element since it corresponds to the true sample
        let masked_logits = logits.clone().slice([1..num_samples, 0..batch, 0..seq, 0..1]).mask_fill(neg_is_pos, f32::NEG_INFINITY);
        let logits = logits.slice_assign([1..num_samples, 0..batch, 0..seq, 0..1], masked_logits);

        logits.squeeze(3)
    }
}

pub struct PretrainStepOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub loss: Tensor<B, 1>,
    pub perplexity: Tensor<B, 1>,
    pub gradient_norm: f32,
    pub quantizer_temperature: f32,
    pub correct: u32,
}

impl<B: Backend> Adaptor<LossInput<B>> for PretrainStepOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Adaptor<TemperatureInput> for PretrainStepOutput<B> {
    fn adapt(&self) -> TemperatureInput {
        TemperatureInput {
            value: self.quantizer_temperature
        }
    }
}


impl<B: Backend> Adaptor<PerplexityInput> for PretrainStepOutput<B> {
    fn adapt(&self) -> PerplexityInput {
        PerplexityInput {
            value: self.perplexity.to_data().to_vec::<f32>().unwrap()[0]
        }
    }
}

impl<B: Backend> Adaptor<GradientNormIntput> for PretrainStepOutput<B> {
    fn adapt(&self) -> GradientNormIntput {
        GradientNormIntput {
            value: self.gradient_norm
        }
    }
}

impl<B: Backend> Adaptor<CorrectInput> for PretrainStepOutput<B> {
    fn adapt(&self) -> CorrectInput {
        CorrectInput {
            value: self.correct,
        }
    }
}