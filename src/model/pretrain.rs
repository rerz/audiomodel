use std::iter;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

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
use burn::tensor::{Bool, Element, Tensor, TensorData};
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
use crate::metric::gradnorm::{GradientNorm, GradientNormIntput, GradientNormMetric};
use crate::metric::maskingratio::MaskingRatioInput;
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
use crate::ops::{cosine_similarity, GradientMult};
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

    fn scalar<E: Element>(&self) -> E;
}

impl<B: Backend> ScalarExt<B> for Tensor<B, 1> {
    type Elem = B::FloatElem;

    fn scalar<E: Element>(&self) -> E {
        let data = self.to_data();

        assert_eq!(data.num_elements(), 1);
        data.to_vec::<Self::Elem>().unwrap().remove(0).elem()
    }
}

impl<B: Backend> ScalarExt<B> for Tensor<B, 1, Int> {
    type Elem = B::IntElem;

    fn scalar<E: Element>(&self) -> E {
        let data = self.to_data();

        assert_eq!(data.num_elements(), 1);

        data.to_vec::<Self::Elem>().unwrap().remove(0).elem()
    }
}

impl<B: Backend, Q: Quantizer<B>, E: Encoder<B>> Pretrain<B, E, Q> {

    fn quantize(&self, y: Tensor<B, 3,>, masked_time_steps: Tensor<B, 2, Bool>, device: &B::Device) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let extracted_features = self.feature_dropout.forward(y);
        let (quantized_features, prob_perplexity) = self
            .quantizer
            .forward::<true>(extracted_features, masked_time_steps.clone(), device);


        let quantized_features = self.project_quantized.forward(quantized_features);

        (quantized_features, prob_perplexity)
    }

    fn predict(&self, inputs: Tensor<B, 2>, seq_lens: Vec<usize>, masked_time_steps: Tensor<B, 2, Bool>, device: &B::Device,) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>, u32) {
        // y are the unmasked extracted features
        let (x, y, features_penalty, num_total_samples) = self.model.forward(
            AudioModelInput {
                sequences: inputs,
                sequence_lens: seq_lens.clone(),
                masked_time_steps: masked_time_steps.clone(),
            },
            device,
        );

        // only select time steps that have been masked (b x t x c)
        let x = x.select(1, masked_time_steps.clone().nonzero()[1].clone());

        let x = self.project_hidden.forward(x);

        (x, y, features_penalty, num_total_samples)
    }

    pub fn forward(
        &self,
        inputs: Tensor<B, 2>,
        attention_mask: Tensor<B, 2, Bool>,
        seq_lens: Vec<usize>,
        masked_time_steps: Tensor<B, 2, Bool>,
        negative_indices: Tensor<B, 3, Int>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 1>, u32, u32) {
        //println!("input dims {:?}", inputs.dims());

        let receptive_field = self.model.feature_extractor.receptive_field(16_000);

        let [_batch, _time] = inputs.dims();

        let (x, y, features_penalty, num_total_steps) = self.predict(
            inputs,
            seq_lens,
            masked_time_steps.clone(),
            device
        );

        // x and y only contain data for the masked time steps

        let (y, prob_perplexity) = self.quantize(
            y,
            masked_time_steps.clone(),
            device
        );

        let y_dims = y.dims();

        let negative_features = get_negative_samples(
            negative_indices,
            y.clone()
        );


        // quantized features: 1 x batch x time x hidden
        // negative_features: num_negatives x batch x time x hidden
        // projected_features: 1 x batch x time x hidden

        // logits shape: (num_negatives + 1) x batch x seq
        let logits = self.contrastive_logits(
            x.clone().unsqueeze_dim(0),
            y.unsqueeze_dim(0),
            negative_features,
            0.1,
        );

        let logit_dims = logits.dims();
        // logits: (num_negatives + 1) x batch x seq

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

            let correct = max.clone().int().sum().scalar::<u32>() - both.int().sum().scalar::<u32>();
            let count = max.to_data().num_elements();

            (correct, count)
        };

        let masked_indices = masked_time_steps.clone().nonzero()[1].clone();//.clone().to_data().to_vec::<i64>().unwrap();

        // ignoring indices in cross entropy loss seems broken so we just remove them from the tensor
        //let unmasked_logits = Tensor::select(logits, 0, masked_indices);
        let unmasked_logits = logits;

        let target = Tensor::zeros([unmasked_logits.dims()[0]], device);

        let num_code_vectors = self.quantizer.num_groups() * self.quantizer.num_vectors_per_group();

        let logit_dims = unmasked_logits.dims();
        let target_dims = target.dims();

        //let contrastive_loss = cross_entropy_with_logits(unmasked_logits, target.unsqueeze_dim(1));
        let contrastive_loss = self.loss.forward(unmasked_logits, target);//.sum();

        let diversity_loss = (num_code_vectors as f32 - prob_perplexity.clone().scalar::<f32>()) / num_code_vectors as f32;

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



        (x, loss, prob_perplexity, correct, num_total_steps)
    }

    fn contrastive_logits(
        &self,
        predicted_features: Tensor<B, 4>,
        target_features: Tensor<B, 4>,
        negative_features: Tensor<B, 4>,
        logit_temperature: f32,
    ) -> Tensor<B, 3> {
        let target_and_negatives = Tensor::cat(vec![target_features.clone(), negative_features.clone()], 0);

        let pred_dims = predicted_features.dims();
        let target_dims = target_and_negatives.dims();

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

impl<B: Backend> Adaptor<MaskingRatioInput> for PretrainStepOutput<B> {
    fn adapt(&self) -> MaskingRatioInput {
        MaskingRatioInput {
            num_total_samples: self.num_total_samples,
            num_masked_samples: self.num_masked_samples,
        }
    }
}

pub struct PretrainStepOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub loss: Tensor<B, 1>,
    pub perplexity: Tensor<B, 1>,
    pub gradient_norm: f32,
    pub quantizer_temperature: f32,
    pub correct: u32,
    pub num_total_samples: u32,
    pub num_masked_samples: u32,
}

impl<
        B: AutodiffBackend,
        E: Encoder<B> + AutodiffModule<B, InnerModule: ModuleDisplay> + ModuleDisplay,
        Q: Quantizer<B> + AutodiffModule<B, InnerModule: ModuleDisplay> + ModuleDisplay,
    > TrainStep<AudioBatch<B>, PretrainStepOutput<B>> for Pretrain<B, E, Q>
{
    fn step(&self, item: AudioBatch<B>) -> TrainOutput<PretrainStepOutput<B>> {
        let num_masked_samples = item.masked_time_indices.clone().int().sum().scalar();

        let (hidden, loss, perplexity, correct, num_total_steps) = self.forward(
            item.sequences,
            item.padding_mask,
            item.sequence_lens,
            item.masked_time_indices,
            item.sampled_negative_indices,
            &item.device,
        );

        let mut grads = loss.backward();

        // TODO: gradient multiplier on feature extractor?
        let mut gradient_mult = GradientMult {
            multiplier: 0.1,
            grads: &mut grads,
        };
        self.model.feature_extractor.visit(&mut gradient_mult);

        let gradient_norm = {
            let mut gradient_norm = GradientNorm::new(&grads, 1.0);
            self.visit(&mut gradient_norm);
            gradient_norm.total_norm.sqrt()
        };

        self.steps.fetch_add(1, Ordering::Relaxed);

        TrainOutput::new(
            self,
            grads,
            PretrainStepOutput {
                loss,
                hidden,
                perplexity,
                gradient_norm,
                quantizer_temperature: self.quantizer.temperature(),
                correct,
                num_masked_samples,
                num_total_samples: num_total_steps,
            },
        )
    }

    fn optimize<BB, O>(mut self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
    where
        BB: AutodiffBackend,
        O: Optimizer<Self, BB>,
        Self: AutodiffModule<BB>,
    {
        self.quantizer.set_num_steps(self.steps.load(Ordering::Relaxed) as u32);

        optim.step(lr, self, grads)
    }
}
