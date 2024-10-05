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
use burn::train::metric::{
    Adaptor, LossInput, LossMetric, Metric, MetricEntry, MetricMetadata, Numeric,
};
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::renderer::SelectedMetricsRenderer;
use itertools::Itertools;
use num_traits::real::Real;
use parquet::data_type::AsBytes;
use parquet::record::{Row, RowAccessor};
use rand::Rng;

use crate::data::{AudioBatch, AudioBatcher};
use crate::metric::code_perplexity::CodePerplexityInput;
use crate::metric::contrastive_loss::ContrastiveLossInput;
use crate::metric::correct::{CorrectInput, CorrectMetric};
use crate::metric::diversity_loss::DiversityLossInput;
use crate::metric::gradnorm::{GradientNorm, GradientNormIntput, GradientNormMetric};
use crate::metric::maskingratio::MaskingRatioInput;
use crate::metric::perplexity::{PerplexityInput, PerplexityMetric};
use crate::metric::temperature::TemperatureInput;
use crate::mine::get_negative_samples;
use crate::model::{AudioModel, AudioModelConfig, AudioModelInput, AudioModelOutput};

#[derive(Config)]
pub struct PretrainConfig {
    pub model_config: AudioModelConfig,
    pub feature_dropout: f32,
    pub projected_size: usize,
}

impl PretrainConfig {
    pub fn init<B: Backend, EC: EncoderConfig, QC: QuantizerConfig>(
        self,
        input_len: u32,
        encoder_config: EC,
        quantizer_config: QC,
        device: &B::Device,
    ) -> Pretrain<B, EC::Model<B>, QC::Model<B>> {
        let hidden_size = self.model_config.hidden_size;
        let vector_dim = quantizer_config.quantized_dim();
        let last_conv_dim = self.model_config.feature_extractor_config.last_conv_dim();

        Pretrain {
            model: self.model_config.init(input_len, encoder_config, device),
            quantizer: <QC::Model<B> as Quantizer<B>>::new(last_conv_dim, quantizer_config, device),
            feature_dropout: DropoutConfig::new(self.feature_dropout as f64).init(),
            project_hidden: LinearConfig::new(hidden_size, self.projected_size).init(device),
            project_quantized: LinearConfig::new(vector_dim, self.projected_size).init(device),
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
    pub steps: Ignored<Arc<AtomicU64>>,
}

pub struct PretrainInput<B: Backend> {
    input_values: Tensor<B, 2>,
    attention_mask: Tensor<B, 2, Int>,
    masked_time_indices: Tensor<B, 2, Int>,
    sampled_negatives: Tensor<B, 3, Int>,
}

pub struct PretrainOutput<B: Backend> {
    x: Tensor<B, 3>,
    total_loss: Tensor<B, 1>,
    contrastive_loss: Tensor<B, 1>,
    diversity_loss: Tensor<B, 1>,
    prob_perplexity: Tensor<B, 1>,
    code_perplexity: Option<Tensor<B, 1>>,
    num_samples: u32,
    num_correct: u32,
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
    fn quantize(
        &self,
        y: Tensor<B, 3>,
        masked_time_steps: Tensor<B, 2, Bool>,
        training: bool,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Option<Tensor<B, 1>>) {
        let extracted_features = self.feature_dropout.forward(y);
        let (quantized_features, prob_perplexity, code_perplexity) =
            self.quantizer
                .forward(extracted_features, masked_time_steps.clone(), training, device);

        let quantized_features = self.project_quantized.forward(quantized_features);

        (quantized_features, prob_perplexity, code_perplexity)
    }

    fn predict(
        &self,
        inputs: Tensor<B, 2>,
        seq_lens: Vec<u32>,
        masked_time_steps: Tensor<B, 2, Bool>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // y are the unmasked extracted features
        let AudioModelOutput {
            last_hidden_state,
            extracted_features,
            ..
        } = self.model.forward(
            AudioModelInput {
                sequences: inputs,
                sequence_lens: seq_lens.clone(),
                masked_time_steps: Some(masked_time_steps.clone()),
            },
            device,
        );

        // only select time steps that have been masked (BxTxC)
        let x = last_hidden_state.select(1, masked_time_steps.clone().nonzero()[1].clone());
        let x = self.project_hidden.forward(x);

        (x, extracted_features)
    }

    fn count_correct_predictions(&self, logits: Tensor<B, 2>, device: &B::Device) -> (u32, u32) {
         let (count, correct) = if logits.to_data().num_elements() == 0 {
            (0, 0)
        } else {
            let max = logits.clone().argmax(1).equal_elem(0);
            let min = logits.clone().argmin(1).equal_elem(0);

            let max_iter = max.clone().to_data().to_vec::<bool>().unwrap();
            let min_iter = min.clone().to_data().to_vec::<bool>().unwrap();

            let both = iter::zip(max_iter, min_iter).map(|(a, b)| a & b);
            let both = Tensor::<B, 2, Bool>::from_bool(
                TensorData::new(both.collect_vec(), max.shape()),
                device,
            );

            let correct =
                max.clone().int().sum().scalar::<u32>() - both.int().sum().scalar::<u32>();
            let count = max.to_data().num_elements();

             (count as u32, correct)
        };

        (count, correct)
    }

    fn diversity_loss(&self, prob_perplexity: Tensor<B, 1>, num_code_vectors: usize, loss_weight: f32, device: &B::Device) -> Tensor<B, 1> {
        let prob_perplexity =  prob_perplexity.scalar::<f32>();
        let diversity_loss = (num_code_vectors as f32 - prob_perplexity)
            / num_code_vectors as f32;
        let diversity_loss = Tensor::from_floats([diversity_loss * loss_weight], device);

        diversity_loss
    }

    pub fn forward(
        &self,
        // BxT
        inputs: Tensor<B, 2>,
        sequence_lens: Vec<u32>,
        masked_time_steps: Tensor<B, 2, Bool>,
        negative_indices: Tensor<B, 3, Int>,
        training: bool,
        device: &B::Device,
    ) -> PretrainOutput<B> {

        // x and y only contain data for the masked time steps
        let (x, y) =
            self.predict(inputs, sequence_lens, masked_time_steps.clone(), device);

        let (y, prob_perplexity, code_perplexity) = self.quantize(y, masked_time_steps.clone(), training, device);

        // NxBxTxH
        let negative_features = get_negative_samples(negative_indices, y.clone());


        // (N+1)xBxT
        let logits = self.contrastive_logits(
            x.clone().unsqueeze_dim(0),
            y.unsqueeze_dim(0),
            negative_features,
            0.1,
        );

        // (BxH)x(N+1)
        let logits = logits
            .clone()
            .swap_dims(0, 2)
            .reshape([-1, logits.dims()[0] as i32]);

        let target = Tensor::zeros([logits.dims()[0]], device);

        let num_code_vectors = self.quantizer.num_groups() * self.quantizer.num_vectors_per_group();

        let (count, correct) = self.count_correct_predictions(logits.clone(), device);

        let contrastive_loss = self.loss.forward(logits, target);
        let diversity_loss = self.diversity_loss(prob_perplexity.clone(), num_code_vectors, 0.1, device);

        let total_loss = contrastive_loss.clone() + diversity_loss.clone(); // + size_penalty.clone();

        PretrainOutput {
            x,
            total_loss,
            contrastive_loss,
            diversity_loss,
            prob_perplexity,
            code_perplexity,
            num_samples: count,
            num_correct: correct,
        }
    }

    fn contrastive_logits(
        &self,
        // 1xBxTxH
        predicted_features: Tensor<B, 4>,
        // 1xBxTxH
        target_features: Tensor<B, 4>,
        // NxBxTxH
        negative_features: Tensor<B, 4>,
        logit_temperature: f32,
    ) -> Tensor<B, 3> {
        let target_and_negatives =
            Tensor::cat(vec![target_features.clone(), negative_features.clone()], 0);

        // TODO: really not sure about this
        let neg_is_pos = target_features
            .clone()
            .equal(negative_features.clone())
            .all_dim(3)
            .bool_not();

        let logits = cosine_similarity(predicted_features, target_and_negatives, 3);
        let logits = logits / logit_temperature;

        let [num_samples, batch, seq, _sim] = logits.dims();

        let masked_logits = logits
            .clone()
            .slice([1..num_samples, 0..batch, 0..seq, 0..1])
            .mask_fill(neg_is_pos, f32::NEG_INFINITY);
        let logits = logits.slice_assign([1..num_samples, 0..batch, 0..seq, 0..1], masked_logits);

        logits.squeeze(3)
    }

    fn apply_gradient_mult(&self, module: &impl Module<B>, grads: &mut B::Gradients)
        where
            B: AutodiffBackend,
    {
        let mut gradient_mult = GradientMult {
            multiplier: 0.1,
            grads,
        };
        module.visit(&mut gradient_mult);
    }

    fn get_gradient_norm(&self, grads: &B::Gradients, scale: f32) -> f32
        where
            B: AutodiffBackend,
            E: ModuleDisplay, Q: ModuleDisplay
    {
        let mut gradient_norm = GradientNorm::new(grads, scale);
        self.visit(&mut gradient_norm);
        gradient_norm.total_norm.sqrt()
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
            value: self.quantizer_temperature,
        }
    }
}

impl<B: Backend> Adaptor<PerplexityInput> for PretrainStepOutput<B> {
    fn adapt(&self) -> PerplexityInput {
        PerplexityInput {
            value: self.perplexity.to_data().to_vec::<f32>().unwrap()[0],
        }
    }
}

impl<B: Backend> Adaptor<GradientNormIntput> for PretrainStepOutput<B> {
    fn adapt(&self) -> GradientNormIntput {
        GradientNormIntput {
            value: self.gradient_norm,
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

impl<B: Backend> Adaptor<ContrastiveLossInput> for PretrainStepOutput<B> {
    fn adapt(&self) -> ContrastiveLossInput {
        ContrastiveLossInput {
            value: self.contrastive_loss.scalar(),
        }
    }
}

impl<B: Backend> Adaptor<DiversityLossInput> for PretrainStepOutput<B> {
    fn adapt(&self) -> DiversityLossInput {
        DiversityLossInput {
            value: self.diversity_loss.scalar(),
        }
    }
}

pub struct PretrainStepOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub loss: Tensor<B, 1>,
    pub perplexity: Tensor<B, 1>,
    pub diversity_loss: Tensor<B, 1>,
    pub contrastive_loss: Tensor<B, 1>,
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
        let num_masked_samples = item.masked_time_steps.clone().unwrap().int().sum().scalar();
        let PretrainOutput {
            x,
            total_loss,
            contrastive_loss,
            diversity_loss,
            prob_perplexity,
            num_samples,
            num_correct,
            ..
        } = self.forward(
            item.sequences,
            item.sequence_lens,
            item.masked_time_steps.unwrap(),
            item.sampled_negative_indices,
            true,
            &item.device,
        );

        let mut grads = total_loss.backward();

        //self.apply_gradient_mult(&self.model.feature_extractor, &mut grads);

        let gradient_norm = self.get_gradient_norm(&grads, 1.0);

        self.steps.fetch_add(1, Ordering::Relaxed);

        TrainOutput::new(
            self,
            grads,
            PretrainStepOutput {
                loss: total_loss,
                hidden: x,
                perplexity: prob_perplexity,
                gradient_norm,
                quantizer_temperature: self.quantizer.temperature(),
                correct: num_correct,
                num_masked_samples,
                num_total_samples: num_samples,
                contrastive_loss,
                diversity_loss,
            },
        )
    }

    fn optimize<BB, O>(mut self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
        where
            BB: AutodiffBackend,
            O: Optimizer<Self, BB>,
            Self: AutodiffModule<BB>,
    {
        self.quantizer
            .set_num_steps(self.steps.load(Ordering::Relaxed) as u32);

        optim.step(lr, self, grads)
    }
}

pub struct ValidationStepOutput<B: Backend> {
    total_loss: Tensor<B, 1>,
    contrastive_loss: Tensor<B, 1>,
    diversity_loss: Tensor<B, 1>,
    num_correct: u32,
    prob_perplexity: Tensor<B, 1>,
    code_perplexity: Tensor<B, 1>,
}

impl<B: Backend, E: Encoder<B>, Q: Quantizer<B>> ValidStep<AudioBatch<B>, ValidationStepOutput<B>> for Pretrain<B, E, Q> {
    fn step(&self, item: AudioBatch<B>) -> ValidationStepOutput<B> {
        let input_dims = item.sequences.dims();
        let PretrainOutput {
            x,
            total_loss,
            contrastive_loss,
            diversity_loss,
            prob_perplexity,
            num_samples,
            num_correct,
            code_perplexity,
        } = self.forward(
            item.sequences,
            item.sequence_lens,
            item.masked_time_steps.unwrap(),
            item.sampled_negative_indices,
            false,
            &item.device,
        );

        ValidationStepOutput {
            total_loss,
            code_perplexity: code_perplexity.unwrap(),
            prob_perplexity,
            contrastive_loss,
            diversity_loss,
            num_correct,
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for ValidationStepOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.total_loss.clone())
    }
}

impl<B: Backend> Adaptor<CodePerplexityInput> for ValidationStepOutput<B> {
    fn adapt(&self) -> CodePerplexityInput {
        CodePerplexityInput {
            value: self.code_perplexity.scalar(),
        }
    }
}

impl<B: Backend> Adaptor<PerplexityInput> for ValidationStepOutput<B> {
    fn adapt(&self) -> PerplexityInput {
        PerplexityInput {
            value: self.prob_perplexity.scalar(),
        }
    }
}

impl<B: Backend> Adaptor<ContrastiveLossInput> for ValidationStepOutput<B> {
    fn adapt(&self) -> ContrastiveLossInput {
        ContrastiveLossInput {
            value: self.contrastive_loss.scalar(),
        }
    }
}

impl<B: Backend> Adaptor<DiversityLossInput> for ValidationStepOutput<B> {
    fn adapt(&self) -> DiversityLossInput {
        DiversityLossInput {
            value: self.diversity_loss.scalar(),
        }
    }
}

impl<B: Backend> Adaptor<CorrectInput> for ValidationStepOutput<B> {
    fn adapt(&self) -> CorrectInput {
        CorrectInput {
            value: self.num_correct,
        }
    }
}