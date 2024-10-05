use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::Backend;
use burn::tensor::{Bool, Tensor};
use burn::tensor::backend::AutodiffBackend;
use itertools::Itertools;
use ndarray::AssignElem;
use burnx::sequence::mask;
use crate::model::arch::encoder::{Encoder, EncoderConfig};

use crate::model::arch::extractor::{FeatureExtractor, FeatureExtractorConfig};
use crate::model::arch::projection::{FeatureProjection, FeatureProjectionConfig};

pub mod arch;

#[derive(Config)]
pub struct AudioModelConfig {
    pub hidden_size: usize,
    pub feature_extractor_config: FeatureExtractorConfig,
    pub feature_projection_config: FeatureProjectionConfig,
}

impl AudioModelConfig {
    pub fn init<B: Backend, EC: EncoderConfig>(
        self,
        input_len: u32,
        encoder_config: EC,
        device: &B::Device,
    ) -> AudioModel<B, EC::Model<B>> {
        let last_conv_dims = self.feature_extractor_config.last_conv_dim();
        let extractor_output_len = self.feature_extractor_config.output_len::<B>(input_len);

        AudioModel {
            feature_extractor: self.feature_extractor_config.init(device),
            feature_projection: self.feature_projection_config.init(device),
            feature_norm: LayerNormConfig::new(last_conv_dims).init(device),
            encoder: <EC::Model<B> as Encoder<B>>::new(
                encoder_config,
                self.hidden_size,
                extractor_output_len,
                device,
            ),
            mask: Param::from_tensor(Tensor::zeros([self.hidden_size], device)),
            hidden_size: self.hidden_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioModel<B: Backend, E> {
    pub feature_extractor: FeatureExtractor<B>,
    pub feature_projection: FeatureProjection<B>,
    pub feature_norm: LayerNorm<B>,
    pub encoder: E,
    pub mask: Param<Tensor<B, 1>>,
    pub hidden_size: usize,
}

pub struct AudioModelInput<B: Backend> {
    pub sequences: Tensor<B, 2>,
    pub sequence_lens: Vec<u32>,
    pub masked_time_steps: Option<Tensor<B, 2, Bool>>,
}

pub struct AudioModelOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub extracted_features: Tensor<B, 3>,
    pub padding_mask: Tensor<B, 2, Bool>,
    pub masked_time_steps: Option<Tensor<B, 2, Bool>>,
}

impl<B: Backend, E: Encoder<B>> AudioModel<B, E> {
    pub fn forward(
        &self,
        input: AudioModelInput<B>,
        device: &B::Device,
    ) -> AudioModelOutput<B> {
        // input.sequences : B x T
        let x = self.feature_extractor.forward(input.sequences);
        let x = x.swap_dims(1, 2);
        let x = self.feature_norm.forward(x);

        let y = x.clone();
        let y = if let Some(masked_time_steps) = input.masked_time_steps.clone() {
            y.clone()
                .select(1, masked_time_steps.clone().nonzero()[1].clone())
        } else {
            y
        };

        let padding_mask = mask::feature_space_padding_mask(
            x.dims()[1] as u32,
            input.sequence_lens,
            self.feature_extractor.kernel_sizes(),
            self.feature_extractor.strides(),
            device,
        );

        let x = self.project_to_hidden(x);
        let x = self.hidden_dropout(x);

        let x = if let Some(mask_indices) = input.masked_time_steps.clone() {
            BlockMask::apply_mask(
                x,
                mask_indices,
                self.mask.val(),
                padding_mask.clone(),
                device,
            )
        } else {
            x
        };

        let x = self.encoder.forward(x, padding_mask.clone());

        AudioModelOutput {
            last_hidden_state: x,
            extracted_features: y,
            padding_mask,
            masked_time_steps: input.masked_time_steps,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
