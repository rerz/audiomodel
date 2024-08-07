use burn::config::Config;
use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::{Bool, Tensor};
use burn::tensor::backend::AutodiffBackend;
use itertools::Itertools;
use ndarray::AssignElem;

use crate::mask;
use crate::model::encoder::{Encoder, EncoderConfig};
use crate::model::extractor::{FeatureExtractor, FeatureExtractorConfig};
use crate::model::projection::{FeatureProjection, FeatureProjectionConfig};

pub mod encoder;
pub mod extractor;
pub mod posconv;
pub mod projection;
pub mod quantizer;
pub mod pretrain;

#[derive(Config)]
pub struct AudioModelConfig {
    pub hidden_size: usize,
    pub feature_extractor_config: FeatureExtractorConfig,
    pub feature_projection_config: FeatureProjectionConfig,
}

impl AudioModelConfig {
    pub fn init<B: Backend, EC: EncoderConfig>(
        self,
        input_len: usize,
        encoder_config: EC,
    ) -> AudioModel<B, EC::Model<B>> {
        let last_conv_dims = self.feature_extractor_config.last_conv_dim();
        let extractor_output_len = self.feature_extractor_config.output_len::<B>(input_len);

        AudioModel {
            feature_extractor: self.feature_extractor_config.init(),
            feature_projection: self.feature_projection_config.init(),
            encoder: <EC::Model<B> as Encoder<B>>::new(encoder_config, self.hidden_size, extractor_output_len),
            mask: Param::from_tensor(Tensor::zeros([self.hidden_size], &B::Device::default())),
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioModel<B: Backend, E: Module<B>> {
    feature_extractor: FeatureExtractor<B>,
    feature_projection: FeatureProjection<B>,
    encoder: E,
    mask: Param<Tensor<B, 1>>,
}

pub struct AudioModelInput<B: Backend> {
    inputs: Tensor<B, 2>,
    seq_lens: Vec<usize>,
    masked_time_steps: Tensor<B, 2, Bool>,
}

pub struct AudioModelOutput<B: Backend> {
    last_hidden_state: Tensor<B, 3>,
    extracted_features: Tensor<B, 3>,
    hidden_states: Option<Tensor<B, 3>>,
    attentions: Option<Tensor<B, 3>>,
}

impl<B: Backend, E: Encoder<B>> AudioModel<B, E> {
    pub fn forward(
        &self,
        AudioModelInput {
            inputs,
            seq_lens,
            masked_time_steps
        }: AudioModelInput<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let features = self.feature_extractor.forward(inputs);
        let features = features.swap_dims(1, 2);

        let feature_space_attention_mask =
            mask::feature_space_attention_mask(features.dims()[1], seq_lens, self.feature_extractor.kernel_sizes(), self.feature_extractor.strides());

        let (hidden, features) = self.feature_projection.forward(features);

        let hidden = mask::mask_hidden_states(
            hidden,
            masked_time_steps,
            self.mask.val(),
            feature_space_attention_mask.clone(),
        );

        let encoder_output = self.encoder.forward(hidden, feature_space_attention_mask);

        (encoder_output, features)
    }
}
