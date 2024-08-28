use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig};
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
pub mod projection;
pub mod quantizer;
pub mod pretrain;
pub mod posenc;

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
        device: &B::Device,
    ) -> AudioModel<B, EC::Model<B>> {
        let last_conv_dims = self.feature_extractor_config.last_conv_dim();
        let extractor_output_len = self.feature_extractor_config.output_len::<B>(input_len);

        AudioModel {
            feature_extractor: self.feature_extractor_config.init(device),
            feature_projection: self.feature_projection_config.init(device),
            feature_norm: LayerNormConfig::new(last_conv_dims).init(device),
            encoder: <EC::Model<B> as Encoder<B>>::new(encoder_config, self.hidden_size, extractor_output_len, device),
            mask: Param::from_tensor(Tensor::zeros([self.hidden_size], device)),
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
}

pub struct AudioModelInput<B: Backend> {
    sequences: Tensor<B, 2>,
    sequence_lens: Vec<usize>,
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
        input: AudioModelInput<B>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>) {
        // input.sequences : B x T
        let features = self.feature_extractor.forward(input.sequences);
        // features : B x C x T

        let features_penalty = features.clone().powf_scalar(2.0).mean();

        let features = features.swap_dims(1, 2);
        let features = self.feature_norm.forward(features);
        // features : B x T x C

        let y = features.clone();

        let feature_space_padding_mask =
            mask::feature_space_padding_mask(features.dims()[1], input.sequence_lens, self.feature_extractor.kernel_sizes(), self.feature_extractor.strides(), device);


        let [batch, time, channels] = features.dims();
        let steps_to_drop = features.dims()[1] % 2;

        let masked_time_steps = input.masked_time_steps;

        // features : B x T x C
        // y : B x T x C
        let features = self.feature_projection.forward(features);


        let x = mask::mask_hidden_states(
            features,
            masked_time_steps,
            self.mask.val(),
            feature_space_padding_mask.clone(),
        );


        let x = self.encoder.forward(x, feature_space_padding_mask);

        // TODO:
        //                # tpu-comment: reducing the size in a dynamic way causes
        //                 # too many recompilations on xla.
        //                 y = unmasked_features[mask_indices].view(
        //                     unmasked_features.size(0), -1, unmasked_features.size(-1)
        //                 )


        (x, y, features_penalty)
    }
}
