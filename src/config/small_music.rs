use burn::prelude::Backend;

use crate::model::AudioModelConfig;
use crate::model::encoder::burn_transformer::BurnTransformerEncoderConfig;
use crate::model::extractor::FeatureExtractorConfig;
use crate::model::posenc::convpos::PosConvConfig;
use crate::model::pretrain::PretrainConfig;
use crate::model::projection::FeatureProjectionConfig;
use crate::model::quantizer::gumbel::GumbelQuantizerConfig;
use crate::train::ConfigBundle;

pub fn small_music_config<B: Backend>() -> ConfigBundle<BurnTransformerEncoderConfig, GumbelQuantizerConfig> {
    let hidden_size = 128;

    // must be even number of kernels/strides
    let feature_extractor_config = FeatureExtractorConfig {
        conv_dims: vec![1, 128, 128, 128, 128, 128, 128],
        conv_kernels: vec![9, 9, 5, 5, 5, 5],
        conv_strides: vec![5, 5, 2, 2, 2, 2],
    };

    let len = feature_extractor_config.output_len::<B>(50_000);

    let last_conv_dim = feature_extractor_config.last_conv_dim();

    let feature_projection_config = FeatureProjectionConfig {
        hidden_size,
        last_conv_dim,
        dropout: 0.1,
        layer_norm_eps: 1e-05,
    };

    let model_config = AudioModelConfig {
        feature_extractor_config,
        feature_projection_config,
        hidden_size,
    };

    let encoder_config = BurnTransformerEncoderConfig {
        pos_conv_config: PosConvConfig {
            num_groups: 16,
            hidden_size,
            num_embeddings: 64,
        },
        num_layers: 2,
        num_heads: 4,
    };

    let quantizer_config = GumbelQuantizerConfig {
        vector_dim: 256,
        vectors_per_group: 256,
        num_groups: 2,
    };

    ConfigBundle {
        pretrain_config: PretrainConfig {
            model_config,
            projected_size: 128,
            feature_dropout: 0.1,
        },
        encoder_config,
        quantizer_config,
    }
}
