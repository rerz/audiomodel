use burn::module::AutodiffModule;
use burn::prelude::Backend;
use crate::model::AudioModelConfig;
use crate::model::encoder::burn_transformer::BurnTransformerEncoderConfig;
use crate::model::encoder::EncoderConfig;
use crate::model::extractor::FeatureExtractorConfig;
use crate::model::posconv::PosConvConfig;
use crate::model::pretrain::PretrainConfig;
use crate::model::projection::FeatureProjectionConfig;
use crate::model::quantizer::gumbel::GumbelQuantizerConfig;
use crate::model::quantizer::QuantizerConfig;

pub fn small_music_config<B: Backend>() -> (PretrainConfig, BurnTransformerEncoderConfig, GumbelQuantizerConfig) {
    let hidden_size = 256;

    let feature_extractor_config = FeatureExtractorConfig {
        conv_dims: vec![1, 128, 128, 128],
        conv_kernels: vec![10, 3, 3],
        conv_strides: vec![5, 2, 2],
    };

    let last_conv_dim = feature_extractor_config.last_conv_dim();

    let feature_projection_config = FeatureProjectionConfig {
        hidden_size,
        last_conv_dim,
        dropout: 0.0,
        layer_norm_eps: 1e-05,
    };

    let model_config = AudioModelConfig {
        feature_extractor_config,
        feature_projection_config,
        hidden_size,
    };

    let encoder_config = BurnTransformerEncoderConfig {
        pos_conv_config: PosConvConfig {
            num_groups: 8,
            hidden_size,
            num_embeddings: 64,
        },
        num_layers: 2,
        num_heads: 4,
    };

    let quantizer_config = GumbelQuantizerConfig {
        vector_dim: 128,
        vectors_per_group: 100,
        num_groups: 2,
    };

    (PretrainConfig {
        model_config,
        projected_size: 128,
        feature_dropout: 0.0,
    }, encoder_config, quantizer_config)
}
