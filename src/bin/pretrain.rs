use burn::nn::transformer::TransformerEncoderConfig;
use audiomodel::model::AudioModelConfig;
use audiomodel::model::extractor::FeatureExtractorConfig;
use audiomodel::model::pretrain::PretrainConfig;
use audiomodel::model::projection::FeatureProjectionConfig;
use audiomodel::model::quantizer::gumbel::GumbelQuantizerConfig;

fn main() {
    let batch_size = 10;
    let data = audiomodel::util::sample_test_batch(batch_size, 1000, 2000);

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

    };

    let quantizer_config = GumbelQuantizerConfig {

    };


    let pretraining_config = PretrainConfig {
        model_config,
        feature_dropout: 0.0,
        projected_size: 100,
    };

    let model = pretraining_config.init(data.input_len(), )
}