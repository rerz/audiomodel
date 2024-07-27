use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct FeatureProjection<B: Backend> {
    norm: LayerNorm<B>,
    linear: Linear<B>,
    dropout: Dropout,
}

#[derive(Config)]
pub struct FeatureProjectionConfig {
    pub last_conv_dim: usize,
    pub layer_norm_eps: f32,
    pub hidden_size: usize,
    pub dropout: f32,
}

impl FeatureProjectionConfig {
    pub fn init<B: Backend>(self) -> FeatureProjection<B> {
        FeatureProjection {
            linear: LinearConfig::new(self.last_conv_dim, self.hidden_size)
                .init(&B::Device::default()),
            norm: LayerNormConfig::new(self.last_conv_dim).init(&B::Device::default()),
            dropout: DropoutConfig::new(self.dropout as f64).init(),
        }
    }
}

impl<B: Backend> FeatureProjection<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let normed = self.norm.forward(hidden);
        let hidden = self.linear.forward(normed.clone());
        let hidden = self.dropout.forward(hidden);
        (hidden, normed)
    }
}
