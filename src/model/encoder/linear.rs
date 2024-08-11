use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::{Backend, Bool, Tensor};
use crate::model::encoder::{Encoder, EncoderConfig};

#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    dropout: Dropout,
    linear: Linear<B>,
    norm: LayerNorm<B>,
    activation: Gelu
}

impl<B: Backend> EncoderLayer<B> {

    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(in_dim, out_dim).init(device),
            dropout: DropoutConfig::new(0.1).init(),
            norm: LayerNormConfig::new(out_dim).init(device),
            activation: Gelu::new(),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 3>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let hidden = self.dropout.forward(hidden);
        let hidden = self.linear.forward(hidden);
        let hidden = self.norm.forward(hidden);
        let hidden = self.activation.forward(hidden);
        hidden
    }
}

#[derive(Module, Debug)]
pub struct LinearEncoder<B: Backend> {
    layers: Vec<EncoderLayer<B>>
}

impl<B: Backend> LinearEncoder<B> {
    pub fn encode(&self, mut hidden: Tensor<B, 3>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        for layer in &self.layers {
            hidden = layer.forward(hidden, attention_mask.clone());
        }
        hidden
    }
}

impl EncoderConfig for () {
    type Model<B> where B: Backend = LinearEncoder<B>;
}

impl<B: Backend> Encoder<B> for LinearEncoder<B> {
    type Config = ();

    fn new(config: Self::Config, hidden_size: usize, extractor_output_size: usize, device: &B::Device) -> Self {
        LinearEncoder {
            layers: vec![
                EncoderLayer::new(hidden_size, 100, device),
                EncoderLayer::new(100, 100, device),
                EncoderLayer::new(100, hidden_size, device),
            ]
        }
    }

    fn forward(&self, hidden: Tensor<B, 3>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        self.encode(hidden, attention_mask)
    }
}