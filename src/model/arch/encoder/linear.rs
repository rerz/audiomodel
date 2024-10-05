use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::{Backend, Bool, Tensor};

use crate::model::encoder::{Encoder, EncoderConfig};
use crate::model::encoder::burn_transformer::BurnTransformer;

#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    dropout: Dropout,
    linear: Linear<B>,
    norm: LayerNorm<B>,
    activation: Gelu,
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

        //println!("linear hidden dims {:?}", hidden.dims());

        let hidden = self.activation.forward(hidden);
        hidden
    }
}

#[derive(Module, Debug)]
pub struct LinearEncoder<B: Backend> {
    layers: Vec<EncoderLayer<B>>,
}

impl<B: Backend> LinearEncoder<B> {
    pub fn encode(&self, mut hidden: Tensor<B, 3>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        for layer in &self.layers {
            hidden = layer.forward(hidden, attention_mask.clone());
        }
        hidden
    }
}
