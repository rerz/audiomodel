use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::Bool;

pub mod linear;
pub mod transformer;

pub trait EncoderConfig {}

pub trait Encoder<B: Backend> {
    type Config: EncoderConfig;

    fn new(config: Self::Config, hidden_size: usize, extractor_output_size: usize) -> Self;

    fn forward(&self, hidden: Tensor<B, 3>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3>;
}
