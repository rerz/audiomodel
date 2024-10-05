use burn::module::Module;
use burn::prelude::{Backend, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Bool;

pub mod linear;
pub mod transformer;
pub mod burn_transformer;

pub trait EncoderConfig {
    type Model<B>: Encoder<B, Config=Self> where B: Backend;
}

pub trait Encoder<B: Backend>: Module<B> {
    type Config: EncoderConfig;

    fn new(config: Self::Config, hidden_size: usize, extractor_output_size: u32, device: &B::Device) -> Self;

    fn forward(&self, hidden: Tensor<B, 3>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3>;
}
