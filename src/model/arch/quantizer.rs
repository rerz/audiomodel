use burn::module::{AutodiffModule, Module};
use burn::prelude::{Backend, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Bool;

pub mod gumbel;
pub mod knn;

pub trait QuantizerConfig {
    type Model<B>: Quantizer<B, Config=Self> where B: Backend;

    fn quantized_dim(&self) -> usize;
}

pub trait Quantizer<B: Backend>: Module<B> {
    type Config: QuantizerConfig;

    fn new(last_conv_dim: usize, config: Self::Config, device: &B::Device) -> Self;

    fn forward(
        &self,
        features: Tensor<B, 3>,
        mask_time_steps: Tensor<B, 2, Bool>,
        training: bool,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Option<Tensor<B, 1>>);

    fn num_groups(&self) -> usize;

    fn num_vectors_per_group(&self) -> usize;

    fn temperature(&self) -> f32;

    fn set_num_steps(&mut self, num_steps: u32);
}
