use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::Bool;

mod gumbel;
mod knn;

pub trait QuantizerConfig {
    fn quantized_dim(&self) -> usize;
}

pub trait Quantizer<B: Backend> {
    type Config: QuantizerConfig;

    fn new(last_conv_dim: usize, config: Self::Config) -> Self;

    fn forward<const TRAINING: bool>(
        &self,
        features: Tensor<B, 3>,
        mask_time_steps: Tensor<B, 2, Bool>,
    ) -> (Tensor<B, 3>, f32);

    fn num_groups(&self) -> usize;

    fn num_vectors_per_group(&self) -> usize;
}
