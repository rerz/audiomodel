use std::iter;

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Gelu, LayerNorm, LayerNormConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::prelude::{Backend, Tensor};
use itertools::{Itertools, izip};

#[derive(Module, Debug)]
pub struct FeatureExtractorConvLayer<B: Backend> {
    conv: Conv1d<B>,
    norm: LayerNorm<B>,
    activation: Gelu,
}

#[derive(Config)]
pub struct FeatureExtractorConvLayerConfig {
    conv_dim_in: usize,
    conv_dim_out: usize,
    conv_kernel: usize,
    conv_stride: usize,
}

impl FeatureExtractorConvLayerConfig {
    pub fn init<B: Backend>(self) -> FeatureExtractorConvLayer<B> {
        FeatureExtractorConvLayer {
            conv: Conv1dConfig::new(self.conv_dim_in, self.conv_dim_out, self.conv_kernel)
                .with_stride(self.conv_stride)
                .init(&B::Device::default()),
            norm: LayerNormConfig::new(self.conv_dim_out).init(&B::Device::default()),
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> FeatureExtractorConvLayer<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.conv.forward(hidden);

        let hidden = hidden.swap_dims(1, 2);
        let hidden = self.norm.forward(hidden);
        let hidden = hidden.swap_dims(1, 2);

        let hidden = self.activation.forward(hidden);
        hidden
    }
}

impl<B: Backend> From<((&usize, &usize), &usize, &usize)> for FeatureExtractorConvLayer<B> {
    fn from(((in_dim, out_dim), kernel, stride): ((&usize, &usize), &usize, &usize)) -> Self {
        FeatureExtractorConvLayerConfig::new(*in_dim, *out_dim, *kernel, *stride).init()
    }
}

#[derive(Config)]
pub struct FeatureExtractorConfig {
    pub conv_dims: Vec<usize>,
    pub conv_kernels: Vec<usize>,
    pub conv_strides: Vec<usize>,
}

impl FeatureExtractorConfig {
    pub fn init<B: Backend>(self) -> FeatureExtractor<B> {
        let dim_windows = self.conv_dims.iter().tuple_windows::<(_, _)>();

        FeatureExtractor {
            conv_layers: izip!(dim_windows, &self.conv_kernels, &self.conv_strides)
                .map(FeatureExtractorConvLayer::from)
                .collect_vec(),
            kernel_sizes: self.conv_kernels,
            strides: self.conv_strides
        }
    }

    pub fn last_conv_dim(&self) -> usize {
        *self.conv_dims.last().unwrap()
    }

    pub fn output_len<B: Backend>(&self, input_len: usize) -> usize {
        feature_extractor_output_lens::<B>(vec![input_len], &self.conv_kernels, &self.conv_strides)
            [0]
    }
}

#[derive(Module, Debug)]
pub struct FeatureExtractor<B: Backend> {
    kernel_sizes: Vec<usize>,
    strides: Vec<usize>,
    conv_layers: Vec<FeatureExtractorConvLayer<B>>,
}

impl<B: Backend> FeatureExtractor<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
        let mut hidden = input.unsqueeze_dim(1);

        for layer in &self.conv_layers {
            hidden = layer.forward(hidden);
        }

        hidden
    }

    pub fn kernel_sizes(&self) -> &[usize] {
        &self.kernel_sizes
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

pub fn conv1d_output_len<B: Backend>(
    input_lens: Vec<usize>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Vec<usize> {
    input_lens
        .into_iter()
        .map(|len| ((len + 2 * padding - kernel_size) / stride) + 1)
        .collect_vec()
}

pub fn feature_extractor_output_lens<B: Backend>(
    mut sequence_lens: Vec<usize>,
    kernel_sizes: &[usize],
    strides: &[usize],
) -> Vec<usize> {
    for (kernel, stride) in iter::zip(kernel_sizes, strides) {
        sequence_lens = conv1d_output_len::<B>(sequence_lens, *kernel, *stride, 0);
    }

    sequence_lens
}
