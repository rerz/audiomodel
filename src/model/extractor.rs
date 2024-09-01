use std::iter;
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{LibTorch, NdArray};
use burn::backend::ndarray::NdArrayDevice;

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Gelu, Initializer, LayerNorm, LayerNormConfig, PaddingConfig1d};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::prelude::{Backend, Tensor};
use burn::tensor::ops::conv::calculate_conv_output_size;
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
    pub fn init<B: Backend>(self, device: &B::Device) -> FeatureExtractorConvLayer<B> {
        FeatureExtractorConvLayer {
            conv: Conv1dConfig::new(self.conv_dim_in, self.conv_dim_out, self.conv_kernel)
                .with_stride(self.conv_stride)
                .with_initializer(Initializer::KaimingNormal { gain: 0.95, fan_out_only: false })
                .init(device),
            norm: LayerNormConfig::new(self.conv_dim_out).init(device),
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

        //println!("extractor hidden dims {:?}", hidden.dims());

        let hidden = self.activation.forward(hidden);
        hidden
    }
}

#[derive(Config)]
pub struct FeatureExtractorConfig {
    pub conv_dims: Vec<usize>,
    pub conv_kernels: Vec<usize>,
    pub conv_strides: Vec<usize>,
}

impl FeatureExtractorConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> FeatureExtractor<B> {
        let dim_windows = self.conv_dims.iter().tuple_windows::<(_, _)>();

        let mut conv_layers = vec![];
        for i in 0..self.conv_dims.len() - 1 {
            let in_dim = self.conv_dims[i];
            let out_dim = self.conv_dims[i + 1];
            let kernel = self.conv_kernels[i];
            let stride = self.conv_strides[i];

            let layer = FeatureExtractorConvLayerConfig::new(in_dim, out_dim, kernel, stride).init(device);
            conv_layers.push(layer);
        }

        FeatureExtractor {
            conv_layers,
            kernel_sizes: self.conv_kernels,
            strides: self.conv_strides,
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

pub fn has_nan<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> bool {
    tensor.to_data().to_vec::<f32>().unwrap().into_iter().any(|val| val.is_nan())
}

impl<B: Backend> FeatureExtractor<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
        let [_batch, _time] = input.dims();

        let mut hidden = input.unsqueeze_dim(1);

        let [_batch, _channel, _time] = hidden.dims();


        // WHY IS THIS NOT THE SAME AS TIME STEPS IN MASK?
        for layer in &self.conv_layers {
            let hidden_dims = hidden.dims();
            //println!("hidden dims {:?}", hidden_dims);
            hidden = layer.forward(hidden);
        }

        let hidden_dims = hidden.dims();
        //println!("hidden dims: {:?}", hidden_dims);
        hidden
    }

    pub fn kernel_sizes(&self) -> &[usize] {
        &self.kernel_sizes
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn output_len(&self, input_len: usize) -> usize {
        feature_extractor_output_lens::<B>(vec![input_len], &self.kernel_sizes, &self.strides)[0]
    }

        pub fn receptive_field(&self, sample_rate: u32) -> f32 {
        let mut receptive_field = 1.0;
        let mut total_stride = 1.0;

        for (&kernel, &stride) in iter::zip(&self.kernel_sizes, &self.strides) {
            receptive_field += (kernel as f32 - 1.0) * total_stride;
            total_stride *= stride as f32;
        }

        let receptive_field_ms = (receptive_field / sample_rate as f32) * 1000.0;

        receptive_field_ms
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
        //println!("seq len {}", sequence_lens[0]);
        sequence_lens = sequence_lens.into_iter().map(|len| calculate_conv_output_size(*kernel, *stride, 0, 1, len)).collect(); //  conv1d_output_len::<B>(sequence_lens, *kernel, *stride, 0);
    }

    //println!("seq len {}", sequence_lens[0]);

    sequence_lens
}


#[test]
fn test_extractor_output() {
    tch::maybe_init_cuda();

    LibTorch::<f32, i8>::seed(0);

    let device = LibTorchDevice::Cuda(0);
    let input = Tensor::arange(1..1000, &device).unsqueeze_dim::<2>(0).float();

    let extractor: FeatureExtractor<LibTorch> = FeatureExtractorConfig::new(
        vec![1, 512, 512, 512, 512, 512, 512, 512],
        vec![10, 3, 3, 3, 3, 2, 2],
        vec![5, 2, 2, 2, 2, 2, 2],
    ).init(&device);

    let output = extractor.forward(input);

    //println!("{}", output);

}