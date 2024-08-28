pub mod block;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::{Backend, Int};
use burn::tensor::{Bool, Tensor};
use itertools::Itertools;
use crate::mask::block::{BlockMask, BlockMaskConfig};

use crate::model::extractor::feature_extractor_output_lens;

pub trait MaskingStrategy<B: Backend> {
    type Config: Clone + Send + Sync;

    fn get_mask_indices(
        batch_and_seq_dims: [usize; 2],
        seq_lens: Vec<usize>,
        mask_config: Self::Config,
        device: &B::Device,
    ) -> Tensor<B, 2, Bool>;

    fn apply_mask(
        &self,
        hidden: Tensor<B, 3>,
        mask_time_indices: Tensor<B, 2, Bool>,
        mask: Tensor<B, 1>,
        _attention_mask: Tensor<B, 2, Bool>,
        _device: &B::Device,
    ) -> Tensor<B, 3> {
        let hidden_shape = hidden.shape();

        let mask_indices = mask_time_indices
            .unsqueeze_dim::<3>(2)
            .expand(hidden_shape.clone());
        let mask = mask.expand(hidden_shape);

        let hidden = Tensor::mask_where(hidden, mask_indices, mask);

        hidden
    }
}

#[test]
fn test_get_mask_indices() {
    NdArray::<f32, i8>::seed(0);

    let indices: Tensor<NdArray, 2, Bool> =
        BlockMask::get_mask_indices([2, 20], vec![20, 20], BlockMaskConfig {
            min_masks: 1,
            mask_len: 10,
            mask_prob: 0.1,
        }, &NdArrayDevice::Cpu);

    println!("{indices}")
}

pub fn feature_space_padding_mask<B: Backend>(
    feature_vec_len: usize,
    sequence_lens: Vec<usize>,
    kernel_sizes: &[usize],
    strides: &[usize],
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let batch_size = sequence_lens.len();

    let output_lens = feature_extractor_output_lens::<B>(sequence_lens, kernel_sizes, strides);
    let output_lens = output_lens.into_iter().map(|e| e as i32).collect_vec();
    let output_lens = Tensor::<B, 1, Int>::from_ints(output_lens.as_slice(), device);

    let range: Tensor<B, 2, Int> =
        Tensor::arange(0..feature_vec_len as i64, device).expand([batch_size as i32, -1]);

    let output_lens = output_lens.unsqueeze_dim(1);
    let mask = range.clone().lower(output_lens.clone());

    mask
}

#[test]
fn test_feature_space_attention_mask() {
    let device = NdArrayDevice::Cpu;

    let kernels = [10, 5, 3];
    let strides = [2, 2, 2];

    let feature_size = feature_extractor_output_lens::<NdArray>(vec![100], &kernels, &strides);

    let attention_mask = feature_space_padding_mask::<NdArray>(feature_size[0], vec![50], &kernels, &strides, &device);
}

// hidden: batch x sequence x hidden
fn mask_hidden_time<B: Backend>(
    hidden: Tensor<B, 3>,
    mask_time_indices: Tensor<B, 2, Bool>,
    mask: Tensor<B, 1>,
    _attention_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let hidden_shape = hidden.shape();

    let mask_indices = mask_time_indices
        .unsqueeze_dim::<3>(2)
        .expand(hidden_shape.clone());
    let mask = mask.expand(hidden_shape);

    let hidden = Tensor::mask_where(hidden, mask_indices, mask);

    hidden
}

pub fn mask_hidden_states<B: Backend>(
    hidden: Tensor<B, 3>,
    mask_indices: Tensor<B, 2, Bool>,
    mask_value: Tensor<B, 1>,
    attention_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let hidden = mask_hidden_time(hidden, mask_indices, mask_value, attention_mask);

    hidden
}
