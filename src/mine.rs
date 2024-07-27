use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::{Backend, Bool, Float, Int};
use burn::tensor::{Distribution, Tensor};
use itertools::Itertools;
use rand::{Rng, thread_rng};

use crate::pad::pad_sequences;

pub fn sample_negative_indices<B: Backend>(
    [batch, seq]: [usize; 2],
    num_negatives: usize,
    mask_time_indices: Tensor<B, 2, Bool>,
    sequence_lens: Vec<usize>,
) -> Tensor<B, 3, Int> {
    let seq_len_range = Tensor::<B, 1, Int>::arange(0..seq as i64, &B::Device::default());

    let mut sampled_negative_indices =
        Tensor::<B, 3, Int>::zeros([batch, seq, num_negatives], &B::Device::default());

    for (batch, seq_len) in (0..batch).zip(sequence_lens) {
        let seq_len = seq_len - 1;
        let batch_mask = mask_time_indices
            .clone()
            .slice([batch..batch + 1, 0..seq])
            .squeeze::<1>(0);

        let batch_range = seq_len_range
            .clone()
            .select(0, batch_mask.clone().nonzero()[0].clone());

        let feature_range = Tensor::<B, 2, Int>::expand(
            Tensor::arange(0..(seq_len as i64) + 1, &B::Device::default()).unsqueeze_dim(1),
            [(seq_len + 1) as usize, num_negatives],
        )
        .clone();

        let sampled_indices = Tensor::<B, 2>::random(
            [(seq_len + 1) as usize, num_negatives],
            Distribution::Uniform(0.0, seq_len as f64),
            &B::Device::default(),
        )
        .int();

        let greater_mask = Tensor::greater_equal(sampled_indices.clone(), feature_range);
        let inced = sampled_indices.clone() + 1;
        let sampled_indices = sampled_indices.mask_where(greater_mask, inced);

        let expanded = batch_range.expand([sampled_indices.dims()[0] as i32, -1]);
        let result = Tensor::gather(expanded, 1, sampled_indices);

        sampled_negative_indices = sampled_negative_indices.clone().slice_assign(
            [
                batch..batch + 1,
                0..(seq_len as usize) + 1,
                0..num_negatives,
            ],
            result.unsqueeze_dim(0),
        );
    }

    sampled_negative_indices
}

pub fn get_negative_samples<B: Backend>(
    sampled_negative_indices: Tensor<B, 3, Int>,
    features: Tensor<B, 3, Float>,
) -> Tensor<B, 4> {
    let [_, _, hidden] = features.dims();
    let [_, _, num_negatives] = sampled_negative_indices.dims();

    // add hidden dim
    let expanded_idx =
        sampled_negative_indices
            .unsqueeze_dim::<4>(3)
            .expand([-1, -1, -1, hidden as i32]);

    // add num_negatives dim
    let features = features
        .unsqueeze_dim::<4>(2)
        .expand([-1, -1, num_negatives as i32, -1]);

    let negative_features = Tensor::gather(features, 1, expanded_idx);
    let negative_features = negative_features.permute([2, 0, 1, 3]);

    negative_features
}
