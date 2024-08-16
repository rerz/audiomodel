use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::{Backend, Bool, Float, Int};
use burn::tensor::{Distribution, Tensor};

fn masked_indices_for_batch_item<B: Backend>(
    sequence_len: usize,
    sequence_len_range: Tensor<B, 1, Int>,
    time_mask: Tensor<B, 2, Bool>,
    batch_item_idx: usize,
) -> Tensor<B, 1, Int> {
    let batch_mask = time_mask
        .slice([batch_item_idx..batch_item_idx + 1, 0..sequence_len])
        .squeeze::<1>(0);
    let batch_range = sequence_len_range
        .select(0, batch_mask.clone().nonzero()[0].clone());

    batch_range
}

fn avoid_sampling_positive<B: Backend>(unpadded_len: usize, num_negatives: usize, negatives: Tensor<B, 2, Int>, device: &B::Device) -> Tensor<B, 2, Int> {
    let feature_range = Tensor::<B, 2, Int>::expand(
        Tensor::arange(0..(unpadded_len as i64) + 1, device).unsqueeze_dim(1),
        [unpadded_len + 1, num_negatives],
    )
        .clone();

    let greater_mask = Tensor::greater_equal(negatives.clone(), feature_range.clone());
    let inced = negatives.clone() + 1;
    let sampled_indices = negatives.mask_where(greater_mask, inced);

    sampled_indices
}

pub fn sample_negative_indices<B: Backend>(
    [batch, seq]: [usize; 2],
    num_negatives: usize,
    mask_time_indices: Tensor<B, 2, Bool>,
    sequence_lens: Vec<usize>,
    device: &B::Device,
) -> Tensor<B, 3, Int> {
    let seq_len_range = Tensor::<B, 1, Int>::arange(0..seq as i64, device);

    let mut sampled_negative_indices =
        Tensor::<B, 3, Int>::zeros([batch, seq, num_negatives], device);

    for (batch_idx, unpadded_len) in (0..batch).zip(sequence_lens) {
        let unpadded_len = unpadded_len - 1;
        let batch_masked_indices = masked_indices_for_batch_item(
            seq,
            seq_len_range.clone(),
            mask_time_indices.clone(),
            batch_idx,
        );

        if batch_masked_indices.dims()[0] == 0 {
            continue;
        }

        let negatives_for_sequence = Tensor::<B, 2>::random(
            [unpadded_len + 1, num_negatives],
            Distribution::Uniform(0.0, unpadded_len as f64),
            device,
        ).int();

        let negatives_for_sequence = negatives_for_sequence + batch_idx as u32 * seq as u32;

        // seq_len x num_negatives
        let negatives_for_sequence = avoid_sampling_positive(unpadded_len, num_negatives, negatives_for_sequence, device);
        let selected_negatives = negatives_for_sequence.select(0, batch_masked_indices.clone());

        let batch_masked_indices = batch_masked_indices.unsqueeze_dim::<2>(1).expand([-1, num_negatives as i32]);

        let batch_negatives = Tensor::<B, 2, Int>::zeros([unpadded_len + 1, num_negatives], device);
        let batch_negatives = Tensor::scatter(batch_negatives, 0, batch_masked_indices, selected_negatives);

        sampled_negative_indices = sampled_negative_indices.clone().slice_assign(
            [
                batch_idx..batch_idx + 1,
                0..(unpadded_len as usize) + 1,
                0..num_negatives,
            ],
            batch_negatives.unsqueeze_dim(0),
        );
    }

    sampled_negative_indices
}

#[test]
fn test_sample_negatives() {
    let device = NdArrayDevice::Cpu;

    sample_negative_indices::<NdArray>(
        [2, 20],
        3,
        Tensor::<NdArray, 2, Int>::ones([2, 20], &device).bool(),
        vec![20, 20],
        &device
    );
}

#[test]
fn test_get_negative_samples() {
        let device = NdArrayDevice::Cpu;

    let negative_indices = sample_negative_indices::<NdArray>(
        [2, 20],
        3,
        Tensor::<NdArray, 2, Int>::ones([2, 20], &device).bool(),
        vec![20, 20],
        &device
    );

    get_negative_samples(negative_indices, Tensor::random([2, 20, 5], Distribution::Default, &device));
}

pub fn get_negative_samples<B: Backend>(
    sampled_negative_indices: Tensor<B, 3, Int>,
    features: Tensor<B, 3, Float>,
) -> Tensor<B, 4> {
    let [batch, seq, hidden] = features.dims();
    let [_, _, num_negatives] = sampled_negative_indices.dims();

    let features = features.reshape([-1, hidden as i32]);

    let negative_samples = Tensor::select(features.clone(), 0, sampled_negative_indices.reshape([-1]));

    let negative_samples = negative_samples.reshape([batch as i32, seq as i32, -1, hidden as i32]);
    let negative_samples = negative_samples.permute([2, 0, 1, 3]);
    // add hidden dim
    // let sampled_negative_indices =
    //     sampled_negative_indices.clone()
    //         .unsqueeze_dim::<4>(3)
    //         .expand([-1, -1, -1, hidden as i32]);
    //
    // // add num_negatives dim
    // let features = features
    //     .expand([num_negatives as i32, -1, -1, -1])
    //     .permute([1, 2, 0, 3]);
    //
    // let negative_features = Tensor::gather(features, 1, sampled_negative_indices);
    // let negative_features = negative_features.permute([2, 0, 1, 3]);
    //
    // let negative_dims = negative_features.dims();

    //negative_features
    negative_samples
}
