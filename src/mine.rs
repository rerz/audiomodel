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
        Tensor::arange(0..(unpadded_len as i64), device).unsqueeze_dim(1),
        [unpadded_len, num_negatives],
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

        let masked_dims = batch_masked_indices.dims();

        if batch_masked_indices.dims()[0] == 0 {
            continue;
        }

        let negatives_for_sequence = Tensor::<B, 2>::random(
            [seq, num_negatives],
            Distribution::Uniform(0.0, (seq - 1) as f64),
            device,
        ).int();

        // TODO: is this needed?
        //let negatives_for_sequence = negatives_for_sequence + batch_idx as u32 * seq as u32;

        // seq_len x num_negatives
        let negatives_for_sequence = avoid_sampling_positive(seq, num_negatives, negatives_for_sequence, device);
        let selected_negatives = negatives_for_sequence.select(0, batch_masked_indices.clone());

        let batch_masked_indices = batch_masked_indices.unsqueeze_dim::<2>(1).expand([-1, num_negatives as i32]);

        let batch_negatives = Tensor::<B, 2, Int>::zeros([seq, num_negatives], device);
        let batch_negatives = Tensor::scatter(batch_negatives, 0, batch_masked_indices, selected_negatives);

        sampled_negative_indices = sampled_negative_indices.clone().slice_assign(
            [
                batch_idx..batch_idx + 1,
                0..(seq as usize),
                0..num_negatives,
            ],
            batch_negatives.unsqueeze_dim(0),
        );
    }

    let sampled_negative_dims = sampled_negative_indices.dims();
    // batch x time x num_negatives

    // tensor where masked time steps each have n negative indices, else all 0
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

    get_negative_samples(negative_indices, Tensor::random([2, 20, 1, 5], Distribution::Default, &device));
}

pub fn get_negative_samples<B: Backend>(
    sampled_negative_indices: Tensor<B, 3, Int>,
    targets: Tensor<B, 3, Float>,
) -> Tensor<B, 4> {
    let [batch, time, num_negatives] = sampled_negative_indices.dims();
    let [batch, masked_time,  hidden] = targets.dims();

    let sampled_negative_indices = sampled_negative_indices.unsqueeze_dim::<4>(3).expand([batch, time, num_negatives, hidden]);
    let targets = targets.unsqueeze_dim::<4>(2).expand([batch, time, num_negatives, hidden]);

    let idx_dims = sampled_negative_indices.dims();
    let tgt_dims = targets.dims();

    let negative_features = Tensor::gather(targets, 1, sampled_negative_indices);
    let neg_dims = negative_features.dims();
    let negative_features = negative_features.permute([2, 0, 1, 3]);

    let negative_dims = negative_features.dims();

    //negative_features

    // batch x time x num_negatives x hidden
    negative_features
}
