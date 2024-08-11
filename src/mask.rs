use burn::prelude::{Backend, ElementConversion, Int};
use burn::tensor::{Bool, Distribution, Tensor};
use itertools::Itertools;
use rand::distributions::Uniform;
use rand::Rng;

use crate::model::extractor::feature_extractor_output_lens;

pub fn get_mask_indices<B: Backend>(
    [batch, seq]: [usize; 2],
    seq_lens: Vec<usize>,
    mask_prob: f32,
    mask_len: usize,
    min_masks: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    assert!(mask_len < seq);

    let eps: f32 = rand::thread_rng().sample(Uniform::new(0.0, 1.0));
    let num_masked_spans = |input_len: usize| {
        let num_masked = (mask_prob * input_len as f32 / mask_len as f32 + eps) as usize;
        let mut num_masked = usize::max(num_masked, min_masks);

        if num_masked * mask_len > seq {
            num_masked = seq / mask_len;
        }

        if input_len - (mask_len - 1) < num_masked {
            num_masked = usize::max(input_len - (mask_len - 1), 0);
        }

        num_masked
    };

    let max_num_spans = num_masked_spans(seq);

    let mut mask = vec![];
    for seq_len in seq_lens {
        let num_masked_spans = num_masked_spans(seq_len);

        let mut starting_indices = Tensor::<B, 1>::random(
            [num_masked_spans],
            Distribution::Uniform(0.0, (seq_len - mask_len - 1) as f64),
            device,
        )
            .int();

        let dummy_idx = if starting_indices.dims()[0] == 0 {
            (seq - 1) as u32
        } else {
            starting_indices.to_data().to_vec::<B::IntElem>().unwrap()[0].elem::<u32>()
        };

        let padding = Tensor::ones([max_num_spans.checked_sub(num_masked_spans).unwrap_or(0)], device) * dummy_idx;

        if padding.dims()[0] != 0 {
            starting_indices = Tensor::cat(
                vec![
                    starting_indices,
                    padding, // TODO: figure out why this underflows
                ],
                0,
            );
        }

        let starting_indices = starting_indices.unsqueeze_dim::<2>(0);
        mask.push(starting_indices);
    }

    let mask = Tensor::cat(mask, 0);
    let mask = mask.unsqueeze_dim::<3>(2);
    let mask = mask.expand([batch, max_num_spans, mask_len]);
    let mask = mask.reshape([batch, max_num_spans * mask_len]);

    let offsets =
        Tensor::arange(0..mask_len as i64, device).unsqueeze_dims::<3>(&[0, 1]);
    let offsets = offsets.expand([batch, max_num_spans, mask_len]);
    let offsets = offsets.reshape([batch, max_num_spans * mask_len]);

    let mask = mask.clone() + offsets.clone();

    let mask_indices = Tensor::<B, 2, Int>::zeros([batch, seq], device);
    let mask_indices = Tensor::scatter(
        mask_indices,
        1,
        mask,
        Tensor::ones([batch, max_num_spans * mask_len], device),
    );

    mask_indices.bool()
}

pub fn feature_space_attention_mask<B: Backend>(
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

    let range: Tensor<B, 2, Int> = Tensor::arange(0..feature_vec_len as i64, device)
        .expand([batch_size as i32, -1]);

    let output_lens = output_lens.unsqueeze_dim(1);
    let mask = range.clone().lower(output_lens.clone());

    mask
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

fn mask_hidden_features<B: Backend>() {
    todo!()
}

pub fn mask_hidden_states<B: Backend>(
    hidden: Tensor<B, 3>,
    mask_indices: Tensor<B, 2, Bool>,
    mask_value: Tensor<B, 1>,
    attention_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let hidden = mask_hidden_time(hidden, mask_indices, mask_value, attention_mask);

    // TODO: the default w2v2 config does not use feature masking so not implemented yet

    hidden
}
