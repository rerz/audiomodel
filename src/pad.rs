use burn::prelude::Backend;
use burn::tensor::{Bool, Int, Tensor};
use itertools::Itertools;
use unzip3::Unzip3;
use crate::model::arch::extractor::feature_extractor_output_lens;

pub fn feature_space_padding_mask<B: Backend>(
    feature_vec_len: u32,
    sequence_lens: Vec<u32>,
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

pub enum Padding {
    Left(usize),
    Right(usize),
    Both(usize, usize),
}

pub enum PaddingType {
    LongestSequence,
    Explicit(u32),
}

pub fn trim_sequence(mut sequence: Vec<f32>, len: u32) -> (Vec<f32>, u32) {
    let original_len = sequence.len() as u32;

    if len <= original_len {
        return (sequence, original_len);
    }

    sequence.truncate(len as usize);

    (sequence, original_len)
}

pub fn pad_sequence(mut sequence: Vec<f32>, len: u32) -> (Vec<f32>, u32) {
    let original_len = sequence.len() as u32;

    assert!(len >= original_len);

    sequence.extend(vec![0.0; (len - original_len) as usize]);

    (sequence, original_len)
}

fn pad_or_trim(sequences: Vec<Vec<f32>>, len: u32) -> Vec<(Vec<f32>, u32)> {
    sequences
        .into_iter()
        .map(|sequence| match sequence.len() {
            original_len if len < original_len as u32 => trim_sequence(sequence, len),
            original_len if len >= original_len as u32 => pad_sequence(sequence, len),
            original_len => (sequence, original_len as u32),
        })
        .collect::<Vec<_>>()
}

pub struct PaddedSequences<B: Backend> {
    pub sequences: Tensor<B, 2>,
    pub attention_mask: Tensor<B, 2, Bool>,
    pub sequence_lens: Vec<u32>,
}

pub fn pad_sequences<B: Backend>(sequences: Vec<Vec<f32>>, padding: PaddingType, device: &B::Device) -> PaddedSequences<B> {
    let max_len = match padding {
        PaddingType::Explicit(len) => len,
        PaddingType::LongestSequence => sequences.iter().map(Vec::len).max().unwrap() as u32
    };

    let sequences_and_lens = pad_or_trim(sequences, max_len);

    let (sequences, masks, lens) = sequences_and_lens
        .into_iter()
        .map(|(sequence, mut len)| {
            if len >= max_len {
                len = max_len;
            }

            let mut mask = vec![1usize; len as usize];
            let result = u32::checked_sub(sequence.len() as u32, len);
            if result.is_none() {
                panic!("{}", format!("{} - {}", sequence.len(), len));
            }
            mask.extend(vec![0; sequence.len() - len as usize]);
            let sequence = Tensor::<B, 1>::from_floats(&*sequence, device);
            let mask = Tensor::<B, 1, Int>::from_ints(&*mask, device).bool();

            (sequence, mask, len)
        })
        .unzip3();

    PaddedSequences {
        sequences: Tensor::stack(sequences, 0),
        attention_mask: Tensor::stack(masks, 0),
        sequence_lens: lens,
    }
}