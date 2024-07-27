use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};
use unzip3::Unzip3;

use crate::data::AudioBatch;

pub enum Padding {
    Left(usize),
    Right(usize),
    Both(usize, usize),
}

pub enum PaddingType {
    LongestSequence,
    Explicit(usize),
}

pub fn trim_sequence(mut sequence: Vec<f32>, len: usize) -> (Vec<f32>, usize) {
    let original_len = sequence.len();

    assert!(len < original_len);

    sequence.truncate(len);

    (sequence, original_len)
}

pub fn pad_sequence(mut sequence: Vec<f32>, len: usize) -> (Vec<f32>, usize) {
    let original_len = sequence.len();

    assert!(len >= original_len);

    sequence.extend(vec![0.0; len - original_len]);

    (sequence, original_len)
}

fn pad_or_trim(sequences: Vec<Vec<f32>>, len: usize) -> Vec<(Vec<f32>, usize)> {
    sequences
        .into_iter()
        .map(|sequence| match sequence.len() {
            original_len if len < original_len => trim_sequence(sequence, len),
            original_len if len >= original_len => pad_sequence(sequence, len),
            original_len => (sequence, original_len),
        })
        .collect::<Vec<_>>()
}

pub fn pad_sequences<B: Backend>(sequences: Vec<Vec<f32>>, padding: PaddingType) -> AudioBatch<B> {
    let sequences_and_lens = match padding {
        PaddingType::Explicit(length) => pad_or_trim(sequences, length),
        PaddingType::LongestSequence => {
            let max_len = sequences.iter().map(Vec::len).max().unwrap();
            pad_or_trim(sequences, max_len)
        }
    };

    let (sequences, masks, lens) = sequences_and_lens
        .into_iter()
        .map(|(sequence, len)| {
            let mut mask = vec![1; len];
            mask.extend(vec![0; sequence.len() - len]);

            let sequence = Tensor::<B, 1>::from_floats(&*sequence, &B::Device::default());
            let mask = Tensor::<B, 1, Int>::from_ints(&*mask, &B::Device::default());

            (sequence, mask, len)
        })
        .unzip3();

    AudioBatch {
        sequences: Tensor::stack(sequences, 0),
        attention_mask: Tensor::stack(masks, 0),
        sequence_lens: lens,
    }
}