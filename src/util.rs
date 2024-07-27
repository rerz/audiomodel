use burn::prelude::Backend;
use itertools::Itertools;
use rand::{Rng, thread_rng};
use crate::data::AudioBatch;
use crate::pad::{pad_sequences, PaddingType};

pub(crate) fn sample_sequence(min_len: usize, max_len: usize) -> Vec<f32> {
    let seq_len = thread_rng().gen_range(min_len..max_len);
    let seq = thread_rng()
        .sample_iter(rand::distributions::Uniform::new(0.0, 1.0))
        .take(seq_len)
        .collect_vec();

    seq
}

pub(crate) fn sample_test_batch<B: Backend>(
    batch_size: usize,
    min_len: usize,
    max_len: usize,
) -> AudioBatch<B> {
    let seqs = (0..batch_size)
        .map(|idx| sample_sequence(min_len, max_len))
        .collect_vec();
    let padded = pad_sequences::<B>(seqs, PaddingType::LongestSequence);
    padded
}