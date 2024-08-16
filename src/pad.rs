use burn::prelude::Backend;
use burn::tensor::{Bool, Int, Tensor};
use unzip3::Unzip3;

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

    assert_eq!(sequence.len(), len);

    (sequence, original_len)
}

pub fn pad_sequence(mut sequence: Vec<f32>, len: usize) -> (Vec<f32>, usize) {
    let original_len = sequence.len();

    assert!(len >= original_len);

    sequence.extend(vec![0.0; len - original_len]);

    assert_eq!(sequence.len(), len);

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

pub struct PaddedSequences<B: Backend> {
    pub sequences: Tensor<B, 2>,
    pub attention_mask: Tensor<B, 2, Bool>,
    pub sequence_lens: Vec<usize>,
}

pub fn pad_sequences<B: Backend>(sequences: Vec<Vec<f32>>, padding: PaddingType, device: &B::Device) -> PaddedSequences<B> {
    let max_len = match padding {
        PaddingType::Explicit(len) => len,
        PaddingType::LongestSequence => sequences.iter().map(Vec::len).max().unwrap()
    };

    let sequences_and_lens = pad_or_trim(sequences, max_len);

    let (sequences, masks, lens) = sequences_and_lens
        .into_iter()
        .map(|(sequence, mut len)| {
            if len >= max_len {
                len = max_len;
            }

            let mut mask = vec![1usize; len];
            let result = usize::checked_sub(sequence.len(), len);
            if result.is_none() {
                panic!("{}", format!("{} - {}", sequence.len(), len));
            }
            mask.extend(vec![0; sequence.len() - len]);
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