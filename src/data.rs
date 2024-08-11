use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{InMemDataset, SqliteDataset};
use burn::prelude::{Backend, Bool, Int, Tensor};
use soa_derive::StructOfArray;

use crate::mask::get_mask_indices;
use crate::mine::sample_negative_indices;
use crate::model::AudioModelConfig;
use crate::model::extractor::feature_extractor_output_lens;
use crate::pad::{pad_sequences, PaddingType};

pub struct AudioDataset {
    inner: SqliteDataset<AudioItem>,
}

pub struct AudioDatasetWithNegatives<B: Backend> {
    inner: InMemDataset<AudioItemWithMaskAndNegatives<B>>
}

#[derive(Clone, Debug)]
pub struct AudioBatch<B: Backend> {
    pub device: B::Device,
    pub sequences: Tensor<B, 2>,
    pub sequence_lens: Vec<usize>,
    pub attention_mask: Tensor<B, 2, Bool>,
    pub masked_time_indices: Tensor<B, 2, Bool>,
    pub sampled_negative_indices: Tensor<B, 3, Int>,
}

#[derive(Clone)]
pub struct AudioBatcher<B: Backend> {
    config: AudioModelConfig,
    device: B::Device,
}

impl<B: Backend> AudioBatcher<B> {
    pub fn new(config: AudioModelConfig, device: B::Device) -> Self {
        Self {
            config,
            device
        }
    }
}

impl<B: Backend> Batcher<AudioItem, AudioBatch<B>> for AudioBatcher<B> {
    fn batch(&self, items: Vec<AudioItem>) -> AudioBatch<B> {
        let batch = items.len();

        let soa = AudioItemVec::from_iter(items);

        let padded = pad_sequences(soa.audio.clone(), PaddingType::Explicit(10_000), &self.device);
        let max_seq_len = padded.sequences.dims()[1];

        let padded_seq_lens = padded.sequence_lens.clone();

        let extracted_features_seq_len = feature_extractor_output_lens::<B>(
            vec![max_seq_len],
            &self.config.feature_extractor_config.conv_kernels,
            &self.config.feature_extractor_config.conv_strides
        )[0];

        let extracted_seq_lens = feature_extractor_output_lens::<B>(
            padded_seq_lens,
            &self.config.feature_extractor_config.conv_kernels,
            &self.config.feature_extractor_config.conv_strides,
        );

        let masked_time_indices = get_mask_indices(
            [batch, extracted_features_seq_len],
            extracted_seq_lens.clone(),
            0.2,
            3,
            0,
            &self.device,
        );

        let sampled_negative_indices =
            sample_negative_indices(
                [batch, extracted_features_seq_len],
                11,
                masked_time_indices.clone(),
                extracted_seq_lens,
                &self.device
            );

        AudioBatch {
            device: self.device.clone(),
            sequences: padded.sequences,
            attention_mask: padded.attention_mask,
            sequence_lens: padded.sequence_lens,
            masked_time_indices,
            sampled_negative_indices,
        }
    }
}

impl<B: Backend> AudioBatch<B> {
    pub fn input_len(&self) -> usize {
        self.sequences.dims()[1]
    }
}

#[derive(StructOfArray, Clone, Debug)]
pub struct AudioItem {
    pub audio: Vec<f32>,
    pub seq_len: usize,
    pub song_id: usize,
    pub genre_id: usize,
    pub genre: String,
    pub sr: u32,
}

pub struct AudioItemWithMaskAndNegatives<B: Backend> {
    item: AudioItem,
    mask: Tensor<B, 1, Int>
}