use std::collections::{HashMap, HashSet};

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::data::dataset::{HuggingfaceDatasetLoader, InMemDataset, SqliteDataset};
use burn::prelude::{Backend, Bool, Int, Tensor};
use itertools::Itertools;
use rayon::iter::ParallelBridge;
use samplerate::ConverterType;
use serde::{Deserialize, Serialize};
use soa_derive::StructOfArray;

use crate::io::read_bytes_inferred;
use crate::mask::block::{BlockMask, BlockMaskConfig};
use crate::mask::MaskingStrategy;
use crate::mine::sample_negative_indices;
use crate::model::AudioModelConfig;
use crate::model::extractor::feature_extractor_output_lens;
use crate::pad::{pad_sequences, PaddingType};

pub struct AudioDataset {
    pub inner: InMemDataset<AudioSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicGenresItem {
    audio_bytes: Vec<u8>,
    audio_path: Option<String>,
    song_id: i64,
    genre_id: i64,
    genre: String,
}

impl From<MusicGenresItem> for AudioSample {
    fn from(value: MusicGenresItem) -> Self {
        let (audio, sr) = read_bytes_inferred(&value.audio_bytes, 16_000);
        AudioSample {
            audio,
            sr,
            id: value.song_id,
            group_label: value.genre_id,
        }
    }
}

use rayon::iter::ParallelIterator;

impl AudioDataset {
    pub fn music_genres() -> (AudioDataset, AudioDataset) {
        let dataset_train = HuggingfaceDatasetLoader::new("lewtun/music_genres")
            .dataset::<MusicGenresItem>("train")
            .unwrap()
            .iter()
            .par_bridge()
            .map(|track| track.into())
            .collect::<Vec<_>>();

        let dataset_test = HuggingfaceDatasetLoader::new("lewtun/music_genres")
            .dataset::<MusicGenresItem>("test")
            .unwrap()
            .iter()
            .par_bridge()
            .map(|track| track.into())
            .collect::<Vec<_>>();

        (Self { inner: InMemDataset::new(dataset_train) }, Self { inner: InMemDataset::new(dataset_test) })
    }

    pub fn music_genres_small() -> AudioDataset {
        let dataset = HuggingfaceDatasetLoader::new("lewtun/music_genres_small")
            .dataset::<MusicGenresItem>("train")
            .unwrap()
            .iter()
            .par_bridge()
            .map(|track| track.into())
            .collect::<Vec<_>>();

        Self {
            inner: InMemDataset::new(dataset),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AudioBatch<B: Backend> {
    pub device: B::Device,
    pub sequences: Tensor<B, 2>,
    pub sequence_lens: Vec<usize>,
    pub padding_mask: Tensor<B, 2, Bool>,
    pub masked_time_indices: Tensor<B, 2, Bool>,
    pub sampled_negative_indices: Tensor<B, 3, Int>,
}

#[derive(Clone)]
pub struct AudioBatcher<B: Backend, M: MaskingStrategy<B>> {
    padding_len: usize,
    config: AudioModelConfig,
    device: B::Device,
    mask_config: M::Config,
}

impl<B: Backend, M: MaskingStrategy<B>> AudioBatcher<B, M> {
    pub fn new(
        pad_to_len: usize,
        config: AudioModelConfig,
        mask_config: M::Config,
        device: B::Device,
    ) -> Self {
        Self {
            config,
            device,
            padding_len: pad_to_len,
            mask_config,
        }
    }
}

impl<B: Backend, M: MaskingStrategy<B>> Batcher<AudioSample, AudioBatch<B>> for AudioBatcher<B, M> {
    fn batch(&self, items: Vec<AudioSample>) -> AudioBatch<B> {
        let batch = items.len();

        let audio = items.into_iter().map(|audio| audio.audio).collect_vec();

        let padded = pad_sequences(
            audio.clone(),
            PaddingType::Explicit(self.padding_len),
            &self.device,
        );
        let max_seq_len = padded.sequences.dims()[1];

        let padded_seq_lens = padded.sequence_lens.clone();

        let extracted_features_seq_len = feature_extractor_output_lens::<B>(
            vec![max_seq_len],
            &self.config.feature_extractor_config.conv_kernels,
            &self.config.feature_extractor_config.conv_strides,
        )[0];

        let extracted_seq_lens = feature_extractor_output_lens::<B>(
            padded_seq_lens,
            &self.config.feature_extractor_config.conv_kernels,
            &self.config.feature_extractor_config.conv_strides,
        );

        let masked_time_indices = M::get_mask_indices(
            [batch, extracted_features_seq_len],
            extracted_seq_lens.clone(),
            self.mask_config.clone(),
            &self.device,
        );

        let mask_dims = masked_time_indices.dims();

        let num = masked_time_indices.clone().nonzero()[1].clone().dims()[0];

        let sampled_negative_indices = sample_negative_indices(
            [batch, num],
            100,
            masked_time_indices.clone(),
            extracted_seq_lens,
            &self.device,
        );

        //println!("batch seq dims {:?}", padded.sequences.dims());

        AudioBatch {
            device: self.device.clone(),
            sequences: padded.sequences,
            padding_mask: padded.attention_mask,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioSample {
    pub audio: Vec<f32>,
    pub id: i64,
    pub group_label: i64,
    pub sr: u32,
}
