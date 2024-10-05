mod music_genres;

use std::collections::{HashMap, HashSet};

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::data::dataset::{HuggingfaceDatasetLoader, InMemDataset, SqliteDataset};
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::{Backend, Bool, Int, Tensor};
use itertools::Itertools;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use samplerate::ConverterType;
use serde::{Deserialize, Serialize};
use soa_derive::StructOfArray;
use burnx::sequence::mask::{MaskingStrategy, NoneMask};
use burnx::sequence::mine::SamplingStrategy;

use crate::io::read_bytes_inferred;
use crate::mine::sample_negative_indices;
use crate::model::AudioModelConfig;
use crate::model::extractor::feature_extractor_output_lens;
use crate::pad::{pad_sequences, PaddingType};



pub struct AudioDataset<D: Dataset<AudioSample>> {
    pub inner: D,
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

impl<D: Dataset<AudioSample>> AudioDataset<D> {
    pub fn music_genres() -> (AudioDataset<D>, AudioDataset<D>) {
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

        (
            Self {
                inner: InMemDataset::new(dataset_train),
            },
            Self {
                inner: InMemDataset::new(dataset_test),
            },
        )
    }

    pub fn music_genres_small() -> AudioDataset {


        Self {
            inner: InMemDataset::new(dataset),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AudioBatch<B: Backend> {
    pub device: B::Device,
    pub sequences: Tensor<B, 2>,
    pub sequence_lens: Vec<u32>,
    pub padding_mask: Tensor<B, 2, Bool>,
    pub masked_time_steps: Option<Tensor<B, 2, Bool>>,
    pub sampled_negative_indices: Tensor<B, 3, Int>,
}

#[derive(Clone)]
pub struct AudioBatcher<B: Backend, M: MaskingStrategy<B>> {
    pad_to_len: u32,
    num_negatives: usize,
    config: AudioModelConfig,
    device: B::Device,
    mask_config: M::Config,
    apply_mask: bool,
    norm: LayerNorm<B>,
}

impl<B: Backend> AudioBatcher<B, NoneMask> {
    pub fn inference(pad_to_len: u32, config: AudioModelConfig, device: B::Device) -> Self {
        let norm = LayerNormConfig::new(pad_to_len as usize).init(&device);
        Self {
            config,
            norm,
            pad_to_len,
            apply_mask: false,
            mask_config: (),
            num_negatives: 0,
            device,
        }
    }
}

impl<B: Backend, M: MaskingStrategy<B>> AudioBatcher<B, M> {
    pub fn pretrain(
        pad_to_len: u32,
        num_negatives: usize,
        config: AudioModelConfig,
        mask_config: M::Config,
        apply_mask: bool,
        device: B::Device,
    ) -> Self {
        let norm = LayerNormConfig::new(pad_to_len as usize).init(&device);
        Self {
            config,
            num_negatives,
            device,
            pad_to_len,
            mask_config,
            apply_mask,
            norm,
        }
    }
}

impl<B: Backend, M: MaskingStrategy<B>, S: SamplingStrategy<B>> Batcher<AudioSample, AudioBatch<B>> for AudioBatcher<B, M> {
    fn batch(&self, items: Vec<AudioSample>) -> AudioBatch<B> {
        let batch = items.len();

        let audio = items.into_iter().map(|audio| audio.audio).collect_vec();

        let padded = pad_sequences(
            audio.clone(),
            PaddingType::Explicit(self.pad_to_len),
            &self.device,
        );
        let max_seq_len = padded.sequences.dims()[1] as u32;

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

        let masked_time_indices = if self.apply_mask {
            Some(M::get_mask_indices(
                [batch, extracted_features_seq_len as usize],
                extracted_seq_lens.clone(),
                self.mask_config.clone(),
                &self.device,
            ))
        } else {
            None
        };

        let num = masked_time_indices.clone().unwrap().nonzero()[1].clone().dims()[0];

        let sampled_negative_indices = sample_negative_indices(
            [batch, num],
            self.num_negatives,
            masked_time_indices.clone().unwrap(),
            extracted_seq_lens,
            &self.device,
        );

        //println!("batch seq dims {:?}", padded.sequences.dims());

        let sequences = self.norm.clone().no_grad().forward(padded.sequences);

        AudioBatch {
            device: self.device.clone(),
            sequences,
            padding_mask: padded.attention_mask,
            sequence_lens: padded.sequence_lens,
            masked_time_steps: masked_time_indices,
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
