use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::Dataset;
use burn::data::dataset::{HuggingfaceDatasetLoader, SqliteDatasetWriter};
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;

use audiomodel::data::{AudioSample, MusicGenresItem};
use audiomodel::io::resample;

type B = Autodiff<Wgpu>;

fn main() {
    let mut resampled_dataset = SqliteDatasetWriter::new("music_genres.sqlite", false).unwrap();

    HuggingfaceDatasetLoader::new("lewtun/music_genres")
        .dataset::<MusicGenresItem>("train")
        .unwrap()
        .iter()
        .par_bridge()
        .map(|track| track.into())
        .for_each(|item: AudioSample| {
            resampled_dataset
                .write(
                    "train",
                    &AudioSample {
                        audio: resample(item.audio, item.sr, 16_000),
                        sr: 16_000,
                        id: item.id,
                        group_label: item.group_label,
                    },
                )
                .unwrap();
        });

    HuggingfaceDatasetLoader::new("lewtun/music_genres")
        .dataset::<MusicGenresItem>("test")
        .unwrap()
        .iter()
        .par_bridge()
        .map(|track| track.into())
        .for_each(|item: AudioSample| {
            resampled_dataset
                .write(
                    "test",
                    &AudioSample {
                        audio: resample(item.audio, item.sr, 16_000),
                        sr: 16_000,
                        id: item.id,
                        group_label: item.group_label,
                    },
                )
                .unwrap();
        });

    resampled_dataset.set_completed().unwrap();
}
