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

    let train_samples = HuggingfaceDatasetLoader::new("lewtun/music_genres")
        .dataset::<MusicGenresItem>("train")
        .unwrap()
        .iter()
        .par_bridge()
        .map(|track| track.into())
        .map(|item: AudioSample| AudioSample {
                        audio: resample(item.audio, item.sr, 16_000),
                        sr: 16_000,
                        id: item.id,
                        group_label: item.group_label,
                    })
        .collect::<Vec<_>>();

    resampled_dataset.write_all("train", &train_samples).unwrap();

    let test_samples = HuggingfaceDatasetLoader::new("lewtun/music_genres")
        .dataset::<MusicGenresItem>("test")
        .unwrap()
        .iter()
        .par_bridge()
        .map(|track| track.into())
        .map(|item: AudioSample| AudioSample {
                        audio: resample(item.audio, item.sr, 16_000),
                        sr: 16_000,
                        id: item.id,
                        group_label: item.group_label,
                    }).collect::<Vec<_>>();

    resampled_dataset.write_all("test", &test_samples).unwrap();

    resampled_dataset.set_completed().unwrap();
}
