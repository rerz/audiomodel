use burn::data::dataloader::Dataset;
use burn::data::dataset::HuggingfaceDatasetLoader;
use rayon::iter::ParallelBridge;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicGenresItem {
    audio_bytes: Vec<u8>,
    audio_path: Option<String>,
    song_id: i64,
    genre_id: i64,
    genre: String,
}

pub struct MusicGenres {}

impl MusicGenres {
    pub fn small() {
        let mut dataset = HuggingfaceDatasetLoader::new("lewtun/music_genres_small")
            .dataset::<MusicGenresItem>("train")
            .unwrap()
            .iter()
            .collect::<Vec<_>>();

        let len = dataset.len();
        let train_split = 0.8;

        let num_train_items = (len as f32 * train_split) as usize;

        let dataset_train = dataset.drain(..num_train_items).collect::<Vec<_>>();
        let dataset_test = dataset.drain(..).collect::<Vec<_>>();
    }

    pub fn normal() {}
}