

//impl Batcher<>

use burn::backend::{Autodiff, LibTorch, NdArray, Wgpu};
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::data::dataloader::Dataset;
use burn::data::dataset::{SqliteDataset, SqliteDatasetWriter};
use audiomodel::data::{AudioDataset, AudioSample};
use audiomodel::io::resample;

fn main() {
    let dataset_train = SqliteDataset::<AudioSample>::from_db_file("music_genres.sqlite", "train").unwrap();
    let dataset_test = SqliteDataset::<AudioSample>::from_db_file("music_genres.sqlite", "test").unwrap();

    audiomodel::train::pretrain::<Autodiff<LibTorch>>(LibTorchDevice::Cuda(0), 60_000, dataset_train, dataset_test);
}