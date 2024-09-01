

//impl Batcher<>

use burn::backend::{Autodiff, Candle, CudaJit, LibTorch, NdArray, Wgpu};
use burn::backend::candle::CandleDevice;
use burn::backend::cuda_jit::CudaDevice;
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
    // samples ~500_000 len

    audiomodel::train::pretrain::<Autodiff<LibTorch>>(LibTorchDevice::Cuda(0), 100_000, dataset_train, dataset_test);
}