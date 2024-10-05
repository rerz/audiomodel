//impl Batcher<>

use burn::backend::{Autodiff, LibTorch, Wgpu};
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::data::dataset::SqliteDataset;

use audiomodel::config::small_music::small_music_config;
use audiomodel::data::{AudioDataset, AudioSample};
use audiomodel::mask::block::BlockMaskConfig;
use audiomodel::train::{ConfigBundle, TrainConfig};

type B = Wgpu;

fn main() {
    let dataset_train = AudioDataset::music_genres_small();
    let dataset_test = AudioDataset::music_genres_small();
    // samples ~500_000 len

    let mask_config = BlockMaskConfig {
        mask_prob: 0.65,
        mask_len: 10,
        min_masks: 3,
    };

    let model_config = small_music_config::<B>();

    audiomodel::train::pretrain::<Autodiff<B>, _, _>(
        WgpuDevice::DiscreteGpu(0),
        TrainConfig {
            mask_config,
            input_len: 50_000,
        },
        model_config,
        dataset_train.inner,
        dataset_test.inner,
    );
}