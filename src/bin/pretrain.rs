use burn::backend::{Autodiff, LibTorch, Wgpu};
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::wgpu::WgpuDevice;

use audiomodel::model::pretrain::pretrain;

type B = Autodiff<LibTorch>;

pub struct ParquetBatcher {}

//impl Batcher<>

fn main() {
    pretrain::<Autodiff<LibTorch>>(LibTorchDevice::Cuda(0));
}