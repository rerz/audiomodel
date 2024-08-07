use burn::backend::{Autodiff, LibTorch, NdArray};
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::libtorch::LibTorchDevice;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamWConfig;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::train::LearnerBuilder;
use audiomodel::model::AudioModelConfig;
use audiomodel::model::encoder::transformer::{TransformerEncoder, TransformerEncoderConfig};
use audiomodel::model::extractor::FeatureExtractorConfig;
use audiomodel::model::pretrain::{Pretrain, pretrain, PretrainConfig};
use audiomodel::model::projection::FeatureProjectionConfig;
use audiomodel::model::quantizer::gumbel::{GumbelQuantizer, GumbelQuantizerConfig};
use audiomodel::util::download_hf_dataset;

type B = Autodiff<LibTorch>;

pub struct ParquetBatcher {

}

//impl Batcher<>

fn main() {
    pretrain::<Autodiff<LibTorch>>(LibTorchDevice::default());
}