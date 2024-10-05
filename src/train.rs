use burn::data::dataloader::{DataLoaderBuilder, Dataset};
use burn::data::dataset::SqliteDataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module, ModuleDisplay};
use burn::optim::AdamWConfig;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::train::LearnerBuilder;
use burn::train::metric::{LearningRateMetric, LossMetric};
use ndarray::Data;

use crate::data::{AudioBatcher, AudioSample};
use crate::mask::block::{BlockMask, BlockMaskConfig};
use crate::metric::code_perplexity::CodePerplexityMetric;
use crate::metric::contrastive_loss::ContrastiveLossMetric;
use crate::metric::correct::CorrectMetric;
use crate::metric::diversity_loss::DiversityLossMetric;
use crate::metric::gradnorm::GradientNormMetric;
use crate::metric::maskingratio::MaskingRatio;
use crate::metric::perplexity::PerplexityMetric;
use crate::metric::temperature::TemperatureMetric;
use crate::model::AudioModel;
use crate::model::encoder::{Encoder, EncoderConfig};
use crate::model::pretrain::{Pretrain, PretrainConfig};
use crate::model::quantizer::{Quantizer, QuantizerConfig};
use crate::ops::PolynomialDecay;

pub struct TrainConfig {
    pub input_len: u32,
    pub mask_config: BlockMaskConfig,
}

pub struct ConfigBundle<EC, QC> {
    pub pretrain_config: PretrainConfig,
    pub encoder_config: EC,
    pub quantizer_config: QC,
}

impl<EC: EncoderConfig, QC: QuantizerConfig> ConfigBundle<EC, QC> {
    pub fn init<B: Backend>(self, input_len: u32, device: &B::Device) -> Pretrain<B, <EC as EncoderConfig>::Model<B>, <QC as QuantizerConfig>::Model<B>> {
        let model: Pretrain<B, _, _> =
            self.pretrain_config
            .clone()
            .init(input_len, self.encoder_config, self.quantizer_config, device);

        model
    }
}

pub trait ModelConstructor {
    type Backend: Backend;
    type Model<B>;
}

type Model<C> = <C as ModelConstructor>::Model<<C as ModelConstructor>::Backend>;

pub fn pretrain<B: AutodiffBackend, EC: EncoderConfig, QC: QuantizerConfig>(
    device: B::Device,
    train_config: TrainConfig,
    model_configs: ConfigBundle<EC, QC>,
    dataset_train: impl Dataset<AudioSample> + 'static,
    dataset_test: impl Dataset<AudioSample> + 'static,
) -> AudioModel<B, <EC as EncoderConfig>::Model<B>>
    where
        <EC as EncoderConfig>::Model<B>: AutodiffModule<B> + ModuleDisplay,
        <<EC as EncoderConfig>::Model<B> as AutodiffModule<B>>::InnerModule: ModuleDisplay,
        <QC as QuantizerConfig>::Model<B>: AutodiffModule<B> + ModuleDisplay,
        <<QC as QuantizerConfig>::Model<B> as AutodiffModule<B>>::InnerModule: ModuleDisplay,
        <<EC as EncoderConfig>::Model<B> as AutodiffModule<B>>::InnerModule:
        Encoder<<B as AutodiffBackend>::InnerBackend>,
        <<QC as QuantizerConfig>::Model<B> as AutodiffModule<B>>::InnerModule:
        Quantizer<<B as AutodiffBackend>::InnerBackend>,
        <EC as EncoderConfig>::Model<B>: 'static,
        <QC as QuantizerConfig>::Model<B>: 'static,
{
    //tch::maybe_init_cuda();

    let input_len = train_config.input_len;
    let mask_config = train_config.mask_config;

    let ConfigBundle {
        pretrain_config,
        quantizer_config,
        encoder_config,
    } = model_configs;

    let model: Pretrain<B, _, _> =
        pretrain_config
            .clone()
            .init(input_len, encoder_config, quantizer_config, &device);

    let batcher = AudioBatcher::<B, BlockMask>::training(
        input_len,
        100,
        pretrain_config.model_config.clone(),
        mask_config.clone(),
        true,
        device.clone(),
    );
    let data_loader_train = DataLoaderBuilder::new(batcher)
        .batch_size(4)
        .num_workers(1)
        .shuffle(0)
        .build(dataset_train);

    let batcher = AudioBatcher::<B::InnerBackend, BlockMask>::training(
        input_len,
        100,
        pretrain_config.model_config,
        mask_config.clone(),
        true,
        device.clone(),
    );
    let data_loader_test = DataLoaderBuilder::new(batcher)
        .batch_size(4)
        .num_workers(1)
        .shuffle(0)
        .build(dataset_test);

    //TrainStep::step(&model, data_loader_train.iter().next().unwrap());

    let optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let scheduler = PolynomialDecay::new(0.005, 0.0, 1.0, 600_000, 200_000);

    let learner = LearnerBuilder::new(".out")
        .devices(vec![device])
        // train metrics
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(PerplexityMetric::<B>::default())
        .metric_train_numeric(GradientNormMetric::<B>::default())
        .metric_train_numeric(LearningRateMetric::default())
        .metric_train_numeric(TemperatureMetric::<B>::default())
        .metric_train_numeric(CorrectMetric::<B>::default())
        .metric_train_numeric(MaskingRatio::<B>::default())
        .metric_train_numeric(ContrastiveLossMetric::<B>::default())
        .metric_train_numeric(DiversityLossMetric::<B>::default())
        // validation metrics
        .metric_valid_numeric(LossMetric::new())
        .metric_valid_numeric(DiversityLossMetric::<B>::default())
        .metric_valid_numeric(ContrastiveLossMetric::<B>::default())
        .metric_valid_numeric(PerplexityMetric::<B>::default())
        .metric_valid_numeric(CodePerplexityMetric::<B>::default())
        .metric_valid_numeric(CorrectMetric::<B>::default())
        .num_epochs(20)
        .summary()
        .build(model, optimizer, scheduler);
    let trained = learner.fit(data_loader_train, data_loader_test);

    let encoder = trained.model;

    encoder
}
