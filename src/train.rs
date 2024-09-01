use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::SqliteDataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::tensor::backend::AutodiffBackend;
use burn::train::LearnerBuilder;
use burn::train::metric::{LearningRateMetric, LossMetric};

use crate::data::{AudioBatcher, AudioSample};
use crate::mask::block::{BlockMask, BlockMaskConfig};
use crate::metric::correct::CorrectMetric;
use crate::metric::gradnorm::GradientNormMetric;
use crate::metric::maskingratio::MaskingRatio;
use crate::metric::perplexity::PerplexityMetric;
use crate::metric::temperature::TemperatureMetric;
use crate::model::encoder::burn_transformer::BurnTransformer;
use crate::model::encoder::Encoder;
use crate::model::pretrain::Pretrain;
use crate::model::quantizer::gumbel::GumbelQuantizer;
use crate::model::quantizer::Quantizer;
use crate::ops::PolynomialDecay;

pub fn pretrain<B: AutodiffBackend>(
    device: B::Device,
    input_len: usize,
    dataset_train: SqliteDataset<AudioSample>,
    dataset_test: SqliteDataset<AudioSample>,
) {
    tch::maybe_init_cuda();

    let (pretrain_config, encoder_config, quantizer_config) =
        crate::config::small_music::small_music_config::<B>();

    let model: Pretrain<B, BurnTransformer<B>, GumbelQuantizer<B>> =
        pretrain_config
            .clone()
            .init(input_len, encoder_config, quantizer_config, &device);

    let mask_config = BlockMaskConfig {
        mask_prob: 0.65,
        mask_len: 10,
        min_masks: 1,
    };

    let batcher = AudioBatcher::<B, BlockMask>::new(
        input_len,
        pretrain_config.model_config.clone(),
        mask_config.clone(),
        device.clone(),
    );
    let data_loader_train = DataLoaderBuilder::new(batcher)
        .batch_size(8)
        .num_workers(1)
        .shuffle(0)
        .build(dataset_train);

    let batcher = AudioBatcher::<B::InnerBackend, BlockMask>::new(
        input_len,
        pretrain_config.model_config,
        mask_config.clone(),
        device.clone(),
    );
    let data_loader_test = DataLoaderBuilder::new(batcher)
        .batch_size(1)
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
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(PerplexityMetric::<B>::default())
        .metric_train_numeric(GradientNormMetric::<B>::default())
        .metric_train_numeric(LearningRateMetric::default())
        .metric_train_numeric(TemperatureMetric::<B>::default())
        .metric_train_numeric(CorrectMetric::<B>::default())
        .metric_train_numeric(MaskingRatio::<B>::default())
        .num_epochs(50)
        .summary()
        .build(model, optimizer, scheduler);
    let trained = learner.fit(data_loader_train, data_loader_test);
}
