use std::sync::atomic::Ordering;
use burn::data::dataloader::{DataLoaderBuilder, Dataset};
use burn::data::dataset::SqliteDataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, ModuleDisplay};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::ElementConversion;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{LearnerBuilder, TrainOutput, TrainStep};
use burn::train::metric::{LearningRateMetric, LossMetric};

use crate::data::{AudioBatch, AudioBatcher, AudioSample};
use crate::mask::block::{BlockMask, BlockMaskConfig};
use crate::metric::correct::CorrectMetric;
use crate::metric::gradnorm::{GradientNorm, GradientNormMetric};
use crate::metric::perplexity::PerplexityMetric;
use crate::metric::temperature::TemperatureMetric;
use crate::model::encoder::burn_transformer::BurnTransformer;
use crate::model::encoder::Encoder;
use crate::model::pretrain::{Pretrain, PretrainStepOutput};
use crate::model::quantizer::gumbel::GumbelQuantizer;
use crate::model::quantizer::Quantizer;
use crate::ops::{GradientMult, PolynomialDecay};

impl<
        B: AutodiffBackend,
        E: Encoder<B> + AutodiffModule<B, InnerModule: ModuleDisplay> + ModuleDisplay,
        Q: Quantizer<B> + AutodiffModule<B, InnerModule: ModuleDisplay> + ModuleDisplay,
    > TrainStep<AudioBatch<B>, PretrainStepOutput<B>> for Pretrain<B, E, Q>
{
    fn step(&self, item: AudioBatch<B>) -> TrainOutput<PretrainStepOutput<B>> {


        let (hidden, loss, perplexity, correct) = self.forward(
            item.sequences,
            item.attention_mask,
            item.sequence_lens,
            item.masked_time_indices,
            item.sampled_negative_indices,
            &item.device,
        );

        let mut grads = loss.backward();

        // TODO: gradient multiplier on feature extractor?
        let mut gradient_mult = GradientMult {
            multiplier: 0.1,
            grads: &mut grads,
        };
        self.model.feature_extractor.visit(&mut gradient_mult);

        let gradient_norm = {
            let mut gradient_norm = GradientNorm::new(&grads, 1.0);
            self.visit(&mut gradient_norm);
            gradient_norm.total_norm.sqrt()
        };

        self.steps.fetch_add(1, Ordering::Relaxed);

        TrainOutput::new(
            self,
            grads,
            PretrainStepOutput {
                loss,
                hidden,
                perplexity,
                gradient_norm,
                quantizer_temperature: self.quantizer.temperature(),
                correct
            },
        )
    }

    fn optimize<BB, O>(mut self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
    where
        BB: AutodiffBackend,
        O: Optimizer<Self, BB>,
        Self: AutodiffModule<BB>,
    {
        self.quantizer.set_num_steps(self.steps.load(Ordering::Relaxed) as u32);

        optim.step(lr, self, grads)
    }
}

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
        mask_prob: 0.1,
        mask_len: 100,
        min_masks: 8,
    };

    let batcher = AudioBatcher::<B, BlockMask>::new(
        input_len,
        pretrain_config.model_config.clone(),
        mask_config.clone(),
        device.clone(),
    );
    let data_loader_train = DataLoaderBuilder::new(batcher)
        .batch_size(16)
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

    TrainStep::step(&model, data_loader_train.iter().next().unwrap());

    let optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let scheduler = PolynomialDecay::new(0.0005, 0.0, 1.0, 300_000, 100_000);

    let learner = LearnerBuilder::new(".out")
        .devices(vec![device])
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(PerplexityMetric::<B>::default())
        .metric_train_numeric(GradientNormMetric::<B>::default())
        .metric_train_numeric(LearningRateMetric::default())
        .metric_train_numeric(TemperatureMetric::<B>::default())
        .metric_train_numeric(CorrectMetric::<B>::default())
        .num_epochs(50)
        .summary()
        .build(model, optimizer, scheduler);
    let trained = learner.fit(data_loader_train, data_loader_test);
}
