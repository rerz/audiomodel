use burn::backend::wgpu::select_device;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::nn::{Dropout, DropoutConfig, Gelu, Linear};
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::activation::softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Bool;
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use burn::train::metric::{AccuracyInput, LearningRateMetric, LossMetric};
use crate::config::small_music::small_music_config;
use crate::data::{AudioBatcher, AudioSample};
use crate::metric::code_perplexity::CodePerplexityMetric;
use crate::metric::contrastive_loss::ContrastiveLossMetric;
use crate::metric::correct::CorrectMetric;
use crate::metric::diversity_loss::DiversityLossMetric;
use crate::metric::gradnorm::GradientNormMetric;
use crate::metric::maskingratio::MaskingRatio;
use crate::metric::perplexity::PerplexityMetric;
use crate::metric::temperature::TemperatureMetric;
use crate::model::{AudioModel, AudioModelInput};
use crate::train::ConfigBundle;

#[derive(Module, Debug)]
pub struct MeanPooling<B: Backend> {
    linear: Linear<B>,
    activation: Gelu,
    projection: Linear<B>,
}

fn mean<B: Backend>(x: Tensor<B, 3>, padding_mask: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 2> {
    let [_time, _batch, _channel] = x.dims();
    // x: TxBxC
    if let Some(mask) = padding_mask {
        let mask = mask.swap_dims(0, 1).unsqueeze_dim::<3>(2);
        let x = Tensor::select(x, 0, mask.nonzero()[0].clone());
        //return (x * mask.clone().float()).sum_dim(0) / mask.sum_dim(0);
        return todo!();
    }

    let num_steps = x.dims()[0];
    x.sum_dim(0).squeeze(0) / num_steps as f32
}

impl<B: Backend> MeanPooling<B> {
    pub fn forward(&self, features: Tensor<B, 3>, mask: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 2> {
        let feat = self.linear.forward(features);
        let feat = mean(feat, mask);
        let feat = self.activation.forward(feat);
        let feat = self.projection.forward(feat);
        // BxC
        feat
    }
}

pub struct MeanPoolingConfig {
    model_hidden_size: usize,
    intermediate_dim: usize,
    num_classes: usize,
}

use burn::nn::LinearConfig;

impl MeanPoolingConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> MeanPooling<B> {
        MeanPooling {
            activation: Gelu::new(),
            linear: LinearConfig::new(self.model_hidden_size, self.intermediate_dim).init(device),
            projection: LinearConfig::new(self.intermediate_dim, self.num_classes).init(device)
        }
    }
}

#[derive(Config)]
pub struct ClassifierConfig {
    dropout: f32,
    pooling_dim: usize,
    num_classes: u32,
    num_finetune_steps: usize,
}

impl ClassifierConfig {
    pub fn init<B: Backend, E: Encoder<B>>(self, model: AudioModel<B, E>, device: &B::Device) -> Classifier<B, E> {
        let hidden_size = model.hidden_size();
        Classifier {
            model,
            dropout: DropoutConfig::new(self.dropout as f64).init(),
            pooling: MeanPoolingConfig { model_hidden_size: hidden_size, intermediate_dim: self.pooling_dim, num_classes: self.num_classes as usize }.init(device),
            num_finetune_steps: self.num_finetune_steps
        }
    }
}

#[derive(Module, Debug)]
pub struct Classifier<B: Backend, E> {
    pub model: AudioModel<B, E>,
    pub pooling:  MeanPooling<B>,
    pub dropout: Dropout,
    pub num_finetune_steps: usize
}

impl<B: Backend, E: Encoder<B>> Classifier<B, E> {
    pub fn forward(&self, inputs: Tensor<B, 2>, padding_mask: Tensor<B, 2, Bool>, device: &B::Device) -> Tensor<B, 2> {
        let sequence_lens = padding_mask.int().sum_dim(1).to_data().to_vec::<u32>().unwrap();
        let output = self.model.forward(AudioModelInput {
            sequences: inputs,
            sequence_lens,
            masked_time_steps: None,
        }, device);

        let x = output.last_hidden_state;
        let x = self.dropout.forward(x);
        let x = self.pooling.forward(x, Some(output.padding_mask));

        x
    }
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
    device: B::Device,
    inputs: Tensor<B, 2>,
    padding_mask: Tensor<B, 2, Bool>,
    targets: Option<Tensor<B, 1, Int>>,
    num_classes: u32,
}

pub struct ClassifierOutput<B: Backend> {
    predicted_labels: Tensor<B, 1, Int>,
    label_probabilities: Tensor<B, 1>
}

use burn::module::ModuleDisplay;
use burnx::lr::PolynomialDecay;
use burnx::sequence::mask::BlockMaskConfig;
use crate::model::arch::encoder::Encoder;

impl<B: AutodiffBackend, E: Encoder<B>  + AutodiffModule<B, InnerModule: ModuleDisplay> + ModuleDisplay, > TrainStep<ClassificationBatch<B>, ClassifierOutput<B>> for Classifier<B, E> {
    fn step(&self, item: ClassificationBatch<B>) -> TrainOutput<ClassifierOutput<B>> {
        let output = self.forward(item.inputs, item.padding_mask, &item.device);
        // BxTxC
        let output = output.reshape([-1, item.num_classes as i32]);

        let label_probabilities = softmax(output.clone(), 1).unsqueeze_dim(1);
        let predicted_labels = label_probabilities.clone().argmax(1).unsqueeze_dim(1);

        let loss = CrossEntropyLoss::new(None, &item.device).forward(output, item.targets.unwrap());

        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
             ClassifierOutput {
                predicted_labels,
                label_probabilities,
            }
        )
    }

    fn optimize<BB, O>(mut self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self where BB: AutodiffBackend, O: Optimizer<Self, BB>, Self: AutodiffModule<BB> {
        // TODO
        if 0 > self.num_finetune_steps {
            self.model = self.model.no_grad();
        }

        optim.step(lr, self, grads)
    }
}

impl<B: Backend, E: Encoder<B>> ValidStep<ClassificationBatch<B>, ClassificationOutput<B>> for Classifier<B, E> {
    fn step(&self, item: ClassificationBatch<B>) -> ClassificationOutput<B> {
        let output = self.forward(item.inputs, item.padding_mask, &item.device);
        let output = output.reshape([-1, item.num_classes as i32]);

        let accuracy_input = AccuracyInput::new(output, item.targets.unwrap());

        todo!()
    }
}

pub fn train_classifier<B: AutodiffBackend, E: Encoder<B>>(trained_model: AudioModel<B, E>, train_dataset: impl Dataset<AudioSample>, test_dataset: impl Dataset<AudioSample>, device: B::Device) where E: AutodiffModule<B> + ModuleDisplay, <E as AutodiffModule<B>>::InnerModule: ModuleDisplay, <E as AutodiffModule<B>>::InnerModule: Encoder<<B as AutodiffBackend>::InnerBackend> {
    let bundle = small_music_config();

    let model: Classifier<B, E> = ClassifierConfig::new(0.0, 128, 10, 10_000).init(trained_model, &device);

    let train_batcher = AudioBatcher::training(
        100_000,
        100,
        bundle.pretrain_config.model_config.clone(),
        BlockMaskConfig {
            min_masks: 1,
            mask_len: 10,
            mask_prob: 0.3,
        },
        true,
        &device,
    );
    let data_loader_train = DataLoaderBuilder::new(train_batcher)
        .batch_size(4)
        .num_workers(1)
        .shuffle(0)
        .build(train_dataset);

    let batcher = AudioBatcher::inference(
        100_000,
        bundle.pretrain_config.model_config.clone(),
        &device,
    );
    let data_loader_test = DataLoaderBuilder::new(batcher)
        .batch_size(4)
        .num_workers(1)
        .shuffle(0)
        .build(test_dataset);

    //TrainStep::step(&model, data_loader_train.iter().next().unwrap());

    let optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let scheduler = PolynomialDecay::new(0.005, 0.0, 1.0, 600_000, 200_000);

    let learner = LearnerBuilder::new(".out")
        .devices(vec![device])
        // train metrics
        // validation metrics
        .num_epochs(20)
        .summary()
        .build(model, optimizer, scheduler);
    let trained = learner.fit(data_loader_train, data_loader_test);

    let encoder = trained.model;

}