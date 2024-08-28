use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::{Backend, ElementConversion};
use burn::tensor::{Bool, Distribution, Tensor};
use burn::tensor::activation::softmax;
use burn::tensor::backend::AutodiffBackend;
use crate::model::extractor::has_nan;

use crate::model::quantizer::{Quantizer, QuantizerConfig};

fn sample_gumbel<B: Backend>(shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
    let eps = 1e-20;
    let tensor = Tensor::random(shape, Distribution::Default, device);

    -Tensor::log(-Tensor::log(tensor + eps) + eps)
}

fn gumbel_softmax_sample<B: Backend>(tensor: Tensor<B, 2>, temperature: f32, device: &B::Device) -> Tensor<B, 2> {
    let noise = sample_gumbel(tensor.dims(), device);

    let y = tensor + noise;
    softmax(y / temperature, 1)
}

pub fn gumbel_softmax<B: Backend>(
    tensor: Tensor<B, 2>,
    temperature: f32,
    hard: bool,
    device: &B::Device,
) -> Tensor<B, 2> {
    let y = gumbel_softmax_sample(tensor, temperature, device);

    if !hard {
        return y;
    }

    let max_indices = y.clone().argmax(1);
    let y_hard = y.zeros_like().scatter(1, max_indices.expand([-1, y.dims()[1] as i32]), y.ones_like());

    (y_hard - y.clone()).detach() + y
}

#[derive(Config)]
pub struct GumbelQuantizerConfig {
    pub num_groups: usize,
    pub vectors_per_group: usize,
    pub vector_dim: usize,
}

impl GumbelQuantizerConfig {
    pub fn init<B: Backend>(self, last_conv_dim: usize, device: &B::Device) -> GumbelQuantizer<B> {
        GumbelQuantizer {
            num_groups: self.num_groups,
            vectors_per_group: self.vectors_per_group,
            code_vector_dim: self.vector_dim,
            code_vectors: Param::from_tensor(Tensor::ones(
                [
                    1,
                    self.num_groups * self.vectors_per_group,
                    self.vector_dim / self.num_groups,
                ],
                device,
            )),
            weight_projection: LinearConfig::new(
                last_conv_dim,
                self.num_groups * self.vectors_per_group,
            )
                .init(device),
            temperature: 2.0,
            temperature_scaling: (2.0, 0.5, 0999995)
        }
    }
}

#[derive(Module, Debug)]
pub struct GumbelQuantizer<B: Backend> {
    num_groups: usize,
    vectors_per_group: usize,
    code_vector_dim: usize,
    code_vectors: Param<Tensor<B, 3>>,
    weight_projection: Linear<B>,
    temperature: f32,
    // (2, 0.5, 0.999995) max, min, decay
    temperature_scaling: (f32, f32, f32),
}

pub struct GumbelQuantizerOutput<B: Backend> {
    prob_perplexity: Tensor<B, 1>,
    code_perplexity: Option<Tensor<B, 1>>,
    code_vectors: Tensor<B, 3>,
}

impl QuantizerConfig for GumbelQuantizerConfig {
    type Model<B> = GumbelQuantizer<B> where B: Backend;

    fn quantized_dim(&self) -> usize {
        self.vector_dim
    }
}

impl<B: Backend> Quantizer<B> for GumbelQuantizer<B> {
    type Config = GumbelQuantizerConfig;

    fn new(last_conv_dim: usize, config: Self::Config, device: &B::Device) -> Self {
        config.init(last_conv_dim, device)
    }

    fn forward<const TRAINING: bool>(
        &self,
        features: Tensor<B, 3>,
        mask_time_steps: Tensor<B, 2, Bool>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let output = self.quantize(features, mask_time_steps, TRAINING, device);
        (output.code_vectors, output.prob_perplexity)
    }

    fn num_groups(&self) -> usize {
        self.num_groups
    }

    fn num_vectors_per_group(&self) -> usize {
        self.vectors_per_group
    }

    fn temperature(&self) -> f32 {
        self.temperature
    }

    fn set_num_steps(&mut self, num_steps: u32) {
        self.temperature = f32::max(self.temperature_scaling.0 * self.temperature_scaling.2.powf(num_steps as f32), self.temperature_scaling.1);
    }
}

impl<B: Backend> GumbelQuantizer<B> {
    pub fn quantize(
        &self,
        x: Tensor<B, 3>,
        mask_time_indices: Tensor<B, 2, Bool>,
        training: bool,
        device: &B::Device,
    ) -> GumbelQuantizerOutput<B> {
        let [batch, time, feats] = x.dims();

        let x = self.weight_projection.forward(x);
        let x = x.reshape::<2, _>([(batch * time * self.num_groups) as i32, -1]);

        let avg_probs = softmax(x.clone().reshape([(batch * time) as i32, self.num_groups as i32, -1]), 2).mean_dim(0);
        let prob_perplexity = self.perplexity(avg_probs);

        let temp = self.temperature;

        let (code_vector_probs, code_perplexity) = match training {
            true => self.forward_train(x, mask_time_indices, [batch, time], device),
            false => self.forward_eval(x, mask_time_indices, [batch, time], device),
        };

        let code_vectors_per_group =
            code_vector_probs.unsqueeze_dim::<3>(2) * self.code_vectors.val();

        let code_vectors = code_vectors_per_group.reshape([
            (batch * time) as i32,
            self.num_groups as i32,
            self.vectors_per_group as i32,
            -1,
        ]);
        let code_vectors = code_vectors
            .sum_dim(2)
            .reshape([batch as i32, time as i32, -1]);

        GumbelQuantizerOutput {
            code_perplexity,
            prob_perplexity,
            code_vectors,
        }
    }

    fn perplexity(&self, probs: Tensor<B, 3>) -> Tensor<B, 1> {
        //let extended_mask = mask.clone().flatten::<1>(0, 1).unsqueeze_dims::<3>(&[1, 2]).expand(probs.shape());
        //let probs = Tensor::mask_where(probs.clone(), extended_mask, Tensor::ones_like(&probs));
        //let marginal_probs = probs.sum_dim(0) / mask.int().sum().to_data().to_vec::<B::IntElem>().unwrap()[0].elem::<i32>() as f32;

        let perplexity = Tensor::exp(-Tensor::sum_dim(probs.clone() * Tensor::log(probs + 1e-7), 2));
        let perplexity = perplexity.sum();

        perplexity
    }

    fn forward_train(
        &self,
        hidden: Tensor<B, 2>,
        mask_time_indices: Tensor<B, 2, Bool>,
        [batch, time]: [usize; 2],
        device: &B::Device,
    ) -> (Tensor<B, 2>, Option<Tensor<B, 1>>) {
        let code_vector_probs = gumbel_softmax(hidden.clone(), self.temperature, true, device);
        let code_vector_probs = code_vector_probs.reshape([(batch * time) as i32, -1]);

        (code_vector_probs, None)
    }

    fn forward_eval(
        &self,
        hidden: Tensor<B, 2>,
        mask_time_indices: Tensor<B, 2, Bool>,
        [batch, time]: [usize; 2],
        device: &B::Device,
    ) -> (Tensor<B, 2>, Option<Tensor<B, 1>>) {
        let code_vector_idx = hidden.clone().argmax(1);
        let code_vector_idx = code_vector_idx.reshape([-1, 1]);

        let hard_x = Tensor::zeros_like(&hidden);
        let hard_x = Tensor::scatter(
            hard_x.clone(),
            1,
            code_vector_idx,
            Tensor::ones(hard_x.shape(), device),
        );
        let hard_x =
            hard_x.reshape([(batch * time) as i32, self.num_groups as i32, -1]);

        let hard_probs = hard_x.clone().mean_dim(0);
        let code_perplexity = self.perplexity(hard_probs);

        let code_vector_probs = hard_x.reshape([(batch * time) as i32, -1]);

        (code_vector_probs, Some(code_perplexity))
    }
}
