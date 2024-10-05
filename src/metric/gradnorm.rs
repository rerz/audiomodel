use std::marker::PhantomData;

use burn::module::{ModuleVisitor, ParamId};
use burn::prelude::{Backend, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::metric::state::{FormatOptions, NumericMetricState};

use crate::ops::l2;

pub struct GradientNorm<'a, B: AutodiffBackend> {
    pub scale: f32,
    pub grads: &'a B::Gradients,
    pub total_norm: f32,
}

impl<'a, B: AutodiffBackend> GradientNorm<'a, B> {
    pub fn new(grads: &'a B::Gradients, scale: f32) -> Self {
        Self {
            grads,
            scale,
            total_norm: 0.0,
        }
    }
}

impl<'a, B: AutodiffBackend> ModuleVisitor<B> for GradientNorm<'a, B> {
    fn visit_float<const D: usize>(&mut self, _id: &ParamId, tensor: &Tensor<B, D>) {
        if let Some(grads) = tensor.grad(self.grads) {
            let param_norm = l2(grads / self.scale);
            self.total_norm += param_norm.powf_scalar(2.0).to_data().to_vec::<f32>().unwrap()[0];
        }
    }
}

#[derive(Default)]
pub struct GradientNormMetric<B: Backend> {
    state: NumericMetricState,
    _phantom: PhantomData<B>,
}

pub struct GradientNormIntput {
    pub value: f32,
}

impl<B: Backend> Metric for GradientNormMetric<B> {
    const NAME: &'static str = "Gradient Norm";
    type Input = GradientNormIntput;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        let grad_norm = item.value;
        self.state.update(grad_norm as f64, 1, FormatOptions::new("Gradient Norm").precision(3))
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for GradientNormMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}