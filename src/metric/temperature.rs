use std::marker::PhantomData;
use burn::prelude::Backend;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use crate::model::pretrain::PretrainStepOutput;

#[derive(Default)]
pub struct TemperatureMetric<B: Backend> {
    state: NumericMetricState,
    _phantom: PhantomData<B>
}

pub struct TemperatureInput {
    pub value: f32
}

impl<B: Backend> Metric for TemperatureMetric<B> {
    const NAME: &'static str = "Temperature";
    type Input = TemperatureInput;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        self.state.update(item.value as f64, 1, FormatOptions::new("Temperature").precision(3))
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for TemperatureMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}