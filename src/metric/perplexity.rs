use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::metric::state::{FormatOptions, NumericMetricState};

#[derive(Default)]
pub struct PerplexityMetric<B: Backend> {
    pub state: NumericMetricState,
    _phantom: PhantomData<B>,
}

pub struct PerplexityInput {
    pub value: f32,
}

impl<B: Backend> Metric for PerplexityMetric<B> {
    const NAME: &'static str = "Perplexity";
    type Input = PerplexityInput;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        let perplexity = item.value;
        self.state.update(perplexity as f64, 1, FormatOptions::new("Perplexity").precision(3))
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for PerplexityMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}