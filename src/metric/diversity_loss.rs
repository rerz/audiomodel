use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::metric::state::{FormatOptions, NumericMetricState};

#[derive(Default)]
pub struct DiversityLossMetric<B: Backend> {
    pub state: NumericMetricState,
    _phantom: PhantomData<B>,
}

pub struct DiversityLossInput {
    pub value: f32,
}

impl<B: Backend> Metric for DiversityLossMetric<B> {
    const NAME: &'static str = "Diversity Loss";
    type Input = DiversityLossInput;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        let diversity_loss = item.value;
        self.state.update(diversity_loss as f64, 1, FormatOptions::new("Diversity Loss").precision(3))
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for DiversityLossMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}