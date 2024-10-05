use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::metric::state::{FormatOptions, NumericMetricState};

#[derive(Default)]
pub struct ContrastiveLossMetric<B: Backend> {
    pub state: NumericMetricState,
    _phantom: PhantomData<B>,
}

pub struct ContrastiveLossInput {
    pub value: f32,
}

impl<B: Backend> Metric for ContrastiveLossMetric<B> {
    const NAME: &'static str = "Contrastive Loss";
    type Input = ContrastiveLossInput;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        let contrastive_loss = item.value;
        self.state.update(contrastive_loss as f64, 1, FormatOptions::new("Contrastive Loss").precision(3))
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for ContrastiveLossMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}