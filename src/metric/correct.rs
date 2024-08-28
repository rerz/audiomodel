use std::marker::PhantomData;
use burn::prelude::Backend;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::metric::state::{FormatOptions, NumericMetricState};

#[derive(Default)]
pub struct CorrectMetric<B: Backend> {
    pub state: NumericMetricState,
    _phantom: PhantomData<B>
}

pub struct CorrectInput {
    pub value: u32,
}

impl<B: Backend> Metric for CorrectMetric<B> {
    const NAME: &'static str = "Num Correct";
    type Input = CorrectInput;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        let correct = item.value;
        self.state.update(correct as f64, 1, FormatOptions::new("Num Correct").precision(1))
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for CorrectMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}