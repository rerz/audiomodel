use std::marker::PhantomData;
use burn::prelude::Backend;
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::metric::state::{FormatOptions, NumericMetricState};

#[derive(Default)]
pub struct MaskingRatio<B: Backend> {
    state: NumericMetricState,
    _backend: PhantomData<B>
}

pub struct MaskingRatioInput {
    pub num_total_samples: u32,
    pub num_masked_samples: u32,
}

impl<B: Backend> Metric for MaskingRatio<B> {
    const NAME: &'static str = "Masking Ratio";
    type Input = MaskingRatioInput;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        let ratio = item.num_masked_samples as f32 / item.num_total_samples as f32;
        self.state.update(ratio as f64, 1, FormatOptions::new("Masked %").precision(2))
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for MaskingRatio<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}