pub mod perplexity;
pub mod gradnorm;
pub mod temperature;
pub mod correct;

use std::marker::PhantomData;
use burn::prelude::{Backend, Tensor};
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use burn::train::metric::state::{FormatOptions, NumericMetricState};

