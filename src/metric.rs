use burn::prelude::Backend;
use burn::train::metric::{Metric, Numeric};

pub mod perplexity;
pub mod gradnorm;
pub mod temperature;
pub mod correct;
pub mod maskingratio;
pub mod diversity_loss;
pub mod contrastive_loss;
pub mod code_perplexity;

