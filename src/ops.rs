use burn::LearningRate;
use burn::lr_scheduler::LrScheduler;
use burn::module::{ModuleVisitor, ParamId};
use burn::prelude::{Backend, ElementConversion, Tensor};
use burn::tensor::backend::AutodiffBackend;

pub fn l2<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, 1> {
    let tensor = tensor.flatten::<1>(0, D - 1);
    let squared = tensor.powi_scalar(2);
    let summed = squared.sum();
    let norm = summed.sqrt();
    norm
}

pub fn cosine_similarity<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let dot = Tensor::sum_dim(a.clone() * b.clone(), dim);
    let norm_a = l2(a).to_data().to_vec::<B::FloatElem>().unwrap()[0].elem::<f32>();
    let norm_b = l2(b).to_data().to_vec::<B::FloatElem>().unwrap()[0].elem::<f32>();

    let norm_a = f32::max(norm_a, 1e-8);
    let norm_b = f32::max(norm_b, 1e-8);

    let sim = dot / (norm_a * norm_b);

    sim
}

pub struct PolynomialDecay {
    start_lr: f32,
    end_lr: f32,
    power: f32,
    warmup_factor: f32,
    num_warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl PolynomialDecay {
    pub fn new(start_lr: f32, end_lr: f32, power: f32, total_steps: usize, num_warmup_steps: usize) -> Self {
        Self {
            start_lr,
            end_lr,
            power,
            warmup_factor: 1.0 / num_warmup_steps as f32,
            total_steps,
            num_warmup_steps,
            current_step: 1,
        }
    }
}

impl<B: Backend> LrScheduler<B> for PolynomialDecay {
    type Record = (f32, f32, f32, f32, usize, usize, usize);

    fn step(&mut self) -> LearningRate {
        if self.current_step >= self.total_steps {
            return self.end_lr as f64;
        }

        if self.current_step <= self.num_warmup_steps {
            self.warmup_factor = self.current_step as f32 / self.num_warmup_steps as f32;
            self.current_step += 1;
            return (self.warmup_factor * self.start_lr) as f64;
        }

        let decay = (1.0 - (self.current_step - self.num_warmup_steps) as f32 / (self.total_steps - self.num_warmup_steps) as f32).powf(self.power);
        let lr = (self.start_lr - self.end_lr) * decay + self.end_lr;

        self.current_step += 1;

        lr as f64
    }

    fn to_record(&self) -> Self::Record {
        (
            self.start_lr,
            self.end_lr,
            self.power,
            self.warmup_factor,
            self.num_warmup_steps,
            self.total_steps,
            self.current_step
        )
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            start_lr: record.0,
            end_lr: record.1,
            power: record.2,
            warmup_factor: record.3,
            num_warmup_steps: record.4,
            total_steps: record.5,
            current_step: record.6,
        }
    }
}

pub struct GradientMult<'a, B: AutodiffBackend>  {
    pub multiplier: f32,
    pub grads: &'a mut B::Gradients,
}

impl<'a, B: AutodiffBackend> ModuleVisitor<B> for GradientMult<'a, B> {
    fn visit_float<const D: usize>(&mut self, _id: &ParamId, tensor: &Tensor<B, D>) {
        if let Some(grads) = tensor.grad(self.grads) {
            let multiplied = grads * self.multiplier;
            tensor.grad_replace(self.grads, multiplied);
        }
    }
}