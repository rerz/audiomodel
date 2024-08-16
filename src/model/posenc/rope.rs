use burn::module::Module;
use burn::prelude::{Backend, Tensor};


#[derive(Module, Debug)]
pub struct RotaryPositionEmbedding<B: Backend> {
    inv_freq: Tensor<B, 1>
}

impl<B: Backend> RotaryPositionEmbedding<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            inv_freq: Tensor::ones([1], device).expand([dim / 2]) / Tensor::<B, 1>::from_floats([10_000], device).expand([dim / 2]).powf(Tensor::arange_step(0..dim as i64, 2, device).float() / dim as f32),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 3> {
        let [batch, seq, _] = hidden.dims();

        let t = Tensor::arange(0..seq as i64, device).float();
        let t = t.unsqueeze_dim::<2>(1);

        let inv_freq = self.inv_freq.clone().unsqueeze_dim(0);
        let freqs = t * inv_freq;
        let emb = Tensor::cat(vec![freqs.clone(), freqs], 1);
        emb.unsqueeze_dim(0)
    }
}

fn rotate_half<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mut chunks = x.chunk(2, 2).into_iter();
    let x1 = chunks.next().unwrap();
    let x2 = chunks.next().unwrap();
    Tensor::cat(vec![-x2, x1], 2)
}

#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    dim: usize,
    rope: RotaryPositionEmbedding<B>
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 3> {
        let rope = self.rope.forward(hidden.clone(), device);
        self.apply_rotary_pos_embed(hidden, rope)
    }

    fn apply_rotary_pos_embed(&self, hidden: Tensor<B, 3>, rope: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_rope = (hidden.clone() * rope.clone().cos()) + (rotate_half(hidden) * rope.sin());
        hidden_rope
    }
}