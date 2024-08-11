use burn::config::Config;
use burn::module::Module;
use burn::nn::{Gelu, PaddingConfig1d};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::prelude::{Backend, Tensor};

fn same_pad<B: Backend>(tensor: Tensor<B, 3>, num_posconv_embeddings: usize) -> Tensor<B, 3> {
    let to_remove = if num_posconv_embeddings % 2 == 0 {
        1
    } else {
        0
    };

    let [batch, seq, hidden] = tensor.dims();

    let hidden = tensor.slice([0..batch, 0..seq, 0..hidden - to_remove]);

    hidden
}

#[derive(Module, Debug)]
pub struct PosConv<B: Backend> {
    conv: Conv1d<B>,
    activation: Gelu,
}

#[derive(Config)]
pub struct PosConvConfig {
    hidden_size: usize,
    num_embeddings: usize,
    num_groups: usize,
}

impl PosConvConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> PosConv<B> {
        PosConv {
            conv: Conv1dConfig::new(self.hidden_size, self.hidden_size, self.num_embeddings)
                .with_stride(1)
                .with_padding(PaddingConfig1d::Explicit(self.num_embeddings / 2))
                .with_groups(self.num_groups)
                .init(device),
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> PosConv<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = hidden.swap_dims(1, 2);
        let hidden = self.conv.forward(hidden);
        let hidden = same_pad(hidden, 10); // TODO: CHANGEME
        let hidden = self.activation.forward(hidden);
        let hidden = hidden.swap_dims(1, 2);

        hidden
    }
}
