use burn::module::Module;
use burn::nn::RotaryEncoding;
use burn::prelude::{Backend, Tensor};

#[derive(Module, Debug)]
pub struct RotaryEncoder<B: Backend> {
    inner: RotaryEncoding<B>
}