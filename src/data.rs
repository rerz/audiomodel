use burn::prelude::{Backend, Int, Tensor};

pub struct AudioBatch<B: Backend> {
    pub sequences: Tensor<B, 2>,
    pub sequence_lens: Vec<usize>,
    pub attention_mask: Tensor<B, 2, Int>,
}

impl<B: Backend> AudioBatch<B> {
    pub fn input_len(&self) -> usize {
        self.sequences.dims()[1]
    }
}
