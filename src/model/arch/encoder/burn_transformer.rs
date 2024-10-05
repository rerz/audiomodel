use burn::config::Config;
use burn::module::Module;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::prelude::{Backend, Bool, Tensor};

use crate::model::encoder::{Encoder, EncoderConfig};
use crate::model::posenc::convpos::{PosConv, PosConvConfig};

#[derive(Config)]
pub struct BurnTransformerEncoderConfig {
    pub pos_conv_config: PosConvConfig,
    pub num_heads: usize,
    pub num_layers: usize,
}

impl EncoderConfig for BurnTransformerEncoderConfig {
    type Model<B> = BurnTransformer<B> where B: Backend;
}

#[derive(Module, Debug)]
pub struct BurnTransformer<B: Backend> {
    pos_conv: PosConv<B>,
    transformer_encoder: TransformerEncoder<B>,
}

impl<B: Backend> BurnTransformer<B> {
    pub fn encode(&self, hidden: Tensor<B, 3>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let [_batch, _time, _hidden] = hidden.dims();

        let hidden_conv = self.pos_conv.forward(hidden.clone());
        let hidden = hidden_conv + hidden;

        // TODO: pad to multiple?
        // let hidden = hidden.swap_dims(0, 1);

        let hidden = self.transformer_encoder.forward(
            TransformerEncoderInput::new(hidden).mask_attn(attention_mask.unsqueeze_dim(1)),
        );

        //hidden.swap_dims(0, 1)

        hidden
    }
}

impl<B: Backend> Encoder<B> for BurnTransformer<B> {
    type Config = BurnTransformerEncoderConfig;

    fn new(
        config: Self::Config,
        hidden_size: usize,
        extractor_output_size: u32,
        device: &B::Device,
    ) -> Self {
        let pos_conv = config.pos_conv_config.init(device);

        Self {
            pos_conv,
            transformer_encoder: TransformerEncoderConfig::new(
                hidden_size,
                hidden_size,
                config.num_heads,
                config.num_layers,
            )
                .init(device),
        }
    }

    fn forward(&self, hidden: Tensor<B, 3>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        self.encode(hidden, attention_mask)
    }
}
