use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Gelu, LayerNorm, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{Bool, Tensor};
use burn::tensor::activation::softmax;
use crate::model::encoder::Encoder;
use crate::model::posconv::PosConv;

#[derive(Config)]
pub struct EncoderAttentionConfig {
    embed_dim: usize,
    num_heads: usize,
    bias: bool,
    dropout: f32,
}

impl EncoderAttentionConfig {
    pub fn init<B: Backend>(self) -> EncoderAttention<B> {
        let embed_dim = self.embed_dim;
        let head_dim = self.embed_dim / self.num_heads;
        EncoderAttention {
            head_dim,
            num_heads: self.num_heads,
            embed_dim,
            dropout: DropoutConfig::new(self.dropout as f64).init(),
            scaling: f32::powf(head_dim as f32, -0.5),
            k_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(self.bias).init(&B::Device::default()),
            v_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(self.bias).init(&B::Device::default()),
            q_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(self.bias).init(&B::Device::default()),
            out_proj: LinearConfig::new(embed_dim, embed_dim).with_bias(self.bias).init(&B::Device::default()),
        }
    }
}

#[derive(Module, Debug)]
pub struct EncoderAttention<B: Backend> {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    scaling: f32,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    q_proj: Linear<B>,
    out_proj: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> EncoderAttention<B> {
    fn reshape(&self, tensor: Tensor<B, 3>, seq_len: i32, batch: usize) -> Tensor<B, 4> {
        tensor.reshape([batch as i32, seq_len, self.num_heads as i32, self.head_dim as i32]).swap_dims(1, 2)
    }

    pub fn forward(&self, hidden: Tensor<B, 3>, attention_mask: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, time, _] = hidden.dims();

        let query_states = self.q_proj.forward(hidden.clone()) * self.scaling;

        let key_states = self.reshape(self.k_proj.forward(hidden.clone()), -1, batch);
        let value_states = self.reshape(self.v_proj.forward(hidden.clone()), -1, batch);

        let projected_shape = [(batch * self.num_heads) as i32, -1, self.head_dim as i32];

        let query_states = self.reshape(query_states, time as i32, batch).reshape(projected_shape);
        let key_states = key_states.reshape(projected_shape);
        let values_states = value_states.reshape(projected_shape);

        let source_len = key_states.dims()[1];
        let key_states = key_states.swap_dims(1, 2);
        let attention_weights = Tensor::matmul(query_states, key_states);

        let attention_weights = attention_weights.reshape([batch, self.num_heads, time, source_len]) + attention_mask;
        let attention_weights = attention_weights.reshape([batch * self.num_heads, time, source_len]);

        let attention_weights = softmax(attention_weights, 2);

        let attention_probs = self.dropout.forward(attention_weights);

        let attention_output = Tensor::matmul(attention_probs, values_states);
        let attention_output = attention_output.reshape([batch, self.num_heads, time, self.head_dim]);
        let attention_output = attention_output.swap_dims(1, 2);
        let attention_output = attention_output.reshape([batch, time, self.embed_dim]);
        let attention_output = self.out_proj.forward(attention_output);

        attention_output
    }
}

#[derive(Module, Debug)]
pub struct EncoderFeedForward<B: Backend> {
    dropout: Dropout,
    dense: Linear<B>,
    activation: Gelu,
    output_dense: Linear<B>,
    output_dropout: Dropout,
}

impl<B: Backend> EncoderFeedForward<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.dense.forward(hidden);
        let hidden = self.activation.forward(hidden);
        let hidden = self.dropout.forward(hidden);
        let hidden = self.output_dense.forward(hidden);
        let hidden = self.output_dropout.forward(hidden);
        hidden
    }
}

#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    encoder_attention: EncoderAttention<B>,
    dropout: Dropout,
    layer_norm: LayerNorm<B>,
    feed_forward: EncoderFeedForward<B>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> TransformerEncoderLayer<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>, attention_mask: Tensor<B, 4>) -> Tensor<B, 3> {
        let residual_attention = hidden.clone();
        let hidden = self.layer_norm.forward(hidden);
        let hidden = self.encoder_attention.forward(hidden, attention_mask);
        let hidden = self.dropout.forward(hidden);
        let hidden = residual_attention + hidden;
        let hidden = hidden.clone() + self.feed_forward.forward(self.final_layer_norm.forward(hidden));

        hidden
    }
}

#[derive(Config)]
pub struct TransformerEncoderConfig {

}

impl TransformerEncoderConfig {
    pub fn init<B: Backend>(self) -> TransformerEncoder<B> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    pos_conv: PosConv<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
    layers: Vec<TransformerEncoderLayer<B>>,
}

impl<B: Backend> TransformerEncoder<B> {
    pub fn encode(&self, hidden: Tensor<B, 3>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let expanded_attention_mask = attention_mask.clone().unsqueeze_dim::<3>(2).repeat(2, hidden.dims()[2]);
        let hidden = hidden.mask_fill(expanded_attention_mask.bool_not(), 0);

        let attention_mask = attention_mask.unsqueeze_dims(&[1, 2]).bool_not().float();
        let attention_mask = attention_mask * f32::neg_infinity();
        let attention_mask = attention_mask.expand([attention_mask.dims()[0], 1, attention_mask.dims()[3], attention_mask.dims()[3]]);

        let pos_embeddings = self.pos_conv.forward(hidden.clone());
        let hidden = hidden + pos_embeddings;
        let mut hidden = self.dropout.forward(hidden);

        for layer in &self.layers {
            hidden = layer.forward(hidden, attention_mask.clone());
        }

        let hidden = self.layer_norm.forward(hidden);

        hidden
    }
}

impl<B: Backend> Encoder<B> for TransformerEncoder<B> {
    type Config = TransformerEncoderConfig;

    fn new(config: Self::Config, hidden_size: usize, extractor_output_size: usize) -> Self {
        config.init()
    }

    fn forward(&self, hidden: Tensor<B, 3>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        self.encode(hidden, attention_mask)
    }
}