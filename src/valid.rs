use burn::prelude::Backend;
use burn::train::ValidStep;
use crate::data::AudioBatch;
use crate::model::encoder::Encoder;
use crate::model::pretrain::Pretrain;
use crate::model::quantizer::Quantizer;

impl<B: Backend, E: Encoder<B>, Q: Quantizer<B>> ValidStep<AudioBatch<B>, ()> for Pretrain<B, E, Q> {
    fn step(&self, item: AudioBatch<B>) -> () {
        ()
    }
}