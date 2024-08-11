use std::io::Cursor;
use std::iter;
use itertools::Itertools;
use rmp3::{DecoderOwned, Frame};

pub trait Format {
    fn read(bytes: &[u8]) -> (Vec<f32>, u32);
}

pub struct Mp3;

impl Format for Mp3 {
    fn read(bytes: &[u8]) -> (Vec<f32>, u32) {
        read_mp3(bytes)
    }
}

fn read_mp3(bytes: &[u8]) -> (Vec<f32>, u32) {
    let mut decoder = DecoderOwned::new(bytes);

    let (samples, sample_rates): (Vec<_>, Vec<_>) = iter::from_fn(|| {
        let frame = decoder.next();

        match frame {
            Some(Frame::Audio(audio)) => Some((audio.samples().to_vec(), audio.sample_rate())),
            _ => None,
        }
    })
    .unzip();

    let sample_rates = sample_rates.into_iter().unique().collect_vec();

    if sample_rates.len() > 1 {
        panic!("variable sample rate?")
    }

    let sample_rate = sample_rates[0];

    let samples = samples
        .into_iter()
        .map(|x| x.to_vec())
        .flatten()
        .collect_vec();

    (samples, sample_rate)
}

pub struct Wav;

impl Format for Wav {
    fn read(bytes: &[u8]) -> (Vec<f32>, u32) {
        read_wav(bytes)
    }
}


fn read_wav(bytes: &[u8]) -> (Vec<f32>, u32) {
    let mut wav = wavers::Wav::<f32>::new(Box::new(Cursor::new(bytes.to_vec()))).unwrap();
    let sr = wav.sample_rate() as u32;
    let samples = wav.read().unwrap();

    (samples.to_vec(), sr)
}

pub fn read_bytes<F: Format>(bytes: &[u8]) -> (Vec<f32>, u32) {
    F::read(bytes)
}