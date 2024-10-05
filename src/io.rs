use std::io::Cursor;
use std::iter;
use color_eyre::owo_colors::OwoColorize;

use itertools::Itertools;
use rmp3::{DecoderOwned, Frame};
use samplerate::ConverterType;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default;

// TODO: move to symphonia based decoders

pub fn read_audio_bytes(bytes: &[u8]) -> (Vec<f32>, u32) {
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let hint = Hint::new();
    let meta_opts = MetadataOptions::default();
    let format_opts = FormatOptions::default();
    let probed = symphonia::default::get_probe().format(&hint, mss, &format_opts, &meta_opts).unwrap();
    let mut format = probed.format;
    let track = format.tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    let mut decoder = default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .expect("unsupported codec");

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap();

    let mut pcm_data = Vec::new();
    while let Ok(packet) = format.next_packet() {
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet).unwrap() {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            _ => panic!(),
        }
    }

    (pcm_data, sample_rate)
}

pub trait Format {
    fn read(bytes: &[u8]) -> (Vec<f32>, u32);
}

pub struct Mp3;

impl Format for Mp3 {
    fn read(bytes: &[u8]) -> (Vec<f32>, u32) {
        read_mp3(bytes)
    }
}

pub fn resample(audio: Vec<f32>, from_sr: u32, to_sr: u32) -> Vec<f32> {
    samplerate::convert(from_sr, to_sr, 1, ConverterType::SincFastest, &audio).unwrap()
}

fn average_interleaved(input: &[f32]) -> Vec<f32> {
    input
        .chunks(2)
        .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
        .collect()
}

fn read_mp3(bytes: &[u8]) -> (Vec<f32>, u32) {
    let mut decoder = DecoderOwned::new(bytes);
    let (samples, sample_rates): (Vec<_>, Vec<_>) = iter::from_fn(|| {
        let frame = decoder.next();

        match frame {
            Some(Frame::Audio(audio)) => {
                let channels = audio.channels();

                Some((audio.samples().to_vec(), audio.sample_rate()))
            }
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

    let samples = average_interleaved(&samples);

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

pub fn read_bytes_inferred(bytes: &[u8], resample_to: usize) -> (Vec<f32>, u32) {
    let ty = infer::get(bytes).unwrap();

    match ty.mime_type() {
        "audio/x-wav" => read_wav(bytes),
        "audio/mpeg" => read_mp3(bytes),
        "audio/ogg" => read_ogg(bytes),
        _ => panic!("could not infer file format")
    }
}

pub fn read_bytes<F: Format>(bytes: &[u8], resample_to: u32) -> (Vec<f32>, u32) {
    let (audio, sr) = F::read(bytes);

    if resample_to != sr {
        return (resample(audio, sr, resample_to), resample_to);
    }

    (audio, sr)
}

fn read_ogg(bytes: &[u8]) -> (Vec<f32>, u32) {
    let mut reader = lewton::inside_ogg::OggStreamReader::new(Box::new(Cursor::new(bytes.to_vec()))).unwrap();
    let sr = reader.ident_hdr.audio_sample_rate;

    let mut samples = vec![];

    while let Some(packet) = reader.read_dec_packet_generic::<Vec<Vec<f32>>>().unwrap() {
        let mean = packet[0]
            .iter()
            .enumerate()
            .map(|(i, _)| packet.iter().map(|c| c[i]).sum::<f32>() / packet.len() as f32)
            .collect_vec();

        samples.extend(mean);
    }

    (samples, sr)
}