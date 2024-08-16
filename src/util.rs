use std::fs::File;

use burn::prelude::Backend;
use hf_hub::{Repo, RepoType};
use hf_hub::api::sync::{Api, ApiRepo};
use itertools::Itertools;
use parquet::file::reader::SerializedFileReader;
use parquet::record::Row;
use rand::{Rng, thread_rng};

use crate::pad::{pad_sequences, PaddedSequences, PaddingType};

pub(crate) fn sample_sequence(min_len: usize, max_len: usize) -> Vec<f32> {
    let seq_len = thread_rng().gen_range(min_len..max_len);
    let seq = thread_rng()
        .sample_iter(rand::distributions::Uniform::new(0.0, 1.0))
        .take(seq_len)
        .collect_vec();

    seq
}

pub fn sample_test_batch<B: Backend>(
    batch_size: usize,
    min_len: usize,
    max_len: usize,
    device: &B::Device,
) -> PaddedSequences<B> {
    let seqs = (0..batch_size)
        .map(|idx| sample_sequence(min_len, max_len))
        .collect_vec();
    let padded = pad_sequences::<B>(seqs, PaddingType::LongestSequence, device);
    padded
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("ApiError : {0}")]
    ApiError(#[from] hf_hub::api::sync::ApiError),

    #[error("IoError : {0}")]
    IoError(#[from] std::io::Error),

    #[error("ParquetError : {0}")]
    ParquetError(#[from] parquet::errors::ParquetError),
}

fn sibling_to_parquet(
    rfilename: &str,
    repo: &ApiRepo,
) -> Result<SerializedFileReader<File>, Error> {
    let local = repo.get(rfilename)?;
    let file = File::open(local)?;
    let reader = SerializedFileReader::new(file)?;
    Ok(reader)
}

pub enum Split<T> {
    Test(T),
    Train(T),
}

pub fn download_hf_dataset(api: &Api, dataset_id: String) -> Result<(impl Iterator<Item=Row>, impl Iterator<Item=Row>), Error> {
    let repo = Repo::with_revision(
        dataset_id,
        RepoType::Dataset,
        "refs/convert/parquet".to_string(),
    );
    let repo = api.repo(repo);
    let info = repo.info()?;

    let mut train = vec![];
    let mut test = vec![];

    for sibling in info.siblings {
        let filename = sibling.rfilename;
        if filename.ends_with(".parquet") {
            let reader_result = sibling_to_parquet(&filename, &repo);
            if filename.contains("test") {
                test.push(reader_result.unwrap());
            } else if filename.contains("train") {
                train.push(reader_result.unwrap());
            }
        }
    }

    Ok((
        train.into_iter().flat_map(|file| file.into_iter().filter_map(Result::ok)),
        test.into_iter().flat_map(|file| file.into_iter().filter_map(Result::ok))
    ))
}