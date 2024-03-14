import random
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")

import numpy as np
from pathlib import Path
import polars as pl
from scipy.signal import butter, lfilter
from torch.utils.data import Dataset

from src.settings import TARGET_COLS, SAMPLE_RATE, EEG_DURATION, EEG_GROUP_IDX  # noqa


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def get_sample_weights(df):
    return (
        df.with_columns(
            pl.reduce(
                lambda x, y: x.cast(pl.String) + "_" + y.cast(pl.String),
                pl.col(["eeg_id"] + TARGET_COLS),
            ).alias("unq_row_id")
        )
        .with_columns(
            pl.len().over("patient_id").alias("patient_sample_count"),
            pl.len().over("unq_row_id").alias("eeg_sample_count"),
        )
        .with_columns(
            (1 / (pl.col("eeg_sample_count") / pl.col("patient_sample_count"))).alias(
                "sample_weight"
            ),
        )
        .with_columns(
            (
                pl.col("sample_weight")
                / pl.col("sample_weight").sum().over("patient_id")
            ).alias("sample_weight")
        )
    )


def norm_target_cols(df):
    norm_targets = df.select(TARGET_COLS).to_numpy()
    target_sums = norm_targets.sum(axis=1, keepdims=True)
    norm_targets = norm_targets / target_sums
    return df.with_columns(
        *[pl.Series(target, norm_targets[:, i]) for i, target in enumerate(TARGET_COLS)]
    ).with_columns(pl.Series("num_votes", target_sums.flatten()))


class HMSTrainEEGData(Dataset):
    def __init__(self, df, data_dir):
        self.patient_ids = df["patient_id"].unique().to_list()
        self.df = df
        self.data_dir = data_dir
        self.df = get_sample_weights(self.df)
        self.df = norm_target_cols(self.df)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_df = self.df.filter(pl.col("patient_id") == patient_id)
        idx = random.choices(
            range(len(patient_df)), weights=patient_df["sample_weight"].to_numpy()
        )[0]
        patient_df = patient_df[idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        # offset = patient_df["eeg_label_offset_seconds"].iloc[0]
        eeg_sub_id = patient_df["eeg_sub_id"].iloc[0]
        data = load_eeg_data(self.data_dir, eeg_id, eeg_sub_id)
        targets = patient_df[TARGET_COLS].values.flatten()
        # targets = targets / targets.sum()
        return data.astype(np.float32), targets.astype(np.float32)


class HMSTrainEEGDataV2(Dataset):
    def __init__(self, df, data_dir, transforms=None):
        self.df = df  # .unique(subset=["eeg_id", *TARGET_COLS])
        self.df = get_sample_weights(self.df)
        self.unq_ids = self.df["eeg_id"].unique().to_list()
        self.data_dir = data_dir
        self.df = norm_target_cols(self.df)
        self.transforms = None

    def __len__(self):
        return len(self.unq_ids)

    def __getitem__(self, idx):
        unq_id = self.unq_ids[idx]
        patient_df = self.df.filter(pl.col("eeg_id") == unq_id)
        idx = random.choices(
            range(len(patient_df)),  # weights=patient_df["sample_weight"].to_numpy()
        )[0]
        patient_df = patient_df[idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        # offset = patient_df["eeg_label_offset_seconds"].iloc[0]
        eeg_sub_id = patient_df["eeg_sub_id"].iloc[0]
        data = load_eeg_data(self.data_dir, eeg_id, eeg_sub_id)
        if self.transforms is not None:
            for tfm in self.transforms:
                data = tfm(data)
        targets = patient_df[TARGET_COLS].values.flatten()
        # targets = targets / targets.sum()
        return {
            "data": data.astype(np.float32),
            "targets": targets.astype(np.float32),
            "eeg_id": eeg_id.astype(np.int32),
            "eeg_sub_id": eeg_sub_id.astype(np.int32),
        }


class HMSValEEGData(Dataset):
    def __init__(self, df, data_dir, transforms=None):
        self.df = df
        self.data_dir = data_dir
        self.df = norm_target_cols(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_df = self.df[idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        # offset = patient_df["eeg_label_offset_seconds"].iloc[0]
        eeg_sub_id = patient_df["eeg_sub_id"].iloc[0]
        data = load_eeg_data(self.data_dir, eeg_id, eeg_sub_id)

        targets = patient_df[TARGET_COLS].values.flatten()
        # targets = targets / targets.sum()
        return {
            "data": data.astype(np.float32),
            "targets": targets.astype(np.float32),
            "eeg_id": eeg_id.astype(np.int64),
            "eeg_sub_id": eeg_sub_id.astype(np.int64),
        }


def fix_nulls(data):
    for i in range(data.shape[1]):
        if np.all(np.isnan(data[:, i])):
            data[:, i] = 0
        else:
            mean = np.nanmedian(data[:, i])
            data[:, i] = np.nan_to_num(data[:, i], nan=mean)

    return data


def load_eeg_data(data_dir, eeg_id, eeg_sub_id):
    npy_path = data_dir / f"{eeg_id}_{eeg_sub_id}.npy"
    data = np.load(npy_path)
    data = fix_nulls(data)
    data = butter_lowpass_filter(data)
    out = np.zeros((data.shape[0], 17), dtype=np.float32)
    for i, (j, k) in enumerate(EEG_GROUP_IDX):
        out[:, i] = data[:, j] - data[:, k]
    out[:, 16] = data[:, -1]
    out = np.clip(out, -1024, 1024)
    # out = np.log1p(np.abs(out)) * np.sign(out) / 3
    out = out / 100
    return out[8:-8, :]
    # df = pl.read_parquet(pq_path)
    # offset = int(offset * SAMPLE_RATE)
    # data = (
    #     df[offset : offset + SAMPLE_RATE * EEG_DURATION : 2, :19]
    #     .fill_null(0)
    #     .to_numpy()
    # )
    # data = np.clip(data, -2048, 2048)
    # data = np.log1p(np.abs(data)) * np.sign(data) / 6
    # return data
