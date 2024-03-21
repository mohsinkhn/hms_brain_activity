import random

import numpy as np
from pathlib import Path
import polars as pl
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from torch.utils.data import Dataset

from src.settings import TARGET_COLS, SAMPLE_RATE, EEG_DURATION, EEG_GROUP_IDX  # noqa


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def notch_filter(data, fs, freq):
    b, a = iirnotch(freq, 30, fs)
    y = filtfilt(b, a, data, axis=0)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs=200, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=1):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def butter_highpass_filter(data, cutoff_freq=0.1, sampling_rate=200, order=1):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
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
            (pl.sum_horizontal(TARGET_COLS)).alias("total_votes"),
        )
        .with_columns(
            (
                # pl.when(pl.col("total_votes") > 20)
                # .then(1)
                # .when((pl.col("total_votes") <= 20) & (pl.col("total_votes") > 10))
                # .then(0.75)
                # .otherwise(0.5)
                pl.col("total_votes")
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


class HMSTrain(Dataset):
    def __init__(self, df, data_dir, low_f=0.5, high_f=50, order=5, transforms=None):
        self.df = df  # .unique(subset=["eeg_id", *TARGET_COLS])
        self.df = get_sample_weights(self.df)
        self.unq_ids = self.df["eeg_id"].unique().to_list()
        self.data_dir = data_dir
        self.df = norm_target_cols(self.df)
        self.low_f = low_f
        self.high_f = high_f
        self.order = order
        self.transforms = transforms

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
        data = load_eeg_data(
            self.data_dir, eeg_id, eeg_sub_id, self.low_f, self.high_f, self.order
        )
        if self.transforms is not None:
            for tfm in self.transforms:
                data = tfm(data)
        targets = patient_df[TARGET_COLS].values.flatten()
        # targets = targets / targets.sum()
        sample_weight = patient_df["sample_weight"].iloc[0]
        return {
            "data": data.astype(np.float32),
            "targets": targets.astype(np.float32),
            "sample_weight": sample_weight.astype(np.float32),
            "eeg_id": eeg_id.astype(np.int32),
            "eeg_sub_id": eeg_sub_id.astype(np.int32),
        }


class HMSVal(Dataset):
    def __init__(self, df, data_dir, low_f=0.5, high_f=50, order=5, transforms=None):
        self.df = df
        self.data_dir = data_dir
        self.low_f = low_f
        self.high_f = high_f
        self.order = order
        self.df = norm_target_cols(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_df = self.df[idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        # offset = patient_df["eeg_label_offset_seconds"].iloc[0]
        eeg_sub_id = patient_df["eeg_sub_id"].iloc[0]
        data = load_eeg_data(
            self.data_dir, eeg_id, eeg_sub_id, self.low_f, self.high_f, self.order
        )

        targets = patient_df[TARGET_COLS].values.flatten()
        # targets = targets / targets.sum()
        return {
            "data": data.astype(np.float32),
            "targets": targets.astype(np.float32),
            "eeg_id": eeg_id.astype(np.int64),
            "eeg_sub_id": eeg_sub_id.astype(np.int64),
            "num_votes": patient_df["num_votes"].iloc[0].astype(np.float32),
        }


class HMSTest(Dataset):
    def __init__(self, df, data_dir, low_f=0.5, high_f=50, order=5, transforms=None):
        self.df = df
        self.data_dir = data_dir
        self.low_f = low_f
        self.high_f = high_f
        self.order = order

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_df = self.df[idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        eeg_sub_id = patient_df["eeg_sub_id"].iloc[0]
        data = load_eeg_data(
            self.data_dir, eeg_id, eeg_sub_id, self.low_f, self.high_f, self.order
        )

        return {
            "data": data.astype(np.float32),
            "eeg_id": eeg_id.astype(np.int64),
        }


def fix_nulls(data):
    for i in range(data.shape[1]):
        if np.all(np.isnan(data[:, i])):
            data[:, i] = 0
        else:
            mean = np.nanmedian(data[:, i])
            data[:, i] = np.nan_to_num(data[:, i], nan=mean)

    return data


def load_eeg_data(data_dir, eeg_id, eeg_sub_id, low_f=0.5, high_f=40, order=5):
    npy_path = data_dir / f"{eeg_id}_{eeg_sub_id}.npy"
    data = np.load(npy_path)
    data = fix_nulls(data)
    # data = butter_bandpass_filter(
    #     data, lowcut=low_f, highcut=high_f, fs=SAMPLE_RATE, order=order
    # )
    out = np.zeros((data.shape[0], 16), dtype=np.float32)
    for i, (j, k) in enumerate(EEG_GROUP_IDX):
        out[:, i] = data[:, j] - data[:, k]
    out = out.astype(np.float64)
    out = notch_filter(out, SAMPLE_RATE, 60)
    out = butter_lowpass_filter(
        out, cutoff_freq=high_f, sampling_rate=SAMPLE_RATE, order=order
    )
    out = butter_highpass_filter(
        out, cutoff_freq=low_f, sampling_rate=SAMPLE_RATE, order=order
    )
    out = np.clip(out, -500, 500)
    # out = np.log1p(np.abs(out)) * np.sign(out) / 3
    out = out / 500
    return out[8:-8, :].astype(np.float32)
