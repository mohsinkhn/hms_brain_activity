from collections import defaultdict
import random

import numpy as np
from pathlib import Path
import polars as pl
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from torch.utils.data import Dataset
from tqdm.auto import tqdm

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
            )
            .alias("sample_weight")
            .clip(1, 20)
        )
    )


def norm_target_cols(df):
    norm_targets = df.select(TARGET_COLS).to_numpy().astype(np.float32)
    target_sums = norm_targets.sum(axis=1, keepdims=True)
    norm_targets = norm_targets / target_sums
    return df.with_columns(
        *[pl.Series(target, norm_targets[:, i]) for i, target in enumerate(TARGET_COLS)]
    ).with_columns(pl.Series("num_votes", target_sums.flatten()))


class HMSTrain(Dataset):
    def __init__(
        self,
        df,
        data_dir,
        pseudo_df=None,
        pseudo_weight=0.5,
        low_f=0.5,
        high_f=50,
        order=5,
        transforms=None,
        scale="log",
        remove_edge="post",
    ):
        self.df = df  # .unique(subset=["eeg_id", *TARGET_COLS])
        self.df = get_sample_weights(self.df)
        self.unq_ids = self.df["eeg_id"].unique().to_list()
        self.data_dir = data_dir
        self.df = norm_target_cols(self.df)
        if pseudo_df is not None:
            self.df = self.df.join(pseudo_df, on=["eeg_id", "eeg_sub_id"], how="left")
            self.df = self.df.with_columns(
                *[
                    pl.when(pl.col("num_votes") < 10)
                    .then(
                        pl.col(target)
                        + pseudo_weight * pl.col(f"{target}_pred").fill_null(0)
                    )
                    .otherwise(pl.col(target))
                    .alias(target)
                    for target in TARGET_COLS
                ]
            )
            self.df = norm_target_cols(self.df)
        self.low_f = low_f
        self.high_f = high_f
        self.order = order
        self.transforms = transforms
        self.scale = scale
        self.remove_edge = remove_edge

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
            self.data_dir,
            eeg_id,
            eeg_sub_id,
            self.low_f,
            self.high_f,
            self.order,
            self.remove_edge,
        )
        if self.transforms is not None:
            for tfm in self.transforms:
                data, _, _ = tfm(data)
        if self.scale == "constant":
            data = data / 100
        elif self.scale == "log":
            data = np.log1p(np.abs(data)) * np.sign(data)
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
    def __init__(
        self,
        df,
        data_dir,
        low_f=0.5,
        high_f=50,
        order=5,
        transforms=None,
        scale="log",
        remove_edge="post",
    ):
        self.df = df
        self.df = sample_sub_ids(self.df)
        self.data_dir = data_dir
        self.low_f = low_f
        self.high_f = high_f
        self.order = order
        self.df = norm_target_cols(self.df)
        self.scale = scale
        self.remove_edge = remove_edge
        self.eegs, self.eeg_mapping = load_eegs(
            self.df, data_dir, low_f, high_f, order, remove_edge
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_df = self.df[idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        # offset = patient_df["eeg_label_offset_seconds"].iloc[0]
        eeg_sub_id = patient_df["eeg_sub_id"].iloc[0]
        data = self.eegs[self.eeg_mapping[(eeg_id, eeg_sub_id)]]
        # data = load_eeg_data(
        #     self.data_dir,
        #     eeg_id,
        #     eeg_sub_id,
        #     self.low_f,
        #     self.high_f,
        #     self.order,
        #     self.remove_edge,
        # )
        if self.scale == "constant":
            data = data / 100
        elif self.scale == "log":
            data = np.log1p(np.abs(data)) * np.sign(data)
        data = data.astype(np.float32)
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
    def __init__(
        self,
        df,
        data_dir,
        low_f=0.5,
        high_f=50,
        order=5,
        transforms=None,
        scale="log",
        remove_edge="post",
    ):
        self.df = df
        self.data_dir = data_dir
        self.low_f = low_f
        self.high_f = high_f
        self.order = order
        self.scale = scale
        self.remove_edge = remove_edge

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_df = self.df[idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        eeg_sub_id = patient_df["eeg_sub_id"].iloc[0]
        data = load_eeg_data(
            self.data_dir,
            eeg_id,
            eeg_sub_id,
            self.low_f,
            self.high_f,
            self.order,
            self.remove_edge,
        )
        if self.scale == "constant":
            data = data / 100
        elif self.scale == "log":
            data = np.log1p(np.abs(data)) * np.sign(data)
        data = data.astype(np.float32)
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


def load_eeg_data(
    data_dir, eeg_id, eeg_sub_id, low_f=0.5, high_f=40, order=5, remove_edge="post"
):
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
    if remove_edge == "pre":
        out = out[8:-8]
    out = butter_highpass_filter(
        out, cutoff_freq=low_f, sampling_rate=SAMPLE_RATE, order=order
    )
    # out = out - np.median(out, axis=0)
    out = np.clip(out, -1000, 1000)
    # out = np.log1p(np.abs(out)) * np.sign(out)
    # out = out / 100
    if remove_edge == "post":
        out = out[8:-8]
    # out = out[8:-8]

    return out


class HMSTrainv2(Dataset):
    def __init__(
        self,
        df,
        data_dir,
        pseudo_df=None,
        pseudo_weight=0.5,
        low_f=0.2,
        high_f=50,
        order=5,
        transforms=None,
        scale="log",
        remove_edge="post",
    ):
        self.df = df  # .unique(subset=["eeg_id", *TARGET_COLS])
        self.df = sample_sub_ids(self.df)
        print(self.df.shape)
        self.df = get_sample_weights(self.df)
        self.unq_ids = self.df["eeg_id"].unique().to_list()
        self.eegs, self.eeg_mapping = load_eegs(
            self.df, data_dir, low_f, high_f, order, remove_edge
        )
        self.data_dir = data_dir
        if pseudo_df is not None:
            self.df = self.df.join(pseudo_df, on=["eeg_id", "eeg_sub_id"], how="left")
            self.df = self.df.with_columns(
                *[
                    pl.when(pl.col("total_votes") < 10)
                    .then(
                        pl.col(target)
                        + pseudo_weight * pl.col(f"{target}_pred").fill_null(0)
                    )
                    .otherwise(pl.col(target))
                    .alias(target)
                    for target in TARGET_COLS
                ]
            )
        self.df = norm_target_cols(self.df)
        self.low_f = low_f
        self.high_f = high_f
        self.order = order
        self.transforms = transforms
        self.sample_ids = np.zeros_like(np.array(self.unq_ids), dtype=np.int64)
        self.scale = scale
        self.remove_edge = remove_edge

    def __len__(self):
        return len(self.unq_ids)

    def __getitem__(self, idx):
        unq_id = self.unq_ids[idx]
        patient_df = self.df.filter(pl.col("eeg_id") == unq_id)
        sample_idx = int(self.sample_ids[idx])
        self.sample_ids[idx] = (sample_idx + 19) % len(patient_df)
        # idx = random.choices(
        #     range(len(patient_df)),  # weights=patient_df["sample_weight"].to_numpy()
        # )[0]

        patient_df = patient_df[sample_idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        # offset = patient_df["eeg_label_offset_seconds"].iloc[0]
        eeg_sub_id = patient_df["eeg_sub_id"].iloc[0]
        data = self.eegs[self.eeg_mapping[(eeg_id, eeg_sub_id)]]
        # data = load_eeg_data(
        #     self.data_dir,
        #     eeg_id,
        #     eeg_sub_id,
        #     self.low_f,
        #     self.high_f,
        #     self.order,
        #     self.remove_edge,
        # )
        if self.transforms is not None:
            for tfm in self.transforms:
                data, _, _ = tfm(data)
        if self.scale == "constant":
            data = data / 100
        elif self.scale == "log":
            data = np.log1p(np.abs(data)) * np.sign(data)
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


class HMSTrainv3(Dataset):
    def __init__(
        self,
        df,
        data_dir,
        pseudo_df=None,
        pseudo_weight=0.5,
        low_f=0.2,
        high_f=50,
        order=5,
        transforms=None,
        scale="log",
        remove_edge="post",
    ):
        self.df = df  # .unique(subset=["eeg_id", *TARGET_COLS])
        self.df = sample_sub_ids(self.df)
        print(self.df.shape)
        self.df = get_sample_weights(self.df)
        self.unq_ids = self.df["eeg_id"].unique().to_list()
        self.eegs, self.eeg_mapping = load_eegs(
            self.df, data_dir, low_f, high_f, order, remove_edge
        )
        self.data_dir = data_dir
        if pseudo_df is not None:
            self.df = self.df.join(pseudo_df, on=["eeg_id", "eeg_sub_id"], how="left")
            self.df = self.df.with_columns(
                *[
                    pl.when(pl.col("total_votes") < 10)
                    .then(
                        pl.col(target)
                        + pseudo_weight * pl.col(f"{target}_pred").fill_null(0)
                    )
                    .otherwise(pl.col(target))
                    .alias(target)
                    for target in TARGET_COLS
                ]
            )
        self.df = norm_target_cols(self.df)
        self.low_f = low_f
        self.high_f = high_f
        self.order = order
        self.transforms = transforms
        self.sample_ids = np.zeros_like(np.array(self.unq_ids), dtype=np.int64)
        self.scale = scale
        self.remove_edge = remove_edge

    def __len__(self):
        return len(self.unq_ids)

    def __getitem__(self, idx):
        unq_id = self.unq_ids[idx]
        patient_df = self.df.filter(pl.col("eeg_id") == unq_id)
        sample_idx = int(self.sample_ids[idx])
        self.sample_ids[idx] = (sample_idx + 19) % len(patient_df)
        # idx = random.choices(
        #     range(len(patient_df)),  # weights=patient_df["sample_weight"].to_numpy()
        # )[0]

        patient_df = patient_df[sample_idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        # offset = patient_df["eeg_label_offset_seconds"].iloc[0]
        eeg_sub_id = patient_df["eeg_sub_id"].iloc[0]
        spec_id = patient_df["spectrogram_id"].iloc[0]
        spec_offset = patient_df["spectrogram_label_offset_seconds"].iloc[0]
        data = self.eegs[self.eeg_mapping[(eeg_id, eeg_sub_id)]]
        # data = load_eeg_data(
        #     self.data_dir,
        #     eeg_id,
        #     eeg_sub_id,
        #     self.low_f,
        #     self.high_f,
        #     self.order,
        #     self.remove_edge,
        # )
        spec = load_spec_data(self.data_dir, spec_id, spec_offset)
        targets = patient_df[TARGET_COLS].values.flatten()

        if self.transforms is not None:
            for tfm in self.transforms:
                data, spec, targets = tfm(data, spec, targets)
        if self.scale == "constant":
            data = data / 100
        elif self.scale == "log":
            data = np.log1p(np.abs(data)) * np.sign(data)
        # targets = targets / targets.sum()
        sample_weight = patient_df["sample_weight"].iloc[0]

        return {
            "data": data.astype(np.float32),
            "targets": targets.astype(np.float32),
            "sample_weight": sample_weight.astype(np.float32),
            "eeg_id": eeg_id.astype(np.int32),
            "eeg_sub_id": eeg_sub_id.astype(np.int32),
            "spec": spec.astype(np.float32),
        }


class HMSValv3(Dataset):
    def __init__(
        self,
        df,
        data_dir,
        low_f=0.5,
        high_f=50,
        order=5,
        transforms=None,
        scale="log",
        remove_edge="post",
    ):
        self.df = df
        self.df = sample_sub_ids(self.df)
        self.data_dir = data_dir
        self.low_f = low_f
        self.high_f = high_f
        self.order = order
        self.df = norm_target_cols(self.df)
        self.scale = scale
        self.remove_edge = remove_edge
        self.eegs, self.eeg_mapping = load_eegs(
            self.df, data_dir, low_f, high_f, order, remove_edge
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_df = self.df[idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        # offset = patient_df["eeg_label_offset_seconds"].iloc[0]
        eeg_sub_id = patient_df["eeg_sub_id"].iloc[0]
        spec_id = patient_df["spectrogram_id"].iloc[0]
        spec_offset = patient_df["spectrogram_label_offset_seconds"].iloc[0]
        data = self.eegs[self.eeg_mapping[(eeg_id, eeg_sub_id)]]
        # data = load_eeg_data(
        #     self.data_dir,
        #     eeg_id,
        #     eeg_sub_id,
        #     self.low_f,
        #     self.high_f,
        #     self.order,
        #     self.remove_edge,
        # )
        if self.scale == "constant":
            data = data / 100
        elif self.scale == "log":
            data = np.log1p(np.abs(data)) * np.sign(data)
        data = data.astype(np.float32)
        targets = patient_df[TARGET_COLS].values.flatten()
        # targets = targets / targets.sum()
        return {
            "data": data.astype(np.float32),
            "targets": targets.astype(np.float32),
            "eeg_id": eeg_id.astype(np.int64),
            "eeg_sub_id": eeg_sub_id.astype(np.int64),
            "spec": load_spec_data(self.data_dir, spec_id, spec_offset).astype(
                np.float32
            ),
            "num_votes": patient_df["num_votes"].iloc[0].astype(np.float32),
        }


class HMSTrainPre(Dataset):
    def __init__(
        self,
        df,
        data_dir,
        pseudo_df=None,
        pseudo_weight=0.5,
        low_f=0.2,
        high_f=50,
        order=5,
        transforms=None,
        scale="log",
        remove_edge="pre",
    ):
        self.df = df  # .unique(subset=["eeg_id", *TARGET_COLS])
        self.unq_ids = self.df["eeg_id"].unique().to_list()
        self.data_dir = data_dir
        self.df = self.df.with_columns(
            pl.col("target").map_dict(
                {
                    "bckg": 0,
                    "tcsz": 1,
                    "fnsz": 2,
                    "cpsz": 3,
                    "gnsz": 4,
                    "absz": 5,
                    "seiz": 6,
                    "tnsz": 7,
                    "spsz": 8,
                    "mysz": 9,
                }
            )
        )
        self.low_f = low_f
        self.high_f = high_f
        self.order = order
        self.transforms = transforms
        self.scale = scale
        self.remove_edge = remove_edge

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
        data = (
            load_eeg_data(
                self.data_dir,
                eeg_id,
                eeg_sub_id,
                self.low_f,
                self.high_f,
                self.order,
                self.remove_edge,
            )
            * 10**6
        )
        if self.transforms is not None:
            for tfm in self.transforms:
                data = tfm(data)
        if self.scale == "constant":
            data = data / 100
        elif self.scale == "log":
            data = np.log1p(np.abs(data)) * np.sign(data)
        new_data = np.zeros(shape=(9984, 16), dtype=np.float32)
        new_data[: data.shape[0], :] = data[:]
        data = new_data[:]
        targets = patient_df["target"].values.flatten()
        # targets = targets / targets.sum()
        return {
            "data": data.astype(np.float32),
            "targets": targets.astype(int),
            # "eeg_id": eeg_id.astype(np.int32),
            # "eeg_sub_id": eeg_sub_id.astype(np.int32),
        }


def load_eegs(df, data_dir, low_f=0.5, high_f=40, order=5, remove_edge="post"):
    data = np.zeros((len(df), 9984, 16), dtype=np.float32)
    mapping = defaultdict(tuple)
    for i, row in tqdm(enumerate(df.iter_rows(named=True))):
        eeg_id = row["eeg_id"]
        eeg_sub_id = row["eeg_sub_id"]
        data[i] = load_eeg_data(
            data_dir, eeg_id, eeg_sub_id, low_f, high_f, order, remove_edge
        )
        mapping[i] = (eeg_id, eeg_sub_id)
    rev_mapping = {v: k for k, v in mapping.items()}
    return data, rev_mapping


def sample_sub_ids(df: pl.DataFrame, n_samples: int = 10):
    # Take maximum 20 samples from each eeg_id
    df = (
        df.with_columns(
            pl.col("eeg_sub_id").max().over("eeg_id").alias("max_sub_id"),
        )
        .with_columns(
            pl.when(pl.col("max_sub_id") >= n_samples)
            .then(pl.col("eeg_sub_id") / pl.col("max_sub_id") * n_samples)
            .otherwise(pl.col("eeg_sub_id"))
            .alias("sampled_sub_id")
            .round()
        )
        .unique(subset=["eeg_id", "sampled_sub_id"])
    )
    return df


def load_spec_data(data_dir, spec_id, spec_offset):
    npy_path = (
        Path(str(data_dir).replace("eegs", "spectrograms")) / f"{spec_id}.parquet.npy"
    )
    data = np.load(npy_path)
    data = np.nan_to_num(data, nan=0)
    data = np.clip(data, np.exp(-4), np.exp(6))
    start = np.searchsorted(data[:, 0, 0], spec_offset, side="right")
    data = np.log(data[start : start + 300, 1:, :])
    mean = data.mean()
    std = data.std()
    data = (data - mean) / (std + 1e-6)
    return data
