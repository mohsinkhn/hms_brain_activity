from typing import Union, Dict, Any

import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.settings import EEG_GROUP_IDX, SAMPLE_RATE, EEG_DURATION, TARGET_COLS
from src.utils.filtering import (
    butter_lowpass_filter,
    butter_highpass_filter,
    notch_filter,
)


def fix_nulls(data):
    med = np.median(data)
    return np.nan_to_num(data, nan=0, posinf=0.0, neginf=0.0)


class Preprocessor(object):
    def __init__(
        self, low_f, high_f, notch=60, order=5, scale="simple", kind="montage", fs=200
    ):
        self.fs = fs
        self.lowcut = low_f
        self.highcut = high_f
        self.notch = notch
        self.order = order
        self.kind = kind
        self.scale = scale
        self.fs = fs

    def __call__(self, data):
        data = fix_nulls(data)
        data = notch_filter(data, self.fs, self.notch)
        data = butter_lowpass_filter(data, self.highcut, self.fs, self.order)
        data = data[8:-8]
        if self.lowcut > 0:
            data = butter_highpass_filter(data, self.lowcut, self.fs, self.order)
        data = np.clip(data, -1024, 1024)
        if self.kind == "montage":
            data = self.montage(data)
        if self.scale == "log":
            data = self.log_scale(data)
        else:
            data = self.simple_scale(data)
        return np.ascontiguousarray(data)

    def log_scale(self, data):
        return np.log1p(np.abs(data)) * np.sign(data)

    def simple_scale(self, data):
        return np.clip(data, -600, 600) / 6

    def montage(self, data):
        out = np.zeros((data.shape[0], 16), dtype=np.float32)
        for i, (j, k) in enumerate(EEG_GROUP_IDX):
            out[:, i] = data[:, j] - data[:, k]
        return out


class HMSTrainData(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: Union[str, Path],
        preprocessor: Preprocessor,
        unq_batch: str = "eeg_id",
        max_weight=20,
        transforms=None,
    ):
        self.preprocessor = preprocessor
        self.unq_batch = unq_batch
        self.max_weight = max_weight
        self.transforms = transforms

        df = df.sort_values(by=["eeg_id"] + TARGET_COLS)
        self.egg_ids, self.offsets, self.targets = parse_dataframe(df)
        if unq_batch == "eeg_id":
            unq_cols = ["eeg_id"]
        else:
            unq_cols = ["eeg_id"] + TARGET_COLS

        df["group"] = (~df[unq_cols].duplicated()).cumsum() - 1
        df["idx"] = range(len(df))
        self.unq_start_ids = df.groupby("group")["idx"].first().tolist()
        self.unq_lens = df.groupby("group")["idx"].count().tolist()
        unq_eeg_ids = df["eeg_id"].unique().tolist()
        self.eegs_data = load_data(unq_eeg_ids, eeg_dir)

    def __len__(self):
        return len(self.unq_start_ids)

    def __getitem__(self, index):
        row_idx = self.unq_start_ids[index]
        delta = np.random.randint(0, self.unq_lens[index])
        row_idx += delta
        eeg_id = self.egg_ids[row_idx]
        offset = self.offsets[row_idx]

        # Get EEG data
        eeg_data = self.eegs_data[eeg_id]
        eeg_data = eeg_data[
            int(offset * SAMPLE_RATE) : int((offset + EEG_DURATION) * SAMPLE_RATE)
        ]
        eeg_data = self.preprocessor(eeg_data)

        # Get target
        target = self.targets[row_idx]
        total_votes = target.sum()
        target = target / total_votes

        for tfm in self.transforms:
            eeg_data, target = tfm(eeg_data, target)

        return {
            "eeg_data": eeg_data.astype(np.float32),
            "targets": target.astype(np.float32),
            "eeg_id": int(eeg_id),
            "offset": int(offset),
            "sample_weight": float(min(self.max_weight, total_votes)),
            "total_votes": int(total_votes),
        }


class HMSTestData(Dataset):
    def __init__(
        self, df: pd.DataFrame, eeg_dir: Union[str, Path], preprocessor: Preprocessor
    ):
        self.eeg_ids, self.offsets, self.targets = parse_dataframe(df)
        self.preprocessor = preprocessor

        self.unq_eeg_ids = df["eeg_id"].unique().tolist()
        self.eegs_data = load_data(self.unq_eeg_ids, eeg_dir)

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, index):
        eeg_id = self.eeg_ids[index]
        offset = self.offsets[index]

        # Get EEG data
        eeg_data = self.eegs_data[eeg_id]
        eeg_data = eeg_data[
            int(offset * SAMPLE_RATE) : int((offset + EEG_DURATION) * SAMPLE_RATE)
        ]
        eeg_data = self.preprocessor(eeg_data)

        target = self.targets[index]
        total_votes = target.sum()
        if total_votes == 0:
            total_votes = 1
            target = target + 1
        target = target / total_votes

        return {
            "eeg_data": eeg_data.astype(np.float32),
            "targets": target.astype(np.float32),
            "eeg_id": int(eeg_id),
            "offset": int(offset),
            "total_votes": int(total_votes),
        }


class HMSTestDataKG(Dataset):
    def __init__(
        self, df: pd.DataFrame, eeg_dir: Union[str, Path], preprocessor: Preprocessor
    ):
        self.eeg_ids, self.offsets, self.targets = parse_dataframe(df)
        self.preprocessor = preprocessor

        self.unq_eeg_ids = df["eeg_id"].unique().tolist()
        self.eeg_dir = eeg_dir
        # self.eegs_data = load_data(self.unq_eeg_ids, eeg_dir)

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, index):
        eeg_id = self.eeg_ids[index]
        offset = self.offsets[index]

        # Get EEG data
        eeg_data = np.load(Path(self.eeg_dir) / f"{eeg_id}.npy")
        eeg_data = eeg_data[
            int(offset * SAMPLE_RATE) : int((offset + EEG_DURATION) * SAMPLE_RATE)
        ]
        eeg_data = self.preprocessor(eeg_data)

        target = self.targets[index]
        total_votes = target.sum()
        if total_votes == 0:
            total_votes = 1
            target = target + 1
        target = target / total_votes

        return {
            "eeg_data": eeg_data.astype(np.float32),
            "targets": target.astype(np.float32),
            "eeg_id": int(eeg_id),
            "offset": int(offset),
            "total_votes": int(total_votes),
        }


def load_data(eeg_ids, eeg_dir):
    eegs_data = {}
    for eeg_id in tqdm(eeg_ids):
        eeg_data = np.load(Path(eeg_dir) / f"{eeg_id}.npy")
        eegs_data[eeg_id] = eeg_data
    return eegs_data


def parse_dataframe(df):
    eeg_ids = df["eeg_id"].astype(int).tolist()
    if "eeg_label_offset_seconds" not in df.columns:
        df["eeg_label_offset_seconds"] = 0
    offsets = df["eeg_label_offset_seconds"].astype(int).tolist()
    if TARGET_COLS[0] not in df.columns:
        df[TARGET_COLS] = 1
    targets = df[TARGET_COLS].astype(np.float32).values
    return eeg_ids, offsets, targets
