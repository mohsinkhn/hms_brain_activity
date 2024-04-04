from typing import Union, Dict, Any

import numpy as np
from pathlib import Path
import polars as pl
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.settings import EEG_GROUP_IDX, SAMPLE_RATE, EEG_DURATION, TARGET_COLS
from src.utils.filtering import (
    butter_lowpass_filter,
    butter_highpass_filter,
    notch_filter,
)


class Preprocessor(object):
    def __init__(
        self, lowcut, highcut, notch=60, order=5, scale="simple", kind="montage", fs=200
    ):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch = notch
        self.order = order
        self.kind = kind
        self.scale = scale
        self.fs = fs

    def __call__(self, data):
        data = notch_filter(data, self.fs, self.notch)
        data = butter_lowpass_filter(data, self.highcut, self.fs, self.order)
        data = data[8:-8]
        data = butter_highpass_filter(data, self.lowcut, self.fs, self.order)
        data = np.clip(data, -1024, 1024)
        if self.kind == "montage":
            data = self.montage(data)
        if self.scale == "log":
            data = self.log_scale(data)
        else:
            data = self.simple_scale(data)
        return data

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
        df: pl.DataFrame,
        eeg_dir: Union[str, Path],
        preprocessor: Preprocessor,
        unq_batch: tuple = None,
        max_weight=20,
    ):
        self.df = df
        self.preprocessor = preprocessor
        self.unq_batch = {c: i for i, c in enumerate(unq_batch)}
        self.max_weight = max_weight

        self.unq_ids = df[list(unq_batch.keys())].unique().rows()
        eeg_ids = df["eeg_id"].unique().to_list()
        self.eegs_data = load_data(eeg_ids, eeg_dir)

    def __len__(self):
        return len(self.unq_ids)

    def __getitem__(self, index) -> Dict[Any]:
        unq_id = self.unq_ids[index]
        df_ = self.df.filter(
            pl.reduce(
                lambda a, b: a & b,
                [pl.col(c) == unq_id[i] for c, i in self.unq_batch.items()],
            )
        ).sample(1)
        eeg_id = df_["eeg_id"][0]
        offset = df_["eeg_label_offset_seconds"][0]

        # Get EEG data
        eeg_data = self.eegs_data[eeg_id]
        eeg_data = eeg_data[
            int(offset * SAMPLE_RATE) : int((offset + EEG_DURATION) * SAMPLE_RATE)
        ]
        eeg_data = self.preprocessor(eeg_data)

        # Get target
        target = df_[TARGET_COLS].to_numpy()[0]
        total_votes = target.sum()
        target = target / total_votes

        return {
            "eeg_data": eeg_data,
            "targets": target,
            "eeg_id": eeg_id,
            "offset": offset,
            "sample_weight": min(self.max_weight, total_votes),
        }


class HMSTestData(Dataset):
    def __init__(
        self, df: pl.DataFrame, eeg_dir: Union[str, Path], preprocessor: Preprocessor
    ):
        self.df = df
        self.preprocessor = preprocessor

        self.eeg_ids = df["eeg_id"].unique().to_list()
        self.eegs_data = load_data(self.eeg_ids, eeg_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Dict[Any]:
        df_ = self.df[index]
        eeg_id = df_["eeg_id"][0]
        offset = df_["eeg_label_offset_seconds"][0]

        # Get EEG data
        eeg_data = self.eegs_data[eeg_id]
        eeg_data = eeg_data[
            int(offset * SAMPLE_RATE) : int((offset + EEG_DURATION) * SAMPLE_RATE)
        ]
        eeg_data = self.preprocessor(eeg_data)

        if TARGET_COLS[0] in df_.columns:
            target = df_[TARGET_COLS].to_numpy()[0]
            total_votes = target.sum()
            target = target / total_votes
        else:
            target = np.zeros((6,))
            total_votes = 1

        return {
            "eeg_data": eeg_data,
            "eeg_id": eeg_id,
            "offset": offset,
            "targets": target,
            "total_votes": total_votes,
        }


def load_data(self, eeg_ids, eeg_dir):
    eegs_data = {}
    for eeg_id in tqdm(self.eeg_ids):
        eeg_data = np.load(eeg_dir / f"{eeg_id}.npy")
        eegs_data[eeg_id] = eeg_data
    return eegs_data
