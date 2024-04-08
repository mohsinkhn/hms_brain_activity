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

from src.nn_datasets.components.eegdataset_clean import Preprocessor, load_data


class HMSTrainSpecData(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: Union[str, Path],
        spec_dir: Union[str, Path],
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
        self.egg_ids, self.offsets, self.spec_ids, self.spec_offsets, self.targets = (
            parse_dataframe(df)
        )
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
        unq_spec_ids = df["spectrogram_id"].unique().tolist()
        self.spec_data = load_data(unq_spec_ids, spec_dir)

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

        spec_id = self.spec_ids[row_idx]
        spec_data = self.spec_data[spec_id]
        spec_offset = self.spec_offsets[row_idx]
        start = np.searchsorted(spec_data[:, 0, 0], spec_offset)
        end = np.searchsorted(spec_data[:, 0, 0], spec_offset + 300)
        spec_data_out = np.zeros((300, 100, 4))
        spec_data_out[: end - start] = spec_data[start:end, 1:, :]
        spec_data = spec_data_out[:]
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
            "spec_data": spec_data.astype(np.float32),
        }


class HMSTestSpecData(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: Union[str, Path],
        spec_dir: Union[str, Path],
        preprocessor: Preprocessor,
    ):
        self.eeg_ids, self.offsets, self.spec_ids, self.spec_offsets, self.targets = (
            parse_dataframe(df)
        )
        self.preprocessor = preprocessor

        self.unq_eeg_ids = df["eeg_id"].unique().tolist()
        self.eegs_data = load_data(self.unq_eeg_ids, eeg_dir)
        unq_spec_ids = df["spectrogram_id"].unique().tolist()
        self.spec_data = load_data(unq_spec_ids, spec_dir)

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

        spec_id = self.spec_ids[index]
        spec_data = self.spec_data[spec_id]
        spec_offset = self.spec_offsets[index]
        start = np.searchsorted(spec_data[:, 0, 0], spec_offset)
        end = np.searchsorted(spec_data[:, 0, 0], spec_offset + 300)
        spec_data_out = np.zeros((300, 100, 4))
        spec_data_out[: end - start] = spec_data[start:end, 1:, :]
        spec_data = spec_data_out[:]

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
            "spec_data": spec_data.astype(np.float32),
        }


def parse_dataframe(df):
    eeg_ids = df["eeg_id"].astype(int).tolist()
    spec_ids = df["spectrogram_id"].astype(int).tolist()
    if "eeg_label_offset_seconds" not in df.columns:
        df["eeg_label_offset_seconds"] = 0
    if "spectrogram_label_offset_seconds" not in df.columns:
        df["spectrogram_label_offset_seconds"] = 0
    offsets = df["eeg_label_offset_seconds"].astype(int).tolist()
    spec_offsets = df["spectrogram_label_offset_seconds"].astype(int).tolist()
    if TARGET_COLS[0] not in df.columns:
        df[TARGET_COLS] = 1
    targets = df[TARGET_COLS].astype(np.float32).values
    return eeg_ids, offsets, spec_ids, spec_offsets, targets
