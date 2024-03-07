import random
import polars as pl
from torch.utils.data import Dataset

from src.settings import TARGET_COLS, TRAIN_EEGS, SAMPLE_RATE, EEG_DURATION  # noqa


def load_eeg_data(eeg_id, offset):
    pq_path = f"{TRAIN_EEGS}/{eeg_id}.parquet"
    df = pl.read_parquet(pq_path)
    offset = int(offset * SAMPLE_RATE)
    return df[offset : offset + SAMPLE_RATE * EEG_DURATION].to_numpy()


class HMSTrainEEGData(Dataset):
    def __init__(self, patient_ids, df):
        self.patient_ids = patient_ids
        self.df = df

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
        offset = patient_df["eeg_label_offset_seconds"].iloc[0]
        data = load_eeg_data(eeg_id, offset)
        targets = patient_df[TARGET_COLS]
        return data, targets


class HMSValEEGData(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        patient_df = self.df[idx].to_pandas()
        eeg_id = patient_df["eeg_id"].iloc[0]
        offset = patient_df["eeg_label_offset_seconds"].iloc[0]
        data = load_eeg_data(eeg_id, offset)
        targets = patient_df[TARGET_COLS]
        return data, targets
