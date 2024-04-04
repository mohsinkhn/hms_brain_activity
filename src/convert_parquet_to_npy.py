import argparse
from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from src.settings import TARGET_COLS, SAMPLE_RATE, EEG_DURATION  # noqa


def convert_parquet_to_npy(df, data_dir, out_dir):
    """Convert the parquet data to numpy format.

    :param df: The dataframe with the eeg ids data.
    """
    df_ = df.unique(subset=["eeg_id"])
    for i in tqdm(range(len(df_))):
        eeg_id = df_["eeg_id"][i]
        eeg_data = pl.read_parquet(Path(data_dir) / f"{eeg_id}.parquet").to_numpy()
        np.save(Path(out_dir) / f"{eeg_id}.npy", eeg_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    filepath = (
        Path(args.data_dir) / "train.csv"
        if args.train
        else Path(args.data_dir) / "test.csv"
    )
    df = pl.read_csv(filepath)
    convert_parquet_to_npy(df, args.data_dir, args.out_dir)
