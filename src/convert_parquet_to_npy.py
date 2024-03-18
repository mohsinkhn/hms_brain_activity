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
        if "_" in eeg_id:
            eeg_id_base = eeg_id.split("_")[0]
        else:
            eeg_id_base = eeg_id
        if "eeg_label_offset_seconds" in df.columns:
            offset = df_["eeg_label_offset_seconds"][i]
            eeg_df = pl.read_parquet(Path(data_dir) / f"{eeg_id_base}.parquet")
            eeg_data = eeg_df[
                int(offset * SAMPLE_RATE) : int((offset + EEG_DURATION) * SAMPLE_RATE)
            ].to_numpy()
            np.save(Path(out_dir) / f"{eeg_id}.npy", eeg_data)
        else:
            eeg_df = pl.read_parquet(Path(data_dir) / f"{eeg_id_base}.parquet")
            eeg_data = eeg_df.to_numpy()
            np.save(Path(out_dir) / f"{eeg_id}.npy", eeg_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        df = pl.read_csv(Path(args.data_dir) / "train.csv")
        convert_parquet_to_npy(
            df.with_columns(
                (
                    pl.col("eeg_id").cast(pl.String)
                    + "_"
                    + pl.col("eeg_sub_id").cast(pl.String)
                ).alias("eeg_id")
            ),
            Path(args.data_dir) / "train_eegs",
            Path(args.out_dir) / "train_eegs",
        )
    else:
        df = pl.read_csv(Path(args.data_dir) / "test.csv")
        df = df.with_columns(pl.lit(0).alias("eeg_sub_id"))
        df = df.with_columns(
            (
                pl.col("eeg_id").cast(pl.String)
                + "_"
                + pl.col("eeg_sub_id").cast(pl.String)
            ).alias("eeg_id")
        )
        (Path(args.out_dir) / "test_eegs").mkdir(exist_ok=True, parents=True)
        convert_parquet_to_npy(
            df, Path(args.data_dir) / "test_eegs", Path(args.out_dir) / "test_eegs"
        )
