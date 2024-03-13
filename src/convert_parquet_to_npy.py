from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from src.settings import TARGET_COLS, SAMPLE_RATE, EEG_DURATION  # noqa


def convert_parquet_to_npy(df, data_dir):
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
            # means = np.nanmedian(eeg_data, axis=1)
            # means = np.nan_to_num(means, nan=0)
            # eeg_data = eeg_data - means[:, None]
            # eeg_data = np.nan_to_num(eeg_data, nan=0)
            # eeg_data = np.clip(eeg_data, -4096, 4096)
            # eeg_data = np.log1p(np.abs(eeg_data)) * np.sign(eeg_data) / 4
            # eeg_data = eeg_data.astype(np.float32)
            np.save(Path(data_dir) / f"{eeg_id}.npy", eeg_data)


if __name__ == "__main__":
    df = pl.read_csv(Path("data") / "train.csv")
    convert_parquet_to_npy(
        df.with_columns(
            (
                pl.col("eeg_id").cast(pl.String)
                + "_"
                + pl.col("eeg_sub_id").cast(pl.String)
            ).alias("eeg_id")
        ),
        Path("data") / "train_eegs",
    )
