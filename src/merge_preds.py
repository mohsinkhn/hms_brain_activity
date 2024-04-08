import argparse
import pandas as pd
import polars as pl
from pathlib import Path

from src.settings import TARGET_COLS
from src.utils.custom import get_comp_score


if __name__ == "__main__":
    df1 = pd.read_csv("logs/train/val_outputs/effb1_clean/val_preds.csv")
    df1 = df1.sort_values(by=["eeg_id", "eeg_label_offset_seconds"]).reset_index(
        drop=True
    )
    df2 = pd.read_csv("logs/train/val_outputs/tfm_clean/val_preds.csv")
    df2 = df2.sort_values(by=["eeg_id", "eeg_label_offset_seconds"]).reset_index(
        drop=True
    )
    df3 = pd.read_csv("logs/train/val_outputs/tfm_clean_seed2/val_preds.csv")
    df3 = df3.sort_values(by=["eeg_id", "eeg_label_offset_seconds"]).reset_index(
        drop=True
    )
    print(get_comp_score(pl.DataFrame(df1).filter(pl.col("total_votes") > 3)))
    print(get_comp_score(pl.DataFrame(df2).filter(pl.col("total_votes") > 3)))
    print(get_comp_score(pl.DataFrame(df3).filter(pl.col("total_votes") > 3)))

    df = df1.copy()
    for target in TARGET_COLS:
        df[f"{target}_pred"] = (
            df1[f"{target}_pred"] + df2[f"{target}_pred"] + df3[f"{target}_pred"]
        ) / 3
    df.to_csv("logs/train/val_outputs/ensemble/val_preds_clean_v2.csv", index=False)
    print(get_comp_score(pl.DataFrame(df).filter(pl.col("total_votes") > 3)))
    print(get_comp_score(pl.DataFrame(df).filter(pl.col("total_votes") >= 10)))
