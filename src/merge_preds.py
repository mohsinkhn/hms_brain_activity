import argparse
import pandas as pd
import polars as pl
from pathlib import Path

from src.settings import TARGET_COLS
from src.utils.custom import get_comp_score


if __name__ == "__main__":
    df1 = pd.read_csv("logs/train/val_outputs/effb1_clean/val_preds.csv")
    df2 = pd.read_csv("logs/train/val_outputs/tfm_clean/val_preds.csv")
    print(get_comp_score(pl.DataFrame(df1).filter(pl.col("total_votes") > 3)))
    print(get_comp_score(pl.DataFrame(df2).filter(pl.col("total_votes") > 3)))
    df = df1.copy()
    for target in TARGET_COLS:
        df[f"{target}_pred"] = (df1[f"{target}_pred"] + df2[f"{target}_pred"]) / 2
    df.to_csv("logs/train/val_outputs/ensemble/val_preds_clean.csv", index=False)
    print(get_comp_score(pl.DataFrame(df).filter(pl.col("total_votes") > 3)))
