import argparse
from pathlib import Path

import polars as pl
import numpy as np

from src.settings import TARGET_COLS
from src.utils.custom import get_comp_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    for i in [0, 1, 2, 3, 4]:
        data = pl.read_csv(Path(args.data_path) / f"{i}/val_preds.csv")
        # data = data.filter(pl.col("num_votes") > 7)
        score = get_comp_score(data.filter(pl.col("num_votes") > 7))
        print(f"Score {i}: {score}")
        if i == 0:
            final_data = data
        else:
            final_data = final_data.vstack(data)

    score = get_comp_score(final_data.filter(pl.col("num_votes") > 7))
    print(f"Final Score: {score}")
    final_data.write_csv(Path(args.data_path) / "val_preds.csv")
