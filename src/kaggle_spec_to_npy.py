from pathlib import Path

import argparse
import polars as pl
import numpy as np
from tqdm.auto import tqdm


def kaggle_spec_to_npy(filepaths, output_folder):
    for filepath in tqdm(filepaths):
        df = pl.read_parquet(filepath)
        data = []
        for ctype in ["LL", "RL", "LP", "RP"]:
            cols = ["time"] + [c for c in df.columns if ctype in c]
            np_data = df.select(cols).to_numpy()
            data.append(np_data)
        data = np.stack(data, axis=-1)
        output_path = Path(output_folder) / Path(filepath).name
        np.save(str(output_path).replace(".csv", ".npy"), data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default="./data/train_spectrograms")
    parser.add_argument("--output_folder", default="./data/train_spectrograms")
    args = parser.parse_args()
    filepaths = list(Path(args.input_folder).glob("*.parquet"))
    kaggle_spec_to_npy(filepaths, args.output_folder)
