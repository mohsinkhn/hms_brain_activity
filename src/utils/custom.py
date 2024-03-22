import numpy as np
import polars as pl
import torch

from src.settings import TARGET_COLS
from src.kaggle_metric import score


def val_to_dataframe(data, means):
    # data["preds"] = correct_means(data["preds"], np.array(means))
    # data["preds"] = norm_preds(data["preds"])
    for i, col in enumerate(TARGET_COLS):
        data[f"{col}_pred"] = data["preds"][:, i]
        data[f"{col}_true"] = data["y"][:, i]
    del data["preds"]
    del data["y"]
    return pl.DataFrame(data)


def test_to_dataframe(data, means):
    # data["preds"] = correct_means(data["preds"], np.array(means))
    for i, col in enumerate(TARGET_COLS):
        data[f"{col}"] = data["preds"][:, i]
    del data["preds"]
    return pl.DataFrame(data)


def get_comp_score(data: pl.DataFrame, group_col="eeg_id"):
    data = (
        data.group_by(group_col)
        .agg(
            [
                *[
                    pl.col(f"{col}_pred").sum().clip(1e-8, 100).alias(f"{col}_pred")
                    for col in TARGET_COLS
                ],
                *[
                    pl.col(f"{col}_true").sum().alias(f"{col}_true")
                    for col in TARGET_COLS
                ],
            ]
        )
        .with_columns(
            pl.sum_horizontal(*[f"{col}_pred" for col in TARGET_COLS]).alias(
                "sum_votes"
            ),
            pl.sum_horizontal(*[f"{col}_true" for col in TARGET_COLS]).alias(
                "sum_votes_true"
            ),
        )
        .with_columns(
            [
                *[
                    (pl.col(f"{col}_pred") / pl.col("sum_votes")).alias(f"{col}_pred")
                    for col in TARGET_COLS
                ],
                *[
                    (pl.col(f"{col}_true") / pl.col("sum_votes_true")).alias(
                        f"{col}_true"
                    )
                    for col in TARGET_COLS
                ],
            ]
        )
    )
    pred_data = data.select(
        pl.col(group_col),
        *[pl.col(f"{col}_pred").alias(col) for col in TARGET_COLS],
    ).to_pandas()
    true_data = data.select(
        pl.col(group_col),
        *[pl.col(f"{col}_true").alias(col) for col in TARGET_COLS],
    ).to_pandas()
    return score(solution=true_data, submission=pred_data, row_id_column_name=group_col)


def correct_means(y, target_means):
    yhat = np.log(y / (1 - y))
    target_means = np.log(target_means / (1 - target_means))
    target_means = target_means.reshape(1, -1)
    yhat = yhat / yhat.mean(keepdims=True, axis=0) * target_means
    yhat = 1 / (1 + np.exp(-yhat))
    return yhat


def norm_preds(preds):
    norm_preds = preds / preds.sum(axis=1, keepdims=True)
    return norm_preds
