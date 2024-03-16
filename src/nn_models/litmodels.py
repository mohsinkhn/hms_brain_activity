from pathlib import Path
from typing import Any, Dict, Tuple

import lightning as L
import numpy as np
import polars as pl
from timm.optim import AdamW, Nadam, MADGRAD
import torch
from torch import nn
import torch.nn.functional as F
import wandb

from src.kaggle_metric import score
from src.settings import TARGET_COLS
from src.nn_datasets.eegdataset import load_eeg_data
from src.utils.plot_batches import plot_zoomed_batch
from src.utils.custom import get_comp_score, val_to_dataframe


class KLDivLossWithLogits(nn.KLDivLoss):
    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, y, t, sample_weight=None):
        y = nn.functional.log_softmax(y, dim=1)
        loss = super().forward(y, t)
        if sample_weight is not None:
            sample_weight = sample_weight / sample_weight.sum()
            loss = (loss.sum(dim=1) * sample_weight).sum()
        else:
            loss = loss.sum() / y.size(0)

        return loss


class LitModel(L.LightningModule):
    """PL Model"""

    def __init__(
        self,
        net: nn.Module,
        optimizer: Any,
        scheduler: torch.optim.lr_scheduler,
        scheduler_interval: str,
        compile: bool,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = net
        self.validation_step_outputs = []
        self.criterion = KLDivLossWithLogits()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch["data"], batch["targets"]
        logits = self.forward(x)
        loss = self.criterion(logits, y, sample_weight=batch.get("sample_weight", None))
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.step(batch)
        self.post_process_validation_step(loss, preds, y, batch, batch_idx)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        data = {}
        for batch_data in self.validation_step_outputs:
            for k, v in batch_data.items():
                if k not in data:
                    data[k] = []
                data[k].append(v.cpu())
        for k, v in data.items():
            data[k] = torch.cat(v, dim=0).numpy()
        self.post_process_validation_epoch_end(data)
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        _, preds, _ = self.step(batch)
        return preds

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # if "conv2d" not in self.model.__class__.__name__.lower():
        #     optimizer = self.hparams.optimizer(params=self.model.parameters())
        # else:
        conv2d_params = [
            kv[1] for kv in self.model.named_parameters() if ".conv2d" in kv[0]
        ]
        other_params = [
            kv[1] for kv in self.model.named_parameters() if ".conv2d" not in kv[0]
        ]
        optimizer = self.hparams.optimizer(
            [
                {
                    "params": conv2d_params,
                    "lr": self.hparams.optimizer.keywords["lr"] * 0.1,
                    "weight_decay": self.hparams.optimizer.keywords["weight_decay"]
                    * 0.1,
                },
                {
                    "params": other_params,
                },
            ]
        )
        # optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": self.hparams.scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def post_process_validation_step(
        self, loss, preds, y, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        preds = F.softmax(preds, dim=1)
        self.validation_step_outputs.append(
            {
                "preds": preds,
                "y": y,
                "eeg_id": batch["eeg_id"],
                "eeg_sub_id": batch["eeg_sub_id"],
                "num_votes": batch["num_votes"],
            }
        )

    def post_process_validation_epoch_end(self, data) -> None:
        data_df = val_to_dataframe(data)
        val_score = get_comp_score(data_df)
        self.log("val/score", val_score, on_step=False, on_epoch=True, prog_bar=False)

        val_score2 = get_comp_score(data_df.filter(pl.col("num_votes") > 7))
        self.log("val/score2", val_score2, on_step=False, on_epoch=True, prog_bar=False)

        # table of means
        means = data_df.select(
            *[pl.mean(f"{col}_pred").alias(f"{col}_pred") for col in TARGET_COLS],
            *[pl.mean(f"{col}_true").alias(f"{col}_true") for col in TARGET_COLS],
        ).to_pandas()
        wandb.log({"means preds/y": wandb.Table(dataframe=means)})

        # Plot confusion matrix
        preds = data_df.select(
            *[pl.col(f"{col}_pred") for col in TARGET_COLS]
        ).to_numpy()
        y = data_df.select(*[pl.col(f"{col}_true") for col in TARGET_COLS]).to_numpy()
        preds_label = np.argmax(preds, axis=1)
        y_label = np.argmax(y, axis=1)
        cm = wandb.plot.confusion_matrix(
            y_true=y_label, preds=preds_label, class_names=TARGET_COLS
        )
        wandb.log({"val/cm": cm})

        # error plots
        data_df = (
            data_df.with_columns(
                pl.reduce(
                    lambda x, y: x + y,
                    [
                        (pl.col(f"{col}_true") - pl.col(f"{col}_pred")).abs()
                        for col in TARGET_COLS
                    ],
                ).alias("error_sum")
            )
            .sort("error_sum", descending=True)
            .head(50)
            .sample(8)
            .to_pandas()
        )
        batch_x, batch_y, preds = [], [], []
        for i, row in data_df.iterrows():
            np_data = load_eeg_data(
                Path("./data/train_eegs"),
                int(row["eeg_id"]),
                int(row["eeg_sub_id"]),
            )
            batch_x.append(np_data)
            batch_y.append([row[f"{col}_true"] for col in TARGET_COLS])
            preds.append([row[f"{col}_pred"] for col in TARGET_COLS])
        batch_x = np.stack(batch_x)
        batch_y = np.array(batch_y)
        preds = np.array(preds)
        plot_zoomed_batch(
            batch_x,
            batch_y,
            preds=preds,
            start=4000,
            n=2000,
            save_path=f"./val_errors/tmp.jpg",
        )
        wandb.log({"val/errors": wandb.Image(f"./val_errors/tmp.jpg")})
