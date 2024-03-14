from pathlib import Path
from typing import Any, Dict, Tuple

import lightning as L
import numpy as np
import pandas as pd
from timm.optim import AdamW, Nadam, MADGRAD
import torch
from torch import nn
import torch.nn.functional as F
import wandb

from src.kaggle_metric import score
from src.settings import TARGET_COLS
from src.nn_datasets.eegdataset import load_eeg_data
from src.utils.plot_batches import plot_batch


class KLDivLossWithLogits(nn.KLDivLoss):
    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        loss = super().forward(y, t)

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
        # y = F.softmax(y, dim=1)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        # preds = torch.argmax(logits, dim=1)
        return loss, F.softmax(logits, dim=1), y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.step(batch)
        self.validation_step_outputs.append(
            {
                "preds": preds,
                "y": y,
                "eeg_id": batch["eeg_id"],
                "eeg_sub_id": batch["eeg_sub_id"],
            }
        )
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat([x["preds"] for x in self.validation_step_outputs], dim=0)
        y = torch.cat([x["y"] for x in self.validation_step_outputs], dim=0)
        eeg_ids = torch.cat([x["eeg_id"] for x in self.validation_step_outputs], dim=0)
        eeg_sub_ids = torch.cat(
            [x["eeg_sub_id"] for x in self.validation_step_outputs], dim=0
        )
        preds = pd.DataFrame(preds.cpu().numpy(), columns=TARGET_COLS)
        preds["eeg_id"] = eeg_ids.cpu().numpy()
        preds_agg = preds.groupby("eeg_id")[TARGET_COLS].sum().reset_index()
        preds_agg[TARGET_COLS] = preds_agg[TARGET_COLS] / preds_agg[TARGET_COLS].sum(
            axis=1
        ).values.reshape(-1, 1)
        y = pd.DataFrame(y.cpu().numpy(), columns=TARGET_COLS)
        y["eeg_id"] = eeg_ids.cpu().numpy()
        y_agg = y.groupby("eeg_id")[TARGET_COLS].sum().reset_index()
        y_agg[TARGET_COLS] = y_agg[TARGET_COLS] / y_agg[TARGET_COLS].sum(
            axis=1
        ).values.reshape(-1, 1)
        self.log(
            "val/score",
            score(y_agg, preds_agg, row_id_column_name="eeg_id"),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.validation_step_outputs = []

        # Log averages of predictions
        avg_preds = preds_agg[TARGET_COLS].mean().tolist()
        avg_y = y_agg[TARGET_COLS].mean().tolist()
        table = wandb.Table(data=[avg_preds, avg_y], columns=TARGET_COLS)
        wandb.log({"means preds/y": table})

        # Plot confusion matrix
        preds_label = np.argmax(preds[TARGET_COLS].values, axis=1)
        y_label = np.argmax(y[TARGET_COLS].values, axis=1)
        cm = wandb.plot.confusion_matrix(
            y_true=y_label, preds=preds_label, class_names=TARGET_COLS
        )
        wandb.log({"val/cm": cm})

        # Plot some of the batches with errors
        error_idx = np.where(preds_label != y_label)[0]
        if len(error_idx) > 0:
            error_idx = np.random.choice(error_idx, size=min(8, len(error_idx)))
            batch_x = []
            batch_y = y[TARGET_COLS].values[error_idx]
            for idx in error_idx:
                np_data = load_eeg_data(
                    Path("./data/train_eegs"), eeg_ids[idx], eeg_sub_ids[idx]
                )[8:-8]
                batch_x.append(np_data)
            batch_x = np.stack(batch_x)
            Path("./val_errors").mkdir(exist_ok=True, parents=True)
            plot_batch(
                batch_x,
                batch_y,
                preds=preds[TARGET_COLS].values[error_idx],
                save_path="./val_errors/{eeg_ids[idx]}_{eeg_sub_ids[idx]}.jpg",
            )

            wandb.log(
                {
                    f"val/errors": wandb.Image(
                        "./val_errors/{eeg_ids[idx]}_{eeg_sub_ids[idx]}.jpg",
                    )
                }
            )

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
        #     conv2d_parameters = self.model.conv2d.parameters()
        #     other_parameters = [
        #         p for p in self.model.model.parameters() if p not in conv2d_parameters
        #     ]
        #     optimizer = self.hparams.optimizer(
        #         [
        #             {
        #                 "params": conv2d_parameters,
        #                 "lr": self.optimizer.lr * 0.1,
        #                 "weight_decay": self.optimizer.weight_decay * 0.1,
        #             },
        #             {
        #                 "params": other_parameters,
        #                 "lr": self.optimizer.lr * 1.0,
        #                 "weight_decay": self.optimizer.weight_decay * 1.0,
        #             },
        #         ]
        #     )
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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

    # def configure_optimizers(self):
    #     if self.config.opti.name == "madgrad":
    #         optimizer = MADGRAD(
    #             self.parameters(),
    #             lr=self.config.opti.lr,
    #             weight_decay=self.config.opti.wd,
    #         )
    #     else:
    #         optimizer = Nadam(
    #             self.parameters(),
    #             lr=self.config.opti.lr,
    #             weight_decay=self.config.opti.wd,
    #         )

    #     if self.config.scheduler.name == "onecycle":
    #         scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #             optimizer,
    #             total_steps=self.trainer.estimated_stepping_batches,
    #             max_lr=self.config.opti.lr,
    #             div_factor=self.config.scheduler.get("div_factor", 10),
    #             pct_start=self.config.scheduler.get("pct_start", 0),
    #             final_div_factor=self.config.scheduler.get("final_div_factor", 100),
    #         )
    #         interval = "step"

    #     else:
    #         scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #             optimizer,
    #             gamma=self.config.scheduler.gamma,
    #             milestones=self.config.scheduler.steps,
    #         )
    #         interval = "epoch"
    #     return [optimizer], [{"scheduler": scheduler, "interval": interval}]
