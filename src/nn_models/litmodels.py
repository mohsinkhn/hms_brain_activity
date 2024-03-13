from typing import Any, Dict, Tuple

import lightning as L
import pandas as pd
from timm.optim import AdamW, Nadam, MADGRAD
import torch
from torch import nn
import torch.nn.functional as F

from src.kaggle_metric import score
from src.settings import TARGET_COLS


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
        x, y = batch
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
        self.validation_step_outputs.append({"preds": preds, "y": y})
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat([x["preds"] for x in self.validation_step_outputs], dim=0)
        y = torch.cat([x["y"] for x in self.validation_step_outputs], dim=0)
        preds = pd.DataFrame(preds.cpu().numpy(), columns=TARGET_COLS)
        preds["id"] = ["id_" + str(x) for x in range(len(preds))]

        y = pd.DataFrame(y.cpu().numpy(), columns=TARGET_COLS)
        y["id"] = ["id_" + str(x) for x in range(len(y))]
        self.log(
            "val/score",
            score(y, preds, row_id_column_name="id"),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
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
