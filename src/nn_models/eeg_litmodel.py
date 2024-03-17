from typing import Any

import torch
from torch import nn
import timm
from src.nn_models.litmodels import LitModel


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


class EEGModel(LitModel):
    def __init__(
        self,
        net: nn.Module,
        optimizer: Any,
        scheduler: torch.optim.lr_scheduler,
        scheduler_interval: str,
        differential_lr: bool,
        compile: bool,
    ):
        super().__init__(net, optimizer, scheduler, scheduler_interval, compile)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": self.scheduler_interval,
                "frequency": 1,
            },
        }
