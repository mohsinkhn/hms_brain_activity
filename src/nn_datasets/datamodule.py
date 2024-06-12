from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np
import os
import segyio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.nn_datasets.components.gprdataset import GPRDataset
from src.utils.helper_functions import collate_fn



class GPRDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        window_size: int = 1000,
        stride: int = 800,
        pin_memory: bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = GPRDataset(self.hparams.data_dir, self.hparams.window_size, self.hparams.stride)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.data_train, self.data_val = random_split(dataset, [train_size, val_size])
        
        self.data_test = self.data_val

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )
