from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
import numpy as np
import polars as pl
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.nn_datasets.eegdataset import HMSTrainEEGData, HMSValEEGData, HMSTrainEEGDataV2
from src.convert_parquet_to_npy import convert_parquet_to_npy


class EEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        num_folds: int = 5,
        fold_id: int = 0,
        transforms: Any = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 6

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        if stage == "fit" or stage is None:
            self.data_train, self.data_val = self._prepare_data()

        if stage == "test" or stage is None:
            self.data_test = self._prepare_test_data()

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {"fold_id": self.hparams.fold_id, "num_folds": self.hparams.num_folds}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def _prepare_data(self) -> Tuple[Dataset, Dataset]:
        """Prepare the training and validation data.

        :return: A tuple with the training and validation datasets.
        """
        df = pl.read_csv(Path(self.hparams.data_dir) / "train.csv")
        # convert_parquet_to_npy(
        #     df.with_columns(
        #         (
        #             pl.col("eeg_id").cast(pl.String)
        #             + "_"
        #             + pl.col("eeg_sub_id").cast(pl.String)
        #         ).alias("eeg_id")
        #     ),
        #     Path(self.hparams.data_dir) / "train_eegs",
        # )
        patient_ids = np.array(df["patient_id"].unique().to_list())
        kf = KFold(n_splits=self.hparams.num_folds, shuffle=True, random_state=786)
        train_idx, val_idx = list(kf.split(patient_ids))[self.hparams.fold_id]
        train_df = df.filter(pl.col("patient_id").is_in(patient_ids[train_idx]))
        val_df = df.filter(pl.col("patient_id").is_in(patient_ids[val_idx]))

        train_dataset = HMSTrainEEGDataV2(
            train_df,
            Path(self.hparams.data_dir) / "train_eegs",
            transforms=self.transforms,
        )
        val_dataset = HMSValEEGData(
            val_df, Path(self.hparams.data_dir) / "train_eegs", transforms=None
        )

        return train_dataset, val_dataset

    def _prepare_test_data(self) -> Dataset:
        """Prepare the test data.

        :return: The test dataset.
        """
        df = pl.read_csv(Path(self.hparams.data_dir) / "test.csv")
        return HMSValEEGData(df, Path(self.hparams.data_dir) / "test_eegs")
