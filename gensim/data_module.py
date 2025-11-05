#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable, Union, Dict
import os.path

# External modules
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
import numpy as np


# Internal modules
from .dataset import NeXtSIMDataset
from .utils import unmask_tensor


main_logger = logging.getLogger(__name__)


__all__ = [
    "SurrogateDataModule"
]


def collate_fn(batch: Dict[str, torch.Tensor]):
    """
    Custom collate fn that converts into float tensor and stack along first
    dimension.
    """
    keys = batch[0].keys()
    return {
        key: torch.as_tensor(
            np.stack([d[key] for d in batch], axis=0),
            dtype=torch.float32
        )
        for key in keys
    }


class SurrogateDataModule(pl.LightningDataModule):
    _UNMASK_KEYS = ["states", "forcings", "degree_days"]

    def __init__(
            self,
            data_path: str,
            aux_path: str,
            state_variables: Iterable[str] = ("sit", ),
            forcing_variables: Iterable[str] = ("t2m", "d2m", "u10m", "v10m"),
            delta_t: int = 2,
            n_input_steps: int = 1,
            n_rollout_steps: int = 1,
            base_resolution: float = 12.5,
            batch_size: int = 64,
            val_batch_size: int = 16,
            n_train_samples: int = None,
            n_workers: int = 4,
            pin_memory: bool = True,
            zip_path: str = None,
            fast: bool = True,
            suffix: str = "",
            threads_limit: Union[int, str] = "shared",
    ):
        super().__init__()
        self._train_dataset = None
        self._val_dataset = None
        self._predict_dataset = None
        self._test_dataset = None
        self.delta_t = delta_t
        self.n_input_steps = n_input_steps
        self.n_rollout_steps = n_rollout_steps
        self.base_resolution = base_resolution
        self.data_path = data_path
        self.aux_path = aux_path
        self.forcing_variables = forcing_variables
        self.state_variables = state_variables
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.n_train_samples = n_train_samples
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.zip_path = zip_path
        self.fast = fast
        self.suffix = suffix
        self.threads_limit = threads_limit

    def setup(self, stage: str) -> None:
        if stage == "fit":
            file_name = f"train{self.suffix}.zarr"
            train_dataset = NeXtSIMDataset(
                os.path.join(self.data_path, file_name),
                aux_path=self.aux_path,
                base_resolution=self.base_resolution,
                delta_t=self.delta_t,
                n_cycles=self.n_input_steps+self.n_rollout_steps,
                state_variables=self.state_variables,
                forcing_variables=self.forcing_variables,
                threads_limit=self.threads_limit,
            )
            if self.n_train_samples is not None:
                train_dataset, _ = random_split(
                    train_dataset, (
                        self.n_train_samples,
                        len(train_dataset)-self.n_train_samples
                    )
                )
            self._train_dataset = train_dataset
        if stage in ("fit", "validate"):
            file_name = f"validation{self.suffix}.zarr"
            self._val_dataset = NeXtSIMDataset(
                os.path.join(self.data_path, file_name),
                aux_path=self.aux_path,
                base_resolution=self.base_resolution,
                delta_t=self.delta_t,
                n_cycles=self.n_input_steps+self.n_rollout_steps,
                state_variables=self.state_variables,
                forcing_variables=self.forcing_variables,
                threads_limit=self.threads_limit
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset, batch_size=self.batch_size,
            shuffle=True, pin_memory=self.pin_memory,
            num_workers=self.n_workers, persistent_workers=self.n_workers > 0,
            drop_last=True, collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=self.pin_memory,
            num_workers=self.n_workers, persistent_workers=self.n_workers > 0,
            collate_fn=collate_fn
        )

    def on_after_batch_transfer(
            self,
            batch: Dict[str, torch.Tensor],
            dataloader_idx
    ) -> Dict[str, torch.Tensor]:
        for k in self._UNMASK_KEYS:
            batch[k] = unmask_tensor(batch[k], batch["mask"])
        return batch
