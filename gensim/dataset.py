#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable, Union, Any, Tuple, Dict

# External modules
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
import xarray as xr
import tensorstore as ts


# Internal modules
from . import utils

main_logger = logging.getLogger(__name__)


__all__ = [
    "NeXtSIMDataset",
]


class NeXtSIMDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            aux_path: str,
            base_resolution: float = 12.5,
            delta_t: int = 2,
            n_cycles: int = 2,
            state_variables: Iterable[str] = utils.STATE_VARIABLES_DEFAULT,
            forcing_variables: Iterable[str] = utils.FORCING_VARIABLES_DEFAULT,
            threads_limit: Union[str, int] = "shared",
    ):
        super().__init__()
        self._dataset = None
        self.data_path = data_path
        self.use_forcing = bool(forcing_variables)
        self.delta_t = delta_t
        self.n_cycles = n_cycles
        self.threads_limit = threads_limit
        self.base_resolution = base_resolution

        self._load_aux_data(aux_path)
        self._set_metadata(data_path)
        self._setup_indices(state_variables, forcing_variables)

    @property
    def dataset(self) -> Any:
        if self._dataset is None:
            self._dataset = self._setup_dataset()
        return self._dataset

    def _set_metadata(
            self,
            data_path: str
    ) -> None:
        zarr_ds = zarr.open(data_path, mode="r")
        self.avail_variables = zarr_ds["var_names"][:]
        self.dataset_len = zarr_ds["datacube"].shape[0]

    def _setup_indices(
            self,
            state_variables: Iterable[str],
            forcing_variables: Iterable[str]
    ):
        self.state_idx = np.array(utils.identify_var_idx(
            self.avail_variables, state_variables
        ))
        self.forcing_idx = np.array(utils.identify_var_idx(
            self.avail_variables, forcing_variables
        ))
        self.degree_days_idx = np.array(utils.identify_var_idx(
            self.avail_variables,
            ["pdd_month", "fdd_month", "pdd_year", "fdd_year"]
        ))

    def _load_aux_data(self, aux_path: str):
        with xr.open_dataset(aux_path) as ds_aux:
            mesh = ds_aux[["x_coord", "y_coord"]].to_array(dim="coord_dim")
            # Conversion into km
            self.mesh = mesh.values / 1_000
            self.mask = ds_aux["mask"].values

    def _setup_dataset(self) -> Any:
        dataset = ts.open(
            {
                'driver': 'zarr',
                'metadata_key': '.zarray',
                'kvstore': {
                    'driver': 'file',
                    'path': f'{self.data_path:s}/datacube',
                    'context': {
                        'file_io_concurrency': {'limit': self.threads_limit},
                    }
                },
                'context': {
                    'data_copy_concurrency': {'limit': self.threads_limit}
                }
            }
        ).result()
        return dataset

    def _to_time_index(self, idx: int) -> np.ndarray:
        return np.arange(
            0, self.delta_t * self.n_cycles, self.delta_t
        ) + idx

    def _unmask_values(self, array: np.ndarray) -> np.ndarray:
        unmasked = np.zeros(
            (*array.shape[:-1], *self.mask.shape), dtype=array.dtype
        )
        unmasked[..., self.bool_mask] = array
        return unmasked

    def _read_states_forcings(
        self,
        idx: int,
    ) -> Tuple[Any, Any]:
        trajectory = self.dataset[self._to_time_index(idx)].read().result()
        trajectory = np.nan_to_num(trajectory)
        states = trajectory[:, self.state_idx]
        forcings = trajectory[:, self.forcing_idx] \
            if self.use_forcing else np.zeros_like(states[:, :1])
        degree_days = trajectory[0, self.degree_days_idx]
        return states, forcings, degree_days

    def __len__(self) -> int:
        return self.dataset_len - self.delta_t * self.n_cycles + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        states, forcings, degree_days = self._read_states_forcings(idx)

        sample = {
            "states": states,
            "forcings": forcings,
            "degree_days": degree_days,
            "mesh": self.mesh,
            "mask": self.mask[None],
            "resolution": np.array([self.base_resolution])
        }
        return sample
