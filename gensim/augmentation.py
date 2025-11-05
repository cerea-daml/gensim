#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple, Dict
import abc

# External modules
import torch
import torch.nn.functional as F

import netCDF4
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.signal import correlate

# Internal modules
from .utils import batch_to_tensor, tensor_to_batch


main_logger = logging.getLogger(__name__)


__all__ = [
    "VFlipTransformation",
    "HFlipTransformation",
    "RotateTransformation",
    "PatchGenerator"
]


class DataTransformation(torch.nn.Module):
    def __init__(self, probability: float = 0.) -> None:
        super().__init__()
        self.probability = probability

    @abc.abstractmethod
    def _transform_func(self, field: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def forward(
            self,
            fields: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.rand(
            fields.size(0),
            device=fields.device,
            dtype=fields.dtype
        )
        labels = labels < self.probability
        fields = torch.where(
            labels.view(
                [labels.size(0)] + [1] * (fields.ndim-1)
            ).expand_as(fields),
            self._transform_func(fields),
            fields
        )
        return fields, labels


class VFlipTransformation(DataTransformation):
    def _transform_func(self, field):
        return torch.flip(field, dims=(-2,))


class HFlipTransformation(DataTransformation):
    def _transform_func(self, field):
        return torch.flip(field, dims=(-1,))


class RotateTransformation(DataTransformation):
    def _transform_func(self, field):
        return torch.flip(field, dims=(-2,)).swapdims(-2, -1)


class DataAugmentation(torch.nn.Module):
    def __init__(self, **transformations: DataTransformation) -> None:
        super().__init__()
        self.transformations = torch.nn.ModuleList(
            list(transformations.values())
        )

    @torch.no_grad()
    def forward(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        in_tensor = batch_to_tensor(batch)
        labels = []
        for transform in self.transformations:
            out_tensor, label = transform(in_tensor)
            labels.append(label)
        batch = tensor_to_batch(out_tensor, batch)
        labels = torch.stack(labels, dim=1).to(dtype=batch["states"].dtype)
        return batch, labels


def extract_patches(
    in_tensor: torch.Tensor,
    sampled_top: torch.Tensor,
    sampled_left: torch.Tensor,
    sampled_levels: torch.Tensor,
    size_y: int,
    size_x: int,
    pad_y: int,
    pad_x: int,
    n_patches: int
) -> torch.Tensor:
    nb, nc, _, _ = in_tensor.shape
    in_tensor = F.pad(
        in_tensor, (pad_y, pad_y, pad_x, pad_x), mode="replicate"
    )
    out_tensor = torch.empty(
        nb*n_patches, nc, size_y, size_x,
        device=in_tensor.device, dtype=in_tensor.dtype
    )
    for k, (start_top, start_left, level) in enumerate(
            zip(sampled_top, sampled_left, sampled_levels)
    ):
        end_top = start_top + size_y * level
        end_left = start_left + size_x * level
        curr_patch = in_tensor[
            k % nb, :, start_top:end_top, start_left:end_left
        ]
        if level > 1:
            curr_patch = curr_patch.view(
                nc, size_y, level, size_x, level
            ).mean(dim=(-3, -1))
        out_tensor[k] = curr_patch
    return out_tensor


class PatchGenerator(torch.nn.Module):
    def __init__(
            self,
            mask_path: str = 'data/auxiliary/ds_aux.nc',
            n_patches: int = 8,
            coarse_levels: Tuple[int, ...] = (1, ),
            coarse_probs: Tuple[float, ...] = (1., ),
            patch_size: Tuple[int, int] = (64, 64),
            overlap_size: Tuple[int, int] = (8, 8),
            valid_threshold: float = 0.1,
            train_with_overlap: bool = True
    ):
        super().__init__()

        self.n_patches = n_patches
        self.overlap_size = overlap_size
        self.valid_threshold = valid_threshold
        self.train_with_overlap = train_with_overlap

        self.max_level = max(coarse_levels)
        self.skip_average = self.max_level == 1
        self.patch_size = (
            patch_size[0] + 2 * overlap_size[0],
            patch_size[1] + 2 * overlap_size[1],
        ) if train_with_overlap else patch_size
        self.pad_size = (
            self.overlap_size[0] * self.max_level,
            self.overlap_size[1] * self.max_level,
        )

        self.register_buffer(
            "coarse_levels",
            torch.as_tensor(coarse_levels, dtype=torch.long)
        )
        coarse_probs = torch.as_tensor(coarse_probs, dtype=torch.float32)
        coarse_probs /= coarse_probs.sum()
        self.register_buffer("coarse_probs", coarse_probs)

        valid_masks = self._estimate_valid_masks(
            mask_path, coarse_levels, valid_threshold
        )
        self.register_buffer("valid_masks", valid_masks)

    def state_dict(self, *args, **kwargs) -> None:
        return None

    def _read_mask(self, mask_path: str) -> np.ndarray:
        with netCDF4.Dataset(mask_path) as ds:
            mask = ds.variables["mask"][:]
        padded_mask = np.pad(
            mask,
            (
                (self.pad_size[0], self.pad_size[0]),
                (self.pad_size[1], self.pad_size[1])
            ),
            mode="edge"
        )
        return padded_mask

    def _estimate_valid_mean(self, mask: np.ndarray, level: int) -> np.ndarray:
        # Get the minimum in the coarse graining level.
        mask_minimum = minimum_filter(mask, size=(level, level))
        border_min = level // 2
        mask_minimum = mask_minimum[
            border_min:mask_minimum.shape[0]-border_min+1,
            border_min:mask_minimum.shape[1]-border_min+1
        ]
        # Calculate the mean of the mask in the coarse graining level.
        kernel = np.zeros((
            (self.patch_size[0] - 1) * level + 1,
            (self.patch_size[1] - 1) * level + 1,
        ))
        kernel[::level, ::level] = 1.
        kernel /= kernel.sum()
        valid_mean = correlate(mask_minimum, kernel, mode="valid")
        ocean_mean = np.zeros_like(mask)
        ocean_mean[:valid_mean.shape[0], :valid_mean.shape[1]] = valid_mean
        return ocean_mean

    def _estimate_valid_masks(
            self,
            mask_path: str,
            coarse_levels: Tuple[int],
            valid_threshold: float = 0.1
    ) -> torch.Tensor:
        mask = self._read_mask(mask_path)
        valid_masks = [
            self._estimate_valid_mean(mask, level) > valid_threshold
            for level in coarse_levels
        ]
        valid_masks = torch.stack([
            torch.as_tensor(mask, dtype=torch.float32) for mask in valid_masks
        ], dim=0)
        return valid_masks

    def _sample_indices(
            self, sampled_level_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_mask = self.valid_masks[
            sampled_level_idx
        ].view(sampled_level_idx.size(0), -1)
        flat_mask = flat_mask.float()
        flat_mask = flat_mask / flat_mask.sum(dim=1, keepdims=True)
        sampled_indices = torch.multinomial(
            flat_mask, num_samples=1, replacement=True,
        )
        sampled_top = sampled_indices // self.valid_masks.size(-1)
        sampled_left = sampled_indices % self.valid_masks.size(-1)
        return sampled_top, sampled_left

    def _extract_patch(
            self, in_tensor: torch.Tensor, start_top: torch.Tensor,
            start_left: torch.Tensor, level: torch.Tensor
    ) -> torch.Tensor:
        end_top = start_top + self.patch_size[0]*level
        end_left = start_left + self.patch_size[1]*level
        return in_tensor[..., start_top:end_top, start_left:end_left]

    def _coarse_grain(
            self, in_tensor: torch.Tensor, level: torch.Tensor
    ) -> torch.Tensor:
        target_shape = (
            *in_tensor.shape[:-2], self.patch_size[0], level,
            self.patch_size[1], level
        )
        return in_tensor.view(target_shape).mean(dim=(-3, -1))

    @torch.no_grad()
    def forward(
            self,
            batch: Dict[str, torch.Tensor],
            resolution: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        nb = batch["states"].size(0)

        # Sampling
        sampled_level_idx = torch.multinomial(
            self.coarse_probs,
            num_samples=nb * self.n_patches,
            replacement=True
        )
        sampled_level_idx = sampled_level_idx.sort().values
        sampled_levels = self.coarse_levels[sampled_level_idx]
        sampled_top, sampled_left = self._sample_indices(sampled_level_idx)

        # Prepare tensors
        in_tensor = batch_to_tensor(batch)
        out_tensor = extract_patches(
            in_tensor, sampled_top, sampled_left, sampled_levels,
            self.patch_size[0], self.patch_size[1],
            self.pad_size[0], self.pad_size[1],
            self.n_patches
        )

        # Convert output back into dictionary
        batch = tensor_to_batch(out_tensor, batch)
        batch["mask"] = 1 - (batch["mask"] < 1).to(batch["mask"])
        resolution = (
            sampled_levels[:, None] * resolution.repeat(self.n_patches, 1)
        )
        return batch, resolution.to(batch["states"])
