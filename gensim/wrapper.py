#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple

# External modules
import torch
import torch.nn.functional as F
from einops import rearrange

# Internal modules

main_logger = logging.getLogger(__name__)


class PaddedUnfoldLayer(torch.nn.Module):
    def __init__(
        self,
        patch_size: Tuple[int, int] = (64, 64),
        overlap_size: Tuple[int, int] = (8, 8),
        padding_mode: str = "replicate"
    ):
        super().__init__()
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.padding_mode = padding_mode
        self.kernel_size = (
            patch_size[0]+overlap_size[0]*2, patch_size[1]+overlap_size[1]*2
        )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        padded_tensor = F.pad(
            in_tensor,
            (
                self.overlap_size[1], self.overlap_size[1],
                self.overlap_size[0], self.overlap_size[0],
            ),
            mode=self.padding_mode
        )
        unfolded_tensor = F.unfold(
            padded_tensor, kernel_size=self.kernel_size, stride=self.patch_size
        )
        rearranged_tensor = rearrange(
            unfolded_tensor, "b (c h w) k -> (b k) c h w",
            h=self.kernel_size[0], w=self.kernel_size[1]
        )
        return rearranged_tensor


class PatchedNetwork(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Module,
            patch_size: Tuple[int, int] = (64, 64),
            overlap_size: Tuple[int, int] = (8, 8),
            padding_mode: str = "replicate"
    ) -> None:
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.padding_mode = padding_mode
        self.kernel_size = (
            patch_size[0]+overlap_size[0]*2, patch_size[1]+overlap_size[1]*2
        )
        self.inner_slices = (
            slice(overlap_size[0], -overlap_size[0]),
            slice(overlap_size[1], -overlap_size[1]),
        )
        self.unfold_layer = PaddedUnfoldLayer(
            patch_size=patch_size,
            overlap_size=overlap_size,
            padding_mode=padding_mode
        )

    def fold_tensor(
            self,
            in_tensor: torch.Tensor,
            output_shape: Tuple[int, int]
    ) -> torch.Tensor:
        sliced_tensor = in_tensor[
            ..., self.inner_slices[0], self.inner_slices[1]
        ]
        folded_tensor = rearrange(
            sliced_tensor, "(b hp wp) c h w -> b c (hp h) (wp w)",
            hp=output_shape[0]//self.patch_size[0],
            wp=output_shape[1]//self.patch_size[1]
        )
        return folded_tensor

    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            labels: torch.Tensor,
            pseudo_time: torch.Tensor,
            resolution: torch.Tensor
    ):
        b, _, h, w = in_tensor.shape
        unfolded_tensor = self.unfold_layer(in_tensor)
        n_patches = unfolded_tensor.size(0) // b
        mesh = self.unfold_layer(mesh)
        mask = self.unfold_layer(mask)
        output = self.model(
            unfolded_tensor,
            mesh=mesh,
            mask=mask,
            labels=labels.repeat(n_patches, 1),
            pseudo_time=pseudo_time.repeat(n_patches, 1),
            resolution=resolution.repeat(n_patches, 1)
        )
        prediction = self.fold_tensor(output, (h, w))
        return prediction
