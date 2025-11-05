#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable, Tuple
from math import inf

# External modules
import torch

# Internal modules
from .utils import mask_tensor


main_logger = logging.getLogger(__name__)


class GaussianEncoder(torch.nn.Module):
    def __init__(
            self,
            mean: Iterable[float] = (0., ),
            std: Iterable[float] = (1., ),
            eps: float = 1E-9
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean)[:, None, None])
        self.register_buffer("std", torch.tensor(std)[:, None, None])
        self.eps = eps

    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normed_tensor = (in_tensor-self.mean) / (self.std + self.eps)
        return normed_tensor * mask, mesh, mask
    

class GaussianDecoder(torch.nn.Module):
    def __init__(
            self,
            mean: Iterable[float] = (0., ),
            std: Iterable[float] = (1., ),
            lower_bound:  Iterable[float] = (-inf, ),
            upper_bound:  Iterable[float] = (inf, ),
            **kwargs
    ):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean)[:, None, None]
        )
        self.register_buffer(
            "std", torch.tensor(std)[:, None, None]
        )
        self.register_buffer(
            "lower_bound", torch.tensor(lower_bound)[:, None, None]
        )
        self.register_buffer(
            "upper_bound", torch.tensor(upper_bound)[:, None, None]
        )

    def get_latent_bounds(
            self,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_bound = self.to_latent(self.lower_bound, first_guess, mask)
        lower_bound = torch.where(
            torch.isfinite(lower_bound),
            lower_bound,
            -torch.inf,
        )
        upper_bound = self.to_latent(self.upper_bound, first_guess, mask)
        upper_bound = torch.where(
            torch.isfinite(upper_bound),
            upper_bound,
            torch.inf,
        )
        return lower_bound, upper_bound

    def to_latent(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        normed_residual = (in_tensor - first_guess - self.mean) / self.std
        return mask_tensor(normed_residual, mask)

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        prediction = first_guess + in_tensor * self.std + self.mean
        physical_prediction = prediction.clamp(
            min=self.lower_bound, max=self.upper_bound
        )
        return mask_tensor(physical_prediction, mask)
