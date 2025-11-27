#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple, Optional

# External modules
import torch

# Internal modules
from .utils import (
    get_empty_labels, generate_noise, get_latent_states
)
from .wrapper import PatchedNetwork


main_logger = logging.getLogger(__name__)


class GenSIMForecastModule(torch.nn.Module):
    _LABELS_DIMS: int = 3

    def __init__(
            self,
            network: torch.nn.Module,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            sampler: torch.nn.Module,
            patching: bool = True,
            patch_size: Tuple[int, int] = (64, 64),
            overlap_size: Tuple[int, int] = (8, 8),
    ):
        super().__init__()
        self.network = network
        self.encoder = encoder
        self.decoder = decoder

        # For sampling
        self.patching = patching
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.overlap_slices = (
            slice(self.overlap_size[0], -self.overlap_size[0]),
            slice(self.overlap_size[1], -self.overlap_size[1]),
        )

        # Instantiate sampler if provided
        self.sampler = sampler
        self.sampler.model = self.network

        # Set inference model with deactivated compilation
        self.set_inference_model(compile_model=False)

    def forward(
            self,
            states: torch.Tensor,
            forcings: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            resolution: torch.Tensor,
            degree_days: torch.Tensor
    ) -> torch.Tensor:
        encoded, latent_mesh, latent_mask = get_latent_states(
            states, forcings, mesh, mask, degree_days, self.encoder
        )

        labels = get_empty_labels(encoded, self._LABELS_DIMS)

        first_guess = states[:, -1]
        initial_states = generate_noise(first_guess, mask)
        latent_bounds = self.decoder.get_latent_bounds(first_guess, mask)
        dynamics = self.sampler.sample(
            states=initial_states,
            encoded=encoded,
            mesh=latent_mesh,
            mask=latent_mask,
            labels=labels,
            resolution=resolution,
            latent_bounds=latent_bounds
        )
        return self.decoder(
            dynamics,
            first_guess=first_guess,
            mask=mask
        )

    def set_inference_model(
            self,
            compile_model: bool = False,
            padding_mode: str = 'replicate',
            **compile_kwargs
    ) -> None:
        model = self.network
        if compile_model:
            model = torch.compile(model, **compile_kwargs)
            main_logger.info(
                f"Model compilation activated with {compile_kwargs}"
            )
        if self.patching:
            model = PatchedNetwork(
                model=model,
                patch_size=self.patch_size,
                overlap_size=self.overlap_size,
                padding_mode=padding_mode
            )
            main_logger.info(
                f"Patched inference model with {tuple(self.patch_size)} "
                f"as patch size, {tuple(self.overlap_size)} as overlap size, "
                f"and {padding_mode} as padding."
            )
        self.inference_model = model
        if self.sampler is not None:
            self.sampler.model = self.inference_model
        return None