#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from math import sqrt

# External modules
import torch

# Internal modules


main_logger = logging.getLogger(__name__)


__all__ = [
    "Embedder",
    "LogScaleModel"
]


class RandomFourierEmbedding(torch.nn.Module):
    def __init__(
            self,
            n_in_features: int = 1,
            n_features: int = 512,
            scale: float = 1.,
    ):
        super().__init__()
        half_dim = n_features // 2
        self.register_buffer(
            "frequencies",
            torch.randn(n_in_features, half_dim) / scale
        )
        self.scaling = 1/sqrt(half_dim)

    def forward(
            self, in_tensor: torch.Tensor
    ) -> torch.Tensor:
        embedding = in_tensor@self.frequencies
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding * self.scaling


class Embedder(torch.nn.Module):
    def __init__(
            self,
            n_embedding: int = 512,
            n_time_in: int = 1,
            n_res_in: int = 1,
            n_augment_in: int = 3,
    ):
        super().__init__()
        self.n_embedding = n_embedding
        self.n_time_in = n_time_in
        self.n_res_in = n_res_in
        self.n_augment_in = n_augment_in
        if n_time_in > 0:
            self.time_embedder = torch.nn.Sequential(
                RandomFourierEmbedding(n_time_in, n_embedding, scale=0.15),
                torch.nn.Linear(n_embedding, n_embedding),
                torch.nn.SiLU(),
                torch.nn.Linear(n_embedding, n_embedding),
            )
        else:
            self.register_module("time_embedder", None)
        if n_augment_in > 0:
            self.augment_embedder = torch.nn.Sequential(
                RandomFourierEmbedding(n_augment_in, n_embedding, scale=1.),
                torch.nn.Linear(n_embedding, n_embedding),
                torch.nn.SiLU(),
                torch.nn.Linear(n_embedding, n_embedding),
            )
        else:
            self.register_module("augment_embedder", None)
        if n_res_in > 0:
            self.resolution_embedder = torch.nn.Sequential(
                RandomFourierEmbedding(n_res_in, n_embedding, scale=1.5),
                torch.nn.Linear(n_embedding, n_embedding),
                torch.nn.SiLU(),
                torch.nn.Linear(n_embedding, n_embedding),
            )
        else:
            self.register_module("resolution_embedder", None)
        self.embedding_activation = torch.nn.SiLU()

    def forward(
            self,
            in_tensor: torch.Tensor,
            pseudo_time: torch.Tensor,
            labels: torch.Tensor,
            resolution: torch.Tensor,
    ) -> torch.Tensor:
        # Embedding
        embedding = torch.zeros(
            in_tensor.size(0), self.n_embedding,
            device=in_tensor.device, layout=in_tensor.layout,
            dtype=in_tensor.dtype
        )
        if self.time_embedder is not None:
            embedding.add_(
                self.time_embedder(pseudo_time)
            )
        if self.augment_embedder is not None:
            embedding.add_(
                self.augment_embedder(labels)
            )
        if self.resolution_embedder is not None:
            embedding.add_(
                self.resolution_embedder(resolution.log())
            )
        embedding = self.embedding_activation(embedding)
        return embedding


class LogScaleModel(torch.nn.Module):
    def __init__(
        self,
        n_embedding: int = 512,
        n_time_in: int = 1,
        n_res_in: int = 1,
        n_augment_in: int = 3,
        n_vars: int = 6
    ):
        super().__init__()
        self.embedder = Embedder(
            n_embedding=n_embedding,
            n_time_in=n_time_in,
            n_res_in=n_res_in,
            n_augment_in=n_augment_in
        )
        self.out_layer = torch.nn.Linear(n_embedding, n_vars)
        torch.nn.init.zeros_(self.out_layer.weight)
        torch.nn.init.zeros_(self.out_layer.bias)

    def forward(
            self,
            pseudo_time: torch.Tensor,
            labels: torch.Tensor,
            resolution: torch.Tensor
    ) -> torch.Tensor:
        features = self.embedder(pseudo_time, pseudo_time, labels, resolution)
        return self.out_layer(features)
