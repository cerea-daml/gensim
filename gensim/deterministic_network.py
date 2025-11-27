#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable, Tuple, List
import math

# External modules
import torch
from einops import rearrange, reduce

# Internal modules
from .network import Embedder, MLPLayer, TransformerBlock
from .utils import ToChannelsLastWrapper, mask_tensor


main_logger = logging.getLogger(__name__)


def mask_reduce(mask: torch.Tensor, factor: int) -> torch.Tensor:
    reduced_mask = reduce(
        mask,
        "b c (h hp) (w wp) -> b c h w",
        reduction="max",
        hp=factor, wp=factor
    )
    return reduced_mask


def field_rms_norm(
        in_tensor: torch.Tensor, scale: torch.Tensor, eps: float = 1E-6
) -> torch.Tensor:
    in_f32 = in_tensor.to(torch.float32)
    norm = in_f32.pow(2).mean(dim=1, keepdims=True) + eps
    normed_f32 = in_f32 * torch.rsqrt(norm)
    return normed_f32.to(in_tensor) * scale[..., None, None]


class ConvNeXtBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 512,
            n_embedding: int = 256,
            kernel_size: Tuple[int, int] = (7, 7),
            mult: int = 1,
            dropout_rate: float = 0.,
            padding_mode: str = 'zeros'
    ):
        super().__init__()
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=((kernel_size[0]-1)//2, (kernel_size[1]-1)//2),
            padding_mode=padding_mode,
            groups=in_channels,
            bias=False
        )
        self.scale_layer = torch.nn.Linear(n_embedding, in_channels)
        self.mlp_layer = ToChannelsLastWrapper(MLPLayer(
            n_features=in_channels,
            mult=mult,
            dropout_rate=dropout_rate,
        ), channel_dim=1)

    def forward(
            self,
            in_tensor: torch.Tensor,
            mask: torch.Tensor,
            embedding: torch.Tensor,
    ) -> torch.Tensor:
        branch_tensor = self.conv_layer(in_tensor)
        scale_tensor = self.scale_layer(embedding) + 1
        branch_tensor = field_rms_norm(branch_tensor, scale_tensor)
        branch_tensor = self.mlp_layer(branch_tensor)
        branch_tensor = mask_tensor(branch_tensor, mask)
        return in_tensor + branch_tensor


class ConvDownBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_embedding: int = 512,
    ):
        super().__init__()
        self.scale_layer = torch.nn.Linear(n_embedding, in_channels)
        self.down_layer = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0,
            bias=False
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedding: torch.Tensor,
    ) -> torch.Tensor:
        scale_tensor = self.scale_layer(embedding) + 1
        normed_tensor = field_rms_norm(in_tensor, scale_tensor)
        down_tensor = self.down_layer(normed_tensor)
        return down_tensor


class ConvUpBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_embedding: int = 512,
    ):
        super().__init__()
        self.scale_layer = torch.nn.Linear(n_embedding, in_channels)
        self.gate_layer = torch.nn.Linear(n_embedding, out_channels)
        self.up_sample_conv = torch.nn.Conv2d(
            in_channels, out_channels*4, bias=False, kernel_size=1
        )
        
        # Initialize as nearest neighbor interpolation
        weight = torch.empty(out_channels, in_channels, 1, 1)
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        weight = torch.repeat_interleave(weight, 4, dim=0)
        self.up_sample_conv.weight.data.copy_(weight)
        
        self.up_sample_shuffle = torch.nn.PixelShuffle(2)

    def forward(
            self,
            in_tensor: torch.Tensor,
            shortcut: torch.Tensor,
            mask: torch.Tensor,
            embedding: torch.Tensor,
    ) -> torch.Tensor:
        scale_tensor = self.scale_layer(embedding) + 1
        normed_tensor = field_rms_norm(in_tensor, scale_tensor)
        upsampled_tensor = self.up_sample_conv(normed_tensor)
        upsampled_tensor = self.up_sample_shuffle(upsampled_tensor)
        gate_tensor = self.gate_layer(embedding) + 0.5
        out_tensor = torch.lerp(
            upsampled_tensor,
            shortcut.to(upsampled_tensor.dtype),
            gate_tensor[..., None, None].to(upsampled_tensor.dtype)
        )
        out_tensor = mask_tensor(out_tensor, mask)
        return out_tensor


class Tokenizer(torch.nn.Module):
    def __init__(
            self, patch_size: Tuple[int, int], lengthscale: float = 100.
) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.lengthscale = lengthscale

    def tokenize_tensor(
            self, in_tensor: torch.Tensor
    ) -> torch.Tensor:
        tokens = rearrange(in_tensor, "b c h w -> b (h w) c")
        return tokens

    def tokenize_mesh(self, mesh: torch.Tensor) -> torch.Tensor:
        tokens_mesh = reduce(
            mesh, "b c (h hp) (w wp) -> b (h w) c", reduction="mean",
            hp=self.patch_size[0], wp=self.patch_size[1]
        ) / self.lengthscale
        return tokens_mesh

    def tokenize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        tokens_mask = reduce(
            mask, "b c (h hp) (w wp) -> b (h w) c", reduction="max",
            hp=self.patch_size[0], wp=self.patch_size[1]
        )
        return tokens_mask
        
    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.tokenize_tensor(in_tensor)
        tokens_mesh = self.tokenize_mesh(mesh)
        tokens_mask = self.tokenize_mask(mask)
        return tokens, tokens_mesh, tokens_mask


class Bottleneck(torch.nn.Module):
    def __init__(
            self,
            n_features: int = 512,
            n_blocks: int = 8,
            n_embedding: int = 256,
            n_heads: int = 8,
            mult: int = 1,
            patch_size: Tuple[int, int] = (4, 4),
            lengthscale: float = 1_000.,
            dropout_mlp: float = 0.,
    ):
        super().__init__()
        self.tokenizer = Tokenizer(patch_size, lengthscale=lengthscale)
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                n_features=n_features,
                n_features_head=n_features//n_heads,
                n_heads=n_heads,
                n_embedding=n_embedding,
                n_rope_features=n_features//n_heads//2,
                mult=mult,
                dropout_mlp=dropout_mlp
            )
            for _ in range(n_blocks)
        ])

    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        tokens, tokens_mesh, tokens_mask = self.tokenizer(
            in_tensor, mesh, mask
        )
        for b in self.blocks:
            tokens = b(tokens, tokens_mesh, tokens_mask, embedding)
        out_tensor = rearrange(
            tokens, "b (h w) c -> b c h w",
            h=in_tensor.size(-2), w=in_tensor.size(-1)
        )
        return out_tensor


class DownBlocks(torch.nn.Module):
    def __init__(
            self,
            n_features: int,
            n_embedding: int,
            n_down_blocks: Iterable[int],
            channel_mul: Iterable[int],
            mult: int = 1,
            dropout_conv: float = 0.,
            padding_mode: str = "zeros"
    ):
        super().__init__()
        self.n_depth = len(n_down_blocks)
        self.blocks = torch.nn.ModuleList()
        channel_mul = [1] + list(channel_mul)
        for k, n_blocks in enumerate(n_down_blocks):
            curr_features = int(n_features * channel_mul[k])
            out_features = int(n_features * channel_mul[k + 1])
            self.blocks.append(torch.nn.Module())
            self.blocks[-1].blocks = torch.nn.ModuleList([
                ConvNeXtBlock(
                    curr_features,
                    n_embedding=n_embedding,
                    mult=mult,
                    padding_mode=padding_mode,
                    dropout_rate=dropout_conv,
                )
                for _ in range(n_blocks)
            ])
            self.blocks[-1].downscale = ConvDownBlock(
                curr_features, out_features, n_embedding=n_embedding
            )

    def forward(
            self,
            features: torch.Tensor,
            mask: torch.Tensor,
            embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        down_tensors = []
        for k, block in enumerate(self.blocks):
            curr_mask = mask_reduce(mask, 2**k)
            for b in block.blocks:
                features = b(
                    features,
                    mask=curr_mask,
                    embedding=embedding
                )
            down_tensors.append(features)
            features = block.downscale(
                features,
                embedding=embedding
            )
        return features, down_tensors


class UpBlocks(torch.nn.Module):
    def __init__(
            self,
            n_features: int,
            n_embedding: int,
            n_up_blocks: Iterable[int],
            channel_mul: Iterable[int],
            mult: int = 1,
            dropout_conv: float = 0.,
            padding_mode: str = "zeros"
    ) -> None:
        super().__init__()
        self.n_depth = len(n_up_blocks)
        self.up_blocks = torch.nn.ModuleList()
        channel_mul = [1] + list(channel_mul)
        for k, n_blocks in enumerate(n_up_blocks):
            curr_features = int(n_features*channel_mul[self.n_depth-k])
            out_features = int(n_features*channel_mul[self.n_depth-k-1])
            self.up_blocks.append(torch.nn.Module())
            self.up_blocks[-1].upscale = ConvUpBlock(
                curr_features, out_features, n_embedding=n_embedding,
            )
            self.up_blocks[-1].blocks = torch.nn.ModuleList([
                ConvNeXtBlock(
                    out_features,
                    n_embedding=n_embedding,
                    mult=mult,
                    padding_mode=padding_mode,
                    dropout_rate=dropout_conv,
                )
                for _ in range(n_blocks)
            ])

    def forward(
            self,
            features: torch.Tensor,
            shortcuts: List[torch.Tensor],
            mask: torch.Tensor,
            embedding: torch.Tensor
    ):
        for k, block in enumerate(self.up_blocks):
            idx = self.n_depth-k-1
            curr_mask = mask_reduce(mask, factor=2**idx)
            features = block.upscale(
                features,
                shortcut=shortcuts[idx],
                mask=curr_mask,
                embedding=embedding
            )
            for b in block.blocks:
                features = b(
                    features,
                    mask=curr_mask,
                    embedding=embedding
                )
        return features


class UViTHead(torch.nn.Module):
    def __init__(
            self,
            n_features: int = 64,
            n_output: int = 5,
            n_embedding: int = 256,
    ):
        super().__init__()
        self.scale_layer = torch.nn.Linear(n_embedding, n_features)
        self.activation = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(
            n_features, n_output, kernel_size=1, bias=True
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            mask: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        scale_tensor = self.scale_layer(embedding) + 1
        normed = field_rms_norm(in_tensor, scale_tensor)
        activated = self.activation(normed)
        conv_out = self.conv(activated)
        conv_out = mask_tensor(conv_out, mask)
        return conv_out


class UViT(torch.nn.Module):
    def __init__(
            self,
            n_input: int = 11,
            n_output: int = 5,
            n_features: int = 64,
            n_bottleneck: int = 8,
            n_down_blocks: Iterable[int] = (3, 3, 3),
            n_up_blocks: Iterable[int] = (3, 3, 3),
            channel_mul: Iterable[int] = (1, 2, 4),
            n_embedding: int = 256,
            n_time_in: int = 1,
            n_res_in: int = 1,
            n_augment_in: int = 3,
            n_heads: int = 8,
            mult: int = 1,
            lengthscale: float = 1_000.,
            dropout_conv: float = 0.,
            dropout_mlp: float = 0.,
            padding_mode: str = "zeros",
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_depth = len(n_down_blocks)
        self.n_embedding = n_embedding

        self.embedder = Embedder(
            n_embedding=n_embedding,
            n_time_in=n_time_in,
            n_res_in=n_res_in,
            n_augment_in=n_augment_in,
        )
        self.in_encoder = torch.nn.Conv2d(n_input+1, n_features, 1, bias=False)
        self.down_blocks = DownBlocks(
            n_features=n_features,
            n_embedding=n_embedding,
            n_down_blocks=n_down_blocks,
            channel_mul=channel_mul,
            mult=mult,
            dropout_conv=dropout_conv,
            padding_mode=padding_mode
        )
        self.bottleneck = Bottleneck(
            n_features*channel_mul[-1],
            n_blocks=n_bottleneck,
            n_embedding=n_embedding,
            n_heads=n_heads,
            mult=mult,
            lengthscale=lengthscale,
            dropout_mlp=dropout_mlp,
            patch_size=(2**self.n_depth, 2**self.n_depth)
        )
        self.up_blocks = UpBlocks(
            n_features=n_features,
            n_embedding=n_embedding,
            n_up_blocks=n_up_blocks,
            channel_mul=channel_mul,
            mult=mult,
            dropout_conv=dropout_conv,
        )
        self.head = UViTHead(
            n_features=n_features,
            n_output=n_output,
            n_embedding=n_embedding
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            labels: torch.Tensor,
            resolution: torch.Tensor,
            pseudo_time: torch.Tensor = None,
    ) -> torch.Tensor:
        embedding = self.embedder(
            in_tensor, pseudo_time=pseudo_time, labels=labels,
            resolution=resolution
        )
        
        in_pad = torch.nn.functional.pad(
            in_tensor, (0, 0, 0, 0, 0, 1), mode="constant", value=1.
        )
        in_masked = mask_tensor(in_pad, mask)
        features = self.in_encoder(in_masked)

        features, shortcuts = self.down_blocks(features, mask, embedding)
        features = self.bottleneck(features, mesh, mask, embedding)
        features = self.up_blocks(features, shortcuts, mask, embedding)

        output = self.head(features, mask, embedding)
        return output
