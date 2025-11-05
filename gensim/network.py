#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple, Iterable
from math import sqrt, inf

# External modules
import torch
import torch.nn.functional as F
import torch.nn
from einops import rearrange, reduce


# Fallback for flash attention
try:
    from flash_attn import flash_attn_func
    USE_FLASH_ATTN = True
except ImportError:
    USE_FLASH_ATTN = False
    flash_attn_func = None

# Internal modules
from .embedding import Embedder


main_logger = logging.getLogger(__name__)


__all__ = [
    "Transformer"
]


def mask_tensor(in_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=in_tensor.dtype)
    return in_tensor * mask


def self_attention(
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor
) -> torch.Tensor:
    """
    Self-attention function for flash attention, falling back to
    `scaled_dot_product_attention` if not available.
    """
    if USE_FLASH_ATTN:
        return flash_attn_func(
            q_proj.to(torch.bfloat16),
            k_proj.to(torch.bfloat16),
            v_proj.to(torch.bfloat16),
            dropout_p=0.,
            softmax_scale=None,
            causal=False
        ).to(q_proj)
    else:
        attn_out = F.scaled_dot_product_attention(
            q_proj.transpose(1, 2),
            k_proj.transpose(1, 2),
            v_proj.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.,
            scale=None,
            is_causal=False
        )
        return attn_out.transpose(1, 2)


def estimate_sine_features(
        mesh: torch.Tensor,
        freqs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mesh_f32 = mesh.float()
    freqs_f32 = freqs.float()
    embedded = torch.einsum(
        "blc,chk->blhk", mesh_f32, freqs_f32
    )
    sin_embedded = embedded.sin()
    cos_embedded = embedded.cos()
    return sin_embedded.to(mesh.dtype), cos_embedded.to(mesh.dtype)


def apply_rope(
        in_tensor: torch.Tensor,
        features: Tuple[torch.Tensor, torch.Tensor],
        n_features: int = 32
) -> torch.Tensor:
    # Rotate the query and key using the computed sine features
    rotated_first = in_tensor[..., :n_features] * features[1] - \
        in_tensor[..., n_features:2*n_features] * features[0]
    rotated_second = in_tensor[..., :n_features] * features[0] + \
        in_tensor[..., n_features:2*n_features] * features[1]
    return torch.cat((
        rotated_first,
        rotated_second,
        in_tensor[..., 2*n_features:]
    ), dim=-1)


class RopeLayer(torch.nn.Module):
    def __init__(
            self,
            n_features: int = 32,
            n_heads: int = 8,
            min_theta: float = 0.,
            max_theta: float = 0.333333,
            random_angle: bool = True,
            device=None, dtype=None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.n_heads = n_heads
        self.n_features = n_features
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.random_angle = random_angle
        self.freqs = torch.nn.Parameter(
            torch.empty(2, n_heads, n_features, **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        factory_kwargs = {
            'device': self.freqs.device,
            'dtype': self.freqs.dtype
        }
        # Span magnitude of frequencies along number of features
        mag = 1 / torch.logspace(
            self.min_theta, self.max_theta, self.n_features//2, base=10.,
            **factory_kwargs
        )[None, :]
        if self.random_angle:
            # Get a random angle per head
            angles = torch.rand(
                self.n_heads, 1, **factory_kwargs
            ) * 2 * torch.pi
        else:
            angles = torch.zeros(self.n_heads, 1, **factory_kwargs)
        # Frequencies in x direction
        freqs_x = torch.cat([
            mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)
        ], dim=-1)
        # Frequencies in y direction
        freqs_y = torch.cat([
            mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)
        ], dim=-1)
        # stack them
        freqs = torch.stack([freqs_x, freqs_y], dim=0)
        self.freqs.data.copy_(freqs)

    def forward(
            self,
            query_tensor: torch.Tensor,
            key_tensor: torch.Tensor,
            mesh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sine_features = estimate_sine_features(mesh, self.freqs)
        return (
            apply_rope(query_tensor, sine_features, self.n_features),
            apply_rope(key_tensor, sine_features, self.n_features)
        )


class SelfAttentionLayer(torch.nn.Module):
    def __init__(
            self,
            n_features: int = 512,
            n_features_head: int = 64,
            n_heads: int = 8,
            n_rope_features: int = 32
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_features_head = n_features_head
        self.n_heads = n_heads

        self.qkv_layer = torch.nn.Linear(
            n_features,
            n_features_head * n_heads * 3,
            bias=False
        )
        self.q_norm = torch.nn.RMSNorm(n_features_head)
        self.k_norm = torch.nn.RMSNorm(n_features_head)
        if n_rope_features > 0:
            self.rope_layer = RopeLayer(
                n_features=n_rope_features,
                n_heads=n_heads
            )
        else:
            self.register_buffer("rope_layer", None)
        self.out_layer = torch.nn.Linear(
            n_features_head * n_heads, n_features, bias=False
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = in_tensor.shape

        # Project layers
        qkv_proj = self.qkv_layer(in_tensor)
        qkv_proj = qkv_proj.view(B, T, 3, self.n_heads, self.n_features_head)
        q_proj, k_proj, v_proj = qkv_proj.unbind(dim=2)

        # QK normalization and apply RoPE
        q_proj = self.q_norm(q_proj)
        k_proj = self.k_norm(v_proj)
        if self.rope_layer is not None:
            q_proj, k_proj = self.rope_layer(q_proj, k_proj, mesh)

        out = self_attention(q_proj, k_proj, v_proj)
        out = out.reshape(B, T, -1)
        return self.out_layer(out)


class MLPLayer(torch.nn.Module):
    def __init__(
            self,
            n_features,
            mult: int = 1,
            dropout_rate: float = 0.,
    ):
        super().__init__()
        self.hidden_features = n_features * mult
        self.in_layer = torch.nn.Linear(
            n_features, self.hidden_features*2, bias=False
        )
        self.activation = torch.nn.SiLU()
        self.out_layer = torch.nn.Sequential()
        if dropout_rate > 0.:
            self.out_layer.append(torch.nn.Dropout(p=dropout_rate))
        self.out_layer.append(
            torch.nn.Linear(
                self.hidden_features, n_features, bias=False
            )
        )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        branch, gate = self.in_layer(in_tensor).chunk(2, dim=-1)
        out_tensor = self.out_layer(
            branch * self.activation(gate)
        )
        return out_tensor


class TransformerBlock(torch.nn.Module):
    def __init__(
            self,
            n_features: int = 512,
            n_features_head: int = 64,
            n_heads: int = 8,
            n_embedding: int = 256,
            n_rope_features: int = 16,
            mult: int = 1,
            dropout_mlp: float = 0.,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.gate_layer = torch.nn.Linear(
            n_embedding, n_features*2, bias=False
        )
        self.pre_attention_norm = torch.nn.RMSNorm(
            n_features, elementwise_affine=False
        )
        self.self_attention = SelfAttentionLayer(
            n_features=n_features,
            n_features_head=n_features_head,
            n_heads=n_heads,
            n_rope_features=n_rope_features,
        )
        self.pre_mlp_norm = torch.nn.RMSNorm(
            n_features, elementwise_affine=False
        )
        self.mlp_layer = MLPLayer(
            n_features=n_features,
            mult=mult,
            dropout_rate=dropout_mlp,
        )

    def apply_attention_branch(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            gate: torch.Tensor
    ) -> torch.Tensor:
        residual = self.pre_attention_norm(in_tensor) * gate
        residual = self.self_attention(residual, mesh)
        residual = mask_tensor(residual, mask)
        return residual

    def apply_mlp_branch(
            self,
            in_tensor: torch.Tensor,
            mask: torch.Tensor,
            gate: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.pre_mlp_norm(in_tensor) * gate
        residual = self.mlp_layer(residual)
        residual = mask_tensor(residual, mask)
        return residual

    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        gates = (self.gate_layer(embedding)[:, None] + 1).chunk(2, dim=-1)
        residual = self.apply_attention_branch(
            in_tensor, mesh, mask, gates[0]
        )
        out_tensor = in_tensor + residual
        residual = self.apply_mlp_branch(in_tensor, mask, gates[1])
        out_tensor = out_tensor + residual
        return out_tensor

class Tokenizer(torch.nn.Module):
    def __init__(
            self,
            n_input: int,
            n_features: int,
            patch_size: int = 4,
            lengthscale: float = 100.
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.patch_size = patch_size
        self.in_encoder = torch.nn.Linear(
            (n_input+1) * patch_size * patch_size,
            n_features, bias=False
        )

    def tokenize_tensor(
            self, in_tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # Padding with channel of ones to enable "bias-like" model
        in_pad = torch.nn.functional.pad(
            in_tensor, (0, 0, 0, 0, 0, 1), mode="constant", value=1.
        )
        in_masked = mask_tensor(in_pad, mask)

        tokens = rearrange(
            in_masked,
            "b c (h hp) (w wp) -> b (h w) (c hp wp)",
            hp=self.patch_size, wp=self.patch_size
        )
        tokens = self.in_encoder(tokens)
        return tokens

    def tokenize_mesh(self, mesh: torch.Tensor) -> torch.Tensor:
        tokens = reduce(
            mesh,
            "b c (h hp) (w wp) -> b (h w) c",
            "mean",
            hp=self.patch_size, wp=self.patch_size
        ) / self.lengthscale
        return tokens

    def tokenize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        tokens_mask = reduce(
            mask,
            "b c (h hp) (w wp) -> b (h w) c",
            "max",
            hp=self.patch_size, wp=self.patch_size
        )
        return tokens_mask

    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.tokenize_tensor(in_tensor, mask)
        tokens_mesh = self.tokenize_mesh(mesh).to(tokens)
        tokens_mask = self.tokenize_mask(mask).to(tokens)
        return tokens, tokens_mesh, tokens_mask


class Head(torch.nn.Module):
    def __init__(
            self,
            n_output: int,
            n_features: int,
            n_embedding: int,
            patch_size: int = 4
    ):
        super().__init__()
        self.n_features = n_features
        self.n_embedding = n_embedding
        self.n_output = n_output
        self.patch_size = patch_size
        self.gate_layer = torch.nn.Linear(n_embedding, n_features, bias=False)
        self.in_norm = torch.nn.RMSNorm(n_features, elementwise_affine=False)
        self.in_activation = torch.nn.ReLU()
        self.out_layer = torch.nn.Linear(
            n_features, n_output*patch_size**2, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights = torch.empty(self.n_output, self.n_features)
        # So that expected output variance = 1
        torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")
        # Initialize as nearest neighbor interpolation
        weights = weights.repeat_interleave(self.patch_size**2, dim=0)
        # Copy the weights to the output layer
        self.out_layer.weight.data.copy_(weights)

    def forward(
            self,
            tokens: torch.Tensor,
            mask: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        h = mask.size(-2) // self.patch_size
        w = mask.size(-1) // self.patch_size
        gate = self.gate_layer(embedding)[:, None] + 1
        tokens = self.in_norm(tokens) * gate
        tokens = self.in_activation(tokens)
        output = self.out_layer(tokens)
        output = rearrange(
            output,
            "b (h w) (c h2 w2) -> b c (h h2) (w w2)",
            h=h, w=w, h2=self.patch_size, w2=self.patch_size
        )
        output = mask_tensor(output, mask)
        return output


class Transformer(torch.nn.Module):
    def __init__(
            self,
            n_input: int = 11,
            n_output: int = 5,
            n_features: int = 512,
            n_blocks: int = 8,
            n_embedding: int = 256,
            n_time_in: int = 1,
            n_res_in: int = 1,
            n_augment_in: int = 3,
            n_features_head: int = 64,
            n_heads: int = 8,
            n_rope_features: int = 32,
            mult: int = 1,
            dropout_mlp: float = 0.,
            lengthscale: float = 100.,
            patch_size: int = 4,
            long_skips: bool = True,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_blocks = n_blocks
        self.n_features = n_features
        self.n_skips = (n_blocks-1)//2 if long_skips else 0
        self.long_skips = long_skips
        self.patch_size = patch_size
        self.embedder = Embedder(
            n_embedding=n_embedding,
            n_time_in=n_time_in,
            n_res_in=n_res_in,
            n_augment_in=n_augment_in,
        )
        self.tokenizer = Tokenizer(
            n_input=n_input,
            n_features=n_features,
            patch_size=patch_size,
            lengthscale=lengthscale
        )
        self.in_blocks = torch.nn.ModuleList([
            TransformerBlock(
                n_features=n_features,
                n_features_head=n_features_head,
                n_heads=n_heads,
                n_embedding=n_embedding,
                n_rope_features=n_rope_features,
                mult=mult,
                dropout_mlp=dropout_mlp,
            )
            for _ in range(self.n_skips)
        ])
        self.bottleneck = torch.nn.ModuleList([
            TransformerBlock(
                n_features=n_features,
                n_features_head=n_features_head,
                n_heads=n_heads,
                n_embedding=n_embedding,
                n_rope_features=n_rope_features,
                mult=mult,
                dropout_mlp=dropout_mlp,
            )
            for _ in range(self.n_blocks - 2*self.n_skips)
        ])
        self.out_blocks = torch.nn.ModuleList([
            TransformerBlock(
                n_features=n_features,
                n_features_head=n_features_head,
                n_heads=n_heads,
                n_embedding=n_embedding,
                n_rope_features=n_rope_features,
                mult=mult,
                dropout_mlp=dropout_mlp,
            )
            for _ in range(self.n_skips)
        ])
        self.skip_embedding = torch.nn.Linear(
            n_embedding, n_features*(self.n_skips+1), bias=False
        )
        self.head = Head(
            n_output=n_output,
            n_features=n_features,
            n_embedding=n_embedding,
            patch_size=patch_size
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            pseudo_time: torch.Tensor,
            labels: torch.Tensor,
            resolution: torch.Tensor
    ) -> torch.Tensor:
        embedding = self.embedder(
            in_tensor, pseudo_time=pseudo_time, labels=labels,
            resolution=resolution
        )
        tokens, tokens_mesh, tokens_mask = self.tokenizer(
            in_tensor, mesh, mask
        )
        tokens_mesh /= resolution.unsqueeze(1)

        gates = self.skip_embedding(embedding)[:, None, :]
        gates = gates.chunk(self.n_skips+1, dim=2)
        skips = [tokens]

        # Downward path
        for block in self.in_blocks:
            tokens = block(tokens, tokens_mesh, tokens_mask, embedding)
            skips.append(tokens)

        # Bottleneck
        for block in self.bottleneck:
            tokens = block(tokens, tokens_mesh, tokens_mask, embedding)

        # Upward path
        skips = skips[::-1]
        for k, block in enumerate(self.out_blocks):
            tokens = torch.lerp(skips[k], tokens, gates[k].expand_as(tokens))
            tokens = block(tokens, tokens_mesh, tokens_mask, embedding)

        # Add input tokens if long skips are activated
        if self.long_skips:
            tokens = torch.lerp(
                skips[-1], tokens, gates[-1].expand_as(tokens)
            )
        output = self.head(tokens, mask, embedding)
        return output
