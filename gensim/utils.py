#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable, Dict, List, Tuple
from functools import wraps
import math

# External modules
import xarray as xr
import zarr
import numpy as np
import cartopy.crs as ccrs

import torch

# Internal modules


main_logger = logging.getLogger(__name__)


STATE_VARIABLES_DEFAULT = ("sit", "sic", "sid", "siu", "siv", "snt")
FORCING_VARIABLES_DEFAULT = ("tus", "rhus", "uas", "vas")


def identify_var_idx(
        keys: Iterable[str],
        query: Iterable[str]
) -> List[int]:
    keys = list(keys)
    return [keys.index(var) for var in query]


def determine_start_idx(zarr_path: str, start_date: str = None) -> int:
    if start_date is None:
        return 0
    with xr.open_dataset(zarr_path) as ds_time:
        time_index = ds_time.indexes["time"]
        return time_index.get_loc(start_date)


def load_zarr(
        zarr_path: str,
        variables: Iterable[str, ]
) -> Dict[str, np.ndarray]:
    dataset = zarr.open(zarr_path, mode="r")
    return {var: dataset[var][...] for var in variables}


def batch_to_tensor(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    nb, _, _, ny, nx = batch["states"].shape
    combined_tensor = torch.cat(
        [v.reshape(nb, -1, ny, nx) for v in batch.values()], dim=1
    )
    return combined_tensor


def tensor_to_batch(
        output: torch.Tensor,
        original_batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    nb, _, _, ny, nx = original_batch["states"].shape
    split_tensors = output.split(
        [v.numel()//nb//ny//nx for v in original_batch.values()], dim=1
    )
    input_shapes = [v.shape for v in original_batch.values()]
    reshaped_tensors = [
        tensor.reshape(-1, *input_shapes[k][1:-2], *tensor.shape[-2:])
        for k, tensor in enumerate(split_tensors)
    ]
    output_batch = dict(zip(original_batch.keys(), reshaped_tensors))
    return output_batch


def split_wd_params(
        model: torch.nn.Module
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    # From minGPT https://github.com/karpathy/minGPT
    # Explanation: https://github.com/karpathy/minGPT/pull/24
    decay_params = set()
    no_decay_params = set()
    no_grad_params = set()
    for name, param in model.named_parameters():
        parent_module = model.get_submodule(".".join(name.split(".")[:-1]))
        decay = (
            name.endswith('weight')
            and not isinstance(parent_module, torch.nn.GroupNorm)
            and "norm" not in name
            and "mod" not in name
            and "embedding" not in name
            and "embedder" not in name
            and "log_scale" not in name
            and "ema" not in name
            and "qk_scaling" not in name
        )
        if decay and param.requires_grad:
            decay_params.add(name)
        elif param.requires_grad:
            no_decay_params.add(name)
        else:
            no_grad_params.add(name)

    # Check if all parameters are considered
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay_params & no_decay_params & no_grad_params
    union_params = decay_params | no_decay_params | no_grad_params
    missing_keys = param_dict.keys() - union_params
    if len(inter_params) != 0:
        raise AssertionError(
            "Parameters {0:s} made it into different sets!".format(
                str(inter_params)
            )
        )
    if len(missing_keys) != 0:
        raise AssertionError(
            "Parameters {0:s} were not separated into sets!".format(
                missing_keys
            )
        )

    # Convert into lists of parameters
    decay_params = [param_dict[pn] for pn in sorted(list(decay_params))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay_params))]
    return decay_params, no_decay_params


def mask_tensor(in_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=in_tensor.dtype)
    return in_tensor * mask


def unmask_tensor(
        field: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Unmask a given field by given mask. The mask is converted into a boolean
    tensor and all elements where the mask is False are set to zero.
    """
    target_shape = (*field.shape[:-1], *mask.shape[-2:])
    mask_shape = [mask.size(0)] \
        + [1,] * (field.ndim-2) \
        + list(mask.shape[-2:])
    new_mask = mask.to(dtype=torch.bool)
    new_mask = new_mask.view(mask_shape).expand(target_shape)
    new_tensor = torch.zeros(
        target_shape, device=field.device, dtype=field.dtype
    )
    new_tensor[new_mask] = field.flatten()
    return new_tensor


def masked_average(
        to_average: torch.Tensor,
        mask: torch.Tensor,
        dim=None
) -> torch.Tensor:
    """
    Computes the masked average of a tensor.

    Args:
        to_average: The tensor to compute the average over.
        mask: A boolean mask to apply.
        dim: The dimension along which to compute the average.

    Returns:
        A tensor with the masked average.
    """
    expanded_mask = mask.expand_as(to_average)
    masked_sum = (to_average * expanded_mask).sum(dim=dim)
    return masked_sum / expanded_mask.sum(dim=dim).clamp(min=1)


def sample_uniform_time(
        template_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Samples time indices as tensor between [0, 1]. The time indices are
    equidistant distributed to reduce the training variance as proposed in
    Kingma et al., 2021.

    Parameters
    ----------
    template_tensor : torch.Tensor
        The template tensor with `n` dimensions and shape (batch size, *).

    Returns
    -------
    sampled_time : torch.Tensor
        The time tensor sampled for each sample independently. The resulting
        shape is (batch size, *) with `n` dimensions filled by ones. The
        tensor lies on the same device as the input.
    """
    time_shape = torch.Size(
        [template_tensor.size(0)] + [1, ] * (template_tensor.ndim - 1)
    )
    # Draw initial time
    time_shift = torch.rand(
        1, dtype=template_tensor.dtype, device=template_tensor.device
    )
    # Equidistant timing
    sampled_time = torch.linspace(
        0, 1, template_tensor.size(0) + 1,
        dtype=template_tensor.dtype, device=template_tensor.device
    )[:template_tensor.size(0)]
    # Estimate time
    sampled_time = (time_shift + sampled_time) % 1
    sampled_time = sampled_time.reshape(time_shape)
    return sampled_time


_const = math.log(math.sqrt(2 * math.pi))


def neglogpdf(value: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log probability density function for a normal 
    distribution.

    Args:
        value: The normalized value to compute the PDF for.
        scale: The logarithm of the scale parameter.

    Returns:
        The log probability density function value.
    """
    return 0.5 * (value.pow(2)) + log_scale + _const


def neglogcdf(value: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative logarithm of the cumulative distribution function for 
    a normal distribution.

    Args:
        value: The normalized value to compute the CDF for.

    Returns:
        The cumulative distribution function value.
    """    
    return -torch.special.log_ndtr(value)
