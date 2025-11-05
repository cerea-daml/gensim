#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
from typing import Tuple, Optional
import logging

# External modules
import torch

# Internal modules


main_logger = logging.getLogger(__name__)


def get_schedule(
        n_steps: int, scale: float = 0.1, shift: float = 0.
) -> torch.Tensor:
    assert scale >= 0, "Scale must be non-negative"
    pseudo_time = torch.linspace(0., 1., n_steps+1)
    if scale > 0:
        main_logger.info(
            f"Activated sigmoid schedule with {n_steps:d} steps,"
            f"{scale:.1f} as scale, and {shift:.1f} as shift."
        )
        sigmoid_time = torch.sigmoid(scale * (pseudo_time-0.5) + shift)
        normalizer = sigmoid_time[-1] - sigmoid_time[0]
        pseudo_time = (sigmoid_time - sigmoid_time[0]) / normalizer
    return pseudo_time


class FlowMatchingSampler(object):
    def __init__(
            self,
            model: torch.nn.Module,
            n_steps: int = 20,
            schedule_scale: float = 0.1,
            schedule_shift: float = 0.,
            second_order: bool = True,
            censoring: bool = True
    ) -> None:
        self.model = model
        self.n_steps = n_steps
        self.schedule = get_schedule(n_steps, schedule_scale, schedule_shift)
        self.second_order = second_order
        self.censoring = censoring
        
    def activate_bounding(
            self,
            pseudo_time: torch.Tensor,
            latent_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        latent_given = latent_bounds is not None
        time_bound = (pseudo_time < 1).all()
        return latent_given and time_bound and self.censoring

    @staticmethod
    def _bound_grad(
            states: torch.Tensor,
            grad: torch.Tensor,
            pseudo_time: torch.Tensor,
            latent_bounds: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # Estimate Eulerian prediction
        prediction = states + (1-pseudo_time.view(-1, 1, 1, 1)) * grad

        # Bound prediction
        prediction = prediction.clamp(
            min=latent_bounds[0], max=latent_bounds[1]
        )

        # Estimate gradient from bounded prediction
        grad = (prediction - states) / (1 - pseudo_time.view(-1, 1, 1, 1))
        return grad

    def compute_grad(
            self,
            states: torch.Tensor,
            encoded: torch.Tensor,
            pseudo_time: torch.Tensor,
            latent_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **model_kwargs
    ) -> torch.Tensor:
        in_tensor = torch.cat((states, encoded), dim=1)
        grad = self.model(
            in_tensor, pseudo_time=pseudo_time, **model_kwargs
        )
        if self.activate_bounding(pseudo_time, latent_bounds):
            grad = self._bound_grad(states, grad, pseudo_time, latent_bounds)
        return grad

    def update_state(
            self,
            state: torch.Tensor,
            grad: torch.Tensor,
            curr_time: torch.Tensor,
            next_time: torch.Tensor,
    ) -> torch.Tensor:
        return state + (next_time - curr_time) * grad

    @torch.no_grad()
    def sample(
        self,
        states: torch.Tensor,
        encoded: torch.Tensor,
        latent_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        pseudo_time = torch.ones(
            states.size(0), 1, device=encoded.device
        )
        for idx, next_time in enumerate(self.schedule[1:]):
            curr_time = self.schedule[idx]

            # Prediction
            pseudo_time.fill_(curr_time)
            grad = self.compute_grad(
                states=states,
                encoded=encoded,
                pseudo_time=pseudo_time,
                latent_bounds=latent_bounds,
                **model_kwargs
            )
            if self.second_order and next_time < 1:
                # Do Euler step
                next_states = self.update_state(
                    states, grad, curr_time, next_time
                )

                # Prediction for DPM step
                pseudo_time.fill_(next_time)
                grad_next = self.compute_grad(
                    states=next_states,
                    encoded=encoded,
                    pseudo_time=pseudo_time,
                    latent_bounds=latent_bounds,
                    **model_kwargs
                )

                # Update gradient with second-order update
                grad = 0.5 * (grad_next + grad)

            states = self.update_state(states, grad, curr_time, next_time)
        return states
