#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Dict, Tuple, Any, Optional

# External modules
import torch
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Internal modules
from .embedding import LogScaleModel
from .utils import (
    sample_uniform_time, masked_average, neglogpdf, neglogcdf, split_wd_params, 
    mask_tensor
)
from .wrapper import PatchedNetwork


main_logger = logging.getLogger(__name__)


class FlowMatchingModel(pl.LightningModule):
    _LABELS_DIMS: int = 3

    def __init__(
            self,
            network: OmegaConf,
            encoder: OmegaConf,
            decoder: OmegaConf,
            sampler: OmegaConf,
            lr: float = 1E-4,
            lr_warmup: int = 5000,
            total_steps: int = 250000,
            weight_decay: float = 1E-3,
            ema_rate: float = 0.999,
            patching: bool = True,
            patch_size: Tuple[int, int] = (64, 64),
            overlap_size: Tuple[int, int] = (8, 8),
            train_with_overlap: bool = True,
            censoring: bool = True,
            optimize_scale: bool = True,
            epsilon: float = 1E-5,
            patch_generator: Optional[OmegaConf] = None,
            train_augmentation: Optional[OmegaConf] = None,
    ):
        super().__init__()
        self.inference_model = None

        # Neural networks
        self.network = instantiate(network)
        self.encoder = instantiate(encoder)
        self.decoder = instantiate(decoder)

        self.log_scale_model = LogScaleModel(
            n_embedding=self.network.embedder.n_embedding,
            n_time_in=self.network.embedder.n_time_in,
            n_res_in=self.network.embedder.n_res_in,
            n_augment_in=self.network.embedder.n_augment_in,
            n_vars=network.n_output
        )
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.network,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                ema_rate
            ),
            device="cpu"
        )
        self.ema_model.requires_grad_(False)
        self.ema_model = self.ema_model.eval()

        # For sampling
        self.patching = patching
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.train_with_overlap = train_with_overlap
        self.overlap_slices = (
            slice(self.overlap_size[0], -self.overlap_size[0]),
            slice(self.overlap_size[1], -self.overlap_size[1]),
        )
        self.censoring = censoring

        # Instantiate sampler
        self.sampler = instantiate(sampler, model=None)

        # Training parameters
        self.ema_rate = ema_rate
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.normalized_gammas = torch.ones(1)
        self.optimize_scale = optimize_scale

        self.patch_generator = instantiate(patch_generator)
        self.train_augmentation = instantiate(train_augmentation)
        self.train_time_sampler = sample_uniform_time

        # If needed for divison
        self.epsilon = epsilon

        # To set inference model with deactivated compilation
        self.set_inference_model(compile_model=False)

        # To enable the optimization of log scale
        self.automatic_optimization = False

        # To enable training continuation from partial checkpoint
        self.strict_loading = False

        # To save all given parameters
        self.save_hyperparameters()

    def forward(
            self,
            states: torch.Tensor,
            forcings: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            resolution: torch.Tensor,
            degree_days: torch.Tensor
    ) -> torch.Tensor:
        encoded, latent_mesh, latent_mask = self._get_latent_states(
            states, forcings, mesh, mask, degree_days
        )
        first_guess = states[:, -1]
        labels = self._get_empty_labels(encoded)
        dynamics = self.forecast_func(
            first_guess=first_guess,
            encoded=encoded,
            mesh=latent_mesh,
            mask=latent_mask,
            labels=labels,
            resolution=resolution
        )
        return self.decoder(
            dynamics,
            first_guess=first_guess,
            mask=mask
        )

    def _remove_overlap(self, field: torch.Tensor) -> torch.Tensor:
        if not self.train_with_overlap:
            return field
        return field[..., self.overlap_slices[0], self.overlap_slices[1]]

    def _get_empty_labels(self, template_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            template_tensor.size(0), self._LABELS_DIMS,
            dtype=template_tensor.dtype, device=template_tensor.device
        )
    
    def _generate_noise(
            self,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        noise = torch.randn_like(first_guess)
        return mask_tensor(noise, mask)

    def _get_latent_states(
            self,
            states: torch.Tensor,
            forcings: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            degree_days: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states_in = states.reshape(states.size(0), -1, *states.shape[-2:])
        forcings_in = forcings.reshape(
            forcings.size(0), -1, *forcings.shape[-2:]
        )
        forcings_in = torch.cat((forcings_in, degree_days), dim=-3)
        in_tensor = torch.cat((states_in, forcings_in), dim=-3)
        encoded, latent_mesh, latent_mask = self.encoder(in_tensor, mesh, mask)
        return encoded, latent_mesh, latent_mask

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
                patch_size=tuple(self.patch_size),
                overlap_size=tuple(self.overlap_size),
                padding_mode=padding_mode
            )
            main_logger.info(
                f"Patched inference model with {tuple(self.patch_size)} "
                f"as patch size, {tuple(self.overlap_size)} as overlap size, "
                f"and {padding_mode} as padding."
            )
        self.inference_model = torch.no_grad()(model)
        self.sampler.model = self.inference_model
        return None

    def forecast_func(
            self,
            first_guess: torch.Tensor,
            encoded: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            labels: torch.Tensor,
            resolution: torch.Tensor
    ) -> torch.Tensor:
        initial_states = self._generate_noise(first_guess, mask)
        latent_bounds = self.decoder.get_latent_bounds(first_guess, mask)
        return self.sampler.sample(
            states=initial_states,
            encoded=encoded,
            mesh=mesh,
            mask=mask,
            labels=labels,
            resolution=resolution,
            latent_bounds=latent_bounds
        )

    def estimate_loss(
            self,
            batch: Dict[str, torch.Tensor],
            resolution: torch.Tensor,
            labels: torch.Tensor,
            prefix: str = "train"
    ) -> Dict[str, torch.Tensor]:
        # Check if scores should be synced
        sync_dist = prefix != "train"

        # Input data
        encoded, latent_mesh, latent_mask = self._get_latent_states(
            batch["states"][:, :-1], batch["forcings"],
            batch["mesh"], batch["mask"], batch["degree_days"]
        )

        # Get linear interpolant
        residual = self.decoder.to_latent(
            batch["states"][:, -1], batch["states"][:, -2],
            batch["mask"]
        )
        noise = torch.randn_like(residual)
        sampled_time = self.train_time_sampler(residual)
        noised_residual = sampled_time * residual \
            + (1-sampled_time) * noise

        # Get input and target
        in_tensor = torch.cat(
            (noised_residual, encoded), dim=1
        )
        prediction = self.network(
            in_tensor,
            mesh=latent_mesh,
            mask=latent_mask,
            pseudo_time=sampled_time.view(-1, 1),
            labels=labels,
            resolution=resolution
        )

        log_scale = self.log_scale_model(
            pseudo_time=sampled_time.view(-1, 1),
            labels=labels,
            resolution=resolution
        )[:, :, None, None]

        # Estimate loss
        velocity = residual - noise
        error = (velocity - prediction) / (log_scale.exp() + self.epsilon)
        loss = neglogpdf(error, log_scale)

        if self.censoring:
            # Add censoring at lower bound
            loss = torch.where(
                torch.eq(batch["states"][:, -1], self.decoder.lower_bound),
                neglogcdf(error),
                loss
            )
            # Add censoring at upper bound
            loss = torch.where(
                torch.eq(batch["states"][:, -1], self.decoder.upper_bound),
                neglogcdf(-error),
                loss
            )

        loss = masked_average(
            self._remove_overlap(loss),
            mask=self._remove_overlap(latent_mask)
        )
        self.log(
            f'{prefix}/loss', loss,
            batch_size=in_tensor.size(0),
            prog_bar=True, sync_dist=sync_dist,
        )
        return {
            "loss": loss,
            "sampled_time": sampled_time,
            "residual": residual,
            "prediction": prediction,
            "velocity": velocity,
            "encoded": encoded,
            "noise": noise,
            "noised_residual": noised_residual,
            "latent_mesh": latent_mesh,
            "latent_mask": latent_mask
        }

    def estimate_auxiliary_losses(
            self,
            batch: Dict[str, torch.Tensor],
            outputs: Dict[str, torch.Tensor],
            resolution: torch.Tensor,
            labels: torch.Tensor,
            prefix: str = "train",
    ) -> Any:
        # Check if scores should be synced
        sync_dist = prefix != "train"

        # Bugfix for CUDAgraph?!
        first_guess = batch["states"][:, -2].nan_to_num(0.)
        latent_bounds = self.decoder.get_latent_bounds(
            first_guess, outputs["latent_mask"]
        )

        # Estimate x out of prediction with clipping
        delta_t = 1 - outputs["sampled_time"]
        predicted_residual = outputs["noised_residual"] \
            + delta_t * outputs["prediction"]
        predicted_residual = predicted_residual.clamp(
            latent_bounds[0], latent_bounds[1]
        )

        # Predict the velocity out of the clipped residual
        clipped_prediction = (
            predicted_residual - outputs["noised_residual"]
        ) / (delta_t + self.epsilon)

        # Auxiliary loss for x with clipping
        loss_x = (outputs["residual"] - predicted_residual).pow(2)
        loss_x = masked_average(
            self._remove_overlap(loss_x),
            mask=self._remove_overlap(outputs["latent_mask"])
        )

        # Auxiliary loss for velocity with clipping
        loss_flow = (outputs["velocity"] - clipped_prediction).pow(2)
        loss_flow = masked_average(
            self._remove_overlap(loss_flow),
            mask=self._remove_overlap(outputs["latent_mask"])
        )

        # Auxiliary loss for epsilon with clipping
        predicted_eps = outputs["noised_residual"] \
            - outputs["sampled_time"] * clipped_prediction
        loss_eps = (outputs["noise"] - predicted_eps).pow(2)
        loss_eps = masked_average(
            self._remove_overlap(loss_eps),
            mask=self._remove_overlap(outputs["latent_mask"])
        )

        # Deterministic x prediction
        initial_time = torch.zeros_like(outputs["sampled_time"].view(-1, 1))
        with torch.no_grad():
            det_prediction = self.network(
                torch.cat((outputs["noise"], outputs["encoded"]), dim=1),
                mesh=outputs["latent_mesh"],
                mask=outputs["latent_mask"],
                pseudo_time=initial_time,
                labels=labels,
                resolution=resolution
            )
        det_residual = outputs["noise"] + det_prediction
        det_residual = det_residual.clamp(latent_bounds[0], latent_bounds[1])

        # Deterministic loss
        loss_det = (outputs["residual"] - det_residual).pow(2)
        loss_det = masked_average(
            self._remove_overlap(loss_det),
            mask=self._remove_overlap(outputs["latent_mask"])
        )

        # Approximate path length by estimating difference between prediction
        # at t=0 and sampled t
        path_length = (
            det_residual - outputs["noise"] - clipped_prediction
        ).pow(2)
        path_length = masked_average(
            self._remove_overlap(path_length),
            mask=self._remove_overlap(outputs["latent_mask"])
        )

        self.log_dict(
            {
                f'{prefix}/loss_flow': loss_flow,
                f'{prefix}/loss_eps': loss_eps,
                f'{prefix}/loss_x': loss_x,
                f'{prefix}/loss_det': loss_det,
                f'{prefix}/path_length': path_length,
            },
            batch_size=outputs["encoded"].size(0),
            prog_bar=False,
            sync_dist=sync_dist,
        )
        return None

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema_model.update_parameters(self.network)

    def on_train_end(self) -> None:
        torch.optim.swa_utils.update_bn(
            self.trainer.train_dataloader, self.ema_model
        )

    def _preprocess_train_step(self) -> Dict[str, Any]:
        [optimizer_net, optimizer_scale] = self.optimizers()
        optimizer_scale._on_before_step = lambda: self.trainer.profiler.start(
            "optimizer_step"
        )
        optimizer_scale._on_after_step = lambda: self.trainer.profiler.stop(
            "optimizer_step"
        )
        optimizer_net.zero_grad()
        optimizer_scale.zero_grad()
        return {"net": optimizer_net, "scale": optimizer_scale}

    def _postprocess_train_step(
            self,
            loss: torch.Tensor,
            optimizers: Dict[str, Any]
    ) -> None:
        self.manual_backward(loss)
        self.clip_gradients(
            optimizers["net"], gradient_clip_val=1.,
            gradient_clip_algorithm="norm"
        )
        optimizers["net"].step()
        if self.optimize_scale:
            optimizers["scale"].step()
        scheduler_net = self.lr_schedulers()
        scheduler_net.step()

    def training_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        optimizers = self._preprocess_train_step()

        resolution = batch.pop("resolution", 12.)
        if self.patch_generator is not None:
            batch, resolution = self.patch_generator(batch, resolution)
        if self.train_augmentation is not None:
            batch, labels = self.train_augmentation(batch)
        else:
            labels = self._get_empty_labels(batch["states"])
        outputs = self.estimate_loss(batch, resolution, labels, prefix="train")
        self._postprocess_train_step(outputs["loss"], optimizers)
        return outputs

    def validation_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
    ) -> torch.Tensor:
        resolution = batch.pop("resolution", 12.)
        if self.patch_generator is not None:
            batch, resolution = self.patch_generator(batch, resolution)
        labels = self._get_empty_labels(batch["states"])

        outputs = self.estimate_loss(
            batch, resolution, labels, prefix="val"
        )
        self.estimate_auxiliary_losses(
            batch, outputs, resolution, labels, prefix="val"
        )
        return outputs["loss"]

    def test_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
    ) -> torch.Tensor:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(
            self
    ) -> Any:
        # To get rid of unusual imports when only inference is performed.
        from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        wd_params, nowd_params = split_wd_params(self.network)
        optimizer_net = torch.optim.AdamW([
            {"params": wd_params, "weight_decay": self.weight_decay},
            {"params": nowd_params, "weight_decay": 0.0}
        ], lr=self.lr, betas=(0.9, 0.99))
        optimizer_scale = torch.optim.Adam(
            self.log_scale_model.parameters(), lr=self.lr, betas=(0.9, 0.99)
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer_net,
            first_cycle_steps=self.total_steps,
            max_lr=self.lr,
            min_lr=1E-6,
            warmup_steps=self.lr_warmup,
        )
        return [
            optimizer_net, optimizer_scale
        ], [{"scheduler": scheduler, "interval": "step"}]
