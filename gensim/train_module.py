#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Dict, Tuple, Optional, Any

# External modules
import torch
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Internal modules
from .embedding import LogScaleModel
from .utils import (
    remove_overlap, get_empty_labels, generate_noise, get_latent_states,
    masked_average, neglogpdf, neglogcdf, split_wd_params,
    sample_uniform_time
)


main_logger = logging.getLogger(__name__)


class GenSIMTrainModule(pl.LightningModule):
    _LABELS_DIMS: int = 3

    def __init__(
            self,
            network: OmegaConf,
            encoder: OmegaConf,
            decoder: OmegaConf,
            lr: float = 1E-4,
            lr_warmup: int = 5000,
            total_steps: int = 250000,
            weight_decay: float = 1E-3,
            ema_rate: float = 0.999,
            overlap_size: Tuple[int, int] = (8, 8),
            train_with_overlap: bool = True,
            censoring: bool = True,
            optimize_scale: bool = True,
            epsilon: float = 1E-5,
            patch_generator: Optional[OmegaConf] = None,
            train_augmentation: Optional[OmegaConf] = None,
    ):
        super().__init__()

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
        self.train_with_overlap = train_with_overlap
        self.overlap_slices = (
            slice(overlap_size[0], -overlap_size[0]),
            slice(overlap_size[1], -overlap_size[1]),
        )
        self.censoring = censoring

        # Training parameters
        self.ema_rate = ema_rate
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.optimize_scale = optimize_scale

        self.patch_generator = instantiate(patch_generator)
        self.train_augmentation = instantiate(train_augmentation)
        self.train_time_sampler = sample_uniform_time

        # If needed for divison
        self.epsilon = epsilon

        # To enable the optimization of log scale
        self.automatic_optimization = False

        # To enable training continuation from partial checkpoint
        self.strict_loading = False

        # To save all given parameters
        self.save_hyperparameters()

    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            pseudo_time: torch.Tensor,
            labels: torch.Tensor,
            resolution: torch.Tensor
    ) -> torch.Tensor:
        return self.network(
            in_tensor, mesh=mesh, mask=mask, pseudo_time=pseudo_time,
            labels=labels, resolution=resolution
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
        encoded, latent_mesh, latent_mask = get_latent_states(
            batch["states"][:, :-1], batch["forcings"],
            batch["mesh"], batch["mask"], batch["degree_days"], self.encoder
        )

        # Get linear interpolant
        residual = self.decoder.to_latent(
            batch["states"][:, -1], batch["states"][:, -2],
            batch["mask"]
        )
        noise = generate_noise(residual, latent_mask)
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
            remove_overlap(loss, self.train_with_overlap, self.overlap_slices),
            mask=remove_overlap(
                latent_mask, self.train_with_overlap, self.overlap_slices
            )
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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema_model.update_parameters(self.network)

    def on_train_end(self) -> None:
        torch.optim.swa_utils.update_bn(
            self.trainer.train_dataloader, self.ema_model
        )

    def training_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        resolution = batch.pop("resolution", 12.)
        if self.patch_generator is not None:
            batch, resolution = self.patch_generator(batch, resolution)
        if self.train_augmentation is not None:
            batch, labels = self.train_augmentation(batch)
        else:
            labels = get_empty_labels(batch["states"], self._LABELS_DIMS)
        
        # Separate parameters for weight decay
        decay_params, no_decay_params = split_wd_params(self.network)
        scale_params = list(self.log_scale_model.parameters())

        # Optimizers
        optimizer_net = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(0.9, 0.95),
        )
        optimizer_scale = torch.optim.AdamW(
            scale_params,
            lr=self.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

        # Zero gradients
        optimizer_net.zero_grad()
        optimizer_scale.zero_grad()

        # Forward pass
        outputs = self.estimate_loss(batch, resolution, labels, prefix="train")
        
        # Backward pass
        self.manual_backward(outputs["loss"])
        
        # Gradient clipping
        self.clip_gradients(
            optimizer_net, gradient_clip_val=1.,
            gradient_clip_algorithm="norm"
        )
        
        # Optimizer steps
        optimizer_net.step()
        if self.optimize_scale:
            optimizer_scale.step()
        
        # Scheduler step
        from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        scheduler_net = CosineAnnealingWarmupRestarts(
            optimizer_net,
            first_cycle_steps=self.total_steps,
            warmup_steps=self.lr_warmup,
            max_lr=self.lr,
            min_lr=self.lr * 0.1,
        )
        scheduler_net.step()
        
        return outputs

    def validation_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
    ) -> torch.Tensor:
        resolution = batch.pop("resolution", 12.)
        if self.patch_generator is not None:
            batch, resolution = self.patch_generator(batch, resolution)
        labels = get_empty_labels(batch["states"], self._LABELS_DIMS)

        outputs = self.estimate_loss(
            batch, resolution, labels, prefix="val"
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
