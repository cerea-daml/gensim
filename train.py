#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import hydra
from omegaconf import DictConfig, OmegaConf

# Internal modules


main_logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path='.', config_name='config'
)
def train_task(cfg: DictConfig, network_name: str = "surrogate") -> None:
    # Import within main loop to speed up training on Jean Zay
    import wandb
    from hydra.utils import instantiate
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
    from wandb.sdk.service.service import ServiceStartTimeoutError

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    torch.use_deterministic_algorithms(mode=False, warn_only=True)
    torch.set_float32_matmul_precision("medium")
    torch._dynamo.config.suppress_errors = True
    
    main_logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: pl.LightningDataModule = instantiate(cfg.data)
    data_module.setup("fit")

    main_logger.info(f"Instantiating model <{cfg[network_name]._target_}")
    model: pl.LightningModule = instantiate(
        cfg[network_name], _recursive_=False
    )
    model.hparams["batch_size"] = cfg.batch_size
    if cfg.compile:
        # Compile for training
        model.network.compile(mode="reduce-overhead")

    if OmegaConf.select(cfg, "callbacks") is not None:
        callbacks = []
        for _, callback_cfg in cfg.callbacks.items():
            curr_callback: pl.callbacks.Callback = instantiate(callback_cfg)
            callbacks.append(curr_callback)
    else:
        callbacks = None

    training_logger = None
    if OmegaConf.select(cfg, "logger") is not None:
        try:
            training_logger = instantiate(cfg.logger)
        except ServiceStartTimeoutError:
            # Needed for restart on Jean Zay
            # Set wandb to offline
            training_logger = instantiate(
                cfg.logger, mode="offline", offline=True, log_model=False
            )

    if isinstance(training_logger, WandbLogger):
        main_logger.info("Watch gradients and parameters of model")
        training_logger.watch(model, log="all", log_freq=100)

    main_logger.info("Instantiating trainer")
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=training_logger
    )

    main_logger.info("Starting training")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)
    main_logger.info("Training finished")
    wandb.finish()


if __name__ == '__main__':
    train_task()
