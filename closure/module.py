"""
module.py — ClosureLitModule: PyTorch Lightning wrapper for closure models.

This module provides a LightningModule that wraps any ``torch.nn.Module``
(FCNN, ResNet, MLP, CNet, etc.) in Lightning's training protocol.
"""

from __future__ import annotations

__all__ = ["ClosureLitModule"]

import torch
import lightning as L


class ClosureLitModule(L.LightningModule):
    """Lightning module wrapping an arbitrary ``nn.Module`` for training.

    Parameters
    ----------
    network : torch.nn.Module
        The underlying neural network (e.g. FCNN, ResNet, MLP, CNet).
    criterion : str
        Name of a ``torch.nn`` loss class (e.g. ``"MSELoss"``).
    metrics : list[str] or None
        Additional ``torch.nn`` loss names to track (e.g. ``["L1Loss"]``).
    optimizer : str
        Name of a ``torch.optim`` optimizer class (e.g. ``"Adam"``).
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay (L2 regularisation).
    scheduler : str or None
        Name of a ``torch.optim.lr_scheduler`` class, or ``None``.
    scheduler_kwargs : dict or None
        Extra keyword arguments for the scheduler.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        criterion: str = "MSELoss",
        metrics: list[str] | None = None,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler: str | None = "ReduceLROnPlateau",
        scheduler_kwargs: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["network"])
        self.network = network

        # Instantiate criterion
        self.criterion = getattr(torch.nn, self.hparams.criterion)()

        # Instantiate extra metrics
        if self.hparams.metrics:
            self.metric_fns = torch.nn.ModuleList(
                [getattr(torch.nn, m)() for m in self.hparams.metrics]
            )
        else:
            self.metric_fns = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        return self.network(x)

    # ------------------------------------------------------------------
    # Training / validation / test / predict steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        features, targets = batch
        prediction = self(features)
        loss = self.criterion(prediction, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_metrics(prediction, targets, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        prediction = self(features)
        loss = self.criterion(prediction, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_metrics(prediction, targets, prefix="val")
        return loss

    def test_step(self, batch, batch_idx):
        features, targets = batch
        prediction = self(features)
        loss = self.criterion(prediction, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self._log_metrics(prediction, targets, prefix="test")
        return loss

    def predict_step(self, batch, batch_idx):
        features = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(features)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer_cls = getattr(torch.optim, self.hparams.optimizer)
        optimizer = optimizer_cls(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.scheduler is None:
            return optimizer

        scheduler_cls = getattr(torch.optim.lr_scheduler, self.hparams.scheduler)
        scheduler_kw = dict(self.hparams.scheduler_kwargs or {})

        # Backward compatibility: older configs sometimes pass scheduler name
        # inside scheduler_kwargs; remove it before scheduler init.
        scheduler_kw.pop("scheduler", None)
        scheduler = scheduler_cls(optimizer, **scheduler_kw)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log_metrics(self, prediction, targets, prefix: str):
        """Log additional metrics (e.g. L1Loss) if configured."""
        if self.metric_fns is None:
            return
        for fn in self.metric_fns:
            name = f"{prefix}_{fn.__class__.__name__}"
            self.log(name, fn(prediction, targets), on_step=False, on_epoch=True)
