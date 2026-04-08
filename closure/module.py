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

        # Instantiate extra metrics.
        # Each entry in ``metrics`` may be:
        #   - a plain string name resolved first in ``torch.nn``, then in
        #     ``torchmetrics`` (e.g. ``"L1Loss"``, ``"R2Score"``).
        #   - a dict ``{"name": "R2Score", "num_outputs": 6, ...}`` for
        #     metric classes that require constructor arguments (common for
        #     multi-output torchmetrics metrics such as R2Score).
        if self.hparams.metrics:
            self.metric_fns = torch.nn.ModuleList(
                [self._build_metric(m) for m in self.hparams.metrics]
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

        scheduler_kw = dict(self.hparams.scheduler_kwargs or {})
        interval = scheduler_kw.pop("interval", "epoch")
        frequency = scheduler_kw.pop("frequency", 1)

        # Backward compatibility: older configs sometimes pass scheduler name
        # inside scheduler_kwargs; remove it before scheduler init.
        scheduler_kw.pop("scheduler", None)
        scheduler = self._build_scheduler(
            optimizer=optimizer,
            scheduler_name=self.hparams.scheduler,
            scheduler_kwargs=scheduler_kw,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
        }
        if self.hparams.scheduler == "ReduceLROnPlateau":
            scheduler_config["monitor"] = "val_loss"

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_name: str,
        scheduler_kwargs: dict,
    ):
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)

        if scheduler_name in {"SequentialLR", "ChainedScheduler"}:
            scheduler_specs = scheduler_kwargs.pop("schedulers", None)
            if not scheduler_specs:
                raise ValueError(
                    f"{scheduler_name} requires a non-empty 'schedulers' list in scheduler_kwargs."
                )
            schedulers = []
            for spec in scheduler_specs:
                child_name = spec["name"]
                child_kwargs = dict(spec.get("kwargs", {}))
                schedulers.append(
                    self._build_scheduler(
                        optimizer=optimizer,
                        scheduler_name=child_name,
                        scheduler_kwargs=child_kwargs,
                    )
                )
            return scheduler_cls(optimizer, schedulers=schedulers, **scheduler_kwargs)

        return scheduler_cls(optimizer, **scheduler_kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metric(spec) -> torch.nn.Module:
        """Instantiate a metric from a string name or a dict spec.

        String form (no constructor args, resolved via torch.nn then torchmetrics)::

            "L1Loss"
            "R2Score"      # resolved from torchmetrics

        Dict form (for metrics that require constructor arguments)::

            {"name": "R2Score", "num_outputs": 6}
            {"name": "MeanAbsoluteError"}
        """
        if isinstance(spec, dict):
            spec = dict(spec)  # copy so we don't mutate hparams
            name = spec.pop("name")
            kwargs = spec
        else:
            name = spec
            kwargs = {}

        if hasattr(torch.nn, name):
            return getattr(torch.nn, name)(**kwargs)

        try:
            import torchmetrics  # optional dependency
            if hasattr(torchmetrics, name):
                return getattr(torchmetrics, name)(**kwargs)
        except ImportError:
            pass

        raise ValueError(
            f"Metric '{name}' not found in torch.nn or torchmetrics. "
            "Install torchmetrics for metrics such as R2Score, "
            "MeanAbsoluteError, etc."
        )

    def _log_metrics(self, prediction, targets, prefix: str):
        """Log additional metrics (e.g. L1Loss, R2Score) if configured."""
        if self.metric_fns is None:
            return
        for fn in self.metric_fns:
            name = f"{prefix}_{fn.__class__.__name__}"
            self.log(name, fn(prediction, targets), on_step=False, on_epoch=True)
