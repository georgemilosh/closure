"""
module.py — ClosureLitModule: PyTorch Lightning wrapper for closure models.

This module provides a LightningModule that wraps any ``torch.nn.Module``
(FCNN, ResNet, MLP, CNet, etc.) in Lightning's training protocol.
"""

from __future__ import annotations

__all__ = ["ClosureLitModule"]

import logging
import os
import time
from pathlib import Path

import yaml
import torch
import lightning as L
from lightning.pytorch.utilities.model_summary import summarize

from closure.resources import (
    aggregate_gpu_stats,
    cgroup_memory_peak_gb,
    gpu_stats,
    process_tree_ram_gb,
    process_tree_unique_ram_gb,
)


_logger = logging.getLogger("closure.module")


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
        self._train_batches_this_epoch = batch_idx + 1
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

    def on_fit_start(self):
        """Emit run context and model summary to python logs."""
        self._fit_start_time = time.perf_counter()
        self._epoch_ram_gb: list[float] = []
        self._epoch_unique_ram_gb: list[float] = []
        self._epoch_gpu_util_pct: list[float] = []
        self._epoch_gpu_mem_mb: list[float] = []
        # Reset the cgroup peak counter baseline for training.
        # We record the peak at fit_start so we can compute the
        # incremental peak caused by training alone.
        self._ram_peak_gb_at_fit_start: float | None = cgroup_memory_peak_gb()
        local_rank = getattr(self.trainer, "local_rank", -1)
        global_rank = getattr(self.trainer, "global_rank", -1)
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
        _logger.info(
            "LOCAL_RANK: %s - GLOBAL_RANK: %s - CUDA_VISIBLE_DEVICES: [%s]",
            local_rank,
            global_rank,
            visible,
        )
        try:
            _logger.info("\n%s", summarize(self, max_depth=2))
        except Exception as exc:  # pragma: no cover
            _logger.warning("Failed to build model summary: %s", exc)

    def on_train_epoch_end(self):
        """Log epoch-level metrics summary to Python logs."""
        cgroup_ram_gb = process_tree_ram_gb()
        unique_ram_gb = process_tree_unique_ram_gb()
        peak_gb = cgroup_memory_peak_gb()
        gstats = aggregate_gpu_stats(gpu_stats())
        avg_gpu_util = gstats["avg_gpu_utilization_pct"]
        avg_gpu_mem = gstats["avg_gpu_memory_used_mb"]

        self._epoch_ram_gb.append(cgroup_ram_gb)
        self._epoch_unique_ram_gb.append(unique_ram_gb)
        if peak_gb is not None:
            self._ram_peak_gb_training = peak_gb
        if avg_gpu_util is not None:
            self._epoch_gpu_util_pct.append(float(avg_gpu_util))
        if avg_gpu_mem is not None:
            self._epoch_gpu_mem_mb.append(float(avg_gpu_mem))

        metrics = self.trainer.callback_metrics
        epoch = self.trainer.current_epoch
        batches = getattr(self, "_train_batches_this_epoch", "?")
        parts = [f"Epoch {epoch}"]
        parts.append(f"batches={batches}")
        for key in ("train_loss", "val_loss", "lr-Adam"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.6g}")
        parts.append(f"cgroup_ram_gb={cgroup_ram_gb:.3f}")
        parts.append(f"unique_ram_gb={unique_ram_gb:.3f}")
        if peak_gb is not None:
            parts.append(f"cgroup_ram_peak_gb={peak_gb:.3f}")
        if avg_gpu_util is not None:
            parts.append(f"avg_gpu_util={avg_gpu_util:.1f}%")
        if avg_gpu_mem is not None:
            parts.append(f"avg_gpu_mem_mb={avg_gpu_mem:.1f}")
        _logger.info(" | ".join(parts))

    def on_fit_end(self):
        """Emit explicit fit-end marker and write timings.yaml."""
        elapsed_s = None
        if hasattr(self, "_fit_start_time"):
            elapsed_s = time.perf_counter() - self._fit_start_time

        if elapsed_s is None:
            elapsed_msg = "unknown"
        else:
            mins, secs = divmod(elapsed_s, 60.0)
            hrs, mins = divmod(mins, 60.0)
            elapsed_msg = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f} ({elapsed_s:.2f}s)"

        _logger.info(
            "Trainer.fit finished at epoch=%s (max_epochs=%s). Elapsed: %s",
            self.trainer.current_epoch,
            self.trainer.max_epochs,
            elapsed_msg,
        )

        # Write timings.yaml to the logger version directory (rank 0 only).
        if getattr(self.trainer, "global_rank", 0) == 0:
            self._write_timings(elapsed_s)

    def _write_timings(self, fit_elapsed_s: float | None) -> None:
        """Persist timing information to ``timings.yaml`` next to metrics."""
        log_dir = None
        logger = self.trainer.logger
        if logger is not None and hasattr(logger, "log_dir"):
            log_dir = logger.log_dir
        if log_dir is None:
            return

        timings: dict = {}

        # Data loading time from the datamodule.
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "_data_load_time_s"):
            timings["data_loading_s"] = round(dm._data_load_time_s, 3)

        if fit_elapsed_s is not None:
            timings["training_s"] = round(fit_elapsed_s, 3)

        timings["epochs"] = self.trainer.current_epoch
        timings["devices"] = self.trainer.num_devices
        timings["num_nodes"] = self.trainer.num_nodes

        # Per-epoch averages collected from closure.log snapshots.
        if self._epoch_ram_gb:
            timings["avg_ram_gb_per_epoch"] = round(sum(self._epoch_ram_gb) / len(self._epoch_ram_gb), 3)
            timings["avg_cgroup_ram_gb_per_epoch"] = timings["avg_ram_gb_per_epoch"]
        if self._epoch_unique_ram_gb:
            timings["avg_unique_ram_gb_per_epoch"] = round(
                sum(self._epoch_unique_ram_gb) / len(self._epoch_unique_ram_gb),
                3,
            )
        peak_training = getattr(self, "_ram_peak_gb_training", None)
        baseline = getattr(self, "_ram_peak_gb_at_fit_start", None)
        if peak_training is not None:
            timings["peak_ram_gb_during_training"] = round(peak_training, 3)
            timings["peak_cgroup_ram_gb_during_training"] = timings["peak_ram_gb_during_training"]
            if baseline is not None:
                # Net peak attributable to training (excludes loading peak)
                timings["net_peak_ram_gb_training"] = round(max(0.0, peak_training - baseline), 3)
                timings["net_peak_cgroup_ram_gb_training"] = timings["net_peak_ram_gb_training"]
        if self._epoch_gpu_util_pct:
            timings["avg_gpu_utilization_pct_per_epoch"] = round(
                sum(self._epoch_gpu_util_pct) / len(self._epoch_gpu_util_pct), 3
            )
        if self._epoch_gpu_mem_mb:
            timings["avg_gpu_memory_used_mb_per_epoch"] = round(
                sum(self._epoch_gpu_mem_mb) / len(self._epoch_gpu_mem_mb), 3
            )

        # Data-loading phase averages collected in the datamodule.
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            loading_ram = getattr(dm, "_loading_ram_snapshots_gb", None)
            loading_unique_ram = getattr(dm, "_loading_unique_ram_snapshots_gb", None)
            loading_ram_peak = getattr(dm, "_loading_ram_peak_gb", None)
            loading_gpu_util = getattr(dm, "_loading_gpu_util_snapshots_pct", None)
            loading_gpu_mem = getattr(dm, "_loading_gpu_mem_snapshots_mb", None)
            if loading_ram:
                timings["avg_ram_gb_during_loading"] = round(sum(loading_ram) / len(loading_ram), 3)
                timings["avg_cgroup_ram_gb_during_loading"] = timings["avg_ram_gb_during_loading"]
            if loading_unique_ram:
                timings["avg_unique_ram_gb_during_loading"] = round(
                    sum(loading_unique_ram) / len(loading_unique_ram),
                    3,
                )
            if loading_ram_peak is not None:
                timings["peak_ram_gb_during_loading"] = round(loading_ram_peak, 3)
                timings["peak_cgroup_ram_gb_during_loading"] = timings["peak_ram_gb_during_loading"]
            if loading_gpu_util:
                timings["avg_gpu_utilization_pct_during_loading"] = round(
                    sum(loading_gpu_util) / len(loading_gpu_util),
                    3,
                )
            if loading_gpu_mem:
                timings["avg_gpu_memory_used_mb_during_loading"] = round(
                    sum(loading_gpu_mem) / len(loading_gpu_mem),
                    3,
                )

        out = Path(log_dir) / "timings.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.safe_dump(timings, f, default_flow_style=False)
        _logger.info("Timings written to %s", out)

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
        scheduler_kw.pop("early_stopping", None)
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
