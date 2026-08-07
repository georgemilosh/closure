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
import torch.nn.functional as F
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
        lambda_gradP: float = 0.0,
        lambda_eamb: float = 0.0,
        physics_dx: float = 1.0,
        physics_dy: float = 1.0,
        physics_small: float = 1e-10,
        physics_rho_abs: bool = True,
        physics_relative_loss: bool = True,
        physics_warmup_epochs: int = 0,
        physics_ramp_epochs: int = 0,
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

        self._physics_warned_messages: set[str] = set()

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
        base_loss, gradp_loss, eamb_loss = self._compute_loss_terms(features, prediction, targets)
        physics_scale = self._physics_loss_scale()
        loss = (
            base_loss
            + physics_scale * float(self.hparams.lambda_gradP) * gradp_loss
            + physics_scale * float(self.hparams.lambda_eamb) * eamb_loss
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_base", base_loss, on_step=False, on_epoch=True)
        self.log("train_loss_gradp", gradp_loss, on_step=False, on_epoch=True)
        self.log("train_loss_eamb", eamb_loss, on_step=False, on_epoch=True)
        self.log("train_loss_physics_scale", physics_scale, on_step=False, on_epoch=True)
        self._log_metrics(prediction, targets, prefix="train")
        self._train_batches_this_epoch = batch_idx + 1
        return loss

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        prediction = self(features)
        base_loss, gradp_loss, eamb_loss = self._compute_loss_terms(features, prediction, targets)
        physics_scale = self._physics_loss_scale()
        loss = (
            base_loss
            + physics_scale * float(self.hparams.lambda_gradP) * gradp_loss
            + physics_scale * float(self.hparams.lambda_eamb) * eamb_loss
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_base", base_loss, on_step=False, on_epoch=True)
        self.log("val_loss_gradp", gradp_loss, on_step=False, on_epoch=True)
        self.log("val_loss_eamb", eamb_loss, on_step=False, on_epoch=True)
        self.log("val_loss_physics_scale", physics_scale, on_step=False, on_epoch=True)
        self._log_metrics(prediction, targets, prefix="val")
        return loss

    def test_step(self, batch, batch_idx):
        features, targets = batch
        prediction = self(features)
        loss = self._compute_base_loss(features, prediction, targets)
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
        for key in ("train_loss", "val_loss"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.6g}")
        # Log learning rate(s): include any lr-* keys logged by
        # LearningRateMonitor (covers Adam, AdamW, SGD, SWA, ...). Fall back
        # to reading the optimizer param groups directly when no monitor is
        # attached (or before it has produced a value for the first epoch).
        lr_logged = False
        for key in sorted(metrics.keys()):
            if str(key).startswith("lr-"):
                parts.append(f"{key}={metrics[key]:.6g}")
                lr_logged = True
        if not lr_logged:
            try:
                optimizers = self.trainer.optimizers
            except Exception:  # pragma: no cover
                optimizers = []
            for i, opt in enumerate(optimizers or []):
                for j, pg in enumerate(opt.param_groups):
                    name = f"lr-{type(opt).__name__}"
                    if len(optimizers) > 1:
                        name += f"[{i}]"
                    if len(opt.param_groups) > 1:
                        name += f"/pg{j}"
                    parts.append(f"{name}={pg['lr']:.6g}")
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
        default_interval = "step" if self.hparams.scheduler == "OneCycleLR" else "epoch"
        interval = scheduler_kw.pop("interval", default_interval)
        frequency = scheduler_kw.pop("frequency", 1)

        # Backward compatibility: older configs sometimes pass scheduler name
        # inside scheduler_kwargs; remove it before scheduler init.
        scheduler_kw.pop("scheduler", None)
        scheduler_kw.pop("early_stopping", None)

        # OneCycleLR needs total_steps (or epochs+steps_per_epoch). When not
        # provided we derive it from the trainer so configs can stay generic.
        if self.hparams.scheduler == "OneCycleLR":
            has_total_steps = scheduler_kw.get("total_steps") is not None
            has_epoch_split = (
                scheduler_kw.get("epochs") is not None
                and scheduler_kw.get("steps_per_epoch") is not None
            )
            if not has_total_steps and not has_epoch_split:
                trainer = getattr(self, "trainer", None)
                estimated = getattr(trainer, "estimated_stepping_batches", None)
                if estimated is None or estimated <= 0 or estimated == float("inf"):
                    raise ValueError(
                        "OneCycleLR requires total_steps. Could not infer it from "
                        "trainer.estimated_stepping_batches; set scheduler_kwargs.total_steps "
                        "or scheduler_kwargs.epochs and scheduler_kwargs.steps_per_epoch."
                    )
                scheduler_kw["total_steps"] = int(estimated)
            # Drop explicit None so the scheduler does not receive total_steps=None.
            if scheduler_kw.get("total_steps", 1) is None:
                scheduler_kw.pop("total_steps", None)

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

    def _compute_loss_terms(self, features, prediction, targets):
        """Return base and optional physics-informed loss components."""
        base_loss = self._compute_base_loss(features, prediction, targets)
        zero = prediction.new_zeros(())
        gradp_loss = zero
        eamb_loss = zero

        if float(self.hparams.lambda_gradP) > 0.0:
            gradp_loss = self._gradient_loss(prediction, targets)
        if float(self.hparams.lambda_eamb) > 0.0:
            eamb_loss = self._eamb_proxy_loss(features, prediction, targets)

        return base_loss, gradp_loss, eamb_loss

    def _compute_base_loss(self, features, prediction, targets):
        """Compute the network-specific training loss or the configured default.

        Networks may expose ``compute_training_loss`` when their natural
        training representation differs from their public forward output.  A
        field-aligned pressure model, for example, can return Cartesian tensor
        channels to callers while comparing prediction and target in its local
        frame.  Ordinary networks remain on the unchanged criterion path.
        """
        network_loss = getattr(self.network, "compute_training_loss", None)
        if callable(network_loss):
            return network_loss(features, prediction, targets, self.criterion)
        return self.criterion(prediction, targets)

    def _physics_loss_scale(self) -> float:
        """Epoch-wise multiplier for optional physics loss warmup/ramp."""
        epoch = int(getattr(self, "current_epoch", 0))
        return self._physics_loss_scale_from_epoch(
            epoch,
            int(self.hparams.physics_warmup_epochs),
            int(self.hparams.physics_ramp_epochs),
        )

    @staticmethod
    def _physics_loss_scale_from_epoch(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
        """Compute epoch-wise physics-loss multiplier."""
        warmup_epochs = max(0, int(warmup_epochs))
        ramp_epochs = max(0, int(ramp_epochs))
        if epoch < warmup_epochs:
            return 0.0
        if ramp_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))

    @staticmethod
    def _fd4_derivatives_2d(field: torch.Tensor, dx: float, dy: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D fourth-order derivatives on interior-valid cells.

        Uses the same five-point fourth-order central stencil as Menura:
        ``(-f[i+2] + 8 f[i+1] - 8 f[i-1] + f[i-2]) / (12 dX)``.
        Returns tensors cropped to interior ``[..., 2:-2, 2:-2]``.
        """
        if field.ndim < 2 or field.shape[-2] < 5 or field.shape[-1] < 5:
            raise ValueError("field must have at least 5 points in each spatial dimension")

        # Tensor layout: [..., H, W]. plasma.py convention: axis 0 (H/dim=-2) = x, axis 1 (W/dim=-1) = y.
        # dfdx -> stencil along dim=-2 (H/rows/x), hold last dim interior.
        dfdx = (
            -field[..., 4:, 2:-2]
            + 8.0 * field[..., 3:-1, 2:-2]
            - 8.0 * field[..., 1:-3, 2:-2]
            + field[..., :-4, 2:-2]
        ) / (12.0 * dx)
        # dfdy -> stencil along dim=-1 (W/cols/y), hold second-to-last dim interior.
        dfdy = (
            -field[..., 2:-2, 4:]
            + 8.0 * field[..., 2:-2, 3:-1]
            - 8.0 * field[..., 2:-2, 1:-3]
            + field[..., 2:-2, :-4]
        ) / (12.0 * dy)
        return dfdx, dfdy

    def _gradient_loss(self, prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE between predicted and target pressure gradients."""
        if prediction.ndim != 4 or targets.ndim != 4:
            self._warn_once("Skipping gradP loss: expected 4D tensors [B,C,H,W].")
            return prediction.new_zeros(())
        if prediction.shape[-2] < 5 or prediction.shape[-1] < 5:
            self._warn_once("Skipping gradP loss: need H,W >= 5 for fourth-order stencil.")
            return prediction.new_zeros(())

        dataset = self._get_physics_dataset()
        prediction_phys = self._inverse_targets_for_physics(prediction, dataset)
        targets_phys = self._inverse_targets_for_physics(targets, dataset)

        dpx, dpy = self._fd4_derivatives_2d(prediction_phys, float(self.hparams.physics_dx), float(self.hparams.physics_dy))
        dtx, dty = self._fd4_derivatives_2d(targets_phys, float(self.hparams.physics_dx), float(self.hparams.physics_dy))
        return 0.5 * (self._physics_mse(dpx, dtx) + self._physics_mse(dpy, dty))

    def _physics_mse(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Physics MSE, optionally normalized to a dimensionless relative error."""
        loss = F.mse_loss(prediction, target)
        if not bool(self.hparams.physics_relative_loss):
            return loss

        energy = 0.5 * (
            prediction.detach().square().mean()
            + target.detach().square().mean()
        )
        return loss / (energy + float(self.hparams.physics_small))

    def _get_physics_dataset(self):
        """Return a dataset carrying normalization metadata, if attached."""
        try:
            trainer = self.trainer
        except RuntimeError:
            return None

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None

        for attr in ("train_dataset", "val_dataset", "test_dataset"):
            dataset = getattr(datamodule, attr, None)
            if dataset is not None:
                return dataset
        return None

    @staticmethod
    def _inverse_targets_for_physics(tensor: torch.Tensor, dataset) -> torch.Tensor:
        """Map normalized/prescaled target channels back to physical pressure."""
        if dataset is None:
            return tensor

        channel_count = tensor.shape[1]
        out = tensor

        if bool(getattr(dataset, "scaler_targets", False)):
            mean = getattr(dataset, "targets_mean", None)
            std = getattr(dataset, "targets_std", None)
            if mean is not None and std is not None:
                mean_t = torch.as_tensor(mean[:channel_count], dtype=tensor.dtype, device=tensor.device).reshape(1, -1, 1, 1)
                std_t = torch.as_tensor(std[:channel_count], dtype=tensor.dtype, device=tensor.device).reshape(1, -1, 1, 1)
                out = out * std_t + mean_t

        prescalers = getattr(dataset, "prescaler_targets", None)
        if prescalers is None or prescalers is False:
            return out

        channels = []
        for channel in range(channel_count):
            value = out[:, channel]
            func = prescalers[channel] if channel < len(prescalers) else None
            name = getattr(func, "__name__", func if isinstance(func, str) else "")
            if func is None:
                channels.append(value)
            elif name == "log":
                channels.append(torch.exp(value))
            elif name == "arcsinh":
                channels.append(torch.sinh(value))
            else:
                raise ValueError(
                    f"Unsupported target prescaler '{name}' for physics loss. "
                    "Supported: None, log, arcsinh."
                )
        return torch.stack(channels, dim=1)

    @staticmethod
    def _inverse_feature_channel_for_physics(
        channel: torch.Tensor,
        dataset,
        channel_index: int,
    ) -> torch.Tensor:
        """Map one normalized/prescaled feature channel back to physical units."""
        if dataset is None:
            return channel

        out = channel
        if bool(getattr(dataset, "scaler_features", False)):
            mean = getattr(dataset, "features_mean", None)
            std = getattr(dataset, "features_std", None)
            if mean is not None and std is not None:
                mean_t = torch.as_tensor(mean[channel_index], dtype=channel.dtype, device=channel.device)
                std_t = torch.as_tensor(std[channel_index], dtype=channel.dtype, device=channel.device)
                out = out * std_t + mean_t

        prescalers = getattr(dataset, "prescaler_features", None)
        if prescalers is None or prescalers is False or channel_index >= len(prescalers):
            return out

        func = prescalers[channel_index]
        name = getattr(func, "__name__", func if isinstance(func, str) else "")
        if func is None:
            return out
        if name == "log":
            return torch.exp(out)
        if name == "arcsinh":
            return torch.sinh(out)
        raise ValueError(
            f"Unsupported feature prescaler '{name}' for physics loss. "
            "Supported: None, log, arcsinh."
        )

    @staticmethod
    def _compute_eamb_from_pressure(
        pressure: torch.Tensor,
        rho: torch.Tensor,
        channel_map: dict[str, int],
        dx: float,
        dy: float,
        small: float,
        rho_abs: bool,
    ) -> torch.Tensor:
        """Build E_amb proxy from pressure channels using Menura formulas.

        Sign convention note
        --------------------
        ECsim stores electron density as a **negative** number (charge-sign
        convention).  Menura's ``extract_features_kernel`` negates
        ``density_b`` before feeding it to the NN so the runtime feature is
        **positive**.  Offline ``plasma.py::get_Ohm`` achieves the same result
        by dividing by ``(-rho_e)``.  Training batches drawn from ECsim HDF5
        files therefore carry a **negative** ``rho_e`` feature.

        ``rho_abs=True`` (default) takes ``|rho|`` before forming the
        denominator, making this method convention-agnostic and consistent with
        both the Menura runtime path and the offline diagnostic.  Set
        ``rho_abs=False`` only when rho is guaranteed to be positive.
        """
        pxx = pressure[:, channel_map["Pxx"], ...]
        pxy = pressure[:, channel_map["Pxy"], ...]
        pxz = pressure[:, channel_map["Pxz"], ...]
        pyy = pressure[:, channel_map["Pyy"], ...]
        pyz = pressure[:, channel_map["Pyz"], ...]

        dpxx_dx, _ = ClosureLitModule._fd4_derivatives_2d(pxx, dx, dy)
        dpxy_dx, dpxy_dy = ClosureLitModule._fd4_derivatives_2d(pxy, dx, dy)
        _, dpyy_dy = ClosureLitModule._fd4_derivatives_2d(pyy, dx, dy)
        dpxz_dx, _ = ClosureLitModule._fd4_derivatives_2d(pxz, dx, dy)
        _, dpyz_dy = ClosureLitModule._fd4_derivatives_2d(pyz, dx, dy)

        rho_inner = rho[:, 2:-2, 2:-2]
        rho_denom = rho_inner.abs() if rho_abs else rho_inner
        rho_denom = rho_denom + small

        epx = -(dpxx_dx + dpxy_dy) / rho_denom
        epy = -(dpxy_dx + dpyy_dy) / rho_denom
        epz = -(dpxz_dx + dpyz_dy) / rho_denom
        return torch.stack([epx, epy, epz], dim=1)

    def _eamb_proxy_loss(self, features: torch.Tensor, prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE between E_amb proxies derived from predicted and target pressures."""
        if prediction.ndim != 4 or targets.ndim != 4 or features.ndim != 4:
            self._warn_once("Skipping E_amb loss: expected 4D tensors [B,C,H,W].")
            return prediction.new_zeros(())
        if prediction.shape[1] < 6 or targets.shape[1] < 6:
            self._warn_once("Skipping E_amb loss: requires at least six pressure channels.")
            return prediction.new_zeros(())
        if prediction.shape[-2] < 5 or prediction.shape[-1] < 5:
            self._warn_once("Skipping E_amb loss: need H,W >= 5 for fourth-order stencil.")
            return prediction.new_zeros(())

        channel_map, rho_index = self._resolve_physics_channel_indices(features, prediction)
        if channel_map is None or rho_index is None:
            return prediction.new_zeros(())

        dataset = self._get_physics_dataset()
        prediction_phys = self._inverse_targets_for_physics(prediction, dataset)
        targets_phys = self._inverse_targets_for_physics(targets, dataset)
        rho = self._inverse_feature_channel_for_physics(features[:, rho_index, ...], dataset, rho_index)
        e_pred = self._compute_eamb_from_pressure(
            prediction_phys,
            rho,
            channel_map,
            dx=float(self.hparams.physics_dx),
            dy=float(self.hparams.physics_dy),
            small=float(self.hparams.physics_small),
            rho_abs=bool(self.hparams.physics_rho_abs),
        )
        e_targ = self._compute_eamb_from_pressure(
            targets_phys,
            rho,
            channel_map,
            dx=float(self.hparams.physics_dx),
            dy=float(self.hparams.physics_dy),
            small=float(self.hparams.physics_small),
            rho_abs=bool(self.hparams.physics_rho_abs),
        )
        return self._physics_mse(e_pred, e_targ)

    def _resolve_physics_channel_indices(
        self,
        features: torch.Tensor,
        prediction: torch.Tensor,
    ) -> tuple[dict[str, int] | None, int | None]:
        """Infer pressure/rho channel indices from datamodule metadata or fallbacks."""
        feature_names = None
        target_names = None
        # Lightning's .trainer property raises RuntimeError when no Trainer is
        # attached (e.g. during unit tests or standalone inference). Catch it.
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None

        if datamodule is not None:
            dm_hp = getattr(datamodule, "hparams", {})
            feature_names = dm_hp.get("feature_channel_names")
            target_names = dm_hp.get("target_channel_names")
            dataset = getattr(datamodule, "train_dataset", None)
            if feature_names is None and dataset is not None:
                feature_names = getattr(dataset, "request_features", None)
            if target_names is None and dataset is not None:
                target_names = getattr(dataset, "request_targets", None)

        channel_map = None
        if target_names:
            short_to_full = {
                "Pxx": "Pxx",
                "Pxy": "Pxy",
                "Pxz": "Pxz",
                "Pyy": "Pyy",
                "Pyz": "Pyz",
            }
            lookup = {name.split("_")[0]: i for i, name in enumerate(target_names)}
            if all(key in lookup for key in short_to_full.values()):
                channel_map = {
                    "Pxx": lookup["Pxx"],
                    "Pxy": lookup["Pxy"],
                    "Pxz": lookup["Pxz"],
                    "Pyy": lookup["Pyy"],
                    "Pyz": lookup["Pyz"],
                }

        # Closure default target ordering fallback: Pxx, Pxy, Pxz, Pyy, Pyz, Pzz.
        if channel_map is None and prediction.shape[1] >= 5:
            channel_map = {"Pxx": 0, "Pxy": 1, "Pxz": 2, "Pyy": 3, "Pyz": 4}

        rho_index = None
        if feature_names:
            for i, name in enumerate(feature_names):
                if name.split("_")[0] == "rho":
                    rho_index = i
                    break

        # Closure default feature ordering fallback places rho_e first.
        if rho_index is None and features.shape[1] > 0:
            rho_index = 0

        if channel_map is None:
            self._warn_once("Skipping E_amb loss: could not infer pressure channel indices.")
        if rho_index is None:
            self._warn_once("Skipping E_amb loss: could not infer rho channel index.")

        return channel_map, rho_index

    def _warn_once(self, message: str) -> None:
        """Emit each physics warning at most once per module instance."""
        if message in self._physics_warned_messages:
            return
        self._physics_warned_messages.add(message)
        _logger.warning(message)

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
