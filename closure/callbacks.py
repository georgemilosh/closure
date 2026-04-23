"""
callbacks.py — Custom Lightning callbacks for closure.

Provides monitoring callbacks that replace manual logging
from the old PyNet training loop.
"""

from __future__ import annotations

__all__ = [
    "MemoryMonitorCallback",
    "TimingCallback",
    "TorchScriptCheckpointExportCallback",
]

import copy
import logging
import time
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


_logger = logging.getLogger(__name__)


class MemoryMonitorCallback(L.Callback):
    """Log RAM (and optionally GPU) memory usage at each epoch end."""

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if psutil is not None:
            proc = psutil.Process()
            ram_mb = proc.memory_info().rss / 1024**2
            pl_module.log("ram_usage_mb", ram_mb, on_step=False, on_epoch=True)

        if torch is not None and torch.cuda.is_available() and pl_module.device.type == "cuda":
            gpu_mb = torch.cuda.max_memory_allocated(pl_module.device) / 1024**2
            pl_module.log("gpu_peak_mb", gpu_mb, on_step=False, on_epoch=True)
            torch.cuda.reset_peak_memory_stats(pl_module.device)


class TimingCallback(L.Callback):
    """Log wall-clock time per epoch."""

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        elapsed = time.perf_counter() - self._epoch_start
        pl_module.log("epoch_time_s", elapsed, on_step=False, on_epoch=True)


class TorchScriptCheckpointExportCallback(L.Callback):
    """Mirror saved Lightning checkpoints to TorchScript ``.pt`` files."""

    def __init__(self) -> None:
        self._exported_mtimes: dict[Path, int] = {}

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._sync_checkpoint_exports(trainer, pl_module)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._sync_checkpoint_exports(trainer, pl_module)

    def _sync_checkpoint_exports(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if torch is None or getattr(trainer, "global_rank", 0) != 0:
            return

        for checkpoint_cb in self._model_checkpoint_callbacks(trainer):
            self._sync_model_checkpoint_dir(checkpoint_cb, pl_module)

    def _model_checkpoint_callbacks(self, trainer: L.Trainer) -> list[ModelCheckpoint]:
        return [
            callback
            for callback in getattr(trainer, "callbacks", [])
            if isinstance(callback, ModelCheckpoint)
        ]

    def _sync_model_checkpoint_dir(
        self,
        checkpoint_cb: ModelCheckpoint,
        pl_module: L.LightningModule,
    ) -> None:
        dirpath = getattr(checkpoint_cb, "dirpath", None)
        if not dirpath:
            return

        ckpt_dir = Path(dirpath)
        if not ckpt_dir.exists():
            return

        ckpt_paths = sorted(ckpt_dir.glob("*.ckpt"))
        for ckpt_path in ckpt_paths:
            mtime_ns = ckpt_path.stat().st_mtime_ns
            if self._exported_mtimes.get(ckpt_path) == mtime_ns:
                continue
            self._export_checkpoint_to_torchscript(ckpt_path, pl_module)
            self._exported_mtimes[ckpt_path] = mtime_ns

        live_ckpts = set(ckpt_paths)
        stale_ckpts = [path for path in self._exported_mtimes if path.parent == ckpt_dir and path not in live_ckpts]
        for stale_ckpt in stale_ckpts:
            self._exported_mtimes.pop(stale_ckpt, None)
            pt_path = stale_ckpt.with_suffix(".pt")
            if pt_path.exists():
                pt_path.unlink()

    def _export_checkpoint_to_torchscript(
        self,
        ckpt_path: Path,
        pl_module: L.LightningModule,
    ) -> None:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        network_state_dict = {
            key.removeprefix("network."): value.detach().cpu()
            for key, value in state_dict.items()
            if key.startswith("network.")
        }
        if not network_state_dict:
            raise ValueError(f"Checkpoint does not contain any network.* weights: {ckpt_path}")

        network = copy.deepcopy(pl_module.network).cpu().eval()
        network.load_state_dict(network_state_dict, strict=True)
        scripted_model = torch.jit.script(network)
        pt_path = ckpt_path.with_suffix(".pt")
        scripted_model.save(str(pt_path))
        _logger.info("Exported TorchScript checkpoint -> %s", pt_path)
