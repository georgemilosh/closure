"""
callbacks.py — Custom Lightning callbacks for closure.

Provides monitoring callbacks that replace manual logging
from the old PyNet training loop.
"""

from __future__ import annotations

__all__ = ["MemoryMonitorCallback", "TimingCallback"]

import time

import lightning as L

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


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
