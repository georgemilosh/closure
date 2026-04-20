"""Tests for closure.callbacks — MemoryMonitorCallback, TimingCallback."""

from __future__ import annotations

from unittest.mock import MagicMock

from closure.callbacks import MemoryMonitorCallback, TimingCallback


class TestTimingCallback:
    def test_records_epoch_time(self):
        cb = TimingCallback()
        trainer = MagicMock()
        trainer.current_epoch = 0
        pl_module = MagicMock()

        cb.on_train_epoch_start(trainer, pl_module)
        assert cb._epoch_start is not None

        cb.on_train_epoch_end(trainer, pl_module)
        pl_module.log.assert_called_once()
        name, value = pl_module.log.call_args[0]
        assert name == "epoch_time_s"
        assert value >= 0.0


class TestMemoryMonitorCallback:
    def test_logs_ram(self):
        cb = MemoryMonitorCallback()
        trainer = MagicMock()
        pl_module = MagicMock()

        cb.on_train_epoch_end(trainer, pl_module)
        logged_names = [call[0][0] for call in pl_module.log.call_args_list]
        assert "ram_usage_mb" in logged_names
