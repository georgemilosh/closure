"""Tests for closure.callbacks — monitoring and checkpoint export callbacks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from closure.callbacks import (
    MemoryMonitorCallback,
    TimingCallback,
    TorchScriptCheckpointExportCallback,
)


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


class TestTorchScriptCheckpointExportCallback:
    def test_exports_torchscript_for_saved_checkpoint(self, tmp_path: Path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "epoch=0-step=1.ckpt"

        reference_network = torch.nn.Sequential(torch.nn.Linear(2, 1))
        with torch.no_grad():
            reference_network[0].weight.copy_(torch.tensor([[2.0, -1.0]]))
            reference_network[0].bias.copy_(torch.tensor([0.5]))

        torch.save(
            {
                "state_dict": {
                    f"network.{name}": tensor
                    for name, tensor in reference_network.state_dict().items()
                }
            },
            ckpt_path,
        )

        trainer = MagicMock()
        trainer.global_rank = 0
        trainer.callbacks = [ModelCheckpoint(dirpath=str(ckpt_dir), filename="best")]

        pl_module = MagicMock()
        pl_module.network = torch.nn.Sequential(torch.nn.Linear(2, 1))

        callback = TorchScriptCheckpointExportCallback()
        callback.on_validation_end(trainer, pl_module)

        pt_path = ckpt_path.with_suffix(".pt")
        assert pt_path.exists()

        scripted_model = torch.jit.load(str(pt_path))
        sample = torch.tensor([[3.0, 4.0]])
        expected = reference_network(sample)
        actual = scripted_model(sample)
        torch.testing.assert_close(actual, expected)

    def test_removes_stale_pt_when_checkpoint_is_deleted(self, tmp_path: Path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        stale_ckpt = ckpt_dir / "epoch=0-step=1.ckpt"
        stale_pt = stale_ckpt.with_suffix(".pt")
        stale_pt.write_bytes(b"stale")

        trainer = MagicMock()
        trainer.global_rank = 0
        trainer.callbacks = [ModelCheckpoint(dirpath=str(ckpt_dir), filename="best")]

        pl_module = MagicMock()
        pl_module.network = torch.nn.Sequential(torch.nn.Linear(2, 1))

        callback = TorchScriptCheckpointExportCallback()
        callback._exported_mtimes[stale_ckpt] = 1

        callback.on_validation_end(trainer, pl_module)

        assert not stale_pt.exists()
