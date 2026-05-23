"""Tests for closure.datamodule — ClosureDataModule."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from closure.datamodule import ClosureDataModule


_COMMON = dict(
    data_folder="/fake/data",
    norm_folder="/fake/norm",
    train_samples_file="train.csv",
    val_samples_file="val.csv",
)


class TestClosureDataModuleInit:
    def test_default_values(self):
        dm = ClosureDataModule(**_COMMON)
        assert dm.hparams["batch_size"] == 32
        assert dm.hparams["num_workers"] == 4

    def test_custom_batch_size(self):
        dm = ClosureDataModule(**_COMMON, batch_size=64)
        assert dm.hparams["batch_size"] == 64


class TestChannelResolution:
    def test_no_channel_names_leaves_none(self):
        dm = ClosureDataModule(**_COMMON)
        mock_ds = MagicMock()
        mock_ds.request_features = ["f0", "f1", "f2"]
        mock_ds.request_targets = ["t0", "t1", "t2"]
        dm._resolve_channel_indices(mock_ds)
        assert dm.feature_channels is None
        assert dm.target_channels is None

    def test_target_channel_names_resolved(self):
        dm = ClosureDataModule(**_COMMON, target_channel_names=["t1", "t2"])
        mock_ds = MagicMock()
        mock_ds.request_features = ["f0", "f1"]
        mock_ds.request_targets = ["t0", "t1", "t2"]
        dm._resolve_channel_indices(mock_ds)
        assert dm.target_channels == [1, 2]

    def test_feature_channel_names_resolved(self):
        dm = ClosureDataModule(**_COMMON, feature_channel_names=["f0", "f2"])
        mock_ds = MagicMock()
        mock_ds.request_features = ["f0", "f1", "f2"]
        mock_ds.request_targets = ["t0", "t1"]
        dm._resolve_channel_indices(mock_ds)
        assert dm.feature_channels == [0, 2]


class TestNormFolderResolution:
    def test_explicit_norm_version_dir_takes_precedence(self, tmp_path):
        explicit_dir = tmp_path / "lightning_logs" / "version_9"
        dm = ClosureDataModule(**_COMMON, norm_version_dir=str(explicit_dir))
        resolved = dm._resolve_norm_folder("/base/norm")
        assert resolved == str(explicit_dir.resolve())

    def test_trainer_log_dir_used_when_available(self, tmp_path):
        dm = ClosureDataModule(**_COMMON)
        trainer_log_dir = tmp_path / "lightning_logs" / "version_3"
        # Production case: CLI auto-inferred norm_folder == default_root_dir,
        # so the per-version trainer.log_dir takes precedence.
        dm.trainer = MagicMock(
            log_dir=str(trainer_log_dir),
            default_root_dir="/base/norm",
        )
        resolved = dm._resolve_norm_folder("/base/norm")
        assert resolved == str(trainer_log_dir.resolve())

    def test_explicit_base_norm_folder_overrides_trainer_log_dir(self, tmp_path):
        dm = ClosureDataModule(**_COMMON)
        trainer_log_dir = tmp_path / "lightning_logs" / "version_3"
        # User explicitly set a norm_folder distinct from default_root_dir;
        # it must win over the trainer's per-version log_dir.
        explicit_norm = tmp_path / "explicit_norm"
        dm.trainer = MagicMock(
            log_dir=str(trainer_log_dir),
            default_root_dir=str(tmp_path / "outputs"),
        )
        resolved = dm._resolve_norm_folder(str(explicit_norm))
        assert resolved == str(explicit_norm.resolve())

    def test_base_norm_folder_fallback_without_version_info(self):
        dm = ClosureDataModule(**_COMMON)
        resolved = dm._resolve_norm_folder("/base/norm")
        assert resolved == "/base/norm"
