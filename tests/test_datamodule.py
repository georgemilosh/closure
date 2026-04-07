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
