"""Tests for closure.run_loader — RunLoader convenience class."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from closure.run_loader import RunLoader, _instantiate_network


# ---------------------------------------------------------------------------
# _instantiate_network
# ---------------------------------------------------------------------------
class TestInstantiateNetwork:
    def test_mlp(self):
        cfg = {
            "class_path": "closure.models.MLP",
            "init_args": {
                "feature_dims": [10, 32, 6],
                "activations": ["ReLU", None],
            },
        }
        net = _instantiate_network(cfg)
        assert hasattr(net, "forward")
        out = net(torch.randn(2, 10))
        assert out.shape == (2, 6)

    def test_missing_init_args_ok(self):
        """init_args defaults to empty dict."""
        cfg = {"class_path": "torch.nn.Flatten"}
        net = _instantiate_network(cfg)
        assert isinstance(net, torch.nn.Flatten)


# ---------------------------------------------------------------------------
# RunLoader._find_best_ckpt
# ---------------------------------------------------------------------------
class TestFindBestCkpt:
    def test_finds_best_by_val_loss(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "best-epoch=5-val_loss=0.0123.ckpt").touch()
        (ckpt_dir / "best-epoch=10-val_loss=0.0098.ckpt").touch()
        (ckpt_dir / "last.ckpt").touch()
        found = RunLoader._find_best_ckpt(ckpt_dir)
        assert "0.0098" in found.name

    def test_falls_back_to_last(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "last.ckpt").touch()
        found = RunLoader._find_best_ckpt(ckpt_dir)
        assert found.name == "last.ckpt"

    def test_falls_back_to_first(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "epoch=3.ckpt").touch()
        found = RunLoader._find_best_ckpt(ckpt_dir)
        assert found.name == "epoch=3.ckpt"

    def test_raises_on_empty(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            RunLoader._find_best_ckpt(ckpt_dir)


# ---------------------------------------------------------------------------
# RunLoader construction
# ---------------------------------------------------------------------------
class TestRunLoaderInit:
    def test_repr(self):
        model = MagicMock()
        dm = MagicMock()
        dm.test_dataset = MagicMock()
        dm.test_dataset.request_targets = ["Pxx_e", "Pyy_e"]
        dm.test_dataset.__len__ = MagicMock(return_value=100)
        dm.val_dataset = None
        dm.train_dataset = None

        loader = RunLoader(model=model, datamodule=dm, config={})
        r = repr(loader)
        assert "Pxx_e" in r
        assert "100" in r

    def test_dataset_property_prefers_test(self):
        dm = MagicMock()
        dm.test_dataset = MagicMock()
        dm.val_dataset = MagicMock()
        dm.train_dataset = MagicMock()
        loader = RunLoader(model=MagicMock(), datamodule=dm, config={})
        assert loader.dataset is dm.test_dataset

    def test_dataset_fallback_val(self):
        dm = MagicMock()
        dm.test_dataset = None
        dm.val_dataset = MagicMock()
        loader = RunLoader(model=MagicMock(), datamodule=dm, config={})
        assert loader.dataset is dm.val_dataset

    def test_dataset_raises_when_none(self):
        dm = MagicMock()
        dm.test_dataset = None
        dm.val_dataset = None
        dm.train_dataset = None
        loader = RunLoader(model=MagicMock(), datamodule=dm, config={})
        with pytest.raises(RuntimeError, match="No dataset"):
            _ = loader.dataset


# ---------------------------------------------------------------------------
# RunLoader.from_version_dir
# ---------------------------------------------------------------------------
class TestFromVersionDir:
    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config not found"):
            RunLoader.from_version_dir(tmp_path)

    @patch("closure.run_loader.ClosureLitModule.load_from_checkpoint")
    @patch("closure.run_loader.ClosureDataModule")
    def test_loads_from_version_dir(self, MockDM, mock_load_ckpt, tmp_path):
        # Set up directory structure
        config = {
            "data": {
                "data_folder": "/fake/data",
                "norm_folder": "/fake/norm",
                "train_samples_file": "train.csv",
                "val_samples_file": "val.csv",
            },
            "model": {
                "network": {
                    "class_path": "closure.models.MLP",
                    "init_args": {
                        "feature_dims": [10, 6],
                        "activations": [None],
                    },
                },
                "criterion": "MSELoss",
            },
        }
        (tmp_path / "config.yaml").write_text(yaml.dump(config))
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "best-epoch=1-val_loss=0.01.ckpt").touch()

        # Mock the heavy operations
        mock_module = MagicMock()
        mock_load_ckpt.return_value = mock_module

        mock_dm_instance = MagicMock()
        mock_dm_instance.target_channels = None
        MockDM.return_value = mock_dm_instance

        loader = RunLoader.from_version_dir(tmp_path)
        assert loader.model is mock_module
        assert loader.datamodule is mock_dm_instance
        mock_dm_instance.setup.assert_called_once_with("test")


# ---------------------------------------------------------------------------
# RunLoader.predict / metrics / loss
# ---------------------------------------------------------------------------
class TestRunLoaderMethods:
    @pytest.fixture
    def mock_loader(self):
        """Create a RunLoader with mocked internals."""
        model = MagicMock()
        dm = MagicMock()
        ds = MagicMock()
        ds.request_targets = ["t0", "t1"]
        ds.prescaler_targets = [None, None]
        ds.features = torch.randn(10, 4)
        ds.targets = torch.randn(10, 2)
        ds.targets_std = torch.ones(2)
        ds.targets_mean = torch.zeros(2)
        ds.targets_shape = (10, 2)
        ds.flatten = True
        dm.test_dataset = ds
        dm.target_channels = None

        config = {
            "data": {
                "data_folder": "/fake/data",
                "read_features_targets_kwargs": {"choose_x": [0, 10]},
            }
        }
        return RunLoader(model=model, datamodule=dm, config=config)

    @patch("closure.run_loader.ClosureDataModule._resolve_path", return_value="/resolved/data")
    def test_data_folder_property(self, mock_resolve, mock_loader):
        assert mock_loader.data_folder == "/resolved/data"

    def test_target_channels_property(self, mock_loader):
        assert mock_loader.target_channels is None

    def test_read_features_targets_kwargs(self, mock_loader):
        assert mock_loader.read_features_targets_kwargs == {"choose_x": [0, 10]}

    @patch("closure.evaluation.transform_targets")
    def test_predict_calls_transform_targets(self, mock_tt, mock_loader):
        mock_tt.return_value = (np.zeros((10, 2)), np.ones((10, 2)))
        gt, pred = mock_loader.predict()
        mock_tt.assert_called_once()
        assert gt.shape == (10, 2)
        assert pred.shape == (10, 2)

    @patch("closure.evaluation.evaluate_regression_metrics")
    @patch("closure.evaluation.transform_targets")
    def test_metrics_calls_evaluate(self, mock_tt, mock_erm, mock_loader):
        mock_tt.return_value = (np.zeros((10, 2)), np.ones((10, 2)))
        mock_erm.return_value = "metrics_df"
        result = mock_loader.metrics()
        assert result == "metrics_df"
        mock_erm.assert_called_once()

    @patch("closure.visualization.plot_pred_targets")
    @patch("closure.evaluation.transform_targets")
    def test_plot_calls_plot_pred_targets(self, mock_tt, mock_plot, mock_loader):
        mock_tt.return_value = (np.zeros((10, 2)), np.ones((10, 2)))
        mock_loader.plot("t0")
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args
        assert call_kwargs[1]["target_channels"] is None


# ---------------------------------------------------------------------------
# Training history methods
# ---------------------------------------------------------------------------
def _make_metrics_csv(version_dir: Path, rows: list[dict] | None = None):
    """Write a minimal metrics.csv inside *version_dir*."""
    if rows is None:
        # Mimic real CSVLogger output: lr rows have epoch=NaN
        rows = [
            {"epoch": float("nan"), "train_loss": float("nan"), "val_loss": float("nan"), "lr-Adam": 1e-3},
            {"epoch": 0, "train_loss": 0.5, "val_loss": float("nan"), "lr-Adam": float("nan")},
            {"epoch": 0, "train_loss": float("nan"), "val_loss": 0.4, "lr-Adam": float("nan")},
            {"epoch": float("nan"), "train_loss": float("nan"), "val_loss": float("nan"), "lr-Adam": 8e-4},
            {"epoch": 1, "train_loss": 0.3, "val_loss": float("nan"), "lr-Adam": float("nan")},
            {"epoch": 1, "train_loss": float("nan"), "val_loss": 0.2, "lr-Adam": float("nan")},
            {"epoch": float("nan"), "train_loss": float("nan"), "val_loss": float("nan"), "lr-Adam": 5e-4},
            {"epoch": 2, "train_loss": 0.1, "val_loss": float("nan"), "lr-Adam": float("nan")},
            {"epoch": 2, "train_loss": float("nan"), "val_loss": 0.15, "lr-Adam": float("nan")},
        ]
    df = pd.DataFrame(rows)
    version_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(version_dir / "metrics.csv", index=False)
    return version_dir


class TestMetricsCsvPath:
    def test_raises_without_version_dir(self):
        loader = RunLoader(model=MagicMock(), datamodule=MagicMock(), config={})
        with pytest.raises(FileNotFoundError, match="No metrics.csv"):
            loader._metrics_csv_path()

    def test_raises_when_csv_missing(self, tmp_path):
        loader = RunLoader(model=MagicMock(), datamodule=MagicMock(), config={},
                           version_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="No metrics.csv"):
            loader._metrics_csv_path()

    def test_returns_path_when_exists(self, tmp_path):
        (tmp_path / "metrics.csv").write_text("epoch,train_loss\n0,0.5\n")
        loader = RunLoader(model=MagicMock(), datamodule=MagicMock(), config={},
                           version_dir=tmp_path)
        assert loader._metrics_csv_path() == tmp_path / "metrics.csv"


class TestHistory:
    def test_returns_per_epoch_frame(self, tmp_path):
        _make_metrics_csv(tmp_path)
        loader = RunLoader(model=MagicMock(), datamodule=MagicMock(), config={},
                           version_dir=tmp_path)
        h = loader.history()
        assert list(h.columns) == ["epoch", "train_loss", "val_loss", "lr-Adam"]
        assert len(h) == 3
        assert list(h["epoch"]) == [0, 1, 2]
        # LR values should be filled despite epoch=NaN in raw CSV rows
        assert h["lr-Adam"].notna().all()

    def test_no_lr_column(self, tmp_path):
        rows = [
            {"epoch": 0, "train_loss": 0.5, "val_loss": float("nan")},
            {"epoch": 0, "train_loss": float("nan"), "val_loss": 0.4},
        ]
        _make_metrics_csv(tmp_path, rows)
        loader = RunLoader(model=MagicMock(), datamodule=MagicMock(), config={},
                           version_dir=tmp_path)
        h = loader.history()
        assert "lr-Adam" not in h.columns
        assert len(h) == 1


class TestBestEpoch:
    def test_returns_best_and_final(self, tmp_path):
        _make_metrics_csv(tmp_path)
        loader = RunLoader(model=MagicMock(), datamodule=MagicMock(), config={},
                           version_dir=tmp_path)
        info = loader.best_epoch()
        assert info["best_epoch"] == 2
        assert info["best_val_loss"] == pytest.approx(0.15)
        assert info["final_epoch"] == 2


class TestPlotHistory:
    @patch("matplotlib.pyplot.show")
    def test_plot_history_does_not_error(self, mock_show, tmp_path):
        _make_metrics_csv(tmp_path)
        loader = RunLoader(model=MagicMock(), datamodule=MagicMock(), config={},
                           version_dir=tmp_path)
        loader.plot_history()
        mock_show.assert_called_once()


class TestCompareVersions:
    def test_compares_multiple_versions(self, tmp_path):
        _make_metrics_csv(tmp_path / "version_0", rows=[
            {"epoch": 0, "train_loss": 0.5, "val_loss": 0.4},
        ])
        _make_metrics_csv(tmp_path / "version_1", rows=[
            {"epoch": 0, "train_loss": 0.3, "val_loss": 0.1},
        ])
        result = RunLoader.compare_versions(tmp_path)
        assert len(result) == 2
        assert result.iloc[0]["version"] == "version_1"
        assert result.iloc[0]["best_val_loss"] == pytest.approx(0.1)

    def test_skips_missing_csv(self, tmp_path):
        _make_metrics_csv(tmp_path / "version_0")
        (tmp_path / "version_1").mkdir()  # no metrics.csv
        result = RunLoader.compare_versions(tmp_path)
        assert len(result) == 1

    def test_empty_log_root(self, tmp_path):
        result = RunLoader.compare_versions(tmp_path)
        assert len(result) == 0
