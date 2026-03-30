"""Tests for closure.trainers — Trainer instantiation, setup_device, create_datasets, save_results."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from closure.config import TrainerConfig
from closure.trainers import Trainer, setup_device, create_datasets, save_results


# ---------------------------------------------------------------------------
# setup_device
# ---------------------------------------------------------------------------
class TestSetupDevice:
    def test_cpu_when_no_cuda(self):
        with patch("closure.trainers.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device = torch.device
            cfg = TrainerConfig(device=None)
            dev = setup_device(cfg)
            assert dev == torch.device("cpu")

    def test_explicit_device(self):
        cfg = TrainerConfig(device="cpu")
        dev = setup_device(cfg)
        assert dev == torch.device("cpu")


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------
class TestSaveResults:
    def test_saves_model_and_loss(self, tmp_path):
        # Create a simple model
        model_mock = MagicMock()
        model_mock.model.state_dict.return_value = {"weight": torch.tensor([1.0, 2.0])}

        loss_dict = {"train_loss": [0.5], "val_loss": [0.3], "time": 10.0}
        save_results(model_mock, loss_dict, str(tmp_path))

        assert os.path.exists(tmp_path / "model.pth")
        assert os.path.exists(tmp_path / "loss_dict.pkl")

        with open(tmp_path / "loss_dict.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert loaded["train_loss"] == [0.5]

    def test_creates_directory(self, tmp_path):
        subdir = tmp_path / "sub" / "dir"
        model_mock = MagicMock()
        model_mock.model.state_dict.return_value = {}
        save_results(model_mock, {}, str(subdir))
        assert os.path.isdir(subdir)


# ---------------------------------------------------------------------------
# create_datasets
# ---------------------------------------------------------------------------
class TestCreateDatasets:
    def _make_config(self, tmp_path, mode_test=False):
        csv_path = os.path.join(str(tmp_path), "split.csv")
        pd.DataFrame({"filenames": ["f1.h5", "f2.h5"]}).to_csv(csv_path, index=False)

        return TrainerConfig(
            work_dir=str(tmp_path),
            mode_test=mode_test,
            dataset_kwargs={
                "data_folder": str(tmp_path),
                "train_sample": csv_path,
                "val_sample": csv_path,
                "test_sample": csv_path,
                "flatten": True,
                "scaler_features": True,
                "scaler_targets": True,
                "read_features_targets_kwargs": {
                    "request_features": ["f0"],
                    "request_targets": ["t0"],
                },
            },
        )

    def test_returns_three_datasets(self, tmp_path):
        cfg = self._make_config(tmp_path)
        features = np.random.randn(2, 4, 4, 1).astype(np.float64)
        targets = np.random.randn(2, 4, 4, 1).astype(np.float64)

        with patch("closure.datasets.rp") as mock_rp:
            mock_rp.read_features_targets.return_value = (features, targets)
            train_ds, val_ds, test_ds = create_datasets(cfg)

        assert train_ds is not None
        assert val_ds is not None
        assert test_ds is not None

    def test_mode_test_skips_train_val(self, tmp_path):
        # First create train datasets to generate normalization params
        cfg_train = self._make_config(tmp_path, mode_test=False)
        features = np.random.randn(2, 4, 4, 1).astype(np.float64)
        targets = np.random.randn(2, 4, 4, 1).astype(np.float64)

        with patch("closure.datasets.rp") as mock_rp:
            mock_rp.read_features_targets.return_value = (features, targets)
            create_datasets(cfg_train)

        # Now test mode_test=True (norm params exist on disk)
        cfg = self._make_config(tmp_path, mode_test=True)
        with patch("closure.datasets.rp") as mock_rp:
            mock_rp.read_features_targets.return_value = (features, targets)
            train_ds, val_ds, test_ds = create_datasets(cfg)

        assert train_ds is None
        assert val_ds is None
        assert test_ds is not None


# ---------------------------------------------------------------------------
# TrainerConfig integration with Trainer constructor
# ---------------------------------------------------------------------------
class TestTrainerConfigAcceptance:
    def test_trainer_accepts_config(self, tmp_path):
        """Verify Trainer.__init__ accepts a TrainerConfig without error."""
        csv_path = os.path.join(str(tmp_path), "split.csv")
        pd.DataFrame({"filenames": ["f1.h5", "f2.h5"]}).to_csv(csv_path, index=False)

        dk = {
            "data_folder": str(tmp_path),
            "train_sample": csv_path,
            "val_sample": csv_path,
            "test_sample": csv_path,
            "flatten": True,
            "scaler_features": True,
            "scaler_targets": True,
            "read_features_targets_kwargs": {
                "request_features": ["f0"],
                "request_targets": ["t0"],
            },
        }

        cfg = TrainerConfig(
            work_dir=str(tmp_path),
            mode_test=False,
            dataset_kwargs=dk,
            load_data_kwargs={
                "train_loader_kwargs": {"batch_size": 2},
                "val_loader_kwargs": {"batch_size": 2},
            },
            model_kwargs={
                "model": "MLP",
                "feature_dims": [1, 16, 1],
                "optimizer_kwargs": {
                    "optimizer": "Adam",
                    "criterion": "MSELoss",
                    "lr": 1e-3,
                },
                "scheduler_kwargs": {
                    "epochs": 1,
                    "scheduler": "StepLR",
                    "step_size": 1,
                },
            },
        )

        features = np.random.randn(2, 4, 4, 1).astype(np.float64)
        targets = np.random.randn(2, 4, 4, 1).astype(np.float64)

        with patch("closure.datasets.rp") as mock_rp:
            mock_rp.read_features_targets.return_value = (features, targets)
            trainer = Trainer(config=cfg)

        assert trainer.work_dir == str(tmp_path)
        assert trainer.test_dataset is not None
