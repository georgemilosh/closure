"""Tests for closure.models — model instantiation, forward pass, save/load."""

from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np
import pytest
import torch

from closure.models import PyNet, FCNN, ResNet, MLP


# ---------------------------------------------------------------------------
# FCNN
# ---------------------------------------------------------------------------
class TestFCNN:
    def test_instantiation(self):
        model = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        assert isinstance(model, torch.nn.Module)

    def test_forward_shape(self):
        model = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 8, 32, 32)

    def test_with_activations(self):
        model = FCNN(channels=[3, 16, 8], kernels=[3, 3], activations=["ReLU", None])
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 8, 32, 32)

    def test_with_batchnorm(self):
        model = FCNN(channels=[3, 16, 8], kernels=[3, 3], batch_norms=[True, False])
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 8, 32, 32)


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------
class TestResNet:
    def test_instantiation(self):
        model = ResNet(channels=[3, 16, 8], kernels=[3, 3])
        assert isinstance(model, torch.nn.Module)

    def test_forward_shape(self):
        model = ResNet(channels=[3, 16, 8], kernels=[3, 3])
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 8, 32, 32)

    def test_skip_connection(self):
        model = ResNet(
            channels=[3, 16, 16, 8],
            kernels=[3, 3, 3],
            skip_connect={"2": 0},
        )
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 8, 32, 32)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
class TestMLP:
    def test_instantiation(self):
        model = MLP(feature_dims=[10, 32, 5])
        assert isinstance(model, torch.nn.Module)

    def test_forward_shape(self):
        model = MLP(feature_dims=[10, 32, 5])
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_with_activations(self):
        model = MLP(feature_dims=[10, 32, 5], activations=["ReLU", None])
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_with_dropout(self):
        model = MLP(feature_dims=[10, 32, 5], dropouts=[0.1, None])
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)


# ---------------------------------------------------------------------------
# PyNet
# ---------------------------------------------------------------------------
class TestPyNet:
    def _make_pynet(self, tmp_path=None):
        return PyNet(
            model="FCNN",
            channels=[3, 16, 8],
            kernels=[3, 3],
            optimizer_kwargs={
                "optimizer": "Adam",
                "criterion": "MSELoss",
                "lr": 1e-3,
            },
            scheduler_kwargs={
                "epochs": 2,
                "scheduler": "StepLR",
                "step_size": 1,
            },
            model_path=tmp_path,
        )

    def test_instantiation(self):
        net = self._make_pynet()
        assert net.device == torch.device("cpu")

    def test_predict(self):
        net = self._make_pynet()
        x = torch.randn(4, 3, 16, 16)
        pred = net.predict(x)
        assert pred.shape == (4, 8, 16, 16)

    def test_predict_numpy(self):
        net = self._make_pynet()
        x = np.random.randn(4, 3, 16, 16).astype(np.float32)
        pred = net.predict(x)
        assert pred.shape == (4, 8, 16, 16)

    def test_compute_loss(self):
        net = self._make_pynet()
        a = torch.randn(10)
        b = torch.randn(10)
        loss = net._compute_loss(a, b, torch.nn.MSELoss())
        assert loss.ndim == 0  # scalar

    def test_save_and_load(self, tmp_path):
        net = self._make_pynet(tmp_path=str(tmp_path))
        # Save manually
        os.makedirs(tmp_path, exist_ok=True)
        torch.save(net.model.state_dict(), tmp_path / "model.pth")
        loss_data = {"train_loss": {}, "val_loss": {}, "time": {}}
        with open(tmp_path / "loss_dict.pkl", "wb") as f:
            pickle.dump(loss_data, f)

        # Create a new PyNet and load
        net2 = self._make_pynet(tmp_path=str(tmp_path))
        net2.load(str(tmp_path))
        # Weights should match
        for p1, p2 in zip(net.model.parameters(), net2.model.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_total_parameters(self):
        net = self._make_pynet()
        assert net.total_parameters > 0
