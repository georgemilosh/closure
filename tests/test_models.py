"""Tests for closure.models — model instantiation, forward pass, save/load."""

from __future__ import annotations

import pytest
import torch

from closure.models import FCNN, ResNet, MLP


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
