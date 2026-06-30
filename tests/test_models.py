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

    def test_forward_image_shape(self):
        # Image batch [B, C, H, W] -> per-pixel MLP -> [B, C_out, H, W]
        # (the flatten=false path that lets the physics gradP loss apply).
        model = MLP(feature_dims=[10, 32, 5])
        x = torch.randn(2, 10, 8, 8)
        out = model(x)
        assert out.shape == (2, 5, 8, 8)

    def test_image_matches_pixelwise(self):
        # Applying the MLP to an image must equal applying it pixel-by-pixel.
        torch.manual_seed(0)
        model = MLP(feature_dims=[10, 32, 5], activations=["ReLU", None]).eval()
        x = torch.randn(2, 10, 4, 4)
        img_out = model(x)
        flat_in = x.permute(0, 2, 3, 1).reshape(-1, 10)
        flat_out = model(flat_in).reshape(2, 4, 4, 5).permute(0, 3, 1, 2)
        assert torch.allclose(img_out, flat_out, atol=1e-6)
