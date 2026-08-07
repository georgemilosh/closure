"""Tests for the invariant, field-aligned pressure-tensor model."""

from __future__ import annotations

import math

import pytest
import torch

from closure.models import InvariantFieldAlignedPressureMLP
from closure.module import ClosureLitModule


def _model(*, guide_direction=None, enforce_spd=True):
    return InvariantFieldAlignedPressureMLP(
        feature_dims=[4, 24, 16, 6],
        activations=["SiLU", "SiLU", None],
        dropouts=[0.0, 0.0, 0.0],
        guide_direction=guide_direction,
        enforce_spd=enforce_spd,
    )


def _features(samples=12):
    torch.manual_seed(10)
    x = torch.randn(samples, 10)
    x[:, 0] = -(0.03 + 0.1 * torch.rand(samples))
    x[:, 1:4] = torch.randn(samples, 3) + torch.tensor([0.4, 1.0, -0.2])
    x[:, 4:7] *= 0.3
    x[:, 7:10] *= 0.2
    return x


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class TestFieldAlignedGeometry:
    def test_basis_is_right_handed_and_parallel_last(self):
        model = _model()
        x = _features()
        rotation, _ = model._basis_and_invariants(x)
        identity = torch.eye(3).expand(x.shape[0], 3, 3)
        assert torch.allclose(rotation @ rotation.transpose(1, 2), identity, atol=2e-6)
        assert torch.allclose(torch.linalg.det(rotation), torch.ones(x.shape[0]), atol=2e-6)
        bhat = x[:, 1:4] / torch.linalg.vector_norm(x[:, 1:4], dim=1, keepdim=True)
        assert torch.allclose(rotation[:, 2], bhat, atol=2e-6)

    @pytest.mark.parametrize(
        "magnetic",
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )
    def test_parallel_guide_and_null_fallbacks_are_finite(self, magnetic):
        model = _model()
        x = _features(1)
        x[0, 1:4] = torch.tensor(magnetic)
        rotation, invariants = model._basis_and_invariants(x)
        assert torch.isfinite(rotation).all()
        assert torch.isfinite(invariants).all()
        assert torch.allclose(rotation @ rotation.transpose(1, 2), torch.eye(3)[None], atol=2e-6)

    def test_galilean_invariant_features(self):
        model = _model()
        x = _features()
        boost = torch.tensor([0.7, -0.4, 0.2]).expand(x.shape[0], 3)
        transformed = x.clone()
        magnetic = x[:, 1:4]
        transformed[:, 4:7] = x[:, 4:7] - boost
        transformed[:, 7:10] = x[:, 7:10] + torch.cross(boost, magnetic, dim=1)
        _, original_invariants = model._basis_and_invariants(x)
        _, transformed_invariants = model._basis_and_invariants(transformed)
        assert torch.allclose(original_invariants, transformed_invariants, atol=2e-6)

    def test_rotational_invariants_and_tensor_covariance(self):
        torch.manual_seed(4)
        original = _model().eval()
        spatial_rotation = _rotation_x(0.61)
        rotated_guide = spatial_rotation @ torch.tensor([0.0, 1.0, 0.0])
        rotated = _model(guide_direction=rotated_guide.tolist()).eval()
        rotated.trunk.load_state_dict(original.trunk.state_dict())

        x = _features()
        xr = x.clone()
        for start in (1, 4, 7):
            xr[:, start : start + 3] = x[:, start : start + 3] @ spatial_rotation.T

        _, invariants = original._basis_and_invariants(x)
        _, rotated_invariants = rotated._basis_and_invariants(xr)
        assert torch.allclose(invariants, rotated_invariants, atol=2e-6)

        pressure = original._packed_to_tensor(original(x))
        rotated_pressure = rotated._packed_to_tensor(rotated(xr))
        expected = spatial_rotation[None] @ pressure @ spatial_rotation.T[None]
        assert torch.allclose(rotated_pressure, expected, atol=2e-6)


class TestFieldAlignedModel:
    def test_pixel_and_image_shapes(self):
        model = _model()
        pixels = _features(8)
        assert model(pixels).shape == (8, 6)
        image = pixels.reshape(2, 2, 2, 10).permute(0, 3, 1, 2)
        assert model(image).shape == (2, 6, 2, 2)

    def test_spd_output(self):
        model = _model(enforce_spd=True)
        pressure = model._packed_to_tensor(model(_features()))
        assert torch.linalg.eigvalsh(pressure).min() > 0.0

    def test_field_frame_loss_is_frobenius_error(self):
        model = _model(enforce_spd=False)
        x = _features()
        torch.manual_seed(3)
        prediction = 2.0e-3 * torch.randn(x.shape[0], 6)
        target = 2.0e-3 * torch.randn(x.shape[0], 6)
        actual = model.compute_training_loss(x, prediction, target, torch.nn.MSELoss())
        difference = model._packed_to_tensor(prediction) - model._packed_to_tensor(target)
        expected = difference.square().sum() / (x.shape[0] * 6 * model.pressure_scale**2)
        assert actual == pytest.approx(float(expected), rel=2e-5)

    def test_lightning_module_uses_field_frame_loss_hook(self):
        model = _model(enforce_spd=False)
        module = ClosureLitModule(network=model, criterion="MSELoss", scheduler=None)
        x = _features()
        prediction = model(x)
        target = torch.randn_like(prediction) * 1.0e-3
        expected = model.compute_training_loss(x, prediction, target, module.criterion)
        actual = module._compute_base_loss(x, prediction, target)
        assert torch.allclose(actual, expected)

    def test_gradients_are_finite(self):
        model = _model()
        x = _features()
        prediction = model(x)
        target = torch.randn_like(prediction) * 1.0e-3
        loss = model.compute_training_loss(x, prediction, target, torch.nn.MSELoss())
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    def test_torchscript_export(self):
        model = _model().eval()
        x = _features()
        scripted = torch.jit.script(model)
        assert torch.allclose(scripted(x), model(x), atol=1e-6)
