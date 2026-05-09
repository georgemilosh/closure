"""Tests for closure.evaluation module."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from closure import evaluation as ev


# ---------------------------------------------------------------------------
# parse_score
# ---------------------------------------------------------------------------
class TestParseScore:
    def test_mse(self):
        criterion = ev.parse_score("MSE")
        assert isinstance(criterion, torch.nn.MSELoss)

    def test_l1loss(self):
        criterion = ev.parse_score("L1Loss")
        assert isinstance(criterion, torch.nn.L1Loss)

    def test_r2(self):
        import torchmetrics
        criterion = ev.parse_score("r2")
        assert criterion is torchmetrics.functional.r2_score


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------
class TestComputeLoss:
    def test_mseloss_perfect(self):
        t = torch.ones(10)
        loss = ev.compute_loss(t, t, "MSELoss")
        assert float(loss) == pytest.approx(0.0, abs=1e-7)

    def test_mseloss_numpy(self):
        gt = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        loss = ev.compute_loss(gt, pred, "MSELoss")
        assert float(loss) == pytest.approx(0.0, abs=1e-7)

    def test_r2_perfect(self):
        gt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss = ev.compute_loss(gt, gt, "r2")
        assert float(loss) == pytest.approx(1.0, abs=1e-6)

    def test_r2_numpy(self):
        gt = np.array([1.0, 2.0, 3.0, 4.0])
        loss = ev.compute_loss(gt, gt.copy(), "r2")
        assert float(loss) == pytest.approx(1.0, abs=1e-6)

    def test_mseloss_nonzero(self):
        gt = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([1.5, 2.5, 3.5])
        loss = ev.compute_loss(gt, pred, "MSELoss")
        assert float(loss) == pytest.approx(0.25, abs=1e-6)

    def test_invalid_criterion_raises(self):
        gt = torch.ones(3)
        with pytest.raises((AttributeError, ValueError)):
            ev.compute_loss(gt, gt, 42)


# ---------------------------------------------------------------------------
# evaluate_loss
# ---------------------------------------------------------------------------
def _make_dataset_stub(n_channels=2):
    """Create a minimal dataset mock with the attributes evaluation.py accesses."""
    dataset = MagicMock()
    dataset.prescaler_targets = [None] * n_channels
    dataset.request_targets = [f"target_{i}" for i in range(n_channels)]
    return dataset


class TestEvaluateLoss:
    def test_returns_dict(self):
        dataset = _make_dataset_stub(n_channels=2)
        gt = torch.rand(10, 2)
        pred = gt.clone()
        result = ev.evaluate_loss(dataset, gt, pred, "MSELoss", verbose=False)
        assert isinstance(result, dict)
        assert "total_MSELoss" in result
        assert "target_0_MSELoss" in result
        assert "target_1_MSELoss" in result

    def test_perfect_prediction_zero_loss(self):
        dataset = _make_dataset_stub(n_channels=2)
        gt = torch.rand(10, 2)
        pred = gt.clone()
        result = ev.evaluate_loss(dataset, gt, pred, "MSELoss", verbose=False)
        assert float(result["total_MSELoss"]) == pytest.approx(0.0, abs=1e-6)

    def test_target_channels_subset(self):
        dataset = _make_dataset_stub(n_channels=3)
        gt = torch.rand(10, 3)
        pred = gt.clone()
        result = ev.evaluate_loss(dataset, gt, pred, "MSELoss", target_channels=[0, 2], verbose=False)
        assert len(result) == 3  # total + 2 channels


# ---------------------------------------------------------------------------
# prediction2data
# ---------------------------------------------------------------------------
class TestPrediction2Data:
    def test_simple_targets(self):
        dataset = MagicMock()
        dataset.request_targets = ["Bx", "By"]
        dataset.flatten = False

        data = {"Bx": np.zeros((4, 5, 3)), "By": np.zeros((4, 5, 3))}
        # shape (N, C, H, W) = (3, 2, 4, 5)
        pred = np.random.randn(3, 2, 4, 5)
        result = ev.prediction2data(data, dataset, pred)
        assert result is data
        np.testing.assert_array_equal(result["Bx"], pred[:, 0, ...].transpose([1, 2, 0]))
        np.testing.assert_array_equal(result["By"], pred[:, 1, ...].transpose([1, 2, 0]))

    def test_nested_targets(self):
        dataset = MagicMock()
        dataset.request_targets = ["ions_vx"]
        dataset.flatten = False

        data = {"ions": {"vx": np.zeros((4, 5, 3))}}
        pred = np.random.randn(3, 1, 4, 5)
        ev.prediction2data(data, dataset, pred)
        np.testing.assert_array_equal(data["ions"]["vx"], pred[:, 0, ...].transpose([1, 2, 0]))

    def test_flatten_mode(self):
        dataset = MagicMock()
        dataset.request_targets = ["Bx"]
        dataset.flatten = True

        data = {"Bx": np.zeros((4, 5, 3))}
        # shape (H, W, T, C) when flatten
        pred = np.random.randn(4, 5, 3, 1)
        ev.prediction2data(data, dataset, pred)
        np.testing.assert_array_equal(data["Bx"], pred[..., 0].transpose([1, 2, 0]))


# ---------------------------------------------------------------------------
# unnormalize_output alias
# ---------------------------------------------------------------------------
class TestUnnormalizeAlias:
    def test_alias_points_to_pred_unnormalize(self):
        assert ev.unnormalize_output is ev.pred_unnormalize


# ---------------------------------------------------------------------------
# pred_pressure_gradients_jvp
# ---------------------------------------------------------------------------
class _DummyFlatModel(torch.nn.Module):
    def __init__(self, n_features: int, n_targets: int):
        super().__init__()
        self.layer = torch.nn.Linear(n_features, n_targets)
        self.device = torch.device("cpu")

    def forward(self, x):
        return self.layer(x)


class TestPredPressureGradientsJvp:
    def test_flatten_mode_accepts_pixelwise_input(self):
        nt, nx, ny = 2, 4, 3
        n_features = 5
        n_targets = 2

        dataset = types.SimpleNamespace(
            flatten=True,
            features_shape=(nt, nx, ny, n_features),
            request_targets=["Pxx_e", "Pxy_e"],
            scaler_targets=False,
            prescaler_targets=[None, None],
            targets_std=np.ones((n_targets,), dtype=np.float32),
            targets_mean=np.zeros((n_targets,), dtype=np.float32),
        )

        model = _DummyFlatModel(n_features=n_features, n_targets=n_targets)
        test_features = torch.randn(nt * nx * ny, n_features)
        data = {"rho": {"e": np.ones((nx, ny, nt), dtype=np.float32)}}

        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(0.0, 1.0, ny)

        result = ev.pred_pressure_gradients_jvp(
            data=data,
            test_features=test_features,
            model=model,
            dataset=dataset,
            x=x,
            y=y,
            species="e",
        )

        assert "prediction" in result
        assert "dPdx" in result
        assert "dPdy" in result
        assert result["prediction"]["Pxx_e"].shape == (nx, ny, nt)
        assert result["dPdx"]["Pxy_e"].shape == (nx, ny, nt)


class TestPredUnnormalizeFlatten:
    def test_uses_runtime_data_shape_not_dataset_targets_shape(self):
        nt, nx, ny = 2, 4, 3
        n_features = 5
        n_targets = 2

        dataset = types.SimpleNamespace(
            flatten=True,
            request_targets=["Pxx_e", "Pxy_e"],
            prescaler_targets=[None, None],
            scaler_targets=False,
            # Intentionally wrong/training-like shape to ensure runtime data shape is used.
            targets_shape=(25, 512, 512, n_targets),
        )

        model = _DummyFlatModel(n_features=n_features, n_targets=n_targets)
        test_features = torch.randn(nt * nx * ny, n_features)
        data = {
            "Pxx": {"e": np.zeros((nx, ny, nt), dtype=np.float32)},
            "Pxy": {"e": np.zeros((nx, ny, nt), dtype=np.float32)},
        }

        ev.pred_unnormalize(data, test_features, model, dataset)

        assert data["Pxx"]["e"].shape == (nx, ny, nt)
        assert data["Pxy"]["e"].shape == (nx, ny, nt)


# ---------------------------------------------------------------------------
# backward-compat stubs in utilities.py
# ---------------------------------------------------------------------------
class TestBackwardCompat:
    """Ensure that functions accessed via utilities.py still route correctly."""

    def test_parse_score_via_utilities(self):
        from closure.utilities import parse_score
        criterion = parse_score("MSE")
        assert isinstance(criterion, torch.nn.MSELoss)

    def test_compute_loss_via_utilities(self):
        from closure.utilities import compute_loss
        t = torch.ones(5)
        loss = compute_loss(t, t, "MSELoss")
        assert float(loss) == pytest.approx(0.0, abs=1e-7)

    def test_prediction2data_via_utilities(self):
        from closure.utilities import prediction2data
        dataset = MagicMock()
        dataset.request_targets = ["Bx"]
        dataset.flatten = False
        data = {"Bx": np.zeros((4, 5, 3))}
        pred = np.random.randn(3, 1, 4, 5)
        prediction2data(data, dataset, pred)
        assert data["Bx"].shape == (4, 5, 3)

    def test_unnormalize_output_via_utilities(self):
        from closure.utilities import unnormalize_output
        assert callable(unnormalize_output)
