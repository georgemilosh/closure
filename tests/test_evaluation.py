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
def _make_trainer_stub(n_samples=10, n_channels=2):
    """Create a minimal Trainer mock with the attributes evaluation.py accesses."""
    trainer = MagicMock()
    trainer.test_loader.target_channels = list(range(n_channels))
    trainer.test_dataset.prescaler_targets = [None] * n_channels
    trainer.test_dataset.request_targets = [f"target_{i}" for i in range(n_channels)]
    return trainer


class TestEvaluateLoss:
    def test_returns_dict(self):
        trainer = _make_trainer_stub(n_samples=10, n_channels=2)
        gt = torch.rand(10, 2)
        pred = gt.clone()
        result = ev.evaluate_loss(trainer, gt, pred, "MSELoss", verbose=False)
        assert isinstance(result, dict)
        assert "total_MSELoss" in result
        assert "target_0_MSELoss" in result
        assert "target_1_MSELoss" in result

    def test_perfect_prediction_zero_loss(self):
        trainer = _make_trainer_stub(n_samples=10, n_channels=2)
        gt = torch.rand(10, 2)
        pred = gt.clone()
        result = ev.evaluate_loss(trainer, gt, pred, "MSELoss", verbose=False)
        assert float(result["total_MSELoss"]) == pytest.approx(0.0, abs=1e-6)

    def test_target_channels_none(self):
        trainer = _make_trainer_stub(n_samples=10, n_channels=3)
        trainer.test_loader.target_channels = None
        gt = torch.rand(10, 3)
        pred = gt.clone()
        result = ev.evaluate_loss(trainer, gt, pred, "MSELoss", verbose=False)
        assert len(result) == 4  # total + 3 channels


# ---------------------------------------------------------------------------
# prediction2data
# ---------------------------------------------------------------------------
class TestPrediction2Data:
    def test_simple_targets(self):
        trainer = MagicMock()
        trainer.test_dataset.request_targets = ["Bx", "By"]
        trainer.test_dataset.flatten = False

        data = {"Bx": np.zeros((4, 5, 3)), "By": np.zeros((4, 5, 3))}
        # shape (N, C, H, W) = (3, 2, 4, 5)
        pred = np.random.randn(3, 2, 4, 5)
        result = ev.prediction2data(data, trainer, pred)
        assert result is data
        np.testing.assert_array_equal(result["Bx"], pred[:, 0, ...].transpose([1, 2, 0]))
        np.testing.assert_array_equal(result["By"], pred[:, 1, ...].transpose([1, 2, 0]))

    def test_nested_targets(self):
        trainer = MagicMock()
        trainer.test_dataset.request_targets = ["ions_vx"]
        trainer.test_dataset.flatten = False

        data = {"ions": {"vx": np.zeros((4, 5, 3))}}
        pred = np.random.randn(3, 1, 4, 5)
        ev.prediction2data(data, trainer, pred)
        np.testing.assert_array_equal(data["ions"]["vx"], pred[:, 0, ...].transpose([1, 2, 0]))

    def test_flatten_mode(self):
        trainer = MagicMock()
        trainer.test_dataset.request_targets = ["Bx"]
        trainer.test_dataset.flatten = True

        data = {"Bx": np.zeros((4, 5, 3))}
        # shape (H, W, T, C) when flatten
        pred = np.random.randn(4, 5, 3, 1)
        ev.prediction2data(data, trainer, pred)
        np.testing.assert_array_equal(data["Bx"], pred[..., 0].transpose([1, 2, 0]))


# ---------------------------------------------------------------------------
# unnormalize_output alias
# ---------------------------------------------------------------------------
class TestUnnormalizeAlias:
    def test_alias_points_to_pred_unnormalize(self):
        assert ev.unnormalize_output is ev.pred_unnormalize


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
        trainer = MagicMock()
        trainer.test_dataset.request_targets = ["Bx"]
        trainer.test_dataset.flatten = False
        data = {"Bx": np.zeros((4, 5, 3))}
        pred = np.random.randn(3, 1, 4, 5)
        prediction2data(data, trainer, pred)
        assert data["Bx"].shape == (4, 5, 3)

    def test_unnormalize_output_via_utilities(self):
        from closure.utilities import unnormalize_output
        assert callable(unnormalize_output)
