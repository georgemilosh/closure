"""Tests for physics-informed loss helpers in ClosureLitModule.

Covers:
  5.1  Derivative kernel parity  — _fd4_derivatives_2d vs plasma.highdiff on
       analytical fields (linear, sinusoidal) and vs the numpy reference on
       random data; verifies sign, magnitude, and 4th-order accuracy.

  5.2  E_amb construction        — _compute_eamb_from_pressure index/sign
       correctness including sign-agnosticism across positive and negative
       rho conventions that arise from ECsim vs Menura extract_features_kernel.

  5.3  Training-step smoke test  — enabling lambda_gradP / lambda_eamb
       changes total loss, does not regress baseline with zero weights, and
       emits expected per-component log keys.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from closure import plasma
from closure.module import ClosureLitModule
from closure.models import FCNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_module(lambda_gradP: float = 0.0, lambda_eamb: float = 0.0) -> ClosureLitModule:
    network = FCNN(channels=[6, 8, 6], kernels=[3, 3], activations=["ReLU", None])
    return ClosureLitModule(
        network=network,
        criterion="MSELoss",
        scheduler=None,
        lambda_gradP=lambda_gradP,
        lambda_eamb=lambda_eamb,
        physics_dx=1.0,
        physics_dy=1.0,
        physics_small=1e-10,
        physics_rho_abs=True,
    )


def _make_batch(
    batch: int = 2,
    channels: int = 6,
    h: int = 32,
    w: int = 32,
    rho_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (features, prediction, targets) with smooth spatial content."""
    x = torch.linspace(0.0, 2 * torch.pi, w)
    y = torch.linspace(0.0, 2 * torch.pi, h)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    base = (torch.sin(xx) + torch.cos(yy)).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    targets = base.expand(batch, channels, h, w) * torch.arange(1, channels + 1).view(1, -1, 1, 1).float()
    prediction = targets + 0.05 * torch.randn_like(targets)

    # rho channel (index 0): positive (Menura convention) or negative (ECsim convention)
    rho = torch.abs(base).expand(batch, 1, h, w) * rho_sign
    features = torch.cat([rho, torch.randn(batch, 9, h, w)], dim=1)

    return features, prediction, targets


# ---------------------------------------------------------------------------
# 5.1  Derivative kernel parity
# ---------------------------------------------------------------------------

class TestFd4DerivativesParity:
    """Parity tests between _fd4_derivatives_2d and plasma.highdiff."""

    def test_linear_field_x(self):
        """d/dx (3x) == 3 everywhere on interior cells.

        plasma.py convention: axis 0 (H/dim=-2) = x direction.
        Field must therefore vary along dim=-2.
        """
        h, w = 32, 32
        x = torch.linspace(0.0, 1.0, h)  # x varies along H (dim=-2)
        field = 3.0 * x.view(1, 1, h, 1).expand(1, 1, h, w)
        dx_val = 1.0 / (h - 1)
        dfdx, dfdy = ClosureLitModule._fd4_derivatives_2d(field, dx_val, dx_val)
        np.testing.assert_allclose(dfdx.numpy(), 3.0, atol=1e-4,
                                   err_msg="d/dx of linear field should be constant 3")
        np.testing.assert_allclose(dfdy.numpy(), 0.0, atol=1e-4,
                                   err_msg="d/dy of x-only field should be 0")

    def test_linear_field_y(self):
        """d/dy (2y) == 2 everywhere on interior cells.

        plasma.py convention: axis 1 (W/dim=-1) = y direction.
        Field must therefore vary along dim=-1.
        """
        h, w = 32, 32
        y = torch.linspace(0.0, 1.0, w)  # y varies along W (dim=-1)
        field = 2.0 * y.view(1, 1, 1, w).expand(1, 1, h, w)
        dy_val = 1.0 / (w - 1)
        dfdx, dfdy = ClosureLitModule._fd4_derivatives_2d(field, dy_val, dy_val)
        np.testing.assert_allclose(dfdx.numpy(), 0.0, atol=1e-4)
        np.testing.assert_allclose(dfdy.numpy(), 2.0, atol=1e-4)

    def test_sinusoidal_exact_derivative(self):
        """Interior values of d/dx sin(kx) must match cos(kx) to 4th-order accuracy.

        Field varies along axis 0 (H/dim=-2 = x direction in plasma convention).
        """
        n = 64
        k = 2.0 * np.pi
        x_np = np.linspace(0.0, 1.0, n, endpoint=False)
        dx = x_np[1] - x_np[0]
        # sin(kx) varies along axis 0 (rows = x direction)
        field_np = np.sin(k * x_np)[:, np.newaxis] * np.ones((1, n))

        field_t = torch.from_numpy(field_np).float().unsqueeze(0).unsqueeze(0)
        dfdx, _ = ClosureLitModule._fd4_derivatives_2d(field_t, dx, dx)
        interior = dfdx[0, 0].numpy()
        exact = k * np.cos(k * x_np[2:-2, np.newaxis]) * np.ones((n - 4, n - 4))
        # 4th-order scheme: error ~ (dx)^4; add atol for near-zero cos values.
        np.testing.assert_allclose(interior, exact, rtol=1e-3, atol=1e-5)

    def test_parity_with_plasma_highdiff_random(self):
        """_fd4_derivatives_2d must agree with plasma.highdiff on interior cells.

        plasma.highdiff uses mode='wrap'; _fd4_derivatives_2d does not wrap but
        crops to interior.  They must agree on the 2:-2 interior slice where
        wrap and non-wrap are identical for smooth periodic-like fields.
        """
        rng = np.random.default_rng(42)
        n = 32
        dx = 0.1
        field_np = rng.standard_normal((n, n))

        # numpy reference (no time axis — squeeze last dim for single snap)
        ref_dx = plasma.highdiff(field_np, dx, dx, axis=0, mode="wrap")
        ref_dy = plasma.highdiff(field_np, dx, dx, axis=1, mode="wrap")

        field_t = torch.from_numpy(field_np).float().unsqueeze(0).unsqueeze(0)
        got_dx, got_dy = ClosureLitModule._fd4_derivatives_2d(field_t, dx, dx)

        # Compare only on interior cells (the region where wrap and crop agree
        # for a field with very small boundary contributions).
        np.testing.assert_allclose(
            got_dx[0, 0].numpy(), ref_dx[2:-2, 2:-2], rtol=1e-4, atol=1e-5,
            err_msg="dfdx: torch result must match plasma.highdiff interior",
        )
        np.testing.assert_allclose(
            got_dy[0, 0].numpy(), ref_dy[2:-2, 2:-2], rtol=1e-4, atol=1e-5,
            err_msg="dfdy: torch result must match plasma.highdiff interior",
        )

    def test_spatial_crop_shape(self):
        """Output shape must be interior-cropped by 2 cells on each side."""
        h, w = 24, 36
        field = torch.zeros(1, 1, h, w)
        dx, dy = ClosureLitModule._fd4_derivatives_2d(field, 1.0, 1.0)
        assert dx.shape == (1, 1, h - 4, w - 4), f"dx shape {dx.shape}"
        assert dy.shape == (1, 1, h - 4, w - 4), f"dy shape {dy.shape}"

    def test_minimum_size_guard(self):
        """Raises ValueError for spatial dims < 5."""
        field = torch.zeros(1, 1, 4, 32)
        with pytest.raises(ValueError, match="5"):
            ClosureLitModule._fd4_derivatives_2d(field, 1.0, 1.0)


# ---------------------------------------------------------------------------
# 5.2  E_amb construction — channel index and sign convention
# ---------------------------------------------------------------------------

class TestEambConstruction:
    """Index-order and sign-convention tests for _compute_eamb_from_pressure."""

    # Closure default target order: Pxx=0, Pxy=1, Pxz=2, Pyy=3, Pyz=4, Pzz=5
    CHANNEL_MAP = {"Pxx": 0, "Pxy": 1, "Pxz": 2, "Pyy": 3, "Pyz": 4}

    def _flat_pressure(self, batch: int, h: int, w: int) -> torch.Tensor:
        """Uniform pressure (all derivatives zero → E_amb = 0)."""
        return torch.ones(batch, 6, h, w)

    def _linear_pxx(self, batch: int, h: int, w: int, slope: float = 1.0) -> torch.Tensor:
        """Pxx = slope*x, all others constant.

        x varies along H (dim=-2, axis 0 in plasma convention), so
        dPxx/dx = slope and EPx = -slope / rho.
        """
        x = torch.linspace(0.0, 1.0, h)  # x along H (dim=-2)
        p = torch.ones(batch, 6, h, w)
        p[:, 0, :, :] = slope * x.view(h, 1).expand(h, w)
        return p

    def test_zero_gradient_gives_zero_eamb(self):
        batch, h, w = 2, 20, 20
        pressure = self._flat_pressure(batch, h, w)
        rho = torch.ones(batch, h, w)
        eamb = ClosureLitModule._compute_eamb_from_pressure(
            pressure, rho, self.CHANNEL_MAP, dx=1.0, dy=1.0, small=1e-10, rho_abs=True,
        )
        np.testing.assert_allclose(eamb.numpy(), 0.0, atol=1e-6)

    def test_eamb_shape(self):
        batch, h, w = 3, 20, 24
        pressure = self._flat_pressure(batch, h, w)
        rho = torch.ones(batch, h, w)
        eamb = ClosureLitModule._compute_eamb_from_pressure(
            pressure, rho, self.CHANNEL_MAP, dx=1.0, dy=1.0, small=1e-10, rho_abs=True,
        )
        # Spatial dims are cropped by 2 cells on each side
        assert eamb.shape == (batch, 3, h - 4, w - 4)

    def test_sign_positive_rho(self):
        """EPx = -dPxx/dx / rho with positive rho: result should be negative for positive slope."""
        batch, h, w = 1, 20, 20
        dx = 1.0 / (h - 1)  # Pxx varies along H (x direction)
        pressure = self._linear_pxx(batch, h, w, slope=1.0)
        rho = torch.ones(batch, h, w)
        eamb = ClosureLitModule._compute_eamb_from_pressure(
            pressure, rho, self.CHANNEL_MAP, dx=dx, dy=dx, small=1e-10, rho_abs=True,
        )
        # EPx = -(dPxx/dx + dPxy/dy) / |rho| = -1.0 / 1.0 = -1.0
        np.testing.assert_allclose(eamb[:, 0].numpy(), -1.0, atol=1e-4)

    def test_sign_negative_rho_is_same_as_positive(self):
        """rho_abs=True must make the result identical for rho=-1 and rho=+1.

        This validates the ECsim-convention case: extract_features_kernel
        negates density_b to produce positive rho, but during training the
        raw ECsim value (negative) may appear in the feature channel.
        """
        batch, h, w = 1, 20, 20
        dx = 1.0 / (w - 1)
        pressure = self._linear_pxx(batch, h, w, slope=1.0)
        rho_pos = torch.ones(batch, h, w)
        rho_neg = -torch.ones(batch, h, w)

        eamb_pos = ClosureLitModule._compute_eamb_from_pressure(
            pressure, rho_pos, self.CHANNEL_MAP, dx=dx, dy=dx, small=1e-10, rho_abs=True,
        )
        eamb_neg = ClosureLitModule._compute_eamb_from_pressure(
            pressure, rho_neg, self.CHANNEL_MAP, dx=dx, dy=dx, small=1e-10, rho_abs=True,
        )
        np.testing.assert_allclose(
            eamb_pos.numpy(), eamb_neg.numpy(), atol=1e-7,
            err_msg="rho_abs=True must neutralise ECsim negative-density convention",
        )

    def test_rho_abs_false_flips_sign(self):
        """With rho_abs=False, negative rho flips the sign of E_amb."""
        batch, h, w = 1, 20, 20
        dx = 1.0 / (w - 1)
        pressure = self._linear_pxx(batch, h, w, slope=1.0)
        rho_pos = torch.ones(batch, h, w)
        rho_neg = -torch.ones(batch, h, w)

        eamb_pos = ClosureLitModule._compute_eamb_from_pressure(
            pressure, rho_pos, self.CHANNEL_MAP, dx=dx, dy=dx, small=1e-10, rho_abs=False,
        )
        eamb_neg = ClosureLitModule._compute_eamb_from_pressure(
            pressure, rho_neg, self.CHANNEL_MAP, dx=dx, dy=dx, small=1e-10, rho_abs=False,
        )
        np.testing.assert_allclose(
            eamb_pos.numpy(), -eamb_neg.numpy(), atol=1e-7,
            err_msg="rho_abs=False should invert E_amb when rho sign flips",
        )

    def test_eamb_proxy_matches_offline_get_ohm(self):
        """Interior E_amb proxy must match plasma.get_Ohm EPx/EPy/EPz on a shared sample.

        Uses a smooth sinusoidal pressure field so offline highdiff(wrap) and
        torch fd4(crop) agree on interior cells.
        """
        rng = np.random.default_rng(7)
        nx, ny = 32, 32
        dx = 0.1
        dy = 0.1
        x_np = np.arange(nx) * dx
        y_np = np.arange(ny) * dy
        xx, yy = np.meshgrid(x_np, y_np, indexing="ij")

        # Build smooth pressure and rho arrays matching ECsim sign convention
        Pxx = 2.0 + 0.3 * np.sin(2 * np.pi * xx / (nx * dx))
        Pxy = 0.1 * np.cos(2 * np.pi * yy / (ny * dy))
        Pxz = 0.05 * np.sin(2 * np.pi * (xx + yy) / (nx * dx))
        Pyy = 2.5 + 0.2 * np.cos(2 * np.pi * xx / (nx * dx))
        Pyz = 0.08 * np.sin(2 * np.pi * yy / (ny * dy))
        Pzz = 3.0 * np.ones((nx, ny))
        rho_e_neg = -np.ones((nx, ny))  # ECsim convention: negative

        # Offline reference (plasma.get_Ohm) — adds a time axis
        def _add_t(arr): return arr[..., np.newaxis]
        data = {
            "Bx": _add_t(np.ones((nx, ny))),
            "By": _add_t(np.zeros((nx, ny))),
            "Bz": _add_t(np.ones((nx, ny))),
            "Ex": _add_t(np.zeros((nx, ny))),
            "Ey": _add_t(np.zeros((nx, ny))),
            "Ez": _add_t(np.zeros((nx, ny))),
            "rho": {"e": _add_t(rho_e_neg), "i": _add_t(np.ones((nx, ny)))},
            "Jx": {"e": _add_t(np.zeros((nx, ny))), "i": _add_t(np.zeros((nx, ny)))},
            "Jy": {"e": _add_t(np.zeros((nx, ny))), "i": _add_t(np.zeros((nx, ny)))},
            "Jz": {"e": _add_t(np.zeros((nx, ny))), "i": _add_t(np.zeros((nx, ny)))},
            "Vx": {"e": _add_t(np.zeros((nx, ny))), "i": _add_t(np.zeros((nx, ny)))},
            "Vy": {"e": _add_t(np.zeros((nx, ny))), "i": _add_t(np.zeros((nx, ny)))},
            "Vz": {"e": _add_t(np.zeros((nx, ny))), "i": _add_t(np.zeros((nx, ny)))},
            "Pxx": {"e": _add_t(Pxx), "i": _add_t(np.zeros((nx, ny)))},
            "Pxy": {"e": _add_t(Pxy), "i": _add_t(np.zeros((nx, ny)))},
            "Pxz": {"e": _add_t(Pxz), "i": _add_t(np.zeros((nx, ny)))},
            "Pyy": {"e": _add_t(Pyy), "i": _add_t(np.zeros((nx, ny)))},
            "Pyz": {"e": _add_t(Pyz), "i": _add_t(np.zeros((nx, ny)))},
            "Pzz": {"e": _add_t(Pzz), "i": _add_t(np.zeros((nx, ny)))},
        }
        plasma.get_Ohm(data, [-1.0, 1.0], x_np, y_np)
        ref_epx = data["EPx"][2:-2, 2:-2, 0]  # t=0, interior
        ref_epy = data["EPy"][2:-2, 2:-2, 0]
        ref_epz = data["EPz"][2:-2, 2:-2, 0]

        # Torch version — use |rho| (Menura/positive convention)
        channel_map = {"Pxx": 0, "Pxy": 1, "Pxz": 2, "Pyy": 3, "Pyz": 4}
        pressure_t = torch.from_numpy(
            np.stack([Pxx, Pxy, Pxz, Pyy, Pyz, Pzz], axis=0)
        ).float().unsqueeze(0)  # [1, 6, nx, ny]
        rho_t = torch.from_numpy(np.abs(rho_e_neg)).float().unsqueeze(0)  # [1, nx, ny]

        eamb = ClosureLitModule._compute_eamb_from_pressure(
            pressure_t, rho_t, channel_map, dx=dx, dy=dy, small=1e-10, rho_abs=True,
        )
        got_epx = eamb[0, 0].numpy()
        got_epy = eamb[0, 1].numpy()
        got_epz = eamb[0, 2].numpy()

        np.testing.assert_allclose(got_epx, ref_epx, rtol=1e-3, atol=1e-5,
                                   err_msg="EPx mismatch vs plasma.get_Ohm")
        np.testing.assert_allclose(got_epy, ref_epy, rtol=1e-3, atol=1e-5,
                                   err_msg="EPy mismatch vs plasma.get_Ohm")
        np.testing.assert_allclose(got_epz, ref_epz, rtol=1e-3, atol=1e-5,
                                   err_msg="EPz mismatch vs plasma.get_Ohm")


class TestPhysicsScaling:
    """Physics losses must operate in physical pressure/rho units."""

    def test_inverse_targets_for_physics_undoes_normalization_and_prescaling(self):
        dataset = SimpleNamespace(
            scaler_targets=True,
            targets_mean=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32),
            targets_std=np.array([0.5, 0.5, 0.5, 2.0, 2.0, 2.0], dtype=np.float32),
            prescaler_targets=[np.log, np.log, np.log, np.arcsinh, np.arcsinh, np.arcsinh],
        )
        scaled = torch.zeros(1, 6, 4, 4)
        physical = ClosureLitModule._inverse_targets_for_physics(scaled, dataset)

        expected = torch.tensor(
            [np.exp(1.0), np.exp(2.0), np.exp(3.0), np.sinh(0.1), np.sinh(0.2), np.sinh(0.3)],
            dtype=torch.float32,
        ).view(1, 6, 1, 1)
        torch.testing.assert_close(physical, expected.expand_as(physical))

    def test_inverse_feature_channel_for_physics_undoes_rho_normalization(self):
        dataset = SimpleNamespace(
            scaler_features=True,
            features_mean=np.array([-2.0, 10.0], dtype=np.float32),
            features_std=np.array([4.0, 5.0], dtype=np.float32),
            prescaler_features=[None, None],
        )
        rho_scaled = torch.zeros(2, 8, 8)
        rho_physical = ClosureLitModule._inverse_feature_channel_for_physics(rho_scaled, dataset, 0)
        torch.testing.assert_close(rho_physical, torch.full_like(rho_scaled, -2.0))

    def test_physics_mse_is_relative_by_default(self):
        module = _make_module()
        prediction = torch.full((1, 1, 4, 4), 2.0)
        target = torch.full((1, 1, 4, 4), 1.0)

        got = module._physics_mse(prediction, target)
        expected = torch.tensor(1.0 / 2.5)
        torch.testing.assert_close(got, expected)

    def test_physics_loss_scale_warmup_then_ramp(self):
        scale = ClosureLitModule._physics_loss_scale_from_epoch
        assert scale(epoch=0, warmup_epochs=2, ramp_epochs=3) == 0.0
        assert scale(epoch=1, warmup_epochs=2, ramp_epochs=3) == 0.0
        assert scale(epoch=2, warmup_epochs=2, ramp_epochs=3) == pytest.approx(1.0 / 3.0)
        assert scale(epoch=3, warmup_epochs=2, ramp_epochs=3) == pytest.approx(2.0 / 3.0)
        assert scale(epoch=4, warmup_epochs=2, ramp_epochs=3) == 1.0
        assert scale(epoch=5, warmup_epochs=2, ramp_epochs=3) == 1.0


# ---------------------------------------------------------------------------
# 5.3  Training-step smoke tests
# ---------------------------------------------------------------------------

class TestPhysicsLossSmoke:
    """End-to-end checks that the physics terms integrate correctly into training."""

    def _run_step(self, module: ClosureLitModule, batch):
        """Run training_step and return the logged metrics dict."""
        module.eval()
        logged: dict = {}
        # Monkey-patch module.log to capture values
        module.log = lambda key, val, **kw: logged.update({key: float(val)})
        module.training_step(batch, batch_idx=0)
        return logged

    def test_zero_weights_baseline_matches_mse(self):
        """With lambda=0 the total logged loss equals plain MSELoss."""
        module = _make_module(lambda_gradP=0.0, lambda_eamb=0.0)
        features, prediction, targets = _make_batch()
        # For the smoke test we bypass the network and pass prediction as if it
        # were the network output by patching forward.
        module.forward = lambda x: prediction  # type: ignore[method-assign]
        batch = (features, targets)
        logged = self._run_step(module, batch)

        expected_mse = float(nn.MSELoss()(prediction, targets))
        assert "train_loss" in logged
        assert abs(logged["train_loss"] - expected_mse) < 1e-5, (
            f"Zero-weight loss {logged['train_loss']:.6f} != MSE {expected_mse:.6f}"
        )

    def test_nonzero_gradP_increases_loss(self):
        """A nonzero lambda_gradP on imperfect predictions must raise total loss."""
        module_base = _make_module(lambda_gradP=0.0)
        module_phys = _make_module(lambda_gradP=1.0)
        features, prediction, targets = _make_batch()

        for m in (module_base, module_phys):
            m.forward = lambda x: prediction  # type: ignore[method-assign]

        batch = (features, targets)
        log_base = self._run_step(module_base, batch)
        log_phys = self._run_step(module_phys, batch)

        assert log_phys["train_loss"] > log_base["train_loss"], (
            "Physics loss should increase total loss for imperfect predictions"
        )

    def test_per_component_log_keys_present(self):
        """training_step must log train_loss_base, train_loss_gradp, train_loss_eamb."""
        module = _make_module(lambda_gradP=0.5, lambda_eamb=0.0)
        features, prediction, targets = _make_batch()
        module.forward = lambda x: prediction  # type: ignore[method-assign]
        batch = (features, targets)
        logged = self._run_step(module, batch)

        for key in ("train_loss", "train_loss_base", "train_loss_gradp", "train_loss_eamb"):
            assert key in logged, f"Expected log key '{key}' not found"

    def test_gradp_component_is_zero_for_perfect_prediction(self):
        """If prediction == targets, gradP and eamb losses must be zero."""
        module = _make_module(lambda_gradP=1.0, lambda_eamb=1.0)
        features, _, targets = _make_batch()
        module.forward = lambda x: targets  # perfect prediction
        batch = (features, targets)
        logged = self._run_step(module, batch)

        assert abs(logged["train_loss_gradp"]) < 1e-6, logged["train_loss_gradp"]
        # eamb will also be zero when prediction == targets
        assert abs(logged["train_loss_eamb"]) < 1e-6, logged["train_loss_eamb"]

    def test_eamb_loss_activates_with_positive_lambda(self):
        """lambda_eamb > 0 must yield nonzero eamb component for imperfect predictions."""
        module = _make_module(lambda_gradP=0.0, lambda_eamb=1.0)
        features, prediction, targets = _make_batch()
        module.forward = lambda x: prediction
        batch = (features, targets)
        logged = self._run_step(module, batch)

        assert logged["train_loss_eamb"] >= 0.0
        # Should be nonzero for non-identical prediction/targets
        assert logged["train_loss_eamb"] > 0.0, (
            "E_amb loss should be positive for imperfect predictions"
        )

    def test_val_step_logs_physics_components(self):
        """validation_step must also emit per-component keys."""
        module = _make_module(lambda_gradP=0.3)
        features, prediction, targets = _make_batch()
        module.forward = lambda x: prediction

        logged: dict = {}
        module.log = lambda key, val, **kw: logged.update({key: float(val)})
        module.validation_step((features, targets), batch_idx=0)

        for key in ("val_loss", "val_loss_base", "val_loss_gradp", "val_loss_eamb"):
            assert key in logged, f"Expected val log key '{key}' not found"
