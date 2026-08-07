"""Tests for the Tier-2 flow-gradient invariants.

Three things must hold for these channels to be usable as closure inputs:

1. the derivative operator is *exactly* Menura's fourth-order stencil, so the
   training-time and deployment-time features agree;
2. the four outputs are genuine rotational scalars;
3. they take the analytically correct values on flows whose strain tensor can
   be written down by hand.
"""

import numpy as np
import pytest

from closure.field_invariants import (
    INVARIANT_NAMES,
    _d4,
    flow_gradient_invariants,
    strain_tensor,
)

GRID = 32
SPACING = 0.0390625  # the campaign's dx: 20 d_i over 512 cells


def _coordinates(n=GRID, delta=SPACING):
    axis = np.arange(n) * delta
    return np.meshgrid(axis, axis, indexing="ij")


def _uniform(vector, n=GRID):
    return np.stack([np.full((n, n), component, dtype=float) for component in vector])


class TestStencil:
    def test_matches_menura_fourth_order_stencil(self):
        """_d4 must equal (8(f+1 - f-1) - (f+2 - f-2)) / 12dx, Menura's d4."""
        rng = np.random.default_rng(0)
        field = rng.normal(size=(GRID, GRID))
        expected = (
            8.0 * (np.roll(field, -1, 0) - np.roll(field, 1, 0))
            - (np.roll(field, -2, 0) - np.roll(field, 2, 0))
        ) / (12.0 * SPACING)
        np.testing.assert_allclose(_d4(field, 0, SPACING), expected, rtol=0, atol=0)

    def test_matches_repo_highdiff(self):
        """The repo's highdiff (used by divP) must agree with our stencil."""
        from closure.plasma import highdiff

        rng = np.random.default_rng(1)
        field = rng.normal(size=(GRID, GRID))
        for axis in (0, 1):
            np.testing.assert_allclose(
                _d4(field, axis, SPACING),
                highdiff(field, SPACING, SPACING, axis=axis, mode="wrap"),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_exact_on_a_resolved_sinusoid(self):
        """Fourth-order accuracy: a well-resolved mode differentiates cleanly."""
        x, _ = _coordinates()
        length = GRID * SPACING
        wave = 2 * np.pi / length
        field = np.sin(wave * x)
        np.testing.assert_allclose(
            _d4(field, 0, SPACING), wave * np.cos(wave * x), rtol=2e-4, atol=2e-6
        )


class TestAnalyticFlows:
    """Flows whose rate-of-strain tensor is constant and known by hand."""

    def test_shear_perpendicular_to_b_is_pure_perpendicular_strain(self):
        # u = (a y, 0, 0) with B along z: W_xy = W_yx = a/2, everything else 0.
        # B is normal to the shear plane, so W.b = 0 and only the traceless
        # perpendicular block survives, with norm a/sqrt(2).
        amplitude = 0.7
        _, y = _coordinates()
        velocity = np.stack([amplitude * y, np.zeros_like(y), np.zeros_like(y)])
        invariants = flow_gradient_invariants(
            _uniform((0.0, 0.0, 1.3)), velocity, SPACING, SPACING
        )
        interior = np.s_[4:-4, 4:-4]  # avoid the periodic wrap of the ramp
        assert invariants["Wpar_e"][interior] == pytest.approx(0.0, abs=1e-9)
        assert invariants["divV_e"][interior] == pytest.approx(0.0, abs=1e-9)
        assert invariants["Wmix_e"][interior] == pytest.approx(0.0, abs=1e-9)
        assert invariants["Wperp_e"][interior] == pytest.approx(
            amplitude / np.sqrt(2.0), rel=1e-6
        )

    def test_shear_along_b_is_pure_mixed_block(self):
        # Same shear, but B along x now lies in the shear plane: the mixed
        # (gyroviscous) block picks it up with magnitude a/2 and the
        # perpendicular block vanishes.
        amplitude = 0.7
        _, y = _coordinates()
        velocity = np.stack([amplitude * y, np.zeros_like(y), np.zeros_like(y)])
        invariants = flow_gradient_invariants(
            _uniform((2.0, 0.0, 0.0)), velocity, SPACING, SPACING
        )
        interior = np.s_[4:-4, 4:-4]
        assert invariants["Wpar_e"][interior] == pytest.approx(0.0, abs=1e-9)
        assert invariants["Wmix_e"][interior] == pytest.approx(amplitude / 2.0, rel=1e-6)
        assert invariants["Wperp_e"][interior] == pytest.approx(0.0, abs=1e-7)

    def test_parallel_compression_is_pure_cgl_driver(self):
        # u = (a x, 0, 0) with B along x: div u = a and b.W.b = a, while both
        # agyrotropic channels vanish.  This is the CGL parallel driver alone.
        amplitude = 0.4
        x, _ = _coordinates()
        velocity = np.stack([amplitude * x, np.zeros_like(x), np.zeros_like(x)])
        invariants = flow_gradient_invariants(
            _uniform((1.0, 0.0, 0.0)), velocity, SPACING, SPACING
        )
        interior = np.s_[4:-4, 4:-4]
        assert invariants["Wpar_e"][interior] == pytest.approx(amplitude, rel=1e-6)
        assert invariants["divV_e"][interior] == pytest.approx(amplitude, rel=1e-6)
        assert invariants["Wmix_e"][interior] == pytest.approx(0.0, abs=1e-7)
        assert invariants["Wperp_e"][interior] == pytest.approx(0.0, abs=1e-7)

    def test_strain_tensor_is_symmetric_with_vanishing_z_row(self):
        rng = np.random.default_rng(2)
        velocity = rng.normal(size=(3, GRID, GRID))
        strain, _ = strain_tensor(velocity, SPACING, SPACING)
        np.testing.assert_allclose(strain, np.swapaxes(strain, 0, 1), atol=1e-12)
        # d_z = 0 in a 2-D run, so W_zz vanishes identically.
        np.testing.assert_allclose(strain[2, 2], 0.0, atol=1e-12)


class TestRotationalInvariance:
    def test_invariant_under_ninety_degree_rotation_about_z(self):
        """A rotation of grid *and* vectors must leave the scalars unchanged.

        Only rotations about z preserve the 2-D structure (d_z = 0), and 90
        degrees is exact on a periodic square grid, so this is a numerically
        clean equivariance check.
        """
        rng = np.random.default_rng(3)
        magnetic = rng.normal(size=(3, GRID, GRID))
        velocity = rng.normal(size=(3, GRID, GRID))
        # Smooth the fields so the stencil is not dominated by grid noise.
        for _ in range(4):
            for field in (magnetic, velocity):
                for component in range(3):
                    field[component] = 0.25 * (
                        np.roll(field[component], 1, 0)
                        + np.roll(field[component], -1, 0)
                        + np.roll(field[component], 1, 1)
                        + np.roll(field[component], -1, 1)
                    )

        def rotate(field):
            """Rotate the sampling grid and the vector components together."""
            rotated = np.stack([np.rot90(component) for component in field])
            return np.stack([-rotated[1], rotated[0], rotated[2]])

        base = flow_gradient_invariants(magnetic, velocity, SPACING, SPACING)
        turned = flow_gradient_invariants(
            rotate(magnetic), rotate(velocity), SPACING, SPACING
        )
        for name in INVARIANT_NAMES:
            np.testing.assert_allclose(
                turned[name], np.rot90(base[name]), rtol=1e-9, atol=1e-12
            )

    def test_independent_of_b_sign(self):
        """b -> -b must not change any of the four scalars."""
        rng = np.random.default_rng(4)
        magnetic = rng.normal(size=(3, GRID, GRID))
        velocity = rng.normal(size=(3, GRID, GRID))
        base = flow_gradient_invariants(magnetic, velocity, SPACING, SPACING)
        flipped = flow_gradient_invariants(-magnetic, velocity, SPACING, SPACING)
        for name in INVARIANT_NAMES:
            np.testing.assert_allclose(flipped[name], base[name], rtol=1e-10, atol=1e-14)


class TestNumericalHygiene:
    def test_finite_at_a_magnetic_null(self):
        rng = np.random.default_rng(5)
        velocity = rng.normal(size=(3, GRID, GRID))
        magnetic = np.zeros((3, GRID, GRID))
        invariants = flow_gradient_invariants(magnetic, velocity, SPACING, SPACING)
        for name in INVARIANT_NAMES:
            assert np.isfinite(invariants[name]).all()

    def test_perpendicular_block_never_negative_under_the_sqrt(self):
        """The tensor identity can go slightly negative in floating point."""
        rng = np.random.default_rng(6)
        for seed_shift in range(5):
            magnetic = rng.normal(size=(3, GRID, GRID)) * 10.0**seed_shift
            velocity = rng.normal(size=(3, GRID, GRID))
            invariants = flow_gradient_invariants(magnetic, velocity, SPACING, SPACING)
            assert (invariants["Wperp_e"] >= 0.0).all()
            assert np.isfinite(invariants["Wperp_e"]).all()


class TestModelIntegration:
    """The Tier-2 channels must reach the trunk without breaking export."""

    @staticmethod
    def _model(**kwargs):
        import torch

        from closure.models import InvariantFieldAlignedPressureMLP

        defaults = dict(
            feature_dims=[8, 16, 6],
            activations=["SiLU", None],
            extra_invariant_indices=[10, 11, 12, 13],
            extra_invariant_scales=[0.1, 0.1, 0.1, 0.1],
        )
        defaults.update(kwargs)
        torch.manual_seed(0)
        return InvariantFieldAlignedPressureMLP(**defaults)

    def test_trunk_width_must_match_the_invariants_used(self):
        import torch  # noqa: F401

        with pytest.raises(ValueError, match="8 invariant inputs"):
            self._model(feature_dims=[4, 16, 6])
        # Dropping the circular electron-frame pair narrows the trunk by two.
        with pytest.raises(ValueError, match="6 invariant inputs"):
            self._model(use_electron_frame_invariants=False)

    def test_forward_accepts_the_extra_channels(self):
        import torch

        model = self._model()
        features = torch.randn(64, 14)
        out = model(features)
        assert out.shape == (64, 6)
        assert torch.isfinite(out).all()

    def test_deployable_variant_ignores_the_electric_channels(self):
        """With the electron-frame pair off, E must not influence predictions."""
        import torch

        model = self._model(feature_dims=[6, 16, 6], use_electron_frame_invariants=False)
        features = torch.randn(32, 14)
        baseline = model(features)
        perturbed = features.clone()
        perturbed[:, 7:10] += 5.0  # Ex, Ey, Ez
        torch.testing.assert_close(model(perturbed), baseline)

    def test_extra_channels_do_influence_predictions(self):
        import torch

        model = self._model(feature_dims=[6, 16, 6], use_electron_frame_invariants=False)
        features = torch.randn(32, 14)
        perturbed = features.clone()
        perturbed[:, 10:14] += 1.0
        assert not torch.allclose(model(perturbed), model(features))

    def test_torchscript_export_with_extras(self):
        import torch

        model = self._model().eval()
        scripted = torch.jit.script(model)
        features = torch.randn(16, 14)
        torch.testing.assert_close(scripted(features), model(features))

    def test_predictions_stay_positive_definite(self):
        import numpy as np
        import torch

        model = self._model().eval()
        packed = model(torch.randn(128, 14) * 3.0).detach().numpy()
        for row in packed:
            tensor = np.array([
                [row[0], row[3], row[4]],
                [row[3], row[1], row[5]],
                [row[4], row[5], row[2]],
            ])
            assert np.linalg.eigvalsh(tensor).min() > 0.0
