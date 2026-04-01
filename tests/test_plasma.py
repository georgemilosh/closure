"""Tests for closure.plasma."""

from __future__ import annotations

import numpy as np

from closure import plasma


def _sample_data(nx: int = 8, ny: int = 8, nt: int = 2):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    times = np.arange(nt)
    field = np.repeat((xx + yy)[..., None], nt, axis=2)

    data = {
        "Bx": np.ones((nx, ny, nt)),
        "By": np.ones((nx, ny, nt)) * 0.5,
        "Bz": np.ones((nx, ny, nt)) * 0.25,
        "Ex": np.ones((nx, ny, nt)) * 0.1,
        "Ey": np.ones((nx, ny, nt)) * 0.2,
        "Ez": np.ones((nx, ny, nt)) * 0.3,
        "rho": {"e": -np.ones((nx, ny, nt)), "i": np.ones((nx, ny, nt))},
        "Jx": {"e": np.ones((nx, ny, nt)) * -0.2, "i": np.ones((nx, ny, nt)) * 0.4},
        "Jy": {"e": np.ones((nx, ny, nt)) * -0.1, "i": np.ones((nx, ny, nt)) * 0.2},
        "Jz": {"e": np.ones((nx, ny, nt)) * -0.05, "i": np.ones((nx, ny, nt)) * 0.1},
        "Vx": {"e": field.copy(), "i": field.copy() * 0.5},
        "Vy": {"e": field.copy() * 0.5, "i": field.copy() * 0.25},
        "Vz": {"e": field.copy() * 0.25, "i": field.copy() * 0.125},
        "Pxx": {"e": np.ones((nx, ny, nt)) * 2.0, "i": np.ones((nx, ny, nt)) * 1.0},
        "Pxy": {"e": np.ones((nx, ny, nt)) * 0.1, "i": np.ones((nx, ny, nt)) * 0.05},
        "Pxz": {"e": np.ones((nx, ny, nt)) * 0.1, "i": np.ones((nx, ny, nt)) * 0.05},
        "Pyy": {"e": np.ones((nx, ny, nt)) * 2.5, "i": np.ones((nx, ny, nt)) * 1.5},
        "Pyz": {"e": np.ones((nx, ny, nt)) * 0.1, "i": np.ones((nx, ny, nt)) * 0.05},
        "Pzz": {"e": np.ones((nx, ny, nt)) * 3.0, "i": np.ones((nx, ny, nt)) * 2.0},
        "Ppar": {"e": np.ones((nx, ny, nt)), "i": np.ones((nx, ny, nt))},
        "Pperp": {"e": np.ones((nx, ny, nt)), "i": np.ones((nx, ny, nt))},
    }
    return x, y, data


class TestVectorHelpers:
    def test_do_dot(self):
        out = plasma.do_dot(np.array([1, 2]), np.array([3, 4]), np.array([5, 6]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]))
        np.testing.assert_allclose(out, np.array([9, 12]))

    def test_do_cross(self):
        cx, cy, cz = plasma.do_cross(1, 0, 0, 0, 1, 0)
        assert (cx, cy, cz) == (0, 0, 1)


class TestFiniteDifference:
    def test_highdiff_matches_linear_derivative(self):
        x = np.linspace(0.0, 1.0, 16)
        y = np.linspace(0.0, 1.0, 16)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        field = 3.0 * xx + 2.0 * yy
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dfdx = plasma.highdiff(field, dx, dy, axis=0, mode="nearest")
        dfdy = plasma.highdiff(field, dx, dy, axis=1, mode="nearest")
        np.testing.assert_allclose(dfdx[2:-2, 2:-2], 3.0, atol=1e-6)
        np.testing.assert_allclose(dfdy[2:-2, 2:-2], 2.0, atol=1e-6)


class TestDiagnostics:
    def test_get_ohm_populates_expected_keys(self):
        x, y, data = _sample_data()
        plasma.get_Ohm(data, [-1.0, 1.0], x, y)
        for key in ["EHallx", "EHally", "EHallz", "EMHDx", "EMHDy", "EMHDz", "EPx", "EPy", "EPz"]:
            assert key in data

    def test_get_ps_2d_field_populates_structure(self):
        x, y, data = _sample_data()
        plasma.get_PS_2D_field(data, x, y)
        assert "PiD" in data
        assert "e" in data["PiD"]
        assert data["PiD"]["e"].shape == data["Bx"].shape


class TestSpectra:
    def test_scalar_spectrum_2d_returns_nonnegative_spectrum(self):
        n = 16
        x = np.linspace(0.0, 2 * np.pi, n)
        y = np.linspace(0.0, 2 * np.pi, n)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        field = np.sin(xx)[..., None]
        ky, spec = plasma.scalar_spectrum_2D(field, x, y)
        assert ky.ndim == 1
        assert spec.ndim == 2
        assert np.all(spec >= 0)

    def test_get_spectral_index_on_power_law(self):
        k = np.arange(1, 17, dtype=float)
        spec = k ** -2
        k_reduced, slopes = plasma.get_spectral_index(k, spec, 4)
        assert k_reduced.size == slopes.size
        assert np.all(np.isfinite(slopes))
