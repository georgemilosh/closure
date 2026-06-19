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


class TestCode2AlfvenFallback:
    def test_code2alfven_reads_b0x_and_nb_from_experiment_inp(self, tmp_path):
        run_dir = tmp_path / "iPiC3D-nathan" / "Le2DHGEM_RunID_1"
        run_dir.mkdir(parents=True)
        (run_dir / "RunID_1.inp").write_text(
            "\n".join(
                [
                    "B0x = 0.0249",
                    "rhoINIT = 0.969 0.969 0.23 0.23",
                ]
            )
        )

        arr = np.ones((2, 2, 1), dtype=float)
        data = {
            "Bx": arr.copy(),
            "By": arr.copy(),
            "Bz": arr.copy(),
            "Bmagn": arr.copy(),
            "Emagn": arr.copy(),
            "rho": {"e": arr.copy()},
        }
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 2.0])
        times = [0.0, 1.0]

        x_out, y_out, t_out = plasma.code2alfven(
            data,
            x,
            y,
            times,
            experiment=str(run_dir.resolve()),
        )

        np.testing.assert_allclose(x_out, x * np.sqrt(0.969))
        np.testing.assert_allclose(y_out, y * np.sqrt(0.969))
        np.testing.assert_allclose(t_out, [0.0, 0.0249])

    def test_code2alfven_uses_explicit_b0x_and_infers_only_nb(self, tmp_path):
        run_dir = tmp_path / "Le2DHGEM_RunID_1"
        run_dir.mkdir(parents=True)
        (run_dir / "RunID_1.inp").write_text(
            "\n".join(
                [
                    "B0x = 0.0249",
                    "rhoINIT = 0.969 0.969 0.23 0.23",
                ]
            )
        )

        arr = np.ones((2, 2, 1), dtype=float)
        data = {
            "Bx": arr.copy(),
            "By": arr.copy(),
            "Bz": arr.copy(),
            "Bmagn": arr.copy(),
            "Emagn": arr.copy(),
            "rho": {"e": arr.copy()},
        }

        _, _, t_out = plasma.code2alfven(
            data,
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            [1.0],
            b0x=0.05,
            nb=None,
            experiment=str(run_dir),
        )

        np.testing.assert_allclose(t_out, [0.05])

    def test_find_experiment_inp_file_rejects_relative_path(self):
        with np.testing.assert_raises(ValueError):
            plasma._find_experiment_inp_file("Le2DHGEM_RunID_1")


class TestCode2AlfvenDataOnly:
    """Test code2alfven when x, y, times are omitted (data-only mode)."""

    def test_code2alfven_without_coordinates(self):
        """Data dict is scaled; returns (None, None, None)."""
        b0x, nb = 0.0249, 0.23
        va = b0x / np.sqrt(nb)
        p0 = nb * va ** 2

        arr = np.full((2, 2, 1), 3.0)
        data = {
            "Bx": arr.copy(),
            "By": arr.copy(),
            "Bz": arr.copy(),
            "Bmagn": arr.copy(),
            "Emagn": arr.copy(),
            "rho": {"e": arr.copy()},
            "Pxx": {"e": arr.copy()},
        }

        x_out, y_out, t_out = plasma.code2alfven(data, b0x=b0x, nb=nb)

        assert x_out is None
        assert y_out is None
        assert t_out is None
        np.testing.assert_allclose(data["Bx"], 3.0 / b0x)
        np.testing.assert_allclose(data["rho"]["e"], 3.0 / nb)
        np.testing.assert_allclose(data["Pxx"]["e"], 3.0 / p0)

    def test_code2alfven_normalize_density_false_keeps_rho_raw(self):
        """normalize_density=False leaves rho in code units; B/P still scaled."""
        b0x, nb = 0.0249, 0.23
        va = b0x / np.sqrt(nb)
        p0 = nb * va ** 2
        arr = np.full((2, 2, 1), 3.0)
        data = {
            "Bx": arr.copy(), "By": arr.copy(), "Bz": arr.copy(),
            "rho": {"e": arr.copy()}, "Pxx": {"e": arr.copy()},
        }
        x_out, y_out, t_out = plasma.code2alfven(
            data, x=np.arange(2.0), times=[1.0], b0x=b0x, nb=nb, normalize_density=False
        )
        np.testing.assert_allclose(data["rho"]["e"], 3.0)  # unchanged
        np.testing.assert_allclose(data["Bx"], 3.0 / b0x)  # still normalized
        np.testing.assert_allclose(data["Pxx"]["e"], 3.0 / p0)
        np.testing.assert_allclose(x_out, np.arange(2.0) * np.sqrt(nb))  # length still scaled
        np.testing.assert_allclose(t_out, [1.0 * b0x])  # time still scaled


class TestAlfvenScales:
    def test_alfven_scales_values(self):
        scales = plasma.alfven_scales(0.0249, 0.23)
        assert scales["b0x"] == 0.0249
        assert scales["nb"] == 0.23
        np.testing.assert_allclose(scales["va"], 0.0249 / np.sqrt(0.23))
        np.testing.assert_allclose(scales["p0"], 0.23 * scales["va"] ** 2)
        np.testing.assert_allclose(scales["e0"], scales["va"] * 0.0249)
        np.testing.assert_allclose(scales["j0"], 0.23 * scales["va"])
