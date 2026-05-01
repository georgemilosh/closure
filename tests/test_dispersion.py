"""Tests for closure.dispersion."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from closure.dispersion import (
    HallMHDBackground,
    apply_closure_correction,
    build_dispersion_matrix,
    build_hall_mhd_operator,
    closure_tensor_jacobian_at_equilibrium,
    electron_pressure_tensor_to_electric_jacobian,
    fourier_mode_vector,
    isotropic_electron_closure_electric_jacobian,
    linearize_spatial_model,
    linearize_spatial_model_2d,
    match_eigenbranches,
    project_fourier_jacobian,
    project_fourier_jacobian_2d,
    scan_dispersion_relation,
)


class TestFourierModeVector:
    def test_returns_unit_modulus_mode(self):
        mode = fourier_mode_vector(8, 3)
        assert mode.shape == (8,)
        np.testing.assert_allclose(np.abs(mode), 1.0)

    def test_rejects_non_positive_grid(self):
        with pytest.raises(ValueError, match="positive"):
            fourier_mode_vector(0, 1)


class TestProjectFourierJacobian:
    def test_projects_single_output_kernel(self):
        n_grid = 8
        kernel = np.zeros((n_grid, 2, n_grid), dtype=np.complex128)

        for offset, weight in {0: 2.0, 1: -0.5}.items():
            for out_idx in range(n_grid):
                kernel[out_idx, 0, (out_idx - offset) % n_grid] += weight

        for offset, weight in {0: -1.0, 2: 0.25}.items():
            for out_idx in range(n_grid):
                kernel[out_idx, 1, (out_idx - offset) % n_grid] += weight

        coeffs = project_fourier_jacobian(kernel, k_index=1)
        expected = np.array(
            [
                2.0 - 0.5 * np.exp(-2.0j * np.pi / n_grid),
                -1.0 + 0.25 * np.exp(-4.0j * np.pi / n_grid),
            ]
        )
        np.testing.assert_allclose(coeffs, expected)

    def test_projects_multi_output_kernel(self):
        n_grid = 6
        kernel = np.zeros((n_grid, 2, 2, n_grid), dtype=np.complex128)
        for out_idx in range(n_grid):
            kernel[out_idx, 0, 0, out_idx] = 1.5
            kernel[out_idx, 0, 1, (out_idx - 1) % n_grid] = -0.75
            kernel[out_idx, 1, 0, (out_idx - 2) % n_grid] = 0.5
            kernel[out_idx, 1, 1, out_idx] = -2.0

        coeffs = project_fourier_jacobian(kernel, k_index=2)
        expected = np.array(
            [
                [1.5, -0.75 * np.exp(-4.0j * np.pi / n_grid)],
                [0.5 * np.exp(-8.0j * np.pi / n_grid), -2.0],
            ]
        )
        np.testing.assert_allclose(coeffs, expected)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="jacobian"):
            project_fourier_jacobian(np.zeros((4, 4)), k_index=0)


class _ToySpatialModel:
    def __call__(self, x):
        y0 = 2.0 * x[:, 0] - 0.5 * torch.roll(x[:, 1], shifts=1, dims=1)
        y1 = -1.0 * x[:, 1] + 0.25 * torch.roll(x[:, 0], shifts=-2, dims=1)
        return torch.stack([y0, y1], dim=1)


class TestLinearizeSpatialModel:
    def test_recovers_spatial_jacobian_for_toy_model(self):
        n_grid = 8
        eq = torch.zeros((1, 2, n_grid, 1), dtype=torch.float64)
        jac = linearize_spatial_model(_ToySpatialModel(), eq)

        assert jac.shape == (n_grid, 2, 2, n_grid)

        coeffs = project_fourier_jacobian(jac, k_index=1)
        expected = np.array(
            [
                [2.0, -0.5 * np.exp(-2.0j * np.pi / n_grid)],
                [0.25 * np.exp(4.0j * np.pi / n_grid), -1.0],
            ]
        )
        np.testing.assert_allclose(coeffs, expected)

    def test_rejects_non_1d_feature_layout(self):
        with pytest.raises(ValueError, match="ny=1"):
            linearize_spatial_model(_ToySpatialModel(), torch.zeros((1, 2, 4, 2)))


class TestApplyClosureCorrection:
    def test_subtracts_block_from_selected_rows(self):
        operator = np.eye(4, dtype=np.complex128)
        block = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -0.5, -1.0]])
        corrected = apply_closure_correction(operator, [1, 3], block)

        expected = np.eye(4, dtype=np.complex128)
        expected[1] -= block[0]
        expected[3] -= block[1]
        np.testing.assert_allclose(corrected, expected)

    def test_rejects_mismatched_block_shape(self):
        with pytest.raises(ValueError, match="closure_block"):
            apply_closure_correction(np.eye(3), [1], np.ones((2, 3)))


class TestBuildDispersionMatrix:
    def test_assembles_flux_source_and_closure_terms(self):
        flux = np.array([[1.0, 2.0], [3.0, 4.0]])
        source = np.array([[0.5, -0.5], [1.0, 0.0]])
        closure = np.array([[0.25, 0.75]])
        matrix = build_dispersion_matrix(
            flux,
            k_phys=0.5,
            source_jacobian=source,
            closure_rows=[1],
            closure_block=closure,
        )
        expected = (-0.5j) * flux + source
        expected[1] -= closure[0]
        np.testing.assert_allclose(matrix, expected)

    def test_requires_both_closure_arguments(self):
        with pytest.raises(ValueError, match="provided together"):
            build_dispersion_matrix(np.eye(2), k_phys=1.0, closure_rows=[0])


class TestMatchEigenbranches:
    def test_recovers_candidate_permutation(self):
        reference = np.eye(3, dtype=np.complex128)
        candidate = reference[:, [2, 0, 1]]

        order, overlaps = match_eigenbranches(reference, candidate)
        np.testing.assert_array_equal(order, np.array([1, 2, 0]))
        np.testing.assert_allclose(overlaps, np.ones(3))

    def test_rejects_zero_columns(self):
        reference = np.eye(2, dtype=np.complex128)
        candidate = reference.copy()
        candidate[:, 1] = 0.0

        with pytest.raises(ValueError, match="non-zero"):
            match_eigenbranches(reference, candidate)


class TestHallMHDOperator:
    def test_zero_wavevector_gives_zero_operator(self):
        background = HallMHDBackground(rho0=2.0, B0=(1.0, 0.0, 0.0))
        operator = build_hall_mhd_operator(background, (0.0, 0.0))
        np.testing.assert_allclose(operator, 0.0)

    def test_recovers_parallel_hall_coupling_entries(self):
        background = HallMHDBackground(rho0=2.0, B0=(3.0, 0.0, 0.0))
        operator = build_hall_mhd_operator(background, (4.0, 0.0), hall_scale=0.5)

        np.testing.assert_allclose(operator[0, 1], -8.0j)
        np.testing.assert_allclose(operator[2, 5], 6.0j)
        np.testing.assert_allclose(operator[5, 2], 12.0j)
        np.testing.assert_allclose(operator[5, 6], -12.0)

    def test_isothermal_closure_is_curl_free_at_uniform_density(self):
        background = HallMHDBackground(rho0=5.0, B0=(1.0, -0.5, 0.0))
        closure = isotropic_electron_closure_electric_jacobian(
            (2.0, 3.0),
            background,
            sound_speed_sq=1.7,
        )

        base_operator = build_hall_mhd_operator(background, (2.0, 3.0))
        closure_operator = build_hall_mhd_operator(
            background,
            (2.0, 3.0),
            closure_electric_jacobian=closure,
        )
        np.testing.assert_allclose(closure_operator, base_operator)

    def test_tensor_closure_projects_into_induction_operator(self):
        background = HallMHDBackground(rho0=4.0, B0=(0.0, 0.0, 1.0))
        tensor_jac = np.zeros((6, 7), dtype=np.complex128)
        tensor_jac[0, 0] = 2.5

        closure = electron_pressure_tensor_to_electric_jacobian(
            (2.0, 3.0),
            background,
            tensor_jac,
        )
        operator = build_hall_mhd_operator(
            background,
            (2.0, 3.0),
            closure_electric_jacobian=closure,
        )

        expected_bz_rho = 2.5 * 2.0 * 3.0 / background.rho0
        np.testing.assert_allclose(operator[6, 0], expected_bz_rho)


class TestProjectFourierJacobian2D:
    def test_diagonal_kernel_gives_constant_coefficient(self):
        nx, ny, n_in = 4, 6, 2
        # Local (diagonal in space) kernel: K[x,y,i,x',y'] = A[i] * delta(x,x') * delta(y,y')
        kernel = np.zeros((nx, ny, n_in, nx, ny), dtype=np.complex128)
        for x in range(nx):
            for y in range(ny):
                kernel[x, y, 0, x, y] = 3.0
                kernel[x, y, 1, x, y] = -1.5
        for kx_idx, ky_idx in [(0, 0), (1, 2), (3, 5)]:
            coeffs = project_fourier_jacobian_2d(kernel, kx_idx, ky_idx)
            np.testing.assert_allclose(coeffs, np.array([3.0, -1.5]), atol=1e-12)

    def test_x_shift_gives_phase_in_x(self):
        nx, ny, n_out, n_in = 8, 4, 1, 1
        # K shifts input by +1 in x: K[x,y,o,i,x',y'] = delta(x, x'+1 mod nx) * delta(y, y')
        kernel = np.zeros((nx, ny, n_out, n_in, nx, ny), dtype=np.complex128)
        for x in range(nx):
            for y in range(ny):
                kernel[x, y, 0, 0, (x - 1) % nx, y] = 1.0
        for kx_idx in range(1, nx):
            coeffs = project_fourier_jacobian_2d(kernel, kx_idx, 0)
            expected = np.exp(-2.0j * np.pi * kx_idx / nx)
            np.testing.assert_allclose(coeffs[0, 0], expected, atol=1e-12)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="jacobian"):
            project_fourier_jacobian_2d(np.zeros((4, 4, 4)), 0, 0)


class _ToySpatialModel2D:
    """2D version of the toy model: local scaling plus shifts in x and y.

    y0[x,y] = 2*u0[x,y]   - 0.5*u1[x-1, y]  (shift in x-direction, dims=1)
    y1[x,y] = -1*u1[x,y]  + 0.25*u0[x, y+1] (shift in y-direction, dims=2)
    """

    def __call__(self, x):
        # x shape: (1, 2, nx, ny); x[:,c] has shape (1, nx, ny)
        # dims=1 → x-direction, dims=2 → y-direction in (1,nx,ny)
        y0 = 2.0 * x[:, 0] - 0.5 * torch.roll(x[:, 1], shifts=1, dims=1)
        y1 = -1.0 * x[:, 1] + 0.25 * torch.roll(x[:, 0], shifts=-1, dims=2)
        return torch.stack([y0, y1], dim=1)


class TestLinearizeSpatialModel2D:
    def test_recovers_modal_coefficients(self):
        nx, ny = 8, 6
        eq = torch.zeros((1, 2, nx, ny), dtype=torch.float64)
        jac = linearize_spatial_model_2d(_ToySpatialModel2D(), eq)

        assert jac.shape == (nx, ny, 2, 2, nx, ny)

        # kx=1, ky=0: shift in x only matters for roll(x, dims=2)
        coeffs = project_fourier_jacobian_2d(jac, 1, 0)
        expected = np.array(
            [
                [2.0, -0.5 * np.exp(-2.0j * np.pi / nx)],
                [0.25, -1.0],
            ]
        )
        np.testing.assert_allclose(coeffs, expected, atol=1e-10)

        # kx=0, ky=1: shift in y only matters for roll(x, dims=3)
        coeffs_ky = project_fourier_jacobian_2d(jac, 0, 1)
        expected_ky = np.array(
            [
                [2.0, -0.5],
                [0.25 * np.exp(2.0j * np.pi / ny), -1.0],
            ]
        )
        np.testing.assert_allclose(coeffs_ky, expected_ky, atol=1e-10)


class _ToyDataset:
    """Minimal dataset stub for closure_tensor_jacobian_at_equilibrium tests."""

    def __init__(self):
        self.request_features = ["rho", "Bx"]
        self.request_targets = ["Pxx", "Pyy"]
        self.scaler_features = True
        self.scaler_targets = True
        self.features_mean = np.array([1.0, 0.5], dtype=float)
        self.features_std = np.array([0.2, 0.1], dtype=float)
        self.targets_mean = np.array([0.3, 0.4], dtype=float)
        self.targets_std = np.array([0.05, 0.08], dtype=float)
        self.prescaler_features = [None, None]
        self.prescaler_targets = [None, None]


class _LinearToyModel(torch.nn.Module):
    """Linear model y = W @ x (in normalized space) for exact Jacobian checks."""

    def __init__(self, W):
        super().__init__()
        W_tensor = torch.tensor(W, dtype=torch.float32)
        self.register_buffer("W", W_tensor)

    def forward(self, x):
        # x: (1, n_feat, 1, 1) → y: (1, n_target, 1, 1)
        x_flat = x.reshape(x.shape[0], -1, 1)  # (1, n_feat, 1)
        y_flat = torch.matmul(self.W, x_flat)  # (1, n_target, 1)
        return y_flat.reshape(x.shape[0], -1, 1, 1)

    def eval(self):
        return self


class TestClosureTensorJacobianAtEquilibrium:
    def test_linear_model_recovers_exact_physical_jacobian(self):
        ds = _ToyDataset()
        W_norm = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=float)
        model = _LinearToyModel(W_norm)

        eq = np.array([1.2, 0.6], dtype=float)  # physical equilibrium
        jac, target_names, feature_names = closure_tensor_jacobian_at_equilibrium(
            model, ds, eq
        )

        # Physical Jacobian = diag(t_std) @ W_norm @ diag(1/f_std)
        expected = np.diag(ds.targets_std) @ W_norm @ np.diag(1.0 / ds.features_std)
        np.testing.assert_allclose(jac, expected, rtol=1e-4)
        assert target_names == ds.request_targets
        assert feature_names == ds.request_features

    def test_rejects_wrong_equilibrium_length(self):
        ds = _ToyDataset()
        model = _LinearToyModel(np.eye(2))
        with pytest.raises(ValueError, match="shape"):
            closure_tensor_jacobian_at_equilibrium(model, ds, [1.0, 2.0, 3.0])


class TestScanDispersionRelation:
    def test_returns_correct_shape(self):
        background = HallMHDBackground(rho0=1.0, B0=(1.0, 0.0, 0.0))
        result = scan_dispersion_relation(
            background,
            k_magnitudes=[0.5, 1.0, 2.0],
            angles=[0.0, np.pi / 4],
        )
        assert result["eigenvalues"].shape == (3, 2, 7)
        assert result["k_magnitudes"].shape == (3,)
        assert result["angles"].shape == (2,)

    def test_zero_k_all_eigenvalues_zero(self):
        background = HallMHDBackground(rho0=2.0, B0=(1.0, 0.0, 0.0))
        result = scan_dispersion_relation(background, [0.0], [0.0])
        np.testing.assert_allclose(result["eigenvalues"][0, 0], 0.0)

    def test_closure_fn_changes_eigenvalues(self):
        background = HallMHDBackground(rho0=1.0, B0=(1.0, 0.0, 0.0))

        def closure_fn(kvec, bg):
            return isotropic_electron_closure_electric_jacobian(
                kvec, bg, sound_speed_sq=0.5
            )

        base = scan_dispersion_relation(background, [1.0], [0.3])
        with_closure = scan_dispersion_relation(
            background, [1.0], [0.3], closure_fn=closure_fn
        )
        # isothermal closure is curl-free for uniform rho0, so eigenvalues match
        np.testing.assert_allclose(
            with_closure["eigenvalues"], base["eigenvalues"], atol=1e-12
        )