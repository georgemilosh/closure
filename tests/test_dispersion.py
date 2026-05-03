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
    closure_tensor_fourier_symbol_at_equilibrium,
    closure_tensor_jacobian_at_equilibrium,
    electron_pressure_tensor_to_electric_jacobian,
    fourier_mode_vector,
    hall_mhd_k_vector,
    isotropic_electron_closure_electric_jacobian,
    linearize_spatial_model,
    linearize_spatial_model_2d,
    match_eigenbranches,
    mode_indices_from_physical_wavenumber,
    patch_domain_lengths_from_grid,
    physical_wavenumber_from_mode_indices,
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


class TestFourierModeCalibration:
    def test_patch_domain_lengths_use_simulation_cell_spacing(self):
        lengths = patch_domain_lengths_from_grid(
            patch_shape=(32, 16),
            simulation_domain_lengths=(20.0, 10.0),
            simulation_grid_shape=(512, 256),
        )
        np.testing.assert_allclose(lengths, (1.25, 0.625))

    def test_mode_indices_and_physical_wavenumber_round_trip(self):
        domain_lengths = (1.25, 2.5)
        mode = (2, -3)
        kvec = physical_wavenumber_from_mode_indices(mode, domain_lengths)

        np.testing.assert_allclose(
            kvec,
            (2.0 * 2.0 * np.pi / 1.25, -3.0 * 2.0 * np.pi / 2.5, 0.0),
        )
        assert mode_indices_from_physical_wavenumber(kvec, domain_lengths, patch_shape=(32, 32)) == mode

    def test_mode_indices_reject_patch_nyquist_exceedance(self):
        domain_lengths = (1.0, 1.0)
        kvec = physical_wavenumber_from_mode_indices((9, 0), domain_lengths)

        with pytest.raises(ValueError, match="Nyquist"):
            mode_indices_from_physical_wavenumber(kvec, domain_lengths, patch_shape=(16, 16))


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

    def test_tensor_closure_uses_full_3d_pressure_divergence(self):
        background = HallMHDBackground(rho0=2.0, B0=(1.0, 0.0, 0.25))
        tensor_jac = np.zeros((6, 1), dtype=np.complex128)
        tensor_jac[:, 0] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        closure = electron_pressure_tensor_to_electric_jacobian(
            (0.5, -0.25, 0.75),
            background,
            tensor_jac,
        )

        expected = (-1j / background.rho0) * np.array(
            [
                0.5 * 1.0 - 0.25 * 2.0 + 0.75 * 3.0,
                0.5 * 2.0 - 0.25 * 4.0 + 0.75 * 5.0,
                0.5 * 3.0 - 0.25 * 5.0 + 0.75 * 6.0,
            ]
        )
        np.testing.assert_allclose(closure[:, 0], expected)


class TestHallMHDKVector:
    def test_simulation_plane_geometry_has_zero_kz(self):
        background = HallMHDBackground(rho0=1.0, B0=(-2.0, 0.5, 1.5))
        kvec = hall_mhd_k_vector(background, 3.0, np.pi / 3, geometry="simulation_plane")

        np.testing.assert_allclose(np.linalg.norm(kvec), 3.0)
        np.testing.assert_allclose(kvec[2], 0.0)

    def test_field_aligned_geometry_parallel_at_zero_angle(self):
        background = HallMHDBackground(rho0=1.0, B0=(-2.0, 0.5, 1.5))
        kvec = hall_mhd_k_vector(background, 3.0, 0.0, geometry="field_aligned")
        b0 = np.asarray(background.B0, dtype=float)

        np.testing.assert_allclose(kvec / np.linalg.norm(kvec), b0 / np.linalg.norm(b0))

    def test_simulation_plane_requires_in_plane_field(self):
        background = HallMHDBackground(rho0=1.0, B0=(0.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="x-y projection"):
            hall_mhd_k_vector(background, 1.0, 0.0, geometry="simulation_plane")


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


class _ToySpatialModel2DModule(torch.nn.Module):
    def forward(self, x):
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


class TestClosureTensorFourierSymbolAtEquilibrium:
    def test_recovers_projected_symbol_without_full_jacobian(self):
        ds = _ToyDataset()
        model = _ToySpatialModel2DModule()
        eq = np.array([1.2, 0.6], dtype=float)
        nx, ny = 8, 6

        symbol, target_names, feature_names = closure_tensor_fourier_symbol_at_equilibrium(
            model,
            ds,
            eq,
            mode_indices=(1, 0),
            patch_shape=(nx, ny),
        )

        normalized_symbol = np.array(
            [
                [2.0, -0.5 * np.exp(-2.0j * np.pi / nx)],
                [0.25, -1.0],
            ]
        )
        expected = np.diag(ds.targets_std) @ normalized_symbol @ np.diag(1.0 / ds.features_std)
        np.testing.assert_allclose(symbol, expected, atol=1e-6)
        assert target_names == ds.request_targets
        assert feature_names == ds.request_features

    def test_finite_difference_path_matches_jvp(self):
        ds = _ToyDataset()
        model = _ToySpatialModel2DModule()
        eq = np.array([1.2, 0.6], dtype=float)

        symbol_jvp, _, _ = closure_tensor_fourier_symbol_at_equilibrium(
            model,
            ds,
            eq,
            mode_indices=(0, 1),
            patch_shape=(6, 6),
            method="jvp",
        )
        symbol_fd, _, _ = closure_tensor_fourier_symbol_at_equilibrium(
            model,
            ds,
            eq,
            mode_indices=(0, 1),
            patch_shape=(6, 6),
            method="finite_difference",
            finite_difference_eps=1e-3,
        )

        np.testing.assert_allclose(symbol_fd, symbol_jvp, atol=1e-4)


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

    def test_simulation_plane_geometry_passes_kz_zero_to_closure(self):
        background = HallMHDBackground(rho0=1.0, B0=(1.0, 0.0, 2.0))
        seen = []

        def closure_fn(kvec, bg):
            seen.append(np.asarray(kvec, dtype=float))
            return np.zeros((3, 7), dtype=np.complex128)

        scan_dispersion_relation(
            background,
            [1.0],
            [0.4],
            closure_fn=closure_fn,
            geometry="simulation_plane",
        )
        np.testing.assert_allclose(seen[0][2], 0.0)

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