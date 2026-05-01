"""Helpers for reduced-fluid dispersion analysis with learned closures.

This module intentionally starts small: it provides the generic linear-algebra
building blocks shared by the existing notebook workflows.

- Local closures feed directly into an operator correction matrix.
- Nonlocal closures are first linearized as a spatial Jacobian and then
  projected onto a single Fourier mode.

The higher-level reduced-fluid model can be built on top of these primitives
without coupling notebook-specific logic into the package API.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import torch
except ImportError:  # pragma: no cover - torch is a package dependency in normal use.
    torch = None


def _as_3vector(values, *, name: str) -> np.ndarray:
    """Return a real 3-vector from length-2 or length-3 input."""
    vec = np.asarray(values, dtype=float)
    if vec.shape == (2,):
        return np.array([vec[0], vec[1], 0.0], dtype=float)
    if vec.shape == (3,):
        return vec.astype(float, copy=False)
    raise ValueError(f"{name} must be a length-2 or length-3 vector")


@dataclass(frozen=True)
class HallMHDBackground:
    """Uniform background state for reduced Hall-MHD dispersion analysis.

    Parameters
    ----------
    rho0 : float
        Uniform mass density, used as the Hall-MHD number density scale.
    u0 : array-like, shape ``(2,)`` or ``(3,)``
        Background ion-fluid velocity.
    B0 : array-like, shape ``(2,)`` or ``(3,)``
        Background magnetic field.
    """

    rho0: float
    u0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B0: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self):
        if self.rho0 <= 0.0:
            raise ValueError("rho0 must be positive")
        object.__setattr__(self, "u0", tuple(_as_3vector(self.u0, name="u0")))
        object.__setattr__(self, "B0", tuple(_as_3vector(self.B0, name="B0")))


def isotropic_electron_closure_electric_jacobian(
    kvec,
    background: HallMHDBackground,
    *,
    sound_speed_sq: float,
) -> np.ndarray:
    r"""Electric-field Jacobian for a barotropic electron-pressure closure.

    This represents

    .. math::

        \delta E_{pe} = -\frac{1}{\rho_0} i k \, c_e^2 \, \delta \rho,

    in the primitive-state basis
    ``(rho, ux, uy, uz, Bx, By, Bz)``.

    For uniform ``rho0`` this term is parallel to ``k`` and therefore does not
    modify the induction equation after applying ``-i k x``. The helper is
    still useful because it makes that cancellation explicit in tests and in
    higher-level diagnostics.
    """
    k3 = _as_3vector(kvec, name="kvec")
    jac = np.zeros((3, 7), dtype=np.complex128)
    jac[:, 0] = -(1j * float(sound_speed_sq) / float(background.rho0)) * k3
    return jac


def electron_pressure_tensor_to_electric_jacobian(
    kvec,
    background: HallMHDBackground,
    tensor_jacobian: np.ndarray,
) -> np.ndarray:
    r"""Map a pressure-tensor Jacobian to the linearized electron electric field.

    Parameters
    ----------
    kvec : array-like, shape ``(2,)`` or ``(3,)``
        Fourier wavevector. Only the in-plane ``kx`` and ``ky`` components are
        used by the current 2D reduction.
    background : HallMHDBackground
        Uniform Hall-MHD equilibrium.
    tensor_jacobian : ndarray, shape ``(6, n_state)``
        Jacobian of the symmetric electron pressure tensor components with
        respect to the reduced primitive state. The component order is
        ``(Pxx, Pxy, Pxz, Pyy, Pyz, Pzz)``.

    Returns
    -------
    ndarray
        Electric-field Jacobian with shape ``(3, n_state)`` for the closure
        term ``-(\nabla \cdot P_e) / rho0``.
    """
    tensor = np.asarray(tensor_jacobian, dtype=np.complex128)
    if tensor.ndim != 2 or tensor.shape[0] != 6:
        raise ValueError("tensor_jacobian must have shape (6, n_state)")

    k3 = _as_3vector(kvec, name="kvec")
    kx, ky = k3[0], k3[1]
    scale = -1j / float(background.rho0)
    e_jac = np.zeros((3, tensor.shape[1]), dtype=np.complex128)
    e_jac[0] = scale * (kx * tensor[0] + ky * tensor[1])
    e_jac[1] = scale * (kx * tensor[1] + ky * tensor[3])
    e_jac[2] = scale * (kx * tensor[2] + ky * tensor[4])
    return e_jac


def build_hall_mhd_operator(
    background: HallMHDBackground,
    kvec,
    *,
    hall_scale: float = 1.0,
    closure_electric_jacobian: np.ndarray | None = None,
    ion_pressure_jacobian: np.ndarray | None = None,
    resistivity: float = 0.0,
) -> np.ndarray:
    r"""Build the 2D Hall-MHD linear operator in primitive variables.

    The primitive-state ordering is
    ``(rho, ux, uy, uz, Bx, By, Bz)`` and the linearized system is evaluated
    about a uniform equilibrium with zero background current.

    The model uses cold ions by default. Any prescribed electron closure enters
    through the generalized Ohm law as an electric-field correction matrix
    ``closure_electric_jacobian``. An optional ion-pressure-force Jacobian can
    also be supplied as a ``(3, 7)`` matrix acting on the same primitive state.
    """
    k3 = _as_3vector(kvec, name="kvec")
    b0 = np.asarray(background.B0, dtype=float)
    u0 = np.asarray(background.u0, dtype=float)
    rho0 = float(background.rho0)

    closure_jac = np.zeros((3, 7), dtype=np.complex128)
    if closure_electric_jacobian is not None:
        closure_jac = np.asarray(closure_electric_jacobian, dtype=np.complex128)
        if closure_jac.shape != (3, 7):
            raise ValueError("closure_electric_jacobian must have shape (3, 7)")

    ion_force = np.zeros((3, 7), dtype=np.complex128)
    if ion_pressure_jacobian is not None:
        ion_force = np.asarray(ion_pressure_jacobian, dtype=np.complex128)
        if ion_force.shape != (3, 7):
            raise ValueError("ion_pressure_jacobian must have shape (3, 7)")

    operator = np.zeros((7, 7), dtype=np.complex128)
    identity = np.eye(7, dtype=np.complex128)

    for col in range(7):
        q = identity[:, col]
        rho = q[0]
        u = q[1:4]
        b = q[4:7]

        delta_j = 1j * np.cross(k3, b)
        delta_e = -np.cross(u, b0) - np.cross(u0, b)
        delta_e = delta_e + (float(hall_scale) / rho0) * np.cross(delta_j, b0)
        delta_e = delta_e + closure_jac @ q
        if resistivity:
            delta_e = delta_e + float(resistivity) * delta_j

        drho = -1j * (rho0 * np.dot(k3, u) + rho * np.dot(k3, u0))
        du = -1j * np.dot(k3, u0) * u + np.cross(delta_j, b0) / rho0 - ion_force @ q
        db = -1j * np.cross(k3, delta_e)

        operator[:, col] = np.concatenate(([drho], du, db))

    return operator


def fourier_mode_vector(n_grid: int, k_index: int, *, dtype=np.complex128) -> np.ndarray:
    """Return the periodic Fourier basis vector for an integer mode index."""
    if n_grid <= 0:
        raise ValueError("n_grid must be positive")

    x_idx = np.arange(n_grid, dtype=float)
    phase = 2.0j * np.pi * float(k_index) * x_idx / float(n_grid)
    return np.exp(phase).astype(dtype, copy=False)


def project_fourier_jacobian_2d(
    jacobian: np.ndarray,
    kx_index: int,
    ky_index: int,
) -> np.ndarray:
    r"""Project a 2D translation-invariant Jacobian onto a single Fourier mode.

    Parameters
    ----------
    jacobian : ndarray
        Shape ``(nx, ny, n_in, nx, ny)`` (rank-5, single output) or
        ``(nx, ny, n_out, n_in, nx, ny)`` (rank-6, multiple outputs).
        The first two and last two axes are output/input spatial indices.
    kx_index, ky_index : int
        Integer mode indices along the *x* and *y* directions.

    Returns
    -------
    ndarray
        Modal transfer matrix with shape ``(n_in,)`` for rank-5 input or
        ``(n_out, n_in)`` for rank-6 input.
    """
    kernel = np.asarray(jacobian)
    squeeze_output = False

    if kernel.ndim == 5:
        kernel = kernel[:, :, np.newaxis, :, :, :]
        squeeze_output = True
    elif kernel.ndim != 6:
        raise ValueError(
            "jacobian must have shape (nx, ny, n_in, nx, ny) or "
            "(nx, ny, n_out, n_in, nx, ny)"
        )

    nx, ny, n_out, n_in, in_nx, in_ny = kernel.shape
    if nx != in_nx or ny != in_ny:
        raise ValueError(
            f"jacobian output and input spatial dimensions must match: "
            f"got ({nx},{ny}) vs ({in_nx},{in_ny})"
        )

    e_kx = fourier_mode_vector(nx, kx_index)
    e_ky = fourier_mode_vector(ny, ky_index)
    e_xy = np.outer(e_kx, e_ky)  # (nx, ny)

    coeffs = np.einsum(
        "xy,xyoiXY,XY->oi",
        np.conj(e_xy),
        kernel,
        e_xy,
        optimize=True,
    ) / float(nx * ny)

    return coeffs[0] if squeeze_output else coeffs


def project_fourier_jacobian(jacobian: np.ndarray, k_index: int) -> np.ndarray:
    r"""Project a translation-invariant spatial Jacobian onto one Fourier mode.

    Parameters
    ----------
    jacobian : ndarray
        Either ``(n_grid, n_in, n_grid)`` for a single output channel or
        ``(n_grid, n_out, n_in, n_grid)`` for multiple output channels.
        The first and last axes are physical-space output and input indices.
    k_index : int
        Integer Fourier mode index to project.

    Returns
    -------
    ndarray
        Modal transfer coefficients with shape ``(n_in,)`` for rank-3 input or
        ``(n_out, n_in)`` for rank-4 input.
    """
    kernel = np.asarray(jacobian)
    squeeze_output = False

    if kernel.ndim == 3:
        kernel = kernel[:, np.newaxis, :, :]
        squeeze_output = True
    elif kernel.ndim != 4:
        raise ValueError(
            "jacobian must have shape (n_grid, n_in, n_grid) or "
            "(n_grid, n_out, n_in, n_grid)"
        )

    n_grid, n_out, n_in, input_grid = kernel.shape
    if n_grid != input_grid:
        raise ValueError("jacobian output and input grid dimensions must match")

    e_k = fourier_mode_vector(n_grid, k_index)
    coeffs = np.empty((n_out, n_in), dtype=np.complex128)
    norm = float(n_grid)

    for out_idx in range(n_out):
        for in_idx in range(n_in):
            coeffs[out_idx, in_idx] = np.conj(e_k) @ kernel[:, out_idx, in_idx, :] @ e_k / norm

    if squeeze_output:
        return coeffs[0]
    return coeffs


def linearize_spatial_model(model, equilibrium_features) -> np.ndarray:
    r"""Linearize a periodic spatial model around one homogeneous input state.

    Parameters
    ----------
    model : callable
        Callable accepting a 4D tensor ``(batch, n_in, nx, ny)`` and returning
        a 4D tensor ``(batch, n_out, nx, ny)``.
    equilibrium_features : array-like or torch.Tensor
        Input tensor with shape ``(1, n_in, nx, 1)``. The current helper is
        intentionally restricted to a single batch sample and a single y-cell so
        that the returned Jacobian matches the 1D Fourier-projection utilities.

    Returns
    -------
    ndarray
        Jacobian with shape ``(nx, n_out, n_in, nx)`` suitable for
        :func:`project_fourier_jacobian`.
    """
    if torch is None:
        raise ImportError("linearize_spatial_model requires torch")

    features = torch.as_tensor(equilibrium_features)
    if features.ndim != 4:
        raise ValueError("equilibrium_features must have shape (1, n_in, nx, 1)")
    if features.shape[0] != 1 or features.shape[-1] != 1:
        raise ValueError("only single-sample 1D inputs with ny=1 are currently supported")

    features = features.detach().clone().requires_grad_(True)

    def model_on_slice(z_1d):
        z_full = z_1d.unsqueeze(0).unsqueeze(-1)
        output = model(z_full)
        if output.ndim != 4:
            raise ValueError("model output must have shape (1, n_out, nx, 1)")
        if output.shape[0] != 1 or output.shape[-1] != 1:
            raise ValueError("model output must preserve single-sample 1D structure")
        return output[0, :, :, 0]

    jac = torch.autograd.functional.jacobian(model_on_slice, features[0, :, :, 0])
    jac = jac.detach().cpu().numpy()
    return np.transpose(jac, (1, 0, 2, 3))


def linearize_spatial_model_2d(model, equilibrium_features) -> np.ndarray:
    r"""Linearize a 2D periodic spatial model around a homogeneous input state.

    Parameters
    ----------
    model : callable
        Callable accepting ``(1, n_in, nx, ny)`` and returning
        ``(1, n_out, nx, ny)``.
    equilibrium_features : array-like or torch.Tensor
        Input tensor with shape ``(1, n_in, nx, ny)``.

    Returns
    -------
    ndarray
        Jacobian with shape ``(nx, ny, n_out, n_in, nx, ny)`` suitable for
        :func:`project_fourier_jacobian_2d`.
    """
    if torch is None:
        raise ImportError("linearize_spatial_model_2d requires torch")

    features = torch.as_tensor(equilibrium_features)
    if features.ndim != 4 or features.shape[0] != 1:
        raise ValueError("equilibrium_features must have shape (1, n_in, nx, ny)")

    def _model_slice(z):
        out = model(z.unsqueeze(0))  # (1, n_out, nx, ny)
        if out.shape[0] != 1:
            raise ValueError("model output must preserve batch size of 1")
        return out[0]  # (n_out, nx, ny)

    jac = torch.autograd.functional.jacobian(_model_slice, features[0])
    # jac: (n_out, nx, ny, n_in, nx, ny) → transpose to (nx, ny, n_out, n_in, nx, ny)
    return np.transpose(jac.detach().cpu().numpy(), (1, 2, 0, 3, 4, 5))


def closure_tensor_jacobian_at_equilibrium(
    model,
    dataset,
    equilibrium_values,
) -> tuple[np.ndarray, list[str], list[str]]:
    r"""Linearized pressure-tensor Jacobian of an ML closure at equilibrium.

    Computes :math:`d P_{\alpha\beta} / d q_i` in physical units where
    :math:`q_i` are the physical input features (in the order stored in
    ``dataset.request_features``) and :math:`P_{\alpha\beta}` are the physical
    target components.  The derivative is evaluated at a spatially homogeneous
    equilibrium described by *equilibrium_values*.

    The chain rule through the normalization layers is applied analytically; no
    numerical finite differences are used.

    Parameters
    ----------
    model : ClosureLitModule or callable
        Trained model accepting ``(1, n_feat, 1, 1)`` and returning
        ``(1, n_target, 1, 1)``.
    dataset : DataFrameDataset
        Dataset providing normalization statistics and prescaler metadata.
    equilibrium_values : array-like, shape ``(n_feat,)``
        Physical values for each feature channel in the order of
        ``dataset.request_features``.

    Returns
    -------
    jac_phys : ndarray, shape ``(n_target, n_feat)``
        Physical-space Jacobian at the equilibrium point.
    target_names : list[str]
        Names of the target channels (row order).
    feature_names : list[str]
        Names of the feature channels (column order).
    """
    if torch is None:
        raise ImportError("closure_tensor_jacobian_at_equilibrium requires torch")

    eq = np.asarray(equilibrium_values, dtype=float).ravel()
    n_feat = len(dataset.request_features)
    if eq.shape != (n_feat,):
        raise ValueError(
            f"equilibrium_values must have shape ({n_feat},) to match "
            f"dataset.request_features, got {eq.shape}"
        )

    # --- build normalized equilibrium point ---
    eq_pre = eq.copy()
    for i, func in enumerate(dataset.prescaler_features or []):
        if func is not None:
            eq_pre[i] = func(float(eq[i]))

    if dataset.scaler_features and dataset.features_mean is not None:
        f_mean = np.asarray(dataset.features_mean, dtype=float).ravel()
        f_std = np.asarray(dataset.features_std, dtype=float).ravel()
        eq_norm = (eq_pre - f_mean) / (f_std + 1e-40)
    else:
        f_std = np.ones(n_feat, dtype=float)
        eq_norm = eq_pre.copy()

    # --- autograd Jacobian of model in normalized space ---
    model.eval()
    z0 = torch.tensor(eq_norm, dtype=torch.float32).reshape(-1)

    def _model_flat(z):
        return model(z.reshape(1, n_feat, 1, 1)).reshape(-1)

    jac_norm_np = torch.autograd.functional.jacobian(
        _model_flat, z0
    ).detach().cpu().numpy()  # (n_target, n_feat)

    # --- un-normalized prediction at equilibrium for prescaler derivative ---
    with torch.no_grad():
        y_norm_np = _model_flat(z0).cpu().numpy()

    n_target = len(dataset.request_targets)
    if dataset.scaler_targets and dataset.targets_std is not None:
        t_std = np.asarray(dataset.targets_std, dtype=float).ravel()
        t_mean = np.asarray(dataset.targets_mean, dtype=float).ravel()
        y_pre = y_norm_np * t_std + t_mean
    else:
        t_std = np.ones(n_target, dtype=float)
        y_pre = y_norm_np.copy()

    # --- derivative of output prescaler: d(y_phys)/d(y_pre) ---
    d_output = np.ones(n_target, dtype=float)
    for i, func in enumerate(dataset.prescaler_targets or []):
        if func is not None:
            name = getattr(func, "__name__", "")
            if name == "log":
                d_output[i] = np.exp(float(y_pre[i]))
            elif name == "arcsinh":
                d_output[i] = np.cosh(float(y_pre[i]))
            else:
                raise ValueError(
                    f"Unsupported target prescaler '{name}'. Supported: None, log, arcsinh."
                )
    if dataset.scaler_targets and dataset.targets_std is not None:
        d_output = d_output * t_std

    # --- derivative of input prescaler: d(z_pre)/d(z_phys) ---
    d_input = np.ones(n_feat, dtype=float)
    for i, func in enumerate(dataset.prescaler_features or []):
        if func is not None:
            name = getattr(func, "__name__", "")
            if name == "log":
                d_input[i] = 1.0 / (float(eq[i]) + 1e-40)
            elif name == "arcsinh":
                d_input[i] = 1.0 / np.sqrt(1.0 + float(eq[i]) ** 2)
            else:
                raise ValueError(
                    f"Unsupported feature prescaler '{name}'. Supported: None, log, arcsinh."
                )
    if dataset.scaler_features and dataset.features_std is not None:
        d_input = d_input / (f_std + 1e-40)

    # full chain-rule product: diag(d_output) @ J_norm @ diag(d_input)
    jac_phys = np.diag(d_output) @ jac_norm_np @ np.diag(d_input)
    return jac_phys, list(dataset.request_targets), list(dataset.request_features)


def scan_dispersion_relation(
    background: HallMHDBackground,
    k_magnitudes,
    angles,
    closure_fn=None,
    *,
    hall_scale: float = 1.0,
    ion_pressure_jacobian: np.ndarray | None = None,
    resistivity: float = 0.0,
    sort_by: str = "growth_rate",
) -> dict[str, np.ndarray]:
    r"""Scan Hall-MHD eigenvalues over a grid of 2D wavenumbers.

    For each ``(|k|, theta)`` pair the function builds the linearized operator
    with :func:`build_hall_mhd_operator` and records the seven eigenvalues.

    Parameters
    ----------
    background : HallMHDBackground
        Uniform equilibrium.
    k_magnitudes : array-like, shape ``(n_k,)``
        Physical wavenumber magnitudes :math:`|k|`.
    angles : array-like, shape ``(n_angle,)``
        Propagation angles in radians, measured from the ``B0`` direction.
    closure_fn : callable or None
        ``closure_fn(kvec, background)`` → electric-field Jacobian
        ``(3, 7)``.  If ``None``, no electron closure is applied (cold-electron
        Hall MHD).
    hall_scale : float
        Hall parameter.
    ion_pressure_jacobian : ndarray or None
        Optional ``(3, 7)`` ion-pressure force Jacobian.
    resistivity : float
        Resistivity coefficient.
    sort_by : ``"growth_rate"`` or ``"frequency"``
        Sort eigenvalues by real part (growth rate) descending or by imaginary
        part (frequency) ascending at each grid point.

    Returns
    -------
    dict
        ``"eigenvalues"`` : complex ndarray, shape ``(n_k, n_angle, 7)``
        ``"k_magnitudes"`` : float ndarray, shape ``(n_k,)``
        ``"angles"``       : float ndarray, shape ``(n_angle,)``
    """
    k_arr = np.asarray(k_magnitudes, dtype=float).ravel()
    theta_arr = np.asarray(angles, dtype=float).ravel()
    n_k, n_angle, n_modes = k_arr.size, theta_arr.size, 7

    eigenvalues = np.empty((n_k, n_angle, n_modes), dtype=np.complex128)

    b0 = np.asarray(background.B0, dtype=float)
    b0_hat = b0 / (np.linalg.norm(b0) + 1e-40)

    # unit vector perpendicular to b0 in the xy-plane
    perp = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(b0_hat, perp)) > 0.99:
        perp = np.array([1.0, 0.0, 0.0])
    b0_perp = perp - np.dot(perp, b0_hat) * b0_hat
    b0_perp /= np.linalg.norm(b0_perp) + 1e-40

    for ik, k_mag in enumerate(k_arr):
        for ia, theta in enumerate(theta_arr):
            kvec = k_mag * (np.cos(theta) * b0_hat + np.sin(theta) * b0_perp)
            closure_jac = closure_fn(kvec, background) if closure_fn is not None else None
            op = build_hall_mhd_operator(
                background,
                kvec,
                hall_scale=hall_scale,
                closure_electric_jacobian=closure_jac,
                ion_pressure_jacobian=ion_pressure_jacobian,
                resistivity=resistivity,
            )
            vals, _ = eigensystem(op)
            if sort_by == "growth_rate":
                idx = np.argsort(np.real(vals))[::-1]
            elif sort_by == "frequency":
                idx = np.argsort(np.imag(vals))
            else:
                raise ValueError("sort_by must be 'growth_rate' or 'frequency'")
            eigenvalues[ik, ia] = vals[idx]

    return {"eigenvalues": eigenvalues, "k_magnitudes": k_arr, "angles": theta_arr}


def apply_closure_correction(
    operator: np.ndarray,
    closure_rows: Sequence[int],
    closure_block: np.ndarray,
) -> np.ndarray:
    """Insert learned-closure row corrections into a linear operator.

    The convention follows the existing FNO notebook workflow: the closure term
    enters the evolution equation with a minus sign, so the supplied block is
    subtracted from the selected rows.
    """
    matrix = np.array(operator, dtype=np.complex128, copy=True)
    row_idx = np.asarray(list(closure_rows), dtype=int)
    block = np.asarray(closure_block, dtype=np.complex128)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix")
    if row_idx.ndim != 1 or row_idx.size == 0:
        raise ValueError("closure_rows must contain at least one row index")

    expected_shape = (row_idx.size, matrix.shape[1])
    if block.shape != expected_shape:
        raise ValueError(
            f"closure_block must have shape {expected_shape}, got {block.shape}"
        )

    matrix[row_idx, :] -= block
    return matrix


def build_dispersion_matrix(
    flux_jacobian: np.ndarray,
    *,
    k_phys: float,
    source_jacobian: np.ndarray | None = None,
    closure_rows: Sequence[int] | None = None,
    closure_block: np.ndarray | None = None,
) -> np.ndarray:
    r"""Assemble the reduced-fluid linear operator ``M = -ik A + S + C``."""
    flux = np.asarray(flux_jacobian, dtype=np.complex128)
    if flux.ndim != 2 or flux.shape[0] != flux.shape[1]:
        raise ValueError("flux_jacobian must be a square matrix")

    operator = (-1j * float(k_phys)) * flux

    if source_jacobian is not None:
        source = np.asarray(source_jacobian, dtype=np.complex128)
        if source.shape != flux.shape:
            raise ValueError("source_jacobian must have the same shape as flux_jacobian")
        operator = operator + source

    if closure_rows is not None or closure_block is not None:
        if closure_rows is None or closure_block is None:
            raise ValueError("closure_rows and closure_block must be provided together")
        operator = apply_closure_correction(operator, closure_rows, closure_block)

    return operator


def eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and column-stacked eigenvectors."""
    operator = np.asarray(matrix, dtype=np.complex128)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("matrix must be a square matrix")
    return np.linalg.eig(operator)


def match_eigenbranches(
    reference_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Match eigenbranches by maximizing column-wise eigenvector overlap.

    Parameters
    ----------
    reference_vectors, candidate_vectors : ndarray, shape ``(n_state, n_mode)``
        Eigenvectors stored column-wise.

    Returns
    -------
    permutation : ndarray, shape ``(n_mode,)``
        Column permutation to apply to the candidate arrays.
    overlaps : ndarray, shape ``(n_mode,)``
        Absolute overlaps after applying the optimal permutation.
    """
    ref = np.asarray(reference_vectors, dtype=np.complex128)
    cand = np.asarray(candidate_vectors, dtype=np.complex128)

    if ref.ndim != 2 or cand.ndim != 2 or ref.shape != cand.shape:
        raise ValueError("reference_vectors and candidate_vectors must be 2D arrays with matching shape")

    ref_norm = np.linalg.norm(ref, axis=0)
    cand_norm = np.linalg.norm(cand, axis=0)
    if np.any(ref_norm == 0.0) or np.any(cand_norm == 0.0):
        raise ValueError("eigenvectors must be non-zero")

    ref_unit = ref / ref_norm[None, :]
    cand_unit = cand / cand_norm[None, :]
    overlap = np.abs(ref_unit.conj().T @ cand_unit)
    row_ind, col_ind = linear_sum_assignment(1.0 - overlap)
    order = col_ind[np.argsort(row_ind)]
    aligned = overlap[np.arange(overlap.shape[0]), order]
    return order, aligned


__all__ = [
    "HallMHDBackground",
    "apply_closure_correction",
    "build_dispersion_matrix",
    "build_hall_mhd_operator",
    "closure_tensor_jacobian_at_equilibrium",
    "electron_pressure_tensor_to_electric_jacobian",
    "eigensystem",
    "fourier_mode_vector",
    "isotropic_electron_closure_electric_jacobian",
    "linearize_spatial_model",
    "linearize_spatial_model_2d",
    "match_eigenbranches",
    "project_fourier_jacobian",
    "project_fourier_jacobian_2d",
    "scan_dispersion_relation",
]