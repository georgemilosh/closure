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
from scipy.linalg import expm
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
    J0 : array-like, shape ``(2,)`` or ``(3,)``
        Local background total current.  The default preserves the uniform,
        zero-current equilibrium used by the original helpers.  A nonzero
        value is required for a frozen-coefficient linearization in a current
        sheet because both ``u_e = u_i - J/rho`` and ``J x B / rho`` then have
        first-order density and magnetic-field terms.
    """

    rho0: float
    u0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B0: tuple[float, float, float] = (1.0, 0.0, 0.0) # default B0 along x for Harris sheet, but this is arbitrary
    J0: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.rho0 <= 0.0:
            raise ValueError("rho0 must be positive")
        object.__setattr__(self, "u0", tuple(_as_3vector(self.u0, name="u0")))
        object.__setattr__(self, "B0", tuple(_as_3vector(self.B0, name="B0")))
        object.__setattr__(self, "J0", tuple(_as_3vector(self.J0, name="J0")))


# Channel orders at the MENURA/TorchScript boundary.  Keeping these explicit is
# important: MENURA stores the tensor diagonals first, while the reduced-fluid
# pressure-divergence helper above uses the conventional packed-symmetric order.
MENURA_FEATURE_NAMES = (
    "rho_e",
    "Bx",
    "By",
    "Bz",
    "Vx_e",
    "Vy_e",
    "Vz_e",
    "Ex",
    "Ey",
    "Ez",
    "Wxx_e",
    "Wyy_e",
    "Wzz_e",
    "Wxy_e",
    "Wxz_e",
    "Wyz_e",
)
MENURA_PRESSURE_COMPONENTS = ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")
DISPERSION_PRESSURE_COMPONENTS = ("Pxx", "Pxy", "Pxz", "Pyy", "Pyz", "Pzz")


def menura_fourth_order_derivative_wavenumber(kvec, cell_size: float) -> np.ndarray:
    r"""Return the real modified wavenumber of MENURA's fourth-order derivative.

    MENURA applies

    ``D f = [8(f[i+1]-f[i-1])-(f[i+2]-f[i-2])] / (12 dx)``.

    For a Fourier mode this is ``i * k_eff`` with
    ``k_eff = [8 sin(k dx)-sin(2 k dx)]/(6 dx)``.  A two-component input is
    interpreted as a 2-D MENURA mode and receives an exactly zero z component.
    """
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")
    k3 = _as_3vector(kvec, name="kvec")
    phase = k3 * float(cell_size)
    return (8.0 * np.sin(phase) - np.sin(2.0 * phase)) / (6.0 * float(cell_size))


def menura_fourth_order_laplacian_symbol(kvec, cell_size: float) -> float:
    r"""Return MENURA's fourth-order Laplacian Fourier symbol (non-positive)."""
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")
    k3 = _as_3vector(kvec, name="kvec")
    phase = k3 * float(cell_size)
    one_dim = (-2.0 * np.cos(2.0 * phase) + 32.0 * np.cos(phase) - 30.0)
    return float(np.sum(one_dim) / (12.0 * float(cell_size) ** 2))


def menura_binomial_filter_transfer(kvec, cell_size: float, *, passes: int = 1) -> float:
    r"""Return the transfer of MENURA's 2-D 3x3 binomial smoother.

    One pass is the separable ``[1, 2, 1]/4`` stencil in x and y, hence
    ``cos(kx*dx/2)^2 cos(ky*dx/2)^2``.  The 2-D production kernel never filters
    along z.  ``passes=0`` is exactly the identity.
    """
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")
    if int(passes) != passes or passes < 0:
        raise ValueError("passes must be a non-negative integer")
    k3 = _as_3vector(kvec, name="kvec")
    one_pass = np.cos(0.5 * k3[0] * cell_size) ** 2 * np.cos(
        0.5 * k3[1] * cell_size
    ) ** 2
    return float(one_pass ** int(passes))


def _cross_product_matrix(vector) -> np.ndarray:
    """Matrix ``C`` such that ``C @ x == vector x x``."""
    x, y, z = _as_3vector(vector, name="vector")
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.complex128
    )


def menura_electron_velocity_jacobian(
    kvec,
    background: HallMHDBackground,
    *,
    cell_size: float,
) -> np.ndarray:
    r"""Return ``d u_e / d q`` for MENURA features and reduced primitives.

    The primitive order is ``(rho, ui_x, ui_y, ui_z, Bx, By, Bz)``.  Around a
    locally frozen state, MENURA's legacy moment convention gives
    ``u_e = u_i - J/rho`` and ``delta J = i k_eff x delta B``.  At nonzero
    local current the density column is therefore ``J0/rho0**2``; dropping it
    silently changes the sheet-state loop while leaving lobe states almost
    unchanged.
    """
    k_eff = menura_fourth_order_derivative_wavenumber(kvec, cell_size)
    jac = np.zeros((3, 7), dtype=np.complex128)
    jac[:, 0] = np.asarray(background.J0, dtype=float) / float(background.rho0) ** 2
    jac[:, 1:4] = np.eye(3)
    jac[:, 4:7] = -(1j / float(background.rho0)) * _cross_product_matrix(k_eff)
    return jac


def menura_strain_feature_jacobian(
    kvec,
    background: HallMHDBackground,
    *,
    cell_size: float,
    filter_passes: int = 4,
) -> np.ndarray:
    r"""Return ``d(Wxx,Wyy,Wzz,Wxy,Wxz,Wyz)/d q`` for a MENURA mode."""
    k_eff = menura_fourth_order_derivative_wavenumber(kvec, cell_size)
    ue_jac = menura_electron_velocity_jacobian(kvec, background, cell_size=cell_size)
    kx, ky, _ = k_eff
    jac = np.zeros((6, 7), dtype=np.complex128)
    jac[0] = 1j * kx * ue_jac[0]
    jac[1] = 1j * ky * ue_jac[1]
    # Wzz is identically zero in MENURA's two-dimensional feature kernel.
    jac[3] = 0.5j * (kx * ue_jac[1] + ky * ue_jac[0])
    jac[4] = 0.5j * kx * ue_jac[2]
    jac[5] = 0.5j * ky * ue_jac[2]
    return jac * menura_binomial_filter_transfer(
        kvec, cell_size, passes=filter_passes
    )


def menura_feature_jacobian(
    kvec,
    background: HallMHDBackground,
    *,
    cell_size: float,
    strain_filter_passes: int = 4,
    electric_feature_jacobian: np.ndarray | None = None,
) -> np.ndarray:
    r"""Map reduced primitive perturbations to the 16 MENURA model features.

    The density feature includes MENURA's ECsim conversion ``rho_e=-rho/(4*pi)``.
    Electric-field inputs are algebraic rather than evolved variables in the
    reduced system; they default to zero and can be supplied explicitly for a
    model that actually consumes them.
    """
    jac = np.zeros((len(MENURA_FEATURE_NAMES), 7), dtype=np.complex128)
    jac[0, 0] = -1.0 / (4.0 * np.pi)
    jac[1:4, 4:7] = np.eye(3)
    jac[4:7] = menura_electron_velocity_jacobian(
        kvec, background, cell_size=cell_size
    )
    if electric_feature_jacobian is not None:
        electric = np.asarray(electric_feature_jacobian, dtype=np.complex128)
        if electric.shape != (3, 7):
            raise ValueError("electric_feature_jacobian must have shape (3, 7)")
        jac[7:10] = electric
    jac[10:16] = menura_strain_feature_jacobian(
        kvec,
        background,
        cell_size=cell_size,
        filter_passes=strain_filter_passes,
    )
    return jac


def menura_pressure_primitive_jacobian(
    pressure_feature_jacobian: np.ndarray,
    feature_jacobian: np.ndarray,
) -> np.ndarray:
    r"""Compose a physical MENURA pressure/feature Jacobian with ``d feature/d q``.

    The returned rows are reordered for
    :func:`electron_pressure_tensor_to_electric_jacobian`.
    """
    pressure_feature = np.asarray(pressure_feature_jacobian, dtype=np.complex128)
    feature = np.asarray(feature_jacobian, dtype=np.complex128)
    if pressure_feature.shape != (6, len(MENURA_FEATURE_NAMES)):
        raise ValueError(
            f"pressure_feature_jacobian must have shape (6, {len(MENURA_FEATURE_NAMES)})"
        )
    if feature.shape != (len(MENURA_FEATURE_NAMES), 7):
        raise ValueError(
            f"feature_jacobian must have shape ({len(MENURA_FEATURE_NAMES)}, 7)"
        )
    # MENURA: Pxx,Pyy,Pzz,Pxy,Pxz,Pyz -> packed symmetric: Pxx,Pxy,Pxz,Pyy,Pyz,Pzz
    return (pressure_feature @ feature)[[0, 3, 4, 1, 5, 2]]


def build_menura_closure_operator(
    background: HallMHDBackground,
    kvec,
    pressure_feature_jacobian: np.ndarray,
    *,
    cell_size: float,
    strain_filter_passes: int = 4,
    eamb_filter_passes: int = 0,
    hall_scale: float = 1.0,
    resistivity: float = 0.0,
    hyper_resistivity: float = 0.0,
) -> np.ndarray:
    r"""Compose a learned closure with MENURA's exact discrete spatial symbols.

    Hyper-resistivity enters Ohm's law as ``-eta_hyp Laplacian(J)``.  On a
    Fourier mode this is an additional positive mode-dependent resistivity.
    Output-side smoothing, when requested, multiplies only the pressure-derived
    ``E_amb`` term and is therefore suitable for the explicitly diagnostic arm.
    """
    k_eff = menura_fourth_order_derivative_wavenumber(kvec, cell_size)
    feature_jac = menura_feature_jacobian(
        kvec,
        background,
        cell_size=cell_size,
        strain_filter_passes=strain_filter_passes,
    )
    tensor_jac = menura_pressure_primitive_jacobian(
        pressure_feature_jacobian, feature_jac
    )
    closure_electric = electron_pressure_tensor_to_electric_jacobian(
        k_eff, background, tensor_jac
    )
    closure_electric *= menura_binomial_filter_transfer(
        kvec, cell_size, passes=eamb_filter_passes
    )
    eta_mode = float(resistivity) - float(hyper_resistivity) * menura_fourth_order_laplacian_symbol(
        kvec, cell_size
    )
    return build_hall_mhd_operator(
        background,
        k_eff,
        hall_scale=hall_scale,
        closure_electric_jacobian=closure_electric,
        resistivity=eta_mode,
    )


def operator_amplification(operator: np.ndarray, timestep: float) -> dict[str, float]:
    r"""Return modal and worst-vector amplification of ``exp(timestep*operator)``."""
    matrix = np.asarray(operator, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix")
    if timestep < 0.0:
        raise ValueError("timestep must be non-negative")
    propagator = expm(float(timestep) * matrix)
    eigenvalues = np.linalg.eigvals(matrix)
    spectral = float(np.exp(float(timestep) * np.max(np.real(eigenvalues))))
    transient = float(np.linalg.svd(propagator, compute_uv=False)[0])
    return {
        "spectral_radius": spectral,
        "largest_singular_value": transient,
        "max_growth_rate": float(np.max(np.real(eigenvalues))),
    }


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
        Fourier wavevector. A length-2 vector is interpreted as ``(kx, ky, 0)``.
    background : HallMHDBackground
        Uniform Hall-MHD equilibrium.
    tensor_jacobian : ndarray, shape ``(6, n_state)``
        Jacobian of the symmetric electron pressure tensor components with
        respect to the reduced primitive state. The component order is
        ``(Pxx, Pxy, Pxz, Pyy, Pyz, Pzz)``.

        In the Hall-MHD notebook ``n_state == 7`` and the columns are
        ``(rho, ux, uy, uz, Bx, By, Bz)``. Electric-field components are not
        columns because this reduced model treats ``E`` as an algebraic Ohm-law
        quantity rather than an evolved primitive variable.

    Returns
    -------
    ndarray
        Electric-field Jacobian with shape ``(3, n_state)`` for the closure
        term ``-(\nabla \cdot P_e) / rho0``.

    Notes
    -----
    For a 3D wavevector ``(kx, ky, kz)`` the row mapping is
    ``Ex <- kx * dPxx + ky * dPxy + kz * dPxz``,
    ``Ey <- kx * dPxy + ky * dPyy + kz * dPyz``, and
    ``Ez <- kx * dPxz + ky * dPyz + kz * dPzz``, multiplied by ``-i / rho0``.
    Simulation-plane scans recover the previous 2D expression by setting
    ``kz = 0``.
    """
    tensor = np.asarray(tensor_jacobian, dtype=np.complex128)
    if tensor.ndim != 2 or tensor.shape[0] != 6:
        raise ValueError("tensor_jacobian must have shape (6, n_state)")

    k3 = _as_3vector(kvec, name="kvec")
    kx, ky, kz = k3
    scale = -1j / float(background.rho0)
    e_jac = np.zeros((3, tensor.shape[1]), dtype=np.complex128)
    e_jac[0] = scale * (kx * tensor[0] + ky * tensor[1] + kz * tensor[2])
    e_jac[1] = scale * (kx * tensor[1] + ky * tensor[3] + kz * tensor[4])
    e_jac[2] = scale * (kx * tensor[2] + ky * tensor[4] + kz * tensor[5])
    return e_jac


def hall_mhd_k_vector(
    background: HallMHDBackground,
    k_magnitude: float,
    theta: float,
    *,
    geometry: str = "field_aligned",
) -> np.ndarray:
    r"""Construct a Hall-MHD wavevector for a requested scan geometry.

    Parameters
    ----------
    background : HallMHDBackground
        Uniform Hall-MHD equilibrium.
    k_magnitude : float
        Desired wavevector magnitude.
    theta : float
        Propagation angle in radians.
    geometry : {``"field_aligned"``, ``"simulation_plane"``}
        ``"field_aligned"`` measures ``theta`` from the full 3D background
        magnetic field. ``"simulation_plane"`` constrains ``kz = 0`` and
        measures ``theta`` from the in-plane projection of ``B0``.

    Returns
    -------
    ndarray, shape ``(3,)``
        Wavevector in the global ``(x, y, z)`` basis.
    """
    k_mag = float(k_magnitude)
    b0 = np.asarray(background.B0, dtype=float)
    mode = geometry.lower().replace("-", "_")

    if mode in {"field_aligned", "full_b0", "full"}:
        b0_norm = np.linalg.norm(b0)
        if b0_norm <= 0.0:
            raise ValueError("background.B0 must be non-zero for field_aligned geometry")
        b0_hat = b0 / b0_norm
        seed = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(b0_hat, seed)) > 0.99:
            seed = np.array([1.0, 0.0, 0.0])
        b0_perp = seed - np.dot(seed, b0_hat) * b0_hat
        b0_perp /= np.linalg.norm(b0_perp) + 1e-40
        return k_mag * (np.cos(theta) * b0_hat + np.sin(theta) * b0_perp)

    if mode in {"simulation_plane", "xy", "in_plane"}:
        b0_xy = np.array([b0[0], b0[1], 0.0], dtype=float)
        b0_xy_norm = np.linalg.norm(b0_xy)
        if b0_xy_norm <= 0.0:
            raise ValueError(
                "background.B0 must have a non-zero x-y projection for simulation_plane geometry"
            )
        b0_hat = b0_xy / b0_xy_norm
        b0_perp = np.array([-b0_hat[1], b0_hat[0], 0.0], dtype=float)
        return k_mag * (np.cos(theta) * b0_hat + np.sin(theta) * b0_perp)

    raise ValueError("geometry must be 'field_aligned' or 'simulation_plane'")


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
    about a locally frozen equilibrium.  ``J0=0`` recovers the historical
    uniform-background operator.  Nonzero ``J0`` includes every algebraic
    first-order term from ``J x B / rho`` in Ohm's law and ion momentum.  As a
    WKB/frozen-coefficient analysis it does not include gradients of the
    background pressure or flow; those are lower-order in the high-k gate.

    The model uses cold ions by default. Any prescribed electron closure enters
    through the generalized Ohm law as an electric-field correction matrix
    ``closure_electric_jacobian``. An optional ion-pressure-force Jacobian can
    also be supplied as a ``(3, 7)`` matrix acting on the same primitive state.
    """
    k3 = _as_3vector(kvec, name="kvec")
    b0 = np.asarray(background.B0, dtype=float)
    u0 = np.asarray(background.u0, dtype=float)
    j0 = np.asarray(background.J0, dtype=float)
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
        hall_force = (
            np.cross(delta_j, b0)
            + np.cross(j0, b)
            - (rho / rho0) * np.cross(j0, b0)
        )
        delta_e = delta_e + (float(hall_scale) / rho0) * hall_force
        delta_e = delta_e + closure_jac @ q
        if resistivity:
            delta_e = delta_e + float(resistivity) * delta_j

        drho = -1j * (rho0 * np.dot(k3, u) + rho * np.dot(k3, u0))
        lorentz = (
            np.cross(delta_j, b0)
            + np.cross(j0, b)
            - (rho / rho0) * np.cross(j0, b0)
        ) / rho0
        du = -1j * np.dot(k3, u0) * u + lorentz - ion_force @ q
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


def patch_domain_lengths_from_grid(
    patch_shape: tuple[int, int],
    simulation_domain_lengths: tuple[float, float],
    simulation_grid_shape: tuple[int, int],
) -> tuple[float, float]:
    """Return the physical size of a 2D patch from simulation grid metadata."""
    if len(patch_shape) != 2 or len(simulation_domain_lengths) != 2 or len(simulation_grid_shape) != 2:
        raise ValueError("patch_shape, simulation_domain_lengths, and simulation_grid_shape must be length-2")

    patch_nx, patch_ny = (int(patch_shape[0]), int(patch_shape[1]))
    domain_lx, domain_ly = (float(simulation_domain_lengths[0]), float(simulation_domain_lengths[1]))
    grid_nx, grid_ny = (int(simulation_grid_shape[0]), int(simulation_grid_shape[1]))
    if patch_nx <= 0 or patch_ny <= 0:
        raise ValueError("patch_shape entries must be positive")
    if domain_lx <= 0.0 or domain_ly <= 0.0:
        raise ValueError("simulation_domain_lengths entries must be positive")
    if grid_nx <= 0 or grid_ny <= 0:
        raise ValueError("simulation_grid_shape entries must be positive")

    return (patch_nx * domain_lx / grid_nx, patch_ny * domain_ly / grid_ny)


def physical_wavenumber_from_mode_indices(
    mode_indices: tuple[int, int],
    domain_lengths: tuple[float, float],
) -> np.ndarray:
    r"""Map integer patch Fourier indices to a physical simulation-plane wavevector.

    The Fourier basis vector uses ``exp(2*pi*i*m*j/N)``. If the physical patch
    length is ``L_patch``, the corresponding wavenumber is ``k = 2*pi*m/L_patch``.
    """
    if len(mode_indices) != 2 or len(domain_lengths) != 2:
        raise ValueError("mode_indices and domain_lengths must be length-2")

    mx, my = (int(mode_indices[0]), int(mode_indices[1]))
    lx, ly = (float(domain_lengths[0]), float(domain_lengths[1]))
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("domain_lengths entries must be positive")

    return np.array([2.0 * np.pi * mx / lx, 2.0 * np.pi * my / ly, 0.0], dtype=float)


def mode_indices_from_physical_wavenumber(
    kvec: tuple[float, float] | tuple[float, float, float] | np.ndarray,
    domain_lengths: tuple[float, float],
    *,
    patch_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    r"""Map a physical simulation-plane wavevector to nearest patch Fourier indices.

    This is the inverse of :func:`physical_wavenumber_from_mode_indices`, up to
    nearest-integer rounding. Supplying ``patch_shape`` enables a Nyquist check.
    """
    k3 = _as_3vector(kvec, name="kvec")
    if len(domain_lengths) != 2:
        raise ValueError("domain_lengths must be length-2")
    lx, ly = (float(domain_lengths[0]), float(domain_lengths[1]))
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("domain_lengths entries must be positive")

    mx = int(np.rint(k3[0] * lx / (2.0 * np.pi)))
    my = int(np.rint(k3[1] * ly / (2.0 * np.pi)))

    if patch_shape is not None:
        if len(patch_shape) != 2:
            raise ValueError("patch_shape must be a length-2 tuple")
        nx, ny = (int(patch_shape[0]), int(patch_shape[1]))
        if nx <= 0 or ny <= 0:
            raise ValueError("patch_shape entries must be positive")
        if abs(mx) > nx // 2 or abs(my) > ny // 2:
            raise ValueError(
                f"mode {(mx, my)} exceeds the patch Nyquist range for patch_shape={patch_shape}"
            )

    return (mx, my)


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


def _prescaler_name(func) -> str:
    return "" if func is None else getattr(func, "__name__", "")


def _feature_prescaler_derivative(func, values) -> np.ndarray:
    name = _prescaler_name(func)
    arr = np.asarray(values, dtype=float)
    if not name:
        return np.ones_like(arr, dtype=float)
    if name == "log":
        return 1.0 / (arr + 1e-40)
    if name == "arcsinh":
        return 1.0 / np.sqrt(1.0 + arr**2)
    raise ValueError(f"Unsupported feature prescaler '{name}'. Supported: None, log, arcsinh.")


def _target_inverse_prescaler_derivative(func, values) -> np.ndarray:
    name = _prescaler_name(func)
    arr = np.asarray(values, dtype=float)
    if not name:
        return np.ones_like(arr, dtype=float)
    if name == "log":
        return np.exp(arr)
    if name == "arcsinh":
        return np.cosh(arr)
    raise ValueError(f"Unsupported target prescaler '{name}'. Supported: None, log, arcsinh.")


def _normalized_equilibrium_and_input_derivative(dataset, equilibrium_values):
    eq = np.asarray(equilibrium_values, dtype=float).ravel()
    n_feat = len(dataset.request_features)
    if eq.shape != (n_feat,):
        raise ValueError(
            f"equilibrium_values must have shape ({n_feat},) to match "
            f"dataset.request_features, got {eq.shape}"
        )

    eq_pre = eq.copy()
    d_input = np.ones(n_feat, dtype=float)
    for i, func in enumerate(dataset.prescaler_features or []):
        if func is not None:
            eq_pre[i] = func(float(eq[i]))
            d_input[i] = _feature_prescaler_derivative(func, eq[i])

    if dataset.scaler_features and dataset.features_mean is not None:
        f_mean = np.asarray(dataset.features_mean, dtype=float).ravel()
        f_std = np.asarray(dataset.features_std, dtype=float).ravel()
        eq_norm = (eq_pre - f_mean) / (f_std + 1e-40)
        d_input = d_input / (f_std + 1e-40)
    else:
        eq_norm = eq_pre.copy()

    return eq, eq_norm, d_input


def _target_output_derivative_field(dataset, y_norm_np: np.ndarray) -> np.ndarray:
    n_target = len(dataset.request_targets)
    if y_norm_np.shape[0] != n_target:
        raise ValueError(
            f"model returned {y_norm_np.shape[0]} targets, expected {n_target}"
        )

    if dataset.scaler_targets and dataset.targets_std is not None:
        t_std = np.asarray(dataset.targets_std, dtype=float).reshape(n_target, 1, 1)
        t_mean = np.asarray(dataset.targets_mean, dtype=float).reshape(n_target, 1, 1)
        y_pre = y_norm_np * t_std + t_mean
    else:
        t_std = np.ones((n_target, 1, 1), dtype=float)
        y_pre = y_norm_np.copy()

    d_output = np.ones_like(y_pre, dtype=float)
    for i, func in enumerate(dataset.prescaler_targets or []):
        if func is not None:
            d_output[i] = _target_inverse_prescaler_derivative(func, y_pre[i])
    return d_output * t_std


def closure_tensor_fourier_symbol_at_equilibrium(
    model,
    dataset,
    equilibrium_values,
    mode_indices,
    *,
    patch_shape: tuple[int, int] = (32, 32),
    method: str = "jvp",
    finite_difference_eps: float = 1e-4,
) -> tuple[np.ndarray, list[str], list[str]]:
    r"""Return the pressure-tensor Fourier symbol of a spatial closure.

    The closure is evaluated around a homogeneous physical equilibrium. For each
    input feature channel, this applies a unit-amplitude Fourier perturbation on
    the periodic ``patch_shape`` grid, differentiates the model response, and
    projects the output back onto the same mode. The result is the modal transfer
    matrix :math:`\widehat{\delta P}(k) / \widehat{\delta f}(k)` in physical
    units.

    This helper is intended for convolutional or otherwise spatial closures. It
    avoids materializing the full spatial Jacobian, whose memory scales like
    ``nx**2 * ny**2 * n_in * n_out``.

    Parameters
    ----------
    model : callable
        Trained spatial model accepting ``(1, n_feat, nx, ny)`` and returning
        ``(1, n_target, nx, ny)``.
    dataset : DataFrameDataset
        Dataset providing feature/target normalization and prescaler metadata.
    equilibrium_values : array-like, shape ``(n_feat,)``
        Physical feature values in ``dataset.request_features`` order.
    mode_indices : tuple[int, int]
        Integer Fourier mode indices ``(kx_index, ky_index)`` on the patch grid.
    patch_shape : tuple[int, int]
        Spatial patch size used for the modal perturbation.
    method : {``"jvp"``, ``"finite_difference"``}
        Directional-derivative method. ``"jvp"`` falls back to finite
        differences if the model backend does not support JVP.
    finite_difference_eps : float
        Centered finite-difference amplitude in normalized-feature units.

    Returns
    -------
    jac_phys : ndarray, shape ``(n_target, n_feat)``
        Complex Fourier-symbol transfer matrix in physical units.
    target_names, feature_names : list[str]
        Channel names for rows and columns.
    """
    if torch is None:
        raise ImportError("closure_tensor_fourier_symbol_at_equilibrium requires torch")

    if len(patch_shape) != 2:
        raise ValueError("patch_shape must be a length-2 tuple")
    nx, ny = (int(patch_shape[0]), int(patch_shape[1]))
    if nx <= 0 or ny <= 0:
        raise ValueError("patch_shape entries must be positive")
    kx_index, ky_index = (int(mode_indices[0]), int(mode_indices[1]))

    _, eq_norm, d_input = _normalized_equilibrium_and_input_derivative(dataset, equilibrium_values)
    n_feat = len(dataset.request_features)
    n_target = len(dataset.request_targets)
    if hasattr(model, "eval"):
        model.eval()

    z0 = torch.tensor(eq_norm, dtype=torch.float32).reshape(1, n_feat, 1, 1).expand(1, n_feat, nx, ny).clone()

    with torch.no_grad():
        y0_norm = model(z0)
    if y0_norm.ndim != 4 or y0_norm.shape[0] != 1:
        raise ValueError("model output must have shape (1, n_target, nx, ny)")
    if y0_norm.shape[1] != n_target or y0_norm.shape[-2:] != (nx, ny):
        raise ValueError(
            f"model output must have shape (1, {n_target}, {nx}, {ny}), got {tuple(y0_norm.shape)}"
        )

    output_derivative = _target_output_derivative_field(dataset, y0_norm[0].detach().cpu().numpy())
    e_kx = fourier_mode_vector(nx, kx_index)
    e_ky = fourier_mode_vector(ny, ky_index)
    mode = np.outer(e_kx, e_ky)
    mode_real = torch.tensor(mode.real, dtype=z0.dtype).reshape(1, 1, nx, ny)
    mode_imag = torch.tensor(mode.imag, dtype=z0.dtype).reshape(1, 1, nx, ny)
    norm = float(nx * ny)

    requested_method = method.lower().replace("-", "_")
    if requested_method not in {"jvp", "finite_difference", "finite_difference_only"}:
        raise ValueError("method must be 'jvp' or 'finite_difference'")

    def model_fn(z):
        return model(z)

    def directional_output(tangent):
        if requested_method == "jvp":
            try:
                _, tangent_out = torch.autograd.functional.jvp(
                    model_fn,
                    (z0,),
                    (tangent,),
                    create_graph=False,
                    strict=False,
                )
                return tangent_out.detach().cpu().numpy()[0]
            except Exception:
                pass
        eps = float(finite_difference_eps)
        if eps <= 0.0:
            raise ValueError("finite_difference_eps must be positive")
        with torch.no_grad():
            plus = model(z0 + eps * tangent)
            minus = model(z0 - eps * tangent)
        return ((plus - minus) / (2.0 * eps)).detach().cpu().numpy()[0]

    coeffs = np.empty((n_target, n_feat), dtype=np.complex128)
    for feature_idx in range(n_feat):
        tangent_real = torch.zeros_like(z0)
        tangent_imag = torch.zeros_like(z0)
        tangent_real[:, feature_idx : feature_idx + 1] = float(d_input[feature_idx]) * mode_real
        tangent_imag[:, feature_idx : feature_idx + 1] = float(d_input[feature_idx]) * mode_imag

        response_real = directional_output(tangent_real) * output_derivative
        response_imag = directional_output(tangent_imag) * output_derivative
        projected_real = np.einsum("xy,oxy->o", np.conj(mode), response_real, optimize=True) / norm
        projected_imag = np.einsum("xy,oxy->o", np.conj(mode), response_imag, optimize=True) / norm
        coeffs[:, feature_idx] = projected_real + 1j * projected_imag

    return coeffs, list(dataset.request_targets), list(dataset.request_features)


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

    Examples
    --------
    For a pressure model trained with six target channels and ten input
    features, this returns ``jac_phys.shape == (6, 10)``. The row order is not
    rearranged here; callers should use ``target_names`` and ``feature_names``
    to map the Jacobian into any model-specific basis, such as the Hall-MHD
    primitive state ``(rho, ux, uy, uz, Bx, By, Bz)``.
    """
    if torch is None:
        raise ImportError("closure_tensor_jacobian_at_equilibrium requires torch")

    _, eq_norm, d_input = _normalized_equilibrium_and_input_derivative(dataset, equilibrium_values)
    n_feat = len(dataset.request_features)

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
            d_output[i] = _target_inverse_prescaler_derivative(func, y_pre[i])
    if dataset.scaler_targets and dataset.targets_std is not None:
        d_output = d_output * t_std

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
    geometry: str = "field_aligned",
) -> dict[str, np.ndarray]:
    r"""Scan Hall-MHD eigenvalues over a grid of wavevectors.

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
    geometry : {``"field_aligned"``, ``"simulation_plane"``}
        Wavevector geometry passed to :func:`hall_mhd_k_vector`.

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

    for ik, k_mag in enumerate(k_arr):
        for ia, theta in enumerate(theta_arr):
            kvec = hall_mhd_k_vector(background, k_mag, theta, geometry=geometry)
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
    "DISPERSION_PRESSURE_COMPONENTS",
    "HallMHDBackground",
    "MENURA_FEATURE_NAMES",
    "MENURA_PRESSURE_COMPONENTS",
    "apply_closure_correction",
    "build_dispersion_matrix",
    "build_hall_mhd_operator",
    "build_menura_closure_operator",
    "closure_tensor_fourier_symbol_at_equilibrium",
    "closure_tensor_jacobian_at_equilibrium",
    "electron_pressure_tensor_to_electric_jacobian",
    "eigensystem",
    "fourier_mode_vector",
    "hall_mhd_k_vector",
    "isotropic_electron_closure_electric_jacobian",
    "linearize_spatial_model",
    "linearize_spatial_model_2d",
    "match_eigenbranches",
    "menura_binomial_filter_transfer",
    "menura_electron_velocity_jacobian",
    "menura_feature_jacobian",
    "menura_fourth_order_derivative_wavenumber",
    "menura_fourth_order_laplacian_symbol",
    "menura_pressure_primitive_jacobian",
    "menura_strain_feature_jacobian",
    "mode_indices_from_physical_wavenumber",
    "patch_domain_lengths_from_grid",
    "physical_wavenumber_from_mode_indices",
    "project_fourier_jacobian",
    "project_fourier_jacobian_2d",
    "scan_dispersion_relation",
    "operator_amplification",
]
