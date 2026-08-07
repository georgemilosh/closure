"""Tier-2 flow-gradient invariants for field-aligned pressure closures.

Motivation
----------
The electron pressure tensor obeys

    d_t P + div(u_e P) + P.grad(u_e) + (P.grad(u_e))^T + div(Q)
        + Omega_ce [b x P - P x b] = 0 ,

so the local drivers of anisotropy and agyrotropy are the *flow gradients*
(compression and strain) together with the field geometry.  This module turns
the rate-of-strain tensor of the electron bulk flow into four rotational
scalars that a field-aligned closure can consume pointwise:

``Wpar_e``   b.W.b                     -- CGL parallel-pressure driver
``divV_e``   tr(grad u_e) = div u_e    -- compressional driver
``Wmix_e``   |(I - bb).W.b|            -- gyroviscous (agyrotropic) driver
``Wperp_e``  ||traceless perp block||  -- perpendicular shear driver

with ``W = 0.5 (grad u + grad u^T)`` the symmetric rate-of-strain tensor.

Why these and not the electron-frame electric field
---------------------------------------------------
``E + u_e x B`` is *identically* ``-(div P_e)/n + eta J - eta_h lap J`` under
Menura's Ohm's law, so feeding it to the closure closes a
``P_e -> E -> div P_e`` loop and, worse, injects the hyper-resistive term,
which has no counterpart in the kinetic training data (measured: 20% of that
invariant's rms over the reconnection region, 45% of E_amb at the X-point).
The invariants here are built from ``B`` and ``u_e = (J_tot - J_i)/rho`` only.
``J_tot`` comes from Ampere's law (curl B), so nothing here depends on P_e:
the feedback loop is cut by construction.

Train/deploy parity
-------------------
Spatial derivatives use the *same* fourth-order central stencil Menura uses,

    d_x f = (8 (f_{i+1} - f_{i-1}) - (f_{i+2} - f_{i-2})) / (12 dx) ,

which is exactly :func:`closure.plasma.highdiff` with its default
coefficients.  Menura's closure call is pointwise per cell, so the derivative
*must* be evaluated by the caller on both sides; defining it once here keeps
the two implementations comparable.  ``tests/test_field_invariants.py``
asserts stencil parity, rotational invariance and analytic values.

Conventions
-----------
All inputs are expected in Alfven units on a uniform 2-D grid with periodic
wrap, indexed ``[x, y]``.  The simulation is invariant along ``z``
(``d_z = 0``), so the third row of ``grad u`` vanishes while ``u`` keeps all
three components.  Outputs then carry units of ``Omega_ci`` (``v_A / d_i``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "INVARIANT_NAMES",
    "strain_tensor",
    "flow_gradient_invariants",
]

#: Channel names, in the order :func:`flow_gradient_invariants` returns them.
INVARIANT_NAMES = ("Wpar_e", "divV_e", "Wmix_e", "Wperp_e")

#: Packed Cartesian components of the rate-of-strain tensor, in the same order
#: the pressure tensor uses: [xx, yy, zz, xy, xz, yz].  Unlike
#: :data:`INVARIANT_NAMES` these are tensor *components*, not invariants: they
#: are meant to be rotated into the model's local field frame, where each one
#: then carries the same parity as the pressure component it drives.
STRAIN_TENSOR_NAMES = ("Wxx_e", "Wyy_e", "Wzz_e", "Wxy_e", "Wxz_e", "Wyz_e")

#: Index pairs of :data:`STRAIN_TENSOR_NAMES` into a symmetric 3x3 tensor.
STRAIN_TENSOR_INDEX_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

#: Fourth-order central first-derivative weights (Menura's stencil).
_D4 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0


def _d4(field: np.ndarray, axis: int, delta: float) -> np.ndarray:
    """Fourth-order central derivative with periodic wrap along ``axis``.

    Written with ``np.roll`` rather than a convolution so the expression is
    literally Menura's stencil and can be read against ``kernels_fields.cuh``.
    """
    return (
        8.0 * (np.roll(field, -1, axis) - np.roll(field, 1, axis))
        - (np.roll(field, -2, axis) - np.roll(field, 2, axis))
    ) / (12.0 * delta)


def strain_tensor(
    velocity: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the symmetric rate-of-strain tensor and the flow divergence.

    Parameters
    ----------
    velocity : ndarray, shape ``(3, nx, ny)``
        Bulk-flow components ``(u_x, u_y, u_z)`` in Alfven units.  Trailing
        axes beyond the two spatial ones are broadcast over.
    dx, dy : float
        Grid spacing in ion inertial lengths.

    Returns
    -------
    W : ndarray, shape ``(3, 3, nx, ny)``
        ``W_ij = 0.5 (d_i u_j + d_j u_i)``, with ``d_z = 0``.
    divergence : ndarray, shape ``(nx, ny)``
        ``div u = d_x u_x + d_y u_y`` (the ``d_z u_z`` term vanishes in 2-D).
    """
    velocity = np.asarray(velocity)
    if velocity.ndim < 3 or velocity.shape[0] != 3:
        raise ValueError("velocity must have shape (3, nx, ny[, ...])")

    # grad[i, j] = d_i u_j; the d_z row is identically zero in a 2-D run.
    grad = np.zeros((3, 3) + velocity.shape[1:], dtype=float)
    for j in range(3):
        grad[0, j] = _d4(velocity[j], axis=0, delta=dx)
        grad[1, j] = _d4(velocity[j], axis=1, delta=dy)

    strain = 0.5 * (grad + np.swapaxes(grad, 0, 1))
    divergence = grad[0, 0] + grad[1, 1]
    return strain, divergence


def flow_strain_components(
    velocity: np.ndarray,
    dx: float,
    dy: float,
) -> dict[str, np.ndarray]:
    """Return the six independent Cartesian components of ``W``.

    These are the parity-matched inputs: rotated into the local field frame
    they transform exactly as the pressure components they drive, so
    ``W_12`` shares the parity of ``P_12`` under ``b -> -b`` where a rotational
    magnitude cannot.  See ``diagnostics/.../fieldframe_closure``.
    """
    strain, _ = strain_tensor(velocity, dx, dy)
    return {
        name: strain[i, j]
        for name, (i, j) in zip(STRAIN_TENSOR_NAMES, STRAIN_TENSOR_INDEX_PAIRS)
    }


def flow_gradient_invariants(
    magnetic: np.ndarray,
    velocity: np.ndarray,
    dx: float,
    dy: float,
    field_epsilon: float = 1.0e-12,
) -> dict[str, np.ndarray]:
    """Return the four Tier-2 rotational scalars as a name -> array mapping.

    Parameters
    ----------
    magnetic : ndarray, shape ``(3, nx, ny)``
        Magnetic field in Alfven units; only its direction is used.
    velocity : ndarray, shape ``(3, nx, ny)``
        Electron bulk flow in Alfven units.
    dx, dy : float
        Grid spacing in ion inertial lengths.
    field_epsilon : float
        Floor on ``|B|`` used to build ``b``.  At an exact magnetic null the
        parallel direction is undefined; the projections below then degrade
        smoothly rather than producing NaNs.

    Returns
    -------
    dict of ndarray
        Keys are :data:`INVARIANT_NAMES`, each of shape ``(nx, ny)`` and in
        units of ``Omega_ci``.
    """
    magnetic = np.asarray(magnetic, dtype=float)
    if magnetic.ndim < 3 or magnetic.shape[0] != 3:
        raise ValueError("magnetic must have shape (3, nx, ny[, ...])")
    if magnetic.shape[1:] != np.shape(velocity)[1:]:
        raise ValueError("magnetic and velocity must share their grid shape")

    strain, divergence = strain_tensor(velocity, dx, dy)

    bmag = np.sqrt((magnetic**2).sum(axis=0))
    bhat = magnetic / np.maximum(bmag, field_epsilon)

    # W.b  (contract the second index)
    strain_dot_b = np.einsum("ij...,j...->i...", strain, bhat)
    # b.W.b : the CGL parallel-pressure driver.
    parallel = np.einsum("i...,i...->...", bhat, strain_dot_b)
    # (I - bb).W.b : the mixed parallel/perpendicular block whose magnitude is
    # the leading gyroviscous (agyrotropic) source, Braginskii Pi ~ W / Omega_ce.
    mixed_vector = strain_dot_b - parallel * bhat
    mixed = np.sqrt((mixed_vector**2).sum(axis=0))

    # Traceless perpendicular block.  Its Frobenius norm follows from tensor
    # invariants, avoiding an explicit perpendicular basis (and therefore any
    # gauge choice): with P = I - bb,
    #     ||P W P||^2 = ||W||^2 - 2|W.b|^2 + (b.W.b)^2
    # and removing its trace, tr(P W P) = tr(W) - b.W.b, subtracts
    # 0.5 * tr(P W P)^2 because the perpendicular identity has norm^2 = 2.
    strain_norm_sq = np.einsum("ij...,ij...->...", strain, strain)
    strain_dot_b_sq = np.einsum("i...,i...->...", strain_dot_b, strain_dot_b)
    perp_block_sq = strain_norm_sq - 2.0 * strain_dot_b_sq + parallel**2
    perp_trace = divergence - parallel
    perp_traceless_sq = perp_block_sq - 0.5 * perp_trace**2
    perpendicular = np.sqrt(np.maximum(perp_traceless_sq, 0.0))

    return {
        "Wpar_e": parallel,
        "divV_e": divergence,
        "Wmix_e": mixed,
        "Wperp_e": perpendicular,
    }
