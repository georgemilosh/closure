"""Plasma-physics and spectral analysis helpers for ``closure``.

This module contains numerical differentiation, field diagnostics,
pressure-strain analysis, filtering, and spectrum utilities that were
previously mixed into ``closure.utilities``.
"""

from __future__ import annotations

__all__ = [
    "alfven_scales",
    "apply_filter",
    "code2alfven",
    "do_cross",
    "do_dot",
    "get_Az",
    "get_Az_3D",
    "get_D",
    "get_J_perp",
    "get_J_perp_3D",
    "get_Ohm",
    "get_Ohm_3D",
    "get_PS_2D",
    "get_PS_2D_field",
    "get_PS_3D",
    "get_PS_3D_field",
    "get_T",
    "get_W",
    "get_agyrotropy",
    "get_spectral_index",
    "find_xo_points",
    "highdiff",
    "track_xo_points",
    "scalar_spectrum_2D",
    "scale_filtering",
    "vector_spectrum_2D",
]

from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as nd


ArrayLike = np.ndarray
DataDict = dict[str, Any]


def _parse_first_float(value: str) -> float:
    """Parse the first float token from a string value."""
    for token in value.replace(",", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    raise ValueError(f"No float value found in: {value!r}")


def _parse_float_list(value: str) -> list[float]:
    """Parse all float tokens from a string value."""
    out: list[float] = []
    for token in value.replace(",", " ").split():
        try:
            out.append(float(token))
        except ValueError:
            continue
    if not out:
        raise ValueError(f"No float values found in: {value!r}")
    return out


def _find_experiment_inp_file(experiment: str) -> Path:
    """Locate an experiment ``.inp`` file from an absolute experiment path.

    If the resolved path lives under ``/readonly/`` and no ``.inp`` file is
    found there (the snapshot may lag behind live writes), the function retries
    with the leading ``/readonly`` prefix stripped so that the live filesystem
    is used instead.
    """
    import re as _re

    exp_path = Path(experiment).expanduser()

    if not exp_path.is_absolute():
        raise ValueError(
            "experiment must be an absolute path to an experiment directory or .inp file"
        )

    def _try_path(p: Path):
        # Accept explicit .inp file path.
        if p.is_file() and p.suffix == ".inp":
            return p.resolve()
        # Accept explicit directory path containing an .inp file.
        if p.is_dir():
            inp_files = sorted(p.glob("*.inp"))
            if inp_files:
                return inp_files[0].resolve()
        return None

    result = _try_path(exp_path)
    if result is not None:
        return result

    # /readonly/ is a periodic snapshot; fall back to live path if needed.
    exp_str = str(exp_path)
    live_str = _re.sub(r"^/readonly(?=/)", "", exp_str)
    if live_str != exp_str:
        live_path = Path(live_str)
        result = _try_path(live_path)
        if result is not None:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f".inp file not found at snapshot path {exp_path!r}; "
                f"falling back to live path {live_path!r}"
            )
            return result

    raise FileNotFoundError(
        f"Could not locate experiment path {exp_path}. "
        "Provide experiment as an absolute path to a run folder or .inp file."
    )


def _read_b0x_nb_from_inp(inp_path: Path) -> tuple[float, float]:
    """Read ``B0x`` and first entry of ``rhoINIT`` (nb) from an iPiC input file."""
    b0x_value: float | None = None
    nb_value: float | None = None

    for raw_line in inp_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key == "B0x":
            b0x_value = _parse_first_float(value)
        elif key == "rhoINIT":
            rho_values = _parse_float_list(value)
            if len(rho_values) < 1:
                raise ValueError(
                    f"rhoINIT in {inp_path} must contain at least 1 value to infer nb"
                )
            nb_value = rho_values[0]

    if b0x_value is None:
        raise ValueError(f"B0x not found in {inp_path}")
    if nb_value is None:
        raise ValueError(f"rhoINIT (1st value for nb) not found in {inp_path}")
    return b0x_value, nb_value


def scalar_spectrum_2D(field: ArrayLike, x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Compute a scalar 2D isotropic spectrum from time-dependent data."""
    lx = x[-1] - x[0]
    ly = y[-1] - y[0]
    nxc = len(x)
    nyc = len(y)
    t = np.arange(field.shape[-1])

    field_ft = np.fft.rfft2(field[0:-1, 0:-1, :], axes=(0, 1))
    spec_2d = (abs(field_ft) ** 2) / ((nxc * nyc) ** 2)
    spec_2d[:, 1:-1, :] *= 2
    kx = np.fft.fftfreq(nxc - 1, x[1] - x[0]) * 2 * np.pi
    ky = np.fft.rfftfreq(nyc - 1, y[1] - y[0]) * 2 * np.pi

    spec_1d = np.zeros((nxc // 2 + 1, len(t)))
    for iy in range(len(ky)):
        for ix in range(len(kx)):
            index = round(np.sqrt((lx * kx[ix] / (2 * np.pi)) ** 2 + (ly * ky[iy] / (2 * np.pi)) ** 2))
            if index <= (nxc // 2):
                spec_1d[index, :] += spec_2d[ix, iy, :]

    return ky, spec_1d[:-1]


def vector_spectrum_2D(
    field_x: ArrayLike,
    field_y: ArrayLike,
    field_z: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[ArrayLike, ArrayLike]:
    """Compute an isotropic 2D spectrum for a vector field."""
    if len(x.shape) == 1:
        lx = x[-1] - x[0]
        ly = y[-1] - y[0]
        x_axis = x
        y_axis = y
    elif len(x.shape) == 2:
        lx = x[-1, 0] - x[0, 0]
        ly = y[0, -1] - y[0, 0]
        x_axis = x[:, 0]
        y_axis = y[0, :]
    else:
        raise ValueError("X and Y must be 1D or 2D arrays")

    t = np.arange(field_x.shape[-1])
    nxc = len(x)
    nyc = len(y)

    field_x_ft = np.fft.rfft2(field_x[0:-1, 0:-1, :], axes=(0, 1))
    field_y_ft = np.fft.rfft2(field_y[0:-1, 0:-1, :], axes=(0, 1))
    field_z_ft = np.fft.rfft2(field_z[0:-1, 0:-1, :], axes=(0, 1))

    spec_2d = (abs(field_x_ft) ** 2 + abs(field_y_ft) ** 2 + abs(field_z_ft) ** 2) / ((nxc * nyc) ** 2)
    spec_2d[:, 1:-1, :] *= 2
    kx = np.fft.fftfreq(nxc - 1, x_axis[1] - x_axis[0]) * 2 * np.pi
    ky = np.fft.rfftfreq(nyc - 1, y_axis[1] - y_axis[0]) * 2 * np.pi

    spec_1d = np.zeros((nxc // 2 + 1, len(t)))
    for iy in range(len(ky)):
        for ix in range(len(kx)):
            index = round(np.sqrt((lx * kx[ix] / (2 * np.pi)) ** 2 + (ly * ky[iy] / (2 * np.pi)) ** 2))
            if index <= (nxc // 2):
                spec_1d[index, :] += spec_2d[ix, iy, :]

    return ky, spec_1d


def get_spectral_index(k: ArrayLike, spec: ArrayLike, n_points: int) -> tuple[ArrayLike, ArrayLike]:
    """Fit local log-log slopes of a spectrum."""
    from scipy.optimize import curve_fit

    def line(xval: ArrayLike, a: float, b: float) -> ArrayLike:
        return a * xval + b

    xvals = np.log10(k[1:])
    yvals = np.log10(spec[1:])

    k_reduced = []
    slopes = []
    for i in range(len(k) // n_points):
        params, _ = curve_fit(
            line,
            xvals[i * n_points : (i + 1) * n_points],
            yvals[i * n_points : (i + 1) * n_points],
            sigma=yvals[i * n_points : (i + 1) * n_points],
        )
        k_reduced.append(np.mean(k[i * n_points + 1 : (i + 1) * n_points + 1]))
        slopes.append(params[0])

    return np.array(k_reduced), np.array(slopes)


def alfven_scales(
    b0x: float,
    nb: float,
) -> dict[str, float]:
    """Return the Alfvén reference scales for a given ``b0x`` and ``nb``.

    Returns
    -------
    dict
        Keys: ``b0x``, ``nb``, ``va``, ``j0``, ``p0``, ``e0``.
    """
    if b0x is None or nb is None:
        raise ValueError(
            f"alfven_scales requires non-None values for b0x and nb. "
            f"Got b0x={b0x}, nb={nb}"
        )
    va = b0x / np.sqrt(nb)
    return {
        "b0x": b0x,
        "nb": nb,
        "va": va,
        "j0": nb * va,
        "p0": nb * va ** 2,
        "e0": va * b0x,
    }


def code2alfven(
    data: DataDict,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    times: list[float] | None = None,
    b0x: float | None = None,
    nb: float | None = None,
    experiment: str | None = None,
) -> tuple[ArrayLike | None, ArrayLike | None, list[float] | None]:
    """Rescale code units to Alfven units.

    The *data* dictionary is modified **in-place**.

    When ``b0x`` or ``nb`` are missing and ``experiment`` is provided,
    values are inferred from ``*.inp``:
    - ``B0x`` line for ``b0x``
    - first entry of ``rhoINIT`` for ``nb``

    ``experiment`` must be an absolute path to either the run directory
    containing an ``.inp`` file or the ``.inp`` file itself.

    ``x``, ``y`` and ``times`` are optional.  When ``None`` the
    corresponding coordinate transform is skipped and ``None`` is
    returned in that position.
    """
    if (b0x is None or nb is None) and experiment is not None:
        try:
            inferred_b0x, inferred_nb = _read_b0x_nb_from_inp(_find_experiment_inp_file(experiment))
            if b0x is None:
                b0x = inferred_b0x
            if nb is None:
                nb = inferred_nb
            print(f"Inferred b0x={b0x} and nb={nb} from {experiment!r}")
        except FileNotFoundError as e:
            raise ValueError(
                f"Could not infer b0x and nb from experiment: {e}. "
                f"Please provide b0x and nb directly."
            ) from e
        except ValueError as e:
            raise ValueError(
                f"Failed to parse b0x or nb from experiment .inp file: {e}. "
                f"Please provide b0x and nb directly."
            ) from e

    if b0x is None or nb is None:
        raise ValueError(
            f"code2alfven requires b0x and nb. Provide them directly or set experiment to infer from *.inp. "
            f"Got b0x={b0x}, nb={nb}"
        )

    sc = alfven_scales(b0x, nb)
    va = sc["va"]
    j0 = sc["j0"]
    p0 = sc["p0"]
    e0 = sc["e0"]

    for field_name in ["Bx", "By", "Bz"]:
        try:
            data[field_name] = data[field_name] / b0x
        except Exception:
            pass
    if "Bmagn" in data:
        data["Bmagn"] = data["Bmagn"] / b0x

    for field_name in [
        "Ex",
        "Ey",
        "Ez",
        "EPx",
        "EPy",
        "EPz",
        "EHallx",
        "EHally",
        "EHallz",
        "Ohmresx",
        "Ohmresy",
        "Ohmresz",
    ]:
        try:
            data[field_name] = data[field_name] / e0
        except Exception:
            pass
    if "Emagn" in data:
        data["Emagn"] = data["Emagn"] / e0

    for field_name in ["Jx", "Jy", "Jz", "Jmagn"]:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec] / j0
        except Exception:
            pass

    for field_name in ["Jtotx", "Jtoty", "Jtotz"]:
        try:
            data[field_name] = data[field_name] / j0
        except Exception:
            pass

    for field_name in ["rho"]:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec] / nb
        except Exception:
            pass

    for field_name in ["Vx", "Vy", "Vz"]:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec] / va
        except Exception:
            pass

    for field_name in ["Pxx", "Pxy", "Pxz", "Pyx", "Pyy", "Pyz", "Pzx", "Pzy", "Pzz", "Ppar", "Pperp"]:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec] / p0
        except Exception:
            pass

    for field_name in ["qx", "qy", "qz", "EFx", "EFy", "EFz"]:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec] / (p0 * va)
        except Exception:
            pass

    for field_name in ["gyro_radius"]:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec] / (va / b0x)
        except Exception:
            pass

    x_out = x * np.sqrt(nb) if x is not None else None
    y_out = y * np.sqrt(nb) if y is not None else None
    t_out = [t * b0x for t in times] if times is not None else None
    return x_out, y_out, t_out

def find_xo_points(A, x=None, y=None, grad_tol=1e-8, merge_tol=1e-3):
    """
    Find O-points and X-points in a 2D scalar field A sampled on a regular grid.

    Parameters
    ----------
    A : 2D ndarray, shape (nx, ny)
        Scalar field values.
    x, y : 1D ndarrays
        Grid coordinates. If None, use integer indices.
    grad_tol : float
        Tolerance on |grad psi| for accepting a critical point.
    merge_tol : float
        Distance tolerance for merging duplicate roots.

    Returns
    -------
    o_points : list of dict
    x_points : list of dict
    """
    from scipy.interpolate import RectBivariateSpline
    from scipy.optimize import root

    nx, ny = A.shape
    if x is None:
        x = np.arange(nx, dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    if y is None:
        y = np.arange(ny, dtype=float)
    else:
        y = np.asarray(y, dtype=float)

    spline = RectBivariateSpline(x, y, A)

    def grad(z):
        xx, yy = z
        gx = spline.ev(xx, yy, dx=1, dy=0)
        gy = spline.ev(xx, yy, dx=0, dy=1)
        return np.array([gx, gy])

    def hessian(z):
        xx, yy = z
        gxx = spline.ev(xx, yy, dx=2, dy=0)
        gyy = spline.ev(xx, yy, dx=0, dy=2)
        gxy = spline.ev(xx, yy, dx=1, dy=1)
        return np.array([[gxx, gxy],
                         [gxy, gyy]])

    def value(z):
        return float(spline.ev(z[0], z[1]))

    def nearest_indices(z):
        ix = int(np.argmin(np.abs(x - z[0])))
        iy = int(np.argmin(np.abs(y - z[1])))
        return ix, iy

    # candidate seeds: points where gradient magnitude is locally small
    dAx = np.gradient(A, x, axis=0)
    dAy = np.gradient(A, y, axis=1)
    gmag = np.sqrt(dAx**2 + dAy**2)

    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(gmag, size=3, mode="nearest")
    # A cell is a local minimum when its value <= the neighbourhood min
    # (with a small relative tolerance for symmetric fields).
    # Exclude the 1-cell border so seeds stay inside the domain.
    interior_mask = gmag[1:-1, 1:-1] <= local_min[1:-1, 1:-1] * (1 + 1e-10)
    ii, jj = np.nonzero(interior_mask)
    candidates = [(x[i + 1], y[j + 1]) for i, j in zip(ii, jj)]

    roots = []
    for seed in candidates:
        # Try hybr first (fast); fall back to lm (Levenberg-Marquardt) which is
        # more robust when the Jacobian is ill-conditioned (e.g. near X-points
        # whose Hessian has mixed-sign eigenvalues).
        # Note: hybr uses xtol (step size); ftol is not a valid option for hybr.
        sol = root(grad, seed, method="hybr", options={"xtol": grad_tol})
        if not sol.success or np.linalg.norm(sol.fun) > grad_tol:
            sol = root(grad, seed, method="lm")

        # Accept based on actual residual, not sol.success alone, which can be
        # False even when the optimizer has converged close enough to a critical
        # point, and can be True when hybr converged to the wrong place.
        if np.linalg.norm(sol.fun) > grad_tol:
            continue

        z = sol.x
        xx, yy = z

        # inside domain
        if not (x[0] <= xx <= x[-1] and y[0] <= yy <= y[-1]):
            continue

        roots.append(z)

    # merge duplicates
    unique = []
    for z in roots:
        if not any(np.linalg.norm(z - q) < merge_tol for q in unique):
            unique.append(z)

    o_points = []
    x_points = []

    for z in unique:
        H = hessian(z)
        eig = np.linalg.eigvalsh(H)
        detH = np.linalg.det(H)
        ix, iy = nearest_indices(z)

        entry = {
            "x": float(z[0]),
            "y": float(z[1]),
            "ix": ix,
            "iy": iy,
            "value": value(z),
            "eigvals": eig,
            "detH": float(detH),
        }

        # Use detH sign for robust classification (the strict eigenvalue
        # comparisons have a silent gap when detH≈0 and one eigenvalue is
        # numerically zero, causing valid critical points to be dropped).
        # detH < 0  → saddle  → X-point
        # detH > 0, trace > 0 → local min → O-min
        # detH > 0, trace < 0 → local max → O-max
        # detH ≈ 0 → degenerate ridge; skip (cannot classify reliably)
        trH = float(eig[0] + eig[1])
        if detH < 0:
            entry["type"] = "X"
            x_points.append(entry)
        elif detH > 0 and trH > 0:
            entry["type"] = "O-min"
            o_points.append(entry)
        elif detH > 0 and trH < 0:
            entry["type"] = "O-max"
            o_points.append(entry)
        else:
            _log = __import__("logging").getLogger(__name__)
            _log.debug("Skipping degenerate critical point at (%.4f, %.4f): detH=%.3e", z[0], z[1], detH)

    import logging as _logging
    _log = _logging.getLogger(__name__)
    # Warnings reference `unique` (the actual output), not any loop variable.
    if not x_points:
        _log.warning(
            "No X-points found (unique roots: %d, candidates: %d)",
            len(unique), len(candidates),
        )
    if not o_points:
        _log.warning(
            "No O-points found (unique roots: %d, candidates: %d)",
            len(unique), len(candidates),
        )
    return o_points, x_points


def track_xo_points(
    data: DataDict,
    X: ArrayLike,
    Y: ArrayLike,
    times: ArrayLike,
    *,
    az_key: str = "Az",
    az_filter: dict[str, Any] | None = None,
    grad_tol: float = 1e-8,
    merge_tol: float = 1e-3,
    max_opoint_jump: float = 3.0,
    rate_sigma_clip: float = 15.0,
) -> dict[str, ArrayLike]:
    """
    Track X- and O-points in Az over time with temporal continuity.

    At each snapshot the function calls ``find_xo_points`` on (optionally
    filtered) Az and applies two stabilisation strategies:

    * **Nearest-neighbour continuity** — when multiple O-points are found,
      prefer the one geometrically closest to the last accepted O-point
      rather than always picking the global Az minimum.
    * **Displacement rejection** — if the selected O-point has moved more
      than *max_opoint_jump* (in the same units as X/Y) since the last
      accepted step, treat it as a tracking artefact: set the O-point
      quantities to NaN for that snapshot and keep the previous accepted
      position as the reference for the next comparison.

    Reconnection rate ``d/dt(Az_O - Az_X)`` is computed after linearly
    interpolating over NaN gaps in the flux series, then applying a
    MAD-based sigma-clip to remove residual single-timestep artefacts.

    Parameters
    ----------
    data : dict
        Field-data dict as returned by ``rp.get_exp_times``.  Must contain
        the Az field under ``az_key`` with shape ``(nx, ny, nt)``.
    X : ndarray, shape (nx, ny) or (nx,)
        Physical x-coordinates (Alfvén units).  2-D meshgrid arrays
        (``indexing="ij"``) are accepted; the first column is extracted.
    Y : ndarray, shape (nx, ny) or (ny,)
        Physical y-coordinates.  2-D meshgrid arrays are accepted; the
        first row is extracted.
    times : array-like, shape (nt,)
        Physical times corresponding to the third axis of Az.
    az_key : str
        Key in *data* for the Az field (default ``"Az"``).
    az_filter : dict or None
        Filter spec forwarded to :func:`apply_filter` before calling
        ``find_xo_points``.  Example::

            {"name": "gaussian_filter", "sigma": 4, "axes": (0, 1)}

        ``None`` (default) skips filtering.
    grad_tol : float
        Gradient-magnitude tolerance passed to :func:`find_xo_points`.
    merge_tol : float
        Duplicate-root merge tolerance passed to :func:`find_xo_points`.
    max_opoint_jump : float
        Maximum O-point displacement between consecutive accepted steps
        (same units as X/Y).  Larger displacements are rejected as
        tracking artefacts.
    rate_sigma_clip : float
        Points in ``recon_rate`` more than *rate_sigma_clip* × MAD from
        the median are set to NaN.  Default 15.0 catches only extreme
        single-step spikes while preserving genuine physics.

    Returns
    -------
    dict with keys (all 1-D arrays of length nt):
        ``"time"``       — physical times
        ``"xpoint_x"``   — X-point x-coordinate
        ``"xpoint_y"``   — X-point y-coordinate
        ``"xpoint_ix"``  — X-point nearest grid index along x
        ``"xpoint_iy"``  — X-point nearest grid index along y
        ``"opoint_x"``   — O-point x-coordinate (NaN where rejected/missing)
        ``"opoint_y"``   — O-point y-coordinate
        ``"opoint_ix"``  — O-point nearest grid index along x (NaN where rejected)
        ``"opoint_iy"``  — O-point nearest grid index along y (NaN where rejected)
        ``"Az_X"``       — Az at the X-point
        ``"Az_O"``       — Az at the O-point (NaN where rejected)
        ``"recon_flux"`` — ``Az_O - Az_X`` (NaN where O-point rejected)
        ``"recon_rate"`` — ``d/dt recon_flux``, gap-interpolated then clipped
    """
    Az_field = np.asarray(data[az_key])
    x_arr = np.asarray(X if X.ndim == 1 else X[:, 0], dtype=float)
    y_arr = np.asarray(Y if Y.ndim == 1 else Y[0, :], dtype=float)
    times_arr = np.asarray(times, dtype=float)
    nt = Az_field.shape[2]

    keys = [
        "xpoint_x", "xpoint_y", "xpoint_ix", "xpoint_iy",
        "opoint_x", "opoint_y", "opoint_ix", "opoint_iy", "Az_X", "Az_O", "recon_flux",
    ]
    result: dict[str, ArrayLike] = {k: np.full(nt, np.nan) for k in keys}

    prev_opoint: tuple[float, float] | None = None

    for t in range(nt):
        Az_t = Az_field[..., t]
        Az_smooth = apply_filter(Az_t, filters=az_filter) if az_filter is not None else Az_t

        o_pts, x_pts = find_xo_points(
            Az_smooth, x=x_arr, y=y_arr,
            grad_tol=grad_tol, merge_tol=merge_tol,
        )
        # Skip only if *both* are absent — if just one type is missing (common
        # in early/late phases of long runs like ECsim) still record what we have.
        if not x_pts and not o_pts:
            continue

        if x_pts:
            xpoint = max(x_pts, key=lambda p: p["value"])
            ix, iy = xpoint["ix"], xpoint["iy"]
            result["xpoint_x"][t]  = xpoint["x"]
            result["xpoint_y"][t]  = xpoint["y"]
            result["xpoint_ix"][t] = ix
            result["xpoint_iy"][t] = iy
            result["Az_X"][t]      = float(Az_field[ix, iy, t])

        if o_pts:
            # Prefer O-point nearest to previous location for temporal continuity.
            if len(o_pts) == 1 or prev_opoint is None:
                opoint = min(o_pts, key=lambda p: p["value"])
            else:
                opoint = min(
                    o_pts,
                    key=lambda p: np.hypot(p["x"] - prev_opoint[0], p["y"] - prev_opoint[1]),
                )

            # Reject O-point if it jumped further than max_opoint_jump from the
            # last accepted position — keeps prev_opoint unchanged as reference.
            opoint_ok = prev_opoint is None or (
                np.hypot(opoint["x"] - prev_opoint[0], opoint["y"] - prev_opoint[1])
                <= max_opoint_jump
            )

            if opoint_ok:
                result["opoint_x"][t]   = float(opoint["x"])
                result["opoint_y"][t]   = float(opoint["y"])
                result["opoint_ix"][t]  = opoint["ix"]
                result["opoint_iy"][t]  = opoint["iy"]
                result["Az_O"][t]       = float(opoint["value"])
                prev_opoint             = (float(opoint["x"]), float(opoint["y"]))

        # recon_flux requires both points
        if np.isfinite(result["Az_X"][t]) and np.isfinite(result["Az_O"][t]):
            result["recon_flux"][t] = result["Az_O"][t] - result["Az_X"][t]

    # Reconnection rate: interpolate over NaN gaps before differentiating to
    # avoid large spurious spikes at gap boundaries.
    flux = result["recon_flux"].copy()
    finite = np.isfinite(flux)
    if finite.sum() >= 2:
        flux_interp = np.interp(times_arr, times_arr[finite], flux[finite])
        rate = np.gradient(flux_interp, times_arr, edge_order=2)
        finite_rate = rate[np.isfinite(rate)]
        if finite_rate.size > 4 and rate_sigma_clip > 0:
            med = np.median(finite_rate)
            mad = np.median(np.abs(finite_rate - med)) * 1.4826
            if mad > 0:
                rate[np.abs(rate - med) > rate_sigma_clip * mad] = np.nan
        rate[~finite] = np.nan
    else:
        rate = np.full(nt, np.nan)

    result["recon_rate"] = rate
    result["time"] = times_arr
    return result


def do_dot(fx: ArrayLike, fy: ArrayLike, fz: ArrayLike, gx: ArrayLike, gy: ArrayLike, gz: ArrayLike) -> ArrayLike:
    """Return the dot product of two vector fields."""
    return fx * gx + fy * gy + fz * gz


def do_cross(
    fx: ArrayLike,
    fy: ArrayLike,
    fz: ArrayLike,
    gx: ArrayLike,
    gy: ArrayLike,
    gz: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Return the cross product of two vector fields."""
    return fy * gz - fz * gy, fz * gx - fx * gz, fx * gy - fy * gx


def get_PS_3D_field(data: DataDict, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> None:
    """Compute pressure-strain diagnostics for a 3D field dictionary in place."""
    data["QJ"] = {}
    data["Qomega"] = {}
    data["QD"] = {}
    data["PiD"] = {}
    data["Ptheta"] = {}
    data["PS"] = {}
    data["theta"] = {}
    data["Dxx"] = {}
    data["Dyy"] = {}
    data["Dzz"] = {}
    data["Dxy"] = {}
    data["Dxz"] = {}
    data["Dyz"] = {}
    data["Ppar"] = {}
    data["Pperp"] = {}
    data["P"] = {}
    data["J*(E+VxB)"] = {}
    data["Jtotx"] = np.sum([data["Jx"][species] for species in data["Jx"].keys()], axis=0)
    data["Jtoty"] = np.sum([data["Jy"][species] for species in data["Jy"].keys()], axis=0)
    data["Jtotz"] = np.sum([data["Jz"][species] for species in data["Jz"].keys()], axis=0)
    e_field = np.array([data["Ex"], data["Ey"], data["Ez"]]).transpose(1, 2, 3, 4, 0)
    b_field = np.array([data["Bx"], data["By"], data["Bz"]]).transpose(1, 2, 3, 4, 0)
    j2 = data["Jtotx"] ** 2 + data["Jtoty"] ** 2 + data["Jtotz"] ** 2
    data["QJ"] = 0.25 * j2 / np.mean(j2, axis=(0, 1, 2))
    for species in data["rho"].keys():
        j_field = np.array([data["Jx"][species], data["Jy"][species], data["Jz"][species]]).transpose(1, 2, 3, 4, 0)
        v_field = np.array([data["Vx"][species], data["Vy"][species], data["Vz"][species]]).transpose(1, 2, 3, 4, 0)
        data["J*(E+VxB)"][species] = np.sum(j_field * (e_field + np.cross(v_field, b_field)), axis=-1)
        uxx = np.gradient(data["Vx"][species], x, axis=0, edge_order=2)
        uxy = np.gradient(data["Vx"][species], y, axis=1, edge_order=2)
        uyx = np.gradient(data["Vy"][species], x, axis=0, edge_order=2)
        uyy = np.gradient(data["Vy"][species], y, axis=1, edge_order=2)
        uzx = np.gradient(data["Vz"][species], x, axis=0, edge_order=2)
        uzy = np.gradient(data["Vz"][species], y, axis=1, edge_order=2)
        uxz = np.gradient(data["Vx"][species], z, axis=2, edge_order=2)
        uyz = np.gradient(data["Vy"][species], z, axis=2, edge_order=2)
        uzz = np.gradient(data["Vz"][species], z, axis=2, edge_order=2)
        omega2 = (uzy - uyz) ** 2 + (uxz - uzx) ** 2 + (uyx - uxy) ** 2
        data["Qomega"][species] = 0.25 * omega2 / np.mean(omega2, axis=(0, 1, 2))
        data["P"][species] = (data["Pxx"][species] + data["Pyy"][species] + data["Pzz"][species]) / 3
        data["Ppar"][species] = (
            data["Pxx"][species] * data["Bx"] ** 2
            + data["Pyy"][species] * data["By"] ** 2
            + data["Pzz"][species] * data["Bz"] ** 2
            + 2 * data["Pxy"][species] * data["Bx"] * data["By"]
            + 2 * data["Pxz"][species] * data["Bx"] * data["Bz"]
            + 2 * data["Pyz"][species] * data["By"] * data["Bz"]
        ) / (data["By"] ** 2 + data["Bx"] ** 2 + data["Bz"] ** 2)
        data["Pperp"][species] = (
            data["Pxx"][species] + data["Pyy"][species] + data["Pzz"][species] - data["Ppar"][species]
        ) / 2
        data["theta"][species] = uxx + uyy + uzz
        data["PS"][species] = (
            -data["Pxx"][species] * uxx
            - data["Pxy"][species] * uxy
            - data["Pxy"][species] * uyx
            - data["Pyy"][species] * uyy
            - data["Pxz"][species] * uzx
            - data["Pyz"][species] * uzy
            - data["Pxz"][species] * uxz
            - data["Pyz"][species] * uyz
            - data["Pzz"][species] * uzz
        )
        data["Ptheta"][species] = data["P"][species] * data["theta"][species]
        data["Dxx"][species] = uxx - data["theta"][species] / 3
        data["Dyy"][species] = uyy - data["theta"][species] / 3
        data["Dzz"][species] = uzz - data["theta"][species] / 3
        data["Dxy"][species] = (uxy + uyx) / 2
        data["Dxz"][species] = (uxz + uzx) / 2
        data["Dyz"][species] = (uyz + uzy) / 2
        dsum = (
            data["Dxx"][species] ** 2
            + data["Dyy"][species] ** 2
            + data["Dzz"][species] ** 2
            + 2 * (data["Dxy"][species] ** 2 + data["Dxz"][species] ** 2 + data["Dyz"][species] ** 2)
        )
        data["QD"][species] = 0.25 * dsum / np.mean(dsum, axis=(0, 1, 2))
        data["PiD"][species] = (
            -(data["Pxx"][species] - data["P"][species]) * (uxx - data["theta"][species] / 3)
            - (data["Pyy"][species] - data["P"][species]) * (uyy - data["theta"][species] / 3)
            - (data["Pzz"][species] - data["P"][species]) * (uzz - data["theta"][species] / 3)
            - data["Pxy"][species] * (uyx + uxy)
            - data["Pxz"][species] * (uzx + uxz)
            - data["Pyz"][species] * (uzy + uyz)
        )


def get_PS_2D_field(data: DataDict, x: ArrayLike, y: ArrayLike) -> None:
    """Compute pressure-strain diagnostics for a 2D field dictionary in place."""
    data["QJ"] = {}
    data["Qomega"] = {}
    data["QD"] = {}
    data["PiD"] = {}
    data["Ptheta"] = {}
    data["PS"] = {}
    data["theta"] = {}
    data["Dxx"] = {}
    data["Dyy"] = {}
    data["Dzz"] = {}
    data["Dxy"] = {}
    data["Dxz"] = {}
    data["Dyz"] = {}
    data["Ppar"] = {}
    data["Pperp"] = {}
    data["P"] = {}
    data["J*(E+VxB)"] = {}
    data["Jtotx"] = np.sum([data["Jx"][species] for species in data["Jx"].keys()], axis=0)
    data["Jtoty"] = np.sum([data["Jy"][species] for species in data["Jy"].keys()], axis=0)
    data["Jtotz"] = np.sum([data["Jz"][species] for species in data["Jz"].keys()], axis=0)
    e_field = np.array([data["Ex"], data["Ey"], data["Ez"]]).transpose(1, 2, 3, 0)
    b_field = np.array([data["Bx"], data["By"], data["Bz"]]).transpose(1, 2, 3, 0)
    j2 = data["Jtotx"] ** 2 + data["Jtoty"] ** 2 + data["Jtotz"] ** 2
    data["QJ"] = 0.25 * j2 / np.mean(j2, axis=(0, 1))
    for species in data["rho"].keys():
        j_field = np.array([data["Jx"][species], data["Jy"][species], data["Jz"][species]]).transpose(1, 2, 3, 0)
        v_field = np.array([data["Vx"][species], data["Vy"][species], data["Vz"][species]]).transpose(1, 2, 3, 0)
        data["J*(E+VxB)"][species] = np.sum(j_field * (e_field + np.cross(v_field, b_field)), axis=-1)
        uxx = np.gradient(data["Vx"][species], x, axis=0, edge_order=2)
        uxy = np.gradient(data["Vx"][species], y, axis=1, edge_order=2)
        uyx = np.gradient(data["Vy"][species], x, axis=0, edge_order=2)
        uyy = np.gradient(data["Vy"][species], y, axis=1, edge_order=2)
        uzx = np.gradient(data["Vz"][species], x, axis=0, edge_order=2)
        uzy = np.gradient(data["Vz"][species], y, axis=1, edge_order=2)
        omega2 = (uzy) ** 2 + (-uzx) ** 2 + (uyx - uxy) ** 2
        data["Qomega"][species] = 0.25 * omega2 / np.mean(omega2, axis=(0, 1))
        data["P"][species] = (data["Pxx"][species] + data["Pyy"][species] + data["Pzz"][species]) / 3
        data["Ppar"][species] = (
            data["Pxx"][species] * data["Bx"] ** 2
            + data["Pyy"][species] * data["By"] ** 2
            + data["Pzz"][species] * data["Bz"] ** 2
            + 2 * data["Pxy"][species] * data["Bx"] * data["By"]
            + 2 * data["Pxz"][species] * data["Bx"] * data["Bz"]
            + 2 * data["Pyz"][species] * data["By"] * data["Bz"]
        ) / (data["By"] ** 2 + data["Bx"] ** 2 + data["Bz"] ** 2)
        data["Pperp"][species] = (
            data["Pxx"][species] + data["Pyy"][species] + data["Pzz"][species] - data["Ppar"][species]
        ) / 2
        data["theta"][species] = uxx + uyy
        data["PS"][species] = (
            -data["Pxx"][species] * uxx
            - data["Pxy"][species] * uxy
            - data["Pxy"][species] * uyx
            - data["Pyy"][species] * uyy
            - data["Pxz"][species] * uzx
            - data["Pyz"][species] * uzy
        )
        data["Ptheta"][species] = data["P"][species] * data["theta"][species]
        data["Dxx"][species] = uxx - data["theta"][species] / 3
        data["Dyy"][species] = uyy - data["theta"][species] / 3
        data["Dzz"][species] = -data["theta"][species] / 3
        data["Dxy"][species] = (uxy + uyx) / 2
        data["Dxz"][species] = uzx / 2
        data["Dyz"][species] = uzy / 2
        dsum = (
            data["Dxx"][species] ** 2
            + data["Dyy"][species] ** 2
            + data["Dzz"][species] ** 2
            + 2 * (data["Dxy"][species] ** 2 + data["Dxz"][species] ** 2 + data["Dyz"][species] ** 2)
        )
        data["QD"][species] = 0.25 * dsum / np.mean(dsum, axis=(0, 1))
        data["PiD"][species] = (
            -(data["Pxx"][species] - data["P"][species]) * (uxx - data["theta"][species] / 3)
            - (data["Pyy"][species] - data["P"][species]) * (uyy - data["theta"][species] / 3)
            - (data["Pzz"][species] - data["P"][species]) * (-data["theta"][species] / 3)
            - data["Pxy"][species] * (uyx + uxy)
            - data["Pxz"][species] * uzx
            - data["Pyz"][species] * uzy
        )


def get_PS_2D(data: DataDict, x: ArrayLike, y: ArrayLike) -> None:
    """Apply :func:`get_PS_2D_field` to each experiment entry in ``data``."""
    for experiment in data.keys():
        get_PS_2D_field(data[experiment], x, y)


def get_PS_3D(data: DataDict, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> None:
    """Apply :func:`get_PS_3D_field` to each experiment entry in ``data``."""
    for experiment in data.keys():
        get_PS_3D_field(data[experiment], x, y, z)


def get_Ohm_3D(
    data: DataDict,
    qom: list[float],
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    coeff: ArrayLike | None = None,
    small: float = 1e-10,
) -> None:
    """3D extension of :func:`get_Ohm`.

    Expects scalar fields shaped ``(nx, ny, nz, ...)`` (e.g. with a trailing
    time axis) and computes Ohm-law decomposition consistently in all three
    spatial directions.
    """
    b_field = np.array([data["Bx"], data["By"], data["Bz"]]).transpose(1, 2, 3, 4, 0)
    e_field = np.array([data["Ex"], data["Ey"], data["Ez"]]).transpose(1, 2, 3, 4, 0)
    b2 = (data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)[..., np.newaxis]
    data["ExB/B^2"] = np.cross(e_field, b_field) / b2
    data["Jtotx"] = np.sum([data["Jx"][species] for species in data["Jx"].keys()], axis=0)
    data["Jtoty"] = np.sum([data["Jy"][species] for species in data["Jy"].keys()], axis=0)
    data["Jtotz"] = np.sum([data["Jz"][species] for species in data["Jz"].keys()], axis=0)
    j_field = np.array([data["Jtotx"], data["Jtoty"], data["Jtotz"]]).transpose(1, 2, 3, 4, 0)
    data["EHallx"], data["EHally"], data["EHallz"] = (
        np.cross(j_field, b_field) / (-data["rho"]["e"] + small)[..., np.newaxis]
    ).transpose(4, 0, 1, 2, 3)

    norm = 0
    data["uCMx"] = 0
    data["uCMy"] = 0
    data["uCMz"] = 0
    for i, species in enumerate(data["rho"].keys()):
        data["uCMx"] += (data["rho"][species] / qom[i]) * data["Vx"][species]
        data["uCMy"] += (data["rho"][species] / qom[i]) * data["Vy"][species]
        data["uCMz"] += (data["rho"][species] / qom[i]) * data["Vz"][species]
        norm += data["rho"][species] / qom[i]
    data["uCMx"] /= norm
    data["uCMy"] /= norm
    data["uCMz"] /= norm
    ucm = np.array([data["uCMx"], data["uCMy"], data["uCMz"]]).transpose(1, 2, 3, 4, 0)
    data["EMHDx"], data["EMHDy"], data["EMHDz"] = -np.cross(ucm, b_field).transpose(4, 0, 1, 2, 3)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    inv_rho_e = 1.0 / (-data["rho"]["e"] + small)

    # Pressure-gradient electric field: E_P = -(div P_e) / (-rho_e)
    data["EPx"] = -(
        highdiff(data["Pxx"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap")
        + highdiff(data["Pxy"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap")
        + highdiff(data["Pxz"]["e"], dx, dy, coeff=coeff, axis=2, dz=dz, mode="wrap")
    ) * inv_rho_e
    data["EPy"] = -(
        highdiff(data["Pxy"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap")
        + highdiff(data["Pyy"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap")
        + highdiff(data["Pyz"]["e"], dx, dy, coeff=coeff, axis=2, dz=dz, mode="wrap")
    ) * inv_rho_e
    data["EPz"] = -(
        highdiff(data["Pxz"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap")
        + highdiff(data["Pyz"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap")
        + highdiff(data["Pzz"]["e"], dx, dy, coeff=coeff, axis=2, dz=dz, mode="wrap")
    ) * inv_rho_e

    # Electron inertia term: (m_e / e) * (V_e . grad) V_e
    for component in ["x", "y", "z"]:
        v_c = data[f"V{component}"]["e"]
        data[f"mVgradV{component}/e"] = (
            highdiff(v_c, dx, dy, coeff=coeff, axis=0, mode="wrap") * data["Vx"]["e"] / qom[0]
            + highdiff(v_c, dx, dy, coeff=coeff, axis=1, mode="wrap") * data["Vy"]["e"] / qom[0]
            + highdiff(v_c, dx, dy, coeff=coeff, axis=2, dz=dz, mode="wrap") * data["Vz"]["e"] / qom[0]
        )


def get_J_perp_3D(
    data: DataDict,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    coeff: ArrayLike | None = None,
) -> None:
    """3D extension of :func:`get_J_perp`.

    Computes diamagnetic and curvature perpendicular-current contributions
    using a full 3D pressure gradient and 3D unit-vector derivative.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    b_field = np.array([data["Bx"], data["By"], data["Bz"]]).transpose(1, 2, 3, 4, 0)
    data["gradPperpx"] = highdiff(data["Pperp"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap")
    data["gradPperpy"] = highdiff(data["Pperp"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap")
    data["gradPperpz"] = highdiff(data["Pperp"]["e"], dx, dy, coeff=coeff, axis=2, dz=dz, mode="wrap")
    grad_pperp = np.array([data["gradPperpx"], data["gradPperpy"], data["gradPperpz"]]).transpose(1, 2, 3, 4, 0)
    b2 = np.sum(b_field ** 2, axis=-1, keepdims=True)
    data["cross(B,DPperp)/B^2"] = np.cross(b_field, grad_pperp) / b2
    data["b"] = b_field / np.sqrt(b2)
    # (b . grad) b in 3D
    data["b*Db"] = (
        data["b"][..., 0, np.newaxis] * highdiff(data["b"], dx, dy, coeff=coeff, axis=0, mode="wrap")
        + data["b"][..., 1, np.newaxis] * highdiff(data["b"], dx, dy, coeff=coeff, axis=1, mode="wrap")
        + data["b"][..., 2, np.newaxis] * highdiff(data["b"], dx, dy, coeff=coeff, axis=2, dz=dz, mode="wrap")
    )
    data["(Ppar - Pperp) cros(B, b*Db)/B^2"] = (
        (data["Ppar"]["e"] - data["Pperp"]["e"])[..., np.newaxis]
        * np.cross(b_field, data["b*Db"])
        / b2
    )


def get_Az_3D(x: ArrayLike, y: ArrayLike, data: DataDict) -> None:
    """3D extension of :func:`get_Az`.

    Computes the out-of-plane vector potential ``Az(x, y)`` independently per
    ``z``-slice (and per trailing axis, e.g. time) using the same path-integral
    formulation as the 2D version: ``Az = ∫₀ʸ Bx(x, y') dy' - ∫₀ˣ By(x', 0) dx'``.
    """
    bx = data["Bx"]
    by = data["By"]
    nx, ny = bx.shape[0], bx.shape[1]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    f_field = np.zeros_like(bx)
    g_field = np.zeros_like(bx)

    for iy in range(1, ny):
        g_field[:, iy] = g_field[:, iy - 1] + (bx[:, iy - 1] + bx[:, iy]) * dy / 2
    # Use By along the y=0 line as the x-integration boundary, broadcast over
    # remaining axes (z, t, ...).
    for ix in range(1, nx):
        f_field[ix] = f_field[ix - 1] - (by[ix - 1, 0:1] + by[ix, 0:1]) * dx / 2

    data["Az"] = f_field + g_field


def apply_filter(
    field: ArrayLike,
    density: ArrayLike | None = None,
    filters: dict[str, Any] | str = {"name": "uniform_filter", "size": 3, "mode": "wrap", "axes": (0, 1)},
) -> ArrayLike:
    """Apply a scipy.ndimage filter, optionally density-weighted."""
    filters_copy = filters.copy() if isinstance(filters, dict) else filters
    if not isinstance(filters_copy, dict):
        filters_object = getattr(nd, filters_copy)
        filter_kwargs = {}
    else:
        filters_name = filters_copy.pop("name", None)
        filters_object = getattr(nd, filters_name)
        filter_kwargs = filters_copy
        if isinstance(filter_kwargs.get("axes"), list):
            filter_kwargs["axes"] = tuple(filter_kwargs["axes"])

    if density is not None:
        if field.shape == density.shape:
            return filters_object(field * density, **filter_kwargs) / filters_object(density, **filter_kwargs)
        return filters_object(field * density[..., np.newaxis], **filter_kwargs) / filters_object(
            density[..., np.newaxis],
            **filter_kwargs,
        )
    return filters_object(field, **filter_kwargs)


def scale_filtering(
    data: DataDict,
    x: ArrayLike,
    y: ArrayLike,
    qom: list[float],
    verbose: bool = False,
    filters: dict[str, Any] = {"name": "uniform_filter", "size": 100, "mode": "wrap", "axes": (0, 1)},
) -> None:
    """Compute filtered plasma quantities in place."""
    auxiliary: DataDict = {}
    for fields in ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]:
        auxiliary[f"{fields}_bar"] = apply_filter(data[fields], filters=filters)

    for fields in ["Vx", "Vy", "Vz", "Bx", "By", "Bz", "Ex", "Ey", "Ez"]:
        auxiliary[f"{fields}_favre"] = {}

    data["E2_bar"] = (auxiliary["Ex_bar"] ** 2 + auxiliary["Ey_bar"] ** 2 + auxiliary["Ez_bar"] ** 2) / (8 * np.pi)
    data["B2_bar"] = (auxiliary["Bx_bar"] ** 2 + auxiliary["By_bar"] ** 2 + auxiliary["Bz_bar"] ** 2) / (8 * np.pi)
    data["Ef_favre"] = {}
    data["PIuu"] = {}
    data["PIbb"] = {}
    data["PS"] = {}
    data["-Ptheta"] = {}
    data["JdotE"] = {}
    auxiliary["rho_bar"] = {}
    b_field = np.array([data["Bx"], data["By"], data["Bz"]]).transpose(1, 2, 3, 0)
    e_bar = np.array([auxiliary["Ex_bar"], auxiliary["Ey_bar"], auxiliary["Ez_bar"]]).transpose(1, 2, 3, 0)
    for i, species in enumerate(data["rho"].keys()):
        for fields in ["Vx", "Vy", "Vz"]:
            auxiliary[f"{fields}_favre"][species] = apply_filter(
                data[fields][species],
                density=data["rho"][species],
                filters=filters,
            )
        for fields in ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]:
            auxiliary[f"{fields}_favre"][species] = apply_filter(
                data[fields],
                density=data["rho"][species],
                filters=filters,
            )
        auxiliary["rho_bar"][species] = apply_filter(data["rho"][species], filters=filters)
        data["Ef_favre"][species] = 0.5 * auxiliary["rho_bar"][species] * (
            auxiliary["Vx_favre"][species] ** 2
            + auxiliary["Vy_favre"][species] ** 2
            + auxiliary["Vz_favre"][species] ** 2
        ) / qom[i]
        b_favre = np.array(
            [
                auxiliary["Bx_favre"][species],
                auxiliary["By_favre"][species],
                auxiliary["Bz_favre"][species],
            ]
        ).transpose(1, 2, 3, 0)
        e_favre = np.array(
            [
                auxiliary["Ex_favre"][species],
                auxiliary["Ey_favre"][species],
                auxiliary["Ez_favre"][species],
            ]
        ).transpose(1, 2, 3, 0)
        tau_e = e_favre - e_bar
        v_favre = np.array(
            [
                auxiliary["Vx_favre"][species],
                auxiliary["Vy_favre"][species],
                auxiliary["Vz_favre"][species],
            ]
        ).transpose(1, 2, 3, 0)
        data["PIbb"][species] = -auxiliary["rho_bar"][species] * np.sum(tau_e * v_favre, axis=-1)
        data["JdotE"][species] = auxiliary["rho_bar"][species] * np.sum(e_favre * v_favre, axis=-1)

        v_field = np.array([data["Vx"][species], data["Vy"][species], data["Vz"][species]]).transpose(1, 2, 3, 0)
        tau_b = apply_filter(np.cross(v_field, b_field), density=data["rho"][species], filters=filters) - np.cross(v_favre, b_favre)
        dv_favre: dict[str, ArrayLike] = {}
        for component in ["x", "y", "z"]:
            dv_favre[f"{component}x"] = np.gradient(auxiliary[f"V{component}_favre"][species], x, axis=0, edge_order=2)
            dv_favre[f"{component}y"] = np.gradient(auxiliary[f"V{component}_favre"][species], y, axis=1, edge_order=2)
        data["-Ptheta"][species] = 0
        for component in ["x", "y", "z"]:
            data["-Ptheta"][species] += apply_filter(data[f"P{component}{component}"][species], filters=filters)
        data["-Ptheta"][species] *= -(dv_favre["xx"] + dv_favre["yy"]) / 3
        data["PIuu"][species] = 0
        data["PS"][species] = 0
        for component1, component2 in zip(["x", "x", "y", "y", "z", "z"], ["x", "y", "x", "y", "x", "y"]):
            pbar = apply_filter(data[f"P{component1}{component2}"][species], filters=filters)
            if verbose:
                print(f"adding: Pbar{component1}{component2} * nabla dVfavre_{component1}d{component2}")
            data["PS"][species] += -pbar * dv_favre[f"{component1}{component2}"]
            tauu = apply_filter(
                data[f"V{component1}"][species] * data[f"V{component2}"][species],
                density=data["rho"][species],
                filters=filters,
            ) - auxiliary[f"V{component1}_favre"][species] * auxiliary[f"V{component2}_favre"][species]
            data["PIuu"][species] += -auxiliary["rho_bar"][species] * tauu * dv_favre[f"{component1}{component2}"] / qom[i]
        data["PIuu"][species] += -auxiliary["rho_bar"][species] * np.sum(tau_b * v_favre, axis=-1)


def get_T(data: DataDict, qom: list[float]) -> None:
    """Compute species temperatures and beta values in place."""
    data["T"] = {}
    data["T_par"] = {}
    data["T_perp"] = {}
    data["beta_par"] = {}
    bx = data["Bx"] / np.sqrt(data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)
    by = data["By"] / np.sqrt(data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)
    bz = data["Bz"] / np.sqrt(data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)
    for i, species in enumerate(data["rho"].keys()):
        data["T"][species] = (
            data["Pxx"][species] + data["Pyy"][species] + data["Pzz"][species]
        ) / (3 * data["rho"][species] * np.sign(qom[i]))
        data["T_par"][species] = (
            data["Pxx"][species] * bx**2
            + data["Pyy"][species] * by**2
            + data["Pzz"][species] * bz**2
            + 2 * (data["Pxy"][species] * bx * by + data["Pxz"][species] * bx * bz + data["Pyz"][species] * by * bz)
        ) / (data["rho"][species] * np.sign(qom[i]))
        data["T_perp"][species] = (3 * data["T"][species] - data["T_par"][species]) / 2
        data["beta_par"][species] = (
            8
            * np.pi
            * data["T_par"][species]
            * (data["rho"][species] * np.sign(qom[i]))
            / (data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)
        )


def get_agyrotropy(data: DataDict) -> None:
    """Compute agyrotropy for all species in place."""
    data["agyrotropy"] = {}
    bx = data["Bx"] / np.sqrt(data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)
    by = data["By"] / np.sqrt(data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)
    bz = data["Bz"] / np.sqrt(data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)
    for species in data["rho"].keys():
        i1 = data["Pxx"][species] + data["Pyy"][species] + data["Pzz"][species]
        i2 = (
            data["Pxx"][species] * data["Pyy"][species]
            + data["Pxx"][species] * data["Pzz"][species]
            + data["Pyy"][species] * data["Pzz"][species]
            - (data["Pxy"][species] ** 2 + data["Pxz"][species] ** 2 + data["Pyz"][species] ** 2)
        )
        p_par = (
            data["Pxx"][species] * bx**2
            + data["Pyy"][species] * by**2
            + data["Pzz"][species] * bz**2
            + 2 * (data["Pxy"][species] * bx * by + data["Pxz"][species] * bx * bz + data["Pyz"][species] * by * bz)
        )
        data["agyrotropy"][species] = 1 - 4 * i2 / ((i1 - p_par) * (i1 + 3 * p_par))


def highdiff(
    data: ArrayLike,
    dx: float,
    dy: float,
    coeff: ArrayLike | None = None,
    axis: int = 0,
    dz: float | None = None,
    **kwargs: Any,
) -> ArrayLike:
    """Compute a fourth-order central finite-difference derivative.

    Supports ``axis`` 0 (x), 1 (y), and 2 (z). For ``axis=2`` the optional
    ``dz`` argument must be provided. The 2D signature ``(data, dx, dy, ...)``
    is preserved for backward compatibility.
    """
    if coeff is None:
        coeff = np.array([-1, 8, 0, -8, 1]) / 12.0

    if axis == 0:
        dx_kernel = coeff.reshape((-1,) + (1,) * (data.ndim - 1))
        return nd.convolve(data, dx_kernel, output=float, **kwargs) / dx
    if axis == 1:
        dy_kernel = coeff.reshape((1, -1) + (1,) * (data.ndim - 2))
        return nd.convolve(data, dy_kernel, output=float, **kwargs) / dy
    if axis == 2:
        if dz is None:
            raise ValueError("highdiff(axis=2) requires the keyword argument 'dz'.")
        if data.ndim < 3:
            raise ValueError("highdiff(axis=2) requires data with at least 3 dimensions.")
        dz_kernel = coeff.reshape((1, 1, -1) + (1,) * (data.ndim - 3))
        return nd.convolve(data, dz_kernel, output=float, **kwargs) / dz
    raise ValueError("Invalid axis. Use 0, 1 or 2.")


def get_Ohm(
    data: DataDict,
    qom: list[float],
    x: ArrayLike,
    y: ArrayLike,
    coeff: ArrayLike | None = None,
    small: float = 1e-10,
) -> None:
    """Compute Ohm-law related diagnostic fields in place."""
    b_field = np.array([data["Bx"], data["By"], data["Bz"]]).transpose(1, 2, 3, 0)
    e_field = np.array([data["Ex"], data["Ey"], data["Ez"]]).transpose(1, 2, 3, 0)
    data["ExB/B^2"] = np.cross(e_field, b_field) / (data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)[..., np.newaxis]
    data["Jtotx"] = np.sum([data["Jx"][species] for species in data["Jx"].keys()], axis=0)
    data["Jtoty"] = np.sum([data["Jy"][species] for species in data["Jy"].keys()], axis=0)
    data["Jtotz"] = np.sum([data["Jz"][species] for species in data["Jz"].keys()], axis=0)
    j_field = np.array([data["Jtotx"], data["Jtoty"], data["Jtotz"]]).transpose(1, 2, 3, 0)
    data["EHallx"], data["EHally"], data["EHallz"] = (np.cross(j_field, b_field) / (-data["rho"]["e"] + small)[..., np.newaxis]).transpose(3, 0, 1, 2)
    norm = 0
    data["uCMx"] = 0
    data["uCMy"] = 0
    data["uCMz"] = 0
    for i, species in enumerate(data["rho"].keys()):
        data["uCMx"] += (data["rho"][species] / qom[i]) * data["Vx"][species]
        data["uCMy"] += (data["rho"][species] / qom[i]) * data["Vy"][species]
        data["uCMz"] += (data["rho"][species] / qom[i]) * data["Vz"][species]
        norm += data["rho"][species] / qom[i]
    data["uCMx"] /= norm
    data["uCMy"] /= norm
    data["uCMz"] /= norm
    ucm = np.array([data["uCMx"], data["uCMy"], data["uCMz"]]).transpose(1, 2, 3, 0)
    data["EMHDx"], data["EMHDy"], data["EMHDz"] = -np.cross(ucm, b_field).transpose(3, 0, 1, 2)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    data["EPx"] = -(
        highdiff(data["Pxx"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap")
        + highdiff(data["Pxy"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap")
    ) / (-data["rho"]["e"] + small)
    data["EPy"] = -(
        highdiff(data["Pxy"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap")
        + highdiff(data["Pyy"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap")
    ) / (-data["rho"]["e"] + small)
    data["EPz"] = -(
        highdiff(data["Pxz"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap")
        + highdiff(data["Pyz"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap")
    ) / (-data["rho"]["e"] + small)

    data["mVgradVx/e"] = (
        highdiff(data["Vx"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap") * data["Vx"]["e"] / qom[0]
        + highdiff(data["Vx"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap") * data["Vy"]["e"] / qom[0]
    )
    data["mVgradVy/e"] = (
        highdiff(data["Vy"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap") * data["Vx"]["e"] / qom[0]
        + highdiff(data["Vy"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap") * data["Vy"]["e"] / qom[0]
    )
    data["mVgradVz/e"] = (
        highdiff(data["Vz"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap") * data["Vx"]["e"] / qom[0]
        + highdiff(data["Vz"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap") * data["Vy"]["e"] / qom[0]
    )


def get_J_perp(data: DataDict, x: ArrayLike, y: ArrayLike, coeff: ArrayLike | None = None) -> None:
    """Calculate perpendicular current contributions from pressure gradients."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    b_field = np.array([data["Bx"], data["By"], data["Bz"]]).transpose(1, 2, 3, 0)
    data["gradPperpx"] = highdiff(data["Pperp"]["e"], dx, dy, coeff=coeff, axis=0, mode="wrap")
    data["gradPperpy"] = highdiff(data["Pperp"]["e"], dx, dy, coeff=coeff, axis=1, mode="wrap")
    data["gradPperpz"] = np.zeros_like(data["gradPperpx"])
    grad_pperp = np.array([data["gradPperpx"], data["gradPperpy"], data["gradPperpz"]]).transpose(1, 2, 3, 0)
    data["cross(B,DPperp)/B^2"] = np.cross(b_field, grad_pperp) / np.sum(b_field**2, axis=3, keepdims=True)
    data["b"] = b_field / np.sqrt(np.sum(b_field**2, axis=3, keepdims=True))
    data["b*Db"] = (
        data["b"][..., 0, np.newaxis] * highdiff(data["b"], dx, dy, coeff=coeff, axis=0, mode="wrap")
        + data["b"][..., 1, np.newaxis] * highdiff(data["b"], dx, dy, coeff=coeff, axis=1, mode="wrap")
    )
    data["(Ppar - Pperp) cros(B, b*Db)/B^2"] = (
        (data["Ppar"]["e"] - data["Pperp"]["e"])[..., np.newaxis]
        * np.cross(b_field, data["b*Db"])
        / np.sum(b_field**2, axis=3, keepdims=True)
    )


def get_Az(x: ArrayLike, y: ArrayLike, data: DataDict) -> None:
    """Compute the out-of-plane vector potential ``Az`` in place."""
    nx = data["Bx"].shape[0]
    ny = data["Bx"].shape[1]
    nz = data["Bx"].shape[2]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    f_field = np.zeros((nx, ny, nz))
    g_field = np.zeros((nx, ny, nz))

    for iy in range(1, ny):
        g_field[:, iy, :] = g_field[:, iy - 1, :] + (data["Bx"][:, iy - 1, :] + data["Bx"][:, iy, :]) * dy / 2

    for iy in range(ny):
        for ix in range(1, nx):
            f_field[ix, iy, :] = f_field[ix - 1, iy, :] - (data["By"][ix - 1, 0, :] + data["By"][ix, 0, :]) * dx / 2
    data["Az"] = f_field + g_field


def get_W(data: DataDict) -> None:
    """Compute species work terms for each experiment."""
    for experiment in data.keys():
        data[experiment]["W"] = {}
        for species in data[experiment]["rho"].keys():
            data[experiment]["W"][species] = do_dot(
                data[experiment]["Ex"],
                data[experiment]["Ey"],
                data[experiment]["Ez"],
                data[experiment]["Jx"][species],
                data[experiment]["Jy"][species],
                data[experiment]["Jz"][species],
            )


def get_D(data: DataDict) -> None:
    """Compute ``J·J``-style diagnostic terms for each experiment."""
    for experiment in data.keys():
        data[experiment]["D"] = {}
        for species in data[experiment]["rho"].keys():
            data[experiment]["D"][species] = do_dot(
                data[experiment]["Jx"][species],
                data[experiment]["Jy"][species],
                data[experiment]["Jz"][species],
                data[experiment]["Jx"][species],
                data[experiment]["Jy"][species],
                data[experiment]["Jz"][species],
            )
