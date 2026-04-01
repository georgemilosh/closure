"""Plasma-physics and spectral analysis helpers for ``closure``.

This module contains numerical differentiation, field diagnostics,
pressure-strain analysis, filtering, and spectrum utilities that were
previously mixed into ``closure.utilities``.
"""

from __future__ import annotations

__all__ = [
    "apply_filter",
    "code2alfven",
    "do_cross",
    "do_dot",
    "get_Az",
    "get_D",
    "get_J_perp",
    "get_Ohm",
    "get_PS_2D",
    "get_PS_2D_field",
    "get_PS_3D_field",
    "get_T",
    "get_W",
    "get_agyrotropy",
    "get_spectral_index",
    "highdiff",
    "scalar_spectrum_2D",
    "scale_filtering",
    "vector_spectrum_2D",
]

from typing import Any

import numpy as np
import scipy.ndimage as nd


ArrayLike = np.ndarray
DataDict = dict[str, Any]


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


def code2alfven(
    data: DataDict,
    x: ArrayLike,
    y: ArrayLike,
    times: list[float],
    b0x: float,
    nb: float,
) -> tuple[ArrayLike, ArrayLike, list[float]]:
    """Rescale code units to Alfven units."""
    va = b0x / np.sqrt(nb)
    j0 = nb * va
    p0 = nb * va**2
    e0 = va * b0x

    for field_name in ["Bx", "By", "Bz"]:
        try:
            data[field_name] = data[field_name] / b0x
        except Exception:
            pass
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

    return x * np.sqrt(nb), y * np.sqrt(nb), [t * b0x for t in times]


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
    data["QJ"] = 0.25 * j2 / np.mean(j2, axis=(0, 1))
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
    **kwargs: Any,
) -> ArrayLike:
    """Compute a fourth-order central finite-difference derivative."""
    if coeff is None:
        coeff = np.array([-1, 8, 0, -8, 1]) / 12.0

    if axis == 0:
        dx_kernel = coeff.reshape((-1,) + (1,) * (data.ndim - 1))
        return nd.convolve(data, dx_kernel, output=float, **kwargs) / dx
    if axis == 1:
        dy_kernel = coeff.reshape((1, -1) + (1,) * (data.ndim - 2))
        return nd.convolve(data, dy_kernel, output=float, **kwargs) / dy
    raise ValueError("Invalid axis. Use 0 or 1.")


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
