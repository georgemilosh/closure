"""Notebook-style field diagnostics and CSV exports.

This module provides reusable building blocks for command-line diagnostics:
multi-field panel plots, profile cuts saved to CSV, reconnection-rate CSVs,
and overlay plots from exported CSV files.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_FIELDS_TO_READ",
    "FieldSpec",
    "animate_field_panels",
    "apply_normalization",
    "build_profiles_dataframe",
    "compute_common_diagnostics",
    "discover_available_snapshots",
    "discover_menura_iterations",
    "discover_menura_runs",
    "export_bands_dataframe",
    "export_reconnection_dataframe",
    "get_cmap",
    "load_experiment_data",
    "load_menura_data",
    "normalize_field_name",
    "parse_field_specs",
    "plot_csv_overlay",
    "plot_field_panels",
    "resolve_field_data",
    "select_snapshot_indices",
]

import ast
import contextlib
import fnmatch
import importlib
import io
import logging
import operator
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from closure import plasma, read_pic as rp
from closure.config import load_paths

logger = logging.getLogger(__name__)


DEFAULT_FIELDS_TO_READ = {
    "B": True,
    "B_ext": False,
    "divB": True,
    "E": True,
    "E_ext": False,
    "rho": True,
    "J": True,
    "P": True,
    "PI": False,
    "Heat_flux": False,
    "N": False,
    "Qrem": False,
}

# Which coarse read flag(s) each base field name needs, keyed lowercase (species
# suffix already stripped by parse_field_specs). Only plain fields that map to a
# base read group are listed; anything absent is treated as derived/processed and
# triggers a safe fall back to DEFAULT_FIELDS_TO_READ in _read_flags_for_specs.
_FIELD_BASE_TO_FLAGS = {
    "rho": ("rho",),
    "n": ("N",),
    "qrem": ("Qrem",),
    "divb": ("divB",),
    "b": ("B",), "bx": ("B",), "by": ("B",), "bz": ("B",),
    "bmagn": ("B",), "az": ("B",),
    "e": ("E",), "ex": ("E",), "ey": ("E",), "ez": ("E",),
    "j": ("J",), "jx": ("J",), "jy": ("J",), "jz": ("J",),
    "jmagn": ("J",), "jtotx": ("J",), "jtoty": ("J",), "jtotz": ("J",),
    "jx-tot": ("J",), "jy-tot": ("J",), "jz-tot": ("J",), "jinplane": ("J",),
    "v": ("J", "rho"), "vx": ("J", "rho"), "vy": ("J", "rho"),
    "vz": ("J", "rho"), "vmagn": ("J", "rho"),
    "p": ("P",),
    "pxx": ("P",), "pxy": ("P",), "pxz": ("P",),
    "pyy": ("P",), "pyz": ("P",), "pzz": ("P",),
    "pyx": ("P",), "pzx": ("P",), "pzy": ("P",),
    "ppar": ("P",), "pperp": ("P",),
    "pi": ("PI",),
    "pixx": ("PI",), "pixy": ("PI",), "pixz": ("PI",),
    "piyy": ("PI",), "piyz": ("PI",), "pizz": ("PI",),
    "piyx": ("PI",), "pizx": ("PI",), "pizy": ("PI",),
}

# Read groups that are computed inside read_pic.read_data from other groups, so
# enabling them requires enabling their inputs too (P/PI are built from J, rho
# and — for Ppar/Pperp — B; velocity from J and rho).
_READ_FLAG_DEPS = {
    "P": ("P", "J", "rho", "B"),
    "PI": ("PI", "J", "rho"),
}


def _read_flags_for_specs(specs: "list[FieldSpec]") -> dict | None:
    """Minimal ``fields_to_read`` covering ``specs``, or ``None`` to read all.

    Returns ``None`` (meaning: fall back to ``DEFAULT_FIELDS_TO_READ``) whenever a
    requested field is a derived/processed quantity we cannot map to a base read
    group, since those diagnostics may depend on the full field set.
    """
    if not specs:
        return None
    flags = {key: False for key in DEFAULT_FIELDS_TO_READ}
    for spec in specs:
        mapped = _FIELD_BASE_TO_FLAGS.get(spec.name.lower())
        if mapped is None:
            return None
        for flag in mapped:
            for dep in _READ_FLAG_DEPS.get(flag, (flag,)):
                flags[dep] = True
    return flags

SPECIES_LABELS = {"e", "i"}
FIELD_ALIASES = {
    "jztot": "Jz-tot",
    "jz-tot": "Jz-tot",
    "jz_tot": "Jz-tot",
    "jxtot": "Jx-tot",
    "jx-tot": "Jx-tot",
    "jx_tot": "Jx-tot",
    "jytot": "Jy-tot",
    "jy-tot": "Jy-tot",
    "jy_tot": "Jy-tot",
    "pid": "PiD",
    "epx": "EPx",
    "epy": "EPy",
    "epz": "EPz",
    "betapar-betaperp": "beta_par - beta_perp",
    "beta_par-beta_perp": "beta_par - beta_perp",
    "beta_par_minus_beta_perp": "beta_par - beta_perp",
}


@dataclass(frozen=True)
class FieldSpec:
    """A field request plus an optional species selector."""

    name: str
    species: str | None = None

    @property
    def label(self) -> str:
        return self.name if self.species is None else f"{self.name}_{self.species}"


def _parse_list_arg(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] in "([{" and cleaned[-1] in ")]}":
            cleaned = cleaned[1:-1]
        raw_values = cleaned.split(",")
    else:
        raw_values = list(value)

    out = []
    for item in raw_values:
        token = str(item).strip().strip("\"'")
        if token:
            out.append(token)
    return out


def normalize_field_name(field_name: str) -> str:
    """Normalize common notebook/CLI aliases without changing unknown names."""
    cleaned = field_name.strip()
    key = cleaned.lower().replace(" ", "")
    return FIELD_ALIASES.get(key, cleaned)


def parse_field_specs(fields: str | Iterable[str]) -> list[FieldSpec]:
    """Parse comma-separated field names, accepting suffixes such as ``P_e``."""
    specs = []
    for raw_field in _parse_list_arg(fields):
        normalized = normalize_field_name(raw_field)
        if "_" in normalized:
            base, maybe_species = normalized.rsplit("_", 1)
            if maybe_species in SPECIES_LABELS:
                specs.append(FieldSpec(base, maybe_species))
                continue
        specs.append(FieldSpec(normalized, None))
    return specs


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "diagnostic"


def _as_axis(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(X)
    y_arr = np.asarray(Y)
    x_axis = x_arr if x_arr.ndim == 1 else x_arr[:, 0]
    y_axis = y_arr if y_arr.ndim == 1 else y_arr[0, :]
    return np.asarray(x_axis, dtype=float), np.asarray(y_axis, dtype=float)


def discover_available_snapshots(files_path: str | Path, experiment: str) -> list[int]:
    """Return sequential snapshot indices for an experiment directory."""
    experiment_dir = Path(files_path).expanduser() / experiment
    filenames = rp._collect_experiment_filenames(str(experiment_dir))
    return list(range(len(filenames)))


def discover_menura_iterations(files_path: str | Path, experiment: str) -> list[int]:
    """Return sorted Menura iteration numbers for an experiment request."""
    run_id, run_path = _resolve_menura_request(experiment, files_path)
    run_root = _resolve_menura_run_root(run_id, run_path)
    files = sorted((run_root / "products").glob("B_it*_rank_0_0.npy"))
    iterations = []
    for file_path in files:
        match = re.search(r"B_it(\d+)_rank_0_0\.npy$", file_path.name)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def select_snapshot_indices(available_indices: list[int], choose_times: str | None) -> list[int] | None:
    """Select snapshot indices from ``all``, an int, comma list, or slice string."""
    if choose_times is None or choose_times.lower() in {"none", "all"}:
        return None

    if ":" in choose_times:
        parts = [part.strip() for part in choose_times.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid choose_times slice {choose_times!r}; expected start:end[:step]")
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if step == 0:
            raise ValueError("choose_times slice step must not be zero")
        selected = available_indices[slice(start, end, step)]
        if not selected:
            raise ValueError(f"choose_times={choose_times!r} selected no snapshots")
        return selected

    if "," in choose_times:
        selected = [int(token.strip()) for token in choose_times.split(",") if token.strip()]
    else:
        selected = [int(choose_times)]

    n = len(available_indices)
    if n == 0:
        raise ValueError(
            f"choose_times={choose_times!r} requested but no snapshots are available "
            f"(0 found). Check the data path and run name."
        )
    bad = [idx for idx in selected if idx < 0 or idx >= n]
    if bad:
        raise ValueError(f"Snapshot indices {bad} out of range 0..{n - 1}")
    return selected


def _resolve_files_path(files_path: str | Path | None) -> str:
    if files_path is not None:
        return str(Path(files_path).expanduser())
    try:
        return load_paths().get("data_dir", "./data")
    except Exception:
        return "./data"


def _resolve_menura_request(experiment: str, files_path: str | Path) -> tuple[str, Path]:
    """Resolve a Menura experiment label to ``(run_id, parent_path)``.

    Examples
    --------
    ``experiment='iso_GEM'``, ``files_path='/.../R0'`` -> run_id ``iso_GEM``,
    parent path ``/.../R0``.

    ``experiment='R0/iso_GEM'``, ``files_path='/.../nathan5-12'`` -> run_id
    ``iso_GEM``, parent path ``/.../nathan5-12/R0``.
    """
    root = Path(files_path).expanduser()
    exp_path = Path(experiment)
    if exp_path.is_absolute():
        return exp_path.name, exp_path.parent

    candidate = root / exp_path
    if len(exp_path.parts) >= 2:
        return exp_path.parts[-1], root.joinpath(*exp_path.parts[:-1])
    if candidate.exists() and (candidate / "products").exists():
        return candidate.name, candidate.parent
    return experiment, root


def _resolve_menura_run_root(run_id: str, run_path: Path) -> Path:
    run_root = run_path / f"run_{run_id}"
    if run_root.exists():
        return run_root
    run_root = run_path / run_id
    if run_root.exists():
        return run_root
    return run_path


def _is_menura_run_dir(path: Path) -> bool:
    """Return True when ``path`` holds a Menura run (``products/B_it*`` files)."""
    products = path / "products"
    if not products.is_dir():
        return False
    return next(products.glob("B_it*_rank_0_0.npy"), None) is not None


def _menura_label_dir(run_dir: Path) -> Path:
    """Map a physical run folder back to its experiment-request folder.

    ``_resolve_menura_run_root`` wraps a request ``<id>`` as ``run_<id>``; undo
    that so the experiment label points at the folder the loader expects.
    """
    name = run_dir.name
    if name.startswith("run_"):
        return run_dir.parent / name[len("run_") :]
    return run_dir


def discover_menura_runs(files_path: str | Path, experiment: str) -> list[str]:
    """Recursively discover Menura runs at or beneath an experiment request.

    A folder is treated as a Menura run when it contains
    ``products/B_it*_rank_0_0.npy`` files. When ``experiment`` already resolves
    to a single run it is returned unchanged; otherwise every run found beneath
    it is returned as an experiment label relative to ``files_path`` (so it can
    be passed straight back to :func:`load_menura_data`).
    """
    root = Path(files_path).expanduser()
    exp_path = Path(experiment)
    base = exp_path if exp_path.is_absolute() else root / exp_path

    if not base.is_dir():
        # Let the normal loader produce a meaningful error downstream.
        return [experiment]

    for candidate in (base, base / f"run_{base.name}", base / base.name):
        if _is_menura_run_dir(candidate):
            return [experiment]

    labels: list[str] = []
    for products in base.rglob("products"):
        run_dir = products.parent
        if not _is_menura_run_dir(run_dir):
            continue
        label_dir = _menura_label_dir(run_dir)
        try:
            labels.append(str(label_dir.relative_to(root)))
        except ValueError:
            labels.append(str(label_dir))

    return sorted(dict.fromkeys(labels)) or [experiment]


def _menura_analysis_dir(files_path: Path, override: str | Path | None) -> Path | None:
    if override is not None:
        return Path(override).expanduser()
    paths = load_paths()
    if "menura_analysis_dir" in paths:
        return Path(paths["menura_analysis_dir"]).expanduser()
    for parent in [files_path, *files_path.parents]:
        candidate = parent / "analysis"
        if (candidate / "read_menura.py").exists():
            return candidate
    return None


def _import_read_menura(files_path: Path, analysis_dir: str | Path | None = None):
    resolved_analysis_dir = _menura_analysis_dir(files_path, analysis_dir)
    if resolved_analysis_dir is not None:
        analysis_path = str(resolved_analysis_dir)
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
    try:
        return importlib.import_module("read_menura")
    except ImportError as exc:
        raise ImportError(
            "Could not import read_menura. Set --menura-analysis-dir or "
            "menura_analysis_dir in paths.yaml, or add Menura analysis to PYTHONPATH."
        ) from exc


def _normalization_sample_values(data: dict, nb_factor: float = 1.0) -> tuple[float, float]:
    try:
        b0x = -float(np.asarray(data["Bx"])[0, 0, 0])
        nb = nb_factor * float(np.nanmax(np.asarray(data["rho"]["i"])[..., 0]))
    except Exception as exc:
        raise ValueError("alfven-sample normalization requires Bx and rho_i fields") from exc
    return b0x, nb


def apply_normalization(
    data: dict,
    X: np.ndarray,
    Y: np.ndarray,
    times: Iterable[float],
    *,
    normalization: str = "none",
    experiment_path: str | Path | None = None,
    b0x: float | None = None,
    nb: float | None = None,
    nb_factor: float = 1.0,
    normalize_density: bool = True,
) -> tuple[np.ndarray, np.ndarray, list | np.ndarray]:
    """Apply optional field/axis/time normalization in-place.

    ``normalize_density=False`` keeps the density field in code units while still
    casting every other field/axis/time into Alfven units.
    """
    if normalization == "none":
        return X, Y, times
    if normalization == "alfven-infer":
        return plasma.code2alfven(
            data,
            x=X,
            y=Y,
            times=list(times),
            b0x=b0x,
            nb=nb,
            experiment=str(experiment_path) if experiment_path is not None else None,
            normalize_density=normalize_density,
        )
    if normalization == "alfven-sample":
        if b0x is None or nb is None:
            sample_b0x, sample_nb = _normalization_sample_values(data, nb_factor=nb_factor)
            b0x = sample_b0x if b0x is None else b0x
            nb = sample_nb if nb is None else nb
        return plasma.code2alfven(data, x=X, y=Y, times=list(times), b0x=b0x, nb=nb, normalize_density=normalize_density)
    if normalization == "alfven-explicit":
        if b0x is None or nb is None:
            raise ValueError("alfven-explicit normalization requires both --b0x and --nb")
        return plasma.code2alfven(data, x=X, y=Y, times=list(times), b0x=b0x, nb=nb, normalize_density=normalize_density)
    raise ValueError(f"Unknown normalization mode: {normalization!r}")


def _compute_current_totals(data: dict) -> None:
    for component in ("x", "y", "z"):
        key = f"J{component}"
        if key in data and isinstance(data[key], dict) and "e" in data[key] and "i" in data[key]:
            data[f"J{component}-tot"] = data[key]["e"] + data[key]["i"]


def _has_species_components(data: dict, field_names: Iterable[str], species: str = "e") -> bool:
    return all(
        name in data and isinstance(data[name], dict) and species in data[name]
        for name in field_names
    )


def compute_common_diagnostics(data: dict, X: np.ndarray, Y: np.ndarray, qom: list) -> None:
    """Compute the notebook-style derived fields when their inputs exist."""
    x_axis, y_axis = _as_axis(X, Y)
    _compute_current_totals(data)

    if all(field in data for field in ("Bx", "By", "Bz")):
        data["Bmagn"] = np.sqrt(data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)
        if "Az" not in data:
            try:
                plasma.get_Az(x_axis, y_axis, data)
            except Exception:
                pass

    if all(field in data for field in ("Bx", "By", "Bz", "Ex", "Ey", "Ez", "rho", "Jx", "Jy", "Jz")):
        try:
            plasma.get_Ohm(data, qom, x_axis, y_axis)
        except Exception:
            pass

    if _has_species_components(data, ("rho", "Jx", "Jy", "Jz", "Vx", "Vy", "Vz", "Pxx", "Pxy", "Pxz", "Pyy", "Pyz", "Pzz")):
        try:
            plasma.get_PS_2D_field(data, x_axis, y_axis)
        except Exception:
            pass
        try:
            plasma.get_agyrotropy(data)
        except Exception:
            pass
        try:
            plasma.get_J_perp(data, x_axis, y_axis)
        except Exception:
            pass

    if "Bmagn" in data and "Ppar" in data and "Pperp" in data:
        if isinstance(data["Ppar"], dict) and isinstance(data["Pperp"], dict) and "e" in data["Ppar"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                data["Ppar/Pperp"] = data["Ppar"]["e"] / data["Pperp"]["e"]
                data["(Ppar - Pperp)/B^2"] = (data["Ppar"]["e"] - data["Pperp"]["e"]) / data["Bmagn"] ** 2
                data["beta_par"] = {
                    spec: 8 * np.pi * data["Ppar"][spec] / data["Bmagn"] ** 2
                    for spec in data["Ppar"]
                    if spec in data["Pperp"]
                }
                data["beta_perp"] = {
                    spec: 8 * np.pi * data["Pperp"][spec] / data["Bmagn"] ** 2
                    for spec in data["Pperp"]
                }
                if "e" in data["beta_par"] and "e" in data["beta_perp"]:
                    data["beta_par - beta_perp"] = data["beta_par"]["e"] - data["beta_perp"]["e"]

    if _has_species_components(data, ("Jx", "Jy")):
        data["Jinplane_e"] = np.sqrt(data["Jx"]["e"] ** 2 + data["Jy"]["e"] ** 2)

    for component in ("x", "y", "z"):
        e_key = f"E{component}"
        hall_key = f"EHall{component}"
        mhd_key = f"EMHD{component}"
        if e_key in data and hall_key in data and mhd_key in data:
            data[f"E{component}-EHall{component}-EMHD{component}"] = data[e_key] - data[hall_key] - data[mhd_key]


def load_experiment_data(
    experiment: str,
    files_path: str | Path | None = None,
    *,
    backend: str = "ecsim",
    choose_times: str | list[int] | None = None,
    choose_species: list[str] | None = None,
    choose_x: tuple[int, int] | None = None,
    choose_y: tuple[int, int] | None = None,
    processed: bool = False,
    alfven_units: bool = False,
    normalization: str = "none",
    b0x: float | None = None,
    nb: float | None = None,
    nb_factor: float = 1.0,
    normalize_density: bool = True,
    menura_analysis_dir: str | Path | None = None,
    menura_scale_ranges: bool = False,
    menura_base_nx: int = 512,
    fields_to_read: dict | None = None,
    request_fields: "str | Iterable[str] | None" = None,
    verbose: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray, list, list]:
    """Load one experiment and optionally add normalization/diagnostics.

    When ``fields_to_read`` is not given but ``request_fields`` is, only the read
    groups needed by those fields are loaded (falling back to the full default
    set if any requested field is a derived/processed quantity). This avoids
    reading — and failing on — fields the caller never asked for (e.g. ``divB``).
    """
    if alfven_units and normalization == "none":
        normalization = "alfven-infer"
    if backend == "auto":
        try:
            resolved_files_path = _resolve_files_path(files_path)
            discover_available_snapshots(resolved_files_path, experiment)
            backend = "ecsim"
        except Exception:
            backend = "menura"
    if backend == "menura":
        return load_menura_data(
            experiment,
            files_path=files_path,
            choose_times=choose_times,
            choose_x=choose_x,
            choose_y=choose_y,
            processed=processed,
            normalization=normalization,
            b0x=b0x,
            nb=nb,
            nb_factor=nb_factor,
            normalize_density=normalize_density,
            analysis_dir=menura_analysis_dir,
            scale_ranges=menura_scale_ranges,
            base_nx=menura_base_nx,
            verbose=verbose,
        )
    if backend != "ecsim":
        raise ValueError(f"Unknown diagnostics backend: {backend!r}")

    resolved_files_path = _resolve_files_path(files_path)
    selected_times = choose_times
    if isinstance(choose_times, str):
        available = discover_available_snapshots(resolved_files_path, experiment)
        selected_times = select_snapshot_indices(available, choose_times)

    if fields_to_read is not None:
        read_flags = dict(fields_to_read)
    else:
        scoped = _read_flags_for_specs(parse_field_specs(request_fields)) if request_fields else None
        read_flags = dict(scoped if scoped is not None else DEFAULT_FIELDS_TO_READ)
    choose_x_list = list(choose_x) if choose_x is not None else None
    choose_y_list = list(choose_y) if choose_y is not None else None

    for _ in range(len(read_flags) + 1):
        try:
            data, X, Y, qom, times = rp.get_exp_times(
                [experiment],
                resolved_files_path,
                read_flags,
                choose_species=choose_species or ["e", "i"],
                choose_times=selected_times,
                choose_x=choose_x_list,
                choose_y=choose_y_list,
                verbose=verbose,
            )
            break
        except KeyError as exc:
            match = re.search(r"'([^']+) is not a file in the archive'", str(exc))
            missing_field = match.group(1) if match else None
            if missing_field in read_flags and read_flags[missing_field]:
                read_flags[missing_field] = False
                if verbose:
                    print(f"Field {missing_field!r} missing; disabling it and retrying.")
                continue
            raise
    else:
        raise RuntimeError("Failed to load data after disabling missing fields")

    run_data = data[experiment]

    X, Y, times = apply_normalization(
        run_data,
        X,
        Y,
        times,
        normalization=normalization,
        experiment_path=Path(resolved_files_path) / experiment,
        b0x=b0x,
        nb=nb,
        nb_factor=nb_factor,
        normalize_density=normalize_density,
    )
    # AFTER apply_normalization: the totals are derived keys the normalization
    # pass does not know about, so computing them first left J*-tot in code
    # units (e.g. 1/b0x ~ 40x too small under alfven-infer).
    _compute_current_totals(run_data)

    run_data["X"] = X
    run_data["Y"] = Y
    run_data["qom"] = qom
    run_data["times"] = times

    if processed:
        compute_common_diagnostics(run_data, X, Y, qom)

    return run_data, X, Y, qom, times


def _menura_scale(run_id: str, run_path: Path, base_nx: int = 512) -> tuple[int, int]:
    root = _resolve_menura_run_root(run_id, run_path)
    files = sorted(root.glob("products/B_it*_rank_0_0.npy"))
    if not files:
        raise FileNotFoundError(f"No B field files under {root}/products")
    nx = np.load(files[0], mmap_mode="r").shape[1] - 4
    return max(int(round(nx / base_nx)), 1), nx


def _scale_range(value: tuple[int, int] | None, scale: int) -> tuple[int, int] | None:
    if value is None:
        return None
    return int(value[0] * scale), int(value[1] * scale)


def _select_menura_iterations(available_iterations: list[int], choose_times: str | list[int] | None) -> list[int] | None:
    if choose_times is None:
        return None
    if isinstance(choose_times, list):
        return [available_iterations[index] for index in choose_times]
    selected_indices = select_snapshot_indices(list(range(len(available_iterations))), choose_times)
    if selected_indices is None:
        return None
    return [available_iterations[index] for index in selected_indices]


def load_menura_data(
    experiment: str,
    *,
    files_path: str | Path | None = None,
    choose_times: str | list[int] | None = None,
    choose_x: tuple[int, int] | None = None,
    choose_y: tuple[int, int] | None = None,
    processed: bool = False,
    normalization: str = "none",
    b0x: float | None = None,
    nb: float | None = None,
    nb_factor: float = 1.0,
    normalize_density: bool = True,
    analysis_dir: str | Path | None = None,
    scale_ranges: bool = False,
    base_nx: int = 512,
    verbose: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray, list, list]:
    """Load a Menura run using ``read_menura`` and return diagnostics data."""
    resolved_files_path = Path(_resolve_files_path(files_path)).expanduser()
    run_id, run_path = _resolve_menura_request(experiment, resolved_files_path)
    read_menura = _import_read_menura(resolved_files_path, analysis_dir)
    iterations = discover_menura_iterations(resolved_files_path, experiment)
    if not iterations:
        run_root = _resolve_menura_run_root(run_id, run_path)
        raise FileNotFoundError(
            f"No Menura snapshots found for experiment {experiment!r}: "
            f"no 'B_it*_rank_0_0.npy' files under {run_root / 'products'}. "
            f"Check --files-path ({resolved_files_path}) and the run name "
            f"(resolved run_id={run_id!r}, run_path={run_path})."
        )
    selected_iterations = _select_menura_iterations(iterations, choose_times)
    scale = 1
    if scale_ranges:
        scale, _ = _menura_scale(run_id, run_path, base_nx=base_nx)
    choose_x_eff = _scale_range(choose_x, scale)
    choose_y_eff = _scale_range(choose_y, scale)

    kwargs = {
        "iters2": selected_iterations,
        "path": str(run_path),
    }
    if choose_x_eff is not None:
        kwargs["ix_min"], kwargs["ix_max"] = choose_x_eff
    if choose_y_eff is not None:
        kwargs["iy_min"], kwargs["iy_max"] = choose_y_eff

    stdout_cm = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with stdout_cm:
        data, _md, _times_all, _iters, times, _beta0, _poly_ind, _x, _y, X, Y = read_menura.read_menura(
            run_id,
            **kwargs,
        )

    qom = [-np.inf, 1.0]
    X, Y, times = apply_normalization(
        data,
        X,
        Y,
        times,
        normalization=normalization,
        experiment_path=run_path / run_id,
        b0x=b0x,
        nb=nb,
        nb_factor=nb_factor,
        normalize_density=normalize_density,
    )
    data["X"] = X
    data["Y"] = Y
    data["qom"] = qom
    data["times"] = times
    data["timeunit"] = r" $\omega_{pi}^{-1}$"
    # AFTER apply_normalization, for the same reason as in
    # load_experiment_data: J*-tot are derived keys the normalization pass
    # does not rescale.
    _compute_current_totals(data)

    if processed:
        compute_common_diagnostics(data, X, Y, qom)
    return data, X, Y, qom, times


def resolve_field_data(data: dict, spec: FieldSpec) -> tuple[np.ndarray, str | None]:
    """Resolve a field specification to an array and concrete species."""
    if spec.name not in data:
        available = ", ".join(sorted(str(key) for key in data.keys()))
        raise KeyError(f"Field {spec.name!r} not found. Available fields: {available}")

    field_obj = data[spec.name]
    if isinstance(field_obj, dict):
        available_species = list(field_obj.keys())
        species = spec.species
        if species is None:
            if len(available_species) == 1:
                species = available_species[0]
            elif "e" in field_obj:
                species = "e"
            else:
                raise KeyError(f"Field {spec.name!r} has species {available_species}; specify a suffix such as _e")
        if species not in field_obj:
            raise KeyError(f"Species {species!r} not available for {spec.name!r}. Choices: {available_species}")
        return np.asarray(field_obj[species]), species

    return np.asarray(field_obj), None


def get_cmap(field_name: str, cmap: str = "auto") -> str:
    """Return a notebook-like default colormap for a field."""
    if cmap != "auto":
        return cmap
    positive_default_fields = {
        "rho",
        "Pxx",
        "Pyy",
        "Pzz",
        "Ppar/Pperp",
        "Bmagn",
        "agyrotropy",
        "beta_par",
        "beta_perp",
    }
    return "viridis" if field_name in positive_default_fields else "seismic"


def _field_limits(values: np.ndarray, robust_quantile: float = 0.995) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    has_pos = np.any(finite > 0)
    has_neg = np.any(finite < 0)
    if has_pos and has_neg:
        vmax = float(np.quantile(np.abs(finite), robust_quantile))
        vmax = max(vmax, 1e-12)
        return -vmax, vmax
    if np.nanmax(finite) <= 0:
        vmax = float(np.quantile(np.abs(finite), robust_quantile))
        return 0.0, max(vmax, 1e-12)
    return 0.0, max(float(np.quantile(finite, robust_quantile)), 1e-12)


def _panel_values(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmax(finite) <= 0:
        return -values
    return values


def _panel_grid_figsize(X: np.ndarray, Y: np.ndarray, nrows: int, ncols: int) -> tuple[float, float]:
    x_axis, y_axis = _as_axis(X, Y)
    x_span = float(np.nanmax(x_axis) - np.nanmin(x_axis))
    y_span = float(np.nanmax(y_axis) - np.nanmin(y_axis))
    panel_width = 4.4
    if x_span > 0 and y_span > 0:
        panel_height = panel_width * y_span / x_span
    else:
        panel_height = 3.0
    panel_height = min(max(panel_height, 2.1), 3.8)
    return ((panel_width + 0.55) * ncols, (panel_height + 0.7) * nrows)


def _draw_panel(
    fig,
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    values: np.ndarray,
    *,
    cmap_name: str,
    vmin: float,
    vmax: float,
    title: str,
):
    """Draw one labelled, colorbar-decorated field panel and return its mesh."""
    im = ax.pcolormesh(X, Y, values, vmin=vmin, vmax=vmax, cmap=cmap_name)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im, cax=cax)
    return im


def plot_field_panels(
    data: dict,
    X: np.ndarray,
    Y: np.ndarray,
    field_specs: list[FieldSpec],
    *,
    run_name: str,
    time_index: int = 0,
    time_value: float | None = None,
    output: str | Path,
    ncols: int | None = None,
    cmap: str = "auto",
    figsize: tuple[float, float] | None = None,
    robust_quantile: float = 0.995,
) -> Path:
    """Save a grid of requested fields, similar to notebook ``plot_requested_fields``."""
    if not field_specs:
        raise ValueError("At least one field is required")
    ncols_eff = ncols or int(np.ceil(np.sqrt(len(field_specs))))
    nrows = int(np.ceil(len(field_specs) / ncols_eff))
    figsize_eff = figsize or _panel_grid_figsize(X, Y, nrows, ncols_eff)
    fig, axes = plt.subplots(nrows, ncols_eff, figsize=figsize_eff, squeeze=False)

    for index, spec in enumerate(field_specs):
        ax = axes.ravel()[index]
        field_data, species = resolve_field_data(data, spec)
        if field_data.ndim == 3:
            values = field_data[..., time_index]
        elif field_data.ndim == 2:
            values = field_data
        else:
            raise ValueError(f"Field {spec.label!r} must be 2D or 3D, got shape {field_data.shape}")
        values_to_plot = _panel_values(values)
        vmin, vmax = _field_limits(values_to_plot, robust_quantile=robust_quantile)
        label = spec.name if species is None else f"{spec.name}_{species}"
        title = f"{run_name}: {label}"
        if time_value is not None:
            title += f", t={time_value:.4g}"
        _draw_panel(
            fig,
            ax,
            X,
            Y,
            values_to_plot,
            cmap_name=get_cmap(spec.name, cmap),
            vmin=vmin,
            vmax=vmax,
            title=title,
        )

    for ax in axes.ravel()[len(field_specs) :]:
        ax.axis("off")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45, w_pad=0.85, h_pad=1.0)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


_TITLE_BAND_INCHES = 0.4


def _frame_slice(field_data: np.ndarray, time_index: int) -> np.ndarray:
    """One 2D frame of a field, tolerating static (2D) fields."""
    return field_data[..., time_index] if field_data.ndim == 3 else field_data


def _panel_time_limits(
    field_data: np.ndarray, time_indices: list[int], robust_quantile: float
) -> tuple[float, float, float]:
    """Sign flip and color limits for one panel, fixed over every plotted frame.

    Computing the limits once over the whole animated window is what keeps the
    colorbar — and therefore the perceived amplitude — from flickering between
    frames, unlike :func:`plot_field_panels`, which only ever sees one snapshot.
    """
    if field_data.ndim == 3 and time_indices != list(range(field_data.shape[2])):
        stack = np.take(field_data, time_indices, axis=2)
    else:
        stack = field_data
    flipped = _panel_values(stack)
    sign = 1.0 if flipped is stack else -1.0
    vmin, vmax = _field_limits(flipped, robust_quantile=robust_quantile)
    return sign, vmin, vmax


def _resolve_time_indices(time_indices: Iterable[int] | None, n_time: int) -> list[int]:
    if time_indices is None:
        return list(range(n_time))
    resolved = []
    for raw_index in time_indices:
        index = int(raw_index)
        if index < 0:
            index += n_time
        if index < 0 or index >= n_time:
            raise IndexError(f"time index {raw_index} out of range 0..{n_time - 1}")
        resolved.append(index)
    if not resolved:
        raise ValueError("At least one time index is required")
    return resolved


def _movie_writer(output: Path) -> str:
    return "ffmpeg" if output.suffix.lower() in {".mp4", ".m4v", ".mov", ".avi", ".webm"} else "pillow"


def animate_field_panels(
    data: dict,
    X: np.ndarray,
    Y: np.ndarray,
    field_specs: list[FieldSpec],
    *,
    run_name: str,
    times: Iterable[float] | None = None,
    time_indices: Iterable[int] | None = None,
    output: str | Path,
    ncols: int | None = None,
    cmap: str = "auto",
    figsize: tuple[float, float] | None = None,
    robust_quantile: float = 0.995,
    fps: int = 5,
    dpi: int = 150,
    frames_dir: str | Path | None = None,
    writer: str | None = None,
) -> Path:
    """Animate the :func:`plot_field_panels` grid over time into a GIF or MP4.

    The same panel layout, colormaps and sign convention as the still figure are
    reused, with two differences: color limits are computed once over all plotted
    frames so the panels do not flicker, and the run name plus running time move
    to the figure title so each panel keeps only its field label. Fields loaded as
    a single 2D snapshot stay static. The writer follows the ``output`` suffix
    (``.gif`` → pillow, ``.mp4`` → ffmpeg) unless ``writer`` says otherwise. When
    ``frames_dir`` is given, every frame is also written there as
    ``frame_<time index>.png``.
    """
    if not field_specs:
        raise ValueError("At least one field is required")
    times_arr = np.asarray(list(times) if times is not None else [], dtype=float)

    resolved: list[tuple[FieldSpec, str | None, np.ndarray]] = []
    n_time: int | None = None
    for spec in field_specs:
        field_data, species = resolve_field_data(data, spec)
        if field_data.ndim not in (2, 3):
            raise ValueError(f"Field {spec.label!r} must be 2D or 3D, got shape {field_data.shape}")
        if field_data.ndim == 3:
            n_time = field_data.shape[2] if n_time is None else min(n_time, field_data.shape[2])
        resolved.append((spec, species, field_data))
    if n_time is None:
        n_time = max(int(times_arr.size), 1)
    indices = _resolve_time_indices(time_indices, n_time)

    panels = []
    for spec, species, field_data in resolved:
        sign, vmin, vmax = _panel_time_limits(field_data, indices, robust_quantile)
        panels.append((spec, species, field_data, sign, vmin, vmax))

    def frame_title(time_index: int) -> str:
        if time_index < times_arr.size:
            unit = data.get("timeunit", "") if isinstance(data.get("timeunit"), str) else ""
            return f"{run_name}, t={times_arr[time_index]:.4g}{unit}"
        return f"{run_name}, frame {time_index}"

    ncols_eff = ncols or int(np.ceil(np.sqrt(len(panels))))
    nrows = int(np.ceil(len(panels) / ncols_eff))
    if figsize is None:
        panel_width, panel_height = _panel_grid_figsize(X, Y, nrows, ncols_eff)
        figsize_eff = (panel_width, panel_height + _TITLE_BAND_INCHES)
    else:
        figsize_eff = figsize
    fig, axes = plt.subplots(nrows, ncols_eff, figsize=figsize_eff, squeeze=False)

    meshes = []
    for index, (spec, species, field_data, sign, vmin, vmax) in enumerate(panels):
        label = spec.name if species is None else f"{spec.name}_{species}"
        meshes.append(
            _draw_panel(
                fig,
                axes.ravel()[index],
                X,
                Y,
                sign * _frame_slice(field_data, indices[0]),
                cmap_name=get_cmap(spec.name, cmap),
                vmin=vmin,
                vmax=vmax,
                # Run name and time live in the figure title, so the panels only
                # need their field label (run names here are long paths).
                title=label,
            )
        )

    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")

    suptitle = fig.suptitle(frame_title(indices[0]))
    # The figure title carries the running time, so it gets a band of its own
    # instead of being laid over the panel titles. The frames of an animation all
    # share one canvas size, so bbox_inches="tight" is not available here.
    fig.tight_layout(pad=0.45, w_pad=0.85, h_pad=1.0, rect=(0.0, 0.0, 1.0, 1.0 - _TITLE_BAND_INCHES / figsize_eff[1]))

    def update(frame: int):
        time_index = indices[frame]
        for mesh, (_, _, field_data, sign, _, _) in zip(meshes, panels, strict=True):
            mesh.set_array(sign * _frame_slice(field_data, time_index))
        suptitle.set_text(frame_title(time_index))
        return [*meshes, suptitle]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # blit=False because the animated suptitle lives outside the panel axes and
    # would otherwise never be redrawn.
    anim = animation.FuncAnimation(fig, update, frames=len(indices), blit=False)
    anim.save(out_path, dpi=dpi, fps=fps, writer=writer or _movie_writer(out_path))

    if frames_dir is not None:
        frames_path = Path(frames_dir)
        frames_path.mkdir(parents=True, exist_ok=True)
        for frame, time_index in enumerate(indices):
            update(frame)
            fig.savefig(frames_path / f"frame_{time_index:04d}.png", dpi=dpi)

    plt.close(fig)
    return out_path


def _resolve_cut_index(axis: np.ndarray, *, cut_index: int | None, cut_value: float | None) -> int:
    if cut_index is not None and cut_value is not None:
        raise ValueError("Specify only one of cut_index or cut_value")
    if cut_index is None and cut_value is None:
        return len(axis) // 2
    if cut_index is not None:
        idx = int(cut_index)
        if idx < 0:
            idx += len(axis)
        if idx < 0 or idx >= len(axis):
            raise IndexError(f"cut_index {cut_index} out of range 0..{len(axis) - 1}")
        return idx
    return int(np.argmin(np.abs(axis - float(cut_value))))


def build_profiles_dataframe(
    data: dict,
    X: np.ndarray,
    Y: np.ndarray,
    field_specs: list[FieldSpec],
    *,
    run_name: str,
    times: Iterable[float] | None = None,
    time_indices: Iterable[int] | None = None,
    projection: str = "y",
    cut_index: int | None = None,
    cut_value: float | None = None,
) -> pd.DataFrame:
    """Build a long-format CSV-ready dataframe of 1D field cuts."""
    if projection not in {"x", "y"}:
        raise ValueError("projection must be 'x' or 'y'")
    x_axis, y_axis = _as_axis(X, Y)
    times_arr = np.asarray(list(times) if times is not None else [])
    indices = list(time_indices) if time_indices is not None else [0]

    rows = []
    for spec in field_specs:
        field_data, species = resolve_field_data(data, spec)
        for time_index in indices:
            if field_data.ndim == 3:
                values_2d = field_data[..., time_index]
            elif field_data.ndim == 2:
                values_2d = field_data
            else:
                raise ValueError(f"Field {spec.label!r} must be 2D or 3D, got shape {field_data.shape}")

            if projection == "y":
                cut_axis = "x"
                idx = _resolve_cut_index(x_axis, cut_index=cut_index, cut_value=cut_value)
                coord = y_axis
                values = values_2d[idx, :]
                resolved_cut_value = x_axis[idx]
            else:
                cut_axis = "y"
                idx = _resolve_cut_index(y_axis, cut_index=cut_index, cut_value=cut_value)
                coord = x_axis
                values = values_2d[:, idx]
                resolved_cut_value = y_axis[idx]

            time_value = float(times_arr[time_index]) if times_arr.size else np.nan
            field_label = spec.name if species is None else f"{spec.name}_{species}"
            for coordinate, value in zip(coord, values):
                rows.append(
                    {
                        "diagnostic": "profile",
                        "run": run_name,
                        "field": spec.name,
                        "species": species or "",
                        "field_label": field_label,
                        "time_index": int(time_index),
                        "time": time_value,
                        "projection": projection,
                        "cut_axis": cut_axis,
                        "cut_index": int(idx),
                        "cut_value": float(resolved_cut_value),
                        "coord": float(coordinate),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _add_notebook_recon_normalization(frame: pd.DataFrame, data: dict) -> None:
    """Add notebook-style normalized reconnection columns in-place.

    Reproduces the cell-6 plotting transform from ``fullres.ipynb``::

        time_norm      = time * |Bx[0, 0, 0]|
        recon_rate_norm = -recon_rate * sqrt(-rho_e[0, 0, 0] * 4*pi) / Bx[0, 0, 0]**2

    The sample values ``Bx[0, 0, 0]`` and ``rho_e[0, 0, 0]`` are read from the
    in-memory ``data`` *after* any normalization has been applied, exactly as
    the notebook references ``d['Bx'][0, 0, 0]`` / ``d['rho']['e'][0, 0, 0]`` at
    plot time. The sign flip makes the growth-phase rate positive (so a log axis
    no longer dives at sign changes) and the ``sqrt(4*pi*rho)/B**2`` factor casts
    the rate into normalized Alfven units.
    """
    try:
        b0x_sample = float(np.asarray(data["Bx"])[0, 0, 0])
        rho_e0 = float(np.asarray(data["rho"]["e"])[0, 0, 0])
    except Exception as exc:
        raise ValueError(
            "recon_normalization='notebook' requires Bx and rho_e fields"
        ) from exc
    if b0x_sample == 0.0:
        raise ValueError("recon_normalization='notebook' requires Bx[0,0,0] != 0")
    scale = np.sqrt(-rho_e0 * 4.0 * np.pi) / b0x_sample**2
    frame["time_norm"] = frame["time"] * abs(b0x_sample)
    frame["recon_rate_norm"] = -frame["recon_rate"] * scale


def export_reconnection_dataframe(
    data: dict,
    X: np.ndarray,
    Y: np.ndarray,
    times: Iterable[float],
    *,
    run_name: str,
    qom: list | None = None,
    az_filter: dict | None = None,
    grad_tol: float = 1e-8,
    merge_tol: float = 1e-3,
    seed_grad_frac: float | None = None,
    recon_normalization: str = "none",
    n_workers: int = 1,
) -> pd.DataFrame:
    """Track X/O points and return reconnection-rate diagnostics as a dataframe.

    ``recon_normalization`` controls extra normalized columns:

    * ``"none"`` (default) — only the raw ``recon_rate`` / ``time`` from
      :func:`plasma.track_xo_points`.
    * ``"notebook"`` — additionally emit ``time_norm`` and ``recon_rate_norm``
      matching ``fullres.ipynb`` (see :func:`_add_notebook_recon_normalization`).
    """
    if recon_normalization not in {"none", "notebook"}:
        raise ValueError(f"Unknown recon_normalization mode: {recon_normalization!r}")
    if "Az" not in data:
        x_axis, y_axis = _as_axis(X, Y)
        plasma.get_Az(x_axis, y_axis, data)
    if qom is not None:
        _compute_current_totals(data)
    result = plasma.track_xo_points(
        data,
        X,
        Y,
        np.asarray(list(times), dtype=float),
        az_filter=az_filter,
        grad_tol=grad_tol,
        merge_tol=merge_tol,
        seed_grad_frac=seed_grad_frac,
        n_workers=n_workers,
    )
    frame = pd.DataFrame(result)
    if recon_normalization == "notebook":
        _add_notebook_recon_normalization(frame, data)
    frame.insert(0, "time_index", np.arange(len(frame), dtype=int))
    frame.insert(0, "run", run_name)
    frame.insert(0, "diagnostic", "reconnection")
    return frame


def export_bands_dataframe(
    data: dict,
    X: np.ndarray,
    Y: np.ndarray,
    times: Iterable[float],
    *,
    run_name: str,
    field: str = "E",
    f_lo: float = 0.15,
    f_hi: float = 0.80,
) -> pd.DataFrame:
    """Band-resolved spectral scalars per snapshot as a dataframe.

    Splits the omnidirectional vector power spectrum of ``field`` (``E`` or
    ``B``) into three wavenumber bands, with edges given as fractions of the
    Nyquist wavenumber so the scalars are comparable across resolutions:

    * ``recon`` — ``k <  f_lo * k_ny``: coherent large-scale (reconnection) field
    * ``wave``  — ``f_lo * k_ny <= k < f_hi * k_ny``: finite-wavelength waves
    * ``grid``  — ``k >= f_hi * k_ny``: grid-scale / checkerboard noise

    Emits one row per snapshot mirroring :func:`export_reconnection_dataframe`
    (``diagnostic``/``run``/``time_index``/``time`` first), with per-band power
    fractions and absolute powers, the wave-to-reconnection contrast ratio, and
    the spectral centroid ``kbar`` as a single combined health index.
    """
    if not 0.0 < f_lo < f_hi <= 1.0:
        raise ValueError(f"Band edges must satisfy 0 < f_lo < f_hi <= 1; got {f_lo}, {f_hi}")
    try:
        fx, fy, fz = data[f"{field}x"], data[f"{field}y"], data[f"{field}z"]
    except KeyError as exc:
        raise ValueError(f"bands diagnostic requires {field}x/{field}y/{field}z fields") from exc

    k, spec = plasma.vector_spectrum_2D(fx, fy, fz, X, Y)
    # vector_spectrum_2D bins radially by integer mode number while returning the
    # rfft ky axis, so the lengths can disagree off-square grids; trim to common.
    n = min(len(k), spec.shape[0])
    k, spec = np.asarray(k[:n], dtype=float), np.asarray(spec[:n], dtype=float)

    k_ny = k[-1]
    lo = k < f_lo * k_ny
    hi = k >= f_hi * k_ny
    mid = ~lo & ~hi

    total = spec.sum(axis=0)
    safe_total = np.where(total > 0, total, np.nan)
    recon_power = spec[lo].sum(axis=0)
    wave_power = spec[mid].sum(axis=0)
    grid_power = spec[hi].sum(axis=0)

    frame = pd.DataFrame(
        {
            "time": np.asarray(list(times), dtype=float),
            "recon_frac": recon_power / safe_total,
            "wave_frac": wave_power / safe_total,
            "grid_frac": grid_power / safe_total,
            "recon_power": recon_power,
            "wave_power": wave_power,
            "grid_power": grid_power,
            "total_power": total,
            "wave_over_recon": wave_power / np.where(recon_power > 0, recon_power, np.nan),
            "kbar": (k[:, None] * spec).sum(axis=0) / safe_total,
            "field": field,
            "k_lo": f_lo * k_ny,
            "k_hi": f_hi * k_ny,
            "k_ny": k_ny,
        }
    )
    frame.insert(0, "time_index", np.arange(len(frame), dtype=int))
    frame.insert(0, "run", run_name)
    frame.insert(0, "diagnostic", "bands")
    return frame


# RunComparePlotter aesthetics (see fullres.ipynb plotter.interactive): cycling
# colors + dash styles, with line width ramping down and alpha ramping up across
# the overlaid series so earlier series read as thick/faint, later as thin/solid.
_OVERLAY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
]
_OVERLAY_DASHES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]


def _overlay_style(
    idx: int,
    n: int,
    *,
    lw: tuple[float, float] = (5.0, 1.5),
    alpha: tuple[float, float] = (0.35, 1.0),
) -> dict:
    """Per-series line style matching RunComparePlotter.style (gradient='line')."""
    lw_max, lw_min = lw
    alpha_min, alpha_max = alpha
    frac = idx / max(n - 1, 1)
    return dict(
        color=_OVERLAY_COLORS[idx % len(_OVERLAY_COLORS)],
        linestyle=_OVERLAY_DASHES[idx % len(_OVERLAY_DASHES)],
        linewidth=lw_max - (lw_max - lw_min) * frac,
        alpha=alpha_min + (alpha_max - alpha_min) * frac,
    )


def _default_overlay_xlabel(x: str, data: pd.DataFrame) -> str:
    """Human-readable x label: the cut coordinate axis, or 'time'."""
    if x == "coord" and "projection" in data.columns:
        projections = data["projection"].dropna().unique()
        if len(projections) == 1:
            return str(projections[0])
    if x in ("time", "time_norm"):
        return "time"
    return x


def _default_overlay_ylabel(y: str, data: pd.DataFrame) -> str:
    """Human-readable y label: the field being plotted, or the rate."""
    if y == "value" and "field_label" in data.columns:
        labels = data["field_label"].dropna().unique()
        if len(labels) == 1:
            return str(labels[0])
        return "value"
    if y in ("recon_rate", "recon_rate_norm"):
        return "reconnection rate"
    return y


def _csv_ref_matches(ref: str, path: str | Path) -> bool:
    """Whether a per-CSV reference names ``path``.

    A reference matches by exact path, by basename equality (either side), by
    path suffix, or as a shell glob against the full path or basename. This lets
    ``--csv-run-pattern reconnection_menura.csv ...`` target a file given by its
    full relative path without retyping it.
    """
    ref = str(ref)
    p = str(path)
    name = Path(p).name
    if ref in (p, name) or Path(ref).name == name:
        return True
    if p.endswith(ref):
        return True
    return fnmatch.fnmatch(p, ref) or fnmatch.fnmatch(name, ref)


def _csv_source_labels(paths: list[str | Path]) -> list[str]:
    """One label per CSV path, distinguishing rows when other grouping
    columns collide across files.

    Two different CSVs can easily share a ``run`` (and ``field_label`` /
    ``projection`` / ``cut_value``) value - e.g. two batches that each ran a
    run named ``iso_GEM_1e-2_Jze.5_r0``. Without something to tell those rows
    apart, the default overlay grouping merges them into a single series and
    ``ax.plot`` connects points from both files sorted purely by x, producing
    a scrambled zig-zag line instead of two overlaid curves - silently, with
    no error. Defaults to the parent directory name, which matches this
    project's ``diagnostics/<batch>/<file>.csv`` layout (e.g. "R0", "R5").

    When that isn't unique - comparing two batches that both use ``R0``,
    ``R5``, ... subdirectories - it lengthens by one path component at a time
    ("nathan5-12_f2/R0" vs "iPiC3D-nathan/R0") rather than jumping straight to
    the absolute path: these labels go in the legend, and a dozen absolute
    paths there overflow the axes and hide the curves.
    """
    resolved = [Path(p).resolve() for p in paths]
    # parts[:-1] drops the file name, which is rarely what distinguishes two
    # exports; depth 0 is the parent directory, and the last step is the full
    # path, so a unique label is always reached.
    for depth in range(max(len(p.parts) for p in resolved)):
        labels = ["/".join(p.parts[:-1][-1 - depth :]) for p in resolved]
        if len(set(labels)) == len(labels):
            return labels
    return [str(p) for p in resolved]


def _apply_overlay_selection(
    frame: pd.DataFrame,
    select: dict[str, list[str]] | None,
    select_patterns: dict[str, list[str]] | None,
) -> pd.DataFrame:
    """Filter one CSV frame by exact-value ``select`` and glob ``select_patterns``."""
    if select:
        for col, values in select.items():
            if col not in frame.columns:
                raise KeyError(f"Cannot filter on {col!r}; available columns: {list(frame.columns)}")
            wanted = [str(v) for v in values]
            frame = frame[frame[col].astype(str).isin(wanted)]
    if select_patterns:
        for col, patterns in select_patterns.items():
            if col not in frame.columns:
                raise KeyError(f"Cannot filter on {col!r}; available columns: {list(frame.columns)}")
            col_str = frame[col].astype(str)
            mask = pd.Series(False, index=frame.index)
            for pattern in patterns:
                mask |= col_str.map(lambda value, pat=str(pattern): fnmatch.fnmatch(value, pat))
            frame = frame[mask]
    return frame


def _overlay_group_rank(
    group_cols: list[str],
    *,
    select: dict[str, list[str]] | None,
    select_patterns: dict[str, list[str]] | None,
    csv_labels: list[str],
):
    """Rank function ordering series the way the user listed them.

    ``--run a,b,c`` (like ``--field`` / ``--select``) reads as an ordering and
    not merely a filter, so the legend - and the color/dash/width gradient of
    ``_overlay_style``, which is driven by series index - follows the listed
    order instead of pandas' alphabetical groupby sort. Glob patterns rank by
    which pattern a value matched first, and ``csv_source`` falls back to the
    order the CSVs were given on the command line. Columns with no explicit
    order, and values within one rank, keep the native groupby order (the sort
    applied with this key is stable).
    """
    orders: list[tuple[dict[str, int], list[str]]] = []
    for col in group_cols:
        explicit = [str(v) for v in (select or {}).get(col, [])]
        patterns = [str(p) for p in (select_patterns or {}).get(col, [])]
        if col == "csv_source" and not explicit and not patterns:
            explicit = [str(label) for label in csv_labels]
        index = {value: i for i, value in enumerate(dict.fromkeys(explicit))}
        orders.append((index, patterns))

    def rank(group_key) -> tuple[int, ...]:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        ranks = []
        for value, (index, patterns) in zip(group_key, orders):
            key = str(value)
            if key in index:
                ranks.append(index[key])
                continue
            matched = next(
                (i for i, pat in enumerate(patterns) if fnmatch.fnmatch(key, pat)),
                len(patterns),
            )
            ranks.append(len(index) + matched)
        return tuple(ranks)

    return rank


_DERIVED_FUNCS = {
    "sqrt": np.sqrt,
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "minimum": np.minimum,
    "maximum": np.maximum,
}

_DERIVED_CONSTS = {"pi": np.pi, "e": np.e}

_DERIVED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_DERIVED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_VECTOR_COMPONENTS = ("x", "y", "z")


def _parse_derived_expression(expression: str) -> ast.Expression:
    """Parse one derived-field expression, rejecting anything non-arithmetic.

    ``^`` is rewritten to ``**`` *before* parsing rather than by binding
    ``BitXor`` to ``pow`` afterwards: as XOR it binds looser than ``/``, so
    ``B^2/(8*pi)`` would silently parse as ``B**(2/(8*pi))``. Textual
    substitution is safe here because the grammar below admits no strings.
    """
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse derived field expression {expression!r}: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _DERIVED_FUNCS:
                name = getattr(node.func, "id", "<expr>")
                raise ValueError(
                    f"Unsupported function {name!r} in {expression!r}; "
                    f"available: {sorted(_DERIVED_FUNCS)}"
                )
            if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                raise ValueError(f"Function calls in {expression!r} take positional arguments only")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _DERIVED_BINOPS:
                raise ValueError(f"Unsupported operator in {expression!r}")
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _DERIVED_UNARYOPS:
                raise ValueError(f"Unsupported unary operator in {expression!r}")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError(f"Only numeric literals are allowed in {expression!r}")
        elif not isinstance(node, (ast.Expression, ast.Name, ast.Load, *_DERIVED_BINOPS, *_DERIVED_UNARYOPS)):
            raise ValueError(f"Unsupported syntax {type(node).__name__} in {expression!r}")
    return tree


def _expression_dependencies(expression: str) -> set[str]:
    """Field labels an expression may need, including vector components.

    ``B`` is not itself a ``field_label`` in the profile CSVs, so a bare name
    also pulls in its ``x``/``y``/``z`` components - see
    ``_derived_expression_env``. Names that don't exist in the data are simply
    never matched by the selection, so over-asking here is harmless.
    """
    tree = _parse_derived_expression(expression)
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} - called
    deps = set()
    for name in names:
        if name in _DERIVED_CONSTS:
            continue
        deps.add(name)
        deps.update(f"{name}{c}" for c in _VECTOR_COMPONENTS)
    return deps


def _eval_derived_node(node: ast.AST, env: dict):
    if isinstance(node, ast.Expression):
        return _eval_derived_node(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise KeyError(
                f"Derived field references {node.id!r}, which is neither an available "
                f"field_label nor a constant; available: {sorted(k for k in env if k not in _DERIVED_CONSTS)}"
            )
        return env[node.id]
    if isinstance(node, ast.BinOp):
        return _DERIVED_BINOPS[type(node.op)](
            _eval_derived_node(node.left, env), _eval_derived_node(node.right, env)
        )
    if isinstance(node, ast.UnaryOp):
        return _DERIVED_UNARYOPS[type(node.op)](_eval_derived_node(node.operand, env))
    if isinstance(node, ast.Call):
        return _DERIVED_FUNCS[node.func.id](*(_eval_derived_node(a, env) for a in node.args))
    raise ValueError(f"Unsupported syntax {type(node).__name__} in derived field expression")


def _derived_expression_env(wide: pd.DataFrame) -> dict:
    """Names an expression can use: every field_label, plus vector magnitudes.

    A name whose components are present but which isn't itself a field_label
    (``B`` given ``Bx``/``By``) resolves to the magnitude of the components that
    *are* there. 2D GEM profile CSVs typically carry only ``Bx``/``By``, so
    ``B`` silently means the in-plane magnitude - which is why the components
    used get logged.
    """
    env = {**_DERIVED_CONSTS}
    env.update({str(col): wide[col] for col in wide.columns})
    bases = {str(col)[:-1] for col in wide.columns if str(col)[-1:] in _VECTOR_COMPONENTS}
    for base in sorted(bases):
        if not base or base in env:
            continue
        components = [f"{base}{c}" for c in _VECTOR_COMPONENTS if f"{base}{c}" in wide.columns]
        env[base] = np.sqrt(sum(wide[c] ** 2 for c in components))
        logger.info("Derived field name %r resolved as the magnitude of %s", base, ", ".join(components))
    return env


def _append_derived_fields(data: pd.DataFrame, derived: dict[str, str], *, value_col: str) -> pd.DataFrame:
    """Append rows for each ``label -> expression`` derived quantity.

    The profile CSVs are long-format (one row per field per coordinate), so the
    quantities an expression combines live on *different* rows. They're pivoted
    to one column per ``field_label`` - keyed on everything that identifies a
    sample (run, csv_source, time, cut, coord) but not on ``field``/``species``,
    which vary by construction - evaluated, and melted back so the result is
    just another ``field_label`` the rest of the overlay path treats normally.
    """
    if "field_label" not in data.columns:
        raise KeyError(f"Derived fields need a 'field_label' column; available: {list(data.columns)}")
    index_cols = [c for c in data.columns if c not in {"field", "species", "field_label", value_col}]
    if not index_cols:
        raise KeyError("Derived fields need at least one column identifying a sample (e.g. coord)")
    wide = data.pivot_table(index=index_cols, columns="field_label", values=value_col, aggfunc="mean")
    env = _derived_expression_env(wide)

    extras = []
    for label, expression in derived.items():
        values = _eval_derived_node(_parse_derived_expression(expression), env)
        frame = wide.index.to_frame(index=False)
        frame[value_col] = np.asarray(values, dtype=float) if np.ndim(values) else float(values)
        frame["field"] = label
        frame["species"] = pd.NA
        frame["field_label"] = label
        frame = frame.dropna(subset=[value_col])
        if frame.empty:
            raise ValueError(
                f"Derived field {label!r} ({expression!r}) evaluated to no finite rows; "
                "check that every field it references was exported for the selected runs"
            )
        extras.append(frame.reindex(columns=data.columns))
    return pd.concat([data, *extras], ignore_index=True)


def plot_csv_overlay(
    csv_paths: list[str | Path],
    *,
    output: str | Path,
    x: str | None = None,
    y: str | None = None,
    group_by: list[str] | None = None,
    title: str | None = None,
    dpi: int = 200,
    logx: bool = False,
    logy: bool = False,
    select: dict[str, list[str]] | None = None,
    select_patterns: dict[str, list[str]] | None = None,
    csv_select_patterns: dict[str, dict[str, list[str]]] | None = None,
    derived: dict[str, str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Path:
    """Overlay long-format profile or reconnection CSV files.

    ``select`` filters rows before plotting: a mapping of column name to the
    list of accepted (string-compared) values, e.g. ``{"field_label": ["P_e"]}``
    to overlay only the ``P_e`` profile. This mirrors a single notebook profile
    cell instead of dumping every field onto one axes. The listed values also
    set the series order: legend entries and the ``_overlay_style`` color/dash
    gradient follow the order given rather than sorting alphabetically. Series
    with no explicit order (columns absent from ``select``, values matched only
    by a pattern, extra ``csv_source`` values) keep their alphabetical order
    after the ones that were named - see ``_overlay_group_rank``.

    ``select_patterns`` filters the same way but matches each (string-compared)
    column value against shell-style glob patterns, e.g.
    ``{"run": ["Le2DHGEM_RunID_*_f2"]}`` to overlay a whole family of runs
    without listing each one. A row is kept when its value matches *any* pattern
    for that column; ``select`` and ``select_patterns`` are combined (AND across
    columns).

    ``csv_select_patterns`` scopes glob filtering to individual CSVs: a mapping
    of a CSV reference (exact path, basename, path suffix, or glob) to a
    ``{column: patterns}`` mapping applied only to rows read from the matching
    CSV(s). Per-CSV patterns override ``select_patterns`` for that column, so a
    CSV with no rule is left unfiltered (e.g. keep the ECsim reference while
    ``run='*r0*'`` filters only the Menura CSV). Every reference must match at
    least one given CSV.

    ``derived`` adds quantities the CSVs don't contain, as a mapping of new
    ``field_label`` to an arithmetic expression over existing ones, e.g.
    ``{"P_e+P_i+B^2/(8*pi)": "P_e+P_i+B^2/(8*pi)"}`` for total pressure. Names
    resolve to field labels, to a vector magnitude when only the components were
    exported (``B`` from ``Bx``/``By``/``Bz``), or to ``pi``/``e``; ``^`` means
    exponentiation and ``sqrt``/``log``/``exp``/trig calls are available. Derived
    labels behave like any other field afterwards - they filter, order, group and
    label exactly as ``select["field_label"]`` entries do, and listing them there
    is what selects them for plotting.

    ``xlabel``/``ylabel`` override the axis labels; when omitted they default to
    what is actually plotted (the cut coordinate / field name / reconnection
    rate) rather than the raw CSV column name.

    Every row is tagged with a ``csv_source`` column (the parent directory
    name of its CSV, or the full path if that's not unique across the given
    files - see ``_csv_source_labels``). It's usable like any other column in
    ``select``/``select_patterns``/``--group-by``, and is folded into the
    grouping automatically whenever it's needed - including when ``group_by``
    was given explicitly: if two CSVs share an identical group key (e.g. the
    same run name reused across two batches), grouping by that key alone
    would connect both files' points into one zig-zagging line, which is
    never a sensible result, so ``csv_source`` is added to tell them apart
    regardless of how ``group_by`` was chosen.
    """
    if not csv_paths:
        raise ValueError("At least one CSV path is required")

    # A derived field is built from field labels the user never asked to plot,
    # so the per-file field_label filter has to be widened to let its inputs
    # through; the requested labels are re-applied after the concat below, once
    # the derived rows exist to be selected.
    requested_fields = [str(f) for f in (select or {}).get("field_label", [])]
    frame_select = select
    if derived:
        needed = set(requested_fields)
        for expression in derived.values():
            needed |= _expression_dependencies(expression)
        frame_select = {**(select or {}), "field_label": sorted(needed)}

    matched_refs: set[str] = set()
    frames = []
    csv_labels = _csv_source_labels(csv_paths)
    for path, csv_label in zip(csv_paths, csv_labels):
        frame = pd.read_csv(path)
        if "csv_source" not in frame.columns:
            frame["csv_source"] = csv_label
        frame_patterns = dict(select_patterns) if select_patterns else {}
        has_csv_rule = False
        if csv_select_patterns:
            for ref, cols in csv_select_patterns.items():
                if _csv_ref_matches(ref, path):
                    matched_refs.add(ref)
                    has_csv_rule = True
                    for col, patterns in cols.items():
                        frame_patterns[col] = list(patterns)
        frame = _apply_overlay_selection(frame, frame_select, frame_patterns or None)
        if has_csv_rule and frame.empty:
            logger.warning("Per-CSV pattern for %s matched no rows in that file", path)
        frames.append(frame)

    if csv_select_patterns:
        unmatched = set(csv_select_patterns) - matched_refs
        if unmatched:
            raise ValueError(
                f"csv_select_patterns references {sorted(unmatched)!r} matched none of the "
                f"CSVs {[str(p) for p in csv_paths]!r}"
            )

    data = pd.concat(frames, ignore_index=True)

    if x is None:
        x = "time" if "recon_rate" in data.columns and "coord" not in data.columns else "coord"
    if y is None:
        y = "recon_rate" if "recon_rate" in data.columns and x == "time" else "value"
    if x not in data.columns or y not in data.columns:
        raise KeyError(f"Requested x={x!r}, y={y!r}; available columns: {list(data.columns)}")

    if derived and not data.empty:
        data = _append_derived_fields(data, derived, value_col=y)
        if requested_fields:
            data = data[data["field_label"].astype(str).isin(requested_fields)]

    if (select or select_patterns or csv_select_patterns or derived) and data.empty:
        raise ValueError(
            f"Selection (select={select!r}, patterns={select_patterns!r}, "
            f"csv_patterns={csv_select_patterns!r}) removed all rows"
        )

    group_cols = group_by or [col for col in ("run", "field_label", "projection", "cut_value") if col in data.columns]
    if not group_cols:
        data["series"] = "series"
        group_cols = ["series"]
    elif "csv_source" not in group_cols and data["csv_source"].nunique() > 1:
        # Applies even when group_by was passed explicitly: two CSVs sharing
        # a group key here means their points would be connected into one
        # zig-zagging line by ax.plot below, which is never a sensible
        # result - there's no legitimate reason to want two files' series
        # spliced together, so this isn't something to opt out of.
        collisions = data.groupby(group_cols, dropna=False)["csv_source"].nunique()
        if (collisions > 1).any():
            group_cols = group_cols + ["csv_source"]

    if xlabel is None:
        xlabel = _default_overlay_xlabel(x, data)
    if ylabel is None:
        ylabel = _default_overlay_ylabel(y, data)

    fig, ax = plt.subplots(figsize=(8, 5))
    group_rank = _overlay_group_rank(
        group_cols,
        select=select,
        select_patterns=select_patterns,
        csv_labels=csv_labels,
    )
    groups = list(data.groupby(group_cols, dropna=False))
    groups.sort(key=lambda item: group_rank(item[0]))
    for idx, (group_key, group) in enumerate(groups):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label = ", ".join(f"{col}={value}" for col, value in zip(group_cols, group_key))
        group = group.sort_values(x)
        ax.plot(group[x], group[y], label=label, **_overlay_style(idx, len(groups)))

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
