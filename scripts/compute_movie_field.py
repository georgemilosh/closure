#!/usr/bin/env python3
"""Generate animated ECsim field movies (GIF/MP4) from closure experiment data.

This script mirrors the interface of the Menura ``compute_movie_field.py`` so that
results from both codes can be compared using the same command-line options.

Usage examples
--------------
# Basic: animate Jz-tot for experiment 'run01'
    python compute_movie_field.py run01

# Specify data root:
    python compute_movie_field.py run01 --files_path /path/to/experiments

# Full kwargs example:
    python compute_movie_field.py run01 \\
        --files_path /data/experiments \\
        --fields Jz-tot \\
        --choose_times 0:50 \\
        --cmap seismic --norm linear --symmetric_limits true \\
        --fps 12 --output_format mp4

# Processed fields (PiD, EP*, agyrotropy, beta_par - beta_perp ...):
    python compute_movie_field.py run01 --fields PiD_e --processed \\
        --cmap RdBu_r --norm symlog --symmetric_limits true --linthresh 1e-4

# Agyrotropy with log scaling:
    python compute_movie_field.py run01 --fields agyrotropy_e --processed \\
        --cmap viridis --norm log --vmin 1e-3 --vmax 1

# Exact manual bounds (no symmetry):
    python compute_movie_field.py run01 --fields EPz_e --processed \\
        --cmap seismic --vmin -0.02 --vmax 0.02 --norm linear

# Smooth MP4 output:
    python compute_movie_field.py run01 --fields Jz-tot \\
        --render_mode imshow --interpolation bilinear \\
        --output_format mp4 --dpi 240 --fps 15

--choose_times format
----------------------
  'all' / 'None'  : all available time snapshots
  single int      : exact sequential index (0 = first snapshot)
  comma list      : specific sequential indices, e.g. 0,5,10,20
  slice           : Python-style index slice, e.g. 0:50 or 10:80:2
                    (negative indices and step supported)

Note: indices always refer to the sorted sequence of available HDF5/NPZ snapshots
(not physical iteration numbers).

--choose_x / --choose_y format
--------------------------------
  Passed as 'start,end'.  Example: --choose_x 10,90

--fields format
----------------
  Comma-separated field names, optionally with species suffix:
    Jz-tot         (total current, always available)
    Bz             (magnetic field component)
    Pxx_e          (electron pressure tensor component)
    agyrotropy_e   (requires --processed)
    PiD_e          (requires --processed)
    beta_par - beta_perp  (alias: betapar-betaperp, requires --processed)

Recipes
-------
1) Diverging signed field with symmetric linear limits:
        python compute_movie_field.py run01 --fields Jz-tot \\
            --cmap RdBu_r --norm linear --symmetric_limits true

2) Exact manual bounds:
        python compute_movie_field.py run01 --fields EPz_e --processed \\
            --cmap seismic --vmin -0.02 --vmax 0.02 --norm linear

3) PiD in notebook-like symlog style:
        python compute_movie_field.py run01 --fields PiD_e --processed \\
            --cmap RdBu_r --norm symlog --symmetric_limits true --linthresh 1e-4

4) Agyrotropy with log scaling:
        python compute_movie_field.py run01 --fields agyrotropy_e --processed \\
            --cmap viridis --norm log --vmin 1e-3 --vmax 1

5) Less pixelated output (smoothed + MP4):
        python compute_movie_field.py run01 --fields Jz-tot \\
            --render_mode imshow --interpolation bilinear --output_format mp4 --dpi 240 --fps 15


Tips
----
- Start with --verbose to print resolved settings and selected time indices.
- For log norm errors, either:
    - switch to --norm symlog, or
    - provide a positive-only field/range and vmin > 0.
- GIF is palette-limited (256 colors). For best visual quality use
    --output_format mp4, and optionally --render_mode imshow with bilinear interpolation.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from closure import plasma, read_pic as rp
from closure.config import load_paths


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


# ---------------------------------------------------------------------------
# Shared helpers (same interface as menura compute_movie_field.py)
# ---------------------------------------------------------------------------

def parse_list_arg(arg_str: str, dtype=str) -> list:
    cleaned = arg_str.strip()
    if len(cleaned) >= 2 and cleaned[0] in "([{" and cleaned[-1] in ")]}":
        cleaned = cleaned[1:-1]

    values = []
    for item in cleaned.split(","):
        token = item.strip().strip("\"'")
        if not token:
            continue
        values.append(dtype(token))
    return values


def parse_range_arg(arg_str: str | None) -> tuple[int, int] | None:
    if arg_str is None:
        return None
    values = parse_list_arg(arg_str, dtype=int)
    if len(values) != 2:
        raise ValueError(f"Range must be 'start,end', got: {arg_str}")
    start, end = values
    if end <= start:
        raise ValueError(f"Range end must be > start, got: {arg_str}")
    return start, end


def parse_fields_arg(fields_arg: str) -> tuple[list[str], list[str | None]]:
    fields_list: list[str] = []
    species_list: list[str | None] = []

    for raw_field in parse_list_arg(fields_arg, dtype=str):
        normalized = normalize_field_name(raw_field)
        if "_" in normalized:
            base, maybe_species = normalized.rsplit("_", 1)
            if maybe_species in SPECIES_LABELS:
                fields_list.append(base)
                species_list.append(maybe_species)
                continue
        fields_list.append(normalized)
        species_list.append(None)

    return fields_list, species_list


def normalize_field_name(field_name: str) -> str:
    cleaned = field_name.strip()
    key = cleaned.lower().replace(" ", "")
    return FIELD_ALIASES.get(key, cleaned)


def select_iterations(available_indices: list[int], choose_times: str | None) -> list[int]:
    """Select a subset of sequential snapshot indices.

    Comma lists and single values match sequential snapshot indices directly.
    The start:end[:step] format uses Python-style index slicing.
    """
    if choose_times is None or choose_times.lower() in {"none", "all"}:
        return available_indices

    n = len(available_indices)

    if ":" in choose_times:
        parts = [p.strip() for p in choose_times.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid interval format '{choose_times}'. Expected start:end[:step].")
        start = int(parts[0]) if parts[0] else None
        end   = int(parts[1]) if parts[1] else None
        step  = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if step == 0:
            raise ValueError("Interval step must not be zero")
        selected = available_indices[slice(start, end, step)]
        if not selected:
            raise ValueError(
                f"Index slice {choose_times!r} selected nothing from {n} available snapshots "
                f"(indices 0..{n - 1})."
            )
        return selected

    if "," in choose_times:
        requested = [int(x.strip()) for x in choose_times.split(",") if x.strip()]
        selected = [i for i in requested if 0 <= i < n]
        if not selected:
            raise ValueError(
                f"None of the requested indices {requested} are in range [0, {n - 1}]."
            )
        return selected

    # Single index.
    target = int(choose_times)
    if not (0 <= target < n):
        raise ValueError(
            f"Index {target} out of range. Available range: 0..{n - 1} ({n} snapshots)."
        )
    return [target]


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def resolve_norm_mode(plot_field: str, norm_arg: str) -> str:
    if norm_arg != "auto":
        return norm_arg
    if plot_field == "PiD":
        return "symlog"
    if plot_field == "agyrotropy":
        return "log"
    return "linear"


def resolve_symmetric_flag(cmap: str, norm_mode: str, finite_data: np.ndarray, symmetric_arg: str) -> bool:
    if symmetric_arg == "true":
        return True
    if symmetric_arg == "false":
        return False
    has_pos = np.any(finite_data > 0)
    has_neg = np.any(finite_data < 0)
    return cmap == "seismic" or norm_mode == "symlog" or (has_pos and has_neg)


def build_norm_and_limits(
    data_for_limits: np.ndarray,
    cmap: str,
    norm_mode: str,
    field_max: float | None,
    vmin_arg: float | None,
    vmax_arg: float | None,
    symmetric_arg: str,
    linthresh: float | None,
    linscale: float,
    log_base: float,
) -> tuple[float, float, mcolors.Normalize | None]:
    finite_data = data_for_limits[np.isfinite(data_for_limits)]
    if finite_data.size == 0:
        raise ValueError("Field has no finite values")

    data_min = float(np.nanmin(finite_data))
    data_max = float(np.nanmax(finite_data))

    vmin = vmin_arg
    vmax = vmax_arg

    if field_max is not None and vmin is None and vmax is None:
        vmax = abs(field_max)
        if cmap == "seismic":
            vmin = -vmax
        else:
            vmin = 0.0

    # If the user explicitly provided both limits, honour them exactly in auto mode.
    if symmetric_arg == "auto" and vmin_arg is not None and vmax_arg is not None:
        symmetric = False
    else:
        symmetric = resolve_symmetric_flag(cmap, norm_mode, finite_data, symmetric_arg)

    if symmetric:
        if vmin is not None and vmax is not None:
            lim = max(abs(vmin), abs(vmax))
        elif vmax is not None:
            lim = abs(vmax)
        elif vmin is not None:
            lim = abs(vmin)
        else:
            lim = max(abs(data_min), abs(data_max))
        vmin, vmax = -lim, lim
    else:
        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max

    if vmax <= vmin:
        raise ValueError(f"Invalid limits: vmin={vmin}, vmax={vmax}")

    norm: mcolors.Normalize | None = None

    if norm_mode == "log":
        positive = finite_data[finite_data > 0]
        if positive.size == 0:
            raise ValueError("Log norm requires positive data after transform")
        if vmin <= 0:
            vmin = max(float(np.nanmin(positive)), 1e-12)
        if vmax <= 0:
            vmax = float(np.nanmax(positive))
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm_mode == "symlog":
        if linthresh is None:
            linthresh = max(abs(vmin) / 100.0, 1e-12)
        norm = mcolors.SymLogNorm(
            linthresh=linthresh,
            linscale=linscale,
            vmin=vmin,
            vmax=vmax,
            base=log_base,
        )

    return vmin, vmax, norm


def resolve_field_data(data: dict, plot_field: str, species: str | None) -> tuple[np.ndarray, str | None]:
    if plot_field not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"Field '{plot_field}' not found. Available fields: {available}")

    field_obj = data[plot_field]
    if isinstance(field_obj, dict):
        available_species = list(field_obj.keys())
        chosen_species = species
        if chosen_species is None:
            if len(available_species) == 1:
                chosen_species = available_species[0]
            else:
                raise KeyError(
                    f"Field '{plot_field}' has species {available_species}. "
                    "Specify one with suffix _e or _i."
                )
        if chosen_species not in field_obj:
            raise KeyError(f"Species '{chosen_species}' not available for '{plot_field}'. Choices: {available_species}")
        return field_obj[chosen_species], chosen_species

    return field_obj, None


def get_cmap(plot_field: str, cmap_arg: str) -> str:
    if cmap_arg != "auto":
        return cmap_arg
    positive_default_fields = {"rho", "Pxx", "Pyy", "Pzz", "agyrotropy", "Ppar/Pperp", "Bmagn", "beta_par", "beta_perp"}
    return "viridis" if plot_field in positive_default_fields else "seismic"


def make_movie(
    field_data: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    times: list | np.ndarray,
    run_id: str,
    plot_field: str,
    species: str | None,
    out_dir: Path,
    cmap: str,
    field_max: float | None,
    vmin: float | None,
    vmax: float | None,
    symmetric_limits: str,
    norm_mode_arg: str,
    linthresh: float | None,
    linscale: float,
    log_base: float,
    render_mode: str,
    interpolation: str,
    output_format: str,
    fps: int,
    dpi: int,
    show_title: bool,
) -> tuple[Path, float, float, float, float]:
    fig, ax = plt.subplots(figsize=(6, 5))

    norm_mode = resolve_norm_mode(plot_field, norm_mode_arg)
    plot_data = field_data

    finite_data = plot_data[np.isfinite(plot_data)]
    if finite_data.size == 0:
        plt.close(fig)
        raise ValueError(f"Field {plot_field} has no finite values")
    data_min = float(np.nanmin(finite_data))
    data_max = float(np.nanmax(finite_data))

    vmin_eff, vmax_eff, norm = build_norm_and_limits(
        data_for_limits=plot_data,
        cmap=cmap,
        norm_mode=norm_mode,
        field_max=field_max,
        vmin_arg=vmin,
        vmax_arg=vmax,
        symmetric_arg=symmetric_limits,
        linthresh=linthresh,
        linscale=linscale,
        log_base=log_base,
    )

    first_frame = plot_data[:, :, 0]
    if render_mode == "imshow":
        extent = (float(X.min()), float(X.max()), float(Y.min()), float(Y.max()))
        imshow_kwargs = {
            "origin": "lower",
            "extent": extent,
            "aspect": "auto",
            "interpolation": interpolation,
            "cmap": cmap,
        }
        if norm is None:
            cax = ax.imshow(first_frame.T, vmin=vmin_eff, vmax=vmax_eff, **imshow_kwargs)
        else:
            cax = ax.imshow(first_frame.T, norm=norm, **imshow_kwargs)
    else:
        if norm is None:
            cax = ax.pcolormesh(X, Y, first_frame, vmin=vmin_eff, vmax=vmax_eff, cmap=cmap)
        else:
            cax = ax.pcolormesh(X, Y, first_frame, cmap=cmap, norm=norm)

    fig.colorbar(cax)
    title_prefix = plot_field if species is None else f"{plot_field}, {species}"
    times_arr = np.asarray(times)
    if show_title:
        ax.set_title(f"{title_prefix}, {run_id}, t = {times_arr[0]:.3g}")

    def update(frame: int):
        frame_data = plot_data[:, :, frame]
        if render_mode == "imshow":
            cax.set_data(frame_data.T)
        else:
            cax.set_array(frame_data.ravel())
        if show_title:
            ax.set_title(f"{title_prefix}, {run_id}, t = {times_arr[frame]:.3g}")
        return (cax,)

    ani = animation.FuncAnimation(fig, update, frames=field_data.shape[2], blit=True)

    safe_field = sanitize_name(plot_field)
    if species is None:
        out_stem = f"{safe_field}_{run_id}_movie"
    else:
        out_stem = f"{safe_field}_{species}_{run_id}_movie"

    if output_format == "mp4":
        out_path = out_dir / f"{out_stem}.mp4"
        ani.save(out_path, dpi=dpi, fps=fps, writer="ffmpeg")
    else:
        out_path = out_dir / f"{out_stem}.gif"
        ani.save(out_path, dpi=dpi, fps=fps)
    plt.close(fig)
    return out_path, data_min, data_max, vmin_eff, vmax_eff


# ---------------------------------------------------------------------------
# ECsim-specific helpers
# ---------------------------------------------------------------------------

def resolve_files_path(override: str | None) -> str:
    if override is not None:
        return override
    try:
        return load_paths().get("data_dir", "./data")
    except Exception:
        return "./data"


def resolve_experiment_and_files_path(
    experiment_arg: str | None,
    files_path_arg: str | None,
    run_path_arg: str | None,
) -> tuple[str, str]:
    """Resolve experiment + files_path from explicit args or Menura-style --run_path."""
    if run_path_arg is not None:
        run_path = Path(run_path_arg).expanduser().resolve()
        return run_path.name, str(run_path.parent)

    if experiment_arg is None:
        raise ValueError("Provide experiment positional argument or --run_path")

    return experiment_arg, resolve_files_path(files_path_arg)


def _parse_first_float(value: str) -> float:
    for token in value.replace(",", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    raise ValueError(f"No float value found in: {value!r}")


def _parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for token in value.replace(",", " ").split():
        try:
            out.append(float(token))
        except ValueError:
            continue
    if not out:
        raise ValueError(f"No float values found in: {value!r}")
    return out


def infer_alfven_b0x_nb(experiment_path: Path) -> tuple[float, float, str]:
    """Infer b0x and nb from .inp using RHO_1 first, then RHO_3, then rhoINIT[0]."""
    inp_path = plasma._find_experiment_inp_file(str(experiment_path))

    b0x_value: float | None = None
    rho_1_value: float | None = None
    rho_3_value: float | None = None
    rho_init_values: list[float] | None = None

    for raw_line in inp_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        key_up = key.upper()

        if key_up == "B0X":
            b0x_value = _parse_first_float(value)
        elif key_up == "RHO_1":
            rho_1_value = _parse_first_float(value)
        elif key_up == "RHO_3":
            rho_3_value = _parse_first_float(value)
        elif key_up == "RHOINIT":
            rho_init_values = _parse_float_list(value)

    if b0x_value is None:
        raise ValueError(f"B0x not found in {inp_path}")

    if rho_1_value is not None:
        return b0x_value, rho_1_value, "RHO_1"
    if rho_3_value is not None:
        return b0x_value, rho_3_value, "RHO_3"
    if rho_init_values is not None and len(rho_init_values) > 0:
        return b0x_value, rho_init_values[0], "rhoINIT[0]"

    raise ValueError(
        f"Could not infer nb from {inp_path}. Expected one of: RHO_1, RHO_3, rhoINIT."
    )


def discover_available_snapshots(files_path: str, experiment: str) -> list[int]:
    """Return sequential indices (0, 1, 2, ...) for available field snapshots."""
    experiment_dir = os.path.join(files_path, experiment)
    filenames = rp._collect_experiment_filenames(experiment_dir)
    return list(range(len(filenames)))


def load_data(
    experiment: str,
    files_path: str,
    snapshot_indices: list[int],
    choose_species: list[str],
    choose_x: tuple[int, int] | None,
    choose_y: tuple[int, int] | None,
    alfven_units: bool,
    verbose: bool,
) -> tuple[dict, np.ndarray, np.ndarray, list, list]:
    """Load ECsim field data via rp.get_exp_times and compute Jxyz-tot."""
    choose_x_list = list(choose_x) if choose_x is not None else None
    choose_y_list = list(choose_y) if choose_y is not None else None

    fields_to_read = dict(DEFAULT_FIELDS_TO_READ)

    # Some iPiC3D archives omit fields such as divB. If a requested field is
    # missing, disable that field and retry automatically.
    for _ in range(len(fields_to_read) + 1):
        try:
            data, X, Y, qom, times = rp.get_exp_times(
                [experiment],
                files_path,
                fields_to_read,
                choose_species=choose_species,
                choose_times=snapshot_indices,
                choose_x=choose_x_list,
                choose_y=choose_y_list,
                verbose=verbose,
            )
            break
        except KeyError as exc:
            message = str(exc)
            match = re.search(r"'([^']+) is not a file in the archive'", message)
            missing_field = match.group(1) if match else None

            if missing_field in fields_to_read and fields_to_read[missing_field]:
                fields_to_read[missing_field] = False
                if verbose:
                    print(f"Field '{missing_field}' missing in archive; disabling it and retrying.")
                continue
            raise
    else:
        raise RuntimeError("Failed to load data after disabling missing fields")

    data = data[experiment]

    # Always compute Ohm's law terms (EPx, EPy, EPz, ExB/B^2, ...).
    plasma.get_Ohm(data, qom, X[:, 0], Y[0, :])

    if alfven_units:
        experiment_path = Path(files_path) / experiment
        b0x, nb, nb_source = infer_alfven_b0x_nb(experiment_path)
        X, Y, times = plasma.code2alfven(data, x=X, y=Y, times=times, b0x=b0x, nb=nb)
        if verbose:
            print(
                f"Applied Alfven units using {experiment_path}: "
                f"B0x={b0x}, nb={nb} (from {nb_source})"
            )

    # Convenient total current fields (must be built after optional unit conversion).
    if "Jz" in data and isinstance(data["Jz"], dict) and "e" in data["Jz"] and "i" in data["Jz"]:
        data["Jz-tot"] = data["Jz"]["e"] + data["Jz"]["i"]
        data["Jx-tot"] = data["Jx"]["e"] + data["Jx"]["i"]
        data["Jy-tot"] = data["Jy"]["e"] + data["Jy"]["i"]

    return data, X, Y, qom, times


def compute_processed_fields(data: dict, X: np.ndarray, Y: np.ndarray, qom: list) -> None:
    """Compute additional diagnostic fields via closure.utilities."""
    try:
        import closure.utilities as ut
    except Exception as exc:
        raise RuntimeError(
            "--processed was requested but closure.utilities could not be imported."
        ) from exc

    x1d = X[:, 0]
    y1d = Y[0, :]

    ut.get_PS_2D_field(data, x1d, y1d)
    ut.get_agyrotropy(data)

    try:
        ut.get_Az(x1d, y1d, data)
    except Exception:
        pass
    try:
        ut.get_J_perp(data, x1d, y1d)
    except Exception:
        pass

    data["Bmagn"] = np.sqrt(data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)

    if "Ppar" in data and "Pperp" in data and isinstance(data["Ppar"], dict) and "e" in data["Ppar"]:
        data["Ppar/Pperp"] = data["Ppar"]["e"] / data["Pperp"]["e"]
        data["(Ppar - Pperp)/B^2"] = (data["Ppar"]["e"] - data["Pperp"]["e"]) / data["Bmagn"] ** 2

        data["beta_par"] = {
            "e": 8 * np.pi * data["Ppar"]["e"] / (data["Bmagn"] ** 2),
            "i": 8 * np.pi * data["Ppar"]["i"] / (data["Bmagn"] ** 2),
        }
        data["beta_perp"] = {
            "e": 8 * np.pi * data["Pperp"]["e"] / (data["Bmagn"] ** 2),
            "i": 8 * np.pi * data["Pperp"]["i"] / (data["Bmagn"] ** 2),
        }
        data["beta_par - beta_perp"] = data["beta_par"]["e"] - data["beta_perp"]["e"]

    if "Jx" in data and "Jy" in data and isinstance(data["Jx"], dict) and "e" in data["Jx"]:
        data["Jinplane_e"] = np.sqrt(data["Jx"]["e"] ** 2 + data["Jy"]["e"] ** 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate animated ECsim field movies (GIF/MP4) from closure experiment data"
    )

    parser.add_argument("experiment", nargs="?", type=str, help="Experiment name (subdirectory of files_path)")
    parser.add_argument(
        "--files_path",
        type=str,
        default=None,
        help="Root directory containing experiment folders (defaults to paths.yaml data_dir)",
    )
    parser.add_argument(
        "--run_path",
        type=str,
        default=None,
        help="Menura-compatible alias: full path to one experiment folder",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="Jz-tot",
        help="Comma-separated fields, optionally with species suffix (e.g. Pxx_e)",
    )
    parser.add_argument(
        "--choose_times",
        type=str,
        default=None,
        help=(
            "'all'/'None': all snapshots; single index: 5; "
            "comma list of indices: 0,5,10; "
            "index slice: start:end[:step] (Python semantics, negatives ok)"
        ),
    )
    parser.add_argument("--choose_x", type=str, default=None, help="X index range as 'start,end'")
    parser.add_argument("--choose_y", type=str, default=None, help="Y index range as 'start,end'")
    parser.add_argument(
        "--choose_species",
        type=str,
        default="e,i",
        help="Comma-separated species labels to load, e.g. e,i or [e,i,e,i] (default: 'e,i')",
    )
    parser.add_argument(
        "--processed",
        action="store_true",
        help="Compute additional diagnostic fields via closure.utilities (agyrotropy, PiD, beta_par - beta_perp, ...)",
    )
    parser.add_argument("--field_max", type=float, default=None, help="Absolute color limit (seismic) or max (viridis)")
    parser.add_argument("--vmin", type=float, default=None, help="Explicit color minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Explicit color maximum")
    parser.add_argument(
        "--symmetric_limits",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether to enforce vmax = -vmin (useful for signed fields)",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="auto",
        choices=["auto", "linear", "log", "symlog"],
        help="Color normalization mode",
    )
    parser.add_argument("--linthresh", type=float, default=None, help="SymLog linthresh (defaults to abs(vmin)/100)")
    parser.add_argument("--linscale", type=float, default=1.0, help="SymLog linscale")
    parser.add_argument("--log_base", type=float, default=10.0, help="Log/SymLog base")
    parser.add_argument("--cmap", type=str, default="auto", help="Colormap name or 'auto'")
    parser.add_argument(
        "--render_mode",
        type=str,
        default="pcolormesh",
        choices=["pcolormesh", "imshow"],
        help="Field rendering backend; imshow can look smoother",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="nearest",
        choices=["nearest", "bilinear", "bicubic"],
        help="Interpolation method used when render_mode=imshow",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="gif",
        choices=["gif", "mp4"],
        help="Animation file format",
    )
    parser.add_argument("--fps", type=int, default=12, help="Animation frames per second")
    parser.add_argument("--dpi", type=int, default=220, help="DPI for generated animation")
    parser.add_argument("--no_title", action="store_true", help="Disable title overlay in the animation")
    parser.add_argument(
        "--alfven_units",
        action="store_true",
        help="Convert loaded fields/axes/time from code units to Alfven units using experiment .inp",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment, files_path = resolve_experiment_and_files_path(
        experiment_arg=args.experiment,
        files_path_arg=args.files_path,
        run_path_arg=args.run_path,
    )
    choose_x = parse_range_arg(args.choose_x)
    choose_y = parse_range_arg(args.choose_y)
    choose_species = parse_list_arg(args.choose_species, dtype=str)
    fields_list, species_list = parse_fields_arg(args.fields)

    available_snapshots = discover_available_snapshots(files_path, experiment)
    if not available_snapshots:
        raise FileNotFoundError(
            f"No field snapshots found for experiment '{experiment}' under {files_path}"
        )

    snapshot_indices = select_iterations(available_snapshots, args.choose_times)

    data, X, Y, qom, times = load_data(
        experiment=experiment,
        files_path=files_path,
        snapshot_indices=snapshot_indices,
        choose_species=choose_species,
        choose_x=choose_x,
        choose_y=choose_y,
        alfven_units=args.alfven_units,
        verbose=args.verbose,
    )

    if args.processed:
        compute_processed_fields(data, X, Y, qom)

    out_dir = Path(files_path) / experiment / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Experiment: {experiment}")
        print(f"Files path: {files_path}")
        print(f"Selected snapshot indices: {snapshot_indices}")
        print(f"Times: {times}")
        print(f"Fields: {list(zip(fields_list, species_list))}")
        print(f"Grid shape: X={X.shape}, Y={Y.shape}")
        print(
            "Color options: "
            f"cmap={args.cmap}, norm={args.norm}, vmin={args.vmin}, vmax={args.vmax}, "
            f"symmetric_limits={args.symmetric_limits}"
        )
        print(
            "Render options: "
            f"render_mode={args.render_mode}, interpolation={args.interpolation}, "
            f"output_format={args.output_format}, fps={args.fps}, dpi={args.dpi}, "
            f"show_title={not args.no_title}"
        )

    for plot_field, species in zip(fields_list, species_list):
        try:
            field_data, chosen_species = resolve_field_data(data, plot_field, species)
            cmap = get_cmap(plot_field, args.cmap)
            out_path, data_min, data_max, vmin_eff, vmax_eff = make_movie(
                field_data=field_data,
                X=X,
                Y=Y,
                times=times,
                run_id=experiment,
                plot_field=plot_field,
                species=chosen_species,
                out_dir=out_dir,
                cmap=cmap,
                field_max=args.field_max,
                vmin=args.vmin,
                vmax=args.vmax,
                symmetric_limits=args.symmetric_limits,
                norm_mode_arg=args.norm,
                linthresh=args.linthresh,
                linscale=args.linscale,
                log_base=args.log_base,
                render_mode=args.render_mode,
                interpolation=args.interpolation,
                output_format=args.output_format,
                fps=args.fps,
                dpi=args.dpi,
                show_title=not args.no_title,
            )
            print(f"Saved animation at {out_path}")
            print(
                f"Field range ({plot_field}{'' if chosen_species is None else '_' + chosen_species}): "
                f"data_min={data_min:.6g}, data_max={data_max:.6g}, "
                f"applied_vmin={vmin_eff:.6g}, applied_vmax={vmax_eff:.6g}"
            )
        except Exception as exc:
            print(f"Skipping {plot_field} ({species}): {exc}")


if __name__ == "__main__":
    main()
