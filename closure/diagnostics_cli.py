"""Command-line entry point for closure field diagnostics."""

from __future__ import annotations

__all__ = ["main"]

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from closure.diagnostics import (
    build_profiles_dataframe,
    export_reconnection_dataframe,
    load_experiment_data,
    parse_field_specs,
    plot_csv_overlay,
    plot_field_panels,
)


logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _parse_pair(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected start,end; got {value!r}")
    if parts[1] <= parts[0]:
        raise argparse.ArgumentTypeError(f"Range end must be greater than start; got {value!r}")
    return parts[0], parts[1]


def _parse_species(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_factor(value: str) -> float:
    cleaned = value.strip().lower().replace("*", "")
    if cleaned == "pi":
        return math.pi
    if cleaned.endswith("pi"):
        factor = cleaned[:-2]
        return float(factor or 1.0) * math.pi
    return float(value)


def _add_load_options(parser: argparse.ArgumentParser, *, default_choose_times: str) -> None:
    parser.add_argument("experiments", nargs="+", help="Experiment names under --files-path")
    parser.add_argument("--backend", choices=["ecsim", "menura", "auto"], default="ecsim", help="Data loader backend")
    parser.add_argument("--files-path", default=None, help="Root directory containing experiment folders or Menura run folders")
    parser.add_argument("--menura-analysis-dir", default=None, help="Directory containing read_menura.py")
    parser.add_argument(
        "--menura-scale-ranges",
        action="store_true",
        help="Scale --choose-x/--choose-y from a base 512-cell run to the Menura run resolution",
    )
    parser.add_argument("--menura-base-nx", type=int, default=512, help="Base Nx used with --menura-scale-ranges")
    parser.add_argument("--choose-times", default=default_choose_times, help="all, int, comma list, or start:end[:step]")
    parser.add_argument("--choose-x", type=_parse_pair, default=None, help="Load x-index range as start,end")
    parser.add_argument("--choose-y", type=_parse_pair, default=None, help="Load y-index range as start,end")
    parser.add_argument("--choose-species", type=_parse_species, default=["e", "i"], help="Comma-separated species labels")
    parser.add_argument("--processed", action="store_true", help="Compute derived plasma diagnostics before exporting")
    parser.add_argument(
        "--normalization",
        choices=["none", "alfven-infer", "alfven-sample", "alfven-explicit"],
        default="none",
        help="Optional code2alfven normalization. alfven-sample matches the notebook b0x/rho_i sample convention.",
    )
    parser.add_argument("--b0x", type=float, default=None, help="Explicit B0x for normalization")
    parser.add_argument("--nb", type=float, default=None, help="Explicit background density for normalization")
    parser.add_argument(
        "--sample-nb-factor",
        type=_parse_factor,
        default=1.0,
        help="Multiplier for rho_i max in alfven-sample mode; accepts values like 1, pi, 4pi",
    )
    parser.add_argument(
        "--no-density-norm",
        dest="normalize_density",
        action="store_false",
        help="Keep density in code units while still casting B and the other fields/axes to Alfven units",
    )
    parser.add_argument("--alfven-units", action="store_true", help="Alias for --normalization alfven-infer")
    parser.add_argument("--verbose", action="store_true", help="Print read_pic loading details")


def _load_for_command(args: argparse.Namespace, experiment: str):
    return load_experiment_data(
        experiment,
        args.files_path,
        backend=args.backend,
        choose_times=args.choose_times,
        choose_species=args.choose_species,
        choose_x=args.choose_x,
        choose_y=args.choose_y,
        processed=args.processed,
        alfven_units=args.alfven_units,
        normalization=args.normalization,
        b0x=args.b0x,
        nb=args.nb,
        nb_factor=args.sample_nb_factor,
        normalize_density=args.normalize_density,
        menura_analysis_dir=args.menura_analysis_dir,
        menura_scale_ranges=args.menura_scale_ranges,
        menura_base_nx=args.menura_base_nx,
        verbose=args.verbose,
    )


def _cmd_fields(args: argparse.Namespace) -> None:
    specs = parse_field_specs(args.fields)
    output_dir = Path(args.output_dir)
    for experiment in args.experiments:
        data, X, Y, qom, times = _load_for_command(args, experiment)
        time_index = args.time_index
        if time_index < 0:
            time_index += len(times)
        if time_index < 0 or time_index >= len(times):
            raise IndexError(f"time_index {args.time_index} out of range for {experiment}: 0..{len(times) - 1}")
        if args.output and len(args.experiments) == 1:
            output = Path(args.output)
        else:
            output = output_dir / experiment / f"fields_t{time_index}.png"
        path = plot_field_panels(
            data,
            X,
            Y,
            specs,
            run_name=experiment,
            time_index=time_index,
            time_value=float(times[time_index]),
            output=output,
            ncols=args.ncols,
            cmap=args.cmap,
        )
        logger.info("Saved field panel: %s", path)


def _cmd_profiles(args: argparse.Namespace) -> None:
    specs = parse_field_specs(args.fields)
    frames = []
    for experiment in args.experiments:
        data, X, Y, qom, times = _load_for_command(args, experiment)
        if args.time_indices is None:
            time_indices = list(range(len(times)))
        else:
            time_indices = [int(part.strip()) for part in args.time_indices.split(",") if part.strip()]
        frames.append(
            build_profiles_dataframe(
                data,
                X,
                Y,
                specs,
                run_name=experiment,
                times=times,
                time_indices=time_indices,
                projection=args.projection,
                cut_index=args.cut_index,
                cut_value=args.cut_value,
            )
        )
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    logger.info("Saved profile CSV: %s (%d rows)", output, sum(len(frame) for frame in frames))


def _write_csv(frame: pd.DataFrame, output: Path, *, mode: str) -> tuple[str, int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {"append", "replace"}:
        raise ValueError(f"Unknown CSV write mode: {mode!r}")
    if mode == "replace" or not output.exists():
        frame.to_csv(output, index=False)
        return "replaced" if mode == "replace" else "created", 0, len(frame)

    try:
        existing_header = list(pd.read_csv(output, nrows=0).columns)
    except pd.errors.EmptyDataError:
        frame.to_csv(output, index=False)
        return "created", 0, len(frame)

    if existing_header != list(frame.columns):
        raise ValueError(
            f"Cannot append to {output}: existing columns {existing_header} "
            f"do not match new columns {list(frame.columns)}"
        )
    existing_rows = sum(1 for _ in output.open()) - 1
    frame.to_csv(output, mode="a", header=False, index=False)
    return "appended", max(existing_rows, 0), len(frame)


def _cmd_reconnection(args: argparse.Namespace) -> None:
    frames = []
    az_filter = None
    if args.az_sigma is not None:
        az_filter = {"name": "gaussian_filter", "sigma": args.az_sigma, "axes": (0, 1)}
    for experiment in args.experiments:
        logger.info(
            "Loading %s with backend=%s, choose_times=%s, normalization=%s",
            experiment,
            args.backend,
            args.choose_times,
            args.normalization,
        )
        data, X, Y, qom, times = _load_for_command(args, experiment)
        logger.info("Loaded %s: grid=%s, snapshots=%d", experiment, data["Bx"].shape[:2], len(times))
        # Reconnection only needs Az (from Bx/By) and the current totals, both of
        # which export_reconnection_dataframe computes itself. Skip the full
        # compute_common_diagnostics battery (Ohm, pressure-strain, agyrotropy,
        # J_perp) — it is ~45 s/snapshot at 1024x512 and unused here.
        frame = export_reconnection_dataframe(
            data,
            X,
            Y,
            times,
            run_name=experiment,
            qom=qom,
            az_filter=az_filter,
            grad_tol=args.grad_tol,
            merge_tol=args.merge_tol,
            seed_grad_frac=args.seed_grad_frac if args.seed_grad_frac > 0 else None,
            recon_normalization=args.recon_normalization,
        )
        logger.info("Computed reconnection diagnostics for %s: %d rows", experiment, len(frame))
        frames.append(frame)
    output = Path(args.output_csv)
    combined = pd.concat(frames, ignore_index=True)
    action, previous_rows, new_rows = _write_csv(combined, output, mode=args.csv_mode)
    logger.info(
        "%s reconnection CSV: %s (%d new rows, %d previous rows)",
        action.capitalize(),
        output,
        new_rows,
        previous_rows,
    )


def _parse_select(pairs: list[str] | None) -> dict[str, list[str]] | None:
    """Parse repeated ``column=val[,val...]`` filters into a select mapping."""
    if not pairs:
        return None
    select: dict[str, list[str]] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--select expects column=value, got {item!r}")
        col, raw = item.split("=", 1)
        col = col.strip()
        values = [v.strip() for v in raw.split(",") if v.strip()]
        if not col or not values:
            raise ValueError(f"--select expects column=value, got {item!r}")
        select.setdefault(col, []).extend(values)
    return select


def _cmd_overlay(args: argparse.Namespace) -> None:
    select = _parse_select(args.select)
    if args.field:
        fields = [f.strip() for f in args.field.split(",") if f.strip()]
        select = {**(select or {}), "field_label": fields}
    path = plot_csv_overlay(
        args.csvs,
        output=args.output,
        x=args.x,
        y=args.y,
        group_by=args.group_by,
        title=args.title,
        dpi=args.dpi,
        logx=args.logx,
        logy=args.logy,
        select=select,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )
    logger.info("Saved overlay figure: %s", path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Notebook-style field diagnostics for closure experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fields = subparsers.add_parser("fields", help="Plot several 2D fields together")
    _add_load_options(fields, default_choose_times="0")
    fields.add_argument("--fields", default="Az,Ey,Ez,rho_e,rho_i,Jz_e,Jz_i,Bx,By,Bz", help="Comma-separated fields")
    fields.add_argument("--time-index", type=int, default=0, help="Loaded time index to plot")
    fields.add_argument("--output", default=None, help="Output path for one experiment")
    fields.add_argument("--output-dir", default="diagnostics", help="Output directory for generated figures")
    fields.add_argument("--ncols", type=int, default=None, help="Number of panel columns")
    fields.add_argument("--cmap", default="auto", help="Colormap or auto")
    fields.set_defaults(func=_cmd_fields)

    profiles = subparsers.add_parser("profiles", help="Export 1D profile cuts to CSV")
    _add_load_options(profiles, default_choose_times="0")
    profiles.add_argument("--fields", required=True, help="Comma-separated fields, e.g. P_e,rho_e,Jz_e,Bx")
    profiles.add_argument("--projection", choices=["x", "y"], default="y", help="Coordinate that varies along the cut")
    profiles.add_argument("--cut-index", type=int, default=None, help="Fixed-axis index for the cut")
    profiles.add_argument("--cut-value", type=float, default=None, help="Fixed-axis coordinate nearest to this value")
    profiles.add_argument("--time-indices", default=None, help="Loaded time indices to export, comma-separated; default all loaded")
    profiles.add_argument("--output-csv", default="diagnostics/profiles.csv", help="Output CSV path")
    profiles.set_defaults(func=_cmd_profiles)

    reconnection = subparsers.add_parser("reconnection", help="Export X/O point and reconnection-rate diagnostics to CSV")
    _add_load_options(reconnection, default_choose_times="all")
    reconnection.add_argument("--output-csv", default="diagnostics/reconnection.csv", help="Output CSV path")
    reconnection.add_argument("--az-sigma", type=float, default=None, help="Optional Gaussian sigma for Az before X/O search (notebook uses 4)")
    reconnection.add_argument("--grad-tol", type=float, default=1e-6, help="Gradient tolerance for X/O root acceptance (notebook default)")
    reconnection.add_argument("--merge-tol", type=float, default=1e-3, help="Duplicate X/O merge tolerance")
    reconnection.add_argument(
        "--seed-grad-frac",
        type=float,
        default=0.05,
        help="Seed the X/O search only from local |grad Az| minima below this fraction of "
        "max|grad Az|; prunes spurious seeds (large speedup at high resolution). Use 0 to disable.",
    )
    reconnection.add_argument(
        "--recon-normalization",
        choices=["none", "notebook"],
        default="none",
        help="Add normalized recon_rate_norm/time_norm columns. 'notebook' reproduces fullres.ipynb cell 6.",
    )
    reconnection.add_argument(
        "--csv-mode",
        choices=["append", "replace"],
        default="append",
        help="Whether to append to or replace --output-csv (default: append)",
    )
    reconnection.set_defaults(func=_cmd_reconnection)

    overlay = subparsers.add_parser("overlay", help="Overlay profile or reconnection CSV files")
    overlay.add_argument("csvs", nargs="+", help="CSV files produced by profiles or reconnection")
    overlay.add_argument("--output", default="diagnostics/overlay.png", help="Output figure path")
    overlay.add_argument("--x", default=None, help="CSV column for x; defaults to coord or time")
    overlay.add_argument("--y", default=None, help="CSV column for y; defaults to value or recon_rate")
    overlay.add_argument("--group-by", nargs="*", default=None, help="Columns defining overlay series")
    overlay.add_argument(
        "--field",
        default=None,
        help="Plot only these field_label values (comma-separated), e.g. P_e or Bx,By. Mirrors one notebook profile cell.",
    )
    overlay.add_argument(
        "--select",
        action="append",
        default=None,
        metavar="COL=VAL",
        help="Filter rows before plotting, e.g. --select run=Le2DHGEM_RunID_0_f2 (repeatable; comma-separated values allowed)",
    )
    overlay.add_argument("--title", default=None, help="Optional plot title")
    overlay.add_argument("--xlabel", default=None, help="Override x-axis label (default: cut coordinate or 'time')")
    overlay.add_argument("--ylabel", default=None, help="Override y-axis label (default: field name or 'reconnection rate')")
    overlay.add_argument("--logx", action="store_true", help="Use a logarithmic x axis")
    overlay.add_argument("--logy", action="store_true", help="Use a logarithmic y axis")
    overlay.add_argument("--dpi", type=int, default=200, help="Saved figure DPI")
    overlay.set_defaults(func=_cmd_overlay)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()