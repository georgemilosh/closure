#!/usr/bin/env python
"""Ez/Jz (or any fields) montages at reconnection-rate thresholds.

For every requested (regime, code) run of a stability campaign and every
selected NN-closure checkpoint, this script:

1. reads the run's ``reconnection_menura.csv`` and finds the FIRST field-dump
   index at which the normalized reconnection rate reaches each requested
   fraction of that run's OWN peak (default: 75% and 90%);
2. loads exactly those field dumps (optionally cropped, same
   ``--choose-x/--choose-y`` convention as ``closure-diagnostics fields``);
3. renders one montage PNG per (regime, code): columns = models side by side,
   rows = <field> @ <fraction> for every combination, using the same styling
   machinery as the ``fields`` CLI (``resolve_field_data`` species handling,
   ``get_cmap`` auto colormaps, ``_field_limits`` robust symmetric color
   scaling, per-panel colorbars).

Models with no usable reconnection data (e.g. runs that crashed before the
second field dump) get an annotated blank column; (regime, code) combinations
with no runs at all (e.g. the stability_campaign2 R5/new GPU-OOM column) are
skipped with a message.

Defaults reproduce the stability_campaign2 figures under
``diagnostics/stability_campaign2/fields_at_recon_thresholds/``:

    python scripts/fields_at_recon_thresholds.py

Everything is adjustable from the command line with the same option names the
``fields`` CLI uses where they overlap, e.g.:

    # different fields, wider crop, another campaign
    python scripts/fields_at_recon_thresholds.py \
        --fields Ez,Jz_i,Bz --choose-x 0,512 --choose-y 0,256 \
        --campaign stability_campaign100ppc

    # a quick look at two models in one regime
    python scripts/fields_at_recon_thresholds.py \
        --regimes R0 --codes new --good-models FCNN_00285 --bad-models MLP_00535

    # different thresholds (fractions of each run's own peak recon rate)
    python scripts/fields_at_recon_thresholds.py --fracs 0.5,0.9
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from closure.diagnostics import (
    _field_limits,
    _panel_values,
    get_cmap,
    load_experiment_data,
    parse_field_specs,
    resolve_field_data,
)

# --------------------------------------------------------------------------
# Defaults. Every one of these can be overridden on the command line; edit
# here only if you want a different permanent default.
# --------------------------------------------------------------------------

#: Campaign name; used to derive both roots below when they are not given.
DEFAULT_CAMPAIGN = "stability_campaign2"

#: Where the per-regime diagnostics CSVs live (<root>/<regime>/reconnection_menura.csv).
DIAGNOSTICS_BASE = "/volume1/scratch/georgem/closure/diagnostics"

#: Where the menura run directories live (<root>/<regime>/<code>_<model>/products/...).
RUNS_BASE = "/esat/cpadata/georgem/2025_112/georgem/menura/runs"

#: Physical regimes (sub-directories of the campaign) and code variants; the
#: run directory for a cell is "<code>_<model_key>".
DEFAULT_REGIMES = ["R5", "R0", "R7", "R12"]
DEFAULT_CODES = ["old", "new"]

#: Checkpoint selection: "good" = completes every started regime/code column
#: of stability_campaign2, "bad" = crashes in most (see stability2_scan.ipynb,
#: survival heatmap). Order matters only for column order in the montage.
DEFAULT_GOOD_MODELS = [
    "FCNN_00285",  # step100 noJnoE/deeper - best: survives all, lowest Ez noise
    "FCNN_00172",  # step100 default/deeper - lowest val_loss, noisy but bounded
    "FCNN_00711",  # prodval0 noJnoE/deeper
    "FCNN_00938",  # prodval0 noJnoE/baseline - worst val_loss yet stable
    "MLP_00643",   # prodval0 noJnoE/deeper
    "MLP_00586",   # serial noJnoE/deeper
]
DEFAULT_BAD_MODELS = [
    "FCNN_00365",  # serial default/shallower - dies within the first dumps
    "FCNN_00435",  # prodval0 default/shallower - high-noise plateau rider
    "FCNN_00611",  # step100 noJnoE/baseline - the step100 that fails at 2000 ppc
    "MLP_00535",   # serial default/baseline - high-noise plateau rider
    "MLP_00596",   # serial noJnoE/shallower
    "MLP_00772",   # prodval0 noJnoE/baseline - campaign-1 "new" model, late blowup
]

#: Fractions of each run's own peak recon_rate_norm; one montage row per
#: (fraction x field).
DEFAULT_FRACS = [0.7, 0.8, 0.90]


def _default_menura_analysis_dir() -> str | None:
    """menura_analysis_dir from the repo's own paths.yaml, cwd-independent.

    ``closure.config.load_paths`` opens ``paths.yaml`` relative to the current
    working directory, so scripts run from anywhere but the repo root would
    silently lose this setting; resolve it from this file's location instead.
    """
    paths_yaml = Path(__file__).resolve().parents[1] / "paths.yaml"
    if paths_yaml.exists():
        import yaml
        value = (yaml.safe_load(paths_yaml.read_text()) or {}).get("menura_analysis_dir")
        if value:
            return str(Path(value).expanduser())
    return None


def _parse_pair(value: str | None) -> tuple[int, int] | None:
    """'a,b' -> (a, b); same syntax as the fields CLI --choose-x/--choose-y."""
    if value is None:
        return None
    parts = [int(v) for v in str(value).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'start,end', got {value!r}")
    return parts[0], parts[1]


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- what to analyze -------------------------------------------------
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN,
                        help="Campaign name used to derive --files-path and --diagnostics-root")
    parser.add_argument("--files-path", default=None,
                        help=f"Root of the run directories (default: {RUNS_BASE}/<campaign>)")
    parser.add_argument("--diagnostics-root", default=None,
                        help=f"Root of the diagnostics CSVs (default: {DIAGNOSTICS_BASE}/<campaign>)")
    parser.add_argument("--regimes", type=_parse_list, default=DEFAULT_REGIMES,
                        help="Comma-separated regime sub-directories")
    parser.add_argument("--codes", type=_parse_list, default=DEFAULT_CODES,
                        help="Comma-separated code variants (run dir prefix)")
    parser.add_argument("--good-models", type=_parse_list, default=DEFAULT_GOOD_MODELS,
                        help="Comma-separated 'good' checkpoint keys (left columns)")
    parser.add_argument("--bad-models", type=_parse_list, default=DEFAULT_BAD_MODELS,
                        help="Comma-separated 'bad' checkpoint keys (right columns)")
    parser.add_argument("--fracs", default=DEFAULT_FRACS,
                        type=lambda v: [float(f) for f in _parse_list(v)],
                        help="Comma-separated fractions of each run's own peak recon rate")
    parser.add_argument("--rate-column", default="recon_rate_norm",
                        help="Column of reconnection_menura.csv used for the thresholds")
    parser.add_argument("--recon-csv", default="reconnection_menura.csv",
                        help="Per-regime reconnection CSV filename")

    # ---- options passed through to the fields-CLI machinery -------------
    parser.add_argument("--fields", default="Ez,Jz",
                        help="Comma-separated fields, same syntax as `closure-diagnostics fields`")
    parser.add_argument("--backend", choices=["ecsim", "menura", "auto"], default="menura",
                        help="Data loader backend")
    parser.add_argument("--choose-x", type=_parse_pair, default="64,256",
                        help="Load x-index range as start,end (empty string for full domain)")
    parser.add_argument("--choose-y", type=_parse_pair, default="86,160",
                        help="Load y-index range as start,end (empty string for full domain)")
    parser.add_argument("--choose-species", type=_parse_list, default=["e", "i"],
                        help="Comma-separated species labels for the loader")
    parser.add_argument("--normalization",
                        choices=["none", "alfven-infer", "alfven-sample", "alfven-explicit"],
                        default="none", help="Optional code2alfven normalization")
    parser.add_argument("--menura-analysis-dir", default=_default_menura_analysis_dir(),
                        help="Directory containing read_menura.py. Defaults to the repo "
                        "paths.yaml value regardless of cwd (the package resolves "
                        "paths.yaml relative to the cwd, so running from outside the "
                        "repo root would otherwise fall back to the first "
                        "menura/analysis dir found above --files-path - e.g. the "
                        "/esat checkout, whose menura_utils still uses the NumPy-1-only "
                        "np.recfromtxt)")
    parser.add_argument("--cmap", default="auto",
                        help="Colormap or 'auto' (seismic for signed fields, viridis for positive)")
    parser.add_argument("--robust-quantile", type=float, default=0.995,
                        help="Quantile for the symmetric color limits of each panel")

    # ---- output ----------------------------------------------------------
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <diagnostics-root>/fields_at_recon_thresholds)")
    parser.add_argument("--dpi", type=int, default=110, help="Saved figure DPI")
    parser.add_argument("--panel-width", type=float, default=3.4,
                        help="Width (inches) of one montage column")
    parser.add_argument("--panel-height", type=float, default=None,
                        help="Height (inches) of one montage row. Default: automatic, "
                        "panel-width x the loaded data's y/x aspect, so the "
                        "equal-aspect panels fill their slots with no blank bands "
                        "for any crop / model count / row count")
    return parser


def threshold_indices(run_frame: pd.DataFrame, fracs: list[float], rate_column: str):
    """First (time_index, time) at which ``rate_column`` reaches each fraction
    of the run's own peak, or None when the run has no usable rate data."""
    s = run_frame.dropna(subset=[rate_column]).sort_values("time_index")
    if s.empty:
        return None
    peak = s[rate_column].max()
    if not np.isfinite(peak) or peak <= 0:
        return None
    out = {}
    for frac in fracs:
        row = s[s[rate_column] >= frac * peak].iloc[0]
        out[frac] = (int(row["time_index"]), float(row["time"]))
    return out


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    files_path = args.files_path or f"{RUNS_BASE}/{args.campaign}"
    diagnostics_root = args.diagnostics_root or f"{DIAGNOSTICS_BASE}/{args.campaign}"
    out_dir = Path(args.output_dir or f"{diagnostics_root}/fields_at_recon_thresholds")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = args.good_models + args.bad_models
    specs = parse_field_specs(args.fields)
    # One montage row per (fraction, field): e.g. Ez@75, Jz@75, Ez@90, Jz@90.
    rows = [(frac, spec) for frac in args.fracs for spec in specs]

    recon = {R: pd.read_csv(f"{diagnostics_root}/{R}/{args.recon_csv}") for R in args.regimes}

    for R in args.regimes:
        runs_present = set(recon[R]["run"].unique())
        for code in args.codes:
            if not any(f"{code}_{m}" in runs_present for m in models):
                print(f"skip {R}/{code}: no runs found in {args.recon_csv}")
                continue

            # ---- pass 1: thresholds + field loads. Loading BEFORE creating
            # the figure lets the subplot slots be sized to the data's own
            # aspect ratio; a fixed slot shape combined with the equal-aspect
            # panels would otherwise leave blank bands whenever the crop is
            # wider/flatter than the slot (as the default 192x74-cell crop is).
            loaded = {}
            for m in models:
                run = f"{code}_{m}"
                th = (threshold_indices(recon[R][recon[R]["run"] == run],
                                        args.fracs, args.rate_column)
                      if run in runs_present else None)
                if th is None:
                    loaded[m] = None
                    continue
                # Load only the threshold dumps (deduplicated: a steep rise can
                # cross several fractions in the same dump).
                idxs = sorted({th[f][0] for f in args.fracs})
                data, X, Y, qom, times = load_experiment_data(
                    run, f"{files_path}/{R}",
                    backend=args.backend,
                    choose_times=",".join(map(str, idxs)),
                    choose_species=args.choose_species,
                    choose_x=args.choose_x,
                    choose_y=args.choose_y,
                    normalization=args.normalization,
                    menura_analysis_dir=args.menura_analysis_dir,
                )
                loaded[m] = (th, data, X, Y, {ti: k for k, ti in enumerate(idxs)})
                print(f"{R}/{code} {m}: " +
                      " ".join(f"{int(f * 100)}%=idx{th[f][0]}" for f in args.fracs), flush=True)

            if all(v is None for v in loaded.values()):
                print(f"skip {R}/{code}: no run has usable {args.rate_column} data")
                continue

            # ---- figure geometry: slot aspect = data aspect (regardless of
            # how many models/rows/fields were requested). panel-height, when
            # given, overrides the automatic value.
            _, _, X0, Y0, _ = next(v for v in loaded.values() if v is not None)
            x_span = float(np.nanmax(X0) - np.nanmin(X0))
            y_span = float(np.nanmax(Y0) - np.nanmin(Y0))
            data_aspect = y_span / x_span if x_span > 0 else 1.0
            panel_h = args.panel_height or args.panel_width * data_aspect
            fig, axes = plt.subplots(
                len(rows), len(models),
                figsize=(args.panel_width * len(models), panel_h * len(rows)),
                squeeze=False,
                gridspec_kw=dict(wspace=0.18, hspace=0.10),
            )

            # ---- pass 2: render.
            for j, m in enumerate(models):
                tag = "good" if m in args.good_models else "bad"
                if loaded[m] is None:
                    for i in range(len(rows)):
                        axes[i][j].axis("off")
                    axes[0][j].set_title(f"[{tag}] {m}\n(no recon data)", fontsize=9)
                    continue
                th, data, X, Y, pos = loaded[m]

                for i, (frac, spec) in enumerate(rows):
                    ax = axes[i][j]
                    ti, tval = th[frac]
                    arr, species = resolve_field_data(data, spec)
                    vals = arr[..., pos[ti]] if arr.ndim == 3 else arr
                    # fields-CLI conventions: flip all-negative fields positive,
                    # robust symmetric limits, auto colormap, panel colorbar.
                    vplot = _panel_values(vals)
                    vmin, vmax = _field_limits(vplot, robust_quantile=args.robust_quantile)
                    im = ax.pcolormesh(X, Y, vplot, vmin=vmin, vmax=vmax,
                                       cmap=get_cmap(spec.name, args.cmap))
                    ax.set_aspect("equal")
                    divider = make_axes_locatable(ax)
                    cbar = fig.colorbar(im, cax=divider.append_axes("right", size="4%", pad=0.05))
                    cbar.ax.tick_params(labelsize=7)
                    label = spec.name if species is None else f"{spec.name}_{species}"
                    ax.text(0.02, 0.94, f"{label}, t={tval:.3g} ({int(frac * 100)}%)",
                            transform=ax.transAxes, fontsize=8, va="top",
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
                    ax.set_xticks([]); ax.set_yticks([])
                    if i == 0:
                        ax.set_title(f"[{tag}] {m}", fontsize=10)
                    if j == 0:
                        # Compact label: at the auto (data-aspect) row height a
                        # long ylabel is taller than its own panel and collides
                        # with the neighboring rows' labels.
                        ax.set_ylabel(f"{spec.label} @{int(frac * 100)}%", fontsize=9)

            crop = ""
            if args.choose_x or args.choose_y:
                crop = (f" (x {args.choose_x[0]}:{args.choose_x[1]}," if args.choose_x else " (") + \
                       (f" y {args.choose_y[0]}:{args.choose_y[1]})" if args.choose_y else ")")
            fig.suptitle(
                f"{args.campaign} {R} / {code} code: {args.fields} at first crossing of "
                + " / ".join(f"{int(f * 100)}%" for f in args.fracs)
                + f" of each run's peak {args.rate_column}{crop}",
                fontsize=14, y=1.0,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            out = out_dir / f"{R}_{code}.png"
            fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print("saved", out, flush=True)


if __name__ == "__main__":
    main()
