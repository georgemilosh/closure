#!/usr/bin/env python
"""Outflow-rectangle diagnostic for the lower-sheet reconnection X point.

Characterises the jet-like current intensification attached to the X point
with one geometric object and one scalar:

* the OUTFLOW RECTANGLE (dotted orange): thin, centred on the X point, angle
  and length optimized jointly so it tracks the intensified current band
  (see outflow_rectangle);
* the RATIO `jz_ratio`: mean |Jz-tot| inside the rectangle over mean
  |Jz-tot| in the rest of the separatrix interior (the region between the
  separatrix branches and the rectangle, bounded in x by the domain edges).
  "How much stronger is the outflow current than the rest of the sheet it
  lives in" - both samples adapt to the run's own geometry.

Per run and per threshold time (first field dump where recon_rate_norm
reaches the requested fraction of that run's own peak):

1. X/O anchors come from the reconnection CSV row - the same pair whose
   Az_O - Az_X defines recon_rate_norm (--local-xo re-derives them from
   unsmoothed Az instead; A/B-tested identical for jet-bearing runs).
2. The separatrix is the Az level set across which median |Jz| jumps most
   sharply (find_separatrix; white contour on the maps).
3. The outflow rectangle is fitted and jz_ratio computed.

Supporting views per case: the x = const Jz cuts at the near-X window edges
(outflow crossings) and the ridge histogram against the corresponding ECsim
kinetic reference (Le2DHGEM_RunID_<n>_f2 for regime R<n>).

By default EVERY model in the campaign/regime folder is processed (the run
directories themselves are the authority on what was run); --models narrows
that to a hand-picked subset.

Example:

    python scripts/jet_metric.py --regime R12 --code new --fracs 0.75,0.9
    python scripts/jet_metric.py --regime R12 --models FCNN_00285,MLP_00535
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from closure import config
from closure.diagnostics import (
    _as_axis,
    _compute_current_totals,
    _field_limits,
    _panel_values,
    get_cmap,
    load_experiment_data,
    parse_field_specs,
    resolve_field_data,
)

# --------------------------------------------------------------------------
# Path roots — resolved from paths.yaml (closure.config.load_paths), all
# overridable on the command line:
#   menura_runs_dir -> RUNS_BASE   menura campaign run trees (<campaign>/<regime>/<run>).
#                                  Distinct from menura_analysis_dir, which points at the
#                                  read_menura/menura_utils checkout used by the loader.
#   data_dir    -> ECSIM_PREPEND   ECsim/iPiC3D data root; the "iPiC3D-nathan" subpath is
#                                  given via --ecsim-files-path and joined onto this prepend
#   diagnostics -> DIAGNOSTICS_BASE  always ./diagnostics under the repo, unless paths.yaml
#                                    overrides it with a diagnostics_dir key
# --------------------------------------------------------------------------
_PATHS = config.load_paths()
_REPO_ROOT = Path(config.__file__).resolve().parents[1]
DIAGNOSTICS_BASE = str(_PATHS.get("diagnostics_dir") or (_REPO_ROOT / "diagnostics"))
RUNS_BASE = _PATHS.get("menura_runs_dir") or "/dodrio/scratch/projects/2026_018/george/menura/runs"
#: ECsim/iPiC3D data root that a relative --ecsim-files-path is prepended with.
ECSIM_PREPEND = _PATHS.get("data_dir") or "."
#: Default ECsim/iPiC3D subpath under ECSIM_PREPEND (see --ecsim-files-path).
ECSIM_FILES = "iPiC3D-nathan"
ECSIM_DIAG = f"{DIAGNOSTICS_BASE}/iPiC3D-nathan"

DEFAULT_CAMPAIGN = "stability_campaign2"
#: Code variants some campaigns prefix their run directories with.
CODES = ("new", "old")

#: Checkpoint directory -> training recipe, as in stability2_scan.ipynb.
RECIPE = {
    "ablations_f2_serial": "serial",
    "production_ablations_f2_val0": "prodval0",
    "production_ablations_f2_step100_val0": "step100",
    "physics_f2_serial": "serial",
}

#: y half-width (d_i) of the band around the X point used for the ridge trace.
BAND_HALFWIDTH_DI = 0.3
#: half-width (d_i) of the "near X" window whose maximum defines the jet peak.
NEAR_WINDOW_DI = 1.25
#: x extent (d_i) of the two flanking background zones just outside the near-X
#: window: the sheet "plateau" the jet is contrasted against.
BG_WINDOW_DI = 2.0
#: x half-width (d_i) of the two small boxes centred on the near-X window edges
#: (the outflow crossings) that feed the primary p99/p50 intensification metric.
EDGE_BOX_HALFWIDTH_DI = 0.25
#: extra vertical half-extent (d_i) added to the ridge band to form those boxes.
#: 0.0 => the boxes span exactly the ridge band (y_X +- band-halfwidth), which
#: puts the aperture edges at the shoulder where the current jumps.
BOX_PAD_DI = 0.0


def _parse_pair(value):
    parts = [int(v) for v in str(value).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'start,end', got {value!r}")
    return parts[0], parts[1]


def _parse_list(value):
    return [v.strip() for v in str(value).split(",") if v.strip()]


def model_description(campaign, regime, run, runs_base=RUNS_BASE):
    """What distinguishes a run's checkpoint, '' if unavailable.

    The checkpoint a run was compiled against is recorded in its
    parameters.h (new code stores a relative model_path_rel, old code an
    absolute model_path), and the directories on the way to it ARE the
    training configuration:

      .../Lightning/<model-set>/<recipe>/<runs*>/<descriptor...>/checkpoints/
      best-epoch=..-val_loss=0.NNNN.pt

    Campaigns nest a different number of descriptors - stability_campaign2
    has one (ablate_noJnoE_P_deeper), physics_campaign_f2 two (the loss
    config and the ablation) - so everything between the model set and
    "checkpoints" is taken rather than a fixed offset. Dropped from it:

    * the ``runs``/``runs_MLP``/``runs_val0`` level, which is packaging
      rather than configuration;
    * any component the run name already states (physics_campaign_f2 names
      its run directories after the loss config), so the title does not say
      the same thing twice.

    Recipes are mapped through RECIPE and the ablation spelled the way
    stability2_scan.ipynb's row_label spells it, so a checkpoint is named
    identically in the panel titles and in the campaign heatmaps. The 4-digit
    number in a stability run key is that val_loss x 1e4, so it is not
    repeated here.
    """
    src = Path(runs_base) / campaign / regime / run / "src" / "parameters.h"
    try:
        text = src.read_text()
    except OSError:
        return ""
    m = (re.search(r'model_path_rel\s*=\s*"([^"]+\.pt)"', text)
         or re.search(r'\bmodel_path\s*=\s*"([^"]+\.pt)"', text))
    if not m:
        return ""
    parts = m.group(1).split("/")
    if "checkpoints" not in parts:
        return ""
    end = parts.index("checkpoints")
    # Start after Lightning/<model-set>/; without that anchor fall back to the
    # single-descriptor layout, which is what the absolute old-code paths and
    # everything before this campaign used.
    start = parts.index("Lightning") + 2 if "Lightning" in parts else end - 2
    fields = [p for p in parts[max(start, 0):end]
              if not p.startswith("runs") and p not in run]
    return "  ".join(RECIPE.get(p, p).removeprefix("ablate_").replace("_P_", "/")
                     for p in fields)


def campaign_runs(campaign, regime, recon=None, runs_base=RUNS_BASE):
    """Every run name of a (campaign, regime), sorted.

    The run directories are the authority on what was launched; the runs
    named in the reconnection CSV are the fallback when those are unreachable
    (diagnostics copied without the raw dumps).
    """
    d = Path(runs_base) / campaign / regime
    if d.is_dir():
        names = sorted(p.name for p in d.iterdir() if p.is_dir())
        if names:
            return names
    if recon is not None and "run" in recon:
        return sorted(str(r) for r in recon["run"].dropna().unique())
    return []


def resolve_code(campaign, regime, code, recon=None, runs_base=RUNS_BASE):
    """The ``<code>_`` prefix this campaign puts on its run directories, or ''.

    Campaigns differ in whether a run name carries a code variant at all:
    stability_campaign2 runs every checkpoint under both (new_FCNN_00172 /
    old_FCNN_00172), while physics_campaign_f2 names each directory after the
    model alone (FCNN_baseline_mse). Auto-detection keeps one command line
    working for both - an explicit --code is honoured, otherwise the prefix is
    used only if the campaign actually has prefixed runs.
    """
    if code and code.lower() != "none":
        return code
    names = campaign_runs(campaign, regime, recon, runs_base)
    if code:                       # explicit "none"
        return ""
    return next((c for c in CODES
                 if any(n.startswith(f"{c}_") for n in names)), "")


def run_name(code, model):
    """Run directory for a model - the code prefix only when there is one."""
    return f"{code}_{model}" if code else model


def discover_models(campaign, regime, code, recon=None, runs_base=RUNS_BASE):
    """Every model of this (campaign, regime, code), sorted.

    The campaign/regime folder is itself the list of what was run - no
    hand-maintained selection to drift out of sync with the campaign. Runs
    that were launched but have no usable dump/CSV row simply come back as
    blank slots in the figure, which is the point: a missing model stays
    visible. With an empty ``code`` the run names ARE the model keys.
    """
    prefix = f"{code}_" if code else ""
    return [n[len(prefix):]
            for n in campaign_runs(campaign, regime, recon, runs_base)
            if n.startswith(prefix)]


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--regime", default="R0", help="Single regime (R<n>)")
    parser.add_argument("--code", default=None, choices=["old", "new", "none"],
                        help="Code variant prefixing the run directories "
                        "(<code>_<model>). Default: detected from the campaign "
                        "- 'new' where runs carry the prefix "
                        "(stability_campaign2), none where the run directory "
                        "IS the model key (physics_campaign_f2). 'none' forces "
                        "the unprefixed form")
    parser.add_argument("--models", type=_parse_list, default=None,
                        help="Comma-separated checkpoint keys. Default: every "
                        "model found in the campaign/regime folder, i.e. every "
                        "<code>_* run directory")
    parser.add_argument("--fracs", default=[0.9],
                        type=lambda v: [float(f) for f in _parse_list(v)],
                        help="Fractions of each run's own peak recon rate")
    parser.add_argument("--field", default="Jz-tot",
                        help="Field for the ridge/map (Jz-tot = total current, "
                        "matching the compute_movie_field.py slide figures; "
                        "use Jz for the electron current)")
    parser.add_argument("--rate-column", default="recon_rate_norm")
    # Crop: the user's lower-sheet window; xpoint_ix/iy in the reconnection
    # CSVs are full-grid (menura) / analysis-crop (ecsim) indices, converted
    # to local ones by subtracting the crop origin.
    parser.add_argument("--choose-x", type=_parse_pair, default="0,256")
    parser.add_argument("--choose-y", type=_parse_pair, default="86,165")
    parser.add_argument("--band-halfwidth-di", type=float, default=BAND_HALFWIDTH_DI,
                        help="HALF-height (d_i) of the y-band, centred on the X point, "
                        "that the along-sheet ridge is traced through: for each x "
                        "column the profile takes J(x) = max|field| over "
                        "y_X +- this. Drawn as the GREEN DASHED box on the maps. "
                        "Set it to about the current-sheet half-thickness: too small "
                        "and a slightly mis-centred or warped sheet drops out of the "
                        "band (profile collapses toward zero, jet_z falls); too large "
                        "and off-sheet structure (separatrix arms, noise) leaks into "
                        "the max, inflating the background and its MAD so jet_z falls "
                        "again - noisy runs are the sensitive ones, clean runs are "
                        "flat over 0.25-1.5. Physical length, so identical framing "
                        "across resolutions. Does NOT affect jet_fwhm, which is "
                        "measured along x")
    parser.add_argument("--near-window-di", type=float, default=NEAR_WINDOW_DI,
                        help="HALF-width (d_i) in x, centred on the X point, of the "
                        "window searched for the jet peak: peak = max J(x) inside it, "
                        "and everything outside is background (see --bg-window-di). "
                        "Drawn as the ORANGE box on the maps. It sets what counts as "
                        "'at the X point': make it comfortably larger than the "
                        "anchor uncertainty (the reconnection-CSV X point can sit "
                        "~0.1-0.3 d_i off the unsmoothed saddle - the current 1.25 "
                        "absorbs that, which is why --local-xo changes nothing) but "
                        "smaller than the distance to neighbouring islands/X points, "
                        "or the peak of an adjacent structure is captured instead. "
                        "Too small also biases jet_z low by clipping the jet's own "
                        "shoulders into the background")
    parser.add_argument("--local-xo", action="store_true",
                        help="Re-identify the global X/O from unsmoothed Az on the "
                        "loaded crop instead of the reconnection-CSV anchors. "
                        "Default off: a same-load comparison showed identical "
                        "metrics for every jet-bearing run (the +-1.25 d_i window "
                        "absorbs the tracker's smoothing offset), and the local "
                        "root solve often finds no saddle on the crop anyway")
    parser.add_argument("--edge-box-halfwidth-di", type=float, default=EDGE_BOX_HALFWIDTH_DI,
                        help="HALF-width (d_i) in x of the two small boxes centred on "
                        "the near-X window edges (x_X +- near-window), i.e. on the "
                        "outflow crossings. Their union is the sample for the primary "
                        "intensification metric p99/p50: both the 'peak' (99th "
                        "percentile) and the 'background' (median) come from the SAME "
                        "vertical cutouts, so the ratio measures current "
                        "intensification as seen through a y cut, not against distant "
                        "x columns")
    parser.add_argument("--box-pad-di", type=float, default=BOX_PAD_DI,
                        help="Extra vertical half-extent (d_i) added to the ridge band "
                        "on each side to form the analysis boxes (drawn orange, and "
                        "shaded green in the x=const cut panel). Widens the y aperture "
                        "so the sampled cutout spans the sheet plus its shoulders")
    parser.add_argument("--bg-window-di", type=float, default=BG_WINDOW_DI,
                        help="x extent (d_i) of the flanking background zones just "
                        "outside the near-X window (the sheet plateau the jet is "
                        "contrasted against); 0 = use the whole rest of the profile")
    parser.add_argument("--ecsim-files-path", default=ECSIM_FILES,
                        help="ECsim/iPiC3D kinetic-reference data location. A relative "
                        "value is joined onto the paths.yaml data_dir prepend "
                        "(default %(default)r -> <data_dir>/iPiC3D-nathan); an absolute "
                        "path is used as-is")
    parser.add_argument("--ecsim-diagnostics", default=ECSIM_DIAG)
    parser.add_argument("--rect-halfthick-di", type=float, default=0.15,
                        help="Half-thickness (d_i) of the thin outflow rectangle")
    parser.add_argument("--rect-init-halflen-di", type=float, default=0.5,
                        help="Half-length (d_i) of the seed rectangle used for the "
                        "angle search")
    parser.add_argument("--rect-angle-range-deg", type=float, default=35.0,
                        help="Angle scan range (+-deg from the x axis) for the "
                        "outflow orientation")
    parser.add_argument("--outflow-gap-di", type=float, default=0.75,
                        help="Maximum sub-threshold stretch (d_i) the growing "
                        "rectangle may bridge (the outflow often dims right at the "
                        "X point or between detached lobes); the final length is "
                        "also EXTENDED by this amount to cover the fade-out")
    parser.add_argument("--plateau-pct", type=float, default=25.0,
                        help="Percentile of the per-segment means along the strip "
                        "used as the sheet-plateau baseline for the stop threshold "
                        "(low enough that a long outflow cannot contaminate it; "
                        "the median failed exactly there)")
    parser.add_argument("--outflow-stop-frac", type=float, default=0.5,
                        help="Growth stops when the newly added end segments' mean "
                        "|Jz| falls below this fraction of the seed-core mean - the "
                        "qualitative threshold defining the outflow extent")
    parser.add_argument("--no-separatrix", action="store_true",
                        help="Skip Az-level-set separatrix detection/overlay (the "
                        "white contour on the maps: the level set, scanned between "
                        "Az_X and the lobe value, across which median |Jz| jumps "
                        "most sharply - the physical sheet boundary)")
    parser.add_argument("--no-ecsim", action="store_true",
                        help="Skip the ECsim reference row")
    # Map rendering, defaults matching the compute_movie_field.py slide style.
    parser.add_argument("--map-cmap", default="viridis",
                        help="Colormap for the field maps (viridis: strong "
                        "negative current = dark on yellow; use viridis_r for "
                        "bright-on-dark, or rainbow_r for the movie-slide look)")
    parser.add_argument("--vmin", type=float, default=-0.2,
                        help="Fixed lower color limit for the maps")
    parser.add_argument("--vmax", type=float, default=0.0,
                        help="Fixed upper color limit for the maps")
    parser.add_argument("--auto-limits", action="store_true",
                        help="Ignore --vmin/--vmax and use the fields-CLI robust "
                        "symmetric limits + auto colormap instead")
    parser.add_argument("--interpolation", default="bilinear",
                        help="imshow interpolation for the maps")
    parser.add_argument("--per-row", type=int, default=3,
                        help="Cases (map + cut pair) per figure row; the whole "
                        "campaign is ~30 models, so widen this to keep the "
                        "montage from growing very tall")
    parser.add_argument("--output-dir", default=None,
                        help="Default: <diagnostics>/<campaign>/jet_metric")
    parser.add_argument("--dpi", type=int, default=130)
    return parser


def threshold_index(df_run, frac, rate_column):
    """First (time_index, time) at which rate reaches frac * own peak."""
    s = df_run.dropna(subset=[rate_column]).sort_values("time_index")
    if s.empty:
        return None
    peak = s[rate_column].max()
    if not np.isfinite(peak) or peak <= 0:
        return None
    row = s[s[rate_column] >= frac * peak].iloc[0]
    return int(row["time_index"]), float(row["time"])


def xpoint_at(df_run, ti, prefix="xpoint"):
    """Tracked X (or O, prefix="opoint") point (ix, iy) at dump ti, falling
    back to the nearest earlier dump with a finite point (the tracker can
    miss single frames)."""
    s = df_run.dropna(subset=[f"{prefix}_ix", f"{prefix}_iy"]).sort_values("time_index")
    s = s[s["time_index"] <= ti]
    if s.empty:
        return None
    row = s.iloc[-1]
    return int(row[f"{prefix}_ix"]), int(row[f"{prefix}_iy"])


def ridge_and_metrics(vals, x_axis, y_axis, ix_local, iy_local, *,
                      band_halfwidth_di, near_window_di, bg_window_di=0.0,
                      edge_box_halfwidth_di=EDGE_BOX_HALFWIDTH_DI,
                      box_pad_di=BOX_PAD_DI, inside_mask=None):
    """Along-sheet ridge profile + condensed jet metrics (see module doc).

    ``inside_mask`` (2D bool, the inside-separatrix region from
    find_separatrix) defines the APERTURE when given: the ridge takes each
    column's max |field| over inside pixels only, the edge-box pool for
    p99/p50 keeps inside pixels only, and aperture_h_di reports the local
    sheet thickness (mean inside-height over the edge-box columns). Every
    quantity then adapts to the run's actual sheet geometry. Without a mask
    (--no-separatrix, or detection failure) the fixed band(+pad) aperture is
    the fallback.

    With ``bg_window_di`` > 0 the ridge background statistics come from the
    two flanking zones just outside the near-X window (the sheet plateau).
    """
    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    hw = max(1, int(round(band_halfwidth_di / dy)))
    band = slice(max(0, iy_local - hw), min(vals.shape[1], iy_local + hw + 1))
    A = np.abs(vals)
    if inside_mask is not None:
        counts = inside_mask.sum(axis=1)
        J = np.where(counts > 0,
                     np.where(inside_mask, A, -np.inf).max(axis=1), np.nan)
    else:
        J = A[:, band].max(axis=1)

    icols = np.arange(len(J))
    xdist = np.abs(icols - ix_local) * dx
    near = xdist <= near_window_di
    if bg_window_di and bg_window_di > 0:
        rest = (xdist > near_window_di) & (xdist <= near_window_di + bg_window_di)
        if np.isfinite(J[rest]).sum() < 4:  # degenerate: fall back to all-rest
            rest = ~near
    else:
        rest = ~near
    bg = float(np.nanmedian(J[rest]))
    mad = float(np.nanmedian(np.abs(J[rest] - bg))) * 1.4826
    near_ok = near & np.isfinite(J)
    if not near_ok.any():
        near_ok = near
    peak_i = int(icols[near_ok][np.nanargmax(J[near_ok])])
    peak = float(J[peak_i])

    half = bg + 0.5 * (peak - bg)
    lo = hi = peak_i
    while lo > 0 and np.isfinite(J[lo - 1]) and J[lo - 1] > half:
        lo -= 1
    while hi < len(J) - 1 and np.isfinite(J[hi + 1]) and J[hi + 1] > half:
        hi += 1

    # Aperture y-extent within the near-X window (cut-panel shading and the
    # fallback drawing extents).
    nw = int(round(near_window_di / dx))
    pad_cells = max(0, int(round(box_pad_di / dy)))
    fb_lo = max(0, band.start - pad_cells)
    fb_hi = min(vals.shape[1], band.stop + pad_cells)
    if inside_mask is not None:
        nb_lo, nb_hi = max(0, ix_local - nw), min(vals.shape[0], ix_local + nw + 1)
        rows = np.flatnonzero(inside_mask[nb_lo:nb_hi].any(axis=0))
        box_y = (int(rows.min()), int(rows.max()) + 1) if rows.size else (fb_lo, fb_hi)
    else:
        box_y = (fb_lo, fb_hi)

    # p99/p50 of |field| within the GREEN BAND as it is actually PLOTTED:
    # the two x = const cut profiles (at x_X +- near-window, the blue/orange
    # verticals on the map), y restricted to the aperture extent. Pooling
    # the band rows over the full x range instead would drag p50 down with
    # the dim far-field stretches of those rows and inflate the ratio far
    # beyond the variation visible in the cut panel.
    ic_l = int(np.clip(ix_local - nw, 0, vals.shape[0] - 1))
    ic_r = int(np.clip(ix_local + nw, 0, vals.shape[0] - 1))
    band_sample = A[[ic_l, ic_r], box_y[0]:box_y[1]]
    b99, b50 = np.nanpercentile(band_sample, 99), np.nanpercentile(band_sample, 50)

    valid = np.isfinite(J)
    metrics = dict(
        band_p99p50=float(b99 / b50) if b50 > 0 else np.inf,
        box_y=box_y,
        jet_z=(peak - bg) / mad if mad > 0 else np.inf,
        jet_contrast=peak / bg if bg > 0 else np.inf,
        jet_fwhm=(hi - lo + 1) * dx,
        participation=float(np.nansum(J) ** 2
                            / (valid.sum() * np.nansum(np.square(J[valid])))),
        peak=peak, background=bg, mad=mad, peak_i=peak_i,
        band_lo_i=band.start, band_hi_i=band.stop - 1,
        bg_mask=rest,
    )
    return J, near, metrics

def find_global_xo(data, x_axis, y_axis):
    """Global X and O point from UNSMOOTHED Az on the loaded crop.

    Rationale: the reconnection CSV's X point comes from az-sigma-4 smoothed
    Az, which is displaced by several cells exactly when a one-sided jet
    makes the X region asymmetric (and during plasmoid birth the sheet holds
    an X-O-X triplet, where the smoothed saddle lands between the true
    saddles). Recomputing Az on the crop and taking the GLOBAL X (max-Az
    saddle, the tracker's own convention) and GLOBAL O (min-Az extremum)
    gives convention-free, unsmoothed anchors. Falls back to the CSV point
    when no saddle is found in the crop.
    """
    from closure import plasma

    if "Az" not in data:
        plasma.get_Az(x_axis, y_axis, data)
    az = np.asarray(data["Az"])[..., 0]
    o_pts, x_pts = plasma.find_xo_points(az, x_axis, y_axis, seed_grad_frac=0.05)
    xp = max(x_pts, key=lambda p: p["value"]) if x_pts else None
    op = min(o_pts, key=lambda p: p["value"]) if o_pts else None
    return xp, op


def find_separatrix(data, vals, x_axis, y_axis, ixl, iyl, *,
                    scan_band_di=1.5, nlevels=80):
    """Separatrix as an Az level set, chosen by the Jz jump across it.

    Az is a stream function, so its level sets are field lines; the
    separatrix is the line through the X point (Az = Az_X). At finite
    resolution the sharpest inside/outside current contrast can sit a level
    or two away, so instead of hard-coding Az_X we scan levels between Az_X
    and the lobe value and keep the one maximising |d median|Jz| / d level| -
    i.e. the level set across which the current jumps from its baseline
    (exactly the "J starts increasing" point seen in the x=const cuts).
    Returns dict(level, az 2D slice, azX, (levels, medJ) profile) or None.
    """
    from closure import plasma

    if "Az" not in data:
        plasma.get_Az(x_axis, y_axis, data)
    az = np.asarray(data["Az"])[..., 0]
    J = np.abs(np.asarray(vals))
    azX = float(az[ixl, iyl])
    dy = float(y_axis[1] - y_axis[0])
    hw = max(2, int(round(scan_band_di / dy)))
    sl = slice(max(0, iyl - hw), min(az.shape[1], iyl + hw + 1))
    az_loc, J_loc = az[:, sl], J[:, sl]
    # Lobe reference: outermost rows of the crop, far from the sheet.
    az_lobe = float(np.median(np.concatenate([az[:, :3].ravel(), az[:, -3:].ravel()])))
    lo, hi = sorted((azX, az_lobe))
    if not np.isfinite(lo) or hi - lo <= 0:
        return None
    levels = np.linspace(lo, hi, nlevels + 2)[1:-1]
    dL = levels[1] - levels[0]
    med = np.array([
        np.median(J_loc[np.abs(az_loc - L) < dL]) if np.any(np.abs(az_loc - L) < dL)
        else np.nan
        for L in levels
    ])
    if np.all(~np.isfinite(med)):
        return None
    grad = np.abs(np.gradient(med, levels))
    k = int(np.nanargmax(grad))
    # Inside-separatrix mask (the aperture): pixels on the Az_X side of the
    # chosen level, restricted to the scan band so far-away structure with
    # coincidentally similar Az cannot leak in.
    side = 1.0 if azX >= az_lobe else -1.0
    inside = np.zeros(az.shape, dtype=bool)
    inside[:, sl] = side * (az[:, sl] - levels[k]) > 0
    return dict(level=float(levels[k]), az=az, azX=azX, az_lobe=az_lobe,
                inside=inside, profile=(levels, med))


def outflow_rectangle(vals, x_axis, y_axis, ixl, iyl, *,
                      halfthick_di=0.15, init_halflen_di=0.5,
                      angle_range_deg=35.0, angle_step_deg=1.0,
                      grow_step_di=0.1, stop_frac=0.5, gap_di=0.75,
                      plateau_pct=25.0):
    """Thin rotated rectangle characterising the magnetized outflow.

    Centred on the X point. Angle and length are optimized JOINTLY: for
    every candidate angle the growth below is run to completion, and the
    winning angle is the one whose intensified extent is LONGEST (tie-break:
    larger seed-core mean) - the outflow direction is by definition the
    direction the intensification extends farthest along.

    Growth at a given angle: start from the seed half-length; keep extending
    while the NEWLY ADDED end segments' mean |Jz| stays above
    baseline + stop_frac * (core - baseline), where baseline is the strip's
    own median (its plateau). Sub-threshold stretches up to gap_di are
    bridged (the outflow often dims at the X point or between detached
    lobes); the last intensified position is then EXTENDED by gap_di so the
    rectangle covers the fade-out rather than clipping at the last bright
    segment.

    Returns dict(angle_deg, halflen_di, outflow_len_di, mean_core, baseline,
    mask (2D bool, the rectangle interior), corners) or None when degenerate.
    """
    A = np.abs(np.asarray(vals))
    x0, y0 = float(x_axis[ixl]), float(y_axis[iyl])
    XX, YY = np.meshgrid(x_axis - x0, y_axis - y0, indexing="ij")
    max_L = 0.5 * float(x_axis[-1] - x_axis[0])
    angles = np.arange(-angle_range_deg, angle_range_deg + 1e-9, angle_step_deg)

    # Pass 1 - a COMMON threshold for every angle. Grading each angle against
    # its own seed would let a tilted direction that starts on a weaker core
    # set itself a lower bar and "win" on length by marching through mediocre
    # current, while the direction along the bright band stops at its first
    # dip. Reference core = the best seed over all angles (the actual jet
    # core); baseline = that direction's strip median (the plateau).
    frames = []
    core_ref, baseline_ref = -np.inf, np.nan
    for th in angles:
        c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
        u = c * XX + s * YY
        v = -s * XX + c * YY
        thick = np.abs(v) <= halfthick_di
        seed = thick & (np.abs(u) <= init_halflen_di)
        if seed.sum() < 4:
            continue
        seed_mean = float(A[seed].mean())
        frames.append((float(th), u, v, thick, seed_mean))
        core_ref = max(core_ref, seed_mean)
    if not frames:
        return None
    # Baseline = the SHEET PLATEAU, estimated as a low percentile of the
    # per-segment means along the best-core strip. The jet is intensification
    # above the rest of the SHEET (not above the near-zero lobe, which lets
    # ordinary sheet current pass and runs every rectangle to the crop edge),
    # but a strip MEDIAN gets contaminated whenever the outflow itself is
    # long, truncating exactly the best cases - a p25 stays on the plateau
    # for outflows covering up to ~75% of the strip.
    th0, u0, v0, thick0, _ = max(frames, key=lambda f: f[4])
    edges = np.arange(init_halflen_di, max_L, grow_step_di)
    seg_means = []
    for lo_e in edges:
        seg = thick0 & (np.abs(u0) > lo_e) & (np.abs(u0) <= lo_e + grow_step_di)
        if seg.sum() >= 2:
            seg_means.append(float(A[seg].mean()))
    if len(seg_means) < 4:
        return None
    baseline_ref = float(np.percentile(seg_means, plateau_pct))
    threshold = baseline_ref + stop_frac * max(core_ref - baseline_ref, 0.0)

    # Pass 2 - grow every angle against that same threshold; the outflow
    # direction is the one that stays intensified the longest.
    best = None
    for th, u, v, thick, seed_mean in frames:
        L = last_good = init_halflen_di
        gap = 0.0
        while L + grow_step_di <= max_L:
            seg = thick & (np.abs(u) > L) & (np.abs(u) <= L + grow_step_di)
            if seg.sum() < 2:
                break
            L += grow_step_di
            if float(A[seg].mean()) >= threshold:
                last_good, gap = L, 0.0
            else:
                gap += grow_step_di
                if gap > gap_di:
                    break
        score = (last_good, seed_mean)
        if best is None or score > best[0]:
            best = (score, th, u, v, seed_mean)
    (last_good, core_mean), th, u, v, _ = best
    baseline = baseline_ref
    Lf = min(last_good + gap_di, max_L)  # cover the fade-out
    thick = np.abs(v) <= halfthick_di
    final = thick & (np.abs(u) <= Lf)
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    e1, e2 = np.array([c, s]), np.array([-s, c])
    ctr = np.array([x0, y0])
    corners = [ctr + su * Lf * e1 + sv * halfthick_di * e2
               for su, sv in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
    return dict(angle_deg=float(th), halflen_di=float(Lf),
                outflow_len_di=2 * float(Lf), mean_core=core_mean,
                baseline=baseline, mask=final, corners=np.array(corners))

def load_case(experiment, files_path, ti, args, **loader_kwargs):
    """One cropped dump; returns (vals_2d, x_axis, y_axis, label, data).
    
    Parameters
    ----------
    experiment : str
        Experiment identifier.
    files_path : str
        Path to the experiment files.
    ti : int
        Time index to load.
    args : object
        Arguments object with attributes: field, choose_x, choose_y.
    **loader_kwargs : dict
        Additional keyword arguments for load_experiment_data.
    
    Returns
    -------
    vals_2d : ndarray
        2D field values.
    x_axis : ndarray
        X-axis coordinates.
    y_axis : ndarray
        Y-axis coordinates.
    label : str
        Field label (name and species if applicable).
    data : dict
        Full data dictionary from load_experiment_data.
    """
    data, X, Y, qom, times = load_experiment_data(
        experiment, files_path,
        choose_times=str(ti),
        choose_x=args.choose_x, choose_y=args.choose_y,
        **loader_kwargs,
    )
    spec = parse_field_specs(args.field)[0]
    if spec.name not in data and spec.name.endswith("-tot"):
        _compute_current_totals(data)  # ECsim loads don't precompute J*-tot
    arr, species = resolve_field_data(data, spec)
    vals = arr[..., 0] if arr.ndim == 3 else arr
    x_axis, y_axis = _as_axis(X, Y)
    label = spec.name if species is None else f"{spec.name}_{species}"
    return np.asarray(vals, dtype=float), x_axis, y_axis, label, data


class CaseError(Exception):
    """A case (run x threshold) that cannot be built - reported, not fatal."""


def blank_case(label, reason):
    """Placeholder tuple for a case that failed to build.

    A failed case KEEPS its slot instead of being dropped: panels are filled
    column-major from the fixed (ECsim, models) x fracs list and nrows follows
    len(cases), so dropping one case renumbers every panel after it and
    reshapes the whole grid. The same model would then sit in a different
    position in each run's figure, which defeats run-by-run comparison.
    """
    return (f"{label}\n[no panel: {reason}]",) + (None,) * 9


def is_blank(case):
    return case[1] is None                       # J is None only for placeholders


def case_column(case):
    """The case label as the metrics table prints it: the title flattened to
    one line, minus - for a failed case - the reason, which is printed after
    the numeric columns instead so it cannot widen them."""
    lines = case[0].split("\n")
    return lines[0] if is_blank(case) else " ".join(lines)


def failure_reason(exc):
    """CaseError carries its own message; anything else shows its type too."""
    return str(exc) if isinstance(exc, CaseError) else f"{type(exc).__name__}: {exc}"


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Relative --ecsim-files-path is taken under the paths.yaml data_dir prepend;
    # an absolute path is honoured as given.
    if not os.path.isabs(args.ecsim_files_path):
        args.ecsim_files_path = os.path.join(ECSIM_PREPEND, args.ecsim_files_path)
    R = args.regime
    diag = f"{DIAGNOSTICS_BASE}/{args.campaign}"
    out_dir = Path(args.output_dir or f"{diag}/jet_metric")
    out_dir.mkdir(parents=True, exist_ok=True)

    recon = pd.read_csv(f"{diag}/{R}/reconnection_menura.csv")

    code = resolve_code(args.campaign, R, args.code, recon=recon)
    tag = code or "all"                          # figure name / titles
    models = args.models
    if models is None:
        models = discover_models(args.campaign, R, code, recon=recon)
        if not models:
            want = f"{code}_* runs" if code else "run directories"
            raise SystemExit(
                f"No {want} found for {args.campaign}/{R} under {RUNS_BASE} or "
                f"in the reconnection CSV; pass --models explicitly")
        print(f"{len(models)} models discovered in {args.campaign}/{R}"
              f"{f' ({code})' if code else ''}: {', '.join(models)}")

    # ---- assemble cases: ECsim kinetic reference first, then the models ----
    cases = []  # (label, J, near, metrics, vals, extent, marks)
    ecsim_J_norm = None
    if not args.no_ecsim:
        n = R.removeprefix("R")
        ec_run = f"Le2DHGEM_RunID_{n}_f2"
        ec_recon = pd.read_csv(f"{args.ecsim_diagnostics}/{R}/reconnection_ecsim.csv")
        for frac in args.fracs:
            pct = int(frac * 100)
            try:
                sub_df = ec_recon[ec_recon["run"] == ec_run]
                th = threshold_index(sub_df, frac, args.rate_column)
                xp = xpoint_at(sub_df, th[0]) if th else None
                if th is None or xp is None:
                    raise CaseError("no usable threshold/X point")
                vals, x_axis, y_axis, flabel, dcase = load_case(
                    ec_run, args.ecsim_files_path, th[0], args,
                    backend="ecsim", choose_species=["e", "i", "e", "i"],
                    normalization="alfven-infer",
                )
                ixl, iyl = xp[0] - args.choose_x[0], xp[1] - args.choose_y[0]
                o_mark = None
                if args.local_xo:
                    xp_l, op_l = find_global_xo(dcase, x_axis, y_axis)
                    if xp_l is not None:
                        ixl, iyl = int(xp_l["ix"]), int(xp_l["iy"])
                    if op_l is not None:
                        o_mark = (int(op_l["ix"]), int(op_l["iy"]))
                else:  # default: the tracker's own rate-defining O point, crop-converted
                    op_csv = xpoint_at(sub_df, th[0], prefix="opoint")
                    if op_csv is not None:
                        o_mark = (op_csv[0] - args.choose_x[0], op_csv[1] - args.choose_y[0])
                        if not (0 <= o_mark[0] < vals.shape[0]):
                            o_mark = None
                sep = None if args.no_separatrix else find_separatrix(
                    dcase, vals, x_axis, y_axis, ixl, iyl)
                J, near, m = ridge_and_metrics(
                    vals, x_axis, y_axis, ixl, iyl,
                    band_halfwidth_di=args.band_halfwidth_di,
                    near_window_di=args.near_window_di,
                    bg_window_di=args.bg_window_di,
                    edge_box_halfwidth_di=args.edge_box_halfwidth_di,
                    box_pad_di=args.box_pad_di,
                    inside_mask=None if sep is None else sep["inside"],
                )
                rect = outflow_rectangle(
                    vals, x_axis, y_axis, ixl, iyl,
                    halfthick_di=args.rect_halfthick_di,
                    init_halflen_di=args.rect_init_halflen_di,
                    angle_range_deg=args.rect_angle_range_deg,
                    stop_frac=args.outflow_stop_frac,
                    gap_di=args.outflow_gap_di,
                    plateau_pct=args.plateau_pct)
                # PRIMARY metric: mean |Jz| inside the rectangle over mean |Jz|
                # in the rest of the separatrix interior (between separatrix and
                # rectangle, bounded in x by the domain edges).
                if rect is not None:
                    rect["jz_ratio"] = np.nan
                    if sep is not None:
                        A_case = np.abs(vals)
                        sheet = sep["inside"] & ~rect["mask"]
                        if rect["mask"].any() and sheet.any():
                            rect["jz_ratio"] = float(A_case[rect["mask"]].mean()
                                                     / A_case[sheet].mean())
                cases.append((f"ECsim {ec_run}\n{flabel}, t={th[1]:.3g} ({pct}%)",
                              J, near, m, vals, (x_axis, y_axis), (ixl, iyl), o_mark, sep, rect))
                if ecsim_J_norm is None:
                    ecsim_J_norm = J / m["background"]
            except Exception as exc:   # keep the slot; report on stdout
                reason = failure_reason(exc)
                print(f"ECsim {ec_run} ({pct}%): {reason}")
                cases.append(blank_case(f"ECsim {ec_run}", f"{pct}%: {reason}"))

    for model in models:
        run = run_name(code, model)
        # Training recipe + ablation variant (see model_description), on the
        # SAME line as the run so the field/time stays the second line. The
        # widest campaign label measures 2.60 in at fontsize 8 against a
        # ~2.65 in panel, so it fits; empty for a run whose parameters.h is
        # gone, and the title is then just the run.
        desc = model_description(args.campaign, R, run)
        head = f"{run}  {desc}" if desc else run
        sub_df = recon[recon["run"] == run]
        for frac in args.fracs:
            pct = int(frac * 100)
            try:
                th = threshold_index(sub_df, frac, args.rate_column)
                xp = xpoint_at(sub_df, th[0]) if th else None
                if th is None or xp is None:
                    raise CaseError("no usable threshold/X point")
                vals, x_axis, y_axis, flabel, dcase = load_case(
                    run, f"{RUNS_BASE}/{args.campaign}/{R}", th[0], args,
                    backend="menura", choose_species=["e", "i"],
                    normalization="none",
                )
                ixl, iyl = xp[0] - args.choose_x[0], xp[1] - args.choose_y[0]
                if not (0 <= ixl < vals.shape[0] and 0 <= iyl < vals.shape[1]):
                    raise CaseError(f"X point {xp} outside the crop")
                o_mark = None
                if args.local_xo:
                    xp_l, op_l = find_global_xo(dcase, x_axis, y_axis)
                    if xp_l is not None:
                        ixl, iyl = int(xp_l["ix"]), int(xp_l["iy"])
                    if op_l is not None:
                        o_mark = (int(op_l["ix"]), int(op_l["iy"]))
                else:  # default: the tracker's own rate-defining O point, crop-converted
                    op_csv = xpoint_at(sub_df, th[0], prefix="opoint")
                    if op_csv is not None:
                        o_mark = (op_csv[0] - args.choose_x[0], op_csv[1] - args.choose_y[0])
                        if not (0 <= o_mark[0] < vals.shape[0]):
                            o_mark = None
                sep = None if args.no_separatrix else find_separatrix(
                    dcase, vals, x_axis, y_axis, ixl, iyl)
                J, near, m = ridge_and_metrics(
                    vals, x_axis, y_axis, ixl, iyl,
                    band_halfwidth_di=args.band_halfwidth_di,
                    near_window_di=args.near_window_di,
                    bg_window_di=args.bg_window_di,
                    edge_box_halfwidth_di=args.edge_box_halfwidth_di,
                    box_pad_di=args.box_pad_di,
                    inside_mask=None if sep is None else sep["inside"],
                )
                rect = outflow_rectangle(
                    vals, x_axis, y_axis, ixl, iyl,
                    halfthick_di=args.rect_halfthick_di,
                    init_halflen_di=args.rect_init_halflen_di,
                    angle_range_deg=args.rect_angle_range_deg,
                    stop_frac=args.outflow_stop_frac,
                    gap_di=args.outflow_gap_di,
                    plateau_pct=args.plateau_pct)
                # PRIMARY metric: mean |Jz| inside the rectangle over mean |Jz|
                # in the rest of the separatrix interior (between separatrix and
                # rectangle, bounded in x by the domain edges).
                if rect is not None:
                    rect["jz_ratio"] = np.nan
                    if sep is not None:
                        A_case = np.abs(vals)
                        sheet = sep["inside"] & ~rect["mask"]
                        if rect["mask"].any() and sheet.any():
                            rect["jz_ratio"] = float(A_case[rect["mask"]].mean()
                                                     / A_case[sheet].mean())
                cases.append((f"{head}\n{flabel}, t={th[1]:.3g} ({pct}%)",
                              J, near, m, vals, (x_axis, y_axis), (ixl, iyl), o_mark, sep, rect))
            except Exception as exc:   # keep the slot; report on stdout
                reason = failure_reason(exc)
                print(f"{run} ({pct}%): {reason}")
                cases.append(blank_case(head, f"{pct}%: {reason}"))

    if all(is_blank(c) for c in cases):
        raise SystemExit("No cases could be built")

    # ---- metrics table ----
    # Case column sized to the longest label actually present - run +
    # checkpoint description + field/time - so the numeric columns stay
    # aligned whatever a campaign names its runs.
    W = max(len(case_column(c)) for c in cases)
    print(f"\n{'case':<{W}} {'jz_ratio':>8} {'p99/p50':>8} {'angle':>7} {'L_out_di':>9}")
    for case in cases:
        flat = case_column(case)
        if is_blank(case):   # reason after the columns, so the table stays aligned
            why = case[0].split("\n")[-1]
            print(f"{flat:<{W}} {'-':>8} {'-':>8} {'-':>7} {'-':>9}  {why}")
            continue
        label, J, near, m, *rest_fields = case
        rect_c = rest_fields[-1]
        if rect_c is None:
            print(f"{flat:<{W}} {'-':>8} {m['band_p99p50']:>8.2f} {'-':>7} {'-':>9}")
        else:
            print(f"{flat:<{W}} {rect_c['jz_ratio']:>8.2f} {m['band_p99p50']:>8.2f} "
                  f"{rect_c['angle_deg']:>+6.1f}° {rect_c['outflow_len_di']:>9.2f}")

    # ---- figure: --per-row cases per row, each case = [field map | cut
    # profile], i.e. a 2*per_row-column grid (histograms dropped). Row height
    # is set from the map's own data aspect so the aspect='equal' maps fill
    # their cells with no vertical whitespace.
    per_row = max(1, min(args.per_row, len(cases)))
    ncases = len(cases)
    nrows = int(np.ceil(ncases / per_row))
    # Horizontal budget. The map is aspect='equal', so its drawn width is fixed
    # by the crop geometry - cell width beyond that just becomes margin around
    # it. The cut is an ordinary line plot with no aspect constraint, so every
    # spare inch goes to IT: it gets the LARGER width ratio, and the gutter is
    # cut to only what text needs. WSPACE is set by the one gutter that carries
    # any: the map's colorbar tick labels overhang into it and the cut's y
    # labels reach back, so below ~0.28 "-0.10" and "0.1" collide. (Was 1.2 /
    # 0.42 - the cut sat narrower than the map with ~1 in of dead space beside
    # every panel; it is ~40% wider now at the same drawn height.)
    MAP_R, CUT_R, WSPACE = 1.6, 1.75, 0.30
    LEFT, RIGHT = 0.04, 0.98                 # also used in subplots_adjust below
    width_ratios = [MAP_R, CUT_R] * per_row
    ncols = 2 * per_row
    fig_w = 7.5 * per_row        # per-case width; the map below lands at the
                                 # size it had before the cut was widened
    x0a, y0a = next(c for c in cases if not is_blank(c))[5]   # blanks carry no axes
    aspect = float((y0a[-1] - y0a[0]) / (x0a[-1] - x0a[0]))
    # Drawn map width, exactly: the horizontal span less the wspace gutters,
    # shared out by width_ratios. row_h follows from it, so an approximation
    # here would show up as vertical whitespace or clipped maps.
    axes_w = (RIGHT - LEFT) * fig_w / (1 + WSPACE * (ncols - 1) / ncols)
    map_w = axes_w * MAP_R / sum(width_ratios)
    map_h = map_w * aspect                   # drawn height of an equal-aspect map
    # Vertical budget, in inches, derived the same way. ROW_GAP is what the
    # text BETWEEN two stacked rows measures: the upper map's x tick labels
    # (0.152) + the lower map's title, plus clearance. The x LABEL is not in
    # that sum - it would add another 0.12 and is why the rows used to
    # collide, so it is drawn only under the last case of each column (as the
    # tick comment below always claimed).
    # Title height at fontsize 8 / pad 3 is linear in line count (measured:
    # 0.122 in for one line, +0.125 per line after). Deriving it from the
    # ACTUAL longest title keeps the rows clear as titles change - they went
    # from two lines to three when the checkpoint description was added.
    title_lines = max(len(c[0].split("\n")) for c in cases if not is_blank(c))
    title_h = 0.122 + 0.125 * (title_lines - 1)
    ROW_GAP = 0.152 + title_h + 0.08
    # TOP_PAD clears the two-line suptitle (~0.49 in) plus the first row's
    # title; BOT_PAD holds the bottom row's tick labels and x label.
    TOP_PAD, BOT_PAD = 0.49 + title_h, 0.36
    # Solve fig_h so the maps keep their full drawn height AND the gaps stay
    # ROW_GAP: (fig_h - pads) = nrows*map_h + (nrows-1)*ROW_GAP. Feeding a
    # bigger hspace to a fixed row height would instead make the maps
    # height-limited, shrinking them back out of the width they just gained.
    row_h = map_h + ROW_GAP * (nrows - 1) / nrows + (TOP_PAD + BOT_PAD) / nrows
    HSPACE = ROW_GAP / map_h
    # The map is aspect='equal' (box h/w = y_span/x_span = aspect), so it draws
    # short and wide. Force the neighbouring cut panel to the SAME drawn height
    # by giving it a matching box aspect, scaled by the width ratio between the
    # two cells - otherwise the cut fills its full cell and dwarfs the map,
    # leaving whitespace around every map. The 0.92 keeps the cut just inside
    # the map's height so its title clears the row above.
    cut_box_aspect = aspect * (MAP_R / CUT_R) * 0.92
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, row_h * nrows),
                             squeeze=False,
                             gridspec_kw=dict(width_ratios=width_ratios,
                                              wspace=WSPACE, hspace=HSPACE))
    # Column-major (vertical-first) placement: consecutive cases fill a column
    # top-to-bottom, so a model's own thresholds (listed consecutively, e.g.
    # 75% then 90%) stay stacked in the same column instead of being split
    # across a row boundary.
    def cell(idx):
        return idx % nrows, idx // nrows      # (row, case-column)

    # The x label hangs under the lowest DRAWN case of each column - not the
    # bottom row, whose slot may be a failed case with no axes to carry it.
    xlabel_at = {k // nrows: k for k, c in enumerate(cases) if not is_blank(c)}
    xlabel_at = set(xlabel_at.values())        # last non-blank per column wins

    for k in range(ncases, nrows * per_row):
        rr0, cp0 = cell(k)
        axes[rr0][2 * cp0].axis("off")
        axes[rr0][2 * cp0 + 1].axis("off")
    for r, case in enumerate(cases):
        rr, cp = cell(r)
        if is_blank(case):
            # Failed case: its cell pair stays blank (only the label, so it is
            # obvious WHICH model dropped out) and every other panel keeps the
            # position it has in the runs where this case succeeds.
            for ax in (axes[rr][2 * cp], axes[rr][2 * cp + 1]):
                ax.axis("off")
            axes[rr][2 * cp].text(0.5, 0.5, case[0], transform=axes[rr][2 * cp].transAxes,
                                  ha="center", va="center", fontsize=8, color="0.4")
            continue
        label, J, near, m, vals, (x_axis, y_axis), (ixl, iyl), o_mark, sep, rect = case
        ax = axes[rr][2 * cp]
        # Movie-style rendering (compute_movie_field.py slides): raw signed
        # field, fixed limits, rainbow_r, bilinear imshow - identical color
        # scale across all rows so amplitudes compare directly. --auto-limits
        # switches back to the fields-CLI robust symmetric scaling.
        if args.auto_limits:
            vplot = _panel_values(vals)
            vmin, vmax = _field_limits(vplot)
            cmap = get_cmap(parse_field_specs(args.field)[0].name)
        else:
            vplot, vmin, vmax, cmap = vals, args.vmin, args.vmax, args.map_cmap
        im = ax.imshow(vplot.T, origin="lower", interpolation=args.interpolation,
                       extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]),
                       vmin=vmin, vmax=vmax, cmap=cmap)
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05)
        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=6)
        ax.plot(x_axis[ixl], y_axis[iyl], "rx", ms=9, mew=2)
        if o_mark is not None and 0 <= o_mark[1] < vals.shape[1]:
            ax.plot(x_axis[o_mark[0]], y_axis[o_mark[1]], "o", ms=9,
                    markerfacecolor="white", markeredgecolor="black", mew=1.8)
        if sep is not None:
            ax.contour(x_axis, y_axis, sep["az"].T, levels=[sep["level"]],
                       colors="white", linewidths=0.9)
        # Where the x = const cut curves (right panel) are taken; line styles
        # and colors match the curves there.
        dxm = float(x_axis[1] - x_axis[0])
        offm = max(1, int(round(args.near_window_di / dxm)))
        for sgn, color, ls in ((-1, "tab:blue", "--"), (+1, "tab:orange", "-")):
            icm = int(np.clip(ixl + sgn * offm, 0, vals.shape[0] - 1))
            ax.axvline(x_axis[icm], color=color, linestyle=ls, linewidth=1.0,
                       alpha=0.85)
        ax.plot(x_axis[m["peak_i"]], y_axis[iyl], "o", ms=7,
                markerfacecolor="none", markeredgecolor="orange", mew=2)
        # Fallback aperture only: with a separatrix the white contour IS the
        # aperture boundary, so the fixed green dashed band would mislead.
        y_lo, y_hi = y_axis[m["band_lo_i"]], y_axis[m["band_hi_i"]]
        if sep is None:
            ax.add_patch(plt.Rectangle((x_axis[0], y_lo), x_axis[-1] - x_axis[0],
                                       y_hi - y_lo, fill=False, edgecolor="tab:green",
                                       linestyle="--", linewidth=1.3))
        # Orange DOTTED rectangle: the outflow rectangle - centred on the X
        # point, rotated to the angle maximising mean |Jz|, grown along that
        # direction until the current intensification ends. Its length IS the
        # outflow extent; nothing else is boxed any more.
        if rect is not None:
            ax.add_patch(plt.Polygon(rect["corners"], closed=True, fill=False,
                                     edgecolor="orange", linestyle=":",
                                     linewidth=1.6))
        # Pin the view to the data. The rectangle is centred on the X point and
        # may reach past the domain edge when that point is off-centre; letting
        # the patch autoscale the axes stretches the map, which then no longer
        # fills its cell (blank strip between the map and its colorbar).
        ax.set_xlim(x_axis[0], x_axis[-1])
        ax.set_ylim(y_axis[0], y_axis[-1])
        ax.set_aspect("equal")
        # Run/case identity on TOP of the map (title), not the y axis.
        ax.set_title(label, fontsize=8, pad=3)
        # Ticks: x outside (below) where there is room; y INSIDE in white, so
        # the short map keeps its full height and the labels stay readable on
        # the dark end of the colormap. Axis labels follow the same rule: the
        # x label sits under the bottom row only, the y label rides inside.
        ax.tick_params(axis="x", direction="out", length=3, width=0.8,
                       labelsize=6, pad=1.5)
        ax.tick_params(axis="y", direction="in", length=3, width=0.8,
                       colors="white", labelsize=6, pad=-14)
        for lbl in ax.get_yticklabels():
            lbl.set_color("white")
            lbl.set_horizontalalignment("left")
        # Only under the LAST case of this column: everywhere else the label
        # would sit in the gap between two rows and collide with the title of
        # the map below (the tick labels alone already fit there).
        if r in xlabel_at:
            ax.set_xlabel("x [d_i]", fontsize=7, labelpad=1)
        ax.text(0.098, 0.5, "y [d_i]", transform=ax.transAxes, rotation=90,
                ha="left", va="center", fontsize=7, color="white")

        ax = axes[rr][2 * cp + 1]
        # x = const cuts across y at the two near-X box edges: where the
        # magnetized reconnection outflow crosses. Signed field, so the
        # outflow current structure (sheet cross-section + jet) is visible.
        dx = float(x_axis[1] - x_axis[0])
        off = max(1, int(round(args.near_window_di / dx)))
        for sgn, style, lab in [(-1, "--", "left edge (x_X - w)"),
                                (+1, "-", "right edge (x_X + w)")]:
            ic = int(np.clip(ixl + sgn * off, 0, vals.shape[0] - 1))
            ax.plot(y_axis, vals[ic, :], style, lw=1.2, label=lab)
            if sep is not None:  # separatrix crossings of this cut: where the
                azc = sep["az"][ic, :] - sep["level"]  # current leaves baseline
                for j in np.flatnonzero(np.diff(np.sign(azc)) != 0):
                    ax.axvline(y_axis[j], color="0.45", linestyle=":", linewidth=0.8)
        ax.axvspan(y_axis[m["box_y"][0]], y_axis[m["box_y"][1] - 1],
                   color="tab:green", alpha=0.12)
        ax.axvline(y_axis[iyl], color="red", lw=0.9, alpha=0.6)
        rtxt = ("no rect" if rect is None else
                f"jz_ratio={rect['jz_ratio']:.2f}  θ={rect['angle_deg']:+.1f}°  "
                f"L_out={rect['outflow_len_di']:.2f} d_i")
        ax.set_title(f"{rtxt}  p99/p50={m['band_p99p50']:.2f}", fontsize=9)
        ax.set_xlim(y_axis[0], y_axis[-1])
        ax.set_box_aspect(cut_box_aspect)   # match the map's drawn height
        ax.legend(fontsize=6, frameon=False)
        ax.tick_params(labelsize=7)

    # Absolute-inch margins: the default subplots margins (top=0.88 etc.) are
    # a fixed FRACTION, so on a tall multi-row figure they become inches of
    # blank at top/bottom. Pin them in inches so only the content remains.
    fig_h = row_h * nrows
    fig.subplots_adjust(left=LEFT, right=RIGHT,
                        top=1 - TOP_PAD / fig_h,   # suptitle (2 lines)
                        bottom=BOT_PAD / fig_h,
                        wspace=WSPACE, hspace=HSPACE)
    fig.suptitle(
        f"{args.campaign} {R}/{tag}: lower-sheet jet metric, field {args.field}, "
        f"thresholds {args.fracs} of peak {args.rate_column}\n"
        "each case = [Jz map | Jz(y) cut at the x=const outflow crossings]; "
        "map: red x=X, white o=O, white=separatrix, dotted orange=outflow rectangle",
        fontsize=12, y=1 - 0.12 / fig_h,
    )
    out = out_dir / f"{R}_{tag}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
