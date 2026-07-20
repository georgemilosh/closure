#!/usr/bin/env python
"""PRELIMINARY jet-significance metric for the lower-sheet X point.

Quantifies the "jet-like" localized intensification of the out-of-plane
current at the reconnection X point relative to the rest of the current
sheet, so the qualitative feature seen in Jz_e maps gets one unbiased number
per snapshot.

Method (per run, per threshold time):

1. Threshold time = first field dump where recon_rate_norm reaches the given
   fraction of the run's own peak (same convention as
   fields_at_recon_thresholds.py); the X-point position comes from the same
   reconnection CSV row (tracker output; empirically always the LOWER sheet
   in these campaigns - a y-band guard enforces it anyway).
2. Ridge profile: J(i) = max over a narrow y-band around the X point of
   |Jz_e|, one value per x column -> the 1D along-sheet current profile.
3. Metrics condensed from the profile:
     jet_z         robust z-score: (peak near X - median of the rest) /
                   (1.4826 * MAD of the rest). "Significance vs the rest of
                   the sheet"; noise raises the MAD, so it self-normalizes.
     jet_contrast  peak / median(rest) - scale-free amplitude ratio.
     jet_fwhm      width (d_i) of the contiguous region around the peak
                   above half prominence - separates a collimated jet from
                   a single hot pixel.
     participation (sum J)^2 / (N sum J^2) - window-free concentration
                   index of the whole profile (1 = uniform sheet).
   The only tunables are physical lengths (y-band half-width, near-X
   window), fixed in d_i across runs and resolutions.
4. Histograms: the distribution of J/median(J) over the sheet, with the
   near-X peak marked; the corresponding ECsim reference run
   (Le2DHGEM_RunID_<n>_f2 for regime R<n>) is overlaid on every panel, so
   "does this closure's sheet look like the kinetic reference" is a direct
   visual comparison in the same normalized units.

Known preliminary simplifications (documented, to revisit):
  * the profile segment is the full --choose-x range, not the O-point-
    bracketed cell of the X point;
  * X points are re-picked per snapshot by the tracker (no temporal
    continuity), so time series of jet_z may hop between multiple X points
    of the sheet.

Example:

    python scripts/jet_metric.py                       # R0/new defaults
    python scripts/jet_metric.py --regime R7 --code old --fracs 0.75,0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
# Defaults (all overridable on the command line)
# --------------------------------------------------------------------------
DIAGNOSTICS_BASE = "/volume1/scratch/georgem/closure/diagnostics"
RUNS_BASE = "/esat/cpadata/georgem/2025_112/georgem/menura/runs"
ECSIM_FILES = "/volume1/scratch/share_dir/iPiC3D-nathan"
ECSIM_DIAG = f"{DIAGNOSTICS_BASE}/iPiC3D-nathan"

DEFAULT_CAMPAIGN = "stability_campaign2"
# Same good/bad selection as fields_at_recon_thresholds.py.
DEFAULT_MODELS = [
    "FCNN_00285", "FCNN_00172", "FCNN_00711", "FCNN_00938", "MLP_00643", "MLP_00586",
    "FCNN_00365", "FCNN_00435", "FCNN_00611", "MLP_00535", "MLP_00596", "MLP_00772",
]

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


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--regime", default="R0", help="Single regime (R<n>)")
    parser.add_argument("--code", default="new", choices=["old", "new"])
    parser.add_argument("--models", type=_parse_list, default=DEFAULT_MODELS,
                        help="Comma-separated checkpoint keys")
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
    parser.add_argument("--ecsim-files-path", default=ECSIM_FILES)
    parser.add_argument("--ecsim-diagnostics", default=ECSIM_DIAG)
    parser.add_argument("--rect-halfthick-di", type=float, default=0.15,
                        help="Half-thickness (d_i) of the thin outflow rectangle")
    parser.add_argument("--rect-init-halflen-di", type=float, default=0.5,
                        help="Half-length (d_i) of the seed rectangle used for the "
                        "angle search")
    parser.add_argument("--rect-angle-range-deg", type=float, default=35.0,
                        help="Angle scan range (+-deg from the x axis) for the "
                        "outflow orientation")
    parser.add_argument("--outflow-gap-di", type=float, default=0.3,
                        help="Maximum sub-threshold stretch (d_i) the growing "
                        "rectangle may bridge (the outflow often dims right at the "
                        "X point or between detached lobes); the final length is "
                        "also EXTENDED by this amount to cover the fade-out")
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

    # PRIMARY metric: intensification through the vertical cutouts at the two
    # outflow crossings (x_X +- near-window), each +-edge_box wide in x. The
    # pooled sample is the separatrix aperture when available (fixed band+pad
    # otherwise); p99 = jet core, p50 = the same cutouts' own median.
    nw = int(round(near_window_di / dx))
    ew = max(1, int(round(edge_box_halfwidth_di / dx)))
    pad_cells = max(0, int(round(box_pad_di / dy)))
    fb_lo = max(0, band.start - pad_cells)
    fb_hi = min(vals.shape[1], band.stop + pad_cells)
    boxes, pool_parts, heights = [], [], []
    for centre in (ix_local - nw, ix_local + nw):
        bx_lo, bx_hi = max(0, centre - ew), min(vals.shape[0], centre + ew + 1)
        if bx_hi <= bx_lo:
            continue
        if inside_mask is not None:
            sub = inside_mask[bx_lo:bx_hi]
            rows = np.flatnonzero(sub.any(axis=0))
            if rows.size == 0:
                continue
            boxes.append((bx_lo, bx_hi, int(rows.min()), int(rows.max()) + 1))
            pool_parts.append(A[bx_lo:bx_hi][sub])
            heights.append(float(sub.sum(axis=1).mean()) * dy)
        else:
            boxes.append((bx_lo, bx_hi, fb_lo, fb_hi))
            pool_parts.append(A[bx_lo:bx_hi, fb_lo:fb_hi].ravel())
            heights.append((fb_hi - fb_lo) * dy)
    pool = np.concatenate(pool_parts) if pool_parts else np.array([np.nan])
    p99, p50 = np.nanpercentile(pool, 99), np.nanpercentile(pool, 50)

    # Aperture y-extent within the near-X window (orange box + cut shading).
    if inside_mask is not None:
        nb_lo, nb_hi = max(0, ix_local - nw), min(vals.shape[0], ix_local + nw + 1)
        rows = np.flatnonzero(inside_mask[nb_lo:nb_hi].any(axis=0))
        box_y = (int(rows.min()), int(rows.max()) + 1) if rows.size else (fb_lo, fb_hi)
    else:
        box_y = (fb_lo, fb_hi)

    valid = np.isfinite(J)
    metrics = dict(
        jet_intens=float(p99 / p50) if p50 > 0 else np.inf,
        jet_p99=float(p99), jet_p50=float(p50),
        aperture_h_di=float(np.mean(heights)) if heights else np.nan,
        boxes=boxes, box_y=box_y,
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
                      grow_step_di=0.1, stop_frac=0.5, gap_di=0.3):
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

    Returns dict(angle_deg, halflen_di, outflow_len_di, max_J, mean_core,
    baseline, corners) or None when degenerate.
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
        if seed_mean > core_ref:
            core_ref = seed_mean
            baseline_ref = float(np.median(A[thick & (np.abs(u) <= max_L)]))
    if not frames:
        return None
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
                outflow_len_di=2 * float(Lf), max_J=float(A[final].max()),
                mean_core=core_mean, baseline=baseline,
                corners=np.array(corners))

def aperture_height_over_rect(sep, rect, x_axis, y_axis):
    """Mean inside-separatrix height (d_i) over the outflow rectangle's
    x-footprint: the sheet thickness along the measured outflow. Replaces the
    legacy definition anchored on the retired p99/p50 edge boxes."""
    cols = ((x_axis >= rect["corners"][:, 0].min())
            & (x_axis <= rect["corners"][:, 0].max()))
    if not cols.any():
        return np.nan
    dy = float(y_axis[1] - y_axis[0])
    heights = sep["inside"][cols].sum(axis=1) * dy
    heights = heights[heights > 0]
    return float(heights.mean()) if heights.size else np.nan


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


def main(argv=None):
    args = build_parser().parse_args(argv)
    R = args.regime
    diag = f"{DIAGNOSTICS_BASE}/{args.campaign}"
    out_dir = Path(args.output_dir or f"{diag}/jet_metric")
    out_dir.mkdir(parents=True, exist_ok=True)

    recon = pd.read_csv(f"{diag}/{R}/reconnection_menura.csv")

    # ---- assemble cases: ECsim kinetic reference first, then the models ----
    cases = []  # (label, J, near, metrics, vals, extent, marks)
    ecsim_J_norm = None
    if not args.no_ecsim:
        n = R.removeprefix("R")
        ec_run = f"Le2DHGEM_RunID_{n}_f2"
        ec_recon = pd.read_csv(f"{args.ecsim_diagnostics}/{R}/reconnection_ecsim.csv")
        for frac in args.fracs:
            sub_df = ec_recon[ec_recon["run"] == ec_run]
            th = threshold_index(sub_df, frac, args.rate_column)
            xp = xpoint_at(sub_df, th[0]) if th else None
            if th is None or xp is None:
                print(f"ECsim {ec_run}: no usable threshold/X point, skipping reference")
                continue
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
                gap_di=args.outflow_gap_di)
            if sep is not None and rect is not None:
                m["aperture_h_di"] = aperture_height_over_rect(sep, rect, x_axis, y_axis)
            cases.append((f"ECsim {ec_run}\n{flabel}, t={th[1]:.3g} ({int(frac*100)}%)",
                          J, near, m, vals, (x_axis, y_axis), (ixl, iyl), o_mark, sep, rect))
            if ecsim_J_norm is None:
                ecsim_J_norm = J / m["background"]

    for model in args.models:
        run = f"{args.code}_{model}"
        sub_df = recon[recon["run"] == run]
        for frac in args.fracs:
            th = threshold_index(sub_df, frac, args.rate_column)
            xp = xpoint_at(sub_df, th[0]) if th else None
            if th is None or xp is None:
                print(f"{run}: no usable threshold/X point at {frac}")
                continue
            vals, x_axis, y_axis, flabel, dcase = load_case(
                run, f"{RUNS_BASE}/{args.campaign}/{R}", th[0], args,
                backend="menura", choose_species=["e", "i"],
                normalization="none",
            )
            ixl, iyl = xp[0] - args.choose_x[0], xp[1] - args.choose_y[0]
            if not (0 <= ixl < vals.shape[0] and 0 <= iyl < vals.shape[1]):
                print(f"{run}: X point ({xp}) outside the crop, skipping")
                continue
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
                gap_di=args.outflow_gap_di)
            if sep is not None and rect is not None:
                m["aperture_h_di"] = aperture_height_over_rect(sep, rect, x_axis, y_axis)
            cases.append((f"{run}\n{flabel}, t={th[1]:.3g} ({int(frac*100)}%)",
                          J, near, m, vals, (x_axis, y_axis), (ixl, iyl), o_mark, sep, rect))

    if not cases:
        raise SystemExit("No cases could be built")

    # ---- metrics table ----
    print(f"\n{'case':<42} {'max|Jz|':>8} {'angle':>7} {'L_out_di':>9} {'ap_h_di':>8}")
    for label, J, near, m, *rest_fields in cases:
        rect_c = rest_fields[-1]
        flat = label.replace("\n", " ")
        if rect_c is None:
            print(f"{flat:<42} {'-':>8} {'-':>7} {'-':>9} {m['aperture_h_di']:>8.2f}")
        else:
            print(f"{flat:<42} {rect_c['max_J']:>8.3f} {rect_c['angle_deg']:>+6.1f}° "
                  f"{rect_c['outflow_len_di']:>9.2f} {m['aperture_h_di']:>8.2f}")

    # ---- figure: one row per case = [field map | ridge profile | histogram] ----
    nrows = len(cases)
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 2.6 * nrows), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.6, 1.2, 1.0],
                                              wspace=0.25, hspace=0.55))
    for r, (label, J, near, m, vals, (x_axis, y_axis), (ixl, iyl), o_mark, sep, rect) in enumerate(cases):
        ax = axes[r][0]
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
        by_lo, by_hi = m["box_y"]
        yb_lo, yb_hi = y_axis[by_lo], y_axis[by_hi - 1]
        if rect is not None:
            ax.add_patch(plt.Polygon(rect["corners"], closed=True, fill=False,
                                     edgecolor="orange", linestyle=":",
                                     linewidth=1.6))
        # Flanking background (plateau) zones: contiguous segments of bg_mask.
        idx = np.flatnonzero(m["bg_mask"])
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(seg) == 0:
                continue
            ax.add_patch(plt.Rectangle(
                (x_axis[seg[0]], yb_lo), x_axis[seg[-1]] - x_axis[seg[0]],
                yb_hi - yb_lo, fill=False, edgecolor="lime", linewidth=1.2,
                linestyle=":"))
        ax.set_aspect("equal")
        ax.set_ylabel(label, fontsize=8)
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
        ax.set_xlabel("x [d_i]", fontsize=7, labelpad=1)
        ax.text(0.098, 0.5, "y [d_i]", transform=ax.transAxes, rotation=90,
                ha="left", va="center", fontsize=7, color="white")

        ax = axes[r][1]
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
                f"max|Jz|={rect['max_J']:.3f}  θ={rect['angle_deg']:+.1f}°  "
                f"L_out={rect['outflow_len_di']:.2f} d_i")
        ax.set_title(f"{rtxt}  ap_h={m['aperture_h_di']:.2f} d_i", fontsize=9)
        ax.set_xlim(y_axis[0], y_axis[-1])
        ax.legend(fontsize=6, frameon=False)
        ax.tick_params(labelsize=7)

        ax = axes[r][2]
        Jn = J / m["background"]
        bins = np.linspace(0, max(3.0, np.max(Jn) * 1.05), 40)
        ax.hist(Jn, bins=bins, density=True, alpha=0.6, color="tab:blue",
                label="this run")
        if ecsim_J_norm is not None and r > 0:
            ax.hist(ecsim_J_norm, bins=bins, density=True, histtype="step",
                    color="black", lw=1.4, label="ECsim ref")
        ax.axvline(m["peak"] / m["background"], color="tab:red", lw=1.4)
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)
        if r == 0:
            axes[r][0].set_title("field map (red x=X point, white o=O point; dotted orange =\n"
                                 "outflow rectangle: angle by max mean |Jz|, length grown to the\n"
                                 "intensification edge; "
                                 "orange o=jet peak; "
                                 "white = separatrix aperture, orange = near-X, lime dots = plateau bg)",
                                 fontsize=9)
            axes[r][1].set_title(f"{args.field}(y) at the near-X box edges (outflow crossings);\n"
                                 "green=aperture extent, red=X-point y\n"
                                 + axes[r][1].get_title(), fontsize=8)
            axes[r][2].set_title("hist of J/median (red line = jet peak)", fontsize=10)

    fig.suptitle(
        f"{args.campaign} {R}/{args.code}: lower-sheet jet metric, field {args.field}, "
        f"thresholds {args.fracs} of peak {args.rate_column}",
        fontsize=13, y=1.0,
    )
    out = out_dir / f"{R}_{args.code}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
