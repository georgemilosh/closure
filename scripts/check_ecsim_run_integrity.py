"""Integrity check for transferred ECsim/iPiC3D run directories.

Purpose
-------
Recover each run's physical initial conditions -- background density and the
asymptotic lobe field B0x -- from the field data itself, and check them
against what the directory claims. The run spreadsheet is deliberately not
consulted: it records what was *intended*, whereas the ``*.inp`` deck and the
``SimulationData.txt`` the code writes at startup ship with the data and are
what a reader would trust.

This matters because they can silently disagree. ``Le2DHGEM_RunID_12`` was
delivered with correct metadata (background ``rhoINIT`` 0.229, ``B0z``
0.00498) but field data whose background density is 0.68 -- the RunID_13/14
configuration. The mismatch only became visible when its ``rho_i`` profile was
overlaid against the corresponding Menura run.

How the quantities are measured
-------------------------------
``rho`` is stored on disk as ``rhoINIT / 4pi``. The background species is
spatially uniform, so its mean recovers ``rhoINIT`` directly. ``B0x`` is the
lobe plateau of ``Bx``: averaging over x first removes the initial
perturbation, which otherwise makes ``max|Bx|`` overshoot (0.025149 against a
true 0.0249). ``B0z`` is the mean of ``Bz``. These are the constants that set
the Alfven normalisation, so they are reported for every run whether or not it
passes.

Snapshots are read from whichever format shipped: the processed ``.npz``, or
the raw iPiC3D h5. Full ``*-Fields_*.h5`` carries the same fields under the
legacy ``/Step#0/Block`` layout; the reduced ``*-fieldB_*.h5`` only writes
``Bxc``/``Bzc`` and *time-averaged* densities that are zero at t=0, so for it
B0x/B0z are checked but density is reported ``--`` rather than a spurious 0.

A duplicate scan additionally compares every pair of t=0 snapshots. Note the
baseline: any two runs in this campaign already share ~half their arrays
bit-for-bit, because every run uses the same Harris species (``rhoINIT``
0.969), the same particles-per-cell and the same seed -- so all ``_0``/``_1``
arrays plus ``Bx, By, Ex, Ey, Ez`` match, and two runs sharing a ``B0z`` also
match on ``Bz``. That is expected and is not reported. What *is* reported is a
pair whose background-species data is identical while the two decks declare
different densities -- the RunID_12 / RunID_7 signature.

Usage
-----
  python scripts/check_ecsim_run_integrity.py \\
      [--files-path /volume1/scratch/share_dir/iPiC3D-nathan] \\
      [--pattern 'Le2DHGEM_RunID_*'] \\
      [--rtol 0.01] \\
      [--no-duplicate-scan] \\
      [--json report.json]

Exits non-zero if any directory is flagged, so it can gate a transfer.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import os
import re
import sys

import numpy as np

FOURPI = 4.0 * np.pi
DEFAULT_FILES_PATH = "/volume1/scratch/share_dir/iPiC3D-nathan"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files-path", default=DEFAULT_FILES_PATH,
                   help="Directory holding the run subdirectories")
    p.add_argument("--pattern", default="Le2DHGEM_RunID_*",
                   help="Glob for run subdirectories, relative to --files-path")
    p.add_argument("--harris-species", type=int, default=1,
                   help="Species index of the Harris-sheet ions")
    p.add_argument("--background-species", type=int, default=3,
                   help="Species index of the background ions (spatially uniform)")
    p.add_argument("--rtol", type=float, default=0.01,
                   help="Relative tolerance for agreement between sources")
    p.add_argument("--no-duplicate-scan", action="store_true",
                   help="Skip the pairwise t=0 comparison (the slow part)")
    p.add_argument("--json", default=None, help="Write the full report to this path")
    return p.parse_args()


# --------------------------------------------------------------------------
# metadata readers
# --------------------------------------------------------------------------

def read_inp(path: str) -> dict:
    """Parse an iPiC3D input deck into a flat key -> string mapping."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0]
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            out[key.strip()] = val.strip()
    return out


def read_simulation_data(path: str) -> dict:
    """Parse the SimulationData.txt the code emits at startup."""
    txt = open(path).read()
    b = re.search(r"Initial magnetic field components\s*=\s*"
                  r"([\d.eE+-]+),\s*([\d.eE+-]+),\s*([\d.eE+-]+)", txt)
    save = re.search(r"Output data is saved in (.*)", txt)
    return dict(
        b0=[float(g) for g in b.groups()] if b else None,
        rho=[float(x) for x in re.findall(r"Initial density\s*=\s*([\d.eE+-]+)", txt)],
        savedir=save.group(1).strip() if save else None,
    )


# --------------------------------------------------------------------------
# measurement from the simulation data
# --------------------------------------------------------------------------

def _gradient_axis(slab: np.ndarray) -> int:
    """Return the axis along which `slab` varies, i.e. the Harris gradient (y).

    Detected rather than assumed, so the script does not depend on the index
    order of a particular writer.
    """
    spreads = []
    for axis in (0, 1):
        prof = slab.mean(axis=1 - axis)          # collapse the uniform direction
        spreads.append(float(prof.max() - prof.min()))
    return int(np.argmax(spreads))


def _read_planes(t0_file: str, harris: int, background: int) -> dict:
    """Load the planes ``read_disk`` needs from a t=0 snapshot, whatever the
    on-disk format the run was delivered in.

    Returns ``bg``/``hs`` (background- and Harris-species density planes) and
    ``bx``/``bz`` (magnetic-field planes), plus ``shape`` (the raw Bx dataset
    shape, including any leading z-slices, used by the duplicate scan) and
    ``narrays``. A density plane is ``None`` when the format does not carry an
    instantaneous species density at t=0.

    Three formats occur in this campaign:

    - ``.npz`` -- the processed snapshots, ``rho_<sp>``/``Bx``/``Bz`` directly.
    - full iPiC3D ``*-Fields_*.h5`` -- the same field names under the legacy
      ``/Step#0/Block/<name>/0`` layout.
    - reduced ``*-fieldB_*.h5`` -- only cell-centred ``Bxc``/``Bzc`` and
      *time-averaged* species charge densities (``rhoc_avg_sp_<sp>``), which
      are still zero at t=0. B0x/B0z are recoverable; density is not, so it is
      reported ``None`` rather than a spurious 0.
    """
    # Slabs may be (nz, ny, nx) with nz of 1 or 2; the z-slices are duplicates.
    if t0_file.endswith(".npz"):
        z = np.load(t0_file)
        return dict(
            bg=np.asarray(z[f"rho_{background}"])[0],
            hs=np.asarray(z[f"rho_{harris}"])[0],
            bx=np.asarray(z["Bx"])[0],
            bz=np.asarray(z["Bz"])[0],
            shape=tuple(int(s) for s in np.asarray(z["Bx"]).shape),
            narrays=len(z.files),
        )

    import h5py

    with h5py.File(t0_file, "r") as n:
        blk = n["Step#0"]["Block"]                       # legacy iPiC3D/ECsim layout
        keys = {k.lower(): k for k in blk.keys()}

        def plane(*names):
            """First present of `names` (case-insensitive), z-slice dropped."""
            for nm in names:
                k = keys.get(nm.lower())
                if k is not None:
                    return np.asarray(blk[k]["0"])[0], k
            return None, None

        # Only the instantaneous species density is trusted; the fieldB
        # `rhoc_avg_sp_*` average is deliberately not used (zero at t=0).
        bg, _ = plane(f"rho_{background}")
        hs, _ = plane(f"rho_{harris}")
        bx, bxk = plane("Bx", "Bxc")
        bz, _ = plane("Bz", "Bzc")
        if bx is None or bz is None:
            raise KeyError(f"no Bx/Bz field found (have {sorted(blk.keys())})")
        return dict(
            bg=bg, hs=hs, bx=bx, bz=bz,
            shape=tuple(int(s) for s in blk[bxk]["0"].shape),
            narrays=len(blk.keys()),
        )


def read_disk(t0_file: str, harris: int, background: int) -> dict:
    """Recover the physical initial conditions from a t=0 field snapshot."""
    p = _read_planes(t0_file, harris, background)
    bg, hs, bx, bz = p["bg"], p["hs"], p["bx"], p["bz"]

    # The Harris density is the natural gradient probe; when a format omits it
    # (fieldB), Bx works just as well -- it reverses across the sheet along y.
    axis = _gradient_axis(hs if hs is not None else bx)

    def profile(a):
        return a.mean(axis=1 - axis)

    # Averaging over x kills the initial perturbation, which is x-dependent;
    # what survives is the lobe plateau, i.e. B0x itself.
    out = dict(
        b0x=float(np.abs(profile(bx)).max()),
        b0x_raw_max=float(np.abs(bx).max()),
        b0z=float(bz.mean()),
        shape=p["shape"],
        narrays=p["narrays"],
        gradient_axis=axis,
    )
    if bg is not None:
        out["rho_background"] = float(bg.mean()) * FOURPI
        out["rho_background_rstd"] = (
            float(bg.std() / abs(bg.mean())) if bg.mean() else float("inf"))
    if hs is not None:
        out["rho_harris_peak"] = float(profile(hs).max()) * FOURPI
    return out


def array_digests(t0_file: str) -> dict:
    """SHA1 per array, so the pairwise scan reads each file once instead of N times.

    Keyed by bare field name in both formats (``Bx``, ``rho_3``, ...) so h5 and
    npz snapshots are self-comparable within their own resolution; cross-format
    pairs never meet because the duplicate scan first filters on array shape.
    """
    out = {}
    if t0_file.endswith(".npz"):
        z = np.load(t0_file)
        for key in z.files:
            out[key] = hashlib.sha1(np.ascontiguousarray(z[key]).tobytes()).hexdigest()
        return out

    import h5py

    with h5py.File(t0_file, "r") as n:
        blk = n["Step#0"]["Block"]
        for key in blk.keys():
            a = np.ascontiguousarray(blk[key]["0"][()])
            out[key] = hashlib.sha1(a.tobytes()).hexdigest()
    return out


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def _close(a, b, rtol: float) -> bool:
    if a is None or b is None:
        return True  # nothing to compare against
    return abs(a - b) <= rtol * max(abs(b), 1e-12)


def _find_t0(run_dir: str):
    """Locate the t=0 field snapshot in whatever format the run was delivered.

    Returns ``(path, ext, prefix)`` -- ``prefix`` is the filename stem up to the
    iteration digits, so ``prefix + '*' + ext`` counts the run's snapshots.
    Non-field ``*_000000`` files (e.g. particle dumps) are excluded by name.
    Preference, most to least informative: processed ``.npz``, then the full
    ``*-Fields`` h5, then the reduced ``*-fieldB`` h5 (density-less at t=0) --
    so a run carrying both h5 variants is read from the full one.
    """
    cands = (glob.glob(os.path.join(run_dir, "*_000000.npz"))
             + glob.glob(os.path.join(run_dir, "*_000000.h5")))
    cands = [c for c in cands if "field" in os.path.basename(c).lower()]
    if not cands:
        return None, None, None

    def rank(c):
        base = os.path.basename(c).lower()
        return (0 if c.endswith(".npz") else 1 if "fieldb" not in base else 2, base)

    t0 = min(cands, key=rank)
    ext = os.path.splitext(t0)[1]
    prefix = re.sub(r"\d+" + re.escape(ext) + "$", "", os.path.basename(t0))
    return t0, ext, prefix


def collect(run_dir: str, harris: int, background: int) -> dict:
    tag = os.path.basename(run_dir.rstrip("/"))
    m = re.match(r".*RunID_(\d+)", tag)
    rec: dict = dict(dir=tag, run_id=int(m.group(1)) if m else None, problems=[])

    decks = glob.glob(os.path.join(run_dir, "*.inp"))
    if decks:
        deck = read_inp(decks[0])
        rho = [float(x) for x in deck.get("rhoINIT", "").split()]
        rec["inp_rho_background"] = rho[background] if len(rho) > background else None
        rec["inp_rho_harris"] = rho[harris] if len(rho) > harris else None
        rec["inp_b0x"] = float(deck["B0x"]) if "B0x" in deck else None
        rec["inp_b0z"] = float(deck["B0z"]) if "B0z" in deck else None
        rec["inp_savedir"] = deck.get("SaveDirName")

    sim_path = os.path.join(run_dir, "SimulationData.txt")
    if os.path.exists(sim_path):
        sim = read_simulation_data(sim_path)
        rec["sim_rho_background"] = sim["rho"][background] if len(sim["rho"]) > background else None
        rec["sim_b0x"] = sim["b0"][0] if sim["b0"] else None
        rec["sim_b0z"] = sim["b0"][2] if sim["b0"] else None
        rec["sim_savedir"] = sim["savedir"]

    t0, ext, prefix = _find_t0(run_dir)
    if t0 is None:
        rec["problems"].append("no t=0 snapshot")
        return rec
    rec["t0_file"] = os.path.basename(t0)
    rec["t0_path"] = t0
    rec["nsnapshots"] = len(glob.glob(os.path.join(run_dir, prefix + "*" + ext)))
    try:
        rec.update(read_disk(t0, harris, background))
    except Exception as exc:  # unreadable / unexpected layout
        rec["problems"].append(f"unreadable t=0 snapshot: {exc!r}")
    return rec


def evaluate(rec: dict, rtol: float) -> None:
    """Populate rec['problems'] by checking the measured values against the deck."""
    measured = dict(rho_background=rec.get("rho_background"),
                    b0x=rec.get("b0x"), b0z=rec.get("b0z"))
    labels = dict(rho_background="background density", b0x="B0x", b0z="B0z")
    fmts = dict(rho_background="%.4f", b0x="%.6f", b0z="%.6f")

    for src in ("inp", "sim"):
        for quantity, value in measured.items():
            declared = rec.get(f"{src}_{quantity}")
            if declared is None or value is None or _close(value, declared, rtol):
                continue
            rec["problems"].append(
                ("%s measured %s != %s %s" %
                 (labels[quantity], fmts[quantity] % value, src, fmts[quantity] % declared)))

    if rec.get("inp_rho_background") is not None and rec.get("sim_rho_background") is not None:
        if not _close(rec["inp_rho_background"], rec["sim_rho_background"], rtol):
            rec["problems"].append("inp disagrees with SimulationData.txt on background density")

    # Sanity-check the species index rather than the physics: a background
    # species sits near std/|mean| ~ 0 (0.003 even with visible particle
    # noise), whereas pointing this at a Harris species gives ~2.4. Only a
    # gross departure means --background-species is wrong; ordinary PIC noise
    # is reported below as a note, since the mean still recovers rhoINIT.
    rstd = rec.get("rho_background_rstd")
    if rstd is not None and rstd > 0.2:
        rec["problems"].append(
            f"species {rec.get('background_species')} is not spatially uniform "
            f"(std/|mean| = {rstd:.3g}) -- wrong --background-species?")
    elif rstd is not None and rstd > 1e-3:
        rec["warnings"] = rec.get("warnings", []) + [
            f"background species carries {100 * rstd:.2g}% particle noise "
            f"(mean still matches; other runs are exactly uniform at t=0)"]

    # Not an integrity failure on its own, but it is the cheap tell that a
    # directory's metadata and its data came from different jobs: RunID_12's
    # deck says /dodrio/... while the SimulationData.txt written at runtime
    # says the LUMI path. Compare full paths, not basenames -- the basenames
    # agree precisely because both name the run.
    a, b = rec.get("inp_savedir"), rec.get("sim_savedir")
    if a and b and a.rstrip("/") != b.rstrip("/"):
        rec["warnings"] = rec.get("warnings", []) + [
            f"SaveDirName differs between deck and runtime: inp={a} vs SimulationData.txt={b}"]


def duplicate_scan(records: list, background: int) -> list:
    """Find run pairs sharing background-species data despite differing densities.

    The decisive array is ``rho_<background>``: two runs initialised at
    different background densities cannot deposit an identical one. Runs
    legitimately configured at the *same* density can, so a hit is only
    interesting when the two decks declare different densities -- which is
    exactly the RunID_12 / RunID_7 case.
    """
    digests = {}
    for rec in records:
        if not rec.get("t0_path"):
            continue
        try:
            digests[rec["dir"]] = array_digests(rec["t0_path"])
        except Exception as exc:
            rec["problems"].append(f"digest failed: {exc!r}")
    by_dir = {r["dir"]: r for r in records}

    key = f"rho_{background}"
    hits = []
    for a, b in itertools.combinations(sorted(digests), 2):
        ra, rb = by_dir[a], by_dir[b]
        if ra.get("shape") != rb.get("shape"):
            continue  # e.g. _f2 (downsampled) vs _npz32 (full grid)
        da, db = digests[a], digests[b]
        if key not in da or key not in db or da[key] != db[key]:
            continue
        common = sorted(set(da) & set(db))
        same = [k for k in common if da[k] == db[k]]
        declared = (ra.get("inp_rho_background"), rb.get("inp_rho_background"))
        contradiction = (declared[0] is not None and declared[1] is not None
                         and abs(declared[0] - declared[1]) > 1e-12)
        hits.append(dict(a=a, b=b, identical=len(same), common=len(common),
                         differ=[k for k in common if da[k] != db[k]],
                         declared=declared, contradiction=contradiction))
    return hits


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _fmt(v, spec="%.4f"):
    return "  --  " if v is None else spec % v


def report(records: list, hits: list, scanned: bool) -> int:
    print("Measured from the t=0 field data (these set the Alfven normalisation):")
    head = ("  %-24s %12s %10s %10s %12s %6s" %
            ("dir", "n_background", "B0x", "B0z", "harris_peak", "nsnap"))
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in records:
        print("  %-24s %12s %10s %10s %12s %6s" % (
            r["dir"], _fmt(r.get("rho_background")), _fmt(r.get("b0x"), "%.6f"),
            _fmt(r.get("b0z"), "%.6f"), _fmt(r.get("rho_harris_peak")),
            r.get("nsnapshots", "-")))

    print()
    print("Measured vs declared:")
    head = ("  %-24s %9s %9s %11s | %9s %9s %11s" %
            ("dir", "inp_rhob", "sim_rhob", "MEAS_rhob", "inp_B0z", "sim_B0z", "MEAS_B0z"))
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in records:
        print("  %-24s %9s %9s %11s | %9s %9s %11s  %s" % (
            r["dir"], _fmt(r.get("inp_rho_background")), _fmt(r.get("sim_rho_background")),
            _fmt(r.get("rho_background")), _fmt(r.get("inp_b0z"), "%.6f"),
            _fmt(r.get("sim_b0z"), "%.6f"), _fmt(r.get("b0z"), "%.6f"),
            "FAIL" if r["problems"] else ""))

    warned = [r for r in records if r.get("warnings")]
    if warned:
        print()
        print("Warnings:")
        for r in warned:
            for w in r["warnings"]:
                print("  %-24s %s" % (r["dir"], w))

    if scanned:
        print()
        print("Duplicate scan (pairs sharing background-species data; a shared Harris")
        print("species and shared in-plane fields are expected and not reported):")
        if not hits:
            print("  none")
        for h in hits:
            print("  %s == %s : %d/%d arrays identical, differ=%s%s" % (
                h["a"], h["b"], h["identical"], h["common"],
                h["differ"] if len(h["differ"]) < 8 else "(many)",
                "   <== but their decks declare %.4g vs %.4g" % h["declared"]
                if h["contradiction"] else "   (both decks declare the same density)"))

    failed = [r for r in records if r["problems"]]
    print()
    print("=" * 72)
    if failed:
        print("FLAGGED %d of %d directories:" % (len(failed), len(records)))
        for r in failed:
            for p in r["problems"]:
                print("  %-24s %s" % (r["dir"], p))
        return 1
    print("All %d directories consistent." % len(records))
    return 0


def main() -> int:
    args = _parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.files_path, args.pattern))
                  if os.path.isdir(d))
    if not dirs:
        print("no run directories matching %s under %s" % (args.pattern, args.files_path),
              file=sys.stderr)
        return 2

    records = []
    for d in dirs:
        rec = collect(d, args.harris_species, args.background_species)
        rec["background_species"] = args.background_species
        evaluate(rec, args.rtol)
        records.append(rec)

    hits = [] if args.no_duplicate_scan else duplicate_scan(records, args.background_species)
    # The scan explains where bad data came from; it does not by itself say
    # which of the pair is wrong. Attribute it as a warning to both, and let
    # the measured-vs-declared check decide the exit status -- so the innocent
    # partner (RunID_7) is not failed alongside the mislabelled one.
    for h in hits:
        if not h["contradiction"]:
            continue
        for rec in records:
            if rec["dir"] in (h["a"], h["b"]):
                other = h["b"] if rec["dir"] == h["a"] else h["a"]
                rec["warnings"] = rec.get("warnings", []) + [
                    f"background-species data is bit-identical to {other} "
                    f"({h['identical']}/{h['common']} arrays match overall)"]

    status = report(records, hits, not args.no_duplicate_scan)

    if args.json:
        for r in records:
            r.pop("t0_path", None)
        with open(args.json, "w") as fh:
            json.dump(dict(files_path=args.files_path, records=records, duplicates=hits),
                      fh, indent=2, default=str)
        print("Wrote %s" % args.json)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
