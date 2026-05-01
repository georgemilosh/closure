"""Offline consistency check: compare closure training-side E_amb proxy to
plasma.get_Ohm reference on a set of ECsim/iPiC3D data files.

Purpose (Phase 6 of physics-informed loss plan)
------------------------------------------------
Before running production training with lambda_gradP / lambda_eamb, this
script establishes:

  1. Interior-cell agreement between _fd4_derivatives_2d and plasma.highdiff
     on the actual simulation data (boundary behaviour is intentionally
     excluded — the two implementations differ there by design).

  2. Per-channel statistics (mean, std, p99) for EPx, EPy, EPz so that
     physics_dx / physics_dy and loss weight lambdas can be set on data
     evidence rather than guesswork.

  3. A handoff checklist printed to stdout confirming sign convention and
     channel ordering are consistent for a given model checkpoint before
     deploying it to Menura.

Usage
-----
  python scripts/check_eamb_offline_consistency.py \\
      --data-file /path/to/ECsim-or-iPiC3D.h5 \\
      [--request-targets Pxx_e Pyy_e Pzz_e Pxy_e Pxz_e Pyz_e] \\
      [--request-features rho_e Bx By Bz Vx_e Vy_e Vz_e Ex Ey Ez] \\
      [--qom -1.0 1.0] \\
      [--time-index 0] \\
      [--dx 0.05] [--dy 0.05] \\
      [--out-dir ./consistency_report]

The script writes a JSON summary and optional PNG spectra to --out-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", required=True, help="Path to HDF5 or pkl simulation snapshot")
    p.add_argument("--request-features", nargs="+",
                   default=["rho_e", "Bx", "By", "Bz", "Vx_e", "Vy_e", "Vz_e", "Ex", "Ey", "Ez"])
    p.add_argument("--request-targets", nargs="+",
                   default=["Pxx_e", "Pyy_e", "Pzz_e", "Pxy_e", "Pxz_e", "Pyz_e"])
    p.add_argument("--qom", nargs="+", type=float, default=[-1.0, 1.0],
                   help="Charge-to-mass ratios [electron, ion, ...]")
    p.add_argument("--time-index", type=int, default=0, help="Time slice index to use")
    p.add_argument("--dx", type=float, default=1.0, help="Grid spacing in x (code units)")
    p.add_argument("--dy", type=float, default=1.0, help="Grid spacing in y (code units)")
    p.add_argument("--out-dir", default="./consistency_report",
                   help="Directory to write JSON summary and PNG plots")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib figure output")
    return p.parse_args()


def _load_snapshot(data_file: str, request_features: list[str], request_targets: list[str],
                   time_index: int):
    """Load features and targets from file using closure read_pic utilities."""
    from closure.read_pic import read_features_targets  # type: ignore[import]
    from closure.read_pic import read_data, parse_simulation_data  # type: ignore[import]
    from closure.read_pic import _augment_fields_to_read_from_requests  # type: ignore[import]
    from closure.utilities import species_to_list

    data_path = Path(data_file)
    try:
        feats, tgts = read_features_targets(
            str(data_path.parent),
            [data_path.name],
            request_features=request_features,
            request_targets=request_targets,
            choose_species=["e", None],
        )
    except ValueError as exc:
        # Single-snapshot files can trigger a squeeze/transposition edge case in
        # read_features_targets/read_files (expects 4D batch output). Fall back
        # to a direct one-file read and explicit channel extraction.
        if "expected 4 dimensions" not in str(exc):
            raise

        fields_to_read = _augment_fields_to_read_from_requests(
            None,
            request_features,
            request_targets,
        )
        sim_data = parse_simulation_data(str(data_path.parent))
        qom = sim_data["qom"]
        data = read_data(
            str(data_path.parent),
            data_path.name,
            fields_to_read,
            qom,
            choose_species=["e", None],
            verbose=False,
        )

        def _extract(channels: list[str]) -> np.ndarray:
            out = []
            for key in species_to_list(channels):
                arr = data[key[0]][key[1]] if isinstance(key, list) else data[key]
                arr = np.asarray(arr)
                if arr.ndim == 3:
                    # read_data returns [H, W, T] for time-dependent fields.
                    arr = arr[..., time_index]
                elif arr.ndim != 2:
                    raise ValueError(f"Unexpected channel shape {arr.shape} for key={key}")
                out.append(arr)
            return np.stack(out, axis=0)

        feats = _extract(request_features)
        tgts = _extract(request_targets)
    # feats/tgts: [C, H, W] or [C, H, W, T]; take time slice
    if feats.ndim == 4:
        feats = feats[..., time_index]
        tgts = tgts[..., time_index]
    return feats, tgts


def _build_data_dict_for_ohm(feats_np: np.ndarray, tgts_np: np.ndarray,
                              feat_names: list[str], tgt_names: list[str]) -> dict:
    """Reconstruct a plasma.get_Ohm-compatible data dict from flat arrays."""
    from closure import plasma  # noqa: F401  (import here for clarity)

    def _idx(names, prefix):
        for i, n in enumerate(names):
            if n.startswith(prefix):
                return i
        return None

    nx, ny = feats_np.shape[1], feats_np.shape[2]

    def _f(arr):
        """Add singleton time axis expected by plasma functions."""
        return arr[..., np.newaxis]

    def _feat(prefix):
        i = _idx(feat_names, prefix)
        return _f(feats_np[i]) if i is not None else _f(np.zeros((nx, ny)))

    def _tgt(prefix):
        i = _idx(tgt_names, prefix)
        return _f(tgts_np[i]) if i is not None else _f(np.zeros((nx, ny)))

    data = {
        "Bx": _feat("Bx"), "By": _feat("By"), "Bz": _feat("Bz"),
        "Ex": _feat("Ex"), "Ey": _feat("Ey"), "Ez": _feat("Ez"),
        "rho": {"e": _feat("rho_e"), "i": _f(np.ones((nx, ny)))},
        "Jx": {"e": _feat("Jx_e"), "i": _f(np.zeros((nx, ny)))},
        "Jy": {"e": _feat("Jy_e"), "i": _f(np.zeros((nx, ny)))},
        "Jz": {"e": _feat("Jz_e"), "i": _f(np.zeros((nx, ny)))},
        "Vx": {"e": _feat("Vx_e"), "i": _f(np.zeros((nx, ny)))},
        "Vy": {"e": _feat("Vy_e"), "i": _f(np.zeros((nx, ny)))},
        "Vz": {"e": _feat("Vz_e"), "i": _f(np.zeros((nx, ny)))},
        "Pxx": {"e": _tgt("Pxx_e"), "i": _f(np.zeros((nx, ny)))},
        "Pxy": {"e": _tgt("Pxy_e"), "i": _f(np.zeros((nx, ny)))},
        "Pxz": {"e": _tgt("Pxz_e"), "i": _f(np.zeros((nx, ny)))},
        "Pyy": {"e": _tgt("Pyy_e"), "i": _f(np.zeros((nx, ny)))},
        "Pyz": {"e": _tgt("Pyz_e"), "i": _f(np.zeros((nx, ny)))},
        "Pzz": {"e": _tgt("Pzz_e"), "i": _f(np.zeros((nx, ny)))},
    }
    return data


def _torch_eamb(tgts_np: np.ndarray, tgt_names: list[str],
                feats_np: np.ndarray, feat_names: list[str],
                dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute EPx/EPy/EPz via the training-side torch helper."""
    from closure.module import ClosureLitModule

    # Map target names to channel indices (strip _e suffix)
    short = {n.split("_")[0]: i for i, n in enumerate(tgt_names)}
    channel_map = {k: short[k] for k in ("Pxx", "Pxy", "Pxz", "Pyy", "Pyz") if k in short}
    if len(channel_map) < 5:
        raise ValueError(f"Missing pressure channels in targets: got {list(short.keys())}")

    rho_idx = next((i for i, n in enumerate(feat_names) if n.startswith("rho_e")), 0)
    rho_np = feats_np[rho_idx]  # [H, W]

    pressure_t = torch.from_numpy(tgts_np).float().unsqueeze(0)  # [1, C, H, W]
    rho_t = torch.from_numpy(rho_np).float().unsqueeze(0)         # [1, H, W]

    eamb = ClosureLitModule._compute_eamb_from_pressure(
        pressure_t, rho_t, channel_map, dx=dx, dy=dy, small=1e-10, rho_abs=True,
    )
    epx = eamb[0, 0].numpy()
    epy = eamb[0, 1].numpy()
    epz = eamb[0, 2].numpy()
    return epx, epy, epz


def _scalar_stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p99": float(np.percentile(np.abs(arr), 99)),
        "max_abs": float(np.abs(arr).max()),
    }


def _high_k_ratio(arr: np.ndarray) -> float:
    """Energy fraction in upper half of k-space (roughness proxy)."""
    ft = np.fft.rfft2(arr)
    power = np.abs(ft) ** 2
    nx, nky = power.shape
    mid_x = nx // 2
    mid_y = nky // 2
    high_k = power[mid_x:, :].sum() + power[:, mid_y:].sum()
    total = power.sum()
    return float(high_k / (total + 1e-30))


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Closure offline E_amb consistency check ===")
    print(f"Data file : {args.data_file}")
    print(f"Time index: {args.time_index}")
    print(f"dx={args.dx}, dy={args.dy}")
    print()

    # 1. Load snapshot
    print("[1/4] Loading snapshot...")
    feats_np, tgts_np = _load_snapshot(
        args.data_file, args.request_features, args.request_targets, args.time_index
    )
    nx, ny = feats_np.shape[1], feats_np.shape[2]
    print(f"      Features shape: {feats_np.shape}, Targets shape: {tgts_np.shape}")

    x_np = np.arange(nx) * args.dx
    y_np = np.arange(ny) * args.dy

    # 2. Offline reference: plasma.get_Ohm
    print("[2/4] Running plasma.get_Ohm (offline reference)...")
    from closure import plasma
    data_dict = _build_data_dict_for_ohm(feats_np, tgts_np, args.request_features, args.request_targets)
    plasma.get_Ohm(data_dict, args.qom, x_np, y_np)
    ref_epx = data_dict["EPx"][2:-2, 2:-2, 0]
    ref_epy = data_dict["EPy"][2:-2, 2:-2, 0]
    ref_epz = data_dict["EPz"][2:-2, 2:-2, 0]

    # 3. Torch training-side proxy
    print("[3/4] Running torch E_amb proxy (_compute_eamb_from_pressure)...")
    torch_epx, torch_epy, torch_epz = _torch_eamb(
        tgts_np, args.request_targets, feats_np, args.request_features, args.dx, args.dy
    )

    # 4. Compare and summarise
    print("[4/4] Computing statistics and agreement metrics...")
    summary: dict = {}

    for name, ref, got in [("EPx", ref_epx, torch_epx),
                            ("EPy", ref_epy, torch_epy),
                            ("EPz", ref_epz, torch_epz)]:
        diff = got - ref
        rel_err = np.abs(diff) / (np.abs(ref) + 1e-30)
        summary[name] = {
            "offline_reference": _scalar_stats(ref),
            "torch_proxy": _scalar_stats(got),
            "absolute_difference": _scalar_stats(diff),
            "relative_error_p99": float(np.percentile(rel_err, 99)),
            "high_k_ratio_reference": _high_k_ratio(ref),
            "high_k_ratio_torch": _high_k_ratio(got),
        }
        print(f"\n  {name}")
        print(f"    ref  mean={ref.mean():.4e} std={ref.std():.4e} p99_abs={np.percentile(np.abs(ref),99):.4e}")
        print(f"    got  mean={got.mean():.4e} std={got.std():.4e} p99_abs={np.percentile(np.abs(got),99):.4e}")
        print(f"    diff p99_rel={summary[name]['relative_error_p99']:.4e}   "
              f"high_k_ratio ref={summary[name]['high_k_ratio_reference']:.4f} "
              f"got={summary[name]['high_k_ratio_torch']:.4f}")

    # -----------------------------------------------------------------------
    # Handoff checklist
    # -----------------------------------------------------------------------
    print("\n=== Handoff checklist for Menura deployment ===")
    checklist = {}

    # (a) Sign convention
    rho_vals = feats_np[[i for i, n in enumerate(args.request_features) if n.startswith("rho_e")][0]]
    rho_sign = "negative (ECsim convention)" if rho_vals.mean() < 0 else "positive (Menura convention)"
    checklist["rho_sign"] = rho_sign
    print(f"  [{'OK' if 'negative' in rho_sign else 'CHECK'}] rho_e sign in features: {rho_sign}")
    print(f"        extract_features_kernel negates density_b → physics_rho_abs=True handles both.")

    # (b) Channel ordering
    tgt_short = [n.split("_")[0] for n in args.request_targets]
    expected_order_closure = ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
    expected_order_menura  = ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]  # pres_elec[0-5]
    order_ok = all(tgt_short[i] == expected_order_closure[i] for i in range(min(len(tgt_short), 6)))
    checklist["target_channel_order"] = tgt_short
    checklist["target_channel_order_ok"] = bool(order_ok)
    print(f"  [{'OK' if order_ok else 'MISMATCH'}] Target channel order: {tgt_short}")
    print(f"        Menura pres_elec order: {expected_order_menura}")

    # (c) E_amb proxy agreement with offline
    max_p99_rel = max(summary[k]["relative_error_p99"] for k in ("EPx", "EPy", "EPz"))
    agreement_ok = max_p99_rel < 0.01
    checklist["eamb_proxy_p99_rel_error"] = float(max_p99_rel)
    checklist["eamb_proxy_agreement_ok"] = bool(agreement_ok)
    print(f"  [{'OK' if agreement_ok else 'WARNING'}] E_amb proxy p99 relative error: {max_p99_rel:.4e} "
          f"(threshold 1e-2)")

    # (d) Smoothness baseline (fill in after ablation training)
    high_k_refs = {k: summary[k]["high_k_ratio_reference"] for k in ("EPx", "EPy", "EPz")}
    checklist["high_k_ratio_baseline"] = high_k_refs
    print(f"  [INFO] High-k ratio baseline (fill into acceptance criteria):")
    for k, v in high_k_refs.items():
        print(f"         {k}: {v:.4f}")
    print()
    print("  After training with physics loss, re-run this script with the model's")
    print("  predictions as --data-file targets to compare high_k_ratio_torch vs baseline.")
    print("  Acceptance criterion: high_k_ratio reduced for all three components.")

    checklist["summary"] = summary
    out_json = out_dir / "eamb_consistency.json"
    out_json.write_text(json.dumps({"checklist": checklist, "details": summary}, indent=2))
    print(f"\nSummary written to {out_json}")

    # -----------------------------------------------------------------------
    # Optional plots
    # -----------------------------------------------------------------------
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for name, ref, got in [("EPx", ref_epx, torch_epx),
                                    ("EPy", ref_epy, torch_epy),
                                    ("EPz", ref_epz, torch_epz)]:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                vmax = np.percentile(np.abs(ref), 99)
                axes[0].imshow(ref.T, origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu_r")
                axes[0].set_title(f"{name} offline ref")
                axes[1].imshow(got.T, origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu_r")
                axes[1].set_title(f"{name} torch proxy")
                diff = got - ref
                dmax = np.percentile(np.abs(diff), 99)
                axes[2].imshow(diff.T, origin="lower", vmin=-dmax, vmax=dmax, cmap="RdBu_r")
                axes[2].set_title(f"{name} diff (p99 rel={summary[name]['relative_error_p99']:.2e})")
                fig.tight_layout()
                out_png = out_dir / f"{name}_consistency.png"
                fig.savefig(out_png, dpi=120)
                plt.close(fig)
                print(f"Plot written to {out_png}")
        except ImportError:
            print("matplotlib not available — skipping plots.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
