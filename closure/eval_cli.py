"""eval_cli.py - Command-line evaluation helper built around RunLoader.

This command mirrors the common notebook-based post-training workflow:
- load a run/checkpoint
- print run config summaries
- print history and best-epoch info
- compute test-set metrics, print them, and write CSV
- save history/metrics figures to ``<run_or_version>/img``
- optionally export per-target field comparison plots
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from closure.run_loader import RunLoader


def _configure_eval_logging(log_file: Path) -> None:
    """Configure console + file logging for closure-eval."""
    root_logger = logging.getLogger()
    if getattr(root_logger, "_closure_eval_configured", False):
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    for handler in (stream_handler, file_handler):
        handler.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    root_logger._closure_eval_configured = True
    logging.getLogger(__name__).info("Evaluation logging initialized -> %s", log_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained closure run and export notebook-style artifacts.",
    )
    source = parser.add_argument_group("Run selection")
    source.add_argument(
        "--version-dir",
        type=Path,
        help="Path to a run/version directory containing config.yaml and checkpoints/.",
    )
    source.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Run directory path. Can point to one run/version folder or to a parent "
            "folder containing many run subdirectories for batch evaluation."
        ),
    )
    source.add_argument(
        "--log-root",
        type=Path,
        help="Directory containing run_*/version_* subfolders; latest one is selected.",
    )
    source.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Optional checkpoint path. Defaults to auto-selected best checkpoint.",
    )
    source.add_argument(
        "--device",
        default="cpu",
        help="Torch device for model loading (for example: cpu, cuda, cuda:0).",
    )
    source.add_argument(
        "--stage",
        default="test",
        choices=["test", "fit", "validate", "predict"],
        help="Datamodule setup stage. Default: test.",
    )

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for csv/img (default: selected run/version directory).",
    )
    outputs.add_argument(
        "--metrics-csv-name",
        default="test_metrics.csv",
        help="Output CSV filename for per-channel metrics.",
    )

    eval_cfg = parser.add_argument_group("Evaluation options")
    eval_cfg.add_argument(
        "--test-samples-file",
        default=None,
        help="Optional override for data.test_samples_file without editing YAML config.",
    )
    eval_cfg.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Target names to render field plots for (default: all requested targets).",
    )
    eval_cfg.add_argument(
        "--plot-step",
        type=int,
        default=3,
        help="Step for plotted sample indices: 0, step, 2*step, ...",
    )
    eval_cfg.add_argument(
        "--max-plots",
        type=int,
        default=None,
        help="Limit number of plotted time slices per target.",
    )
    eval_cfg.add_argument(
        "--robust-quantile",
        type=float,
        default=0.995,
        help="Robust quantile for color limits in field plots.",
    )
    eval_cfg.add_argument(
        "--error-mode",
        default="relative",
        choices=["relative", "absolute", "symmetric_percent"],
        help="Error panel mode in field plots.",
    )
    eval_cfg.add_argument(
        "--signed-target-names",
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of target names treated as signed for "
            "diverging colormap scaling. Default: auto-detect from plotted values."
        ),
    )
    eval_cfg.add_argument(
        "--skip-history-plot",
        action="store_true",
        help="Skip saving training history figure.",
    )
    eval_cfg.add_argument(
        "--skip-metrics-plot",
        action="store_true",
        help="Skip saving channel-wise metrics bar chart.",
    )
    eval_cfg.add_argument(
        "--skip-field-plots",
        action="store_true",
        help="Skip per-target spatial prediction/ground-truth/error plots.",
    )

    return parser.parse_args()


def _select_version_dir(args: argparse.Namespace) -> Path:
    if args.version_dir is not None:
        return args.version_dir.expanduser().resolve()
    if args.run_dir is not None:
        return args.run_dir.expanduser().resolve()
    if args.log_root is not None:
        root = args.log_root.expanduser().resolve()
        candidates = sorted(
            [
                p for p in root.iterdir()
                if p.is_dir() and (p.name.startswith("run_") or p.name.startswith("version_"))
            ]
        )
        if not candidates:
            raise FileNotFoundError(f"No run_*/version_* directories found in {root}")
        return candidates[-1]
    raise ValueError("Provide one of --version-dir, --run-dir, or --log-root")


def _is_complete_run_dir(path: Path) -> bool:
    """Return True when *path* looks like an evaluable run directory."""
    if not path.is_dir():
        return False
    if not (path / "config.yaml").exists():
        return False
    ckpt_dir = path / "checkpoints"
    if not ckpt_dir.is_dir():
        return False
    return any(ckpt_dir.glob("*.ckpt"))


def _candidate_run_dirs(run_root: Path) -> Iterable[Path]:
    """Yield candidate run version directories sorted by name.

    Supports both flat (run_root/version/) and two-level
    (run_root/group/version/) layouts.  If any direct child of *run_root*
    is a complete run directory the flat layout is assumed; otherwise each
    direct child is treated as a group and its children are returned.
    """
    direct = sorted(p for p in run_root.iterdir() if p.is_dir())
    if any(_is_complete_run_dir(p) for p in direct):
        return direct
    # Two-level layout: run_root/physics_tag/run_tag/
    grandchildren: list[Path] = []
    for group in direct:
        grandchildren.extend(sorted(p for p in group.iterdir() if p.is_dir()))
    return grandchildren


def _resolve_eval_dirs(args: argparse.Namespace) -> tuple[list[Path], bool]:
    """Resolve one or many run directories from CLI args.

    Returns
    -------
    eval_dirs : list[Path]
        Directories to evaluate.
    from_run_root : bool
        True when using batch mode from a parent --run-dir folder.
    """
    if args.run_dir is not None and args.version_dir is None and args.log_root is None:
        run_path = args.run_dir.expanduser().resolve()
        if _is_complete_run_dir(run_path):
            return [run_path], False

        if not run_path.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_path}")

        candidates = list(_candidate_run_dirs(run_path))
        if candidates:
            return candidates, True

        raise FileNotFoundError(
            f"No complete run directories found under {run_path}. "
            "A complete run requires config.yaml and checkpoints/*.ckpt"
        )

    return [_select_version_dir(args)], False


def _print_config_summary(run: RunLoader) -> None:
    for section in ("trainer", "data"):
        print(f"\n[{section}]")
        section_cfg = run.config.get(section, {})
        for key, value in section_cfg.items():
            if not isinstance(value, (dict, list)):
                print(f"  {key}: {value}")

    print("\n[model.network]")
    network_cfg = run.config.get("model", {}).get("network", {})
    print(f"  class_path: {network_cfg.get('class_path')}")
    print(f"  init_args:  {network_cfg.get('init_args', {})}")


def _save_history_plot(run: RunLoader, out_img_dir: Path) -> None:
    history = run.history()
    lr_cols = [col for col in history.columns if col.startswith("lr-")]

    physics_val_cols = [c for c in ["val_loss_base", "val_loss_gradp", "val_loss_eamb"] if c in history.columns]
    physics_scale_col = "val_loss_physics_scale" if "val_loss_physics_scale" in history.columns else None
    has_physics = bool(physics_val_cols)
    n_panels = 1 + (1 if has_physics else 0) + (1 if lr_cols else 0)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    panel = 0
    axes[panel].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[panel].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[panel].set_title("Total loss vs epoch")
    axes[panel].set_xlabel("epoch")
    axes[panel].set_ylabel("loss")
    axes[panel].grid(alpha=0.3)
    axes[panel].legend()

    if has_physics:
        panel += 1
        for col in physics_val_cols:
            axes[panel].plot(history["epoch"], history[col], label=col)
        axes[panel].set_title("Physics component losses (val)")
        axes[panel].set_xlabel("epoch")
        axes[panel].set_ylabel("relative loss")
        axes[panel].grid(alpha=0.3)
        if physics_scale_col:
            ax2 = axes[panel].twinx()
            ax2.plot(history["epoch"], history[physics_scale_col],
                     color="grey", linestyle="--", alpha=0.6, label="physics_scale")
            ax2.set_ylabel("physics_scale")
            ax2.set_ylim(-0.05, 1.15)
            ax2.legend(loc="lower right", fontsize=8)
        axes[panel].legend(loc="upper right")

    if lr_cols:
        panel += 1
        for col in lr_cols:
            axes[panel].plot(history["epoch"], history[col], label=col)
        axes[panel].set_title("Learning rate vs epoch")
        axes[panel].set_xlabel("epoch")
        axes[panel].set_ylabel("lr")
        axes[panel].grid(alpha=0.3)
        axes[panel].legend()

    fig.tight_layout()
    output_path = out_img_dir / "history.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved history plot -> {output_path}")


def _save_metrics_plot(metrics_df, out_img_dir: Path) -> None:
    metric_by_channel = metrics_df.set_index("channel").sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    metric_by_channel["r2"].plot.bar(ax=axes[0], color="steelblue", alpha=0.85)
    axes[0].set_title("Channel-wise R2")
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].axhline(0, color="grey", linewidth=0.5)
    axes[0].grid(axis="y", alpha=0.3)

    metric_by_channel["nrmse"].plot.bar(ax=axes[1], color="darkorange", alpha=0.85)
    axes[1].set_title("Channel-wise normalized RMSE")
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.set_xlabel("target channel")

    fig.tight_layout()
    output_path = out_img_dir / "channel_metrics.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics plot -> {output_path}")


def _cleanup_process_memory() -> None:
    """Release as much process memory as possible between batch runs."""
    plt.close("all")
    gc.collect()
    try:
        import torch  # Local import: optional dependency for this cleanup path.

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Keep cleanup best-effort and non-fatal on environments without torch/cuda.
        pass


def _evaluate_single_run(
    version_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> bool:
    """Evaluate one run directory.

    Returns True on success, False if run should be skipped.
    """
    data_overrides = {}
    if args.test_samples_file:
        data_overrides["test_samples_file"] = args.test_samples_file

    run = None
    history = None
    metrics_df = None
    ground_truth = None
    prediction = None

    try:
        run = RunLoader.from_version_dir(
            version_dir,
            stage=args.stage,
            ckpt=args.ckpt,
            device=args.device,
            data_overrides=data_overrides or None,
        )
    except Exception as exc:  # pragma: no cover - defensive skip for partial runs
        print(f"Skipping unfinished or invalid run: {version_dir}")
        print(f"Reason: {exc}")
        logging.getLogger(__name__).warning(
            "Skipping run %s due to load failure: %s", version_dir, exc
        )
        _cleanup_process_memory()
        return False

    try:
        output_img_dir = output_dir / "img"
        output_img_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loaded run directory: {version_dir}")
        print(f"Output directory: {output_dir}")
        if data_overrides:
            print(f"Data overrides: {data_overrides}")

        _print_config_summary(run)

        history = run.history()
        print("\n[history tail]")
        print(history.tail(8).to_string(index=False))

        print("\n[best epoch]")
        print(run.best_epoch())

        if not args.skip_history_plot:
            _save_history_plot(run, output_img_dir)

        metrics_df = run.metrics().sort_values("r2", ascending=False)
        print("\n[test metrics sorted by r2]")
        print(metrics_df.to_string(index=False))

        metrics_csv_path = output_dir / args.metrics_csv_name
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Wrote metrics CSV -> {metrics_csv_path}")

        if not args.skip_metrics_plot:
            _save_metrics_plot(metrics_df, output_img_dir)

        if not args.skip_field_plots:
            targets = args.targets or list(run.dataset.request_targets)
            step = max(1, args.plot_step)
            # Use the pre-flatten time dimension when available.
            # For MLP runs, dataset.targets is flattened to pixel-level rows,
            # which can be orders of magnitude larger than the number of
            # physical snapshots and makes plotting appear stuck.
            time_count = getattr(run.dataset, "targets_shape", run.dataset.targets.shape)[0]
            indices = list(range(0, time_count, step))
            if args.max_plots is not None:
                indices = indices[: max(0, args.max_plots)]

            print("\n[field plots]")
            print(f"targets: {targets}")
            print(f"plot_indices: {indices[:10]}{' ...' if len(indices) > 10 else ''}")

            # Compute predictions once and reuse across target plots.
            # This avoids repeated full-dataset forward passes that can trigger OOM.
            print("Precomputing predictions once for all field plots...")
            ground_truth, prediction = run.predict()

            for target_name in targets:
                print(f"Rendering target {target_name}")
                run.plot(
                    target_name,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    robust_quantile=args.robust_quantile,
                    error_mode=args.error_mode,
                    plot_indices=indices,
                    signed_target_names=args.signed_target_names,
                    show_figure=False,
                    output_dir=str(output_dir),
                )
                gc.collect()

        return True
    finally:
        # Explicitly drop large arrays/dataframes/models before next run in batch mode.
        del prediction
        del ground_truth
        del metrics_df
        del history
        del run
        _cleanup_process_memory()


def main() -> None:
    args = _parse_args()
    eval_dirs, from_run_root = _resolve_eval_dirs(args)

    if from_run_root:
        # In batch mode, bypass Lightning checkpoint migration paths that can
        # become unstable across repeated sequential loads on some HPC stacks.
        os.environ.setdefault("CLOSURE_FORCE_DIRECT_CKPT_LOAD", "1")

        root_output = (
            (args.output_dir.expanduser().resolve())
            if args.output_dir is not None
            else args.run_dir.expanduser().resolve()
        )
        root_output.mkdir(parents=True, exist_ok=True)
        _configure_eval_logging(root_output / "closure-eval.log")

        print(f"Batch mode: evaluating {len(eval_dirs)} run directories under {args.run_dir}")
        completed = 0
        skipped = 0
        for run_dir in eval_dirs:
            print(f"\n===== Evaluating: {run_dir.name} =====")
            output_dir = root_output / run_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)
            _cleanup_process_memory()
            if not _is_complete_run_dir(run_dir):
                print(f"Skipping unfinished run: {run_dir}")
                skipped += 1
                continue

            if _evaluate_single_run(run_dir, args, output_dir):
                completed += 1
            else:
                skipped += 1

        print("\nBatch evaluation complete.")
        print(f"Completed: {completed}")
        print(f"Skipped: {skipped}")
        return

    version_dir = eval_dirs[0]
    output_dir = (args.output_dir or version_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_eval_logging(output_dir / "closure-eval.log")

    if not _evaluate_single_run(version_dir, args, output_dir):
        raise RuntimeError(f"Evaluation failed for run directory: {version_dir}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
