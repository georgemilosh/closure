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
from pathlib import Path

import matplotlib.pyplot as plt

from closure.run_loader import RunLoader


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
        help="Alias of --version-dir (for workflows using run_* directory names).",
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
        default=["Pxy_e", "Pxz_e", "Pyz_e"],
        help="Target names treated as signed for diverging colormap scaling.",
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
    n_panels = 1 + (1 if lr_cols else 0)

    fig, axes = plt.subplots(1, n_panels, figsize=(14, 4))
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_title("Loss vs epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if lr_cols:
        for col in lr_cols:
            axes[1].plot(history["epoch"], history[col], label=col)
        axes[1].set_title("Learning rate vs epoch")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("lr")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

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


def main() -> None:
    args = _parse_args()
    version_dir = _select_version_dir(args)

    data_overrides = {}
    if args.test_samples_file:
        data_overrides["test_samples_file"] = args.test_samples_file

    run = RunLoader.from_version_dir(
        version_dir,
        stage=args.stage,
        ckpt=args.ckpt,
        device=args.device,
        data_overrides=data_overrides or None,
    )

    output_dir = (args.output_dir or version_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
        indices = list(range(0, run.dataset.targets.shape[0], step))
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

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
