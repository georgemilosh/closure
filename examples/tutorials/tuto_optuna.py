#!/usr/bin/env python
"""
Tutorial: Optuna Hyperparameter Sweep for Harris Current Sheet
================================================================

Demonstrates the Optuna integration in closure:

1. Configure and launch an HP sweep via ``harris_optuna_sweep.py``
2. Load and inspect the resulting Optuna study
3. Visualize optimization history, parameter importances, and more
4. Export the best config for production training

Usage
-----
Run from the project root::

    python examples/tutorials/tuto_optuna.py                     # default (12 trials, cpu)
    python examples/tutorials/tuto_optuna.py --n-trials 50 --accelerator gpu
    python examples/tutorials/tuto_optuna.py --no-sweep          # analyse an existing study only
    python examples/tutorials/tuto_optuna.py --help

Prerequisites
-------------
* ``closure`` installed (``pip install -e .``).
* ``optuna`` and ``plotly`` available (included in ``requirements-hp.txt``).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import optuna
import pandas as pd

# Enable closure's logger.info messages (data loading, normalization, shapes)
_closure_logger = logging.getLogger("closure")
_closure_logger.setLevel(logging.INFO)
_stream_fmt = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_stream_fmt)
_closure_logger.addHandler(_stream_handler)


def _attach_file_logger(log_dir: Path) -> None:
    """Add a timestamped FileHandler so every closure.* message is persisted."""
    log_dir.mkdir(parents=True, exist_ok=True)
    file_fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    fh = logging.FileHandler(log_dir / "closure.log")
    fh.setFormatter(file_fmt)
    _closure_logger.addHandler(fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HP sweep tutorial for the Harris current-sheet experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--variant", default="default", choices=["default", "noE", "noJ", "noJnoE"],
                    help="Feature variant (default: default)")
    p.add_argument("--task", default="P", choices=["P", "divP"],
                    help="Target task (default: P)")
    p.add_argument("--search-profile", default="fast", choices=["fast", "balanced", "thorough"],
                    help="Optuna search profile (default: fast)")
    p.add_argument("--n-trials", type=int, default=12,
                    help="Number of Optuna trials (default: 12)")
    p.add_argument("--max-epochs", type=int, default=4,
                    help="Max epochs per trial (default: 4)")
    p.add_argument("--accelerator", default="cpu",
                    help="Lightning accelerator (default: cpu)")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--reset-study", action="store_true", default=True,
                    help="Delete previous study DB before running (default: True)")
    p.add_argument("--no-reset-study", action="store_false", dest="reset_study",
                    help="Keep existing study DB (resume)")
    p.add_argument("--no-sweep", action="store_true",
                    help="Skip the sweep; only analyse an existing study.")
    p.add_argument("--no-plots", action="store_true",
                    help="Skip saving visualization plots.")
    p.add_argument("--output-root", type=str, default=None,
                    help="Root for sweep outputs (default: <project_root>/dev/optuna_tutorial)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Shared parameter subset for visualization (present in every completed trial)
SHARED_PLOT_PARAMS = [
    "architecture", "activation", "batch_norm", "dropout",
    "lr", "weight_decay", "batch_size",
    "gradient_clip_val", "scheduler_family",
]
CONTOUR_PARAMS = ["lr", "weight_decay", "dropout"]


def _save_plotly(fig, path: Path):
    """Save a Plotly figure to static HTML and, if kaleido is available, PNG."""
    fig.write_html(str(path.with_suffix(".html")))
    try:
        fig.write_image(str(path.with_suffix(".png")), scale=2)
    except Exception:
        pass  # kaleido not installed
    print("  Saved:", path.with_suffix(".html"))


# ---------------------------------------------------------------------------
# 1) Run the sweep
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace, output_root: Path, study_name: str,
              db_path: Path, storage: str, study_dir: Path):
    if args.reset_study:
        if db_path.exists():
            db_path.unlink()
            print(f"Removed existing study DB: {db_path}")
        if study_dir.exists():
            shutil.rmtree(study_dir)
            print(f"Removed existing study directory: {study_dir}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "examples" / "optuna" / "harris_optuna_sweep.py"),
        "--variant", args.variant,
        "--task", args.task,
        "--search-profile", args.search_profile,
        "--n-trials", str(args.n_trials),
        "--max-epochs", str(args.max_epochs),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
        "--study-name", study_name,
        "--storage", storage,
        "--output-root", str(output_root),
        "--disable-progress-bar",
    ]

    print("\n--- Running Optuna sweep ---")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    print("\nReturn code:", result.returncode)
    print("--- STDOUT (tail) ---")
    print("\n".join(result.stdout.splitlines()[-25:]))
    if result.returncode != 0:
        print("--- STDERR (tail) ---")
        print("\n".join(result.stderr.splitlines()[-25:]))
        raise RuntimeError("Sweep failed. Inspect stderr above.")


# ---------------------------------------------------------------------------
# 2) Load and inspect the study
# ---------------------------------------------------------------------------

def load_and_inspect(study_name: str, storage: str, output_root: Path):
    print("\n--- Loading study ---")
    study = optuna.load_study(study_name=study_name, storage=storage)

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    print(f"Finished trials : {len(completed)}")
    print(f"Best value      : {study.best_value}")
    print(f"Best trial #    : {study.best_trial.number}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Trial table
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df_sorted = df.sort_values("value", ascending=True)
    print("\nTop 10 trials:")
    print(df_sorted.head(10).to_string(index=False))

    csv_out = output_root / f"{study_name}_trials.csv"
    df_sorted.to_csv(csv_out, index=False)
    print("Saved trial table:", csv_out)

    common_params = sorted(
        set.intersection(*(set(t.params.keys()) for t in completed))
    ) if completed else []
    print("Common params across completed trials:", common_params)

    return study


# ---------------------------------------------------------------------------
# 3) Visualization
# ---------------------------------------------------------------------------

def save_visualizations(study: optuna.Study, output_root: Path, study_name: str):
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
        plot_contour,
    )

    plot_dir = output_root / f"{study_name}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Saving visualizations to {plot_dir} ---")

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if len(completed) < 2:
        print("Not enough completed trials for meaningful plots. Skipping.")
        return

    # Optimisation history
    fig = plot_optimization_history(study)
    _save_plotly(fig, plot_dir / "optimization_history")

    # Parameter importances (shared subset only)
    try:
        fig = plot_param_importances(study, params=SHARED_PLOT_PARAMS)
        _save_plotly(fig, plot_dir / "param_importances")
    except Exception as e:
        print(f"  Skipped param_importances: {e}")

    # Parallel coordinate
    try:
        fig = plot_parallel_coordinate(study, params=SHARED_PLOT_PARAMS)
        _save_plotly(fig, plot_dir / "parallel_coordinate")
    except Exception as e:
        print(f"  Skipped parallel_coordinate: {e}")

    # Slice plot
    try:
        fig = plot_slice(study, params=SHARED_PLOT_PARAMS)
        _save_plotly(fig, plot_dir / "slice")
    except Exception as e:
        print(f"  Skipped slice: {e}")

    # Contour plot
    try:
        fig = plot_contour(study, params=CONTOUR_PARAMS)
        _save_plotly(fig, plot_dir / "contour")
    except Exception as e:
        print(f"  Skipped contour: {e}")


# ---------------------------------------------------------------------------
# 4) Show best exported config
# ---------------------------------------------------------------------------

def show_best_config(output_root: Path, study_name: str):
    best_cfg = output_root / study_name / "best_config.yaml"
    print("\n--- Best config ---")
    if best_cfg.exists():
        print("Path:", best_cfg)
        print("To retrain with the best hyperparameters:\n")
        print(f"  closure-train fit --config {best_cfg} --trainer.fast_dev_run=true\n")
        print(best_cfg.read_text()[:1500])
    else:
        print("best_config.yaml not found at", best_cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "dev" / "optuna_tutorial"
    output_root.mkdir(parents=True, exist_ok=True)
    _attach_file_logger(output_root)

    study_name = f"harris_{args.variant}_{args.task}_tutorial_viz"
    db_path = output_root / f"{study_name}.db"
    storage = f"sqlite:///{db_path}"
    study_dir = output_root / study_name

    print("Study name  :", study_name)
    print("Storage     :", storage)
    print("Output root :", output_root)

    # 1) Run sweep
    if not args.no_sweep:
        run_sweep(args, output_root, study_name, db_path, storage, study_dir)

    # 2) Inspect
    study = load_and_inspect(study_name, storage, output_root)

    # 3) Visualize
    if not args.no_plots:
        save_visualizations(study, output_root, study_name)

    # 4) Best config
    show_best_config(output_root, study_name)

    print("\n=== Optuna tutorial complete ===")
    print("All outputs saved under:", output_root)


if __name__ == "__main__":
    main()
