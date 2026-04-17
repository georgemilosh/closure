#!/usr/bin/env python3
"""Summarize scaling runs stored under a Lightning CSV logger directory.

Examples
--------
python scripts/analyze_scaling_runs.py models/Lightning/Harris/Le/OS1600
python analyze_scaling_runs.py /abs/path/to/OS1600 --metric val_loss
python analyze_scaling_runs.py OS1600 --csv-out scaling_summary.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
import yaml


RUN_PATTERN = re.compile(
    r"^(?:(?P<arch>[A-Za-z0-9]+)_)?(?P<mode>[A-Za-z0-9]+)_(?P<nodes>\d+)n_(?P<gpus_per_node>\d+)g$"
)
DEFAULT_BASE_DIR = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize metrics.csv and timings.yaml for scaling runs under one logger name."
    )
    parser.add_argument(
        "logger_name",
        help="Logger directory name (for example OS1600) or an absolute path to that directory.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Parent directory that contains logger directories. Defaults to current working directory.",
    )
    parser.add_argument(
        "--metric",
        default="val_loss",
        help="Metric column to score runs with from metrics.csv. Default: val_loss.",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Treat larger metric values as better. Default behavior minimizes the metric.",
    )
    parser.add_argument(
        "--sort-by",
        default="total_gpus",
        choices=["run", "mode", "nodes", "gpus_per_node", "total_gpus", "training_s", "best_metric"],
        help="Column used to sort the summary table.",
    )
    parser.add_argument(
        "--csv-out",
        help="CSV output path. Defaults to <LOGGER_NAME>/scaling_summary.csv.",
    )
    return parser.parse_args()


def resolve_logger_dir(logger_name: str, base_dir: str) -> Path:
    logger_path = Path(logger_name).expanduser()
    if logger_path.is_absolute():
        return logger_path
    return Path(base_dir).expanduser().resolve() / logger_path


def collapse_metric_by_epoch(metrics: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in metrics.columns:
        return pd.DataFrame(columns=["epoch", column])

    valid = metrics.dropna(subset=[column]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["epoch", column])

    if "epoch" not in valid.columns:
        valid = valid.reset_index().rename(columns={"index": "epoch"})
        return valid[["epoch", column]]

    valid["epoch"] = valid["epoch"].bfill()
    valid = valid.dropna(subset=["epoch"])
    if valid.empty:
        return pd.DataFrame(columns=["epoch", column])

    collapsed = valid[["epoch", column]].groupby("epoch", as_index=False).last()
    collapsed["epoch"] = collapsed["epoch"].astype(int)
    return collapsed


def read_metric_summary(metrics_path: Path, metric_name: str, maximize: bool) -> dict:
    metrics = pd.read_csv(metrics_path)
    summary: dict[str, float | int | None] = {}

    for loss_name in ("train_loss", "val_loss"):
        collapsed = collapse_metric_by_epoch(metrics, loss_name)
        if collapsed.empty:
            summary[f"final_{loss_name}"] = None
            summary[f"final_{loss_name}_epoch"] = None
            continue
        final_row = collapsed.iloc[-1]
        summary[f"final_{loss_name}"] = float(final_row[loss_name])
        summary[f"final_{loss_name}_epoch"] = int(final_row["epoch"])

    metric_history = collapse_metric_by_epoch(metrics, metric_name)
    if metric_history.empty:
        summary["best_metric"] = None
        summary["best_metric_epoch"] = None
        summary["final_metric"] = None
        summary["final_metric_epoch"] = None
        summary["metric_name"] = metric_name
        return summary

    best_idx = metric_history[metric_name].idxmax() if maximize else metric_history[metric_name].idxmin()
    best_row = metric_history.loc[best_idx]
    final_row = metric_history.iloc[-1]

    summary["best_metric"] = float(best_row[metric_name])
    summary["best_metric_epoch"] = int(best_row["epoch"])
    summary["final_metric"] = float(final_row[metric_name])
    summary["final_metric_epoch"] = int(final_row["epoch"])
    summary["metric_name"] = metric_name
    return summary


def read_timings(timings_path: Path) -> dict:
    if not timings_path.exists():
        return {}
    with timings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def summarize_run(run_dir: Path, metric_name: str, maximize: bool) -> dict:
    run_name = run_dir.name
    summary: dict[str, object] = {
        "run": run_name,
        "path": str(run_dir),
        "arch": "default",
        "mode": "unknown",
        "nodes": None,
        "gpus_per_node": None,
        "total_gpus": None,
    }

    match = RUN_PATTERN.match(run_name)
    if match:
        nodes = int(match.group("nodes"))
        gpus_per_node = int(match.group("gpus_per_node"))
        summary.update(
            {
                "arch": match.group("arch") or "default",
                "mode": match.group("mode"),
                "nodes": nodes,
                "gpus_per_node": gpus_per_node,
                "total_gpus": nodes * gpus_per_node,
            }
        )

    metrics_path = run_dir / "metrics.csv"
    timings_path = run_dir / "timings.yaml"
    summary["has_metrics"] = metrics_path.exists()
    summary["has_timings"] = timings_path.exists()

    if metrics_path.exists():
        summary.update(read_metric_summary(metrics_path, metric_name=metric_name, maximize=maximize))
    else:
        summary.update(
            {
                "best_metric": None,
                "best_metric_epoch": None,
                "final_metric": None,
                "final_metric_epoch": None,
                "final_train_loss": None,
                "final_train_loss_epoch": None,
                "final_val_loss": None,
                "final_val_loss_epoch": None,
                "metric_name": metric_name,
            }
        )

    timings = read_timings(timings_path)
    summary["data_loading_s"] = timings.get("data_loading_s")
    summary["training_s"] = timings.get("training_s")
    summary["epochs"] = timings.get("epochs")
    summary["devices"] = timings.get("devices")
    summary["num_nodes"] = timings.get("num_nodes")
    summary["avg_ram_gb_per_epoch"] = timings.get("avg_ram_gb_per_epoch")
    summary["avg_gpu_utilization_pct_per_epoch"] = timings.get("avg_gpu_utilization_pct_per_epoch")
    summary["avg_gpu_memory_used_mb_per_epoch"] = timings.get("avg_gpu_memory_used_mb_per_epoch")
    summary["avg_ram_gb_during_loading"] = timings.get("avg_ram_gb_during_loading")
    summary["avg_gpu_utilization_pct_during_loading"] = timings.get("avg_gpu_utilization_pct_during_loading")
    summary["avg_gpu_memory_used_mb_during_loading"] = timings.get("avg_gpu_memory_used_mb_during_loading")

    data_loading_s = summary["data_loading_s"]
    training_s = summary["training_s"]
    if data_loading_s is not None and training_s is not None:
        summary["total_runtime_s"] = float(data_loading_s) + float(training_s)
    else:
        summary["total_runtime_s"] = None

    if training_s is not None and summary["epochs"] not in (None, 0):
        summary["sec_per_epoch"] = float(training_s) / float(summary["epochs"])
    else:
        summary["sec_per_epoch"] = None

    if training_s is not None and summary["total_gpus"] not in (None, 0):
        summary["gpu_seconds"] = float(training_s) * float(summary["total_gpus"])
    else:
        summary["gpu_seconds"] = None

    return summary


def add_scaling_columns(frame: pd.DataFrame, maximize: bool) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["speedup"] = pd.NA
    frame["efficiency"] = pd.NA

    for (_, _), group in frame.groupby(["mode", "arch"], sort=False):
        candidates = group.dropna(subset=["training_s", "total_gpus"])
        if candidates.empty:
            continue

        baseline = candidates.sort_values(["total_gpus", "training_s", "run"]).iloc[0]
        base_time = float(baseline["training_s"])
        base_gpus = float(baseline["total_gpus"])

        for idx in group.index:
            row = frame.loc[idx]
            if pd.notna(row["training_s"]) and pd.notna(row["total_gpus"]):
                gpu_ratio = float(row["total_gpus"]) / base_gpus
                speedup = base_time / float(row["training_s"])
                frame.at[idx, "speedup"] = speedup
                frame.at[idx, "efficiency"] = speedup / gpu_ratio if gpu_ratio else pd.NA

    return frame


def discover_runs(logger_dir: Path) -> list[Path]:
    run_dirs = []
    for path in sorted(logger_dir.iterdir()):
        if not path.is_dir():
            continue
        if (path / "metrics.csv").exists() or (path / "timings.yaml").exists():
            run_dirs.append(path)
    return run_dirs


def build_summary(logger_dir: Path, metric_name: str, maximize: bool) -> pd.DataFrame:
    rows = [summarize_run(run_dir, metric_name=metric_name, maximize=maximize) for run_dir in discover_runs(logger_dir)]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return add_scaling_columns(frame, maximize=maximize)


def print_mode_summary(frame: pd.DataFrame, metric_name: str, maximize: bool) -> None:
    for (mode, arch), group in frame.groupby(["mode", "arch"], sort=False):
        print(f"\n[{mode} | {arch}]")

        valid_time = group.dropna(subset=["training_s"])
        if not valid_time.empty:
            fastest = valid_time.loc[valid_time["training_s"].idxmin()]
            print(
                "fastest run: "
                f"{fastest['run']} ({fastest['training_s']:.3f}s training, "
                f"speedup={fastest['speedup'] if pd.notna(fastest['speedup']) else math.nan:.3f})"
            )

        valid_metric = group.dropna(subset=["best_metric"])
        if not valid_metric.empty:
            idx = valid_metric["best_metric"].idxmax() if maximize else valid_metric["best_metric"].idxmin()
            best = valid_metric.loc[idx]
            print(
                "best metric: "
                f"{best['run']} ({metric_name}={best['best_metric']:.6g} at epoch {int(best['best_metric_epoch'])})"
            )


def main() -> None:
    args = parse_args()
    logger_dir = resolve_logger_dir(args.logger_name, args.base_dir)
    if not logger_dir.exists():
        raise FileNotFoundError(f"Logger directory does not exist: {logger_dir}")
    if not logger_dir.is_dir():
        raise NotADirectoryError(f"Logger path is not a directory: {logger_dir}")

    summary = build_summary(logger_dir, metric_name=args.metric, maximize=args.maximize)
    if summary.empty:
        raise FileNotFoundError(f"No runs with metrics.csv or timings.yaml found in: {logger_dir}")

    sort_ascending = args.sort_by != "best_metric" or not args.maximize
    if args.sort_by == "training_s":
        sort_ascending = True
    summary = summary.sort_values(by=args.sort_by, ascending=sort_ascending, na_position="last").reset_index(drop=True)

    display_columns = [
        "run",
        "arch",
        "mode",
        "nodes",
        "gpus_per_node",
        "total_gpus",
        "training_s",
        "data_loading_s",
        "sec_per_epoch",
        "avg_ram_gb_per_epoch",
        "avg_gpu_utilization_pct_per_epoch",
        "avg_gpu_memory_used_mb_per_epoch",
        "avg_ram_gb_during_loading",
        "avg_gpu_utilization_pct_during_loading",
        "avg_gpu_memory_used_mb_during_loading",
        "speedup",
        "efficiency",
        "best_metric",
        "best_metric_epoch",
        "final_metric",
        "final_train_loss",
        "final_val_loss",
    ]
    display = summary[display_columns].copy()

    numeric_cols = display.select_dtypes(include=["number"]).columns
    display[numeric_cols] = display[numeric_cols].round(6)

    print(f"Logger directory: {logger_dir}")
    print(f"Metric: {args.metric} ({'maximize' if args.maximize else 'minimize'})")
    print(display.to_string(index=False))
    print_mode_summary(summary, metric_name=args.metric, maximize=args.maximize)

    csv_path = Path(args.csv_out).expanduser() if args.csv_out else logger_dir / "scaling_summary.csv"
    if not csv_path.is_absolute():
        csv_path = logger_dir / csv_path
    display.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary to {csv_path}")


if __name__ == "__main__":
    main()