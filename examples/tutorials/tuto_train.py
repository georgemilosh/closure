#!/usr/bin/env python
"""
Tutorial: End-to-End Closure Training
======================================

Self-contained tutorial that uses the small fixture dataset shipped with
the repo (``tests/fixtures/ecsim_tiny``).  No ``paths.yaml`` or external
data needed.

Practical workflow demonstrating:

1. Create a YAML experiment config
2. Train with Lightning (custom checkpointing + early stopping)
3. Evaluate with MSE and R² (plus per-channel regression diagnostics)
4. Visualize diagonal and off-diagonal pressure channels
5. Export deployable ``.pt`` artifacts (bundle + TorchScript)

Usage
-----
Run from the project root::

    python examples/tutorials/tuto_train.py            # quick training on fixture data
    python examples/tutorials/tuto_train.py --no-train # skip training, evaluate existing checkpoints
    python examples/tutorials/tuto_train.py --help

Prerequisites
-------------
* Python environment with ``closure`` installed (``pip install -e .``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Enable closure's logger.info messages (data loading, normalization, shapes)
_closure_logger = logging.getLogger("closure")
_closure_logger.setLevel(logging.INFO)
_stream_fmt = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_stream_fmt)
_closure_logger.addHandler(_stream_handler)


def _attach_file_logger(log_dir: Path) -> None:
    """Add a timestamped FileHandler so every closure.* message is persisted.

    Also attaches the same handler to the Lightning logger so that model
    summaries and training progress appear in ``closure.log``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    file_fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    fh = logging.FileHandler(log_dir / "closure.log")
    fh.setFormatter(file_fmt)
    _closure_logger.addHandler(fh)

    # Mirror Lightning output into the same log file.
    lightning_logger = logging.getLogger("lightning.pytorch")
    lightning_logger.addHandler(fh)

matplotlib.use("Agg")  # non-interactive backend; safe on headless machines
import matplotlib.pyplot as plt
import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from closure import ClosureDataModule, ClosureLitModule, MLP, load_paths
import closure.evaluation as ev
from closure.visualization import plot_pred_targets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end Closure training tutorial (uses bundled fixture data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=3,
        help="Maximum training epochs (default: 3)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training; only run evaluation on existing checkpoints.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving visualization plots.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1) Paths
# ---------------------------------------------------------------------------

def setup_paths():
    project_root = Path(__file__).resolve().parents[2]

    # Use paths.yaml for work_dir; override data_dir to point at fixtures
    paths = load_paths(str(project_root / "paths.yaml"))
    data_root = project_root / "tests" / "fixtures"
    work_dir = Path(paths["work_dir"]) / "tuto"
    split_dir = data_root / "ecsim_tiny"  # pre-made train/val/test.csv already there

    config_dir = work_dir / "configs"
    artifact_dir = work_dir / "artifacts"

    for p in [work_dir, config_dir, artifact_dir]:
        p.mkdir(parents=True, exist_ok=True)

    _closure_logger.info("project_root: %s", project_root)
    _closure_logger.info("data_root   : %s", data_root)
    _closure_logger.info("work_dir    : %s", work_dir)
    _closure_logger.info("split_dir   : %s", split_dir)
    _closure_logger.info("config_dir  : %s", config_dir)
    _closure_logger.info("artifact_dir: %s", artifact_dir)

    return project_root, work_dir, split_dir, config_dir, artifact_dir


# ---------------------------------------------------------------------------
# 2) Write experiment config
# ---------------------------------------------------------------------------

def write_experiment_config(config_dir: Path, args: argparse.Namespace) -> Path:
    # Paths prefixed with ./ are resolved against CWD by _resolve_path().
    # norm_folder is a bare identifier resolved against work_dir from paths.yaml.
    data_folder_cfg = "./tests/fixtures/ecsim_tiny"
    norm_folder_cfg = "tuto"

    run_config = {
        "name": "baseline",
        "data": {
            "data_folder": data_folder_cfg,
            "norm_folder": norm_folder_cfg,
            "flatten": True,
            "scaler_features": True,
            "scaler_targets": True,
            "prescaler_features": [
                "arcsinh", "arcsinh", "arcsinh", "arcsinh", "arcsinh", "arcsinh",
                None, None, None, None,
            ],
            "prescaler_targets": ["log", "log", "log", None, None, None],
            "read_features_targets_kwargs": {
                "fields_to_read": {
                    "B": True, "E": True, "B_ext": False, "E_ext": False,
                    "divB": False, "rho": True, "N": False, "Qrem": False,
                    "J": True, "P": True, "PI": True, "divP": False,
                    "Ohmres": False, "Heat_flux": False, "gyro_radius": False,
                    "EF": False,
                },
                "request_features": [
                    "rho_e",
                    "Bx", "By", "Bz",
                    "Jx_e", "Jy_e", "Jz_e",
                    "Vx_e", "Vy_e", "Vz_e",
                ],
                "request_targets": ["Pxx_e", "Pyy_e", "Pzz_e", "Pxy_e", "Pxz_e", "Pyz_e"],
                "choose_species": ["e", None],
                "choose_x": [0, 128],
                "choose_y": [0, 128],
                "verbose": False,
            },
        "train_samples_file": "./tests/fixtures/ecsim_tiny/train.csv",
        "val_samples_file": "./tests/fixtures/ecsim_tiny/val.csv",
        "test_samples_file": "./tests/fixtures/ecsim_tiny/test.csv",
        },
        "model": {
            "feature_dims": [10, 60, 80, 50, 40, 6],
            "activations": ["Tanh", "Tanh", "Tanh", "Tanh", None],
        },
        "optimizer": {
            "criterion": "MSELoss",
            "metrics": ["R2Score"],
            "optimizer": "Adam",
            "lr": 5e-4,
            "weight_decay": 0.0,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_kwargs": {
                "mode": "min", "factor": 0.6, "patience": 4, "threshold": 1e-3,
            },
        },
        "trainer": {
            "batch_size": args.batch_size,
            "num_workers": 8,
            "max_epochs": args.max_epochs,
            "accelerator": "gpu",
            "devices": 1,
            "precision": "16-mixed",
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 25,
        },
    }

    config_path = config_dir / "model.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    _closure_logger.info("Wrote config: %s", config_path)
    _closure_logger.info("\n%s\n...", config_path.read_text()[:1200])
    return config_path


# ---------------------------------------------------------------------------
# 3) Build DataModule, Model, Trainer
# ---------------------------------------------------------------------------

def build_components(config_path: Path, work_dir: Path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimizer"]
    trainer_cfg = cfg["trainer"]

    # Paths in the config use ./ prefix (CWD-relative) or bare identifiers
    # (resolved against paths.yaml); _resolve_path() handles both in setup().
    datamodule = ClosureDataModule(
        data_folder=data_cfg["data_folder"],
        norm_folder=data_cfg["norm_folder"],
        train_samples_file=data_cfg["train_samples_file"],
        val_samples_file=data_cfg["val_samples_file"],
        test_samples_file=data_cfg["test_samples_file"],
        batch_size=trainer_cfg["batch_size"],
        num_workers=trainer_cfg["num_workers"],
        flatten=data_cfg["flatten"],
        scaler_features=data_cfg["scaler_features"],
        scaler_targets=data_cfg["scaler_targets"],
        prescaler_features=data_cfg["prescaler_features"],
        prescaler_targets=data_cfg["prescaler_targets"],
        read_features_targets_kwargs=data_cfg["read_features_targets_kwargs"],
    )

    network = MLP(
        feature_dims=model_cfg["feature_dims"],
        activations=model_cfg["activations"],
    )

    scheduler_kwargs = dict(opt_cfg.get("scheduler_kwargs", {}))
    scheduler_kwargs.pop("scheduler", None)

    module = ClosureLitModule(
        network=network,
        criterion=opt_cfg["criterion"],
        metrics=opt_cfg.get("metrics"),
        optimizer=opt_cfg["optimizer"],
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 0.0),
        scheduler=opt_cfg.get("scheduler"),
        scheduler_kwargs=scheduler_kwargs,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=work_dir / "checkpoints",
        filename="epoch{epoch:03d}-valloss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    logger = CSVLogger(save_dir=str(work_dir), name="lightning_logs")

    requested_accelerator = trainer_cfg["accelerator"]
    requested_devices = trainer_cfg["devices"]
    trainer_precision = trainer_cfg["precision"]

    if requested_accelerator in {"gpu", "cuda"} and not torch.cuda.is_available():
        _closure_logger.info("CUDA unavailable; falling back to CPU.")
        trainer_accelerator = "cpu"
        trainer_devices = 1
        if trainer_precision == "16-mixed":
            trainer_precision = "bf16-mixed"
    else:
        trainer_accelerator = requested_accelerator
        trainer_devices = requested_devices

    trainer = L.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        accelerator=trainer_accelerator,
        devices=trainer_devices,
        precision=trainer_precision,
        gradient_clip_val=trainer_cfg["gradient_clip_val"],
        log_every_n_steps=trainer_cfg["log_every_n_steps"],
        default_root_dir=str(work_dir),
        callbacks=[ckpt_callback, early_stop],
        logger=logger,
    )

    return cfg, datamodule, network, module, trainer, ckpt_callback


# ---------------------------------------------------------------------------
# 4) Train
# ---------------------------------------------------------------------------

def run_training(module, datamodule, trainer, ckpt_callback):
    _closure_logger.info("--- Starting training ---")
    datamodule.setup("fit")
    _closure_logger.info("  train samples: %d", len(datamodule.train_dataset))
    _closure_logger.info("  val samples  : %d", len(datamodule.val_dataset))

    trainer.fit(module, datamodule=datamodule)
    _closure_logger.info("Best checkpoint: %s", ckpt_callback.best_model_path)


# ---------------------------------------------------------------------------
# 5) Evaluate
# ---------------------------------------------------------------------------

def run_evaluation(module, network, datamodule, ckpt_callback, work_dir: Path):
    _closure_logger.info("--- Evaluation ---")

    best_ckpt = ckpt_callback.best_model_path if ckpt_callback.best_model_path else None
    if best_ckpt and Path(best_ckpt).exists():
        module_eval = ClosureLitModule.load_from_checkpoint(best_ckpt, network=network)
        _closure_logger.info("Loaded checkpoint: %s", best_ckpt)
    else:
        module_eval = module
        _closure_logger.info("No checkpoint found, using current module state.")

    datamodule.setup("test")
    ground_truth, prediction = ev.transform_targets(
        module_eval,
        datamodule.test_dataset,
        target_channels=datamodule.target_channels,
        rescale=False,
        renorm=False,
        verbose=False,
        reshape=False,
        test_features=datamodule.test_dataset.features,
    )
    _closure_logger.info("prediction shape : %s", prediction.shape)
    _closure_logger.info("ground_truth shape: %s", ground_truth.shape)

    # MSE and R² reports
    mse_report = ev.evaluate_loss(
        datamodule.test_dataset, ground_truth, prediction, "MSELoss",
        target_channels=datamodule.target_channels, verbose=True,
    )
    r2_report = ev.evaluate_loss(
        datamodule.test_dataset, ground_truth, prediction, "r2",
        target_channels=datamodule.target_channels, verbose=True,
    )

    # Per-channel regression table
    metrics_df = ev.evaluate_regression_metrics(
        datamodule.test_dataset, ground_truth, prediction,
        target_channels=datamodule.target_channels,
    )
    _closure_logger.info("Per-channel regression metrics:\n%s",
                         metrics_df.sort_values("r2", ascending=False).to_string(index=False))

    return module_eval, ground_truth, prediction, metrics_df


# ---------------------------------------------------------------------------
# 6) Plot metrics and predictions
# ---------------------------------------------------------------------------

def save_metric_plots(metrics_df: pd.DataFrame, work_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ms = metrics_df.set_index("channel").sort_index()
    ms["r2"].plot.bar(ax=axes[0], color="steelblue", alpha=0.85)
    axes[0].set_title("Channel-wise R²")
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].grid(axis="y", alpha=0.3)

    ms["nrmse"].plot.bar(ax=axes[1], color="darkorange", alpha=0.85)
    axes[1].set_title("Channel-wise normalized RMSE")
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.set_xlabel("target channel")

    out = work_dir / "channel_metrics.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    _closure_logger.info("Saved: %s", out)


def save_training_curves(work_dir: Path):
    log_root = work_dir / "lightning_logs"
    version_dirs = sorted(log_root.glob("version_*"))
    if not version_dirs:
        _closure_logger.info("No training logs found, skipping curve plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    for vdir in version_dirs[-3:]:
        csv = vdir / "metrics.csv"
        if not csv.exists():
            continue
        m = pd.read_csv(csv)
        if "val_loss" in m.columns:
            val_df = m.dropna(subset=["val_loss"])
            ax1.plot(val_df["epoch"], val_df["val_loss"], label=f"{vdir.name} val")
        if "train_loss" in m.columns:
            tr_df = m.dropna(subset=["train_loss"])
            ax1.plot(tr_df["epoch"], tr_df["train_loss"], ls=":", label=f"{vdir.name} train")
        r2_col = next((c for c in m.columns if "R2Score" in c and "val" in c), None)
        if r2_col:
            r2_df = m.dropna(subset=[r2_col])
            ax2.plot(r2_df["epoch"], r2_df[r2_col], label=f"{vdir.name} val R²")

    ax1.set(xlabel="epoch", ylabel="MSE loss", title="Training / validation loss")
    ax1.grid(alpha=0.3); ax1.legend()
    ax2.set(xlabel="epoch", ylabel="R²", title="R² score")
    ax2.grid(alpha=0.3); ax2.legend()

    out = work_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    _closure_logger.info("Saved: %s", out)


def save_prediction_plots(cfg, datamodule, prediction, ground_truth, work_dir: Path):
    data_cfg = cfg["data"]
    resolved_data_folder = str(ClosureDataModule._resolve_path(data_cfg["data_folder"], "data_dir"))

    n_timesteps = datamodule.test_dataset.targets_shape[0]
    plot_indices = [i for i in [0, 1, 2] if i < n_timesteps]

    # Diagonal example
    plot_pred_targets(
        datamodule.test_dataset, "Pxx_e", prediction, ground_truth,
        data_folder=resolved_data_folder,
        read_features_targets_kwargs=data_cfg["read_features_targets_kwargs"],
        target_channels=datamodule.target_channels,
        output_dir=str(work_dir),
        plot_indices=plot_indices,
        figsize=(12, 8),
        robust_quantile=0.995,
        error_mode="relative",
        signed_target_names=["Pxy_e", "Pxz_e", "Pyz_e"],
    )
    _closure_logger.info("Saved Pxx_e prediction plots to: %s", work_dir)

    # Off-diagonal example
    target_to_plot = (
        "Pxz_e" if "Pxz_e" in datamodule.test_dataset.request_targets
        else datamodule.test_dataset.request_targets[0]
    )
    plot_pred_targets(
        datamodule.test_dataset, target_to_plot, prediction, ground_truth,
        data_folder=resolved_data_folder,
        read_features_targets_kwargs=data_cfg["read_features_targets_kwargs"],
        target_channels=datamodule.target_channels,
        output_dir=str(work_dir),
        plot_indices=plot_indices,
        figsize=(12, 8),
        robust_quantile=0.995,
        error_mode="symmetric_percent",
        error_limit=0.5,
        signed_target_names=["Pxy_e", "Pxz_e", "Pyz_e"],
    )
    _closure_logger.info("Saved %s prediction plots to: %s", target_to_plot, work_dir)


# ---------------------------------------------------------------------------
# 7) Compare runs
# ---------------------------------------------------------------------------

def compare_logged_runs(work_dir: Path):
    log_dirs = [str(p) for p in sorted((work_dir / "lightning_logs").glob("version_*"))[-5:]]
    if not log_dirs:
        _closure_logger.info("No logged runs to compare.")
        return

    run_summary = ev.compare_runs(log_dirs=log_dirs, metric_key="val_loss")
    run_summary = run_summary.sort_values("best_val_loss", na_position="last")
    _closure_logger.info("Run comparison:\n%s", run_summary.to_string(index=False))

    if not run_summary.dropna(subset=["best_val_loss"]).empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = run_summary.dropna(subset=["best_val_loss"]).copy()
        plot_df["run"] = [Path(x).name for x in plot_df["log_dir"]]
        ax.bar(plot_df["run"], plot_df["best_val_loss"], color="teal", alpha=0.85)
        ax.set(ylabel="best val_loss", xlabel="run", title="Best validation loss across runs")
        plt.xticks(rotation=20)
        out = work_dir / "run_comparison.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        _closure_logger.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# 8) Export deployable artifacts
# ---------------------------------------------------------------------------

def export_artifacts(module_eval, cfg, datamodule, work_dir: Path):
    _closure_logger.info("--- Exporting artifacts ---")
    artifact_dir = work_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg["model"]
    ds = datamodule.test_dataset

    # Inference bundle
    bundle_path = artifact_dir / "inference_bundle.pt"
    artifact_bundle = {
        "state_dict": module_eval.network.state_dict(),
        "model_kwargs": {
            "feature_dims": model_cfg["feature_dims"],
            "activations": model_cfg["activations"],
        },
        "request_features": ds.request_features,
        "request_targets": ds.request_targets,
        "target_channels": datamodule.target_channels,
        "feature_channels": datamodule.feature_channels,
        "features_mean": np.asarray(ds.features_mean),
        "features_std": np.asarray(ds.features_std),
        "targets_mean": np.asarray(ds.targets_mean),
        "targets_std": np.asarray(ds.targets_std),
        "prescaler_features": [None if f is None else f.__name__ for f in ds.prescaler_features],
        "prescaler_targets": [None if f is None else f.__name__ for f in ds.prescaler_targets],
        "flatten": ds.flatten,
    }
    torch.save(artifact_bundle, bundle_path)
    _closure_logger.info("Saved inference bundle: %s", bundle_path)

    # TorchScript model
    torchscript_path = artifact_dir / "torchscript.pt"
    module_eval.network.eval()
    example_input = torch.randn(32, model_cfg["feature_dims"][0])
    try:
        scripted = torch.jit.script(module_eval.network.cpu())
    except Exception as err:
        _closure_logger.info("script failed, falling back to trace: %s", err)
        scripted = torch.jit.trace(module_eval.network.cpu(), example_input)
    scripted.save(str(torchscript_path))
    _closure_logger.info("Saved TorchScript model: %s", torchscript_path)

    # Quick sanity check
    jit_model = torch.jit.load(str(torchscript_path), map_location="cpu")
    with torch.no_grad():
        y_jit = jit_model(example_input)
    _closure_logger.info("TorchScript forward check shape: %s", tuple(y_jit.shape))

    return bundle_path, torchscript_path


# ---------------------------------------------------------------------------
# 9) Run manifest
# ---------------------------------------------------------------------------

def save_manifest(config_path, train_csv, val_csv, test_csv, best_ckpt, bundle_path, torchscript_path, work_dir):
    artifact_dir = work_dir / "artifacts"
    manifest = {
        "config_path": str(config_path),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        "bundle_path": str(bundle_path),
        "torchscript_path": str(torchscript_path),
        "work_dir": str(work_dir),
    }
    manifest_path = artifact_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    _closure_logger.info("Saved manifest: %s", manifest_path)
    _closure_logger.info("\n%s", json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1) Paths
    project_root, work_dir, split_dir, config_dir, artifact_dir = setup_paths()
    _attach_file_logger(work_dir)

    # 2) Config
    config_path = write_experiment_config(config_dir, args)

    # 3) Data splits (pre-made CSVs shipped with fixtures)
    _closure_logger.info("--- Using fixture splits ---")
    for name in ["train.csv", "val.csv", "test.csv"]:
        df = pd.read_csv(split_dir / name)
        _closure_logger.info("  %s: %d rows", name, len(df))

    # 4) Build components — paths resolved by _resolve_path() inside DataModule
    cfg, datamodule, network, module, trainer, ckpt_callback = build_components(
        config_path, work_dir,
    )

    # 4) Train
    if not args.no_train:
        run_training(module, datamodule, trainer, ckpt_callback)

    # 5) Evaluate
    module_eval, ground_truth, prediction, metrics_df = run_evaluation(
        module, network, datamodule, ckpt_callback, work_dir,
    )

    # 6) Plots
    if not args.no_plots:
        save_metric_plots(metrics_df, work_dir)
        save_training_curves(work_dir)
        save_prediction_plots(cfg, datamodule, prediction, ground_truth, work_dir)

    # 7) Compare runs
    compare_logged_runs(work_dir)

    # 8) Export artifacts
    bundle_path, torchscript_path = export_artifacts(module_eval, cfg, datamodule, work_dir)

    # 9) Manifest
    best_ckpt = ckpt_callback.best_model_path if ckpt_callback.best_model_path else None
    data_cfg = cfg["data"]
    save_manifest(
        config_path,
        data_cfg["train_samples_file"], data_cfg["val_samples_file"], data_cfg["test_samples_file"],
        best_ckpt, bundle_path, torchscript_path, work_dir,
    )

    _closure_logger.info("=== Tutorial complete ===")
    _closure_logger.info("All outputs saved under: %s", work_dir)


if __name__ == "__main__":
    main()
