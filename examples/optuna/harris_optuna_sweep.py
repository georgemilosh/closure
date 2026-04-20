#!/usr/bin/env python3
"""Optuna sweep tailored to Harris Le2GEM15ppc FCNN experiments.

Usage
-----
    pip install -e ".[hp]"
    python examples/optuna/harris_optuna_sweep.py \
      --variant default \
      --task P \
      --n-trials 50 \
      --storage sqlite:///harris_default_P.db

Path conventions: ``--data-folder`` and ``--train-samples`` / ``--val-samples``
accept paths relative to ``data_dir`` from ``paths.yaml``.  The
``--output-root`` is resolved to an absolute path; all derived trial
directories and norm folders therefore stay absolute and bypass resolution.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

COMMON_FIELDS = {
    "B": True,
    "B_ext": False,
    "divB": False,
    "E": True,
    "E_ext": False,
    "rho": True,
    "J": True,
    "P": True,
    "PI": True,
    "Heat_flux": False,
    "N": False,
    "Qrem": False,
}

FEATURE_SETS = {
    "default": ["rho_e", "Bx", "By", "Bz", "Vx_e", "Vy_e", "Vz_e", "Ex", "Ey", "Ez"],
    "noE": ["rho_e", "Bx", "By", "Bz", "Vx_e", "Vy_e", "Vz_e"],
    "noJ": ["rho_e", "Bx", "By", "Bz", "Ex", "Ey", "Ez"],
    "noJnoE": ["rho_e", "Bx", "By", "Bz"],
}

TASK_SPECS = {
    "P": {
        "targets": ["Pxx_e", "Pyy_e", "Pzz_e", "Pxy_e", "Pxz_e", "Pyz_e"],
        "prescaler_targets": ["log", "log", "log", "arcsinh", "arcsinh", "arcsinh"],
    },
    "divP": {
        "targets": ["EPx", "EPy", "EPz"],
        "prescaler_targets": [None, None, None],
        "extra_fields": {"divP": True},
    },
}

ARCHITECTURES = {
    "4lrs": {
        "hidden": [128, 64, 32],
        "kernels": [3, 5, 5, 3],
    },
    "5lrs": {
        "hidden": [256, 128, 64, 32],
        "kernels": [3, 3, 5, 5, 3],
    },
    "6lrs": {
        "hidden": [256, 128, 64, 32, 16],
        "kernels": [3, 3, 5, 5, 3, 3],
    },
    "7lrs": {
        "hidden": [512, 256, 128, 64, 32, 16],
        "kernels": [3, 3, 3, 5, 5, 3, 3],
    },
}

SHARED_DATA = {
    "choose_x": [0, 512],
    "choose_y": [175, 325],
    "patch_dim": [32, 32],
}

SEARCH_PROFILES = {
    "fast": {
        "n_trials": 20,
        "max_epochs": 40,
        "early_stopping_patience": 8,
        "startup_trials": 5,
        "pruner_warmup_epochs": 3,
        "space": {
            "architectures": ["4lrs", "5lrs"],
            "dropouts": [0.0, 0.05, 0.1, 0.15],
            "batch_sizes": [32, 64],
            "scheduler_families": ["plateau", "warmup_cosine"],
            "lr_range": (1e-4, 3e-3),
            "weight_decay_range": (1e-7, 1e-3),
        },
    },
    "balanced": {
        "n_trials": 50,
        "max_epochs": 80,
        "early_stopping_patience": 15,
        "startup_trials": 10,
        "pruner_warmup_epochs": 5,
        "space": {
            "architectures": ["4lrs", "5lrs", "6lrs", "7lrs"],
            "dropouts": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
            "batch_sizes": [16, 32, 64],
            "scheduler_families": ["plateau", "warmup_cosine"],
            "lr_range": (5e-5, 5e-3),
            "weight_decay_range": (1e-7, 1e-3),
        },
    },
    "thorough": {
        "n_trials": 120,
        "max_epochs": 140,
        "early_stopping_patience": 25,
        "startup_trials": 20,
        "pruner_warmup_epochs": 8,
        "space": {
            "architectures": ["4lrs", "5lrs", "6lrs", "7lrs"],
            "dropouts": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
            "batch_sizes": [16, 32, 64],
            "scheduler_families": ["plateau", "warmup_cosine"],
            "lr_range": (3e-5, 8e-3),
            "weight_decay_range": (1e-8, 3e-3),
        },
    },
}


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    profile = SEARCH_PROFILES[args.search_profile]
    if args.n_trials is None:
        args.n_trials = profile["n_trials"]
    if args.max_epochs is None:
        args.max_epochs = profile["max_epochs"]
    if args.early_stopping_patience is None:
        args.early_stopping_patience = profile["early_stopping_patience"]
    if args.startup_trials is None:
        args.startup_trials = profile["startup_trials"]
    if args.pruner_warmup_epochs is None:
        args.pruner_warmup_epochs = profile["pruner_warmup_epochs"]
    return args


def build_trial_config(trial: "optuna.Trial", args: argparse.Namespace) -> dict:
    space = SEARCH_PROFILES[args.search_profile]["space"]
    features = FEATURE_SETS[args.variant]
    task_spec = deepcopy(TASK_SPECS[args.task])
    targets = task_spec["targets"]

    architecture_name = trial.suggest_categorical("architecture", space["architectures"])
    architecture = ARCHITECTURES[architecture_name]
    activation = trial.suggest_categorical("activation", ["ReLU", "GELU", "ELU"])
    batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    dropout = trial.suggest_categorical("dropout", space["dropouts"])

    lr = trial.suggest_float("lr", space["lr_range"][0], space["lr_range"][1], log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", space["weight_decay_range"][0], space["weight_decay_range"][1], log=True
    )
    batch_size = trial.suggest_categorical("batch_size", space["batch_sizes"])
    gradient_clip_val = trial.suggest_categorical("gradient_clip_val", [0.0, 0.5, 1.0])
    scheduler_family = trial.suggest_categorical("scheduler_family", space["scheduler_families"])

    if scheduler_family == "plateau":
        scheduler_name = "ReduceLROnPlateau"
        scheduler_kwargs = {
            "patience": trial.suggest_int("scheduler_patience", 8, 25),
            "factor": trial.suggest_categorical("scheduler_factor", [0.2, 0.5]),
            "min_lr": trial.suggest_float("min_lr", 1e-6, 1e-4, log=True),
        }
    else:
        warmup_upper = max(4, min(16, args.max_epochs // 3))
        warmup_epochs = trial.suggest_int("warmup_epochs", 2, warmup_upper)
        cosine_eta_min = trial.suggest_float("cosine_eta_min", 1e-6, 5e-4, log=True)
        scheduler_name = "SequentialLR"
        scheduler_kwargs = {
            "milestones": [warmup_epochs],
            "schedulers": [
                {
                    "name": "LinearLR",
                    "kwargs": {
                        "start_factor": trial.suggest_categorical("warmup_start_factor", [0.01, 0.05, 0.1]),
                        "end_factor": 1.0,
                        "total_iters": warmup_epochs,
                    },
                },
                {
                    "name": "CosineAnnealingLR",
                    "kwargs": {
                        "T_max": max(args.max_epochs - warmup_epochs, 1),
                        "eta_min": cosine_eta_min,
                    },
                },
            ],
        }

    channels = [len(features), *architecture["hidden"], len(targets)]
    activations = [activation] * (len(channels) - 2) + [None]
    batch_norms = [batch_norm] * (len(channels) - 2) + [False]
    dropouts = [dropout] * (len(channels) - 2) + [0.0]

    fields_to_read = deepcopy(COMMON_FIELDS)
    for key, value in task_spec.get("extra_fields", {}).items():
        fields_to_read[key] = value

    trial_dir = args.output_root / args.study_name / f"trial_{trial.number:04d}"

    return {
        "trial_dir": str(trial_dir),
        "model": {
            "network": {
                "channels": channels,
                "kernels": architecture["kernels"],
                "activations": activations,
                "batch_norms": batch_norms,
                "dropouts": dropouts,
            },
            "optimizer": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "scheduler": scheduler_name,
            "scheduler_kwargs": scheduler_kwargs,
        },
        "trainer": {
            "gradient_clip_val": gradient_clip_val,
        },
        "data": {
            "data_folder": args.data_folder,
            "norm_folder": str(trial_dir / "norm"),
            "train_samples_file": args.train_samples,
            "val_samples_file": args.val_samples,
            "batch_size": batch_size,
            "num_workers": args.num_workers,
            "flatten": False,
            "scaler_features": True,
            "scaler_targets": True,
            "patch_dim": SHARED_DATA["patch_dim"],
            "prescaler_targets": task_spec["prescaler_targets"],
            "read_features_targets_kwargs": {
                "fields_to_read": fields_to_read,
                "request_features": features,
                "request_targets": targets,
                "choose_species": ["e", None, "e", None],
                "choose_x": SHARED_DATA["choose_x"],
                "choose_y": SHARED_DATA["choose_y"],
                "verbose": False,
            },
        },
    }


def objective(trial: "optuna.Trial") -> float:
    import lightning as L
    import yaml
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from optuna.integration import PyTorchLightningPruningCallback

    from closure.datamodule import ClosureDataModule
    from closure.models import FCNN
    from closure.module import ClosureLitModule

    cfg = build_trial_config(trial, ARGS)
    trial_dir = Path(cfg["trial_dir"])
    trial_dir.mkdir(parents=True, exist_ok=True)

    with (trial_dir / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    network_cfg = cfg["model"]["network"]
    network = FCNN(
        channels=network_cfg["channels"],
        kernels=network_cfg["kernels"],
        activations=network_cfg["activations"],
        batch_norms=network_cfg["batch_norms"],
        dropouts=network_cfg["dropouts"],
    )
    module = ClosureLitModule(
        network=network,
        optimizer=cfg["model"]["optimizer"],
        lr=cfg["model"]["lr"],
        weight_decay=cfg["model"]["weight_decay"],
        scheduler=cfg["model"]["scheduler"],
        scheduler_kwargs=cfg["model"]["scheduler_kwargs"],
    )
    datamodule = ClosureDataModule(**cfg["data"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=ARGS.early_stopping_patience, mode="min"),
        ModelCheckpoint(
            dirpath=trial_dir / "checkpoints",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch}-{val_loss:.5f}",
        ),
        PyTorchLightningPruningCallback(trial, monitor="val_loss"),
    ]

    trainer = L.Trainer(
        default_root_dir=str(trial_dir),
        max_epochs=ARGS.max_epochs,
        accelerator=ARGS.accelerator,
        devices=ARGS.devices,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(trial_dir), name="logs"),
        enable_progress_bar=not ARGS.disable_progress_bar,
        deterministic=ARGS.deterministic,
        log_every_n_steps=ARGS.log_every_n_steps,
        gradient_clip_val=cfg["trainer"]["gradient_clip_val"],
    )

    trainer.fit(module, datamodule=datamodule)

    best_score = trainer.callback_metrics["val_loss"].item()
    trial.set_user_attr("trial_dir", str(trial_dir))
    return best_score


def write_best_summary(study: "optuna.Study", output_dir: Path) -> None:
    import yaml

    best = {
        "number": study.best_trial.number,
        "value": study.best_trial.value,
        "params": study.best_trial.params,
        "trial_dir": study.best_trial.user_attrs.get("trial_dir"),
    }
    with (output_dir / "best_trial.yaml").open("w") as handle:
        yaml.safe_dump(best, handle, sort_keys=False)


def export_best_config(study: "optuna.Study", args: argparse.Namespace, output_dir: Path) -> Path:
    import yaml

    trial_dir = Path(study.best_trial.user_attrs["trial_dir"])
    resolved_path = trial_dir / "resolved_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())

    trainer_root = output_dir / "best_run"
    export = {
        "seed_everything": args.seed,
        "model": {
            "network": {
                "class_path": "closure.models.FCNN",
                "init_args": resolved["model"]["network"],
            },
            "criterion": "MSELoss",
            "optimizer": resolved["model"]["optimizer"],
            "lr": resolved["model"]["lr"],
            "weight_decay": resolved["model"]["weight_decay"],
            "scheduler": resolved["model"]["scheduler"],
            "scheduler_kwargs": resolved["model"].get("scheduler_kwargs", {}),
        },
        "data": resolved["data"],
        "trainer": {
            "max_epochs": args.max_epochs,
            "accelerator": args.accelerator,
            "devices": args.devices,
            "default_root_dir": str(trainer_root),
            "gradient_clip_val": resolved.get("trainer", {}).get("gradient_clip_val", 0.0),
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                    "init_args": {
                        "monitor": "val_loss",
                        "patience": args.early_stopping_patience,
                        "mode": "min",
                    },
                },
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "monitor": "val_loss",
                        "save_top_k": 1,
                        "save_last": True,
                        "mode": "min",
                        "filename": "best-{epoch}-{val_loss:.5f}",
                    },
                },
            ],
            "logger": {
                "class_path": "lightning.pytorch.loggers.CSVLogger",
                "init_args": {
                    "save_dir": str(trainer_root),
                    "name": "logs",
                },
            },
        },
    }

    export_path = Path(args.export_best_config)
    if not export_path.is_absolute():
        export_path = output_dir / export_path
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w") as handle:
        yaml.safe_dump(export, handle, sort_keys=False)
    return export_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna sweep for Harris Le2GEM15ppc FCNN experiments")
    parser.add_argument("--variant", choices=sorted(FEATURE_SETS), required=True)
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--search-profile", choices=sorted(SEARCH_PROFILES), default="balanced")
    parser.add_argument(
        "--data-folder",
        default="ecsim/Harris/Le",
        help="Data folder relative to data_dir in paths.yaml.",
    )
    parser.add_argument(
        "--train-samples",
        default="ecsim/sampling/ecsim/Harris/Le/Le2GEM15ppc/train.csv",
        help="Train CSV relative to data_dir in paths.yaml.",
    )
    parser.add_argument(
        "--val-samples",
        default="ecsim/sampling/ecsim/Harris/Le/Le2GEM15ppc/val.csv",
        help="Val CSV relative to data_dir in paths.yaml.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("optuna_runs/harris"))
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None, help="Optuna storage URL, e.g. sqlite:///harris.db")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--startup-trials", type=int, default=None)
    parser.add_argument("--pruner-warmup-epochs", type=int, default=None)
    parser.add_argument("--export-best-config", default="best_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    ARGS = parser.parse_args()
    ARGS = apply_profile_defaults(ARGS)

    if ARGS.study_name is None:
        ARGS.study_name = f"harris_{ARGS.variant}_{ARGS.task}"

    try:
        import lightning as L
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "This script requires the Lightning and Optuna extras. "
            "Install them with `pip install -e \".[hp]\"`."
        ) from exc

    # Resolve output_root so all derived paths (trial_dir, norm_folder,
    # checkpoints) become absolute and pass through _resolve_path unchanged.
    ARGS.output_root = ARGS.output_root.resolve()

    output_dir = ARGS.output_root / ARGS.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(ARGS.seed, workers=True)

    study = optuna.create_study(
        study_name=ARGS.study_name,
        direction="minimize",
        storage=ARGS.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=ARGS.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=ARGS.startup_trials,
            n_warmup_steps=ARGS.pruner_warmup_epochs,
        ),
    )
    study.optimize(objective, n_trials=ARGS.n_trials)
    write_best_summary(study, output_dir)
    exported_config = export_best_config(study, ARGS, output_dir)

    print("\n=== Best trial ===")
    print(f"  Number: {study.best_trial.number}")
    print(f"  Value (val_loss): {study.best_trial.value:.6f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"  Exported config: {exported_config}")