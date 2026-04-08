#!/usr/bin/env python3
"""Generate Harris Le2GEM15ppc Lightning experiment folders.

This script scaffolds a directory tree similar to the archived legacy layout:

    default/P
    default/divP
    noE/P
    noE/divP
    noJ/P
    noJ/divP
    noJnoE/P
    noJnoE/divP

Each experiment folder receives:

- one Lightning YAML config per architecture sweep entry
- a Slurm-ready ``run.sh`` that launches all configs sequentially

The generated configs target the current Lightning-based training stack via
``closure-train fit --config <file>.yaml``.

Example:

    python scripts/scaffold_harris_experiments.py \
      --output-root models/Harris/Le/Le2GEM15ppc_lightning \
      --repo-dir "$HOME/georgem/closure"
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


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


SHARED_SPEC = {
    "choose_x": [0, 512],
    "choose_y": [175, 325],
    "patch_dim": [32, 32],
    "optimizer": "AdamW",
}

TASK_SPECS = {
    "P": {
        "targets": ["Pxx_e", "Pyy_e", "Pzz_e", "Pxy_e", "Pxz_e", "Pyz_e"],
        "prescaler_targets": ["log", "log", "log", "arcsinh", "arcsinh", "arcsinh"],
        "job_hours": 10,
        "run_names": ["4lrs_es500", "5lrs_es500", "6lrs_es500", "7lrs_es500"],
    },
    "divP": {
        "targets": ["EPx", "EPy", "EPz"],
        "prescaler_targets": [None, None, None],
        "extra_fields": {"divP": True},
        "job_hours": 12,
        "run_names": ["4lrs", "5lrs", "6lrs", "7lrs"],
    },
}

FEATURE_SETS = {
    "default": ["rho_e", "Bx", "By", "Bz", "Vx_e", "Vy_e", "Vz_e", "Ex", "Ey", "Ez"],
    "noE": ["rho_e", "Bx", "By", "Bz", "Vx_e", "Vy_e", "Vz_e"],
    "noJ": ["rho_e", "Bx", "By", "Bz", "Ex", "Ey", "Ez"],
    "noJnoE": ["rho_e", "Bx", "By", "Bz"],
}


def build_variants() -> dict:
    variants = {}
    for variant_name, features in FEATURE_SETS.items():
        variants[variant_name] = {}
        for task_name, task_spec in TASK_SPECS.items():
            spec = deepcopy(SHARED_SPEC)
            spec["features"] = features
            spec.update(deepcopy(task_spec))
            variants[variant_name][task_name] = spec
    return variants


VARIANTS = build_variants()


ARCHITECTURES = {
    4: {
        "hidden": [128, 64, 32],
        "kernels": [3, 5, 5, 3],
        "activations": ["ReLU", "ReLU", "ReLU", None],
        "batch_norms": [True, True, True, False],
    },
    5: {
        "hidden": [256, 128, 64, 32],
        "kernels": [3, 3, 5, 5, 3],
        "activations": ["ReLU", "ReLU", "ReLU", "ReLU", None],
        "batch_norms": [True, True, True, True, False],
    },
    6: {
        "hidden": [256, 128, 64, 32, 16],
        "kernels": [3, 3, 5, 5, 3, 3],
        "activations": ["ReLU", "ReLU", "ReLU", "ReLU", "ReLU", None],
        "batch_norms": [True, True, True, True, True, False],
    },
    7: {
        "hidden": [512, 256, 128, 64, 32, 16],
        "kernels": [3, 3, 3, 5, 5, 3, 3],
        "activations": ["ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "ReLU", None],
        "batch_norms": [True, True, True, True, True, True, False],
    },
}


def build_config(
    experiment_dir: Path,
    run_name: str,
    spec: dict,
    data_folder: str,
    split_root: str,
    num_workers: int,
    max_epochs: int,
    devices: int,
) -> dict:
    n_inputs = len(spec["features"])
    n_outputs = len(spec["targets"])
    arch_depth = int(run_name[0])
    arch = ARCHITECTURES[arch_depth]
    run_dir = experiment_dir / run_name

    fields = deepcopy(COMMON_FIELDS)
    for key, value in spec.get("extra_fields", {}).items():
        fields[key] = value

    return {
        "seed_everything": 42,
        "model": {
            "network": {
                "class_path": "closure.models.FCNN",
                "init_args": {
                    "channels": [n_inputs, *arch["hidden"], n_outputs],
                    "kernels": arch["kernels"],
                    "activations": arch["activations"],
                    "batch_norms": arch["batch_norms"],
                },
            },
            "criterion": "MSELoss",
            "optimizer": spec["optimizer"],
            "lr": 0.001,
            "weight_decay": 1.0e-5,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_kwargs": {
                "patience": 25,
                "factor": 0.2,
                "min_lr": 1.0e-5,
            },
        },
        "data": {
            "data_folder": data_folder,
            "norm_folder": str(experiment_dir),
            "train_samples_file": f"{split_root}/train.csv",
            "val_samples_file": f"{split_root}/val.csv",
            "test_samples_file": f"{split_root}/test.csv",
            "batch_size": 32,
            "num_workers": num_workers,
            "flatten": False,
            "scaler_features": True,
            "scaler_targets": True,
            "patch_dim": spec["patch_dim"],
            "read_features_targets_kwargs": {
                "fields_to_read": fields,
                "request_features": spec["features"],
                "request_targets": spec["targets"],
                "choose_species": ["e", None, "e", None],
                "choose_x": spec["choose_x"],
                "choose_y": spec["choose_y"],
                "verbose": False,
            },
            "prescaler_targets": spec["prescaler_targets"],
        },
        "trainer": {
            "max_epochs": max_epochs,
            "accelerator": "gpu",
            "devices": devices,
            "strategy": "ddp",
            "default_root_dir": str(run_dir),
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                    "init_args": {
                        "monitor": "val_loss",
                        "patience": max_epochs,
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
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": "epoch",
                    },
                },
            ],
            "logger": {
                "class_path": "lightning.pytorch.loggers.CSVLogger",
                "init_args": {
                    "save_dir": str(run_dir),
                    "name": "logs",
                },
            },
        },
    }


def write_run_script(
    experiment_dir: Path,
    variant_name: str,
    target_name: str,
    run_names: list[str],
    repo_dir: str,
    account: str,
    job_hours: int,
) -> None:
    job_name = f"Le2GEM15ppc_{variant_name}_{target_name}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        "#SBATCH --nodes=2",
        "#SBATCH --gres=gpu:4",
        "#SBATCH --ntasks-per-node=4",
        "#SBATCH --cpus-per-task=12",
        "#SBATCH --mem=120G",
        f"#SBATCH --time={job_hours:02d}:00:00",
        "#SBATCH --output=out_%x_%j.out",
        "#SBATCH --error=err_%x_%j.err",
        "",
        "set -euo pipefail",
        "",
        f'REPO_DIR="{repo_dir}"',
        'OUTPUT_DIR="$(pwd)"',
        'export MASTER_PORT="${MASTER_PORT:-12340}"',
        'export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))',
        'export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)',
        "",
        "module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1",
        "module load h5py/3.9.0-foss-2023a",
        "module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1",
        "module load matplotlib/3.7.2-gfbf-2023a",
        "",
        'mkdir -p "$HOME/job_logs"',
        'echo "script=$(basename "$0") pwd=$(pwd)" >> "$HOME/job_logs/job_${SLURM_JOB_ID}.log"',
        "",
        'cd "$REPO_DIR"',
        "",
    ]
    for run_name in run_names:
        lines.append(
            f'srun closure-train fit --config "$OUTPUT_DIR/{run_name}.yaml"'
        )
    lines.append("")
    (experiment_dir / "run.sh").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold Harris Le2GEM15ppc Lightning experiment folders.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/Harris/Le/Le2GEM15ppc_lightning"),
        help="Root directory where the scaffolded experiment tree will be created.",
    )
    parser.add_argument(
        "--repo-dir",
        default="$HOME/georgem/closure",
        help="Repository path to embed into generated Slurm run scripts.",
    )
    parser.add_argument(
        "--data-folder",
        default="/volume1/scratch/share_dir/ecsim/Harris/Le",
        help="Harris data root passed to ClosureDataModule.",
    )
    parser.add_argument(
        "--split-root",
        default="/volume1/scratch/share_dir/ecsim/sampling/ecsim/Harris/Le/Le2GEM15ppc",
        help="Folder containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--account",
        default="2025_112",
        help="Slurm account to embed in generated run.sh files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of dataloader workers written into the YAML configs.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=500,
        help="Maximum epochs written into the YAML configs.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=4,
        help="Number of devices written into the YAML configs.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    created = []
    for variant_name, target_map in VARIANTS.items():
        for target_name, spec in target_map.items():
            experiment_dir = output_root / variant_name / target_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            created.append(experiment_dir)

            for run_name in spec["run_names"]:
                cfg = build_config(
                    experiment_dir=experiment_dir,
                    run_name=run_name,
                    spec=spec,
                    data_folder=args.data_folder,
                    split_root=args.split_root,
                    num_workers=args.num_workers,
                    max_epochs=args.max_epochs,
                    devices=args.devices,
                )
                with (experiment_dir / f"{run_name}.yaml").open("w") as handle:
                    yaml.safe_dump(cfg, handle, sort_keys=False)

            write_run_script(
                experiment_dir=experiment_dir,
                variant_name=variant_name,
                target_name=target_name,
                run_names=spec["run_names"],
                repo_dir=args.repo_dir,
                account=args.account,
                job_hours=spec["job_hours"],
            )

    print("Created experiment folders:")
    for path in created:
        print(f"- {path}")


if __name__ == "__main__":
    main()