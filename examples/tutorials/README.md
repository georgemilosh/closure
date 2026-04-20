# Tutorial Scripts

Quick-start scripts that walk through the main `closure` workflows.
Each script is self-contained and can be run from the **project root**.

## Prerequisites

1. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Scripts

### `tuto_train.py` — End-to-end training

Self-contained tutorial using the small fixture dataset shipped with the repo (`tests/fixtures/ecsim_tiny`). No `paths.yaml` or external data needed.

Covers config creation, Lightning training, evaluation (MSE / R² / per-channel diagnostics), visualization, and artifact export (inference bundle + TorchScript).

```bash
# Train + evaluate + export (uses bundled fixture data)
python examples/tutorials/tuto_train.py

# Skip training, evaluate existing checkpoints only
python examples/tutorials/tuto_train.py --no-train

# Custom epochs and batch size
python examples/tutorials/tuto_train.py --max-epochs 10 --batch-size 512

# See all options
python examples/tutorials/tuto_train.py --help
```

### `tuto_optuna.py` — Optuna hyperparameter sweep

Launches an Optuna study for the Harris current-sheet experiment, then analyses the results with trial tables and interactive Plotly visualizations.

```bash
# Quick sweep (12 trials, 4 epochs each, CPU)
python examples/tutorials/tuto_optuna.py

# Larger sweep on GPU
python examples/tutorials/tuto_optuna.py --n-trials 50 --max-epochs 20 --accelerator gpu

# Analyse an existing study without rerunning
python examples/tutorials/tuto_optuna.py --no-sweep

# See all options
python examples/tutorials/tuto_optuna.py --help
```
