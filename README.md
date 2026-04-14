# closure

closure is a machine learning framework for fluid closure modeling on ECsim and iPiC3D data.

The training stack is now based on PyTorch Lightning.

## Highlights

- Lightning-native training with clear separation between model and data logic.
- YAML-driven experiments through LightningCLI.
- Built-in callbacks for timing and memory monitoring.
- Evaluation and plotting helpers compatible with the new module/datamodule API.

## Core Components

- `closure/module.py`: `ClosureLitModule` (`lightning.LightningModule`)
- `closure/datamodule.py`: `ClosureDataModule` (`lightning.LightningDataModule`)
- `closure/models.py`: network architectures (`MLP`, `FCNN`, `ResNet`, `CNet`)
- `closure/cli.py`: CLI entry point (`closure-train`)
- `closure/callbacks.py`: `MemoryMonitorCallback`, `TimingCallback`
- `closure/evaluation.py`: post-training metrics and prediction transforms
- `closure/visualization.py`: prediction vs ground-truth plotting

## Installation

### Basic Installation

```bash
pip install -e .
```

This installs the core framework with PyTorch, PyTorch Lightning, and essential utilities.

### Optional Dependencies

We provide several optional extras for different use cases:

**Hyperparameter Optimization (Optuna)**

For hyperparameter search with Optuna, install the `hp` extra:

```bash
pip install -e ".[hp]"
```

Includes: `optuna`, `optuna-integration`, `scikit-learn`, `plotly`, `nbformat`

**Jupyter Notebooks**

For interactive notebook development:

```bash
pip install -e ".[notebook]"
```

Includes: `jupyter`, `ipykernel`, `notebook`, `ipywidgets`

**Combined Installation (HP + Notebooks)**

```bash
pip install -e ".[hp,notebook]"
```

**Development**

For development, testing, and linting:

```bash
pip install -e ".[dev]"
```

Includes: `pytest`, `pytest-cov`, `ruff`, `pre-commit`

### GPU/CUDA Support

The package includes PyTorch, torchvision, and torchaudio but defaults to CPU builds. To enable GPU support, **force-reinstall** the PyTorch packages from the appropriate CUDA index (required because pip will otherwise skip the reinstall if versions match):

**CUDA 12.4 (Recommended for driver ≥ 525.60):**

```bash
pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 12.1:**

```bash
pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu121
```

**CPU-only (no GPU):**

```bash
pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cpu
```

> **Note:** Check your NVIDIA driver version with `nvidia-smi`. The driver's CUDA version must be ≥ the toolkit version. For example, driver CUDA 12.8 supports cu124 but **not** cu130.

Verify GPU support after installation:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

### Recommended Installation for Hyperparameter Sweep Workflows

If you want to use the Optuna hyperparameter sweep functionality with GPU acceleration:

```bash
# Install core + hyperparameter optimization + notebooks
pip install -e ".[hp,notebook]"

# Then force-reinstall GPU-enabled PyTorch for your platform (e.g., CUDA 12.4)
pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu124
```

### Quick Start with Requirements Files

We provide pre-made requirements files for common workflows:

**Core only (CPU):**
```bash
pip install -r requirements.txt
```

**Hyperparameter optimization (Optuna + analysis):**
```bash
pip install -r requirements-hp.txt
```

**Development and testing:**
```bash
pip install -r requirements-dev.txt
```

**GPU support with CUDA 12.4:**
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu124
```

**Full stack (HP + Notebooks + Dev — matches `closure-test` env):**
```bash
pip install -r requirements-dev.txt
```

For GPU support, force-reinstall PyTorch from the appropriate CUDA index:
```bash
pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu124
```

See `requirements-gpu.txt` for detailed instructions on GPU installation for different CUDA versions.

## Verifying Installation

Test that everything is installed correctly:

```bash
# Test core imports
python -c "import closure; import lightning; import torch; print('✅ Core packages OK')"

# Test optional imports (if installed with [hp])
python -c "import optuna; import plotly; import sklearn; print('✅ HP packages OK')"

# Test notebook imports (if installed with [notebook])
python -c "import jupyter; import ipykernel; print('✅ Notebook packages OK')"

# Test GPU (if CUDA enabled)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Test CLI
closure-train --help

# Test Optuna sweep (hyperparameter optimization)
python examples/optuna/harris_optuna_sweep.py --help
```

## Quick Start (Python API)

```python
import lightning as L

from closure.datamodule import ClosureDataModule
from closure.models import MLP
from closure.module import ClosureLitModule

network = MLP(feature_dims=[10, 64, 32, 6], activations=["Tanh", "ReLU", None])

module = ClosureLitModule(
    network=network,
    criterion="MSELoss",
    optimizer="Adam",
    lr=5e-4,
    scheduler="ReduceLROnPlateau",
)

datamodule = ClosureDataModule(
    data_folder="/path/to/data",
    norm_folder="/path/to/norm",
    train_samples_file="/path/to/train.csv",
    val_samples_file="/path/to/val.csv",
    test_samples_file="/path/to/test.csv",
    batch_size=512,
    flatten=True,
    read_features_targets_kwargs={
        "request_features": ["rho_e", "Bx", "By", "Bz", "Vx_e", "Vy_e", "Vz_e", "Ex", "Ey", "Ez"],
        "request_targets": ["Pxx_e", "Pyy_e", "Pzz_e", "Pxy_e", "Pxz_e", "Pyz_e"],
    },
)

trainer = L.Trainer(max_epochs=50, accelerator="auto")
trainer.fit(module, datamodule=datamodule)
trainer.test(module, datamodule=datamodule)
```

## Quick Start (CLI)

Use provided YAML configs under `configs/`.

```bash
closure-train fit --config configs/default.yaml
```

Override parameters directly from CLI:

```bash
closure-train fit \
  --config configs/default.yaml \
  --model.network.class_path=closure.models.ResNet \
  --model.lr=1e-3 \
  --data.batch_size=256
```

## Logging and Artifacts

Lightning logging is used by default (CSV logger in configs).

Typical outputs include:

- `lightning_logs/` or configured logger directory
- `metrics.csv`
- checkpoints from `ModelCheckpoint`
- normalized feature/target statistics in `norm_folder`

Legacy files like `loss_dict.pkl` are no longer used.

## Production Setup

This section covers everything needed to go from raw simulation data to
production training runs.

### 1. `paths.yaml`

Create a `paths.yaml` in the repository root (copy from `paths.yaml.example`):

```yaml
work_dir: ./models       # training outputs, checkpoints, normalization stats
data_dir: /scratch/data   # root of your simulation data
```

Relative paths in `paths.yaml` are resolved against the directory that
contains the file.  All config parameters that accept paths use a three-tier
resolution strategy (implemented by `ClosureDataModule._resolve_path`):

| Path form | Example | Resolution |
|-----------|---------|------------|
| Absolute | `/scratch/data/Harris` | Used as-is |
| Dot-relative (`./`, `../`) | `./data/train.csv` | Resolved against the current working directory |
| Bare identifier | `ecsim/Harris/Le` | Joined with the corresponding `paths.yaml` root (`data_dir` or `work_dir`) |

### 2. Data directory structure

Simulation data is stored as HDF5 or pickle files under `data_dir`, organized
by experiment.  Each file contains a single simulation time step:

```
data_dir/
  ecsim/Harris/Le/
    T2D14_filter2/
      T2D-Fields_00500.h5.pkl
      T2D-Fields_01000.h5.pkl
      ...
    T2D15_filter2/
      T2D-Fields_00500.h5.pkl
      ...
```

The files are read by `closure.read_pic.read_features_targets`, which
extracts the requested field channels (B, E, rho, J, P, etc.) and species.

### 3. Creating train/val/test splits

Use `scripts/datasplit.py` to build CSV split files.  Each CSV has a single
`filenames` column listing the data file paths:

```bash
# Training set from two simulation folders (time steps 5000–10000)
python scripts/datasplit.py \
    folders=[T2D14_filter2,T2D15_filter2] \
    name=train.csv \
    root_folder=/scratch/data/ecsim/Harris/Le/ \
    min_number=5000 max_number=10000

# Validation set from a held-out folder
python scripts/datasplit.py \
    folders=[T2D16_filter2] \
    name=val.csv \
    root_folder=/scratch/data/ecsim/Harris/Le/

# Test set
python scripts/datasplit.py \
    folders=[T2D17_filter2] \
    name=test.csv \
    root_folder=/scratch/data/ecsim/Harris/Le/
```

Arguments:

| Argument | Required | Description |
|----------|----------|-------------|
| `folders` | yes | Folder names or paths to search, e.g. `[a,b,c]` |
| `name` | yes | Output CSV filename |
| `root_folder` | no | Root prepended to each folder path |
| `pattern` | no | Glob pattern (default: `T2D-Fields_*`) |
| `min_number` | no | Exclude files with time-step number below this |
| `max_number` | no | Exclude files with time-step number above this |

### 4. Writing a YAML config

Three annotated templates are provided under `configs/`:

| Template | Architecture | Data shape | Use case |
|----------|-------------|------------|----------|
| `configs/default.yaml` | FCNN | 2-D patches | CNN-based closure |
| `configs/mlp.yaml` | MLP | Flattened pixels | Pixel-wise baseline |
| `configs/resnet.yaml` | ResNet | 2-D patches | Deep residual closure |

Copy one and customize.  Key sections explained:

```yaml
data:
  data_folder: ecsim/Harris/Le           # bare → joined with data_dir
  norm_folder: Harris/Le/my_experiment   # bare → joined with work_dir
  train_samples_file: ./splits/train.csv  # ./ → CWD-relative
  val_samples_file: ./splits/val.csv
  test_samples_file: ./splits/test.csv
  flatten: false                          # true for MLP, false for CNN/ResNet
  patch_dim: [32, 32]                     # random crop size (CNN/ResNet only)
  scaler_features: true                   # enable per-channel standardization
  scaler_targets: true
  prescaler_features:                     # per-channel transforms before standardization
    - arcsinh    # rho_e
    - null       # Bx  (no prescaling)
    - ...
  prescaler_targets:
    - log        # Pxx_e (positive-definite diagonal)
    - arcsinh    # Pxy_e (signed off-diagonal)
    - ...
  read_features_targets_kwargs:
    fields_to_read:                       # which HDF5 field groups to load
      B: true
      E: true
      rho: true
      J: true
      P: true
      PI: true
    request_features:                     # specific channels extracted from fields
      - rho_e
      - Bx
      - By
      - Bz
      - Jx_e
      - Jy_e
      - Jz_e
      - Vx_e
      - Vy_e
      - Vz_e
    request_targets:
      - Pxx_e
      - Pyy_e
      - Pzz_e
      - Pxy_e
      - Pxz_e
      - Pyz_e
    choose_species: ['e', null]           # electron species for multi-species data
    choose_x: [0, 512]                    # spatial domain crop
    choose_y: [175, 325]
```

Prescaler guidance:
- `log` — for strictly positive quantities (diagonal pressure)
- `arcsinh` — for quantities that can be negative or span orders of magnitude
- `null` — no prescaling

### 5. Launching training

**Single GPU:**

```bash
closure-train fit --config my_config.yaml
```

**Multi-GPU (DDP):**

```bash
closure-train fit --config my_config.yaml \
    --trainer.devices=4 \
    --trainer.strategy=ddp
```

**Slurm cluster:**

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12

srun closure-train fit --config my_config.yaml
```

### 6. Scaffolding experiment sweeps

For systematic architecture/feature-set sweeps, use
`scripts/scaffold_harris_experiments.py`.  It generates a directory tree
of YAML configs and Slurm `run.sh` scripts:

```bash
python scripts/scaffold_harris_experiments.py \
    --output-root models/Harris/Le/Le2GEM15ppc_lightning \
    --data-folder ecsim/Harris/Le \
    --split-root ecsim/sampling/ecsim/Harris/Le/Le2GEM15ppc \
    --max-epochs 500 --devices 4
```

This creates:

```
Le2GEM15ppc_lightning/
  default/P/          4lrs_es500.yaml  5lrs_es500.yaml  ...  run.sh
  default/divP/       4lrs.yaml        5lrs.yaml        ...  run.sh
  noE/P/              ...
  noJ/P/              ...
  noJnoE/P/           ...
```

Each variant (default, noE, noJ, noJnoE) uses a different feature subset.
Each task (P, divP) uses different targets and prescalers.  The `run.sh`
files are ready to submit with `sbatch`.

### 7. Evaluation and artifact export

After training, load a checkpoint and evaluate:

```python
from closure.module import ClosureLitModule
from closure.evaluation import evaluate_loss, evaluate_regression_metrics, transform_targets

module = ClosureLitModule.load_from_checkpoint("best.ckpt", network=network)
ground_truth, prediction = transform_targets(module, test_dataset, ...)

# Per-channel MSE
evaluate_loss(test_dataset, ground_truth, prediction, "MSELoss", verbose=True)

# Regression metrics table (R², RMSE, Pearson r, etc.)
metrics_df = evaluate_regression_metrics(test_dataset, ground_truth, prediction)
```

Export deployable artifacts:

```python
import torch

# Inference bundle (state dict + normalization stats + metadata)
torch.save({"state_dict": ..., "features_mean": ..., ...}, "inference_bundle.pt")

# TorchScript for deployment
scripted = torch.jit.script(network)
scripted.save("torchscript.pt")
```

See `examples/tutorials/tuto_train.py` for a complete end-to-end example
including evaluation, visualization, and artifact export.

## Examples

- `examples/tutorials/tuto_train.py`: self-contained training tutorial using bundled fixture data
- `examples/tuto_train.ipynb`: real-data tutorial (Lightning update section added at top)
- `examples/tuto_train_synthetic.ipynb`: synthetic-data tutorial (Lightning update section added at top)
- `examples/optuna/optuna_sweep.py`: Optuna sweep example with Lightning
- `examples/optuna/harris_optuna_sweep.py`: Harris Le2GEM15ppc Optuna sweep for FCNN experiments

## Notes on Migration

- The old `Trainer`, `PyNet`, and `closure.trainers` module were removed.
- Use `ClosureLitModule` + `ClosureDataModule` for programmatic workflows.
- Use `closure-train` for config-driven workflows.

## Citing & License

- **Author:** George Miloshevich  
- **License:** MIT License  
- **Projects:** STRIDE, HELIOSKILL

If you use **closure** in your research, please cite:

```bibtex
@article{miloshevich2026electron,
  title = {Electron Neural Closure for Turbulent Magnetosheath Simulations: {{Energy}} Channels},
  author = {Miloshevich, G. and Vranckx, L. and de Oliveira Lopes, F. N. and Dazzi, P. and Arrò, G. and Lapenta, G.},
  year = {2026},
  journal = {Physics of Plasmas},
  volume = {33},
  number = {1},
  pages = {012901},
  issn = {1070-664X},
  doi = {10.1063/5.0300009},
}
```

---

## Further Reading

- [examples/tuto_train.ipynb](closure/examples/tuto_train.ipynb) — Full tutorial notebook
- Source code docstrings for detailed API documentation

---

**closure** is designed for flexibility, reproducibility, and ease of use in scientific ML workflows. Contributions and feedback are welcome!
