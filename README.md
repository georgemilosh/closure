# closure

**closure** is a flexible, reproducible, and distributed-ready machine learning framework for training on fields files of ECsim/iPiC3D Particle in Cell Codes, with a focus on fluid closure. It provides tools for data loading, preprocessing, model training, evaluation, and experiment management, supporting both single-node and distributed (multi-GPU/multi-node) workflows.

---

## What does closure do?

- **Experiment Management:**  
  Organizes experiments in a clear directory structure. Each experiment (called a "run") is stored in its own subfolder, with all configs, logs, and results saved for reproducibility.

- **Flexible Data Handling:**  
  Loads and preprocesses large scientific datasets (e.g., from simulations or experiments), supporting normalization, filtering, patch extraction, and custom feature/target selection.

- **Model Training:**  
  Supports a variety of neural network architectures (MLP, FCNN, ResNet, etc.) and training protocols, including distributed training with PyTorch.

- **Evaluation & Analysis:**  
  Provides utilities for loss/metric evaluation, plotting, and comparison across runs or experiments.

- **Reproducibility:**  
  Every run is fully specified by a `config.json` file, making it easy to reproduce or compare results.

---

## Directory Structure

- `closure/src/` — Core source code (trainers, datasets, models, utilities, etc.)
- `closure/examples/` — Example notebooks and scripts (see `tuto_train_haydn.ipynb`)
- `work_dir/` — User-specified experiment directory; each run is a subfolder (e.g., `work_dir/0`, `work_dir/nosubsample`, etc.)

---

## Quick Start

### 1. Install dependencies

Option 1: Install via conda:
```bash
conda env create -f gputorch.yml
```

Option 2: Install dependencies manually:

- Python 3.x
- PyTorch
- numpy, pandas, joblib, scipy, matplotlib

### 2. Prepare your data

- Organize your simulation or experimental data as described in the tutorial notebook.
- Prepare CSV files listing your train/val/test samples.

### 3. Define configuration dictionaries

See [examples/tuto_train_haydn.ipynb](closure/examples/tuto_train_haydn.ipynb) for a full walkthrough.

```python
from closure.src.trainers import Trainer

dataset_kwargs = {...}      # See notebook for details
load_data_kwargs = {...}
model_kwargs = {...}

trainer = Trainer(
    work_dir="experiments/my_experiment",
    dataset_kwargs=dataset_kwargs,
    load_data_kwargs=load_data_kwargs,
    model_kwargs=model_kwargs,
    device="cuda"
)
trainer.fit()
```

### 4. Command-line usage

```bash
python -m closure.src.trainers --config work_dir=work_dir --config run=run_name --config model_kwargs.model_name=ResNet
```

- Use `--config key=value` to override config values (supports nested keys with dot notation).

---

## How does experiment management work?

- The **Trainer** manages a parent directory (`work_dir`).
- Each **run** is a subfolder (e.g., `work_dir/0`, `work_dir/nosubsample`), containing:
  - `config.json` — full configuration for the run
  - `model.pth` — trained model weights
  - `loss_dict.pkl` — training/validation loss history
  - `run.log` — log file for the run
  - normalization files (`X.pkl`, `y.pkl`)
- To start a new run, change any config (e.g., hyperparameters, model, batch size) and call `trainer.fit(config=new_config)`. A new subfolder is created.
- To reload or compare runs, use the `load_run` method or the provided utilities.

---

## Example Workflow

See [examples/tuto_train_haydn.ipynb](closure/examples/tuto_train_haydn.ipynb) for a step-by-step tutorial, including:

- Setting up dataset and model configs
- Training a model
- Running multiple experiments in the same folder
- Evaluating and comparing results
- Reloading previous runs

---

## Main Components

- **src/trainers.py** — The `Trainer` class: manages configs, logging, datasets, models, and training.
- **src/datasets.py** — Data loading utilities: distributed/serial samplers, normalization, filtering, patch extraction.
- **src/models.py** — Model definitions: MLP, FCNN, ResNet, and the `PyNet` training wrapper.
- **src/utilities.py** — Utility functions: config handling, evaluation, plotting, and more.
- **src/read_pic.py** — Functions for reading iPiC3D/ECsim simulation data in h5 or pickle format.

---

## Citing & License

- **Author:** George Miloshevich  
- **License:** MIT License  
- **Projects:** STRIDE, HELIOSKILL

---

## Further Reading

- [examples/tuto_train_haydn.ipynb](closure/examples/tuto_train_haydn.ipynb) — Full tutorial notebook
- Source code docstrings for detailed API documentation

---

**closure** is designed for flexibility, reproducibility, and ease of use in scientific ML workflows. Contributions and feedback are welcome!