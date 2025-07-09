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

See [examples/tuto_train_haydn.ipynb](examples/tuto_train_haydn.ipynb) for a full walkthrough.

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

A tutorial on the usage of the trainer can be found in `Tutorial_trainer.pdf` in the `examples` folder.

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

### What to Pay Attention To

- **Configuration Consistency:**  
  When running multiple experiments in the same `work_dir`, make sure the dataset configuration (`dataset_kwargs`) is consistent across runs. If you change the dataset, use a new `work_dir` to avoid normalization mismatches.

- **Run Naming:**  
  Use the `run` key in your config to create new subfolders for each experiment. This keeps results organized and prevents accidental overwrites.

- **Distributed Setup:**  
  The script automatically handles distributed training if environment variables (like `WORLD_SIZE`) are set. Make sure your cluster/job environment is configured correctly.

- **Logging:**  
  Each run has its own log file, making it easy to debug and track progress. Logs include information about the host, rank, and device.

- **Reproducibility:**  
  All configs, normalization parameters, and model weights are saved for each run, ensuring you can always reproduce or analyze past experiments.

- **Extensibility:**  
  The Trainer is designed to be flexible—custom models, datasets, and training protocols can be plugged in with minimal changes.

---

## Main Components

- **src/trainers.py** — The `Trainer` class: manages configs, logging, datasets, models, and training.
- **src/datasets.py** — Data loading utilities: distributed/serial samplers, normalization, filtering, patch extraction.
- **src/models.py** — Model definitions: MLP, FCNN, ResNet, and the `PyNet` training wrapper.
- **src/utilities.py** — Utility functions: config handling, evaluation, plotting, and more.
- **src/read_pic.py** — Functions for reading iPiC3D/ECsim simulation data in h5 or pickle format.

## Datasets and Data Loading

closure provides a powerful and flexible system for loading, preprocessing, and sampling scientific datasets, designed for both single-node and distributed (multi-GPU) workflows.

### Key Features

- **Distributed and Serial Sampling:**  
  Efficiently split and shuffle data across multiple GPUs or nodes using the `DistributedSampler` and `SubSampler` classes. This ensures each process gets a unique subset of the data, enabling scalable training.

- **Channel-based Data Loading:**  
  The `ChannelDataLoader` extends PyTorch's DataLoader to allow selection of specific feature and target channels (e.g., physical quantities like density, pressure, etc.), as well as patch-based cropping for image-like data.

- **Flexible Subsampling and Shuffling:**  
  Easily control the fraction of data used for training or validation via the `subsample_rate` and `subsample_seed` parameters. This is useful for quick prototyping or handling very large datasets. This is also useful for oversampling which is necessary if you are using random patch extraction.

- **Patch Extraction:**  
  For image or field data, you can extract random spatial patches on-the-fly, which is useful for data selection or training convolutional models.

- **Feature/Target Normalization:**  
  The `DataFrameDataset` class supports normalization (mean/std) and pre-scaling (e.g., log transforms) of both features and targets. Normalization parameters are saved and reused for reproducibility.

- **Filtering and Transformations:**  
  Apply custom filters (e.g., smoothing, denoising) to features or targets using scipy or custom functions. Supports torchvision-style transforms for data augmentation.

- **CSV/Metadata Integration:**  
  Datasets are typically defined by CSV files listing sample filenames and metadata, making it easy to manage large and complex datasets.

### Example Usage

```python
from closure.src.datasets import DataFrameDataset, ChannelDataLoader

dataset = DataFrameDataset(
    data_folder='/path/to/data',
    norm_folder='/path/to/norm',
    samples_file='/path/to/samples.csv',
    feature_dtype='float32',
    target_dtype='float32',
    scaler_features=None,
    scaler_targets=None,
    transform={'RandomCrop': {'size': (16, 16)}, 'apply': ['train']},
    datalabel='train'
)

loader = ChannelDataLoader(
    dataset,
    batch_size=32,
    feature_channel_names=['Bx', 'By', 'Bz'],
    target_channel_names=['Pxx_e', 'Pyy_e'],
    subsample_rate=0.5,
    subsample_seed=42,
    patch_dim=[16, 16],
)

for features, targets in loader:
    # Training loop here
    pass
```

### Typical Workflow

1. **Prepare CSV files** listing your train/val/test samples.
2. **Configure `dataset_kwargs`** to specify data location, features/targets, normalization, and any filters or transforms.
3. **Create a `DataFrameDataset`** for each split (train/val/test).
4. **Wrap with `ChannelDataLoader`** for batching, shuffling, and distributed support.
5. **Pass loaders to the Trainer** for model training and evaluation.


## Models

closure provides a modular and extensible system for defining, training, and evaluating neural network models, with a focus on scientific and physical data. The models are implemented using PyTorch and are designed to be easily configurable and compatible with distributed training.

### Key Features

- **Unified Training Wrapper (`PyNet`):**  
  The `PyNet` class wraps any supported model architecture and manages the full training lifecycle, including optimizer and scheduler setup, distributed training (via PyTorch DDP), checkpointing, early stopping, and logging. It supports custom optimizers, learning rate schedulers, and metrics.

- **Model Zoo:**  
  Several neural network architectures are provided out-of-the-box:
  - **MLP (Multi-Layer Perceptron):** For tabular or flattened data, with configurable layer sizes, activations, and dropout.
  - **FCNN (Fully Convolutional Neural Network):** For image or field data, supports configurable channels, kernels, activations, batch normalization, and dropout.
  - **ResNet:** Flexible residual network with user-defined skip connections, suitable for deep image-based models.
  - **CNet:** Example convolutional network for image data.

- **Custom Initialization:**  
  Models support custom weight and bias initialization, as well as flexible activation and dropout configuration.

- **Distributed Training:**  
  All models can be wrapped in PyTorch's DistributedDataParallel for multi-GPU or multi-node training.

- **Checkpointing and Reloading:**  
  Models and training histories can be saved and reloaded from disk, supporting experiment reproducibility and resuming.


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
