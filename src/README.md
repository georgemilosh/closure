# STRIDE Trainer

A flexible, reproducible, and distributed-ready trainer class for orchestrating machine learning protocols.

## Overview

The `Trainer` class manages the full lifecycle of a machine learning experiment:
- **Configuration**: Handles experiment and run-specific configs.
- **Logging**: Sets up file and console logging, including distributed setups.
- **Datasets & Dataloaders**: Loads and manages train/val/test datasets and their loaders.
- **Model**: Instantiates or loads models, supports saving/loading weights.
- **Distributed Training**: Supports multi-GPU and multi-node training via PyTorch Distributed.

## Directory Structure

- The `Trainer` manages a parent experiment directory (`work_dir`).
- Each experiment "run" is a subdirectory within `work_dir` (e.g., `work_dir/0`, `work_dir/1`, ...).
- Each run subfolder contains its own `config.json`, model weights, logs, and results.
- This structure allows you to organize multiple experiments under a single `Trainer`, with each run representing a distinct training session (e.g., with different hyperparameters or data splits).
- The `Trainer` can load, manage, and reproduce any run by referencing its subdirectory, ensuring clear separation and reproducibility of experiments.

## Usage

### Command Line

```bash
python -m src.trainers --config work_dir=work_dir --config run=run --config model_kwargs.model_name=ResNet
```

- Use `--config key=value` to update nested config keys. For nested keys, use dot notation (e.g., `model_kwargs.model_name=ResNet`).
- `work_dir` must be specified.

### Example (Python)

```python
from closure.src.trainers import Trainer

trainer = Trainer(
    work_dir="experiments/my_experiment",
    dataset_kwargs={...},
    load_data_kwargs={...},
    model_kwargs={...},
    device="cuda"
)
trainer.fit()
```

## Key Features

- **Flexible configuration**: Easily update configs via command line or Python.
- **Reproducibility**: Each run is isolated in its own folder with full config and logs.
- **Distributed training**: Supports multi-GPU and multi-node setups.
- **Logging**: Both file and console logging, with detailed context (rank, node, etc).
- **Extensible**: Easily plug in new datasets or models.

## Main Methods

- `__init__`: Initializes the Trainer, sets up configs, logging, datasets, and model.
- `comprehend_config`: Processes and applies the configuration.
- `load_data`: Creates data loaders for train/val/test.
- `load_run`: Loads a specific run’s config and model weights.
- `fit`: Trains the model and saves results.

## Notes

- The Trainer object manages a parent directory (the "trainer folder"), specified by its `work_dir` attribute.
- Each run is a subfolder inside this parent directory, typically named after the run (e.g., `work_dir/0`, `work_dir/1`, etc.).
- Each run subfolder contains its own `config.json`, model weights, logs, and results.
- When you want to start a new run, you usually change a few keys in the config (such as hyperparameters or target variables), and the Trainer will save the new configuration and outputs in a new subfolder.
- The Trainer can load a specific run using the `load_run` method, which loads the config and model state from the corresponding subfolder.

## Requirements

- Python 3.x
- PyTorch
- Other dependencies as required by your `datasets`, `models`, and `utilities` modules.

---

**Author:** George Miloshevich  
**License:** MIT License  
**Project:** STRIDE