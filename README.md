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

```bash
pip install -e .
```

Optional hyper-parameter search extras:

```bash
pip install -e ".[hp]"
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

## Examples

- `examples/tuto_train_haydn.ipynb`: real-data tutorial (Lightning update section added at top)
- `examples/tuto_train_synthetic.ipynb`: synthetic-data tutorial (Lightning update section added at top)
- `examples/optuna/optuna_sweep.py`: Optuna sweep example with Lightning

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

- [examples/tuto_train_haydn.ipynb](closure/examples/tuto_train_haydn.ipynb) — Full tutorial notebook
- Source code docstrings for detailed API documentation

---

**closure** is designed for flexibility, reproducibility, and ease of use in scientific ML workflows. Contributions and feedback are welcome!
