# Agent A: ML Pipeline Refactor (Claude Opus 4.6)

You are refactoring the `closure` repository (`georgemilosh/closure`, branch `main`) from a research codebase into a publishable open-source Python package. You are **Agent A** — responsible for the ML/training pipeline. A parallel **Agent B** handles data processing (plasma physics, I/O, scripts). You must not edit Agent B's files.

> **Reference**: The full coordinated plan is at `.github/prompts/plan-closureOpenSourceRefactor.prompt.md`. Read it for full context if anything below is unclear.

---

## Repository Location

```
/volume1/scratch/georgem/closure/
```

Current package directory: `src/` (will be renamed to `closure/` in Phase 0).

---

## Your Role

You own the **ML training pipeline**: Trainer class, models, datasets, dataloaders, evaluation, visualization, config management, packaging, and CI/CD.

---

## Target Architecture (your files highlighted)

After all phases, the package will look like this. Files marked **YOU** are yours to create/edit. Files marked **B** belong to Agent B — read only.

```
closure/                          # repo root
├── closure/                      # Python package (renamed from src/)
│   ├── __init__.py               # YOU — __version__, public API re-exports
│   ├── config.py                 # YOU (NEW) — TrainerConfig dataclass, paths loading
│   ├── trainers.py               # YOU — slim orchestrator using config.py
│   ├── models.py                 # YOU — type hints added; PyNet, FCNN, ResNet, MLP, CNet
│   ├── datasets.py               # YOU — type hints added; DataFrameDataset
│   ├── dataloaders.py            # YOU — type hints added; DistributedSampler, SubSampler
│   ├── evaluation.py             # YOU (NEW) — ML eval functions (from utilities.py)
│   ├── visualization.py          # YOU (NEW) — plotting functions (from utilities.py)
│   ├── plasma.py                 # B (NEW) — DO NOT EDIT
│   ├── utilities.py              # SHARED — you extract, B slims (see rules below)
│   ├── read_pic.py               # B — DO NOT EDIT
│   ├── logconfig.py              # YOU
│   └── runs.py                   # YOU — updated imports
├── scripts/                      # B — DO NOT EDIT
├── tests/
│   ├── conftest.py               # B owns — request fixtures via conftest_ml.py
│   ├── conftest_ml.py            # YOU (optional) — your ML-specific fixtures
│   ├── fixtures/mock_data.h5     # B owns
│   ├── test_config.py            # YOU
│   ├── test_models.py            # YOU
│   ├── test_datasets.py          # YOU
│   ├── test_evaluation.py        # YOU
│   ├── test_visualization.py     # YOU
│   ├── test_trainers.py          # YOU
│   ├── test_plasma.py            # B — DO NOT EDIT
│   ├── test_read_pic.py          # B — DO NOT EDIT
│   └── test_utilities.py         # B — DO NOT EDIT
├── pyproject.toml                # YOU
├── .github/workflows/            # YOU
├── .pre-commit-config.yaml       # YOU
├── LICENSE                       # YOU
└── ... (README, CONTRIBUTING etc. — Agent B)
```

---

## Execution Plan

### Prerequisites: Phase 0 (Shared, Sequential)

Phase 0 is done **before** you start. Verify these are complete before beginning Phase 1A:

1. Branch `refactor/open-source-prep` exists, branched from `main`
2. `src/` has been renamed to `closure/`
3. Dead code removed: `Plot_PIC.py`, `Pressure.py`
4. Scripts moved to `scripts/`: compute_field_images.py, compute_spectrum.py, compute_flux.py, pkl_to_h5.py, ipic3d_to_ecsim_h5.py, convert_pkl_to_npz.py, convert_pth2pt.py, downscale.py, downscale_to_h5.py, datasplit.py, plot_experiment.py, run_compare_runs.py, timing.py, all .sh files
5. `paths.yaml.example` exists at repo root; `paths.yaml` in `.gitignore`
6. Minimal stub `closure/config.py` with `load_paths()` exists
7. Basic `pyproject.toml` with pytest config exists
8. `tests/` directory with `conftest.py` and `tests/fixtures/` exist

**After verifying Phase 0**, create your sub-branch:
```bash
git checkout refactor/open-source-prep
git checkout -b refactor/phase1a-ml-pipeline
```

---

### Phase 1A: ML Pipeline (YOUR WORK)

Execute these steps **in order**. Run `pytest` after each step.

#### Step 1A.1: Extend `config.py` with TrainerConfig dataclass

Extend the stub `closure/config.py` (already has `load_paths()`):
1. Define `TrainerConfig` dataclass with typed fields:
   - `work_dir`, `dataset_kwargs`, `model_kwargs`, `device`, `log_name`, `log_level`, `num_workers`, `force`, `timing_name`
   - Distributed: `world_size`, `rank`, `gpus_per_node`, `local_rank`
2. Add `load_config()` — reads config.json + paths.yaml, returns `TrainerConfig`
3. Move `set_nested_config()` from `utilities.py` to `config.py`
4. **TEST**: Write and run `pytest tests/test_config.py`

#### Step 1A.2: Create `evaluation.py` (extract from utilities.py)

1. Create `closure/evaluation.py`
2. Move these functions from `utilities.py`:
   - `evaluate_loss()`, `compare_runs()`, `compare_metrics()`, `compute_loss()`
   - `transform_features()`, `transform_targets()`, `normalize_input()`
   - `pred_unnormalize()` (aliased as `unnormalize_output`), `prediction2data()`
   - `pred_ground_targets()`, `parse_score()`
3. `evaluation.py` imports `trainers` (one-way — this BREAKS the circular import)
4. Leave **lazy** backward-compat stubs in `utilities.py`:
   ```python
   def transform_targets(*args, **kwargs):  # backward compat
       from closure.evaluation import transform_targets as _f
       return _f(*args, **kwargs)
   ```
   **CRITICAL**: Do NOT use top-level `from closure.evaluation import ...` in utilities.py — that recreates the cycle: `utilities → evaluation → trainers → utilities`.
5. **TEST**: Write and run `pytest tests/test_evaluation.py`

#### Step 1A.3: Create `visualization.py` (extract from utilities.py)

1. Create `closure/visualization.py`
2. Move: `graph_pred_targets()`, `plot_pred_targets()`
3. These import from `evaluation.py`
4. **TEST**: Write `tests/test_visualization.py` — import smoke test, basic call with mock data

#### Step 1A.4: Simplify Trainer class

1. Refactor `trainers.py` to use `TrainerConfig` dataclass instead of raw dict
2. Extract pure functions:
   - `setup_device(config: TrainerConfig) -> torch.device`
   - `create_datasets(config: TrainerConfig) -> tuple[Dataset, ...]`
   - `save_results(model, loss_dict, path) -> None`
3. Keep `Trainer` as slim orchestrator calling these functions
4. Update `Trainer.__init__` to accept `TrainerConfig` (keep backward-compat kwargs)
5. Add `main()` entrypoint functions to `trainers.py` and `runs.py` (for CLI: `closure-train`, `closure-runs`)
6. Update `runs.py` imports
7. **TEST**: Run `pytest tests/test_trainers.py tests/test_models.py tests/test_datasets.py`

#### Step 1A.5: Write remaining tests

1. `tests/test_config.py` — TrainerConfig creation, set_nested_config, load_paths
2. `tests/test_models.py` — PyNet/FCNN/ResNet/MLP instantiation, forward pass shapes, save/load
3. `tests/test_datasets.py` — DataFrameDataset normalization, filtering, prescaling (mock data)
4. `tests/test_evaluation.py` — compute_loss, transform_targets, evaluate_loss
5. `tests/test_trainers.py` — Integration: Trainer(config) → fit() on mock HDF5 data
6. **Run**: `pytest tests/` — all must pass

#### Step 1A.6: Add type hints and docstrings

1. Add type annotations to all public functions in: `trainers.py`, `config.py`, `evaluation.py`, `visualization.py`, `models.py`
2. Standardize to NumPy-style docstrings
3. Add `__all__` to each module

**Commit and push your branch**: `refactor/phase1a-ml-pipeline`

---

### Phase 2: Integration & Packaging (SEQUENTIAL — after Agent B also finishes)

You **lead** packaging. Agent B reviews.

#### Step 2.1: Merge and verify
1. Merge both sub-branches into `refactor/open-source-prep`
2. `pytest tests/` — everything passes
3. `python -c "import closure"` — no circular imports

#### Step 2.2: Create `__init__.py` with public API
```python
from closure.config import TrainerConfig, load_paths, load_config
from closure.trainers import Trainer
from closure.models import PyNet
from closure.datasets import DataFrameDataset
from closure.plasma import get_Ohm, get_PS_2D_field, get_Az, get_J_perp
from closure.read_pic import get_exp_times, read_data_ipic3d, build_XY
__version__ = "0.1.0"
```

#### Step 2.3: Complete `pyproject.toml`
```toml
[project]
name = "closure"
version = "0.1.0"
description = "ML framework for fluid closure of PIC plasma simulations"
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24,<2.0",
    "pandas>=1.5",
    "scipy>=1.10",
    "h5py>=3.8",
    "matplotlib>=3.6",
    "joblib>=1.2",
    "torchmetrics>=1.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.4", "pre-commit>=3.0"]
optuna = ["optuna>=3.0"]
# NOTE: For GPU/CUDA support, install torch separately with the appropriate
# --index-url for your CUDA version. See README for instructions.

[project.scripts]
closure-train = "closure.trainers:main"
closure-runs = "closure.runs:main"

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: marks tests as slow", "gpu: requires GPU"]

[tool.ruff]
line-length = 120
[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
```

#### Step 2.4: Create LICENSE file (MIT)

#### Step 2.5: Verify packaging
1. `pip install -e ".[dev]"`
2. `python -c "from closure import Trainer, TrainerConfig; print('OK')"`
3. `python -c "from closure.plasma import get_Ohm; print('OK')"`
4. `pytest tests/` — all pass
5. `python -m build` — produces sdist + wheel

---

### Phase 3A: CI/CD + Linting (YOUR WORK)

1. Configure Ruff in `pyproject.toml` (already started in 2.3)
2. Create `.pre-commit-config.yaml` (ruff, yaml checks, trailing whitespace)
3. `ruff format .` across entire codebase
4. `ruff check --fix .` for auto-fixable issues
5. Create `.github/workflows/ci.yml` (Python 3.10–3.12 matrix, ruff, pytest)
6. Create `.github/workflows/publish.yml` (tag-triggered PyPI publish)
7. Commit and verify CI passes

---

## Rules & Constraints

### File Ownership — DO NOT violate
| Scope | Files |
|-------|-------|
| **You own** | `closure/trainers.py`, `config.py`, `evaluation.py`, `visualization.py`, `models.py`, `datasets.py`, `dataloaders.py`, `logconfig.py`, `runs.py`, `__init__.py` |
| **You own** | `tests/test_trainers.py`, `test_models.py`, `test_datasets.py`, `test_evaluation.py`, `test_visualization.py`, `test_config.py`, `tests/conftest_ml.py` |
| **You own** | `pyproject.toml`, `.github/workflows/`, `.pre-commit-config.yaml`, `LICENSE` |
| **Shared (you extract, B cleans)** | `closure/utilities.py` |
| **DO NOT EDIT** | `closure/plasma.py`, `read_pic.py`, `utils_ECSIM.py`, `scripts/*`, `tests/conftest.py`, `tests/fixtures/*`, `tests/test_plasma.py`, `test_read_pic.py`, `test_utilities.py`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, etc. |

### Backward-Compat Stubs in utilities.py
When you move functions out of `utilities.py`, leave **lazy** stubs:
```python
def transform_targets(*args, **kwargs):  # backward compat
    from closure.evaluation import transform_targets as _f
    return _f(*args, **kwargs)
```
**NEVER** use top-level `from closure.evaluation import ...` in utilities.py — that recreates the circular import chain (`utilities → evaluation → trainers → utilities`).

### Circular Import Resolution
- `evaluation.py` imports `trainers` (one-way) ✓
- `trainers.py` imports `utilities` (one-way) ✓  
- `utilities.py` does NOT import `evaluation` or `trainers` at top level ✓
- No cycle.

### Test Fixtures
- Agent B owns `tests/conftest.py` and `tests/fixtures/mock_data.h5`
- If you need ML-specific fixtures, create `tests/conftest_ml.py`
- Always run `pytest tests/` (the full suite) after each step

### Coordination with Agent B
- Agent B is extracting plasma physics functions from `utilities.py` into `plasma.py` — do NOT touch those functions
- Agent B will slim `utilities.py` after you extract your functions — your extractions happen first
- Agent B will also update `read_pic.py`'s internal imports for functions that moved to `plasma.py`
- `plasma.py` is pure numpy/scipy — no torch dependency

---

## Verification Checklist (run after all phases)

```bash
pip install -e ".[dev]"
python -c "import closure"
python -c "from closure import Trainer, TrainerConfig; print('OK')"
python -c "from closure.evaluation import transform_targets; print('OK')"
pytest tests/ --cov=closure
ruff check .
ruff format --check .
python -m build
twine check dist/*
```

All must pass with zero errors.
 