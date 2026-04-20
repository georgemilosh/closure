# Phase 1A Deviations Report — Agent A → Agent B

**Branch**: `refactor/phase1a-ml-pipeline`  
**Commit**: `6a58fc5`  
**Date**: 2025 (Phase 1A completion)

This document summarizes where Agent A's actual Phase 1A implementation deviated from the original plan in `agentA-mlPipeline.prompt.md`. Agent B should review these before starting Phase 1B to avoid conflicts.

---

## 1. `tests/conftest.py` was edited (Agent B's file)

**Plan**: Agent A should NOT edit `conftest.py` — it belongs to Agent B.  
**What happened**: The existing `conftest.py` had broken syntax (missing indentation and `file` instead of `__file__`), which prevented *all* pytest collection. Agent A applied a minimal 2-line fix to unblock testing:

```python
# Before (broken)
@pytest.fixture
def fixtures_dir():
return pathlib.Path(file).parent / "fixtures"

# After (fixed)
@pytest.fixture
def fixtures_dir():
    return pathlib.Path(__file__).parent / "fixtures"
```

**Impact for Agent B**: The fix is correct. You can build on it or replace it entirely — Agent A won't touch this file again.

---

## 2. `TrainerConfig` has extra fields not in the plan

**Plan**: Fields listed were `work_dir`, `dataset_kwargs`, `model_kwargs`, `device`, `log_name`, `log_level`, `num_workers`, `force`, `timing_name`, `world_size`, `rank`, `gpus_per_node`, `local_rank`.  
**What happened**: Agent A added two additional fields needed by the existing `Trainer` code:
- `load_data_kwargs: dict | None = None` — used by `Trainer.load_data()`
- `mode_test: bool = False` — controls train vs. test mode

**Impact for Agent B**: None — these are internal to the ML pipeline. You won't need to interact with them.

---

## 3. `evaluation.py` imports `trainers` lazily, not at top-level

**Plan**: "evaluation.py imports trainers (one-way)" — implied a top-level import.  
**What happened**: Only `compare_runs()` and `compare_metrics()` import `trainers`, and they do so lazily inside the function body:

```python
def compare_runs(work_dirs, ...):
    from closure import trainers as tr
    ...
```

**Impact for Agent B**: This is actually *better* than the plan — it reduces coupling and avoids import-time side effects. No action needed.

---

## 4. `datasets.py` and `dataloaders.py` — no type hints added

**Plan**: "type hints added" for both `datasets.py` (DataFrameDataset) and `dataloaders.py` (DistributedSampler, SubSampler).  
**What happened**: Agent A added module-level docstrings and `__all__` exports but did NOT add type annotations to function signatures. These files are otherwise unchanged.

**Impact for Agent B**: These files remain Agent A's to modify in later phases. No conflict for you, but note that the type-hints task is deferred.

---

## 5. `__init__.py` — no public API re-exports yet

**Plan**: `__init__.py` was listed as "YOU — `__version__`, public API re-exports".  
**What happened**: Only `__version__ = "0.1.0"` exists. No re-exports (e.g., `from closure.config import TrainerConfig`).

**Impact for Agent B**: This is intentional — the public API re-exports are a Phase 2 task (Step 2.2). Don't add your own re-exports yet; they'll be done during integration.

---

## 6. `conftest_ml.py` was not created

**Plan**: Agent A could optionally create `tests/conftest_ml.py` for ML-specific fixtures.  
**What happened**: Not created. All ML test fixtures are defined inline within each test file using `@pytest.fixture`.

**Impact for Agent B**: No shared ML fixtures file exists. If you need to reference ML test patterns, look at the individual test files (`test_models.py`, `test_datasets.py`, etc.).

---

## 7. `utilities.py` still has a top-level `try: from . import trainers` block

**Plan**: utilities.py should NOT import trainers or evaluation at top level to avoid circular imports.  
**What happened**: The original `try: import torch; from . import trainers as tr` block at the top of `utilities.py` was left in place. It is guarded by `try/except ImportError` so it doesn't crash, but it does create a soft coupling at import time.

**Impact for Agent B**: When you slim `utilities.py`, you should:
1. Remove this top-level `from . import trainers as tr` import
2. Any remaining functions that need `trainers` should use lazy imports inside the function body (matching the stub pattern Agent A established)

---

## 8. `parse_score()` bug fix — "MSE" → "MSELoss"

**Plan**: No bug fix was mentioned.  
**What happened**: Agent A discovered and fixed a pre-existing bug where `parse_score("MSE")` crashed because `torch.nn.MSE` does not exist. The fix maps `"MSE"` to `"MSELoss"` before `getattr(torch.nn, ...)`.

**Impact for Agent B**: Purely positive — existing code that called `parse_score("MSE")` will now work. No action needed.

---

## 9. CI/CD and linting files not created

**Plan** (Phase 3A): `.github/workflows/ci.yml`, `.github/workflows/publish.yml`, `.pre-commit-config.yaml`.  
**What happened**: These were not created — they are Phase 3A tasks, not Phase 1A.

**Impact for Agent B**: Expected. These will be created after both Phase 1A and 1B merge.

---

## Summary Table

| Item | Planned | Actual | Severity |
|------|---------|--------|----------|
| `conftest.py` edited | Don't edit (Agent B's) | Minimal fix applied | Low — necessary to unblock tests |
| `TrainerConfig` fields | 13 fields | 14 fields (+`load_data_kwargs`, `mode_test`) | None — internal to ML |
| `evaluation.py → trainers` import | Top-level one-way | Lazy inside functions | None — better than planned |
| Type hints on `datasets.py`, `dataloaders.py` | Add type hints | Docstrings only, no type hints | Low — deferred |
| `__init__.py` re-exports | Add public API | Only `__version__` | None — Phase 2 task |
| `conftest_ml.py` | Optional | Not created | None |
| `utilities.py` top-level trainers import | Remove | Left in place | Medium — Agent B should clean up |
| `parse_score("MSE")` bug | Not planned | Fixed | None — beneficial |

---

## Key Takeaway for Agent B

The most important item is **#7**: `utilities.py` still has a top-level `from . import trainers as tr` inside a `try/except` block. When you slim `utilities.py` in Phase 1B, remove this import and ensure any remaining functions use lazy imports. Everything else is either beneficial, internal to ML, or deferred to later phases.
