#!/usr/bin/env python
"""Authoritative test-set evaluation of the production ablation cells.

Loads each cell's best checkpoint via RunLoader (which falls back to direct
torch.load, sidestepping the Lightning `pl_legacy_patch` / `enums` bug that
breaks `closure-train test --ckpt_path` on this stack), runs the model over the
test split in EAGER mode (no dependence on the dead per-job /tmp caches), and
computes per-channel regression metrics in normalised space.

mean(mse) over channels == the MSELoss, i.e. directly comparable to val_loss.

MEMORY: run.metrics() holds all predictions + computes r2/pearson over every
test pixel. The full RunID_5_step100 (201 files x 1023^2) needs ~75 GB and OOMs a
64 GB alloc. TEST_STRIDE subsamples the test files (every Nth) to bound RAM;
~50 full 1023^2 images already give very stable per-channel metrics. Set
TEST_STRIDE=1 (full set) only with a large --mem (>=128 GB).

Env knobs:
  TEST_STRIDE  : keep every Nth test file (default 4 -> ~50 of 201; fits 64 GB)

Writes:
  <cell>/test_metrics.csv     per-channel metrics per cell
  <base>/test_ranking.csv     one aggregate row per cell
"""
import os, gc, csv, glob, traceback
import pandas as pd
from closure.run_loader import RunLoader
try:
    import torch
except Exception:
    torch = None

BASE = os.environ.get("BASE", "models/Lightning/iPiC3D-nathan5-12/production_ablations_step100")
SUB = {"CNN": "runs", "MLP": "runs_MLP"}
CH = {"default": 10, "noE": 7, "noJ": 7, "noJnoE": 4}
BS = {"CNN": 4, "MLP": 8096}                 # eager test batch size (forward-only)
FEATURES = ["default", "noE", "noJ", "noJnoE"]
TARGETS = ["P", "divP"]
ARCHS = os.environ.get("ARCHS", "baseline shallower deeper").split()   # all archs by default
FULL_TEST = os.environ.get("TEST_SPLIT", "./splits/iPiC3D-nathan5-12/RunID_5_step100.csv")  # f2: RunID_5_f2.csv
STRIDE = int(os.environ.get("TEST_STRIDE", "4"))   # f2 test is only 21 files -> use TEST_STRIDE=1


def build_subset_split():
    """Write a strided subset of the test split; return its path (or the full one)."""
    if STRIDE <= 1:
        return FULL_TEST
    with open(FULL_TEST) as fh:
        lines = fh.read().splitlines()
    header, data = lines[0], lines[1:]
    keep = data[::STRIDE]
    out = os.path.abspath(os.path.join(BASE, f"_test_subset_stride{STRIDE}.csv"))
    with open(out, "w") as fh:
        fh.write("\n".join([header] + keep) + "\n")
    print(f"Test subset: every {STRIDE}th file -> {len(keep)}/{len(data)} files ({out})", flush=True)
    return out


TEST_SPLIT = build_subset_split()

rows = []
for model in ["CNN", "MLP"]:
    for feat in FEATURES:
        for targ in TARGETS:
            for arch in ARCHS:
                cell = f"{BASE}/{SUB[model]}/ablate_{feat}_{targ}_{arch}"
                if not glob.glob(os.path.join(cell, "checkpoints", "best-*.ckpt")):
                    print(f"[skip] {model} {feat} {targ} {arch}: no best checkpoint", flush=True)
                    continue
                print(f"=== TEST {model} {feat} {targ} {arch} ===", flush=True)
                run = None
                try:
                    run = RunLoader.from_version_dir(
                        cell, stage="test", device="cuda",
                        data_overrides={"loading_mode": "eager", "batch_size": BS[model],
                                        "test_samples_file": TEST_SPLIT},
                    )
                    m = run.metrics()                      # per-channel, normalised space
                    m.to_csv(os.path.join(cell, "test_metrics.csv"), index=False)
                    rows.append(dict(model=model, feature=feat, target=targ, arch=arch, channels=CH[feat],
                                     test_mse=round(float(m["mse"].mean()), 4),
                                     test_rmse=round(float(m["rmse"].mean()), 4),
                                     test_r2=round(float(m["r2"].mean()), 4),
                                     test_nrmse=round(float(m["nrmse"].mean()), 4)))
                    print(f"  mean_mse={m['mse'].mean():.4f}  mean_r2={m['r2'].mean():.4f}", flush=True)
                except Exception:
                    traceback.print_exc()
                    rows.append(dict(model=model, feature=feat, target=targ, arch=arch,
                                     channels=CH[feat], test_mse="FAIL"))
                finally:
                    del run
                    gc.collect()
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                # write incrementally so partial progress survives a crash/timeout
                pd.DataFrame(rows).to_csv(os.path.join(BASE, "test_ranking.csv"), index=False)

out = pd.DataFrame(rows)
print("\n===== test_ranking.csv =====")
print(out.to_string(index=False))
print(f"\nWrote {os.path.join(BASE, 'test_ranking.csv')}  (TEST_STRIDE={STRIDE})")
print("=== EVAL_TEST_DONE ===")
