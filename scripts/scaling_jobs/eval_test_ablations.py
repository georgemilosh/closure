#!/usr/bin/env python
"""Authoritative per-channel test/val evaluation of the production ablation cells.

For every cell (model × feature × target × arch) this loads the best checkpoint via
RunLoader, runs it over each requested split in EAGER mode, and computes per-channel
regression metrics in normalised space (so mean(mse) over channels == the MSELoss,
directly comparable to val_loss), then plots the per-channel R^2 heatmaps.

RunLoader is used instead of `closure-train test --ckpt_path` because the latter hits
the Lightning `pl_legacy_patch` / `enums` bug on this stack. Test and validation go
through the same path -- the val split is just passed as the loader's `test_samples_file`
(RunID_0 instead of RunID_5) -- so test and val metrics are computed identically.

MEMORY: run.metrics() holds every prediction to compute r2/pearson over all pixels of a
split. The full RunID_5_step100 (201 files x 1023^2) needs ~75 GB and OOMs a 64 GB alloc.
{TEST,VAL}_STRIDE keeps every Nth file to bound RAM (~50 full 1023^2 images already give
stable per-channel metrics); use stride=1 only with a large --mem (>=128 GB).

Env knobs:
  EVAL_SPLITS  which splits to evaluate (default "test val"; "test" for the old behavior)
  TEST_SPLIT   test split csv (default RunID_5_step100.csv; f2: RunID_5_f2_step100.csv)
  TEST_STRIDE  keep every Nth test file (default 4 -> ~50 of 201; fits 64 GB)
  VAL_SPLIT    val split csv (default: read val_samples_file from each cell's config.yaml)
  VAL_STRIDE   keep every Nth val file (default: same as TEST_STRIDE, for matched RAM)
  SKIP_FIGURES set to 1 to skip the heatmaps (e.g. when matplotlib is unavailable)

Writes:
  <cell>/{test,val}_metrics.csv   per-channel metrics, one file per cell per split
  <base>/{test,val}_ranking.csv   one aggregate row per cell (mean over channels)
  <base>/channel_metrics.csv      all cells x both splits, one row per channel (rebuilt at
                                  the end from every *_metrics.csv on disk, so it stays
                                  complete even across separate invocations)
  <base>/figs/fig_channel_r2_<model>_<split>_<target>.png
                                  per-channel R^2 heatmaps (target channel x feature x arch);
                                  self-contained -- no per-folder make_figures.py needed
"""
import os, gc, csv, glob, traceback
import pandas as pd
import yaml
from closure.run_loader import RunLoader
try:
    import torch
except Exception:
    torch = None

BASE = os.environ.get("BASE", "models/Lightning/iPiC3D-nathan5-12/production_ablations_step100")
SUB = {"CNN": "runs", "MLP": "runs_MLP"}
CH = {"default": 10, "noE": 7, "noJ": 7, "noJnoE": 4}
BS = {"CNN": 4, "MLP": 8096}                 # eager batch size (forward-only)
FEATURES = ["default", "noE", "noJ", "noJnoE"]
TARGETS = ["P", "divP"]
ARCHS = os.environ.get("ARCHS", "baseline shallower deeper").split()   # all archs by default

EVAL_SPLITS = os.environ.get("EVAL_SPLITS", "test val").split()
FULL_TEST = os.environ.get("TEST_SPLIT", "./splits/iPiC3D-nathan5-12/RunID_5_step100.csv")  # f2: RunID_5_f2_step100.csv
TEST_STRIDE = int(os.environ.get("TEST_STRIDE", "4"))   # f2 test is only 21 files -> use TEST_STRIDE=1
VAL_SPLIT_ENV = os.environ.get("VAL_SPLIT")             # if unset, read from each cell's config.yaml
VAL_STRIDE = int(os.environ.get("VAL_STRIDE", str(TEST_STRIDE)))


def build_subset_split(full_path, stride, tag):
    """Write a strided subset of `full_path`; return its path (or the full one if stride<=1)."""
    if stride <= 1:
        return full_path
    with open(full_path) as fh:
        lines = fh.read().splitlines()
    header, data = lines[0], lines[1:]
    keep = data[::stride]
    out = os.path.abspath(os.path.join(BASE, f"_{tag}_subset_stride{stride}.csv"))
    with open(out, "w") as fh:
        fh.write("\n".join([header] + keep) + "\n")
    print(f"{tag} subset: every {stride}th file -> {len(keep)}/{len(data)} files ({out})", flush=True)
    return out


def val_split_from_configs():
    """Resolve the val split csv from the first cell config that has one (all cells share it)."""
    for model in ["CNN", "MLP"]:
        for feat in FEATURES:
            for targ in TARGETS:
                for arch in ARCHS:
                    cfg_path = f"{BASE}/{SUB[model]}/ablate_{feat}_{targ}_{arch}/config.yaml"
                    if os.path.exists(cfg_path):
                        vf = yaml.safe_load(open(cfg_path)).get("data", {}).get("val_samples_file")
                        if vf:
                            return vf
    raise RuntimeError("VAL_SPLIT not set and no cell config.yaml with a val_samples_file was found.")


def split_file_for(split):
    """Return the (possibly strided) split csv path for split in {test, val}."""
    if split == "test":
        return build_subset_split(FULL_TEST, TEST_STRIDE, "test")
    if split == "val":
        vf = VAL_SPLIT_ENV or val_split_from_configs()
        return build_subset_split(vf, VAL_STRIDE, "val")
    raise ValueError(f"unknown split {split!r} (expected test|val)")


def eval_split(split):
    """Evaluate every cell on `split`, writing <cell>/<split>_metrics.csv and <base>/<split>_ranking.csv."""
    split_file = split_file_for(split)
    print(f"\n########## SPLIT={split}  ({split_file}) ##########", flush=True)
    rows = []
    for model in ["CNN", "MLP"]:
        for feat in FEATURES:
            for targ in TARGETS:
                for arch in ARCHS:
                    cell = f"{BASE}/{SUB[model]}/ablate_{feat}_{targ}_{arch}"
                    if not glob.glob(os.path.join(cell, "checkpoints", "best-*.ckpt")):
                        print(f"[skip] {split} {model} {feat} {targ} {arch}: no best checkpoint", flush=True)
                        continue
                    print(f"=== {split.upper()} {model} {feat} {targ} {arch} ===", flush=True)
                    run = None
                    try:
                        run = RunLoader.from_version_dir(
                            cell, stage="test", device="cuda",
                            data_overrides={"loading_mode": "eager", "batch_size": BS[model],
                                            "test_samples_file": split_file},
                        )
                        m = run.metrics()                  # per-channel, normalised space
                        m.to_csv(os.path.join(cell, f"{split}_metrics.csv"), index=False)
                        rows.append(dict(model=model, feature=feat, target=targ, arch=arch, channels=CH[feat],
                                         **{f"{split}_mse": round(float(m["mse"].mean()), 4),
                                            f"{split}_rmse": round(float(m["rmse"].mean()), 4),
                                            f"{split}_r2": round(float(m["r2"].mean()), 4),
                                            f"{split}_nrmse": round(float(m["nrmse"].mean()), 4)}))
                        print(f"  mean_mse={m['mse'].mean():.4f}  mean_r2={m['r2'].mean():.4f}", flush=True)
                    except Exception:
                        traceback.print_exc()
                        rows.append(dict(model=model, feature=feat, target=targ, arch=arch,
                                         channels=CH[feat], **{f"{split}_mse": "FAIL"}))
                    finally:
                        del run
                        gc.collect()
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    # write incrementally so partial progress survives a crash/timeout
                    pd.DataFrame(rows).to_csv(os.path.join(BASE, f"{split}_ranking.csv"), index=False)
    return pd.DataFrame(rows)


def write_combined_channel_metrics():
    """Concatenate every <cell>/{test,val}_metrics.csv into one long per-channel table.
    Scans disk (not just this run's results) so channel_metrics.csv is complete even when
    test and val were evaluated in separate invocations."""
    frames = []
    for model, sub in SUB.items():
        for cell in sorted(glob.glob(os.path.join(BASE, sub, "ablate_*"))):
            tag = os.path.basename(cell).replace("ablate_", "")
            try:
                feature, target, arch = tag.split("_", 2)
            except ValueError:
                continue
            for split in ["test", "val"]:
                mpath = os.path.join(cell, f"{split}_metrics.csv")
                if not os.path.exists(mpath):
                    continue
                m = pd.read_csv(mpath)
                m.insert(0, "split", split)
                m.insert(1, "model", model)
                m.insert(2, "feature", feature)
                m.insert(3, "target", target)
                m.insert(4, "arch", arch)
                frames.append(m)
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    dest = os.path.join(BASE, "channel_metrics.csv")
    out.to_csv(dest, index=False)
    print(f"\nWrote {dest}: {len(out)} per-channel rows "
          f"({', '.join(sorted(out['split'].unique()))})")


HEAT_FEAT = ["default", "noJ", "noE", "noJnoE"]   # heatmap row order


def plot_channel_r2_heatmaps():
    """Per-channel R^2 heatmaps from channel_metrics.csv: one PNG per (model × split × target),
    faceted by target channel, rows=feature, cols=arch. Written to <base>/figs/."""
    if os.environ.get("SKIP_FIGURES", "0") == "1":
        return
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(matplotlib unavailable: {exc} -- skipping heatmaps)")
        return
    cm_path = os.path.join(BASE, "channel_metrics.csv")
    if not os.path.exists(cm_path):
        print("(no channel_metrics.csv -- skipping heatmaps)")
        return
    cm = pd.read_csv(cm_path)
    cm["r2"] = pd.to_numeric(cm["r2"], errors="coerce")
    figs_dir = os.path.join(BASE, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    study = os.path.basename(os.path.normpath(BASE))
    for model in SUB:
        for split in sorted(cm["split"].unique()):
            for target in TARGETS:
                sub = cm[(cm.model == model) & (cm.split == split) & (cm.target == target)]
                if sub.empty:
                    continue
                meta = (sub[["channel", "channel_index"]].dropna(subset=["channel"])
                        .drop_duplicates().sort_values("channel_index"))
                channels = meta["channel"].tolist() or sorted(sub.channel.unique())
                archs = [a for a in ARCHS if a in sub.arch.unique()]
                tables = {ch: (sub[sub.channel == ch].groupby(["feature", "arch"], as_index=False)
                               .agg(r2=("r2", "mean")).pivot(index="feature", columns="arch", values="r2")
                               .reindex(index=HEAT_FEAT, columns=archs)) for ch in channels}
                allv = np.concatenate([t.to_numpy().ravel() for t in tables.values()])
                allv = allv[np.isfinite(allv)]
                if allv.size == 0:
                    continue
                vmin, vmax = float(allv.min()), float(allv.max())
                ncols = 3
                nrows = int(np.ceil(len(channels) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                                         constrained_layout=True, sharex=True, sharey=True, squeeze=False)
                axes = axes.ravel()
                im = None
                for idx, ax in enumerate(axes):
                    if idx >= len(channels):
                        ax.axis("off")
                        continue
                    vals = tables[channels[idx]].to_numpy()
                    im = ax.imshow(vals, cmap="YlGnBu", aspect="auto", vmin=vmin, vmax=vmax)
                    ax.set_title(channels[idx], fontsize=11)
                    ax.set_xticks(np.arange(len(archs)))
                    ax.set_xticklabels(archs, fontsize=8)
                    ax.set_yticks(np.arange(len(HEAT_FEAT)))
                    ax.set_yticklabels(HEAT_FEAT, fontsize=8)
                    if idx % ncols == 0:
                        ax.set_ylabel("input features")
                    if idx // ncols == nrows - 1:
                        ax.set_xlabel("architecture")
                    for i in range(vals.shape[0]):
                        for j in range(vals.shape[1]):
                            v = vals[i, j]
                            if np.isfinite(v):
                                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                                        color="white" if v > (vmin + vmax) / 2 else "black")
                if im is not None:
                    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="R²")
                fig.suptitle("Determination score (R²) by target channel, feature ablation, architecture\n"
                             f"{model} — target {target} — {split.upper()} split   [{study}]", fontsize=12)
                fname = f"fig_channel_r2_{model}_{split}_{target}.png"
                fig.savefig(os.path.join(figs_dir, fname), dpi=150, bbox_inches="tight")
                plt.close(fig)
                print("wrote", os.path.join(figs_dir, fname))


for split in EVAL_SPLITS:
    out = eval_split(split)
    print(f"\n===== {split}_ranking.csv =====")
    print(out.to_string(index=False))
    stride = TEST_STRIDE if split == "test" else VAL_STRIDE
    print(f"\nWrote {os.path.join(BASE, f'{split}_ranking.csv')}  ({split.upper()}_STRIDE={stride})")

write_combined_channel_metrics()
plot_channel_r2_heatmaps()
print("=== EVAL_TEST_DONE ===")
