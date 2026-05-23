# Lazy NPZ Loading Tutorial

A practical, end-to-end guide to the lazy data-loading pipeline added to
`closure`. This document explains what the feature does, why it exists,
how to turn it on, and how to tune it — with explicit definitions for
every piece of jargon you will encounter along the way.

If you have never run a training job in this repository before, start by
reading [README.md](../README.md) and [INSTALLATION.md](../INSTALLATION.md).
This tutorial assumes you can already launch a training run with the
**eager** loader and want to know how to switch to the **lazy** loader.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Background and terminology](#2-background-and-terminology)
3. [When (and when not) to use lazy loading](#3-when-and-when-not-to-use-lazy-loading)
4. [Architecture overview](#4-architecture-overview)
5. [Configuration reference](#5-configuration-reference)
6. [The two batch samplers](#6-the-two-batch-samplers)
7. [The single-open NPZ fast path](#7-the-single-open-npz-fast-path)
8. [DDP / multi-GPU safety](#8-ddp--multi-gpu-safety)
9. [Smoke configs and end-to-end recipes](#9-smoke-configs-and-end-to-end-recipes)
10. [Performance tuning checklist](#10-performance-tuning-checklist)
11. [Troubleshooting & FAQ](#11-troubleshooting--faq)
12. [Testing the lazy stack](#12-testing-the-lazy-stack)
13. [Reading the source](#13-reading-the-source)

---

## 1. TL;DR

Add four knobs to the `data:` block of your YAML and you are done:

```yaml
data:
  loading_mode: lazy_npz          # switch from eager (default) to lazy
  sample_cache_size: 1            # FCNN/CNN; for MLP set = chunk_window
  chunk_window: 1                 # only used when flatten: true
  persistent_workers: true        # keep per-worker caches warm across epochs
  prefetch_factor: 2              # standard DataLoader knob
```

Everything else (your model, optimizer, prescalers, normalization,
metrics) is unchanged. The lazy loader is a drop-in replacement for the
eager `DataFrameDataset`: it produces tensors with the **same shapes and
the same normalization statistics**, just without holding the entire
training set in RAM.

Production smoke configs:

- FCNN / CNN: [configs/testing/testing_small_npz_lazy.yaml](../configs/testing/testing_small_npz_lazy.yaml)
- MLP (pixel-wise): [configs/testing/testing_small_npz_lazy_mlp.yaml](../configs/testing/testing_small_npz_lazy_mlp.yaml)

Run either with:

```bash
closure-train fit --config configs/testing/testing_small_npz_lazy.yaml
```

---

## 2. Background and terminology

A handful of terms appear repeatedly. Skim this section once and refer
back as needed.

| Term | Meaning |
|---|---|
| **Eager loading** | The default mode in `closure`. The entire training and validation splits are read from disk into a single in-memory tensor at `setup()` time. Fast at training but limited by RAM. Implemented by `DataFrameDataset`. |
| **Lazy loading** | Only metadata (file list, normalization stats, shape) is kept in RAM. Each `__getitem__` reads one file from disk on demand. Implemented by `LazyNPZDataFrameDataset`. |
| **NPZ** | NumPy's zipped archive format (`.npz`). One file per snapshot; each archived array is a key (e.g. `Bx`, `By`, `Pxx_e`). |
| **Snapshot / file** | One time step from the underlying simulation, stored as one `.npz` (or `.h5` etc.) file on disk. |
| **Sample** | One unit returned by `dataset.__getitem__`. With `flatten=False` a sample is one whole snapshot tensor of shape `(C, H, W)`. With `flatten=True` a sample is one *pixel* — a vector of shape `(C,)`. |
| **`flatten`** | Toggles the two interpretations above. `False` for CNNs/FCNNs that consume images, `True` for pixel-wise MLPs. |
| **Patch** | A spatially cropped sub-region of a snapshot, produced by the `RandomCrop` transform when `patch_dim` is set. Only used with `flatten=False`. |
| **DataLoader worker** | A subprocess spawned by PyTorch's `DataLoader` to load samples in parallel. Set with `num_workers`. Each worker gets its own copy of the dataset object — and therefore its own cache. |
| **LRU cache** | "Least Recently Used" cache. A small dictionary that drops the oldest unused entry when full. We use one per worker, keyed by file index. |
| **Sampler** | The object that decides *which* dataset indices to draw, and in *what order*. PyTorch supports a plain `Sampler` (yields indices) and a `BatchSampler` (yields lists of indices). |
| **Cache hit / miss** | When `dataset.__getitem__(i)` needs file `f`, a *hit* means the file's decoded arrays are already in the LRU cache and no disk I/O happens. A *miss* means we open the file. The two samplers are designed to maximize hits. |
| **Prescaler** | A per-channel monotonic transform applied **before** normalization (e.g. `log`, `arcsinh`). Configured via `prescaler_features` / `prescaler_targets`. Identical in eager and lazy. |
| **Normalization stats** | Per-channel mean and std, stored as `X.pkl` (features) and `y.pkl` (targets) under `norm_folder`. Computed once on the training split and reused for val/test. |
| **DDP** | "Distributed Data Parallel" — Lightning's multi-GPU mode. Multiple Python processes train in lock-step, one per GPU. |
| **`rp.read_features_targets`** | The canonical reader in `closure/read_pic.py` that supports many simulation formats. It reopens a file **once per channel** which can be a bottleneck for `.npz`. |
| **Fast path** | A specialized loader inside `LazyNPZDataFrameDataset` that bypasses `rp.read_features_targets` for plain `.npz` files and opens each file **once** per access. Auto-detected; no configuration needed. |

---

## 3. When (and when not) to use lazy loading

### Use `loading_mode: lazy_npz` when

- Your training split no longer fits in RAM. (Symptom: `setup()` either
  OOM-kills the job or pushes you into swap.)
- You are training on the full iPiC3D production split (hundreds of
  snapshots × multi-channel volumes).
- You want to share data across many concurrent jobs on the cluster
  without each holding a private RAM-resident copy.
- You want to use the `RandomCrop` patch augmentation with many crops
  per file per epoch (oversampling).

### Stick with `loading_mode: eager` (the default) when

- The entire dataset fits comfortably in RAM. Eager is **always faster
  per step** once it has loaded, because every access is a memory copy
  with zero I/O.
- You are running short experiments or unit tests where the one-time
  upfront load is negligible.
- You depend on `subsample_rate < 1.0` (undersampling). Lazy mode
  supports oversampling natively via the sampler, but undersampling on
  pixels is a no-op in lazy + `flatten: true` (a warning is logged).

### Either is fine when

- Your dataset is medium-sized (fits in RAM but is slow to load).
  Eager pays a one-time load cost; lazy pays a smaller per-batch cost
  but eliminates the upfront wait. Pick by preference.

---

## 4. Architecture overview

```
  ┌────────────────────────────────────────────────────────────────┐
  │                      ClosureDataModule                         │
  │                                                                │
  │   loading_mode == "eager"          loading_mode == "lazy_npz"  │
  │   ─────────────────────            ───────────────────────     │
  │   DataFrameDataset                 LazyNPZDataFrameDataset     │
  │   (all files in RAM)               (only metadata in RAM)      │
  │                                                                │
  │   DataLoader(shuffle=True)         _make_lazy_train_loader():  │
  │                                      flatten=False             │
  │                                        OnePatchPerFileBatch... │
  │                                      flatten=True              │
  │                                        FileChunkedSampler      │
  └────────────────────────────────────────────────────────────────┘
              │                                    │
              ▼                                    ▼
       per-step: in-RAM             per-step: disk read of one file
       tensor slice (no I/O)        → per-worker LRU cache
                                    → CHW tensor / pixel vector
```

Key components, with file links:

- [closure/datasets.py](../closure/datasets.py) — `DataFrameDataset` (eager),
  `LazyNPZDataFrameDataset` (lazy), `_FileLRUCache`,
  `OnePatchPerFileBatchSampler`, `FileChunkedSampler`.
- [closure/datamodule.py](../closure/datamodule.py) — `ClosureDataModule`
  that picks the right dataset class and the right sampler based on the
  YAML config.
- [closure/read_pic.py](../closure/read_pic.py) — the canonical reader
  used as a fallback (and as the source of truth for normalization
  semantics).

### Data flow per training step (lazy mode)

1. The sampler yields one or more global indices.
2. `LazyNPZDataFrameDataset.__getitem__(idx)` decodes
   `file_idx = idx // pixels_per_file` (or `idx` itself for `flatten=False`).
3. The LRU cache is consulted. On a hit we skip straight to step 5.
4. On a miss the file is opened, the requested channels are read, the
   array is converted to channel-first `(C, H, W)`, prescaled, and
   normalized. The result is stored in the cache.
5. For `flatten=True` we take one column out of the `(C, H*W)`
   reshape; otherwise we return the full `(C, H, W)` tensor and (if
   training) apply `RandomCrop`.
6. PyTorch batches and pins the result, the GPU consumes it.

The cache is *per-process*: each `DataLoader` worker has its own. The
two samplers are designed so that consecutive indices keep hitting the
same small set of files, maximizing cache hits.

---

## 5. Configuration reference

All of the knobs live under the `data:` block of your YAML. Defaults
are chosen so that omitting them gives you the legacy eager behavior.

```yaml
data:
  # ───── selecting the loader ─────
  loading_mode: lazy_npz     # "eager" (default) | "lazy_npz"

  # ───── caching ─────
  sample_cache_size: 1       # number of decoded files held per worker
  chunk_window: 1            # files held in flight by FileChunkedSampler
                             # (only used when flatten: true)

  # ───── DataLoader knobs ─────
  num_workers: 12            # standard PyTorch knob; >0 enables multiprocessing
  persistent_workers: true   # keep worker procs (+ their caches) alive across epochs
  prefetch_factor: 2         # batches each worker preloads (PyTorch default)
```

Detailed meanings:

- **`loading_mode`**
  - `"eager"` (default): preloads via `DataFrameDataset`. Same behavior
    as before this feature existed.
  - `"lazy_npz"`: use `LazyNPZDataFrameDataset`. Lightning will pick
    the appropriate sampler automatically (see §6).
- **`sample_cache_size`** — Capacity of the per-worker LRU cache,
  measured in **whole files** (each file holds all requested
  features+targets channels for one snapshot).
  - For `flatten=False` (CNN/FCNN with `OnePatchPerFileBatchSampler`):
    `1` is sufficient — each batch touches each file at most once.
  - For `flatten=True` (MLP with `FileChunkedSampler`): set this
    equal to `chunk_window` so the round-robin keeps every file in
    the window resident.
- **`chunk_window`** — Number of files that the `FileChunkedSampler`
  interleaves per round-robin batch. Only used when `flatten: true`
  and `loading_mode: lazy_npz`. Larger values trade more RAM (per
  worker) for less batch-time correlation.
- **`persistent_workers`** — When `true`, PyTorch keeps DataLoader
  worker processes alive between epochs. This is **important** for
  lazy mode because otherwise every epoch destroys the caches and pays
  the cold-read tax again. Default is `null`, which auto-enables this
  for `lazy_npz` whenever `num_workers > 0`.
- **`prefetch_factor`** — Number of batches each worker prefetches.
  Default is `null` → defer to PyTorch (which uses `2`).

The eager loader ignores `sample_cache_size`, `chunk_window`, etc.

---

## 6. The two batch samplers

A **sampler** decides which indices to draw from the dataset, and in
what order. The lazy loader uses different samplers for the
image-mode (`flatten: false`) and pixel-mode (`flatten: true`)
training paths. Lightning wires this up automatically.

### 6.1 `OnePatchPerFileBatchSampler` (image mode)

Used when `flatten: false` and `loading_mode: lazy_npz`. The
**guarantee**: within any single batch, every sample comes from a
different file (i.e. a different time snapshot).

Why this matters:

- Eliminates *within-batch time correlation*: SGD gradient estimates
  benefit from independent samples. Two random crops from the same
  snapshot share large-scale plasma structure and are not iid.
- Plays perfectly with `sample_cache_size: 1`: each worker only needs
  to hold the *one* file it is currently cropping. The next index it
  is asked for belongs to a different file *and* to a different
  worker, so no eviction race ever happens.

How `oversample` is derived: when you set `subsample_rate: 800` in
lazy + `flatten: false` mode, that value becomes the
sampler's `oversample` parameter. Each file is visited 800 times per
epoch, and `RandomCrop` produces a different patch each time. This
replaces the legacy "make a long `Subset` of repeated indices" trick.

The `__len__` formula is exactly:

```
batches_per_epoch = (num_files // batch_size) * oversample
```

(with `drop_last: false` adding one partial batch per pass).

### 6.2 `FileChunkedSampler` (pixel mode / MLP)

Used when `flatten: true` and `loading_mode: lazy_npz`. The sampler
emits pixel-level global indices, but it does so in a structured
order that keeps the LRU cache hot.

The algorithm:

1. Shuffle the file order.
2. Partition into consecutive groups of size `chunk_window`.
3. For each group, shuffle the pixel order within each file.
4. **Round-robin**: emit pixel `0` from file `a`, pixel `0` from file
   `b`, …, pixel `0` from file `w`, then pixel `1` from `a`, etc.
   until each file in the group has had every pixel emitted.

What this buys you:

- Within any chunk of `chunk_window * pixels_per_file` consecutive
  indices, **at most `chunk_window` distinct files are touched**.
  Therefore a per-worker cache of size `chunk_window` never evicts
  inside a group.
- Adjacent samples in a batch come from *different* files,
  giving you cross-file decorrelation while still amortizing the
  decode cost over `pixels_per_file` pixels per file load.
- Every pixel in every file is emitted **exactly once per epoch** —
  no implicit over/undersampling.

Choose `chunk_window` by your RAM budget per worker:

| `chunk_window` | decorrelation | RAM / worker (≈) |
|---|---|---|
| 1 | weakest (each batch may sit on one file at a time) | 1 decoded file |
| 2–4 | good (each batch mixes 2–4 files) | 2–4 decoded files |
| 8+ | excellent, approaches eager | 8+ decoded files |

> **Important pairing:** `sample_cache_size` *must be ≥* `chunk_window`.
> If it is smaller, the sampler's invariant ("within a group, only
> `window` files are live") gets violated by premature evictions and
> you pay extra disk reads. The smoke config sets them equal.

---

## 7. The single-open NPZ fast path

This is an internal optimization. You do not configure it; it is
auto-detected per dataset instance.

### What it solves

The canonical reader `rp.read_features_targets` is generic: it supports
HDF5, raw arrays, derived species fields, Alfvén unit rescaling, and
spatial slicing. To get there it re-opens the input file **once per
requested channel**. For a 10-feature / 6-target setup that means
16 `np.load` calls per sample — 16× more I/O than necessary.

### What the fast path does

When the dataset is constructed, `LazyNPZDataFrameDataset._maybe_fast_load_npz`
probes the first file. If **all** of the following hold:

1. The file extension is `.npz`.
2. `alfven_units: false`.
3. No `filter_features` or `filter_targets` callbacks.
4. No `choose_x` / `choose_y` / `choose_z` slicing in
   `read_features_targets_kwargs`.
5. Every requested field is a plain string (no nested derived-species
   markers).
6. All requested keys exist in the npz archive.

…then the dataset switches to the fast path **permanently**: every
subsequent file load is a single `np.load` followed by an in-memory
stacking of the requested keys into `(C, H, W)`.

If any of those checks fail, the flag is set to `False` and we
permanently fall back to `rp.read_features_targets`. Either way you
get the **same numbers**: the fast path mirrors `rp`'s off-by-one
slicing (`[0:size-1]`) and `(y, x) → (x, y)` transpose so that
normalization stats and per-pixel values are byte-identical between
the two paths.

You will see an INFO log line on the first successful probe:

```
INFO closure.datasets — npz fast path enabled for /path/to/data
```

…or, when something disables it:

```
INFO closure.datasets — npz fast path disabled: snap_42.npz missing keys ['Foo']; falling back to read_features_targets
```

### Caveats

- **Alfvén units** disable the fast path by design — the rescaling
  uses an experiment-level `.inp` file that `rp` handles for you.
- **Custom filters** (`filter_features` / `filter_targets`) also
  disable it. The filter callbacks are arbitrary Python and we do not
  try to introspect them.
- **Spatial slicing** (`choose_x` / `choose_y` / `choose_z`) is
  rare and disables the fast path because reproducing rp's exact
  slicing semantics for non-default ranges is fragile.

For most production npz datasets none of these apply and the fast path
is on.

---

## 8. DDP / multi-GPU safety

When you launch with multiple GPUs Lightning spawns one Python process
per device. **Each process constructs its own dataset** and therefore
its own `LazyNPZDataFrameDataset`.

The only shared on-disk state is the normalization pickle (`X.pkl`,
`y.pkl`) written under `norm_folder`. The lazy loader handles the
multi-process race like this:

```
if norm files do not yet exist:
    if rank == 0:
        compute stats by streaming over training files
        write X.pkl / y.pkl
        torch.distributed.barrier()       # signal "files ready"
    else:
        torch.distributed.barrier()       # wait
        joblib.load(X.pkl / y.pkl)
```

The barrier ensures that no non-zero rank tries to read the pickle
before rank 0 has finished writing it. When `torch.distributed` is not
initialized (single-process runs) the code path skips the barrier
entirely.

This is implemented in `LazyNPZDataFrameDataset._prepare_normalization_params`
and unit-tested in [tests/test_norm_ddp.py](../tests/test_norm_ddp.py).

Once `X.pkl` / `y.pkl` exist on disk every process loads them
independently from `joblib.load` — no barrier needed.

---

## 9. Smoke configs and end-to-end recipes

Two ready-to-run configurations live under `configs/testing/`. They
exist primarily to exercise the lazy stack end-to-end on a small split,
but you can use them as templates.

### 9.1 FCNN / CNN (image mode)

[configs/testing/testing_small_npz_lazy.yaml](../configs/testing/testing_small_npz_lazy.yaml)

```yaml
data:
  flatten: false
  patch_dim: [64, 64]
  batch_size: 4              # must be <= num_train_files (4 in this smoke split)
  loading_mode: lazy_npz
  sample_cache_size: 1
  persistent_workers: true
```

What happens at runtime:

- `LazyNPZDataFrameDataset` is built with shape `(N, H, W, C)` from
  the first probed snapshot (here `1023 × 1023`, 10 features, 6 targets).
- The train loader uses `OnePatchPerFileBatchSampler` with
  `oversample = subsample_rate` (so `subsample_rate: 800` means 800
  random crops per snapshot per epoch).
- Each worker holds one decoded file at a time, applies
  `RandomCrop((64, 64))` per access, and emits a `(C, 64, 64)` patch.

> **`batch_size` constraint for image mode.** The
> `OnePatchPerFileBatchSampler` will refuse `batch_size > num_files`
> at construction time with a clear error message — there literally
> aren't enough distinct files to fill a batch with one patch each.
> For the included smoke split (4 train files) we use `batch_size: 4`.
> For real production splits (hundreds of files) you can use any
> `batch_size ≤ num_train_files` you like, typically 32–256.

### 9.2 MLP (pixel mode)

[configs/testing/testing_small_npz_lazy_mlp.yaml](../configs/testing/testing_small_npz_lazy_mlp.yaml)

```yaml
data:
  flatten: true
  batch_size: 4096
  loading_mode: lazy_npz
  chunk_window: 4
  sample_cache_size: 4   # match chunk_window
  persistent_workers: true
```

What happens at runtime:

- `LazyNPZDataFrameDataset` exposes `N * H * W` pixel-level samples.
- `FileChunkedSampler` interleaves four files at a time, so each
  4096-pixel batch is composed of pixels from four different snapshots.
- The four-file LRU cache stays warm for an entire round-robin pass
  before any eviction happens.

> Note: `subsample_rate` is meaningful here only when ≠ 1.0 to *warn*
> you that it has no effect (one pixel is already one sample). To do
> "more epochs" simply train for more `max_epochs`.

### 9.3 Reproducing the verified smoke run

Both smoke configs were run end-to-end with `fast_dev_run=true` and
verified to produce sensible loss values. To reproduce on Hortense:

```bash
cd /dodrio/scratch/projects/2026_018/george/closure
source activate_hpc.sh

# FCNN / CNN (image mode) — completes in ~15 s
closure-train fit --config configs/testing/testing_small_npz_lazy.yaml \
  --trainer.fast_dev_run=true

# MLP (pixel mode) — completes in ~7 s
closure-train fit --config configs/testing/testing_small_npz_lazy_mlp.yaml \
  --trainer.fast_dev_run=true
```

Expected tail of the FCNN log:

```text
Epoch 0: 100%|████| 1/1 [00:11<00:00, 0.09it/s, val_loss=0.315, train_loss=2.320]
Trainer.fit finished at epoch=1 (max_epochs=1). Elapsed: 00:00:14.97 (14.97s)
```

Expected tail of the MLP log:

```text
Epoch 0: 100%|████| 1/1 [00:06<00:00, 0.15it/s, val_loss=0.253, train_loss=1.010]
Trainer.fit finished at epoch=1 (max_epochs=1). Elapsed: 00:00:07.47 (7.47s)
```

`fast_dev_run=true` runs exactly one train + val batch and exits — the
fastest way to verify a config is wired correctly before launching a
real job.

### 9.4 Inspect what Lightning will actually run

Before launching a long job, dump the merged config:

```bash
closure-train fit --config configs/testing/testing_small_npz_lazy.yaml --print_config | less
```

This shows every default value Lightning will apply, including the
auto-resolved `persistent_workers` and `prefetch_factor`.

### 9.5 More recipes — common scenarios

The eight recipes below cover the situations you are most likely to
hit in practice. Each one shows only the `data:` block (the model
block stays whatever it was before).

#### 9.5.1 Convert a working eager FCNN config to lazy

The smallest possible diff:

```diff
 data:
   ...
+  loading_mode: lazy_npz
+  sample_cache_size: 1
+  persistent_workers: true
```

That is it. The dataset will pick `OnePatchPerFileBatchSampler` because
`flatten: false`, and your `subsample_rate` will be reinterpreted as
the sampler's `oversample`.

#### 9.5.2 Convert a working eager MLP config to lazy

```diff
 data:
   ...
+  loading_mode: lazy_npz
+  chunk_window: 4
+  sample_cache_size: 4
+  persistent_workers: true
```

If you were relying on `subsample_rate < 1.0` for undersampling in
eager mode, you will see a warning at startup and the field will be
ignored. Reduce `max_epochs` instead.

#### 9.5.3 Maximum cross-batch decorrelation (MLP)

Use as much RAM per worker as you can:

```yaml
data:
  loading_mode: lazy_npz
  flatten: true
  chunk_window: 16
  sample_cache_size: 16
  num_workers: 8
  persistent_workers: true
  prefetch_factor: 4
```

This holds 16 decoded files hot per worker (≈ 16 × file size of RAM
per worker × 8 workers). Inside any round-robin pass, 16 distinct
snapshots are interleaved into every batch. Use only if you have the
RAM headroom.

#### 9.5.4 Minimum RAM per worker (MLP)

Trade decorrelation for memory:

```yaml
data:
  loading_mode: lazy_npz
  flatten: true
  chunk_window: 1
  sample_cache_size: 1
  num_workers: 4
```

`chunk_window: 1` means each worker drains a single file completely
before opening the next — best I/O efficiency, weakest per-batch
decorrelation. Adequate when your per-file pixel count is enormous and
you do many epochs.

#### 9.5.5 FCNN with heavy patch oversampling

```yaml
data:
  loading_mode: lazy_npz
  flatten: false
  patch_dim: [128, 128]
  batch_size: 64
  num_workers: 12
  sample_cache_size: 1
  subsample_rate: 1000      # → OnePatchPerFile.oversample = 1000
  persistent_workers: true
```

1000 random `128 × 128` crops per snapshot per epoch — same per-file
I/O as `subsample_rate: 1`, because the cache means we decode each
file only once and then crop 1000 times. This is the killer use case
for `OnePatchPerFileBatchSampler`.

#### 9.5.6 Validation/test only (no training)

The validation and test loaders are *not* lazy-mode-specific in any
interesting way: they use standard PyTorch DataLoaders over the lazy
dataset, with `shuffle=False`. You set them up the same way:

```yaml
data:
  loading_mode: lazy_npz
  val_samples_file: ./splits/my_val.csv
  test_samples_file: ./splits/my_test.csv
  sample_cache_size: 1
  num_workers: 8
  batch_size: 32
```

Then:

```bash
closure-train validate --config my_config.yaml --ckpt_path path/to/checkpoint.ckpt
closure-train test     --config my_config.yaml --ckpt_path path/to/checkpoint.ckpt
```

Both reuse the same normalization pickles written during fit.

#### 9.5.7 Multi-GPU (DDP) lazy training

No special YAML needed — just point Lightning at multiple devices:

```yaml
trainer:
  accelerator: gpu
  devices: 4
  strategy: ddp
data:
  loading_mode: lazy_npz
  num_workers: 8           # workers PER GPU process
  sample_cache_size: 1
  persistent_workers: true
```

Each of the 4 GPU processes builds its own `LazyNPZDataFrameDataset`
and spawns its own 8 DataLoader workers (32 workers total). On the
first run, rank 0 computes normalization stats while ranks 1–3 wait
at the barrier; subsequent runs skip recomputation. See §8.

#### 9.5.8 Resume from checkpoint without recomputing stats

The normalization pickles live under `norm_folder`. If you point a new
run at the same `norm_folder`, the lazy loader detects the existing
`X.pkl` / `y.pkl` and skips the streaming pass entirely. To force a
recompute, delete the pickles:

```bash
rm /path/to/norm_folder/X.pkl /path/to/norm_folder/y.pkl
closure-train fit --config my_config.yaml --ckpt_path last.ckpt
```

### 9.6 Hands-on lab: use the dataset directly from Python

You don't have to go through Lightning to play with the lazy loader.
Here is a self-contained snippet you can run after `source
activate_hpc.sh`:

```python
from closure.datasets import (
    LazyNPZDataFrameDataset,
    OnePatchPerFileBatchSampler,
    FileChunkedSampler,
)
from torch.utils.data import DataLoader

# Build the dataset (image mode, 1 sample = 1 snapshot).
ds = LazyNPZDataFrameDataset(
    data_folder="iPiC3D-nathan",
    norm_folder="./scratch_norm",
    samples_file="./splits/iPiC3D-nathan5-12/small/train_npz.csv",
    datalabel="train",
    flatten=False,
    scaler_features=True,
    scaler_targets=True,
    read_features_targets_kwargs={
        "fields_to_read": {"B": True, "E": True},
        "request_features": ["Bx", "By", "Bz"],
        "request_targets":  ["Ex", "Ey", "Ez"],
    },
    sample_cache_size=1,
)

print(f"num_files={ds.num_files}  shape={ds.features_shape}  "
      f"fast_path={ds._npz_fast_path}")

# Pull one sample directly.
features, targets = ds[0]
print(features.shape, targets.shape)   # (3, H, W), (3, H, W)

# Wrap with the image-mode sampler.
sampler = OnePatchPerFileBatchSampler(
    num_files=ds.num_files, batch_size=2, oversample=3, seed=0,
)
loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
for batch_idx, (xb, yb) in enumerate(loader):
    print(batch_idx, xb.shape, yb.shape)
    if batch_idx >= 2:
        break
```

For the MLP path swap to `flatten=True` and use `FileChunkedSampler`:

```python
ds.flatten = False  # reconstruct as flatten=True instead — see ctor docs
# ... build a flatten=True dataset, then:
sampler = FileChunkedSampler(
    num_files=ds.num_files,
    pixels_per_file=ds._pixels_per_file,
    window=4,
    seed=0,
)
loader = DataLoader(ds, batch_size=128, sampler=sampler, num_workers=0)
```

### 9.7 Worked example: estimating epoch length

Suppose you have:

- `num_train_files = 500`
- `H = W = 256` (so `pixels_per_file = 65 536`)
- `batch_size = 4096`
- `chunk_window = 4`

For the **MLP** (`flatten: true`):

```
samples_per_epoch  = 500 * 65_536          = 32_768_000
batches_per_epoch  = 32_768_000 / 4_096    = 8_000
files_decoded/epoch = 500   (each file is read once per epoch)
```

For the **FCNN** (`flatten: false`, `subsample_rate: 800`,
`batch_size: 32`):

```
oversample         = 800
batches_per_epoch  = (500 / 32) * 800      = 12_500
samples_per_epoch  = 12_500 * 32           = 400_000 (patches)
files_decoded/epoch = 500   (each file is decoded once and cropped 800x)
```

The key insight in both cases is that the **number of file decodes per
epoch equals `num_train_files`** — independent of `batch_size` and
`oversample`. The cache is what makes that possible.

### 9.8 Worked example: estimating worker RAM

A decoded file occupies (after prescale + normalize):

```
RAM_per_file ≈ (C_features + C_targets) * H * W * sizeof(float32)
```

For the 10F/6T iPiC3D production setup at `H = W = 1023`:

```
RAM_per_file ≈ 16 * 1023 * 1023 * 4 bytes ≈ 67 MB
```

Per-worker RAM budget for caching:

```
RAM_per_worker ≈ sample_cache_size * RAM_per_file
```

So `sample_cache_size: 4` ⇒ ≈ 270 MB per worker just for the cache.
Multiply by `num_workers` for total RAM committed to caches across
the DataLoader. Stay well under your slurm allocation.

---

## 10. Performance tuning checklist

In order, these are the dials that matter most:

1. **`num_workers`** — the single biggest lever. Start at the number of
   physical cores you have per GPU (e.g. 12 on Hortense `gpu_rome`),
   then scale up only if you see GPU utilization < 80%.
2. **`persistent_workers: true`** — without this, every new epoch
   discards worker state. Always set it true for lazy mode.
3. **`sample_cache_size`** — set per the rules above (1 for image
   mode, = `chunk_window` for MLP). Increasing further only helps if
   your worker count is so low that the same worker keeps revisiting
   different files within a few batches.
4. **`chunk_window`** (MLP only) — larger means better cross-file mixing
   but more RAM per worker. 2–4 is the sweet spot.
5. **`prefetch_factor`** — bump from 2 to 4 if your storage has high
   latency and your GPU is starving (visible as GPU utilization
   spikes followed by idle dips).
6. **`use_readonly: true`** — *Hortense-specific*. Routes
   `data_folder` through `/readonly` to avoid aggressive Lustre
   page-cache eviction. Zero downside on systems where the mount
   exists.
7. **Storage layout** — many small `.npz` files (one per snapshot) is
   the supported pattern. Avoid a single huge `.h5` file: lazy mode
   exists to load *files*, not to slice into a giant volume.
8. **Batch size** — for MLP, very large batch sizes (4096+) amortize
   the Python-side per-sample overhead. For CNN patches, optimize for
   GPU memory.

What to watch in `closure.log`:

- The "Data loading (train+val) took Xs" line is much smaller in lazy
  mode — it now reflects only the stats-streaming time.
- `Lazy NPZ loading enabled (sample_cache_size=…, chunk_window=…)`
  confirms the right loader was selected.
- `Lazy train loader: OnePatchPerFileBatchSampler` or
  `FileChunkedSampler` confirms the right sampler was selected.
- `npz fast path enabled for /path/to/data` confirms the
  single-open optimization is active.

---

## 11. Troubleshooting & FAQ

### "My validation/test results changed after switching to lazy mode."

They should not have changed. The lazy and eager paths are tested for
bit-equal normalization stats on the same training files (see
[tests/test_lazy_npz_disk.py](../tests/test_lazy_npz_disk.py)).
If you see drift, check:

1. Whether `norm_folder` is being reused across runs (it should be —
   stats are written once and reloaded). A fresh `norm_folder` will
   recompute stats; tiny float rounding differences (~1e-7) are
   expected.
2. Whether you changed `prescaler_*` or the channel list between runs.

### "Training is *slower* in lazy mode."

Make sure:

- `persistent_workers: true` (default-on for lazy with
  `num_workers > 0`, but worth confirming).
- `num_workers` is high enough — single-worker lazy loading is
  inherently I/O-bound.
- The fast path enabled (look for the INFO log). If not, check
  whether you accidentally set `alfven_units: true` or a `filter_*`
  callback.
- The cache size matches the sampler (1 for image, ≥ `chunk_window`
  for MLP).
- Your dataset actually does not fit in RAM. If it does, eager is
  faster and that's fine.

### "OOM during normalization stats computation."

Stats are computed by **streaming** one file at a time (`bypassing the
cache`), so the peak memory should be one decoded file plus
running per-channel float64 sums. If you OOM here, your individual
files are larger than your worker's RAM budget — that's a data layout
problem, not a lazy-loader bug.

### "I see `npz fast path disabled` in the log."

That's informational, not an error. The dataset will still produce
correct results via `rp.read_features_targets`. The two common reasons
are `alfven_units: true` (intentional) or a missing key in the npz
(a real bug in your `request_features` / `request_targets`).

### "Can I mix `.npz` and `.h5` files in one split?"

Yes. The fast path is per-file (the `.npz` extension check does *not*
permanently disable it), so `.h5` files quietly use the slow path
while `.npz` files use the fast path. Mixing formats is unusual but
supported.

### "Will `subsample_rate < 1.0` (undersampling) work?"

- `flatten: false` + lazy: Yes — it would, but it is not the natural
  expression of "draw a random subset of patches". Use the
  `oversample` parameter via `subsample_rate >= 1.0` instead, and
  control epoch length via `max_epochs`.
- `flatten: true` + lazy: No-op. A warning is logged. Use a smaller
  number of `max_epochs` if you want less training compute.

### "I changed my YAML but the run still uses eager mode."

`loading_mode` defaults to `"eager"`. Double-check spelling and
indentation under `data:`. `closure-train fit --print_config` will show
you the resolved value.

### "Do I need to delete `X.pkl` / `y.pkl` after changing channels?"

Yes. The normalization files are not invalidated automatically. If you
change `request_features`, `prescaler_*`, or the training split, delete
the old `norm_folder` (or point at a fresh one) so the lazy loader
recomputes stats.

### "How does the cache interact with multi-worker DataLoaders?"

Each worker process holds its own `_FileLRUCache`. The samplers send
related indices to the same worker most of the time (PyTorch's default
round-robin work assignment), but not always. The cache size guidance
already assumes a small over-budget for cross-worker draws.

---

## 12. Testing the lazy stack

There are four dedicated test files. All run on the HPC env in a few
seconds.

```bash
cd /dodrio/scratch/projects/2026_018/george/closure
source activate_hpc.sh
python -m pytest tests/ -q
```

| File | What it covers |
|---|---|
| [tests/test_lazy_npz_disk.py](../tests/test_lazy_npz_disk.py) | End-to-end parity between eager and lazy on real `.npz` files: shapes, normalization stats, per-pixel values. |
| [tests/test_lazy_npz_samplers.py](../tests/test_lazy_npz_samplers.py) | Unit tests for `OnePatchPerFileBatchSampler` and `FileChunkedSampler`: distinct files per batch, full coverage, reproducibility under `set_epoch`. |
| [tests/test_npz_fast_path.py](../tests/test_npz_fast_path.py) | Detection of the fast path (enabled/disabled in each of the documented conditions) and I/O parity with the slow path. |
| [tests/test_norm_ddp.py](../tests/test_norm_ddp.py) | The DDP barrier in `_prepare_normalization_params`: rank-0 writes + barriers, non-zero ranks barrier + load, single-process path unchanged. |
| [tests/test_cli_fit_lazy.py](../tests/test_cli_fit_lazy.py) | `closure-train fit` smoke runs through the Lightning CLI in lazy mode for both `flatten=true` (MLP) and `flatten=false` (FCNN). |

A single fast smoke during development:

```bash
python -m pytest tests/test_npz_fast_path.py tests/test_lazy_npz_samplers.py -q
```

---

## 13. Reading the source

Pointers into the codebase, ordered by relevance:

- [closure/datasets.py](../closure/datasets.py)
  - `class LazyNPZDataFrameDataset` — the lazy dataset.
  - `_load_file_chw`, `_get_file_arrays` — per-access decode + cache.
  - `_maybe_fast_load_npz`, `_stack_npz_arrays` — single-open fast path.
  - `_prepare_normalization_params` — DDP-safe stats handling.
  - `class _FileLRUCache` — the cache itself.
  - `class OnePatchPerFileBatchSampler`, `class FileChunkedSampler` — samplers.

- [closure/datamodule.py](../closure/datamodule.py)
  - `ClosureDataModule.__init__` — the YAML knobs.
  - `setup` — dataset class selection.
  - `_make_lazy_train_loader` — sampler selection.
  - `_loader_extra_kwargs` — `persistent_workers` / `prefetch_factor` defaults.
  - `on_train_epoch_start` — calls `sampler.set_epoch(...)` every epoch.

- [closure/read_pic.py](../closure/read_pic.py)
  - `read_features_targets`, `read_fieldname` — the canonical reader
    whose semantics (off-by-one slicing, ij-transpose) the fast path
    has to mirror.

---

## Appendix A. Glossary of YAML keys (lazy-relevant only)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `loading_mode` | str | `"eager"` | `"eager"` or `"lazy_npz"`. |
| `sample_cache_size` | int | `1` | Files held in per-worker LRU cache. |
| `chunk_window` | int | `1` | Files interleaved by `FileChunkedSampler` (MLP only). |
| `persistent_workers` | bool/null | `null` | Keep worker processes alive across epochs. Auto-on for lazy + workers. |
| `prefetch_factor` | int/null | `null` | DataLoader prefetch. Defers to PyTorch when `null`. |
| `flatten` | bool | `true` | Pixel mode (`true`) or image mode (`false`). Decides which sampler is used in lazy mode. |
| `patch_dim` | `[H, W]`/null | `null` | Random-crop size; image mode only. |
| `subsample_rate` | float | `1.0` | In lazy + image mode this becomes `oversample`. In lazy + pixel mode it's ignored with a warning. |
| `use_readonly` | bool | `false` | Hortense-specific Lustre `/readonly` redirection. Orthogonal to lazy/eager. |

---

## Appendix B. Minimal lazy YAML to copy-paste

```yaml
seed_everything: 42

model:
  network:
    class_path: closure.models.FCNN
    init_args:
      channels: [3, 16, 3]
      kernels: [3, 3]
      activations: [ReLU, null]
  criterion: MSELoss
  optimizer: Adam
  lr: 0.001

data:
  data_folder: my_dataset
  train_samples_file: ./splits/train.csv
  val_samples_file:   ./splits/val.csv
  test_samples_file:  ./splits/test.csv
  batch_size: 32
  num_workers: 8
  flatten: false
  patch_dim: [64, 64]
  scaler_features: true
  scaler_targets: true
  features_dtype: float32
  targets_dtype: float32

  loading_mode: lazy_npz
  sample_cache_size: 1
  persistent_workers: true
  prefetch_factor: 2

  read_features_targets_kwargs:
    request_features: [Bx, By, Bz]
    request_targets:  [Ex, Ey, Ez]

trainer:
  max_epochs: 50
  accelerator: auto
  devices: auto
  default_root_dir: ./models/my_run
```

That is enough to switch a working eager config to lazy mode without
touching anything else.
