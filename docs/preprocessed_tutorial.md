# Preprocessed-chunk loading mode

This tutorial covers the `loading_mode: preprocessed` path end-to-end:
what files it creates, how long those files live, how to configure it,
and how to verify correctness at increasing scale.

---

## 1. Why preprocessed mode exists

With `lazy_npz`, every time the DataLoader fetches a sample it:

1. Opens the `.npz` file (16 opens for 10F+6T fields via `rp.read_features_targets`)
2. Applies Alfvén-unit rescaling
3. Applies per-channel prescaling (log, arcsinh, …)
4. Applies mean/std normalization

That pipeline costs ~0.5 s per 1023×1023 file.  With a batch of 4
and `num_workers=12`, the GPU still starves.

`preprocessed` does all of the above **once**, writes compact float32
tensors to the node-local SSD (`$TMPDIR`, 480 GB on Hortense GPU nodes),
and then training just reads pre-normalized tensors — no numpy, no file
format parsing, no normalization per batch.

---

## 2. File lifecycle

Two kinds of output are created; they have different lifetimes.

### 2.1 Norm stats — persistent

`X.pkl` and `y.pkl` are written to `norm_folder` (the Lightning
log-dir by default, e.g. `models/Lightning/.../lightning_logs/version_0/`).
These live on `$VSC_SCRATCH` and survive between jobs.

They encode per-channel `(mean, std)` over the **training** split.
Val and test splits reuse the training stats automatically — if the
pkl files are already present, neither pass in `_stream_norm_stats`
re-computes them.

### 2.2 Chunk tensors — ephemeral

`chunk_NNNN.pt` files land in:

```
ssd_cache_dir/{datalabel}_{fingerprint}/chunk_NNNN.pt
```

where `fingerprint` is a 10-char MD5 of
`(samples_file, request_features, request_targets, prescaler_features, prescaler_targets, alfven_units)`.

On Hortense, `ssd_cache_dir` defaults to `$TMPDIR/closure_preprocessed`.
`$TMPDIR` is a per-job directory on the local NVMe SSD set by Slurm and
**wiped automatically when the job ends**.  There is no manual cleanup
needed; there is also no way to reuse chunks across jobs.

If you set `ssd_cache_dir` to a path on `$VSC_SCRATCH` instead, chunks
will persist but you are responsible for cleaning them up (each chunk
file for the production 1023×1023 / 16-channel config is ~300 MB).

> **Key detail**: if `metadata.json` exists in the fingerprint directory
> the entire preprocessing step is skipped — neither the norm stats nor the
> chunk files are rewritten.  Delete the directory (or let Slurm wipe
> `$TMPDIR`) to force a fresh run.

### 2.3 File-to-slot shuffling

During preprocessing, files are assigned to chunk slots in a **shuffled**
order (fixed seed 0) rather than CSV row order.  Slot `i` stores CSV file
`dataset._file_perm[i]`.

This matters when the CSV is time-ordered (consecutive rows are from the
same simulation epoch): without the shuffle, chunk 0 contains files from
simulation time t₀..t_fpb, chunk 1 from t_fpb..t_2fpb, etc., and the
model trains on one temporal block before moving to the next.  With the
shuffle, each chunk is a random sample of the full time range.

The sampler (`ChunkOrderedSampler`) re-shuffles the chunk visit order and
the within-chunk file order every epoch, so training diversity is further
improved on top of the static chunk layout.

The shuffle seed (0) is fixed so the chunk layout is deterministic: the
same dataset always produces the same `file_perm`, making the cache valid
for repeated runs within the same job.

### 2.4 What the fingerprint covers

Changing **any** of these parameters auto-creates a new cache directory:

| Parameter | Covered by fingerprint |
|---|---|
| `train_samples_file` | yes |
| `request_features` | yes |
| `request_targets` | yes |
| `prescaler_features` / `prescaler_targets` | yes |
| `alfven_units` | yes |
| `scaler_features` / `scaler_targets` | no |
| `num_workers`, `batch_size` | no |
| `patch_dim`, `subsample_rate` | no |

Changing only `scaler_features=False` (normalization off) while keeping
everything else the same would silently reuse the previous chunks.
That is intentional: the raw data layout is fixed by the fields and
prescalers, not by whether normalization is applied afterward.  If you
need unnormalized chunks, use a different `ssd_cache_dir`.

---

## 3. Quick start

Minimum YAML changes over a lazy config:

```yaml
data:
  loading_mode: preprocessed
  # ssd_cache_dir defaults to $TMPDIR/closure_preprocessed on Hortense
  chunk_cache_size: 1        # chunk tensors kept per DataLoader worker
```

Everything else (`data_folder`, `train_samples_file`, `batch_size`,
`prescaler_targets`, `alfven_units`, `read_features_targets_kwargs`, …)
is unchanged.

The full smoke config is at
`configs/testing/testing_small_npz_preprocessed.yaml`.

### 3.1 Running on Hortense

```bash
source activate_hpc.sh   # sets up modules + venv

# fast sanity check (1 batch each, no checkpointing)
closure-train fit \
  --config configs/testing/testing_small_npz_preprocessed.yaml \
  --trainer.fast_dev_run=true

# override ssd_cache_dir explicitly if $TMPDIR is not set
closure-train fit \
  --config configs/testing/testing_small_npz_preprocessed.yaml \
  --data.ssd_cache_dir=${TMPDIR:-/tmp}/closure_preprocessed \
  --trainer.fast_dev_run=true
```

The log will show:
```
Preprocessing complete: 4 files → 1 chunks in /tmp/...
PreprocessedChunkDataset | split=train | files=4 | chunks=1 | flatten=False | samples=4
```

On the **second run** with the same `ssd_cache_dir`, the chunk lines
are absent — the cache is fully reused.

### 3.2 CLI performance flags

When running `closure-train fit` directly from the command line (rather than
via a `.sh` script), add these two flags to avoid two common bottlenecks:

```bash
closure-train fit \
  --config configs/iPiC3D-nathan5-12/Runs_7-9-10-11-12_long1000_cosine_swa_step100.yaml \
  --data.loading_mode=preprocessed \
  --data.ssd_cache_dir="${TMPDIR:-/tmp}/closure_benchmark_P" \
  --data.preprocess_chunk_size_gb=2 \
  --data.preprocess_num_workers=8 \
  --data.num_workers=12 \
  --trainer.num_sanity_val_steps=0 \
  ...
```

- `--data.preprocess_num_workers=8` — parallelises the one-time preprocessing
  pass over Lustre using 8 threads.  The default is 1 (sequential); 8 workers
  give roughly 4–5× faster preprocessing with negligible extra RAM
  (≈ 320 MB).  See section 5.6 for details.

- `--trainer.num_sanity_val_steps=0` — skips Lightning's pre-training sanity
  validation check.  With `num_workers=12` and `persistent_workers=True`
  (the default in preprocessed mode), Lightning would otherwise spawn 12
  worker processes, run 2 val batches, and tear them down before training
  starts — adding ~10 s of startup overhead on Lustre.  Skip it if your
  data pipeline is already validated.

> **Note**: `--data.read_features_targets_kwargs.num_workers=4` has no
> effect in preprocessed mode — the preprocessing uses its own thread pool
> controlled by `preprocess_num_workers`.  Remove it from any CLI invocations.

---

## 4. All knobs

```yaml
data:
  loading_mode: preprocessed

  # Path to local NVMe SSD scratch.
  # On Hortense: omit to auto-use $TMPDIR/closure_preprocessed.
  # On local workstation: set to /tmp/closure_preprocessed or similar.
  ssd_cache_dir: null

  # Chunk tensors kept resident per DataLoader worker.
  # 1 is sufficient for FCNN/CNN (ChunkOrderedSampler drains one chunk
  # before moving to the next, so eviction never happens mid-chunk).
  # Increase only if you observe chunk-boundary stalls in profiling.
  chunk_cache_size: 1

  # RAM budget for one chunk during the preprocessing pass.
  # null → auto-estimated as available_RAM * 0.4 / num_gpus.
  # Set explicitly (in GiB) if the auto estimate is wrong:
  #   preprocess_chunk_size_gb: 20.0
  preprocess_chunk_size_gb: null

  # Number of threads used to read files in parallel during the
  # one-time preprocessing pass (does NOT affect training).
  # Default 1 (sequential).  4–8 typically saturates Lustre bandwidth.
  preprocess_num_workers: 1
```

### 4.1 Chunk size calculation

When `preprocess_chunk_size_gb` is null, the code reads
`/proc/meminfo` at preprocessing time:

```
budget = MemAvailable * 0.4 / num_gpus
files_per_chunk = budget / ((C_f + C_t) * H * W * 4 bytes)
```

For the production setup (10F + 6T, 1023×1023, 1 GPU):
- `MemAvailable ≈ 200 GB` on a GPU node → budget ≈ 80 GB
- bytes_per_file ≈ 16 × 1023² × 4 ≈ 67 MB
- → ~1190 files/chunk (entire small split fits in one chunk)

The log line `Chunk size: N files/chunk (... → X MiB/chunk)` confirms
what was computed.

> **Warning**: `null` reads `/proc/meminfo` (node-wide RAM), not the
> Slurm cgroup limit.  On a 251 GB node with `--mem=64G`, the auto
> estimate produces a budget of ~100 GB — far exceeding your allocation.
> **Always set `preprocess_chunk_size_gb` explicitly in Slurm jobs.**
> See section 5.3 for a safe formula.

---

## 5. RAM mechanics and the two parameters

Understanding why these two parameters exist, what they control, and
how they interact is the key to tuning the preprocessed loader without
OOM surprises.

### 5.1 The two phases and their RAM profiles

Preprocessed mode has two temporally separate phases:

**Phase 1 — Preprocessing (one-time)**

Triggered at the start of the first job (or whenever the SSD cache
is cold).  Files are read from Lustre/NFS one by one (or in parallel
if `preprocess_num_workers > 1`), normalized, and written to `.pt`
chunk files on the local SSD.  RAM is dominated by:

- The raw file buffers (one file per thread in flight ≈
  `preprocess_num_workers × bytes_per_file`)
- Linux page cache from files already read (the kernel retains recently
  accessed data in free RAM, which inflates `cgroup_ram` but not real
  process pressure)
- One chunk worth of tensors held in the buffer before saving
  (`preprocess_chunk_size_gb` GiB)

**Phase 2 — Training (every epoch)**

Workers read pre-normalized `.pt` chunks from SSD.  RAM is dominated by:

- Per-worker chunk caches: `num_workers × chunk_cache_size × preprocess_chunk_size_gb`
- Model parameters + GPU activations (separate, lives on GPU)
- Residual page cache from phase 1 (still counted in `cgroup_ram`,
  but the kernel can evict it under pressure — it is not real
  application pressure)

The key insight: **phase 1 sets the peak cgroup allocation, phase 2
determines steady-state working-set RAM.**

### 5.2 cgroup_ram vs unique_ram

The two RAM metrics logged each epoch measure different things:

| Metric | What it counts | Useful for |
|--------|---------------|-----------|
| `cgroup_ram_gb` | All memory in the Slurm cgroup, including kernel page cache | Slurm OOM risk |
| `unique_ram_gb` (PSS) | Private + proportional share of shared pages — actual application pressure | True working-set comparison |

After preprocessing completes on a lightly loaded node, `cgroup_ram`
jumps to 50–60 GB even though `unique_ram` is only ~1 GB.  This is
normal: the kernel cached the 505 raw files it just read, but those
pages are evictable and will be reclaimed if something else needs RAM.

Example from a real run (505 files, 1023×1023, `preprocess_chunk_size_gb=0.5`):

```
# After preprocessing, before training:
cgroup_ram_gb=49.4   unique_ram_gb=0.67   cgroup_ram_peak_gb=59.5

# During training epoch 0 (num_workers=4, chunk_cache_size=1):
cgroup_ram_gb=54.5   unique_ram_gb=3.5    cgroup_ram_peak_gb=59.5
```

`cgroup_ram_peak` (59.5 GB) was set during preprocessing and never
exceeded during training — even with 2.5× more data (43 chunks instead
of 18), the peak stayed flat.  Preprocessing peak RAM is independent
of total dataset size; it depends only on `preprocess_chunk_size_gb`.

### 5.3 Controlling peak RAM with `preprocess_chunk_size_gb`

During preprocessing, the highest-pressure moment is when one chunk's
worth of files has been read into the staging buffer but not yet saved.
The budget formula is:

```
peak_ram ≈ preprocess_chunk_size_gb   (staging buffer)
          + preprocess_num_workers × bytes_per_file   (in-flight reads)
          + page_cache   (kernel-retained, evictable but counted in cgroup)
          + ~3 GB   (Python interpreter, model, misc)
```

Page cache is hard to predict — it scales with how much free RAM the
node has.  On a 251 GB node with only one job running, page cache for
505 files × ~40 MB each ≈ 20 GB extra.  On a busy node the kernel
reclaims those pages and the cache footprint is much smaller.

**Safe formula for Slurm jobs (no page cache surprise):**

```
preprocess_chunk_size_gb ≤ mem_gb - gpu_gb - page_cache_estimate - 3
```

With `mem_gb=64, gpu_gb=6` (A100 context), `page_cache_estimate=15 GB`
(conservative for a 64 GB allocation on a 251 GB node):

```
preprocess_chunk_size_gb ≤ 64 - 6 - 15 - 3 = 40 GB   (upper bound)
```

In practice, you also want headroom:

```
preprocess_chunk_size_gb = floor((mem_gb - gpu_gb - 20) / 2)
                         = floor((64 - 6 - 20) / 2) = 19 GB
```

If you prefer a more conservative setting that gives very low training RAM:

```
preprocess_chunk_size_gb = 0.5   # 43 chunks for 505 files at 1023×1023
```

This keeps `unique_ram` during training at ≈ `num_workers × 0.5 GB = 6 GB`
for 12 workers, at the cost of more chunk switches per epoch (see 5.4).

**Observed safe settings for the production 1023×1023 dataset:**

| `--mem` | `preprocess_chunk_size_gb` | Peak cgroup | Outcome |
|---------|--------------------------|-------------|---------|
| 60 GB   | 0.5                      | 59.5 GB     | just fits (40 MB headroom) |
| 64 GB   | 2.0                      | ~62 GB      | safe |
| 80 GB   | 4.0                      | ~75 GB      | safe, fewer chunks |
| 128 GB  | 20.0                     | ~60 GB*     | ~25 chunks total |

*Page cache is bounded by `mem_gb` so the cgroup peak saturates.

### 5.4 Training RAM: the `num_workers × chunk_cache_size` product

During training, each DataLoader worker holds `chunk_cache_size` chunks
in its private in-process memory.  Because workers are forked processes,
these allocations are NOT shared — they are counted in full per worker.

```
training_unique_ram ≈ num_workers × chunk_cache_size × preprocess_chunk_size_gb
                     + ~2 GB   (main process + overhead)
```

Examples at 1023×1023 with `preprocess_chunk_size_gb=0.5`:

| `num_workers` | `chunk_cache_size` | Training unique_ram |
|---------------|-------------------|---------------------|
| 4             | 1                 | 0.5×4 + 2 = 4 GB   |
| 12            | 1                 | 0.5×12 + 2 = 8 GB  |
| 4             | 2                 | 1.0×4 + 2 = 6 GB   |
| 12            | 2                 | 1.0×12 + 2 = 14 GB |

`chunk_cache_size=1` is enough for FCNN/CNN because `ChunkOrderedSampler`
drains one chunk completely before moving to the next — the active chunk
is never evicted mid-chunk.  Increase it only if you profile chunk-switch
stalls (rare with fast NVMe).

### 5.5 Chunk count, epoch time, and the `preprocess_chunk_size_gb` tradeoff

Smaller `preprocess_chunk_size_gb` → more chunks → more chunk switches
per epoch → longer epoch time.  This is the fundamental tradeoff:

```
num_chunks = ceil(num_files / files_per_chunk)
           = ceil(num_files / (preprocess_chunk_size_gb × 1024³ / bytes_per_file))
```

For 505 files at 1023×1023 (bytes_per_file ≈ 40 MB for 4F+6T):

| `preprocess_chunk_size_gb` | files/chunk | num_chunks | Epoch time* | Training unique_ram† |
|---------------------------|-------------|------------|-------------|----------------------|
| 0.5                       | ~12         | 43         | ~363 s      | ~6 GB (12 workers)  |
| 2.0                       | ~50         | 11         | ~180 s      | ~26 GB              |
| 20.0                      | ~500        | 2          | ~140 s      | ~242 GB             |

\* Measured on A100 at `batch_size=32`, `num_workers=4`. More workers reduce
this significantly.\
† `num_workers=12, chunk_cache_size=1`.

The sweet spot for low RAM and fast training depends on your Slurm
allocation.  For `--mem=64G`: `preprocess_chunk_size_gb=2–4` gives 11–22
chunks, keeps training unique_ram ≤ 50 GB, and reduces epoch time by 2×
compared to 0.5.

### 5.6 How `preprocess_num_workers` affects preprocessing time and RAM

`preprocess_num_workers` only matters during the one-time preprocessing
phase — it has zero effect on training.  It controls how many files are
read from Lustre in parallel using a thread pool.

**Time**: Each 1023×1023 file takes ~1.2 s to read from Lustre.  Two
passes are made (norm stats + chunk building), so with `N` files and 1
worker:

```
preprocessing_time ≈ 2 × num_files × 1.2 s / preprocess_num_workers
```

| `preprocess_num_workers` | 505 files, estimated time |
|--------------------------|--------------------------|
| 1 (default)              | ~1210 s (20 min)          |
| 4                        | ~303 s (5 min)            |
| 8                        | ~152 s (2.5 min)          |

Measured with 1 worker: ~2912 s for norm stats + chunking combined
(includes Alfvén rescaling and prescaling overhead on top of raw I/O).
With `preprocess_num_workers=4` expect ~700–800 s.

> **Note**: Lustre (the Dodrio parallel filesystem) has a bandwidth limit
> per user per node.  Scaling beyond 8 threads rarely gives further
> speedup — the bottleneck shifts from latency to bandwidth.

**RAM**: Each in-flight thread holds one raw file in memory before saving
it to the chunk buffer.  The extra peak RAM during preprocessing is:

```
extra_peak ≈ preprocess_num_workers × bytes_per_file
```

For 1023×1023 with 4 features + 6 targets:
- `bytes_per_file ≈ 40 MB`
- 4 workers → extra 160 MB
- 8 workers → extra 320 MB

This is negligible compared to the chunk buffer (`preprocess_chunk_size_gb`)
and page cache, so increasing `preprocess_num_workers` from 1 to 8 does
not meaningfully raise peak RAM.

**Progress bar**: When `tqdm` is installed (it is on Hortense), you will
see two per-file progress bars during preprocessing:

```
norm stats (features): 100%|████████| 505/505 [03:12<00:00,  2.6file/s]
norm stats (targets):  100%|████████| 505/505 [03:08<00:00,  2.7file/s]
preprocess:            100%|████████| 505/505 [09:47<00:00,  0.9file/s]
```

The `preprocess` bar covers both reading and saving; it updates after
each file is added to its chunk buffer, so it advances in bursts of
`files_per_chunk`.

---

## 6. Choosing between eager, lazy_npz, and preprocessed

Use this table as a first-cut decision guide:

| | Eager | lazy_npz | Preprocessed |
|---|---|---|---|
| **Data fits in RAM?** | yes | no | no (or yes but want fast epochs) |
| **One-time cost** | file I/O × 1 | none | file I/O × 2 + SSD write |
| **Per-epoch cost** | ~0 | high (re-reads all files) | low (SSD reads) |
| **Peak RAM** | full dataset | `sample_cache_size × file_size × num_workers` | `preprocess_chunk_size_gb` |
| **SLURM `--mem` for 505 files, 1023²** | ~64 GB | ~8 GB | ~64 GB (phase 1), ~8 GB (phase 2) |
| **Epoch time (A100, batch=32)** | ~31 s | ~800 s | ~363 s (0.5 GB chunks) |
| **Good for** | ≤200 files or large RAM | any size, RAM-constrained | large datasets, fast epochs needed |

The preprocessed mode peak RAM is set by the preprocessing phase
(`preprocess_chunk_size_gb`), not the total dataset size.  Once the SSD
cache is warm, training RAM is `num_workers × chunk_cache_size × preprocess_chunk_size_gb`
— much lower than eager.

---

## 7. DDP behaviour

Preprocessing is rank-0-only, coordinated by a `torch.distributed.barrier()`:

```
rank 0: _preprocess_and_save() → writes chunks + X.pkl/y.pkl
        torch.distributed.barrier()    ← notifies other ranks
rank 1+:                                torch.distributed.barrier()
        load metadata.json
```

If the metadata already exists on entry (second job, or same job after
a cache hit), all ranks hit the barrier unconditionally so DDP
synchronisation is never broken.

The internal `LazyNPZDataFrameDataset` used inside `_preprocess_and_save`
is created with `scaler=False` in pass 1 (no barrier inside it) and loads
from the pkl written in pass 1 in pass 2 (again no barrier).  There is
no barrier mismatch between ranks.

---

## 8. Parity with lazy_npz

The preprocessing call chain is:

```
_preprocess_and_save
  → LazyNPZDataFrameDataset._get_file_arrays(file_idx, normalize=True)
      → _load_file_chw
          → rp.read_features_targets  (or fast-path single-open)
          → alfven_units rescaling
          → _apply_prescaling_to_sample
          → _normalize_sample
```

This is **exactly** the same code path as `LazyNPZDataFrameDataset.__getitem__`
at training time.  The norm stat algorithm (`_stream_norm_stats`) is also a
direct copy of `_compute_streaming_normalization_params` — same float64
accumulation, same `maximum(variance, 0)` clamp, same cast to `dtype_numpy`.

Parity is therefore a structural guarantee, not just an empirical
observation.  The regression test `test_parity_with_lazy_flatten_false`
guards the invariant on the tiny fixture (B/E fields, no prescalers, no
alfven_units).

---

## 9. Suggested larger tests

The tests below go beyond the unit suite.  Run them in order — each
one adds a new dimension of risk.

### 9.1 Throughput benchmark (5 min, interactive node)

Compare real training throughput between lazy and preprocessed:

```bash
# Lazy baseline — note wall-clock time and it/s in the progress bar
time closure-train fit \
  --config configs/testing/testing_small_npz_lazy.yaml \
  --trainer.limit_train_batches=20 \
  --trainer.max_epochs=1 \
  --trainer.enable_progress_bar=true

# Preprocessed — note preprocessing time + training throughput
# (on a second run the preprocessing is skipped and only training time matters)
time closure-train fit \
  --config configs/testing/testing_small_npz_preprocessed.yaml \
  --data.ssd_cache_dir=${TMPDIR:-/tmp}/closure_bench \
  --trainer.limit_train_batches=20 \
  --trainer.max_epochs=1 \
  --trainer.enable_progress_bar=true
```

`--trainer.limit_train_batches=20 --trainer.max_epochs=1` runs exactly 20 train
batches then one validation pass, so `val_loss` is always logged and callbacks
(`EarlyStopping`, `LearningRateMonitor`) behave correctly.  Using `--trainer.max_steps=20`
instead stops mid-epoch and causes `EarlyStopping` to fail because `val_loss` is
not yet available.

Expected: preprocessing takes ~20 s for 4 files; after that each
training step should be at least 5× faster than lazy.

### 9.2 Multi-worker stress test

Test that per-worker chunk caches don't corrupt data:

```bash
closure-train fit \
  --config configs/testing/testing_small_npz_preprocessed.yaml \
  --data.ssd_cache_dir=${TMPDIR:-/tmp}/closure_workers \
  --data.num_workers=4 \
  --data.chunk_cache_size=1 \
  --trainer.limit_train_batches=50 \
  --trainer.max_epochs=1

```

Watch for `RuntimeError` or loss divergence (NaN/inf) — either would
indicate a cache race between workers.  Loss should behave similarly
to the single-worker run.

### 9.3 Real-data parity check

Verify that the first batch of preprocessed data matches lazy_npz on
the actual 1023×1023 files with prescalers and alfven_units:

```python
# Run from a Python shell with the HPC venv activated.
import torch, numpy as np
from closure.datasets import PreprocessedChunkDataset, LazyNPZDataFrameDataset

RFT = dict(
    fields_to_read={"B": True, "E": True, "rho": True, "J": True, "P": True, "PI": True},
    request_features=["rho_e", "Bx", "By", "Bz", "Vx_e", "Vy_e", "Vz_e", "Ex", "Ey", "Ez"],
    request_targets=["Pxx_e", "Pyy_e", "Pzz_e", "Pxy_e", "Pxz_e", "Pyz_e"],
    choose_species=["e", None, "e", None],
    verbose=False,
)
COMMON = dict(
    data_folder="iPiC3D-nathan",            # resolved via paths.yaml
    norm_folder="/tmp/parity_norm",
    scaler_features=True,
    scaler_targets=True,
    prescaler_features=None,
    prescaler_targets=["log", "log", "log", "arcsinh", "arcsinh", "arcsinh"],
    alfven_units=True,
    read_features_targets_kwargs=RFT,
)
TRAIN_CSV = "splits/iPiC3D-nathan5-12/small/train_npz.csv"

pre = PreprocessedChunkDataset(
    **COMMON,
    samples_file=TRAIN_CSV,
    ssd_cache_dir="/tmp/parity_ssd",
    datalabel="train",
    flatten=False,
    chunk_cache_size=1,
)
lazy = LazyNPZDataFrameDataset(
    **COMMON,
    samples_file=TRAIN_CSV,
    datalabel="train",
    flatten=False,
    sample_cache_size=1,
)
for i in range(min(pre.num_files, 4)):
    pf, pt = pre[i]
    lf, lt = lazy[i]
    np.testing.assert_allclose(pf.numpy(), lf.numpy(), atol=1e-5,
                               err_msg=f"feature mismatch at file {i}")
    np.testing.assert_allclose(pt.numpy(), lt.numpy(), atol=1e-5,
                               err_msg=f"target mismatch at file {i}")
    print(f"file {i}: OK  |feat| = {pf.abs().mean():.4f}  |targ| = {pt.abs().mean():.4f}")
print("All files match.")
```

### 9.4 Multi-epoch training (full small split, Slurm job)

Run the smoke config for a meaningful number of epochs to observe
that val loss decreases and no stalls occur between epochs:

```bash
# In a batch script with --gres=gpu:1 --mem=60G
source activate_hpc.sh
closure-train fit \
  --config configs/testing/testing_small_npz_preprocessed.yaml \
  --trainer.max_epochs=5 \
  --trainer.enable_progress_bar=true
```

The preprocessing step runs only in epoch 0 setup; epochs 1–4 go
straight to chunk loads.  Look for the log line
`Preprocessed train loader: ChunkOrderedSampler FCNN (chunks=N, oversample=800) → M samples/epoch`
which confirms the sampler is active.

### 9.5 DDP barrier test (2 GPUs)

Test that rank-0-only preprocessing + barrier works without deadlock:

```bash
# In a batch script with --gres=gpu:2
closure-train fit \
  --config configs/testing/testing_small_npz_preprocessed.yaml \
  --trainer.devices=2 \
  --trainer.strategy=ddp \
  --trainer.fast_dev_run=true
```

Expected: rank 0 logs `Saved chunk 0 ... → /tmp/.../chunk_0000.pt`;
rank 1 sees no such line.  Both ranks proceed to training after the
barrier.  If there is a deadlock, the job hangs indefinitely at the
`Saved chunk` step.

---

## 10. Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError: ssd_cache_dir not set` | `$TMPDIR` unset (not a Slurm job) | Add `--data.ssd_cache_dir=/tmp/closure_preprocessed` |
| Preprocessing re-runs every epoch | shouldn't happen — metadata persists | Check if `ssd_cache_dir` is under `/tmp` and being wiped; use a persistent path instead |
| `metadata.json` found but chunk files missing | SSD filled up mid-preprocessing, or manual partial delete | Delete the fingerprint directory and rerun |
| `chunk_cache_size=1` causes slow steps | all chunks are being iterated with `num_workers > 1` | Increase `chunk_cache_size` to match how many distinct chunks workers touch simultaneously |
| `val_loss` from preprocessed differs from lazy | different random seed for crops, not a data bug | Use `seed_everything: 42` and same patch_dim to reproduce |
| OOM during preprocessing despite low `preprocess_chunk_size_gb` | `null` was used — code read node-wide `/proc/meminfo`, not your Slurm cgroup | Set `preprocess_chunk_size_gb` explicitly; see section 5.3 for formula |
| Changed `preprocess_chunk_size_gb` but chunks weren't re-built | Chunk size is not part of the fingerprint; old chunks reused | `rm -rf $TMPDIR/closure_preprocessed/train_<fingerprint>` then rerun |
| Preprocessing takes 30–50 min with default settings | `preprocess_num_workers=1` reads files sequentially | Set `--data.preprocess_num_workers=4` (or 8) to parallelize Lustre reads |
| `cgroup_ram` during training is 40–55 GB but `unique_ram` is only 3 GB | Page cache from preprocessing is retained in cgroup but evictable | Normal — `unique_ram` is the real metric; `cgroup_ram` will drop under memory pressure |

---

## 11. Glossary

**Alfvén units**
A physical rescaling applied to raw PIC simulation fields (densities,
velocities, EM fields) to convert from code units into dimensionless
Alfvén units using parameters (`B₀`, `nₑ`) parsed from the experiment's
`.inp` file.  Controlled by `alfven_units: true` in the YAML.  Applied
before prescaling and normalization.

**barrier** (DDP barrier)
`torch.distributed.barrier()` — a synchronisation point where every
process in a distributed job must arrive before any of them can continue.
Used here so that rank 0 finishes writing chunk files before ranks 1+ try
to read them.

**chunk**
A group of files stored together as a single PyTorch tensor file
(`chunk_NNNN.pt`).  Each chunk file holds a dict
`{"features": Tensor(N, C_f, H, W), "targets": Tensor(N, C_t, H, W)}`
where N is the number of files in that chunk.  Chunks exist because the
full dataset is too large to hold in RAM at once.

**chunk_cache_size**
The number of chunk tensors held in each DataLoader worker's in-memory
LRU cache.  Setting it to 1 is sufficient when `ChunkOrderedSampler`
drains one chunk completely before moving to the next, since the active
chunk is never evicted mid-chunk.

**ChunkOrderedSampler**
The custom PyTorch `Sampler` used with `PreprocessedChunkDataset` during
training.  Each epoch it (1) shuffles the order in which chunks are
visited, (2) shuffles indices within each chunk, and (3) yields all
indices of a chunk before moving to the next, so each worker's
`chunk_cache_size=1` cache is never evicted mid-chunk.

**DDP** (DistributedDataParallel)
PyTorch's multi-GPU training strategy.  Each GPU runs its own process
(rank), and gradient updates are synchronised across ranks after each
backward pass.  Rank 0 is the primary process; other ranks are workers.

**file_perm**
The permutation applied during preprocessing that assigns CSV files to
chunk slots in a shuffled order.  `dataset._file_perm[slot]` gives the
CSV row index of the file stored at `slot`.  Stored in `metadata.json`
so the mapping is recoverable after the dataset is reloaded.

**fingerprint**
A 10-character MD5 hash of the dataset configuration
`(samples_file, request_features, request_targets, prescaler_features,
prescaler_targets, alfven_units)`.  Used as part of the cache directory
name so that different experiment configurations never share chunk files
even when pointing at the same `ssd_cache_dir`.

**fingerprint directory**
The subdirectory inside `ssd_cache_dir` that holds chunk files for one
specific split and configuration, named `{datalabel}_{fingerprint}`.
Example: `$TMPDIR/closure_preprocessed/train_bdbf33d434/`.

**flatten**
Controls whether the dataset returns full 2D field maps or individual
pixels.  `flatten=False` (FCNN/CNN mode): `__getitem__` returns one
`(C, H, W)` tensor per file.  `flatten=True` (MLP mode): `__getitem__`
returns one `(C,)` vector per pixel, with `len(dataset) = num_files × H × W`.

**norm stats** (normalization statistics)
Per-channel mean and standard deviation computed over the training split
and saved as `X.pkl` (features) and `y.pkl` (targets) in `norm_folder`.
Applied as `(x − mean) / std` after prescaling.  Computed once on the
training split; val and test splits load and reuse the training stats.

**norm_folder**
The directory where `X.pkl` and `y.pkl` are written.  Defaults to the
Lightning log directory (`default_root_dir/lightning_logs/version_N/`),
which lives on persistent scratch storage.  Distinct from `ssd_cache_dir`.

**oversample**
The number of times each file is visited per training epoch.  In FCNN
mode, each visit draws a different random crop of the full field map
(`patch_dim × patch_dim`), so `oversample=800` gives 800 distinct views
of each 1023×1023 snapshot per epoch.  Passed to `ChunkOrderedSampler`
and controlled by `subsample_rate` in the YAML.

**prescaling**
A non-linear per-channel transformation applied to raw field values
*before* mean/std normalization.  Common choices: `log` (for
positive-definite quantities like pressure diagonal components) and
`arcsinh` (for signed quantities like off-diagonal pressure).  Controlled
by `prescaler_features` / `prescaler_targets` lists in the YAML.

**preprocess_num_workers**
The number of threads used to read simulation files in parallel during
the one-time preprocessing pass.  Default ``1`` (sequential).  Increasing
this speeds up the SSD write phase by reading multiple Lustre files
concurrently; the extra RAM cost is `preprocess_num_workers × bytes_per_file`
(≈ 40 MB per thread for 1023×1023 / 10-channel files).  Has no effect
during training epochs.  4–8 typically saturates Lustre bandwidth without
meaningful RAM cost.

**preprocessing pass**
The one-time step (run at the start of the first job) that reads all raw
simulation files through the full transformation pipeline
(Alfvén units → prescaling → normalization), converts them to float32,
and saves them as chunked `.pt` tensor files on the local SSD.  Subsequent
training epochs skip this step entirely and read the pre-computed tensors.

**slot**
An integer index into the physical chunk layout (0 to num_files − 1).
Slot `i` is stored at position `i % chunk_size` within chunk
`i // chunk_size`.  The relationship between slots and CSV file indices
is given by `file_perm`: `slot i` holds CSV file `file_perm[i]`.

**ssd_cache_dir**
The root directory on the local NVMe SSD where preprocessed chunk files
are written.  On Hortense, defaults to `$TMPDIR/closure_preprocessed`
(ephemeral, wiped at job end).  Can be overridden with
`--data.ssd_cache_dir=<path>` to use a persistent location.

**$TMPDIR**
An environment variable set by Slurm pointing to a per-job directory on
the local NVMe SSD of the compute node.  On Hortense GPU nodes this is
480 GB; on debug/CPU nodes it is 100 GB.  The directory and all its
contents are wiped automatically when the job ends.
