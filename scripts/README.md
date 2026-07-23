# Scripts

## Run the downscaling job

From this folder, submit:

```bash
cd /dodrio/scratch/projects/2026_018/george/closure/scripts
mkdir -p logs
sbatch run_downscale.sh <PATH_TO_DATA> <READ_FOLDER> <WRITE_FOLDER> <ZOOM> [OUTPUT_FORMAT] [NO_FILTERS] [OUTPUT_DTYPE]
```

Example:

```bash
sbatch run_downscale.sh /dodrio/scratch/projects/2025_112/nathan/ Le2DHGEM_RunID_5 Le2DHGEM_RunID_5_ds 0.25

# Optional output format (pkl or npz)
sbatch run_downscale.sh /dodrio/scratch/projects/2025_112/nathan/ Le2DHGEM_RunID_5 Le2DHGEM_RunID_5_ds 0.25 npz

# Optional format conversion mode (disable filter/zoom processing)
sbatch run_downscale.sh /dodrio/scratch/projects/2026_018/share_dir/iPiC3D-nathan/ Le2DHGEM_RunID_5 Le2DHGEM_RunID_5_npz 1.0 npz 1

# Optional output dtype (float32 or float64)
sbatch run_downscale.sh /dodrio/scratch/projects/2026_018/share_dir/iPiC3D-nathan/ Le2DHGEM_RunID_5 Le2DHGEM_RunID_5_npz 1.0 npz 1 float32
```

Notes:
- Keep a trailing slash in `<PATH_TO_DATA>` with the current `downscale.py` path concatenation logic.
- SLURM stdout/stderr are written into `logs/`.
- `downscale.py` copies `SimulationData.txt`, optional `*.inp`, and optional `ConservedQuantities.txt` into the output folder.
- `OUTPUT_FORMAT` defaults to `pkl` if omitted.
- `NO_FILTERS` defaults to `0`; set to `1` (or `true`) to skip filter/zoom processing.
- `OUTPUT_DTYPE` defaults to `float64`; set to `float32` for single-precision output files.

## Verify a transferred ECsim run set

Run this on any newly transferred batch of iPiC3D run directories before using them:

```bash
python scripts/check_ecsim_run_integrity.py
python scripts/check_ecsim_run_integrity.py \
  --files-path /path/to/batch --pattern 'Le2DHGEM_RunID_*' --json report.json
```

It measures each run's background density, `B0x` and `B0z` from the t=0 field data, prints
them, and checks them against the `*.inp` deck and `SimulationData.txt`. Exits non-zero if
any directory disagrees. The run spreadsheet is deliberately not consulted — it records
what was intended, whereas the deck and `SimulationData.txt` ship with the data.

Notes:
- Measured quantities: `rho` is stored on disk as `rhoINIT / 4pi`, and the background
  species is spatially uniform, so its mean recovers `rhoINIT` directly. `B0x` is the lobe
  plateau of `Bx` — averaging over x first removes the initial perturbation, which
  otherwise makes `max|Bx|` overshoot (0.025149 against a true 0.0249). `B0z` is
  `mean(Bz)`. These are the constants that set the Alfven normalisation, so they are
  printed for every run whether or not it passes.
- Motivating case: `Le2DHGEM_RunID_12` arrived with correct metadata (background 0.229,
  `B0z` 0.00498) but field data at background 0.68 — the RunID_13/14 configuration. Its
  t=0 snapshot is bit-identical to RunID_7's in all arrays but `Bz`. Both the `_f2` and
  `_npz32` copies are affected, so the error predates the conversion.
- The duplicate scan reports only pairs whose *background*-species data matches. Any two
  runs in this campaign already share ~half their arrays bit-for-bit (same Harris species
  at 0.969, same ppc, same seed), which is expected and not reported.
- Use `--harris-species` / `--background-species` if the species ordering differs from the
  Double-Harris default (0,1 = sheet electrons/ions; 2,3 = background electrons/ions).
- `--no-duplicate-scan` skips the pairwise pass, which is the slow part.
