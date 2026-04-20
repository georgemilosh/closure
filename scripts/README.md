# Scripts

## Run the downscaling job

From this folder, submit:

```bash
cd /dodrio/scratch/projects/2026_018/george/closure/scripts
mkdir -p logs
sbatch run_downscale.sh <PATH_TO_DATA> <READ_FOLDER> <WRITE_FOLDER> <ZOOM> [OUTPUT_FORMAT] [NO_FILTERS]
```

Example:

```bash
sbatch run_downscale.sh /dodrio/scratch/projects/2025_112/nathan/ Le2DHGEM_RunID_5 Le2DHGEM_RunID_5_ds 0.25

# Optional output format (pkl or npz)
sbatch run_downscale.sh /dodrio/scratch/projects/2025_112/nathan/ Le2DHGEM_RunID_5 Le2DHGEM_RunID_5_ds 0.25 npz

# Optional format conversion mode (disable filter/zoom processing)
sbatch run_downscale.sh /dodrio/scratch/projects/2026_018/share_dir/iPiC3D-nathan/ Le2DHGEM_RunID_5 Le2DHGEM_RunID_5_npz 1.0 npz 1
```

Notes:
- Keep a trailing slash in `<PATH_TO_DATA>` with the current `downscale.py` path concatenation logic.
- SLURM stdout/stderr are written into `logs/`.
- `downscale.py` copies `SimulationData.txt`, optional `*.inp`, and optional `ConservedQuantities.txt` into the output folder.
- `OUTPUT_FORMAT` defaults to `pkl` if omitted.
- `NO_FILTERS` defaults to `0`; set to `1` (or `true`) to skip filter/zoom processing.
