#!/bin/bash
#SBATCH --job-name=bands
#SBATCH --account=2026_018
#SBATCH --partition=cpu_milan_rhel9
#SBATCH --error=logs/bands_%x_%j.err
#SBATCH --output=logs/bands_%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#
# Band diagnostics over every menura run in a campaign folder.
#
# Submit as a batch job:   sbatch scripts/run_bands.sh [folder] [single_run]
# Or run interactively:    bash   scripts/run_bands.sh [folder] [single_run]
# (the #SBATCH lines above are ignored when run with bash).
#
# Args (both optional; fall back to the defaults set below):
#   $1  campaign folder under menura/runs   (e.g. nathan5-12)
#   $2  single run dir to process only that run (else all runs in the folder)

# NOTE: no `set -e` on purpose. If one run's diagnostics fail (e.g. an OOM
# kill on a large run), we log it and continue to the next run rather than
# aborting the whole batch — matching the original interactive behaviour.

# --- Cluster stack -----------------------------------------------------------
# A batch job inherits the cluster module loaded at submit time. Force the CPU
# stack so activate_hpc.sh loads the CPU (non-CUDA) PyTorch modules; otherwise a
# job submitted from a GPU login environment loads mismatched CUDA modules.
module swap cluster/dodrio/cpu_milan_rhel9 2>/dev/null \
  || module load cluster/dodrio/cpu_milan_rhel9 2>/dev/null \
  || true

cd /dodrio/scratch/projects/2026_018/george/closure
source activate_hpc.sh
mkdir -p logs

# --- Configuration -----------------------------------------------------------
folder="${1:-nathan5-12}"
#folder=physics_campaign_f2
#folder=nathan5-12_f2
#folder=stability_campaign
#folder=stability_campaign2
#folder=stability_campaign100ppc
runs_root=/dodrio/scratch/projects/2026_018/george/menura/runs/$folder
diagnostics_root=diagnostics/$folder
experiment_workers=2
# bands has no --num-workers: the spectrum is one vectorized FFT over all
# snapshots, not a per-snapshot search loop, so only --experiment-workers
# (parallel runs) applies. Each worker loads one full run into RAM at once, so
# raising experiment_workers raises peak RAM (and OOM risk) proportionally —
# keep it small unless you also raise --mem above.

# Set single_run to a specific run dir to process only that run, or leave empty "" for all runs
single_run="${2:-}"
#single_run="/dodrio/scratch/projects/2026_018/george/menura/runs/nathan5-12_f2/R5"

if [ -n "$single_run" ]; then
  run_dirs=("$single_run")
else
  mapfile -t run_dirs < <(find "$runs_root" -mindepth 1 -maxdepth 1 -type d)
fi

# --- Main loop ---------------------------------------------------------------
for run_dir in "${run_dirs[@]}"; do
  if [ -d "$run_dir" ]; then
    rel_path=${run_dir#"$runs_root"/}
    rel_path=${rel_path%/}
    out_dir="$diagnostics_root/$rel_path"
    echo "Processing: $run_dir"
    mkdir -p "$out_dir"

    bands_cmd=(
      closure-diagnostics bands
      --backend menura
      --files-path "$run_dir"
      --choose-times all
      --choose-x 0,512
      --choose-y 0,256
      --menura-scale-ranges
      --csv-mode replace
      --output-csv "$out_dir/bands_menura.csv"
      --experiment-workers "$experiment_workers"
    )
    printf 'Running:'
    printf ' %q' "${bands_cmd[@]}"
    printf '\n'
    if ! "${bands_cmd[@]}"; then
      echo "WARNING: band diagnostics failed for $run_dir (continuing to next run)" >&2
    fi
  fi
done
