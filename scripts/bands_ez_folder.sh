# Generic per-campaign driver for Ez-only band diagnostics (full image, no crop).
# Usage:
#   bash scripts/bands_ez_folder.sh <campaign_folder> [experiment_workers] [component]
# e.g.
#   bash scripts/bands_ez_folder.sh stability_campaign2 6
# Loops over the first-level R* dirs of the campaign and writes
# diagnostics/<campaign_folder>/<R>/bands_ez_menura.csv for each.
#
# Memory (measured, all fields x all snapshots held in RAM per worker):
#   510x510 grid, ~102 snaps  -> ~21 GB peak RSS per worker
#   1024x1024 grid, ~51 snaps -> ~40 GB peak RSS per worker
# Keep experiment_workers x per-worker-peak below ~180 GB on this 256 GB machine.

set -u
cd /volume1/scratch/georgem/closure

folder=${1:?usage: bands_ez_folder.sh <campaign_folder> [experiment_workers] [component]}
experiment_workers=${2:-4}
component=${3:-Ez}

runs_root=/esat/cpadata/georgem/2025_112/georgem/menura/runs/$folder
diagnostics_root=diagnostics/$folder

if [ ! -d "$runs_root" ]; then
  echo "No such campaign folder: $runs_root" >&2
  exit 1
fi

mapfile -t run_dirs < <(find "$runs_root" -mindepth 1 -maxdepth 1 -type d)

for run_dir in "${run_dirs[@]}"; do
  if [ -d "$run_dir" ]; then
    rel_path=${run_dir#"$runs_root"/}
    rel_path=${rel_path%/}
    out_dir="$diagnostics_root/$rel_path"
    echo "Processing: $run_dir"
    mkdir -p "$out_dir"

    bands_cmd=(
      python scripts/bands_ez.py
      --backend menura
      --files-path "$run_dir"
      --component "$component"
      --choose-times all
      --csv-mode replace
      --output-csv "$out_dir/bands_ez_menura.csv"
      --experiment-workers "$experiment_workers"
    )
    printf 'Running:'
    printf ' %q' "${bands_cmd[@]}"
    printf '\n'
    "${bands_cmd[@]}"
  fi
done
