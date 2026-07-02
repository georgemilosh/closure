cd /dodrio/scratch/projects/2026_018/george/closure
source activate_hpc.sh
#folder=nathan5-12
#folder=physics_campaign_f2
#folder=nathan5-12_f2
#folder=stability_campaign
folder=stability_campaign2
#folder=stability_campaign100ppc
runs_root=/dodrio/scratch/projects/2026_018/george/menura/runs/$folder
# runs_root=/dodrio/scratch/projects/2026_018/george/menura/runs/$folder
diagnostics_root=diagnostics/$folder
experiment_workers=2
# bands has no --num-workers: the spectrum is one vectorized FFT over all
# snapshots, not a per-snapshot search loop, so only --experiment-workers
# (parallel runs) applies.

# Set single_run to a specific run dir to process only that run, or leave empty "" for all runs
single_run=""
#single_run="/dodrio/scratch/projects/2026_018/george/menura/runs/nathan5-12_f2/R5"

if [ -n "$single_run" ]; then
  run_dirs=("$single_run")
else
  mapfile -t run_dirs < <(find "$runs_root" -mindepth 1 -maxdepth 1 -type d)
fi

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
    "${bands_cmd[@]}"
  fi
done
