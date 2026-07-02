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
num_workers=4

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

    reconnection_cmd=(
      closure-diagnostics reconnection
      --backend menura
      --files-path "$run_dir"
      --choose-times all
      --processed
      --choose-x 0,512
      --choose-y 0,256
      --menura-scale-ranges
      --az-sigma 4
      --recon-normalization notebook
      --csv-mode replace
      --output-csv "$out_dir/reconnection_menura.csv"
      --experiment-workers "$experiment_workers"
      --num-workers "$num_workers"
    )
    printf 'Running:'
    printf ' %q' "${reconnection_cmd[@]}"
    printf '\n'
    "${reconnection_cmd[@]}"

    profiles_cmd=(
      closure-diagnostics profiles
      --backend menura
      --files-path "$run_dir"
      --fields P_e,P_i,rho_e,rho_i,Jz_e,Jz_i,Bx,By
      --projection y
      --choose-times 0
      --processed
      --choose-x 0,512
      --choose-y 0,256
      --menura-scale-ranges
      --output-csv "$out_dir/profiles_menura.csv"
    )
    printf 'Running:'
    printf ' %q' "${profiles_cmd[@]}"
    printf '\n'
    "${profiles_cmd[@]}"
  fi
done
