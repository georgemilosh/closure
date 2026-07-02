cd /dodrio/scratch/projects/2026_018/george/closure
source activate_hpc.sh

files_path=/dodrio/scratch/projects/2026_018/share_dir/iPiC3D-nathan
diagnostics_root=diagnostics/iPiC3D-nathan

for run_dir in "$files_path"/*_f2/; do
  if [ -d "$run_dir" ]; then
    run_name=${run_dir#"$files_path"/}
    run_name=${run_name%/}

    # Le2DHGEM_RunID_<n>_f2  ->  R<n>   (e.g. R0)
    label=R$(echo "$run_name" | sed -E 's/^.*RunID_([0-9]+).*$/\1/')
    out_dir="$diagnostics_root/$label"

    echo "Processing: $run_name -> $out_dir"
    mkdir -p "$out_dir"

    echo "Running: closure-diagnostics reconnection $run_name --files-path $files_path --choose-times all --processed --normalization alfven-infer --sample-nb-factor 1 --choose-species e,i,e,i --choose-x 0,512 --choose-y 0,256 --az-sigma 4 --recon-normalization notebook --csv-mode replace --output-csv $out_dir/reconnection_ecsim.csv"
    closure-diagnostics reconnection "$run_name" \
      --files-path "$files_path" \
      --choose-times all --processed \
      --normalization alfven-infer --sample-nb-factor 1 --choose-species e,i,e,i \
      --choose-x 0,512 --choose-y 0,256 \
      --az-sigma 4 --recon-normalization notebook --csv-mode replace \
      --output-csv "$out_dir/reconnection_ecsim.csv"

    echo "Running: closure-diagnostics profiles $run_name --files-path $files_path --fields P_e,P_i,rho_e,rho_i,Jz_e,Jz_i,Bx,By --projection y --choose-times 0 --processed --normalization alfven-infer --sample-nb-factor 1 --choose-species e,i,e,i --choose-x 0,512 --choose-y 0,256 --output-csv $out_dir/profiles_ecsim.csv"
    closure-diagnostics profiles "$run_name" \
      --files-path "$files_path" \
      --fields P_e,P_i,rho_e,rho_i,Jz_e,Jz_i,Bx,By \
      --projection y --choose-times 0 --processed \
      --normalization alfven-infer --sample-nb-factor 1 --choose-species e,i,e,i \
      --choose-x 0,512 --choose-y 0,256 \
      --output-csv "$out_dir/profiles_ecsim.csv"
  fi
done
