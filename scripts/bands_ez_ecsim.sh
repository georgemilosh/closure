# Ez-only band diagnostics for the iPiC3D/ECsim Le2DHGEM runs on haydn.
# Counterpart of the archived bands_ecsim.csv (alfven-infer normalization,
# reproduced exactly on haydn 2026-07-13), but (a) Ez-only via
# scripts/bands_ez.py and (b) full image — no --choose-x/--choose-y crop —
# matching the menura bands_ez convention. (The archived bands_ecsim.csv
# cropped to the lower current sheet, 0:512 x 0:256 of the 513x513 grid, so
# absolute powers are not directly comparable between the two files.)
#
# Writes diagnostics/iPiC3D-nathan/R<N>/bands_ez_ecsim.csv for every
# Le2DHGEM_RunID_<N>_f2 under the share dir.
#
# Usage:  bash scripts/bands_ez_ecsim.sh
# Light: ~1 GB npz per run, safe to run alongside the menura campaign jobs.

set -u
cd /volume1/scratch/georgem/closure

files_path=/volume1/scratch/share_dir/iPiC3D-nathan
diagnostics_root=diagnostics/iPiC3D-nathan
component=Ez

for run_dir in "$files_path"/Le2DHGEM_RunID_*_f2; do
  run=$(basename "$run_dir")
  n=${run#Le2DHGEM_RunID_}
  n=${n%_f2}
  out_dir="$diagnostics_root/R$n"
  echo "Processing: $run -> $out_dir"
  mkdir -p "$out_dir"

  bands_cmd=(
    python scripts/bands_ez.py "$run"
    --backend ecsim
    --files-path "$files_path"
    --component "$component"
    --choose-times all
    --choose-species e,i,e,i
    --normalization alfven-infer
    --sample-nb-factor 1
    --csv-mode replace
    --output-csv "$out_dir/bands_ez_ecsim.csv"
  )
  printf 'Running:'
  printf ' %q' "${bands_cmd[@]}"
  printf '\n'
  "${bands_cmd[@]}"
done
