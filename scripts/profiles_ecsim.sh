#!/bin/bash
# Mirror the fullres.ipynb profile cells for an ECsim (iPiC3D) run.
#
# Each notebook profile cell plots ONE field along y at x = nx//2, t = time_idx,
# using profile_fns(name, species, time_idx) on the alfven-sample-normalized data.
# This script exports the profile CSV once, then emits one per-field PNG (so a
# single figure shows just that field across the requested runs), matching the
# notebook instead of dumping every field onto one axes.
#
# Usage:
#   scripts/profiles_ecsim.sh [OUTDIR] [EXPERIMENT ...]
# Defaults reproduce R0 (Le2DHGEM_RunID_0_f2); pass more experiment names to
# overlay several runs per field (e.g. Le2DHGEM_RunID_0_f2 Le2DHGEM_RunID_5_f2).
set -euo pipefail

OUTDIR="${1:-diagnostics/profiles_ecsim}"
shift || true
EXPERIMENTS=("${@:-Le2DHGEM_RunID_0_f2}")

FILES_PATH="/volume1/scratch/share_dir/iPiC3D-nathan"
CSV="${OUTDIR}/profiles_ecsim.csv"
# Notebook profile cells (cells 7-14): one field each.
FIELDS=(P_e P_i rho_e rho_i Jz_e Jz_i Bx By)

mkdir -p "${OUTDIR}"

# 1) Export every field once (matches profile_fns: projection y, cut at nx//2,
#    time index 0, alfven-sample normalization with nb = rho_i.max()).
closure-diagnostics profiles "${EXPERIMENTS[@]}" \
  --files-path "${FILES_PATH}" \
  --fields "$(IFS=,; echo "${FIELDS[*]}")" \
  --projection y --choose-times 0 --processed \
  --choose-x 0,512 --choose-y 0,256 --choose-species e,i,e,i \
  --normalization alfven-sample --sample-nb-factor 1 \
  --output-csv "${CSV}"

# 2) One figure per field, mirroring each notebook cell.
for field in "${FIELDS[@]}"; do
  closure-diagnostics overlay "${CSV}" \
    --field "${field}" --x coord --y value --group-by run \
    --title "${field}" \
    --output "${OUTDIR}/profile_${field}.png"
done

echo "Wrote ${CSV} and per-field PNGs under ${OUTDIR}/"
