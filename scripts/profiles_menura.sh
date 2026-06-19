#!/bin/bash
# Mirror the fullres.ipynb profile cells for a Menura run.
#
# Same intent as profiles_ecsim.sh, but uses the menura backend and scales the
# crop ranges from the 512-cell base to the run resolution (--menura-scale-ranges).
# profile_fns cuts along y at x = nx//2, t = time_idx, on alfven-sample-normalized
# data (nb = rho_i.max()). Emits one per-field PNG so each figure shows a single
# field instead of all of them at once.
#
# Usage:
#   scripts/profiles_menura.sh [OUTDIR] [RUN ...]
# Each RUN may be a bare run folder (R0, R5, ...) -> expanded to RUN/$MODEL, or a
# full experiment path (R0/some_other_model) which is used as-is. Pass several to
# overlay them per field, e.g.:
#   scripts/profiles_menura.sh diagnostics/cmp R0 R5 R7
# Override the shared model with MODEL=...; defaults reproduce R0.
set -euo pipefail

OUTDIR="${1:-diagnostics/profiles_menura}"
shift || true
MODEL="${MODEL:-iso_GEM_1e-2_Jze.5_r0_1024x1024}"

# Expand bare run names (no "/") to RUN/$MODEL; keep explicit paths as given.
EXPERIMENTS=()
for run in "${@:-R0}"; do
  case "${run}" in
    */*) EXPERIMENTS+=("${run}") ;;
    *)   EXPERIMENTS+=("${run}/${MODEL}") ;;
  esac
done

FILES_PATH="/volume1/scratch/georgem/menura/runs/GEM/hortense/nathan5-12"
CSV="${OUTDIR}/profiles_menura.csv"
# Notebook profile cells (cells 7-14): one field each.
FIELDS=(P_e P_i rho_e rho_i Jz_e Jz_i Bx By)

mkdir -p "${OUTDIR}"

# 1) Export every field once (projection y, cut at nx//2 by default, time index
#    0, alfven-sample normalization with nb = rho_i.max()).
closure-diagnostics profiles "${EXPERIMENTS[@]}" \
  --backend menura \
  --files-path "${FILES_PATH}" \
  --fields "$(IFS=,; echo "${FIELDS[*]}")" \
  --projection y --choose-times 0 --processed \
  --choose-x 0,512 --choose-y 0,256 --menura-scale-ranges \
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
