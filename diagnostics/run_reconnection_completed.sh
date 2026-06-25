#!/bin/bash
# Run closure-diagnostics reconnection over the COMPLETED menura runs harvested
# from runs.csv (see completed_runs_<tree>.txt, one experiment per line).
#
# One output CSV per tree: the 512 (nathan5-12_f2) and 1024 (nathan5-12) MLP runs
# share the same experiment string (R5/MLP/1e-2/noJnoE_P_baseline, ...), so writing
# them to a single CSV would collide on run_name. Keep them separate.
set -euo pipefail
cd "$(dirname "$0")/.."          # -> closure repo root (where diagnostics/ lives)

RUNS_ROOT=/dodrio/scratch/projects/2026_018/george/menura/runs
ANALYSIS_DIR=/dodrio/scratch/projects/2026_018/george/menura/analysis

run_tree() {
    local tree=$1
    local list="diagnostics/completed_runs_${tree}.txt"
    read -ra experiments < "$list"   # space-separated experiments on one line
    [[ ${#experiments[@]} -gt 0 ]] || { echo "no runs in $list"; return; }
    echo "=== $tree : ${#experiments[@]} runs -> diagnostics/reconnection_${tree}.csv ==="
    closure-diagnostics reconnection "${experiments[@]}" \
        --backend menura \
        --files-path "$RUNS_ROOT/$tree" \
        --menura-analysis-dir "$ANALYSIS_DIR" \
        --choose-times all --processed \
        --normalization alfven-sample --sample-nb-factor 1 \
        --choose-x 0,512 --choose-y 0,256 --menura-scale-ranges \
        --az-sigma 4 --recon-normalization notebook \
        --csv-mode replace \
        --output-csv "diagnostics/reconnection_${tree}.csv"
}

run_tree nathan5-12
run_tree nathan5-12_f2
