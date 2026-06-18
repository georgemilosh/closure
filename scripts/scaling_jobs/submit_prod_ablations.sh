#!/bin/bash
# =============================================================================
# submit_prod_ablations.sh — submit the production ablation matrix as SEPARATE
# SLURM batch jobs (one independent job per combination), so they schedule and
# run in PARALLEL across GPUs instead of serially inside one allocation.
#
# Each job runs ONE feature ablation (FEATURE_SET) of ONE model (cnn|mlp); the
# job script itself loops over the targets (P, divP) and over ARCH_LIST. With
# SPLIT_ARCH=1 (see below) the arch loop is ALSO split into separate jobs, giving
# maximum parallelism (one job per model × feature × arch).
#
# -----------------------------------------------------------------------------
# OPTIONS (all are environment variables; shown with their defaults)
# -----------------------------------------------------------------------------
#   MODELS="cnn mlp"                  Which model(s) to submit. Space-separated
#                                     subset of {cnn, mlp}. One job family each.
#                                       cnn -> train_prod_ablation_cnn_p1023_step100_1n_1g.sh
#                                              (plateau config, bs=16, p=1023, ReduceLROnPlateau)
#                                       mlp -> train_prod_ablation_mlp_step100_1n_1g.sh
#                                              (cosine config, bs=8096, flatten, lr=4e-4)
#
#   FEATURES="default noE noJ noJnoE" Which input-feature ablations to submit.
#                                     Space-separated subset of:
#                                       default = rho_e,B,V_e,E      (10 ch)
#                                       noE     = drop E fields      (7 ch)
#                                       noJ     = drop V_e/currents  (7 ch)
#                                       noJnoE  = rho_e,B only       (4 ch)
#                                     One job per feature (per model), unless
#                                     SPLIT_ARCH=1 (then per feature × arch).
#
#   ARCH_LIST="baseline"             Which network-size variants to run.
#                                     Space-separated subset of:
#                                       baseline  = the tuned FCNN/MLP
#                                       shallower = one fewer layer
#                                       deeper    = one more layer
#                                     SPLIT_ARCH=0: passed whole to each job, which
#                                       runs the archs SERIALLY inside one allocation.
#                                     SPLIT_ARCH=1: the helper submits one job PER arch.
#
#   SPLIT_ARCH=0                     0 = one job per (model, feature); the job runs
#                                       all ARCH_LIST archs back-to-back (fewer jobs,
#                                       longer each).
#                                     1 = one job per (model, feature, arch); maximum
#                                       parallelism (more jobs, each shorter). Use this
#                                       when you have GPUs free and want the matrix done
#                                       in the shortest wall-clock.
#
#   MAX_EPOCHS=<unset>               Optional per-cell epoch cap. If unset, each job
#                                     uses its own default (CNN=150, MLP=3). Override
#                                     e.g. when running the full 3-arch CNN matrix to
#                                     keep 6 cells/job inside the 72 h wall.
#
#   DRY=0                            1 = print the exact sbatch command lines and do
#                                     NOT submit (dry run). Always preview with DRY=1
#                                     first.
#
# -----------------------------------------------------------------------------
# EXAMPLES
# -----------------------------------------------------------------------------
#   # Preview everything (no submission):
#   DRY=1 bash scripts/scaling_jobs/submit_prod_ablations.sh
#
#   # Default: 4 CNN + 4 MLP jobs (one per feature), baseline arch:
#   bash scripts/scaling_jobs/submit_prod_ablations.sh
#
#   # CNN only, all features:
#   MODELS="cnn" bash scripts/scaling_jobs/submit_prod_ablations.sh
#
#   # Just two features, MLP only:
#   MODELS="mlp" FEATURES="default noE" bash scripts/scaling_jobs/submit_prod_ablations.sh
#
#   # Full 3-arch CNN matrix, EACH arch as its own job (max parallelism),
#   # with a lower epoch cap so each fits the wall comfortably:
#   MODELS="cnn" ARCH_LIST="baseline shallower deeper" SPLIT_ARCH=1 MAX_EPOCHS=60 \
#     bash scripts/scaling_jobs/submit_prod_ablations.sh
#   # -> 4 features × 3 archs = 12 independent CNN jobs
#
#   # One single cell, by hand (no helper) — CNN, noE feature, deeper arch:
#   sbatch -J prod_ablate_cnn_noE_deeper \
#     --export=ALL,FEATURE_SET=noE,ARCH_LIST="deeper",MAX_EPOCHS=60 \
#     scripts/scaling_jobs/train_prod_ablation_cnn_p1023_step100_1n_1g.sh
#
# Manual whole-matrix loop (equivalent to the default helper run):
# for FS in default noE noJ noJnoE; do
#  sbatch -J prod_ablate_cnn_${FS} --export=ALL,FEATURE_SET=${FS} \
#    scripts/scaling_jobs/train_prod_ablation_cnn_p1023_step100_1n_1g.sh
#  sbatch -J prod_ablate_mlp_${FS} --export=ALL,FEATURE_SET=${FS} \
#    scripts/scaling_jobs/train_prod_ablation_mlp_step100_1n_1g.sh
# done
# =============================================================================

set -euo pipefail

cd /dodrio/scratch/projects/2026_018/george/closure
JOBDIR=scripts/scaling_jobs

MODELS="${MODELS:-cnn mlp}"
FEATURES="${FEATURES:-default noE noJ noJnoE}"
TARGETS="${TARGETS:-P divP}"
SPLIT_TARGET="${SPLIT_TARGET:-0}"
ARCH_LIST="${ARCH_LIST:-baseline}"
SPLIT_ARCH="${SPLIT_ARCH:-0}"
DRY="${DRY:-0}"

declare -A SCRIPT=(
  [cnn]="${JOBDIR}/train_prod_ablation_cnn_p1023_step100_1n_1g.sh"
  [mlp]="${JOBDIR}/train_prod_ablation_mlp_step100_1n_1g.sh"
)

# Fail fast on a mistyped CONFIG_PATH (else every job dies instantly with a missing config).
if [[ -n "${CONFIG_PATH:-}" && ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
  echo "       (check the path — common typo: '11-2' instead of '11-12')" >&2
  exit 1
fi

# submit <model> <feature_set> <targets> <arch_list> <jobname_suffix>
#   <targets>/<arch_list> are the TARGETS/ARCH_LIST values passed to THIS job (a single value
#   when split, the full list otherwise). <jobname_suffix> is appended to the job name.
submit() {
  local model="$1" fs="$2" targets="$3" arch_list="$4" suffix="$5"
  [[ -n "${SCRIPT[$model]:-}" ]] || { echo "Unknown MODEL '${model}' (expected cnn|mlp)"; exit 1; }
  local jobname="prod_ablate_${model}_${fs}${suffix}"
  # Pass params via the environment + --export=ALL (NOT a comma-list): values like
  # "P divP" / "baseline shallower deeper" contain spaces and would break --export=A,B=val.
  # --export=ALL also keeps MODULEPATH etc. that the job's `module load` needs.
  local -a envs=( "FEATURE_SET=${fs}" "TARGETS=${targets}" "ARCH_LIST=${arch_list}" )
  [[ -n "${MAX_EPOCHS:-}" ]]  && envs+=( "MAX_EPOCHS=${MAX_EPOCHS}" )
  [[ -n "${CONFIG_PATH:-}" ]] && envs+=( "CONFIG_PATH=${CONFIG_PATH}" )   # alt config (e.g. *_val0.yaml)
  [[ -n "${SAVE_DIR:-}" ]]    && envs+=( "SAVE_DIR=${SAVE_DIR}" )         # alt output folder (e.g. *_val0)
  [[ -n "${PATCH_DIM:-}" ]]   && envs+=( "PATCH_DIM=${PATCH_DIM}" )       # CNN only; e.g. [512,512] for f2 downscaled
  [[ -n "${BATCH_SIZE:-}" ]]  && envs+=( "BATCH_SIZE=${BATCH_SIZE}" )
  if [[ "${DRY}" == "1" ]]; then
    printf 'env'; printf ' %q' "${envs[@]}"
    printf ' sbatch -J %q --export=ALL %q\n' "${jobname}" "${SCRIPT[$model]}"
  else
    env "${envs[@]}" sbatch -J "${jobname}" --export=ALL "${SCRIPT[$model]}"
  fi
}

# Build target groups: one job per target if SPLIT_TARGET=1, else one job covering all TARGETS.
if [[ "${SPLIT_TARGET}" == "1" ]]; then tgroups=(${TARGETS}); else tgroups=("${TARGETS}"); fi

n=0
for model in ${MODELS}; do
  for fs in ${FEATURES}; do
    for tg in "${tgroups[@]}"; do
      tsuf=""; [[ "${SPLIT_TARGET}" == "1" ]] && tsuf="_${tg}"
      if [[ "${SPLIT_ARCH}" == "1" ]]; then
        for arch in ${ARCH_LIST}; do
          submit "${model}" "${fs}" "${tg}" "${arch}" "${tsuf}_${arch}"; n=$((n+1))
        done
      else
        submit "${model}" "${fs}" "${tg}" "${ARCH_LIST}" "${tsuf}"; n=$((n+1))
      fi
    done
  done
done

echo "# ${n} job(s) $([[ "${DRY}" == "1" ]] && echo 'previewed (DRY=1)' || echo 'submitted')."
