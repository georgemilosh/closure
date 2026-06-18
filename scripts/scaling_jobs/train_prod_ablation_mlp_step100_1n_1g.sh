#!/bin/bash
# Production ablation driver — MLP (pixel-wise) at the BEST hyperparameters found in
# benchmarks_f2_serial: full-resolution step100 data (1004 files), flatten=true,
# batch_size=8096, num_workers=12 (MLP scales with workers; §4.12), preprocessed
# 0.25 GB chunks, bf16-mixed.  See README §5.16 / §8.10.
#
# NOTE (§5.16): the MLP converges by epoch ~1 and neither more epochs nor more capacity
# lowers its val floor (~0.14).  MAX_EPOCHS defaults to 3 (best.pt = epoch ~1 anyway), and
# ARCH_LIST defaults to "baseline" only — capacity ablation is known to be flat, so the
# deeper/shallower MLP variants add cost without insight.  Set ARCH_LIST="baseline shallower
# deeper" to reproduce the full 3-arch matrix if desired.
#
# ONE JOB PER FEATURE ablation (FEATURE_SET): each submission covers 1 feature x 2 targets
# (P/divP) x ARCH_LIST.  Submit 4 jobs (default/noE/noJ/noJnoE) for the full matrix.
#   features: default / noE / noJ / noJnoE   targets: P / divP
#
# Storage: per-(feature,target) /tmp cache dir, deleted after each combo (see CNN script notes).
#
# Submit all 4 feature jobs:
#   for FS in default noE noJ noJnoE; do
#     sbatch -J prod_ablate_mlp_${FS} --export=ALL,FEATURE_SET=${FS} \
#       scripts/scaling_jobs/train_prod_ablation_mlp_step100_1n_1g.sh
#   done
# Full 3-arch MLP capacity ablation for one feature:
#   sbatch -J prod_ablate_mlp_default --export=ALL,FEATURE_SET=default,ARCH_LIST="baseline shallower deeper" \
#     scripts/scaling_jobs/train_prod_ablation_mlp_step100_1n_1g.sh
#SBATCH --job-name=prod_ablate_mlp_step100
#SBATCH --account=2026_018
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_rome_a100

set -euo pipefail

REPO_DIR="/dodrio/scratch/projects/2026_018/george/closure"
cd "$REPO_DIR"
mkdir -p logs

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
module load PyTorch-Lightning/2.2.1-foss-2023a-CUDA-12.1.1

export PATH="${REPO_DIR}/_shims:${PATH}"
export PYTHONPATH="${REPO_DIR}/_shims:${REPO_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export MPI4PY_RC_INITIALIZE=0
export OMPI_MCA_ess=singleton
export OMPI_MCA_plm=isolated

# --- One feature ablation per job ---
FEATURE_SET="${FEATURE_SET:-default}"
case "$FEATURE_SET" in
  default|noE|noJ|noJnoE) ;;
  *) echo "FEATURE_SET must be one of: default noE noJ noJnoE (got '${FEATURE_SET}')"; exit 1 ;;
esac

# --- Production hyperparameters (best from benchmarks_f2_serial) ---
CONFIG_PATH="${CONFIG_PATH:-configs/iPiC3D-nathan5-12/Runs_7-9-10-11-12_long1000_cosine_swa_step100.yaml}"
ARCH_LIST="${ARCH_LIST:-baseline}"      # capacity is flat for MLP (§5.16.2); baseline-only by default
MAX_EPOCHS="${MAX_EPOCHS:-10}"          # MLP saturates ~epoch 1 (§5.16.1); atomic split affords more room — 10 ep (~7h) gives a full curve before SWA. EarlyStopping/best-ckpt govern.
BATCH_SIZE="${BATCH_SIZE:-8096}"
LR="${LR:-1.0e-4}"                      # lowered from config default 4e-4: gpu516 probe showed 1e-4 best (0.136 vs 0.153); bs=2048 worse, so keep bs=8096
NUM_WORKERS="${NUM_WORKERS:-12}"        # MLP throughput scales with workers (§4.12)
CHUNK_GB="${CHUNK_GB:-0.25}"
PRECISION="${PRECISION:-bf16-mixed}"
SAVE_DIR="${SAVE_DIR:-./models/Lightning/iPiC3D-nathan5-12/production_ablations_step100}"  # override for alt studies (e.g. _val0)
RUNS_DIR="${SAVE_DIR}/runs_MLP"
CLEAN_TMP="${CLEAN_TMP:-1}"

echo "=== Production MLP ablation (full-res step100) | FEATURE_SET=${FEATURE_SET} ==="
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "ARCH_LIST=${ARCH_LIST}  MAX_EPOCHS=${MAX_EPOCHS}  BATCH_SIZE=${BATCH_SIZE}  NUM_WORKERS=${NUM_WORKERS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
echo ""

run_one() {
  local feature_set="$1" target_set="$2" arch="$3" combo_cache="$4"
  local run_tag="ablate_${feature_set}_${target_set}_${arch}"
  local run_dir="${RUNS_DIR}/${run_tag}"

  if [[ -f "${run_dir}/checkpoints/last.ckpt" ]]; then
    echo "== Skipping completed run: ${run_tag} =="; echo ""; return 0
  fi
  echo "== Starting: ${run_tag} (cache=${combo_cache}) =="

  case "$feature_set" in
    default) feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz,Vx_e,Vy_e,Vz_e,Ex,Ey,Ez]'; feature_channels=10 ;;
    noE)     feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz,Vx_e,Vy_e,Vz_e]'; feature_channels=7 ;;
    noJ)     feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz,Ex,Ey,Ez]'; feature_channels=7 ;;
    noJnoE)  feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz]'; feature_channels=4 ;;
  esac

  case "$target_set" in
    P)
      target_override='--data.read_features_targets_kwargs.request_targets=[Pxx_e,Pyy_e,Pzz_e,Pxy_e,Pxz_e,Pyz_e]'
      prescaler_override='--data.prescaler_targets=[log,log,log,arcsinh,arcsinh,arcsinh]'
      target_channels=6
      fields_override='--data.read_features_targets_kwargs.fields_to_read.B=true --data.read_features_targets_kwargs.fields_to_read.B_ext=false --data.read_features_targets_kwargs.fields_to_read.divB=false --data.read_features_targets_kwargs.fields_to_read.E=true --data.read_features_targets_kwargs.fields_to_read.E_ext=false --data.read_features_targets_kwargs.fields_to_read.rho=true --data.read_features_targets_kwargs.fields_to_read.J=true --data.read_features_targets_kwargs.fields_to_read.P=true --data.read_features_targets_kwargs.fields_to_read.PI=true --data.read_features_targets_kwargs.fields_to_read.Heat_flux=false --data.read_features_targets_kwargs.fields_to_read.N=false --data.read_features_targets_kwargs.fields_to_read.Qrem=false --data.read_features_targets_kwargs.fields_to_read.divP=false'
      ;;
    divP)
      target_override='--data.read_features_targets_kwargs.request_targets=[EPx,EPy,EPz]'
      prescaler_override='--data.prescaler_targets=[null,null,null]'
      target_channels=3
      fields_override='--data.read_features_targets_kwargs.fields_to_read.B=true --data.read_features_targets_kwargs.fields_to_read.B_ext=false --data.read_features_targets_kwargs.fields_to_read.divB=false --data.read_features_targets_kwargs.fields_to_read.E=true --data.read_features_targets_kwargs.fields_to_read.E_ext=false --data.read_features_targets_kwargs.fields_to_read.rho=true --data.read_features_targets_kwargs.fields_to_read.J=true --data.read_features_targets_kwargs.fields_to_read.P=true --data.read_features_targets_kwargs.fields_to_read.PI=true --data.read_features_targets_kwargs.fields_to_read.Heat_flux=false --data.read_features_targets_kwargs.fields_to_read.N=false --data.read_features_targets_kwargs.fields_to_read.Qrem=false --data.read_features_targets_kwargs.fields_to_read.divP=true'
      ;;
  esac

  # Architecture override (MLP feature_dims). Baseline = 26.6K (the best-generalising size, §5.16.2).
  if [[ "$arch" == "deeper" ]]; then
    arch_override="--model.network.init_args.feature_dims=[${feature_channels},128,128,128,64,${target_channels}] --model.network.init_args.activations=[SiLU,SiLU,SiLU,SiLU,null] --model.network.init_args.dropouts=[0.0,0.15,0.15,0.1,0.0]"
  elif [[ "$arch" == "shallower" ]]; then
    arch_override="--model.network.init_args.feature_dims=[${feature_channels},128,64,${target_channels}] --model.network.init_args.activations=[SiLU,SiLU,null] --model.network.init_args.dropouts=[0.0,0.1,0.0]"
  elif [[ "$arch" == "baseline" ]]; then
    arch_override="--model.network.init_args.feature_dims=[${feature_channels},128,128,64,${target_channels}] --model.network.init_args.activations=[SiLU,SiLU,SiLU,null] --model.network.init_args.dropouts=[0.0,0.15,0.15,0.0]"
  else
    echo "Unsupported arch: ${arch}"; return 1
  fi

  if srun --cpu-bind=none closure-train fit \
      --config "${CONFIG_PATH}" \
      --trainer.devices=1 --trainer.num_nodes=1 --trainer.strategy=auto \
      --trainer.enable_progress_bar=false \
      --trainer.max_epochs="${MAX_EPOCHS}" \
      --trainer.precision="${PRECISION}" \
      --trainer.num_sanity_val_steps=0 \
      --data.loading_mode=preprocessed \
      --data.ssd_cache_dir="${combo_cache}" \
      --data.chunk_cache_size=1 \
      --data.preprocess_chunk_size_gb="${CHUNK_GB}" \
      --data.preprocess_num_workers=8 \
      --data.flatten=true \
      --data.subsample_rate=1.0 \
      --data.batch_size="${BATCH_SIZE}" \
      --data.num_workers="${NUM_WORKERS}" \
      --data.read_features_targets_kwargs.num_workers=4 \
      --model.lr="${LR}" \
      --model.network=closure.models.MLP \
      --trainer.logger.init_args.save_dir="${SAVE_DIR}" \
      --trainer.logger.init_args.name=runs_MLP \
      --trainer.logger.init_args.version="${run_tag}" \
      --trainer.default_root_dir="${SAVE_DIR}" \
      $feature_override $target_override $prescaler_override $arch_override $fields_override
  then
    echo "OK ${run_tag}"
  else
    echo "FAIL ${run_tag} (continuing)"
  fi
  echo ""
}

total_start=$(date +%s)

# One feature per job: loop targets (each its own shared chunk set), arch INNER, clean after.
feature_set="${FEATURE_SET}"
for target_set in ${TARGETS:-P divP}; do   # TARGETS=P (or divP) to run a single target per job
  # Per-JOB-unique cache dir: prevents two jobs that share a FEATURE_SET (e.g. arch-split
  # jobs, or a resubmission) from racing on the same /tmp chunks or rm-ing each other's
  # chunks when co-scheduled on the same node. Within THIS job the arch loop still reuses it.
  combo_cache="/tmp/closure_prod_mlp_${SLURM_JOB_ID:-$$}_${feature_set}_${target_set}"
  for arch in ${ARCH_LIST}; do
    run_one "$feature_set" "$target_set" "$arch" "$combo_cache"
  done
  if [[ "${CLEAN_TMP}" == "1" ]]; then
    echo "Cleaning combo cache ${combo_cache}"; rm -rf "${combo_cache}" || true; echo ""
  fi
done

total_end=$(date +%s); s=$((total_end - total_start))
echo "=== MLP production ablation (${feature_set}) done: $((s/3600))h $(((s%3600)/60))m ==="
