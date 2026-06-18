#!/bin/bash
# Production ablation driver — CNN (FCNN) at the validated production schedule:
# full-resolution step100 data (1004 files), patch_dim=1023 (full image), oversample=1,
# preprocessed 0.25 GB chunks, bf16-mixed, and the PLATEAU config:
# ReduceLROnPlateau + lr=1e-4 + batch_size=16 (Runs_7-9-10-11-12_long_plateau_step100.yaml).
# bs=16 is the 40GB-A100 ceiling at p=1023 and removes the small-batch BatchNorm oscillation
# that bs=4 suffered; ReduceLROnPlateau then refines on the now-smooth val signal.
#
# ONE JOB PER FEATURE ablation (FEATURE_SET): each submission covers 1 feature x 2 targets
# (P/divP) x ARCH_LIST. Default ARCH_LIST=baseline (at ~9 min/epoch the 2 cells per feature
# already fill ~45 h); pass ARCH_LIST="baseline shallower deeper" + a lower MAX_EPOCHS for the
# full 3-arch matrix. Submit the 4 feature jobs to run in parallel across GPUs.
#   features: default / noE / noJ / noJnoE   targets: P / divP   arch: baseline (default)
#
# Storage: each (feature,target) combo gets its OWN /tmp cache dir (channels differ →
# distinct chunks, ~63 GB each) which is DELETED after the combo's arch variants finish.
#
# Submit all 4 feature jobs:
#   for FS in default noE noJ noJnoE; do
#     sbatch -J prod_ablate_cnn_${FS} --export=ALL,FEATURE_SET=${FS} \
#       scripts/scaling_jobs/train_prod_ablation_cnn_p1023_step100_1n_1g.sh
#   done
# Single feature, baseline arch only:
#   sbatch -J prod_ablate_cnn_noE --export=ALL,FEATURE_SET=noE,ARCH_LIST="baseline" \
#     scripts/scaling_jobs/train_prod_ablation_cnn_p1023_step100_1n_1g.sh
#SBATCH --job-name=prod_ablate_cnn_p1023_step100
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

# --- Production hyperparameters (validated on gpu503, June 2026) ---
# Schedule: ReduceLROnPlateau + lr=1e-4 + bs=16 (the plateau config). bs=16 is the max that
# fits p=1023 on a 40GB A100 (38.7GB; bs=24/32 OOM) and gives 4x the BatchNorm batch ->
# ~3x smoother val than bs=4 (the oscillation was small-batch BN noise). ~9 min/epoch.
CONFIG_PATH="${CONFIG_PATH:-configs/iPiC3D-nathan5-12/Runs_7-9-10-11-12_long_plateau_step100.yaml}"
ARCH_LIST="${ARCH_LIST:-baseline}"      # baseline-arch only: at ~9min/epoch the 2 cells (P/divP) per feature-job already fill the budget
MAX_EPOCHS="${MAX_EPOCHS:-300}"         # atomic split = 1 cell/job, so use the budget: ~300 ep x ~9min ~= 45-50h < 72h. EarlyStopping(patience=100)+plateau stop earlier if converged; best-ckpt saved continuously.
BATCH_SIZE="${BATCH_SIZE:-16}"
PATCH_DIM="${PATCH_DIM:-[1023,1023]}"
SUBSAMPLE_RATE="${SUBSAMPLE_RATE:-1}"   # os=1: one full-image crop per file (no augmentation at p=1023; §10.8)
NUM_WORKERS="${NUM_WORKERS:-0}"          # CNN preprocessed optimum is 0 (§8.7); large patches make IPC net-negative
CHUNK_GB="${CHUNK_GB:-0.25}"
PRECISION="${PRECISION:-bf16-mixed}"
SAVE_DIR="${SAVE_DIR:-./models/Lightning/iPiC3D-nathan5-12/production_ablations_step100}"  # override for alt studies (e.g. _val0)
RUNS_DIR="${SAVE_DIR}/runs"
CLEAN_TMP="${CLEAN_TMP:-1}"             # rm each combo's /tmp chunks after its arch variants finish

echo "=== Production CNN ablation (p=1023, full-res step100) | FEATURE_SET=${FEATURE_SET} ==="
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "ARCH_LIST=${ARCH_LIST}  MAX_EPOCHS=${MAX_EPOCHS}  BATCH_SIZE=${BATCH_SIZE}  PATCH_DIM=${PATCH_DIM}"
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

  # Feature override + input channel count
  case "$feature_set" in
    default) feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz,Vx_e,Vy_e,Vz_e,Ex,Ey,Ez]'; feature_channels=10 ;;
    noE)     feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz,Vx_e,Vy_e,Vz_e]'; feature_channels=7 ;;
    noJ)     feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz,Ex,Ey,Ez]'; feature_channels=7 ;;
    noJnoE)  feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz]'; feature_channels=4 ;;
  esac

  # Target override + output channel count
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

  # Architecture override (FCNN channels)
  if [[ "$arch" == "deeper" ]]; then
    arch_override="--model.network.init_args.channels=[${feature_channels},128,128,128,64,${target_channels}] --model.network.init_args.kernels=[3,3,3,5,3] --model.network.init_args.activations=[SiLU,SiLU,SiLU,SiLU,null] --model.network.init_args.batch_norms=[true,true,true,true,false] --model.network.init_args.dropouts=[0.0,0.15,0.15,0.1,0.0]"
  elif [[ "$arch" == "shallower" ]]; then
    arch_override="--model.network.init_args.channels=[${feature_channels},128,64,${target_channels}] --model.network.init_args.kernels=[3,5,3] --model.network.init_args.activations=[SiLU,SiLU,null] --model.network.init_args.batch_norms=[true,true,false] --model.network.init_args.dropouts=[0.0,0.1,0.0]"
  elif [[ "$arch" == "baseline" ]]; then
    arch_override="--model.network.init_args.channels=[${feature_channels},128,128,64,${target_channels}] --model.network.init_args.kernels=[3,3,5,3] --model.network.init_args.activations=[SiLU,SiLU,SiLU,null] --model.network.init_args.batch_norms=[true,true,true,false] --model.network.init_args.dropouts=[0.0,0.15,0.15,0.0]"
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
      --data.patch_dim="${PATCH_DIM}" \
      --data.subsample_rate="${SUBSAMPLE_RATE}" \
      --data.batch_size="${BATCH_SIZE}" \
      --data.num_workers="${NUM_WORKERS}" \
      --trainer.logger.init_args.save_dir="${SAVE_DIR}" \
      --trainer.logger.init_args.name=runs \
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
  combo_cache="/tmp/closure_prod_cnn_${SLURM_JOB_ID:-$$}_${feature_set}_${target_set}"
  for arch in ${ARCH_LIST}; do
    run_one "$feature_set" "$target_set" "$arch" "$combo_cache"
  done
  if [[ "${CLEAN_TMP}" == "1" ]]; then
    echo "Cleaning combo cache ${combo_cache}"; rm -rf "${combo_cache}" || true; echo ""
  fi
done

total_end=$(date +%s); s=$((total_end - total_start))
echo "=== CNN production ablation (${feature_set}) done: $((s/3600))h $(((s%3600)/60))m ==="
