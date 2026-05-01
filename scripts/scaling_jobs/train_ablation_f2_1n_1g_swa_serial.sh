#!/bin/bash
# Serial ablation driver for all 4x2 feature/target combinations on single GPU.
# Default runs 8 combinations (default, noE, noJ, noJnoE × P, divP) with baseline architecture.
# You can also run shallower/deeper variants in the same job using ARCH_LIST="baseline shallower deeper".
# Select training config via CONFIG_PATH (defaults to cosine SWA).
# Submit with:
#   sbatch --export=ALL,CONFIG_PATH=configs/iPiC3D-nathan5-12/Runs_7-9-10-11-12_f2_plateau_swa.yaml,ARCH_LIST="baseline shallower deeper" scripts/scaling_jobs/train_ablation_f2_1n_1g_swa_serial.sh
#SBATCH --job-name=ipic_ablate_f2_1g_serial
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

echo "=== Starting serial 300-epoch ablation matrix (all 8 combinations) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
echo ""

CONFIG_PATH="${CONFIG_PATH:-configs/iPiC3D-nathan5-12/Runs_7-9-10-11-12_f2_cosine_swa.yaml}"
ARCH_LIST="${ARCH_LIST:-baseline shallower deeper}"
RUNS_DIR="./models/Lightning/iPiC3D-nathan5-12/ablations_f2_serial/runs"
echo "Using CONFIG_PATH=${CONFIG_PATH}"
echo "Using ARCH_LIST=${ARCH_LIST}"
echo ""

# Function to run one ablation combination
run_ablation() {
  local feature_set="$1"
  local target_set="$2"
  local arch="$3"
  local run_tag="ablate_${feature_set}_${target_set}_${arch}"
  local run_dir="${RUNS_DIR}/${run_tag}"

  # Skip reruns only when prior artifacts indicate a completed train+test pass.
  if [[ -s "${run_dir}/test_metrics.csv" && -f "${run_dir}/checkpoints/last.ckpt" ]]; then
    echo "=========================================="
    echo "Skipping completed run: ${run_tag}"
    echo "=========================================="
    echo ""
    return 0
  fi

  echo "=========================================="
  echo "Starting: ${run_tag}"
  echo "=========================================="

  # Determine feature override and channels
  case "$feature_set" in
    default)
      feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz,Vx_e,Vy_e,Vz_e,Ex,Ey,Ez]'
      feature_channels=10
      ;;
    noE)
      feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz,Vx_e,Vy_e,Vz_e]'
      feature_channels=7
      ;;
    noJ)
      feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz,Ex,Ey,Ez]'
      feature_channels=7
      ;;
    noJnoE)
      feature_override='--data.read_features_targets_kwargs.request_features=[rho_e,Bx,By,Bz]'
      feature_channels=4
      ;;
  esac

  # Determine target override and channels
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

  if [[ "$arch" == "deeper" ]]; then
    model_channels_override="--model.network.init_args.channels=[${feature_channels},128,128,128,64,${target_channels}]"
    model_kernels_override="--model.network.init_args.kernels=[3,3,3,5,3]"
    model_acts_override="--model.network.init_args.activations=[SiLU,SiLU,SiLU,SiLU,null]"
    model_bns_override="--model.network.init_args.batch_norms=[true,true,true,true,false]"
    model_drops_override="--model.network.init_args.dropouts=[0.0,0.15,0.15,0.1,0.0]"
  elif [[ "$arch" == "shallower" ]]; then
    model_channels_override="--model.network.init_args.channels=[${feature_channels},128,64,${target_channels}]"
    model_kernels_override="--model.network.init_args.kernels=[3,5,3]"
    model_acts_override="--model.network.init_args.activations=[SiLU,SiLU,null]"
    model_bns_override="--model.network.init_args.batch_norms=[true,true,false]"
    model_drops_override="--model.network.init_args.dropouts=[0.0,0.1,0.0]"
  elif [[ "$arch" == "baseline" ]]; then
    model_channels_override="--model.network.init_args.channels=[${feature_channels},128,128,64,${target_channels}]"
    model_kernels_override="--model.network.init_args.kernels=[3,3,5,3]"
    model_acts_override="--model.network.init_args.activations=[SiLU,SiLU,SiLU,null]"
    model_bns_override="--model.network.init_args.batch_norms=[true,true,true,false]"
    model_drops_override="--model.network.init_args.dropouts=[0.0,0.15,0.15,0.0]"
  else
    echo "Unsupported architecture: ${arch} (expected baseline, shallower, or deeper)"
    return 1
  fi

  if srun --cpu-bind=none closure-train fit \
    --config "${CONFIG_PATH}" \
    --trainer.devices=1 \
    --trainer.num_nodes=1 \
    --trainer.strategy=auto \
    --trainer.enable_progress_bar=false \
    --data.batch_size=32 \
    --data.num_workers=12 \
    --data.read_features_targets_kwargs.num_workers=4 \
    --trainer.logger.init_args.save_dir=./models/Lightning/iPiC3D-nathan5-12/ablations_f2_serial \
    --trainer.logger.init_args.name=runs \
    --trainer.logger.init_args.version="${run_tag}" \
    --trainer.default_root_dir="./models/Lightning/iPiC3D-nathan5-12/ablations_f2_serial" \
    $feature_override \
    $target_override \
    $prescaler_override \
    $model_channels_override \
    $model_kernels_override \
    $model_acts_override \
    $model_bns_override \
    $model_drops_override \
    $fields_override
  then
    echo "✓ ${run_tag} completed successfully"
  else
    echo "✗ ${run_tag} FAILED"
    echo "Continuing with remaining ablations..."
  fi
  echo ""
}

# Run all combinations in series
total_start=$(date +%s)

for arch in ${ARCH_LIST}; do
  for feature_set in default noE noJ noJnoE; do
    for target_set in P divP; do
      run_ablation "$feature_set" "$target_set" "$arch"
    done
  done
done

total_end=$(date +%s)
total_seconds=$((total_end - total_start))
total_hours=$((total_seconds / 3600))
total_mins=$(((total_seconds % 3600) / 60))

echo "=========================================="
echo "All ablations completed!"
echo "Total runtime: ${total_hours}h ${total_mins}m"
echo "=========================================="
