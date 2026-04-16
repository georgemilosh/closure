#!/bin/bash
# Source this file to set up the closure environment using HPC modules.
#
# Usage (on a GPU node):  source activate_hpc.sh
#
# Loads the same HPC toolchain used in production (run.sh), adds
# the lightning shim so `import lightning` works, and puts the closure
# source tree on PYTHONPATH.

_CLOSURE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- HPC modules (same as production run.sh + Lightning) ---
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
module load PyTorch-Lightning/2.2.1-foss-2023a-CUDA-12.1.1

# --- pip packages not covered by modules ---
# Only needed once:
#   pip install --user jsonargparse[signatures] pandas joblib psutil
# (scipy, numpy, pyyaml, torchmetrics come from the modules above)

# --- PATH: shim scripts (closure-train, etc.) ---
export PATH="${_CLOSURE_ROOT}/_shims:${PATH}"

# --- PYTHONPATH: shim + project root ---
export PYTHONPATH="${_CLOSURE_ROOT}/_shims:${_CLOSURE_ROOT}:${PYTHONPATH}"

# For interactive single-GPU use, prevent Lightning from assuming
# distributed SLURM launch.  Remove these lines for multi-node jobs.
unset SLURM_NTASKS
unset SLURM_JOB_NAME
export SLURM_NTASKS=1

echo "closure HPC env ready  (modules + shims + project root on PYTHONPATH)"
