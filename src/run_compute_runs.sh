#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --account=2025_065
#SBATCH --error=down_%x_%j.err
#SBATCH --output=down_%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem 238G
#SBATCH --time=06:00:00

# Load necessary modules
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a
#module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
#module load Optuna/3.6.1-foss-2023b
#module load SciPy-bundle/2022.05-foss-2022a
#module load Optuna/3.6.1-foss-2023b
module load matplotlib/3.7.2-gfbf-2023a

# Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

# Get the argument for the read folder
WORK_DIR=$1
OUTPUT_DIR=$(pwd)/
WORK_DIR="${OUTPUT_DIR}${WORK_DIR}"

REPO_DIR="/dodrio/scratch/projects/2025_065/georgem/2024_109/closure/"

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340 #$(get_free_port)  ### the original script had $(get_free_port) but it was not recognized
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
#module load Optuna/3.6.1-foss-2023b
#module load h5py/3.9.0-foss-2023a
#module load SciPy-bundle/2022.05-foss-2022a
#module load Optuna/3.6.1-foss-2023b
module load matplotlib/3.7.2-gfbf-2023a
#source ~/.bashrc
#mamba activate torch

# Log the job script path
echo "The script you are running has:" >> ~/job_logs/job_${SLURM_JOB_ID}.log
echo "basename: [$(basename "$0")]"  >> ~/job_logs/job_${SLURM_JOB_ID}.log
echo "dirname : [$(dirname "$0")]" >> ~/job_logs/job_${SLURM_JOB_ID}.log
echo "pwd     : [$(pwd)]" >> ~/job_logs/job_${SLURM_JOB_ID}.log

#source ~/.bashrc
#conda activate /dodrio/scratch/projects/starting_2023_110/miniforge3/envs/torch
# Set the number of threads.  #SBATCH --partition=gpu_p100
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the program
cd $REPO_DIR

# Run the downscale.py script with the provided arguments
python -m src.run_compare_runs.py --work_dir $WORK_DIR