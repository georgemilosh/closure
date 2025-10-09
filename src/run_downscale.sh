#!/bin/bash
#SBATCH --job-name=downscale_job
#SBATCH --account=2025_065
#SBATCH --output=downscale_job_%j.log
#SBATCH --error=downscale_job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=03:00:00

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
READ_FOLDER=$1

# Define the path and write folder
PATH_TO_DATA="/dodrio/scratch/projects/2025_012/georgem/ecsim/peppe/"
WRITE_FOLDER="${READ_FOLDER}_filter2"

# Run the downscale.py script with the provided arguments
python /dodrio/scratch/projects/2025_065/georgem/2024_109/closure/src/downscale.py --path $PATH_TO_DATA --read_folder $READ_FOLDER --write_folder $WRITE_FOLDER
