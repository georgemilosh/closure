#!/bin/bash
#SBATCH --job-name=downscale_job
#SBATCH --account=2025_065
#SBATCH --error=down_%x_%j.err
#SBATCH --output=down_%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
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


# Define the path and write folder
PATH_TO_DATA=$1 #"/dodrio/scratch/projects/2025_065/georgem/2024_109/ecsim/peppe/"
READ_FOLDER=$2
WRITE_FOLDER=$3 #"${READ_FOLDER}_filter1"
ZOOM=$4 

# Run the downscale.py script with the provided arguments
python downscale.py --path $PATH_TO_DATA --read_folder $READ_FOLDER --write_folder $WRITE_FOLDER --zoom $ZOOM
