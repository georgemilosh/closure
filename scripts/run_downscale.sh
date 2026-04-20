#!/bin/bash
#SBATCH --job-name=downscale_job
#SBATCH --account=2025_112
#SBATCH --error=logs/down_%x_%j.err
#SBATCH --output=logs/down_%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=06:00:00

if [ "$#" -lt 4 ] || [ "$#" -gt 7 ]; then
	echo "Usage: sbatch run_downscale.sh <PATH_TO_DATA> <READ_FOLDER> <WRITE_FOLDER> <ZOOM> [OUTPUT_FORMAT] [NO_FILTERS] [OUTPUT_DTYPE]"
	echo "Example: sbatch run_downscale.sh /dodrio/scratch/projects/2025_112/nathan/ Le2DHGEM_RunID_5 Le2DHGEM_RunID_5_ds 0.25 npz 1 float32"
	exit 1
fi

# Ensure log folder exists for SLURM output/error files.
mkdir -p logs

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
OUTPUT_FORMAT=${5:-pkl}
NO_FILTERS=${6:-0}
OUTPUT_DTYPE=${7:-float64}

EXTRA_ARGS=()
if [ "$NO_FILTERS" = "1" ] || [ "$NO_FILTERS" = "true" ] || [ "$NO_FILTERS" = "True" ]; then
	EXTRA_ARGS+=(--no_filters)
fi

# Run the downscale.py script with the provided arguments
python -u downscale.py --path "$PATH_TO_DATA" --read_folder "$READ_FOLDER" --write_folder "$WRITE_FOLDER" --zoom "$ZOOM" --output_format "$OUTPUT_FORMAT" --output_dtype "$OUTPUT_DTYPE" "${EXTRA_ARGS[@]}"
