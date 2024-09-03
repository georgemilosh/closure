#!/bin/bash
#SBATCH --account=lp_cmpa
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --job-name=job1-2-8-     # create a short name for your job
#SBATCH --nodes=1               # node count 
#SBATCH --gres=gpu:2             # number of allocated gpus per node ### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --output=%x%j.out
#SBATCH --error=%x%j.err

OUTPUT_DIR="/lustre1/project/stg_00032/georgem/closure/models/dev/dev21/"

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340 #$(get_free_port)  ### the original script had $(get_free_port) but it was not recognized
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


#module load cluster/genius/gpu_p100
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source ~/.bashrc
mamba activate huggingface_hub

srun python -m src.trainers --work_dir "$OUTPUT_DIR" --force --run "1-2-8"
