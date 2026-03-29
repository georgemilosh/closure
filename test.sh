#! /bin/bash


OUTPUT_DIR="/volume1/scratch/georgem/closure/models/peppe/sigma0_haydn/FCNN/P/"
REPO_DIR="/dodrio/scratch/projects/2024_109/closure/"


# Option 1: Single run with complex config (your original example)
#srun python -m src.runs --force \
#  --config work_dir="$OUTPUT_DIR" \
#  --config run="shallow" \
#  --config 'model_kwargs.channels=[10,64,16,6]' \
#  --config 'model_kwargs.activations=["ReLU","ReLU",null]' \
#  --config 'model_kwargs.kernels=[3,5,5]' \
#  --config model_kwargs.optimizer_kwargs.lr=0.001

# Option 2: Learning rate sweep (uncomment to use)
# srun python -m src.runs --force \
#   --config work_dir="$OUTPUT_DIR" \
#   --lr-sweep 0.0001 0.0005 0.001 0.005

# Option 3: Multiple custom runs (uncomment to use)
 MULTIPLE_RUNS='[
   {
     "run": "shallow",
     "config": {
       "model_kwargs.channels": [10, 64, 16, 6],
       "model_kwargs.activations": ["ReLU", "ReLU", null],
       "model_kwargs.kernels": [3, 5, 5],
       "model_kwargs.optimizer_kwargs.lr": 0.001,
       "model_kwargs.scheduler_kwargs.epochs": 2,
       "model_kwargs.batch_norms": [true,true,true,false]
     }
   },
   {
     "run": "deep",
     "config": {
       "model_kwargs.channels": [10, 32, 64, 32, 16, 6],
       "model_kwargs.activations": ["ReLU", "ReLU", "ReLU", "ReLU", "ReLU", null],
       "model_kwargs.kernels": [3, 3, 5, 5, 5, 3],
       "model_kwargs.optimizer_kwargs.lr": 0.0005,
       "model_kwargs.scheduler_kwargs.epochs": 2,
       "model_kwargs.batch_norms": [true,true,true,true,true,false]
     }
   }
 ]'
 python -m src.runs --force --config work_dir="$OUTPUT_DIR" --runs "$MULTIPLE_RUNS"
