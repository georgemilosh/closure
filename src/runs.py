"""
runs.py
Enhanced runner for multiple training experiments.
Imports and uses trainers.py methods without modifying the original file.
This script provides a flexible command-line interface for running single or multiple training experiments,
supporting configuration via command-line arguments, JSON files, or convenience sweep/grid search options.
It imports and utilizes methods from `trainers.py` without modifying the original file.
Features:
- Supports distributed and single-node training environments.
- Allows updating nested configuration keys via command-line.
- Supports learning rate and weight decay sweeps, as well as arbitrary grid searches.
- Can execute multiple runs sequentially, each with its own configuration.
- Handles configuration via command-line, JSON string, or external JSON file.
- Integrates with SLURM and PyTorch Distributed for multi-GPU training.
- Provides logging and timing options.
Usage:
    # Option 1: Single run with complex config (your original example)
        srun python -m src.runs --force \
        --config work_dir="$OUTPUT_DIR" \
        --config run="shallow" \
        --config 'model_kwargs.channels=[10,64,16,6]' \
        --config 'model_kwargs.activations=["ReLU","ReLU",null]' \
        --config 'model_kwargs.kernels=[3,5,5]' \
        --config model_kwargs.optimizer_kwargs.lr=0.001

    # Option 2: Learning rate sweep (uncomment to use)
        srun python -m src.runs --force \
          --config work_dir="$OUTPUT_DIR" \
          --lr-sweep 0.0001 0.0005 0.001 0.005

    # Option 3: Multiple custom runs (uncomment to use)
        MULTIPLE_RUNS='[
          {
            "run": "shallow",
            "config": {
              "model_kwargs.channels": [10, 64, 16, 6],
              "model_kwargs.activations": ["ReLU", "ReLU", null],
               "model_kwargs.kernels": [3, 5, 5],
               "model_kwargs.optimizer_kwargs.lr": 0.001
             }
           },
           {
             "run": "deep",
             "config": {
               "model_kwargs.channels": [10, 32, 64, 32, 16, 6],
               "model_kwargs.activations": ["ReLU", "ReLU", "ReLU", "ReLU", null],
               "model_kwargs.kernels": [3, 3, 5, 5, 5],
               "model_kwargs.optimizer_kwargs.lr": 0.0005
             }
           }
         ]'
        srun python -m src.runs --force --config work_dir="$OUTPUT_DIR" --runs "$MULTIPLE_RUNS"
Arguments:
    --force              : Force the training to start even if the run exists.
    --timing_name        : Name of the timing CSV file.
    --config             : Update nested config keys (key.subkey=value).
    --runs               : JSON string defining multiple runs and their configs.
    --config-file        : Path to JSON file containing run configurations.
    --lr-sweep           : Learning rate values for automatic sweep.
    --wd-sweep           : Weight decay values for automatic sweep.
    --grid-search        : JSON string defining parameter grid for grid search.
Environment Variables (for distributed training):
    WORLD_SIZE, SLURM_PROCID, SLURM_GPUS_ON_NODE, SLURM_CPUS_PER_TASK
Requires:
    - Python 3
    - PyTorch
    - SLURM (for distributed runs)
    - trainers.py, utils.py, logconfig.py in the same package/module

"""

import argparse
import copy
import json
import logging
import os
from socket import gethostname

import torch
import torch.distributed as dist

from . import trainers
from . import utils as ut
from . import logconfig


def parse_config_value(value):
    """Parse config value, trying JSON first, then fallback to string."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def create_lr_runs(lr_values, base_config=None):
    """Create runs with different learning rates."""
    runs = []
    for lr in lr_values:
        run_config = {"model_kwargs.optimizer_kwargs.lr": lr}
        if base_config:
            run_config.update(base_config)
        runs.append({
            "run": f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+"),
            "config": run_config
        })
    return runs


def create_wd_runs(wd_values, base_config=None):
    """Create runs with different weight decay values."""
    runs = []
    for wd in wd_values:
        run_config = {"model_kwargs.optimizer_kwargs.weight_decay": wd}
        if base_config:
            run_config.update(base_config)
        runs.append({
            "run": f"wd{wd:.0e}".replace("e-0", "e-").replace("e+0", "e+"),
            "config": run_config
        })
    return runs


def create_grid_search(param_grid):
    """Create runs for grid search over multiple parameters."""
    import itertools
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    runs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        run_name = "_".join([f"{k.split('.')[-1]}{v:.0e}".replace("e-0", "e-").replace("e+0", "e+") 
                            for k, v in config.items()])
        runs.append({
            "run": run_name,
            "config": config
        })
    return runs


def main():
    logconfig.setup_logging(console_level=logging.INFO)
    
    # Training settings
    parser = argparse.ArgumentParser(description='Enhanced Multi-Run Training')
    parser.add_argument('--force', action=argparse.BooleanOptionalAction,
                        help='Force the training to start even if the run exists')
    parser.add_argument('--timing_name', type=str, default=False,
                        help='Name of the timing CSV file. If not provided no timing file will be created')
    parser.add_argument('--config', action='append', default=None, 
                        help="Update nested config keys. Use 'key.subkey=value' format. For arrays/objects, use JSON syntax.")
    parser.add_argument('--runs', type=str, default=None, 
                        help="JSON string defining multiple runs with their configs. Format: '[{\"run\": \"name1\", \"config\": {\"key\": \"value\"}}, ...]'")
    parser.add_argument('--config-file', type=str, default=None,
                        help="Path to JSON file containing run configurations")
    
    # Convenience methods for common sweeps
    parser.add_argument('--lr-sweep', nargs='+', type=float, 
                        help='Learning rate values for automatic sweep')
    parser.add_argument('--wd-sweep', nargs='+', type=float, 
                        help='Weight decay values for automatic sweep')
    parser.add_argument('--grid-search', type=str, default=None,
                        help='JSON string defining parameter grid for grid search')

    args = parser.parse_args()

    # Extract work_dir from config
    work_dir = None
    if args.config:
        for update in args.config:
            key, value = update.split("=", 1)
            if key == "work_dir":
                work_dir = value
                break

    if work_dir is None:
        raise ValueError("work_dir must be specified in the --config argument")

    # Check if running in a distributed environment
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)

        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
        print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}, \
                    gpus_per_node: {gpus_per_node}, num_workers: {num_workers}")
    else:
        # Single-node setup
        world_size = 1
        rank = 0
        local_rank = 0
        gpus_per_node = min(torch.cuda.device_count(),1) if torch.cuda.is_available() else 0
        num_workers = os.cpu_count() if os.cpu_count() < 32 else 32
        print(f"Running on a single node with {gpus_per_node} GPUs and {num_workers} CPU cores.")

    print(f"Creating Trainer object with work_dir={work_dir}")
    trainer = trainers.Trainer(work_dir=work_dir, world_size=world_size, rank=rank, gpus_per_node=gpus_per_node, 
                              local_rank=local_rank, num_workers=num_workers, force=args.force, timing_name=args.timing_name)

    # Determine which type of run to execute
    runs_config = None
    
    # Handle convenience methods
    if args.lr_sweep:
        runs_config = create_lr_runs(args.lr_sweep)
        print(f"Generated learning rate sweep: {[run['run'] for run in runs_config]}")
    
    elif args.wd_sweep:
        runs_config = create_wd_runs(args.wd_sweep)
        print(f"Generated weight decay sweep: {[run['run'] for run in runs_config]}")
    
    elif args.grid_search:
        param_grid = json.loads(args.grid_search)
        runs_config = create_grid_search(param_grid)
        print(f"Generated grid search with {len(runs_config)} combinations")
    
    # Handle config file
    elif args.config_file is not None:
        try:
            with open(args.config_file, 'r') as f:
                runs_config = json.load(f)
            
            if not isinstance(runs_config, list):
                raise ValueError("Config file must contain a JSON list of runs")
            
            print(f"Loaded {len(runs_config)} runs from {args.config_file}")
        except Exception as e:
            raise ValueError(f"Error processing config file: {e}")

    # Handle command line JSON
    elif args.runs is not None:
        try:
            runs_config = json.loads(args.runs)
            if not isinstance(runs_config, list):
                raise ValueError("--runs must be a JSON list")
            
            print(f"Loaded {len(runs_config)} runs from command line")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --runs argument: {e}")

    # Execute multiple runs
    if runs_config is not None:
        print(f"Running {len(runs_config)} sequential runs...")
        for i, run_spec in enumerate(runs_config):
            if not isinstance(run_spec, dict) or 'run' not in run_spec:
                raise ValueError(f"Each run must be a dict with 'run' key. Got: {run_spec}")
            
            config = copy.deepcopy(trainer.config)
            
            # Apply base config updates from --config
            if args.config is not None:
                for update in args.config:
                    key, value = update.split("=", 1)
                    if key != "work_dir":
                        parsed_value = parse_config_value(value)
                        ut.set_nested_config(config, key, parsed_value)
            
            # Set run name
            config['run'] = run_spec['run']
            
            # Apply run-specific config updates
            if 'config' in run_spec:
                for key, value in run_spec['config'].items():
                    ut.set_nested_config(config, key, value)
            
            print(f"\n{'='*60}")
            print(f"Starting run {i+1}/{len(runs_config)}: {run_spec['run']}")
            print(f"{'='*60}")
            for key, value in run_spec.get('config', {}).items():
                print(f"  {key} = {value}")
            print()
            
            trainer.fit(config=config)
            print(f"Completed run {i+1}/{len(runs_config)}: {run_spec['run']}")
    
    # Handle single run with multiple config changes
    elif args.config is not None:
        config = copy.deepcopy(trainer.config)
        for update in args.config:
            key, value = update.split("=", 1)
            if key != "work_dir":
                parsed_value = parse_config_value(value)
                ut.set_nested_config(config, key, parsed_value)
                print(f"Setting {key} to {parsed_value}")
        trainer.fit(config=config)
    else:
        trainer.fit()

    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"Error destroying process group, possibly because it didn't exist?")
        print(e)


if __name__ == '__main__':
    main()