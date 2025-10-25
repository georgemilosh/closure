
import os
"""
This script compares the results of multiple training runs stored in a specified directory.

The script performs the following steps:
1. Parses command-line arguments to get the path to the trainer folder.
2. Lists all subdirectories (runs) in the specified trainer folder.
3. For each run, loads the loss dictionary from 'loss_dict.pkl' and the configuration from 'config.json'.
4. Uses the loaded data to compare the runs using a specified metric.
5. Saves the comparison results and the loaded data to 'evaluation.pkl' in the trainer folder.

Arguments:
    --work_dir (str): Path to the trainer folder containing the runs.

Outputs:
    evaluation.pkl: A pickle file containing the loss dictionaries, configuration data, and comparison results.

Example usage:
    python -m src.run_compare_runs.py --work_dir /path/to/trainer/folder
"""
import pickle
import argparse
import json
from . import trainers as tr
loss_dicts = {}

# Get the list of all folders in the trainer directory
parser = argparse.ArgumentParser(description='Run comparison of runs.')
parser.add_argument('--work_dir', type=str, required=True, help='Path to the trainer folder')
args = parser.parse_args()

trainer_folder = args.work_dir
print(f'Looking for runs in {trainer_folder}')
runs = [f for f in os.listdir(trainer_folder) if os.path.isdir(os.path.join(trainer_folder, f))]
print(f"Found {len(runs)} runs")

config_runs = {}
for run in runs:
    file_path = os.path.join(trainer_folder, run, 'loss_dict.pkl')
    config_path = os.path.join(trainer_folder, run, 'config.json')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            loss_dicts[run] = pickle.load(f)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_runs[run] = json.load(f)

print(f"parsed configs and losses")


folder_list_with_path = [trainer_folder] * len(loss_dicts)

print(f"work_dirs: {folder_list_with_path}")
print(f"runs: {loss_dicts.keys()}")

loss = tr.ut.compare_runs(work_dirs=folder_list_with_path,
                          runs=loss_dicts.keys(), 
                          mode_test=True, metric=['r2'], rescale=True, renorm=True, verbose=True, 
                          log_name='reading.log', log_level='INFO')
print(f"computed metrics")

output_data = {
    'loss_dicts': loss_dicts,
    'config_runs': config_runs,
    'loss': loss
}

output_file = os.path.join(trainer_folder, 'evaluation.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(output_data, f)
print(f"wrote metrics to the {output_file}")