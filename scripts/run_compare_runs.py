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
    python scripts/run_compare_runs.py --work_dir /path/to/trainer/folder
"""
import os
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
import pickle
import argparse
import json
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from closure import trainers as tr

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




try:
    loss_dicts = {}

    # Get the list of all folders in the trainer directory
    folders = [f for f in os.listdir(args.work_dir) if os.path.isdir(os.path.join(args.work_dir, f))]

    line_styles = itertools.cycle(['-', '--', '-.', ':'])

    plt.figure()
    config_runs = {}
    for folder in folders:
        file_path = os.path.join(args.work_dir, folder, 'loss_dict.pkl')
        config_path = os.path.join(args.work_dir, folder, 'config.json')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                loss_dicts[folder] = {'loss': pickle.load(f)}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_runs[folder] = json.load(f)
            plt.plot(loss_dicts[folder]['loss']['val_loss']['criterion'], label=folder, linestyle=next(line_styles))
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(args.work_dir)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 0.6])
    #plt.xlim([0, 150])
    plt.tight_layout()
    print(f"Saving in: {args.work_dir}/img/loss_comparison.png")
    os.makedirs(os.path.join(args.work_dir, 'img'), exist_ok=True)
    plt.savefig(f'{args.work_dir}/img/loss_comparison.png', dpi=300, bbox_inches='tight')

except Exception as e:
    print(f"An error occurred while plotting: {e}")

try:
    with open(os.path.join(args.work_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    train = pd.read_csv(config['dataset_kwargs']['train_sample'])
    val = pd.read_csv(config['dataset_kwargs']['val_sample'])
    test = pd.read_csv(config['dataset_kwargs']['test_sample'])
    # Extract cycle numbers as integers from filenames
    train['cycle'] = train['filenames'].str.extract(r'_(\d+)\.h5')[0].astype(int)
    val['cycle'] = val['filenames'].str.extract(r'_(\d+)\.h5')[0].astype(int)
    test['cycle'] = test['filenames'].str.extract(r'_(\d+)\.h5')[0].astype(int)
    # Visualize data split by cycle numbers
    fig, ax = plt.subplots(figsize=(12, 4))

    # Create data for plotting
    train_cycles = sorted(train['cycle'].unique())
    val_cycles = sorted(val['cycle'].unique())
    test_cycles = sorted(test['cycle'].unique())

    # Plot each split with different markers and colors
    ax.scatter(train_cycles, [1]*len(train_cycles), s=200, marker='o', label='Train', alpha=0.7, color='blue')
    ax.scatter(val_cycles, [2]*len(val_cycles), s=200, marker='s', label='Validation', alpha=0.7, color='orange')
    ax.scatter(test_cycles, [3]*len(test_cycles), s=200, marker='^', label='Test', alpha=0.7, color='green')

    # Add cycle numbers as text labels
    for cycle in train_cycles:
        ax.text(cycle, 1, str(cycle), ha='center', va='bottom', fontsize=9)
    for cycle in val_cycles:
        ax.text(cycle, 2, str(cycle), ha='center', va='bottom', fontsize=9)
    for cycle in test_cycles:
        ax.text(cycle, 3, str(cycle), ha='center', va='bottom', fontsize=9)

    # Formatting
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Train', 'Validation', 'Test'])
    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('Data Split', fontsize=12)
    ax.set_title('Data Split Distribution Across Cycles', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right')

    # Set x-axis limits with some padding
    all_cycles = train_cycles + val_cycles + test_cycles
    ax.set_xlim([min(all_cycles) - 200, max(all_cycles) + 200])

    plt.tight_layout()
    print(f'Saving in: {args.work_dir}/img/data_split_distribution.png')
    os.makedirs(os.path.join(args.work_dir, 'img'), exist_ok=True)
    plt.savefig(f'{args.work_dir}/img/data_split_distribution.png', dpi=300, bbox_inches='tight')

    # Print summary statistics
    print(f"\nData Split Summary:")
    print(f"Train: {len(train_cycles)} cycles - {train_cycles}")
    print(f"Val:   {len(val_cycles)} cycles - {val_cycles}")
    print(f"Test:  {len(test_cycles)} cycles - {test_cycles}")
    print(f"Total: {len(all_cycles)} cycles")

except Exception as e:
    print(f"An error occurred while visualizing data split: {e}")

try:
    output_data = pickle.load(open(os.path.join(args.work_dir, 'evaluation.pkl'), 'rb'))
    loss = output_data['loss']
    loss2 = loss.copy()
    print(loss2.to_string(index=False))

    selected_columns = loss2.columns[~loss2.columns.str.endswith('_r2')]
    selected_data = loss2[selected_columns]
    columns = selected_data.columns[loss2.columns.get_loc('total_MSELoss'):]

    x = list(loss2['run'])
    df = pd.DataFrame({key: value for key, value in zip(columns, loss2[columns].values.T.tolist())}, index=list(x), 
                    dtype=float)
    

    # Add a horizontal line for the model with the lowest total_MSELoss
    min_total_MSELoss = df['total_MSELoss'].min()
    best_model = df['total_MSELoss'].idxmin()

    columns_r2 = loss2.columns[loss2.columns.str.endswith('_r2')]
    x = list(loss2['run'])

    df = pd.DataFrame({key: value for key, value in zip(columns_r2, loss2[columns_r2].values.T.tolist())}, index=list(x), 
                    dtype=float)
    
    fig = plt.figure(figsize=(15, 5)) 
    ax = fig.gca()
    df.plot.bar(rot=90, ax=ax)
    plt.ylim([-.1, 1])
    plt.ylabel(r'$R^2$ score')
    plt.xlabel('Experiments')

    # Add a horizontal line for the model with the highest total_r2
    max_total_r2 = df['total_r2'].max()
    best_model_r2 = df['total_r2'].idxmax()
    ax.axhline(max_total_r2, color='black', linestyle='--', linewidth=2, label=f'Highest total_r2: {max_total_r2:.2f}')

    # Annotate the best model with bold text
    ax.text(df.index.get_loc(best_model_r2), max_total_r2, f'{max_total_r2:.2f}', ha='center', va='bottom', color='black', weight='bold')

    plt.legend()
    plt.title(args.work_dir)
    plt.tight_layout()
    print(f'Saving in: {args.work_dir}/img/r2_comparison.png')
    os.makedirs(os.path.join(args.work_dir, 'img'), exist_ok=True)
    plt.savefig(f'{args.work_dir}/img/r2_comparison.png', dpi=300, bbox_inches='tight')
except Exception as e:
    print(f"An error occurred while plotting metrics: {e}")




trainer = tr.Trainer(work_dir=trainer_folder, mode_test=True,
                          log_name='reading.log', log_level='INFO')


for run in runs:
    if os.path.isdir(os.path.join(trainer_folder, run)) and os.path.exists(os.path.join(trainer_folder, run, 'config.json')):
        trainer.load_run(run)
        ground_truth_scaled, prediction_scaled = tr.ut.transform_targets(trainer)
        for request_target in trainer.test_dataset.request_targets:
            print(f"Comparing {request_target} for run {run}")
            tr.ut.graph_pred_targets(trainer, request_target, ground_truth_scaled, prediction_scaled)




