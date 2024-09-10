"""
This script collects dictionaries from specified input folders, processes the data, and creates a DataFrame.
It can optionally filter the data based on the number of GPUs and CPUs and save a plot of the data.

Usage:
    python timing.py <input_folders> [--include-subfolders] [--gpus <gpus>] [--cpus <cpus>]

Arguments:
    input_folders: List of input folders containing loss_dict.pkl files (supports wildcards).
    --include-subfolders: Include all subfolders in the search for loss_dict.pkl files.
    --gpus: Number of GPUs to filter for plotting.
    --cpus: Number of CPUs to filter for plotting.

Examples:
    1. Basic usage without plotting:
        python timing.py /path/to/folder1 /path/to/folder2

    2. Including subfolders in the search:
        python timing.py /path/to/folder1 /path/to/folder2 --include-subfolders

    3. Plotting the data with specific GPU and CPU filters:
        python timing.py /path/to/folder1 /path/to/folder2 --gpus 2 --cpus 8

    4. Including subfolders and plotting with specific GPU and CPU filters:
        python timing.py /path/to/folder1 /path/to/folder2 --include-subfolders --gpus 2 --cpus 8

The script performs the following steps:
1. Collects all loss_dict.pkl files from the specified input folders.
2. Loads the data from these files and processes it into a DataFrame.
3. Optionally filters the DataFrame based on the specified number of GPUs and CPUs.
4. Saves the DataFrame to a CSV file named 'timing.csv'.
5. Optionally plots the data and saves the plot to a file named 'time_vs_nodes_gpus_<gpus>_cpus_<cpus>.png'.
"""

import pandas as pd
import pickle
import argparse
import os
import glob
import matplotlib.pyplot as plt

def load_dict_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)['time']
        data['filename'] = file_path.split('/')[-2]
        data['path'] = '/'.join(file_path.split('/')[:-2])
        return data

def find_all_pkl_files(root_folder):
    pkl_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == 'loss_dict.pkl':
                pkl_files.append(os.path.join(dirpath, filename))
    return pkl_files

def main(input_folders, include_subfolders, gpus, cpus):
    data = []
    expanded_folders = []
    for pattern in input_folders:
        expanded_folders.extend(glob.glob(pattern))

    for folder in expanded_folders:
        if include_subfolders:
            pkl_files = find_all_pkl_files(folder)
        else:
            pkl_files = [os.path.join(folder, 'loss_dict.pkl')]

        for file_path in pkl_files:
            if os.path.exists(file_path):
                print(f"File found: {file_path}")
                data.append(load_dict_from_pkl(file_path))
            else:
                print(f"File not found: {file_path}")

    df = pd.DataFrame(data)
    if 'train+val' in df.columns and 'train' in df.columns:
        # Ensure 'train+val' and 'train' are not lists
        df['train+val'] = df['train+val'].apply(lambda x: sum(x) if isinstance(x, list) else x)
        df['train'] = df['train'].apply(lambda x: sum(x) if isinstance(x, list) else x)
        df['val'] = df['train+val'] - df['train']
        df.drop(columns=['train+val'], inplace=True)
    
    # Split the filename column into nodes, task, and cpu
    df[['nodes', 'gpus', 'cpus']] = df['filename'].str.extract(r'(\d+)-(\d+)-(\d+)')
    
    # Reorder columns to place 'val' between 'total' and 'train'
    columns_order = ['nodes', 'gpus', 'cpus', 'total', 'val', 'train'] + [col for col in df.columns if col not in ['nodes', 'gpus', 'cpus', 'total', 'val', 'train']]
    df = df[columns_order]

    df.to_csv('timing.csv', index=False)
    print("DataFrame saved to timing.csv")
    print(df)

    if gpus is not None and cpus is not None:
        # Filter rows where gpus and cpus match the specified values
        filtered_df = df[(df['gpus'] == gpus) & (df['cpus'] == cpus)]
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_df['nodes'], filtered_df['total'], label='Total')
        plt.plot(filtered_df['nodes'], filtered_df['val'], label='Val')
        plt.plot(filtered_df['nodes'], filtered_df['train'], label='Train')
        plt.xlabel('Nodes')
        plt.ylabel('Time')
        plt.title(f'Time vs Nodes (GPUs: {gpus}, CPUs: {cpus})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'time_vs_nodes_gpus_{gpus}_cpus_{cpus}.png')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect dictionaries from input folders and create a DataFrame.")
    parser.add_argument('input_folders', nargs='+', help='List of input folders containing loss_dict.pkl files (supports wildcards)')
    parser.add_argument('--include-subfolders', action='store_true', help='Include all subfolders in the search for loss_dict.pkl files')
    parser.add_argument('--gpus', type=str, help='Number of GPUs to filter for plotting')
    parser.add_argument('--cpus', type=str, help='Number of CPUs to filter for plotting')
    args = parser.parse_args()
    main(args.input_folders, args.include_subfolders, args.gpus, args.cpus)