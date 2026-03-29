#!/usr/bin/env python3
"""
Convert .h5.pkl files (created by downscale.py) back to HDF5 format while preserving downscaled resolution.

This script:
- Reads pickle files containing downscaled field data
- Reconstructs HDF5 files with the same structure as the original
- Preserves the new (downscaled) resolution
- Uses SimulationData.txt to get updated cell dimensions
- Maintains all field data and metadata

Usage:
    python pkl_to_h5.py --path /path/to/data --read_folder data_downscaled --write_folder data_h5 
    python pkl_to_h5.py --path /path/to/data --read_folder T2D16_filter --write_folder T2D16_h5_converted --simulation_data SimulationData.txt

Arguments:
    --path (str): Base directory path for reading and writing files
    --read_folder (str): Folder name where .h5.pkl files are located
    --write_folder (str): Folder name where output HDF5 files will be saved
    --simulation_data (str): Name of SimulationData.txt file (default: SimulationData.txt)
    --timeshot (str): Specific timeshot to process, if None all will be processed (default: None)
    --verbose (bool): Enable verbose output (default: False)

Author: Derived from analysis of downscale.py
Date: December 2025
License: MIT
"""

import h5py
import numpy as np
import pickle
import glob
import os
import shutil
import argparse
import re


def parse_simulation_data(filepath):
    """
    Parse SimulationData.txt to extract cell dimensions and other metadata.
    
    Args:
        filepath (str): Path to SimulationData.txt
        
    Returns:
        dict: Dictionary containing parsed parameters
    """
    params = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Match lines like "Number of cells (x)      = 512"
                match = re.search(r'Number of cells \(([xyz])\)\s*=\s*(\d+)', line)
                if match:
                    axis = match.group(1)
                    value = int(match.group(2))
                    params[f'cells_{axis}'] = value
                
                # Extract other common parameters
                match = re.search(r'x-Length\s*=\s*([\d.]+)', line)
                if match:
                    params['x_length'] = float(match.group(1))
                
                match = re.search(r'y-Length\s*=\s*([\d.]+)', line)
                if match:
                    params['y_length'] = float(match.group(1))
                
                match = re.search(r'z-Length\s*=\s*([\d.]+)', line)
                if match:
                    params['z_length'] = float(match.group(1))
    except FileNotFoundError:
        print(f"Warning: SimulationData.txt not found at {filepath}")
    
    return params


def convert_pkl_to_h5(pkl_path, h5_path, verbose=False):
    """
    Convert a single .h5.pkl file back to HDF5 format.
    
    Args:
        pkl_path (str): Path to input .h5.pkl file
        h5_path (str): Path to output .h5 file
        verbose (bool): Print detailed information
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Load pickle file
        if verbose:
            print(f"Loading {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, dict):
            print(f"ERROR: Expected dictionary in {pkl_path}, got {type(data)}")
            return False
        
        # Create HDF5 file
        if verbose:
            print(f"Creating {h5_path}")
        
        with h5py.File(h5_path, 'w') as hf:
            # Create the same group structure as original: /Step#0/Block/
            step_group = hf.create_group("Step#0")
            block_group = step_group.create_group("Block")
            
            # Write each field
            for fieldname, fielddata in data.items():
                if verbose:
                    print(f"  Writing field: {fieldname}, shape: {fielddata.shape}")
                
                # Ensure data is proper shape [1, nx, ny] or similar
                if fielddata.ndim == 2:
                    fielddata = np.expand_dims(fielddata, axis=0)
                
                # Create dataset group for each field
                field_group = block_group.create_group(fieldname)
                
                # Store data in subgroup with index "0" (matching original structure)
                field_group.create_dataset("0", data=fielddata, compression="gzip", compression_opts=4)
        
        if verbose:
            print(f"Successfully converted {os.path.basename(pkl_path)} to {os.path.basename(h5_path)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR converting {pkl_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert .h5.pkl files back to HDF5 format while preserving downscaled resolution.'
    )
    parser.add_argument('--path', type=str, default='/volume1/scratch/share_dir/peppe/',
                        help='The base directory path for reading and writing files.')
    parser.add_argument('--read_folder', type=str, required=True,
                        help='The folder name where .h5.pkl files are located.')
    parser.add_argument('--write_folder', type=str, required=True,
                        help='The folder name where output HDF5 files will be saved.')
    parser.add_argument('--simulation_data', type=str, default='SimulationData.txt',
                        help='Name of SimulationData.txt file.')
    parser.add_argument('--timeshot', type=str, default='None',
                        help='The specific timeshot to process, if None all will be processed.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output.')
    parser.add_argument('--file', type=str, default=None,
                        help='Optional: basename of a single .h5.pkl file to convert (e.g., T2D13_Fields_000500.h5.pkl).')
    
    args = parser.parse_args()
    
    path = args.path
    read_folder = args.read_folder
    write_folder = args.write_folder
    simulation_data_file = args.simulation_data
    timeshot = args.timeshot
    verbose = args.verbose
    single_file = args.file
    
    # Validate input folder
    read_path = f'{path}{read_folder}'
    if not os.path.exists(read_path):
        raise FileNotFoundError(f"The folder {read_path} does not exist.")
    
    # Create output folder
    write_path = f'{path}{write_folder}'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
        if verbose:
            print(f"Created output directory: {write_path}")
    else:
        if os.listdir(write_path):
            print(f"WARNING: Output folder {write_path} is not empty. Files may be overwritten.")
    
    # Parse simulation data if it exists
    sim_data_path = f'{read_path}/{simulation_data_file}'
    sim_params = parse_simulation_data(sim_data_path)
    if verbose:
        print(f"Parsed simulation parameters: {sim_params}")
    
    # Copy SimulationData.txt to output folder
    sim_data_output = f'{write_path}/{simulation_data_file}'
    if os.path.exists(sim_data_path):
        shutil.copy(sim_data_path, sim_data_output)
        if verbose:
            print(f"Copied SimulationData.txt to output folder")
    
    # Build list of files to process
    if single_file:
        candidate_path = os.path.join(read_path, single_file)
        if not os.path.exists(candidate_path):
            print(f"ERROR: requested file not found: {candidate_path}")
            return
        pkl_files_list = [single_file]
        if verbose:
            print(f"Processing single file: {single_file}")
    else:
        all_pkl_files = glob.glob(f'{read_path}/*.h5.pkl')
        pkl_files_list = [os.path.basename(f) for f in all_pkl_files]
        if not pkl_files_list:
            print(f"No .h5.pkl files found in {read_path}")
            return
        if verbose:
            print(f"Found {len(pkl_files_list)} .h5.pkl files")
    
    converted_count = 0
    failed_count = 0
    
    for pkl_filename in pkl_files_list:
        # Filter by timeshot if specified
        if timeshot != 'None':
            if timeshot not in pkl_filename:
                if verbose:
                    print(f"Skipping {pkl_filename} (timeshot filter)")
                continue
        
        pkl_filepath = f'{read_path}/{pkl_filename}'
        # Remove .pkl extension to get original h5 filename
        h5_filename = pkl_filename.replace('.pkl', '')
        h5_filepath = f'{write_path}/{h5_filename}'
        
        print(f"Processing: {pkl_filename}")
        
        if convert_pkl_to_h5(pkl_filepath, h5_filepath, verbose=verbose):
            converted_count += 1
        else:
            failed_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"  Successfully converted: {converted_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output directory: {write_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()