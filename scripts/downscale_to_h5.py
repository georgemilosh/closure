#!/usr/bin/env python3
"""
This script processes HDF5 files by applying specified filters and saves the processed data as HDF5 files.

This version saves to HDF5 format instead of pickle files, providing better:
- Cross-platform compatibility
- Long-term archival
- Partial data access
- Language interoperability

Arguments:
    --path (str): The base directory path for reading and writing files.
    --read_folder (str): The folder name where input HDF5 files are located.
    --write_folder (str): The folder name where output HDF5 files will be saved.
    --zoom (float): Downsampling factor (default: 0.25)
    --roll_x (int): Shift amount along x-axis (default: 0)
    --roll_y (int): Shift amount along y-axis (default: 0)
    --timeshot (str): Specific timeshot to process, 'None' for all (default: 'None')
    --compression (str): HDF5 compression type: 'gzip', 'lzf', or 'none' (default: 'gzip')
    --compression_level (int): Compression level for gzip (0-9, default: 4)

Variables:
    filters (list): A list of dictionaries specifying the filters to apply.

Processing:
    1. Iterate over each filename in filenames_list.
    2. Load the HDF5 file.
    3. Extract data for each field in the file.
    4. Apply specified filters to the data.
    5. Save the processed data as an HDF5 file in the write_folder.
    6. Copy the SimulationData.txt file from the read_folder to the write_folder.
    7. Modify the 'Number of cells (x)' and 'Number of cells (y)' lines in the SimulationData.txt file.

Note:
    - The script assumes that the HDF5 files have a specific structure with data stored under "/Step#0/Block/".
    - The filters are applied in the order they are listed in the filters variable.
    - Output HDF5 files maintain the same structure as input files for compatibility.

Usage:
    python downscale_to_h5.py --path /volume1/scratch/share_dir/ecsim/peppe/ --read_folder T2D16 --write_folder T2D16_filter --zoom 0.25
    python downscale_to_h5.py --path /volume1/scratch/share_dir/ecsim/peppe/ --read_folder T2D16 --write_folder T2D16_filter --zoom 0.5 --compression gzip --compression_level 6

Author: George Miloshevich
Date: December 2025
License: MIT
"""
import h5py
import numpy as np
import scipy.ndimage as nd
import glob
import os
import shutil
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Process HDF5 files and apply filters, saving to HDF5 format.')
parser.add_argument('--path', type=str, default='/volume1/scratch/share_dir/peppe/', 
                   help='The base directory path for reading and writing files.')
parser.add_argument('--read_folder', type=str, required=True, 
                   help='The folder name where input HDF5 files are located.')
parser.add_argument('--write_folder', type=str, required=True, 
                   help='The folder name where output HDF5 files will be saved.')
parser.add_argument('--zoom', type=float, default=0.25, 
                   help='The amount of zoom/downsampling (default: 0.25).')
parser.add_argument('--roll_x', type=int, default=0, 
                   help='How much to shift the x axis (default: 0).')
parser.add_argument('--roll_y', type=int, default=0, 
                   help='How much to shift the y axis (default: 0).')
parser.add_argument('--timeshot', type=str, default='None', 
                   help='The timeshot to process, if None all timeshots will be processed.')
parser.add_argument('--filename', type=str, default=None,
                   help='Specific filename to process (e.g., T2D-Fields_005000.h5). If specified, only this file will be processed.')
parser.add_argument('--compression', type=str, default='gzip', choices=['gzip', 'lzf', 'none'],
                   help='HDF5 compression type (default: gzip).')
parser.add_argument('--compression_level', type=int, default=4, choices=range(10),
                   help='Compression level for gzip (0-9, default: 4).')
parser.add_argument('--verbose', action='store_true',
                   help='Print verbose output during processing.')

args = parser.parse_args()

path = args.path
read_folder = args.read_folder
write_folder = args.write_folder
zoom = args.zoom
roll_x = args.roll_x
roll_y = args.roll_y
timeshot = args.timeshot
specific_filename = args.filename
compression = None if args.compression == 'none' else args.compression
compression_opts = args.compression_level if compression == 'gzip' else None
verbose = args.verbose

# Define filters
filters = [
    {'name': 'uniform_filter', 'size': 4, 'axes': (1, 2), 'mode': 'wrap'},
    {'name': 'zoom', 'zoom': (1, zoom, zoom), 'mode': 'grid-wrap'}
]

# Check if read_folder exists
if not os.path.exists(f'{path}{read_folder}'):
    raise FileNotFoundError(f"The folder {path}{read_folder} does not exist.")

# Check if write_folder exists, if not create it
if not os.path.exists(f'{path}{write_folder}'):
    os.makedirs(f'{path}{write_folder}')
    logger.info(f"Created output directory: {path}{write_folder}")
else:
    if os.listdir(f'{path}{write_folder}'):  # protect from overwriting existing files
        raise FileExistsError(f"The folder {path}{write_folder} is not empty.")

# Get all filenames in the read_folder
if specific_filename:
    # Process only the specified file
    if not specific_filename.endswith('.h5'):
        specific_filename += '.h5'
    
    full_path = f'{path}{read_folder}/{specific_filename}'
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Specified file {specific_filename} not found in {path}{read_folder}")
    
    filenames_list = [specific_filename]
    logger.info(f"Processing single file: {specific_filename}")
else:
    # Process all HDF5 files in the folder
    all_filenames = glob.glob(f'{path}{read_folder}/*.h5')
    filenames_list = [os.path.basename(f) for f in all_filenames]
    logger.info(f"Found {len(filenames_list)} HDF5 files to process")

if not filenames_list:
    logger.warning(f"No HDF5 files found in {path}{read_folder}")
    exit(0)

if not specific_filename:
    logger.info(f"Found {len(filenames_list)} HDF5 files to process")
logger.info(f"Filters: {filters}")
logger.info(f"Compression: {compression} (level: {compression_opts})" if compression else "Compression: None")

processed_count = 0
skipped_count = 0

for filename in filenames_list:
    if timeshot != 'None':
        if timeshot not in filename:
            skipped_count += 1
            continue
    
    read_filename = f'{path}{read_folder}/{filename}'
    write_filename = f'{path}{write_folder}/{filename}'  # Keep .h5 extension
    
    logger.info(f"Processing: {filename}")
    if verbose:
        logger.info(f"  Reading from: {read_filename}")
        logger.info(f"  Writing to: {write_filename}")
    
    # Load and process the file
    data = {}
    original_shape = None
    
    with h5py.File(read_filename, 'r') as n:
        if "/Step#0/Block/" not in n:
            logger.warning(f"Block object not found in {read_filename}, skipping...")
            skipped_count += 1
            continue
        
        # Iterate over each field
        for fieldname in n[f"/Step#0/Block/"].keys():
            field_data = n[f"/Step#0/Block/{fieldname}/0"][:, :-1, :-1]  # Remove extra point in last dimension
            
            if original_shape is None:
                original_shape = field_data.shape
                if verbose:
                    logger.info(f"  Original shape: {original_shape}")
            
            # Apply filters
            if filters is not None:
                if not isinstance(filters, list):
                    filters = [filters]
                
                for filteri in filters:
                    if verbose:
                        logger.info(f"  Filtering {fieldname} with {filteri['name']}")
                    
                    filters_copy = filteri.copy()
                    filters_name = filters_copy.pop("name", None)
                    filters_object = getattr(nd, filters_name)
                    filter_kwargs = filters_copy
                    
                    # Convert lists to tuples for filter arguments
                    for key, kwarg in filter_kwargs.items():
                        if isinstance(kwarg, list):
                            filter_kwargs[key] = tuple(kwarg)
                    
                    field_data = filters_object(field_data, **filter_kwargs)
                    
                    if verbose:
                        logger.info(f"    Result shape: {field_data.shape}")
            
            # Pad and roll
            field_data = np.pad(field_data, pad_width=((0, 0), (0, 1), (0, 1)), mode='wrap')[0:1, ...]
            field_data = np.roll(field_data, (roll_x, roll_y), axis=(1, 2))
            
            data[fieldname] = field_data
    
    # Write to HDF5 file
    with h5py.File(write_filename, 'w') as out_file:
        # Create the same group structure as input
        step_group = out_file.create_group("/Step#0")
        block_group = step_group.create_group("Block")
        
        for fieldname, field_data in data.items():
            field_group = block_group.create_group(fieldname)
            
            # Create dataset with compression
            field_group.create_dataset(
                "0",
                data=field_data,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True if compression else False  # Shuffle improves compression
            )
            
            if verbose:
                logger.info(f"  Saved {fieldname} with shape {field_data.shape}")
    
    processed_count += 1
    
    # Log file sizes
    input_size = os.path.getsize(read_filename) / (1024**2)  # MB
    output_size = os.path.getsize(write_filename) / (1024**2)  # MB
    compression_ratio = (1 - output_size / input_size) * 100
    
    logger.info(f"  File size: {input_size:.2f} MB → {output_size:.2f} MB ({compression_ratio:+.1f}%)")

# Copy and modify SimulationData.txt
simulation_data_path = f'{path}{write_folder}/SimulationData.txt'
shutil.copy(f'{path}{read_folder}/SimulationData.txt', simulation_data_path)

# Read the file
with open(simulation_data_path, 'r') as file:
    lines = file.readlines()

# Calculate new grid size based on zoom
# Assuming original size is known or extracted from first processed file
if original_shape is not None:
    new_nx = int(original_shape[2] * zoom) + 1  # +1 for the padded point
    new_ny = int(original_shape[1] * zoom) + 1
    
    logger.info(f"Updating SimulationData.txt: grid size to {new_nx}×{new_ny}")
    
    # Modify the specific lines
    for i, line in enumerate(lines):
        if 'Number of cells (x)' in line:
            lines[i] = f'Number of cells (x)      = {new_nx}\n'
        if 'Number of cells (y)' in line:
            lines[i] = f'Number of cells (y)      = {new_ny}\n'
    
    # Write the modified lines back to the file
    with open(simulation_data_path, 'w') as file:
        file.writelines(lines)
else:
    logger.warning("Could not determine grid size, SimulationData.txt not modified")

# Summary
logger.info(f"\n{'='*60}")
logger.info("PROCESSING SUMMARY")
logger.info(f"{'='*60}")
logger.info(f"Files processed: {processed_count}")
logger.info(f"Files skipped: {skipped_count}")
logger.info(f"Output directory: {path}{write_folder}")
logger.info(f"✓ Processing complete!")
