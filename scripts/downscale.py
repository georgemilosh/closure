

"""
This script processes HDF5 files by applying specified filters and saves the processed data as pkl or npz files.

Arguments:
    --path (str): The base directory path for reading and writing files.
    --read_folder (str): The folder name where input HDF5 files are located.
    --write_folder (str): The folder name where output files will be saved.

Variables:
    filters (list): A list of dictionaries specifying the filters to apply.

Processing:
    1. Iterate over each filename in filenames_list.
    2. Load the HDF5 file.
    3. Extract data for each field in the file.
    4. Apply specified filters to the data.
    5. Save the processed data as a pkl or npz file in the write_folder.
    6. Copy auxiliary metadata files (SimulationData.txt, optional *.inp, optional ConservedQuantities.txt) from the read_folder to the write_folder.
    7. Modify the 'Number of cells (x)' and 'Number of cells (y)' lines in the SimulationData.txt file.

Note:
    - The script assumes that the HDF5 files have a specific structure with data stored under "/Step#0/Block/".
    - The filters are applied in the order they are listed in the filters variable.
    - The script prints verbose output if the verbose variable is set to True.

Usage:
    python downscale.py --path /dodrio/scratch/projects/2024_109/ecsim/peppe/ --read_folder T2D16 --write_folder T2D16_filter --zoom 0.25
"""
import h5py
import numpy as np
import scipy.ndimage as nd
import pickle
import glob
import os
import shutil
import argparse
import re

parser = argparse.ArgumentParser(description='Process HDF5 files and apply filters.')
parser.add_argument('--path', type=str, default='/volume1/scratch/share_dir/peppe/', help='The base directory path for reading and writing files.')
parser.add_argument('--read_folder', default='data', type=str, required=True, help='The folder name where input HDF5 files are located.')
parser.add_argument('--write_folder', default='data_filter', type=str, required=True, help='The folder name where output pickle files will be saved.')
parser.add_argument('--zoom', default='0.25', type=str, required=False, help='the amount of zoom.')
parser.add_argument('--roll_x', default='0', type=str, required=False, help='How much we would like to shift the x axis.')
parser.add_argument('--roll_y', default='0', type=str, required=False, help='How much we would like to shift the y axis.')
parser.add_argument('--timeshot', default='None', type=str, required=False, help='The time shot we would like to process, if None all timeshots will be processed.')
parser.add_argument('--output_format', default='pkl', choices=['pkl', 'npz'], required=False, help='Output format for downscaled field files.')
parser.add_argument('--no_filters', action='store_true', help='Disable filtering/zoom processing (format conversion mode).')
parser.add_argument('--output_dtype', default='float64', choices=['float32', 'float64'], required=False, help='Output array dtype for saved fields.')
parser.add_argument('--resume', action='store_true', help='Allow writing into a non-empty output folder and skip files already converted.')
parser.add_argument('--skip_bad_files', action='store_true', help='Skip unreadable/corrupted input files instead of aborting the job.')
args = parser.parse_args()

path = args.path
read_folder = args.read_folder
write_folder = args.write_folder
zoom = float(args.zoom)
roll_x = int(args.roll_x)
roll_y = int(args.roll_y)
timeshot = args.timeshot
output_format = args.output_format
no_filters = args.no_filters
output_dtype = np.float32 if args.output_dtype == 'float32' else np.float64
resume = args.resume
skip_bad_files = args.skip_bad_files


def get_output_basename(input_filename: str, fmt: str) -> str:
    if fmt == 'npz':
        return f"{os.path.splitext(input_filename)[0]}.npz"
    return f"{input_filename}.pkl"


filters = None
if not no_filters:
    filters = [
        {'name': 'uniform_filter', 'size': int(1/zoom), 'axes': (1, 2), 'mode': 'wrap'},
        {'name': 'zoom', 'zoom': (1, zoom, zoom), 'mode': 'grid-wrap'},
    ]
else:
    print('Running with --no_filters: skipping filter/zoom processing.', flush=True)
if not os.path.exists(f'{path}{read_folder}'): # Check if read_folder exists
    raise FileNotFoundError(f"The folder {path}{read_folder} does not exist.")

if not os.path.exists(f'{path}{write_folder}'): # Check if write_folder exists, if not create it
    os.makedirs(f'{path}{write_folder}')
else:
    if os.listdir(f'{path}{write_folder}') and not resume: # protect from overwriting existing files
        raise FileExistsError(f"The folder {path}{write_folder} is not empty.")
    if os.listdir(f'{path}{write_folder}') and resume:
        print(f"Resume mode enabled: existing files in {path}{write_folder} will be kept.", flush=True)

# Get all filenames in the read_folder
all_filenames = sorted(glob.glob(f'{path}{read_folder}/*.h5'))
filenames_list = [os.path.basename(f) for f in all_filenames]
print(f"Found {len(filenames_list)} input .h5 files in {path}{read_folder}", flush=True)
if len(filenames_list) == 0:
    raise FileNotFoundError(
        f"No .h5 files found in {path}{read_folder}. "
        "This script expects ECSIM-style *-Fields_*.h5 inputs; "
        "iPiC3D proc*.hdf inputs require a different conversion step first."
    )

skipped_existing = 0
skipped_bad = []
processed = 0

for filename in filenames_list:
    if timeshot != 'None':
        if timeshot not in filename:
            continue
    read_filename = f'{path}{read_folder}/{filename}'
    write_basename = get_output_basename(filename, output_format)
    write_filename = f'{path}{write_folder}/{write_basename}'
    if resume and os.path.exists(write_filename):
        skipped_existing += 1
        continue
    print(f"Processing {read_filename}", flush=True)
    print(f"Writing to {write_filename}", flush=True)
    # Load the file
    verbose=False
    data = {}
    try:
        with h5py.File(read_filename, 'r') as n:
            print(f"Working on {filename}", flush=True)
            if "/Step#0/Block/" in n:
                # Iterate over each time step
                for fieldname in n[f"/Step#0/Block/"].keys():
                    data[fieldname] = n[f"/Step#0/Block/{fieldname}/0"][:,:-1,:-1] # there is extra point in the last dimension
                    if filters is not None:
                        if not isinstance(filters, list):
                            filters = [filters]
                        for filteri in filters: # apply all filters in succession
                            if verbose:
                                print(f"Filtering {fieldname} from {filename} with {filteri['name']}")
                            filters_copy = filteri.copy()
                            filters_name = filters_copy.pop("name", None)
                            filters_object = getattr(nd, filters_name)
                            filter_kwargs = filters_copy
                            for _, kwarg in filter_kwargs.items():
                                if  isinstance(kwarg, list):
                                    kwarg = tuple(kwarg)  #  configs usually provide lists, but we need tuples
                            data[fieldname] = filters_object(data[fieldname], **filter_kwargs)
                            if verbose:
                                print(f"Resulting shape {data[fieldname].shape}")
                    if verbose:
                        print(data[fieldname].shape)
                    if not no_filters:
                        data[fieldname] = np.pad(data[fieldname], pad_width=((0,0), (0, 1), (0, 1)), mode='wrap')[0:1,...]
                        data[fieldname] = np.roll(data[fieldname], (roll_x, roll_y), axis=(0,1))
                    data[fieldname] = data[fieldname].astype(output_dtype, copy=False)
                if output_format == 'npz':
                    np.savez(write_filename, **data)
                else:
                    with open(write_filename, 'wb') as out_file:
                        pickle.dump(data, out_file)
                processed += 1
            else:
                print(f"Block object not found in {read_filename}", flush=True)
    except OSError as e:
        if skip_bad_files:
            print(f"Skipping unreadable file {read_filename}: {e}", flush=True)
            skipped_bad.append(read_filename)
            continue
        raise
simulation_data_path = f'{path}{write_folder}/SimulationData.txt'
shutil.copy(f'{path}{read_folder}/SimulationData.txt', simulation_data_path)

# Copy optional .inp files if present in the source folder.
inp_files = glob.glob(f'{path}{read_folder}/*.inp')
for inp_file in inp_files:
    shutil.copy(inp_file, f'{path}{write_folder}/{os.path.basename(inp_file)}')

# Copy optional ConservedQuantities.txt if present in the source folder.
conserved_quantities_path = f'{path}{read_folder}/ConservedQuantities.txt'
if os.path.exists(conserved_quantities_path):
    shutil.copy(conserved_quantities_path, f'{path}{write_folder}/ConservedQuantities.txt')


# Read the file
with open(simulation_data_path, 'r') as file:
    lines = file.readlines()

# Modify the specific lines by scaling the existing values by zoom
for i, line in enumerate(lines):
    if 'Number of cells (x)' in line or 'Number of cells (y)' in line:
        left, sep, right = line.partition('=')
        if not sep:
            raise ValueError(f"Malformed line in SimulationData.txt: {line.strip()}")
        try:
            original_cells = int(right.strip())
        except ValueError as e:
            raise ValueError(f"Could not parse number of cells from line: {line.strip()}") from e

        scale_factor = 1.0 if no_filters else zoom
        new_cells = int(round(original_cells * scale_factor))
        lines[i] = f"{left}= {new_cells}\n"
    elif 'Grid resolution' in line:
        left, sep, right = line.partition('=')
        if not sep:
            raise ValueError(f"Malformed line in SimulationData.txt: {line.strip()}")

        values = [int(v) for v in re.findall(r'\d+', right)]
        if len(values) < 2:
            raise ValueError(f"Could not parse grid resolution from line: {line.strip()}")

        scale_factor = 1.0 if no_filters else zoom
        values[0] = int(round(values[0] * scale_factor))
        values[1] = int(round(values[1] * scale_factor))
        lines[i] = f"{left}= {' x '.join(str(v) for v in values)}\n"

# Write the modified lines back to the file
with open(simulation_data_path, 'w') as file:
    file.writelines(lines)

print(
    f"Summary: processed={processed}, skipped_existing={skipped_existing}, skipped_bad={len(skipped_bad)}",
    flush=True,
)
if skipped_bad:
    print("Skipped files:", flush=True)
    for skipped_file in skipped_bad:
        print(skipped_file, flush=True)
