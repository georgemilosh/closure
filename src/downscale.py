

"""
This script processes HDF5 files by applying specified filters and saves the processed data as pickle files.

Arguments:
    --path (str): The base directory path for reading and writing files.
    --read_folder (str): The folder name where input HDF5 files are located.
    --write_folder (str): The folder name where output pickle files will be saved.

Variables:
    filters (list): A list of dictionaries specifying the filters to apply.

Processing:
    1. Iterate over each filename in filenames_list.
    2. Load the HDF5 file.
    3. Extract data for each field in the file.
    4. Apply specified filters to the data.
    5. Save the processed data as a pickle file in the write_folder.
    6. Copy the SimulationData.txt file from the read_folder to the write_folder.
    7. Modify the 'Number of cells (x)' and 'Number of cells (y)' lines in the SimulationData.txt file.

Note:
    - The script assumes that the HDF5 files have a specific structure with data stored under "/Step#0/Block/".
    - The filters are applied in the order they are listed in the filters variable.
    - The script prints verbose output if the verbose variable is set to True.

Usage:
    python downscale.py --path /dodrio/scratch/projects/2024_109/ecsim/peppe/ --read_folder T2D16 --write_folder T2D16_filter
"""
import h5py
import numpy as np
import scipy.ndimage as nd
import pickle
import glob
import os
import shutil
import argparse

filters=[{'name': 'uniform_filter', 'size': 4, 'axes': (1,2), 'mode' : 'wrap'},
                {'name': 'zoom', 'zoom': (1, 0.25, 0.25), 'mode' : 'grid-wrap'}]
parser = argparse.ArgumentParser(description='Process HDF5 files and apply filters.')
parser.add_argument('--path', type=str, default='/volume1/scratch/share_dir/peppe/', help='The base directory path for reading and writing files.')
parser.add_argument('--read_folder', default='data', type=str, required=True, help='The folder name where input HDF5 files are located.')
parser.add_argument('--write_folder', default='data_filter', type=str, required=True, help='The folder name where output pickle files will be saved.')
args = parser.parse_args()

path = args.path
read_folder = args.read_folder
write_folder = args.write_folder


if not os.path.exists(f'{path}{read_folder}'): # Check if read_folder exists
    raise FileNotFoundError(f"The folder {path}{read_folder} does not exist.")

if not os.path.exists(f'{path}{write_folder}'): # Check if write_folder exists, if not create it
    os.makedirs(f'{path}{write_folder}')
else:
    if os.listdir(f'{path}{write_folder}'): # protect from overwriting existing files
        raise FileExistsError(f"The folder {path}{write_folder} is not empty.")

# Get all filenames in the read_folder
all_filenames = glob.glob(f'{path}{read_folder}/*.h5')
filenames_list = [os.path.basename(f) for f in all_filenames]

for filename in filenames_list:
    read_filename = f'{path}{read_folder}/{filename}'
    write_filename = f'{path}{write_folder}/{filename}.pkl'
    print(f"Processing {read_filename}")
    print(f"Writing to {write_filename}")
    # Load the file
    verbose=False
    data = {}
    with h5py.File(read_filename, 'r') as n:
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
                data[fieldname] = np.pad(data[fieldname], pad_width=((0,0), (0, 1), (0, 1)), mode='wrap')[0:1,...]
            with open(write_filename, 'wb') as out_file:
                pickle.dump(data, out_file)
        else:
            print(f"Block object not found in {read_filename}")
simulation_data_path = f'{path}{write_folder}/SimulationData.txt'
shutil.copy(f'{path}{read_folder}/SimulationData.txt', simulation_data_path)


# Read the file
with open(simulation_data_path, 'r') as file:
    lines = file.readlines()

# Modify the specific lines
for i, line in enumerate(lines):
    if 'Number of cells (x)' in line:
        lines[i] = 'Number of cells (x)      = 512\n'
    if 'Number of cells (y)' in line:
        lines[i] = 'Number of cells (y)      = 512\n'

# Write the modified lines back to the file
with open(simulation_data_path, 'w') as file:
    file.writelines(lines)