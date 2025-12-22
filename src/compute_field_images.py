import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import src.trainers as tr
import src.read_pic as rp
import os
import numpy as  np
import shutil
import matplotlib.animation as animation
import src.utilities as ut
# Fields to read.


if len(sys.argv) <= 4:
    print("Usage: python compute_movie_field.py <plot_field> <path> <run_ID>")
    print("Usage: python compute_field_images.py <plot_field> <run_ID> <field_max>")
    sys.exit(1)


if len(sys.argv) == 5:
    field_max = float(sys.argv[4])
else:
    field_max = None

print(f'{len(sys.argv) = }, Setting field_max to {field_max}')

plot_fields = sys.argv[1].strip('[]').split(',')
plot_fields = [field.strip() for field in plot_fields]

fields_list = []
species_list = []
for field in plot_fields:
    if '_' in field:
        parsed_field, species = field.rsplit('_', 1)
        fields_list.append(parsed_field)
        species_list.append(species)
    else:
        fields_list.append(field)
        species_list.append(None)
print(f'{fields_list = }, {species_list = }')

fields_to_read={"B":True,"B_ext":False,"divB":True,"E":True,"E_ext":False,"rho":True,"J":True,
                "P":True,"PI":False,"Heat_flux":False,"N":False,"Qrem":False}
# Path of the folder containing the .h5 files to read.
files_path= sys.argv[2] #"/volume1/scratch/share_dir/ecsim/peppe/" #"/lustre1/project/stg_00032/share_dir/brecht/" # "/users/cpa/francesc/share_dir/SW/data_small/" #"/users/cpa/francesc/share_dir/jincai/dat_FF2D07e/" #="/users/cpa/francesc/share_dir/nn/data/raw_data/"
experiments = [f.name for f in os.scandir(files_path) if f.is_dir()]
print(f"{experiments = }")

run_ID = sys.argv[3] #'data_filter'
data, X, Y, qom, times = rp.get_exp_times([run_ID], files_path, fields_to_read, 
                                          choose_species=['e','i'], verbose=True, 
                                          choose_times=1)
iters2 = range(len(times))

data = data[run_ID]

ut.get_Ohm(data, qom, X[:,0], Y[0,:])

fig, axs = plt.subplots(5, 3, figsize=(15, 20))

data['Jz-tot'] = data['Jz']['e'] + data['Jz']['i']
data['Jx-tot'] = data['Jx']['e'] + data['Jx']['i']
data['Jz-tot'] = data['Jz']['e'] + data['Jz']['i']



for plot_field, species in zip(fields_list, species_list):
    print(f'{plot_field = }, {species = }')

    # Create output directory for frames (single folder per field)
    frames_dir = f'{files_path}/{run_ID}/plots/{run_ID}_frames/{plot_field}'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 5))

    if plot_field in ['rho', 'Pxx', 'Pyy', 'Pzz']:
        cmap = 'viridis'
    else:
        cmap = 'seismic'

    # Initialize the plot with the first frame
    if species is None:
        try:
            shape2 = data[plot_field].shape[2]
        except KeyError:
            print(f"Field {plot_field} not found in data.")
            print(f"{data.keys() = }")
            raise KeyError

        finite_data = data[plot_field][np.isfinite(data[plot_field])]
        # Find the 5th and 95th percentiles for Jz to set vmin and vmax
        #Jz_min = np.percentile(finite_data, 1)
        #Jz_max = np.percentile(finite_data, 99)
        # Find the min and max values for Jz to set vmin and vmax
        if field_max is None:
            field_min = min(np.nanmin(finite_data), np.nanmin(finite_data))/4
            field_max = max(np.nanmax(finite_data), np.nanmax(finite_data))/4
        else:
            field_min = -field_max
        vlimit = max(-field_min, field_max)
        print(f" {data[plot_field].shape = }, {field_min = }, {field_max =}, {iters2 = }")

        # Loop through frames and save each as PNG
        for frame in range(shape2):
            ax.clear()
            if cmap == 'seismic':
                cax = ax.pcolormesh(X,Y,data[plot_field][:, :, frame], vmin=-vlimit, vmax=vlimit, cmap=cmap)
            else:
                cax = ax.pcolormesh(X,Y,np.abs(data[plot_field][:, :, frame]), cmap=cmap, vmin=0, vmax=vlimit)
            fig.colorbar(cax)
            ax.set_title(f'{plot_field}, run {run_ID}, time = {times[frame]:.2f}'+r"$\Omega_{ci}^{-1}$")
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f'frame_{frame:04d}.png')
            fig.savefig(frame_path, dpi=150, bbox_inches='tight')
            fig.clf()
            ax = fig.add_subplot(111)
        
        print(f'Saved {shape2} frames at {frames_dir}')
    else:
        shape2 = data[plot_field][species].shape[2]
        finite_data = data[plot_field][species][np.isfinite(data[plot_field][species])]
        #Jz_min = np.percentile(finite_data, 1)
        #Jz_max = np.percentile(finite_data, 99)
        # Find the min and max values for Jz to set vmin and vmax
        if field_max is None:
            field_min = min(np.nanmin(finite_data), np.nanmin(finite_data))/4
            field_max = max(np.nanmax(finite_data), np.nanmax(finite_data))/4
        else:
            field_min = -field_max 
        vlimit = max(-field_min, field_max)
        print(f" {data[plot_field][species].shape = }, {field_min = }, {field_max =}")
        # Loop through frames and save each as PNG
        for frame in range(shape2):
            ax.clear()
            if cmap == 'seismic':
                cax = ax.pcolormesh(X,Y,data[plot_field][species][:, :, frame], vmin=-vlimit, vmax=vlimit, cmap=cmap)
            else:
                cax = ax.pcolormesh(X,Y,np.abs(data[plot_field][species][:, :, frame]), cmap=cmap, vmin=0, vmax=vlimit)
            fig.colorbar(cax)
            ax.set_title(f'{plot_field}, {species}, run {run_ID}, time = {times[frame]:.2f}'+r"$\Omega_{ci}^{-1}$")
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f'{species}_frame_{frame:04d}.png')
            fig.savefig(frame_path, dpi=150, bbox_inches='tight')
            fig.clf()
            ax = fig.add_subplot(111)
        
        print(f'Saved {shape2} frames at {frames_dir}')


    plt.close(fig)
