import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import src.trainers as tr
import src.read_pic as rp
import os
import numpy as  np
import matplotlib.animation as animation
# Fields to read.


if len(sys.argv) != 4:
    print("Usage: python compute_movie_field.py <plot_field> <run_ID>")
    sys.exit(1)

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
fig, axs = plt.subplots(5, 3, figsize=(15, 20))
fields = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez', 'Jx', 'Jy', 'Jz', 'Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pxz', 'Pyz']


data['Jz-tot'] = data['Jz']['e'] + data['Jz']['i']



for plot_field, species in zip(fields_list, species_list):
    print(f'{plot_field = }, {species = }')
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 5))

    if plot_field in ['rho', 'Pxx', 'Pyy', 'Pzz']:
        cmap = 'viridis'
    else:
        cmap = 'seismic'

    # Initialize the plot with the first frame
    if species is None:
        shape2 = data[plot_field].shape[2]

        finite_data = data[plot_field][np.isfinite(data[plot_field])]
        # Find the 5th and 95th percentiles for Jz to set vmin and vmax
        #Jz_min = np.percentile(finite_data, 1)
        #Jz_max = np.percentile(finite_data, 99)
        # Find the min and max values for Jz to set vmin and vmax
        Jz_min = min(np.nanmin(finite_data), np.nanmin(finite_data))
        Jz_max = max(np.nanmax(finite_data), np.nanmax(finite_data))
        vlimit = max(-Jz_min, Jz_max)
        print(f" {data[plot_field].shape = }, {Jz_min = }, {Jz_max =}, {iters2 = }")

        if cmap == 'seismic':
            cax = ax.pcolormesh(X,Y,data[plot_field][:, :, 0], vmin=-vlimit/4, vmax=vlimit/4, cmap=cmap)
        else:
            cax = ax.pcolormesh(X,Y,np.abs(data[plot_field][:, :, 0]), cmap=cmap, vmin=0, vmax=vlimit/2)
        fig.colorbar(cax)
        ax.set_title(f'{plot_field}, run {run_ID} time = {times[0]:.2f}'+r"$\omega_{pi}^{-1}$")

        def update(frame):
            cax.set_array(data[plot_field][:, :, frame].ravel())
            ax.set_title(f'{plot_field}, run {run_ID}, time = {times[frame]:.2f}'+r"$\omega_{pi}^{-1}$")
            return cax,
    else:
        shape2 = data[plot_field][species].shape[2]
        finite_data = data[plot_field][species][np.isfinite(data[plot_field][species])]
        #Jz_min = np.percentile(finite_data, 1)
        #Jz_max = np.percentile(finite_data, 99)
        # Find the min and max values for Jz to set vmin and vmax
        Jz_min = min(np.nanmin(finite_data), np.nanmin(finite_data))
        Jz_max = max(np.nanmax(finite_data), np.nanmax(finite_data))
        vlimit = max(-Jz_min, Jz_max)
        print(f" {data[plot_field][species].shape = }, {Jz_min = }, {Jz_max =}, {iters2 = }")
        if cmap == 'seismic':
            cax = ax.pcolormesh(X,Y,data[plot_field][species][:, :, 0], vmin=-vlimit/4, vmax=vlimit/4, cmap=cmap)
        else:
            cax = ax.pcolormesh(X,Y,np.abs(data[plot_field][species][:, :, 0]), cmap=cmap, vmin=0, vmax=vlimit/2)
        fig.colorbar(cax)
        ax.set_title(f'{plot_field}, {species}, run {run_ID}, time = {times[0]:.2f}'+r"$\omega_{pi}^{-1}$")

        def update(frame):
            cax.set_array(data[plot_field][species][:, :, frame].ravel())
            ax.set_title(f'{plot_field}, {species}, run {run_ID}, time = {times[frame]:.2f}'+r"$\omega_{pi}^{-1}$")
            return cax,


    fig.set_tight_layout(True)

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=shape2, blit=True)


    # Save the animation as a gif file
    plots_path = os.path.join(files_path, run_ID, 'plots')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    if species is None:
        ani.save(f'{files_path}/{run_ID}/plots/{plot_field}_{run_ID}_movie.gif')
        print('Saved gif at ', f'{files_path}/plots/{run_ID}/{plot_field}_{run_ID}_movie.gif')
    else:
        ani.save(f'{files_path}/{run_ID}/plots/{plot_field}_{species}_{run_ID}_movie.gif')
        print('Saved gif at ', f'{files_path}/plots/{run_ID}/{plot_field}_{species}_{run_ID}_movie.gif')

    plt.close(fig)