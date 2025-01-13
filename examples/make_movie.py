import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import src.trainers as tr
import src.read_pic as rp
import os
import numpy as  np
import matplotlib.animation as animation
# Fields to read.
fields_to_read={"B":True,"B_ext":False,"divB":True,"E":True,"E_ext":False,"rho":True,"J":True,
                "P":True,"PI":False,"Heat_flux":False,"N":False,"Qrem":False}
# Path of the folder containing the .h5 files to read.
files_path="/volume1/scratch/share_dir/peppe/" #"/lustre1/project/stg_00032/share_dir/brecht/" # "/users/cpa/francesc/share_dir/SW/data_small/" #"/users/cpa/francesc/share_dir/jincai/dat_FF2D07e/" #="/users/cpa/francesc/share_dir/nn/data/raw_data/"
experiments = [f.name for f in os.scandir(files_path) if f.is_dir()]
print(f"{experiments = }")

experiment = 'data_filter'
data, X, Y, qom, times = rp.get_exp_times([experiment], files_path, fields_to_read, 
                                          choose_species=['e','i'], verbose=True, 
                                          choose_times=1)


data = data[experiment]
fig, axs = plt.subplots(5, 3, figsize=(15, 20))
fields = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez', 'Jx', 'Jy', 'Jz', 'Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pxz', 'Pyz']

# Create the pcolormesh objects ahead of time
pcolormesh_objects = []
for ax, field in zip(axs.flat, fields):
    if field in ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']:
        pcolormesh = ax.pcolormesh(X, Y, data[field][..., 0], cmap='seismic', 
                                   vmax=data[field][...].max(), vmin=-data[field][...].max())
    else:
        pcolormesh = ax.pcolormesh(X, Y, data[field]['e'][..., 0], cmap='seismic', 
                                   vmax=data[field]['e'][...].max(), vmin=-data[field]['e'][...].max())
    fig.colorbar(pcolormesh, ax=ax)
    ax.set_title(f"{field}, time = {times[0]}"+r"$\omega_{pi}^{-1}$")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    pcolormesh_objects.append(pcolormesh)

plt.tight_layout()

def update_frame(i):
    time = times[i]
    print(f"Time = {time}")
    for ax, pcolormesh, field in zip(axs.flat, pcolormesh_objects, fields):
        if field in ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']:
            pcolormesh.set_array(data[field][..., i].ravel())
        else:
            pcolormesh.set_array(data[field]['i'][..., i].ravel())
        ax.set_title(f"{field}, time = {time}"+r"$\omega_{pi}^{-1}$")

ani = animation.FuncAnimation(fig, update_frame, frames=range(data['Bx'].shape[-1]), repeat=False)
ani.save('../examples/img/animation.gif', writer='imagemagick')