import sys
import os
import numpy as np
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []


sys.path.append('/volume1/scratch/georgem/menura/analysis/')
import menura_utils as mu
from menura_paths import path_remote, path_local

run_ID = 14
run_ID = f'{int(run_ID):03}'
path_loc = f'{path_local}/run_{run_ID}'

path_rem = f'{path_remote}/run_{run_ID}'
it = 1000
remote_label = 'jean-zay'

md = mu.menura_data(path_loc, path_rem, it, remote_label=remote_label,
                 ask_scp=False, print_param=False, full_scp=True )
times = np.arange(md.mp.nb_it_max_cst)*md.mp.dt


beta0 = 2
poly_ind = 1
times2 = times[::20][1:]
iters2 = range(len(times))[::20][1:]
len(times2)
indexmin = 3 # indices to be discarded from the loaded data
indexmax = 3 # indices to be discarded from the loaded data
x = md.grid_x_box[indexmin:-indexmax]
y = md.grid_y_box[indexmin:-indexmax]

import numpy as np
import scipy.ndimage as nd
#iter = 400 # approximate to the nearest 100 assuming there are field outputs every 100


fields = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']
species_fields = ['rho', 'Pxx', 'Pyy', 'Pzz', 'Pyx', 'Pzx', 'Pzy', 'Pxy', 'Pxz', 'Pyz', 'Vx', 'Vy', 'Vz', 'Jx', 'Jy', 'Jz']
fields += species_fields

data = {field: [] for field in fields if field not in species_fields}
data.update({field: {'e' : [], 'i': []} for field in species_fields})
for iter in iters2:
    time = times[iter]
    md = mu.menura_data(path_loc, path_rem, iter, remote_label=remote_label,
                 ask_scp=False, print_param=False, full_scp=True )
    
    Bx, By, Bz = md.load_field('B')[:,indexmin:-indexmax,indexmin:-indexmax]*.01 #*np.sqrt(4*np.pi) # to have CGS units consistent with ECsim
    data['Bx'].append(Bx)
    data['By'].append(By)
    data['Bz'].append(Bz)
    
    Ex, Ey, Ez = md.load_field('E')[:,indexmin:-indexmax,indexmin:-indexmax]*(.01)**2 #*np.sqrt(4*np.pi) #*np.sqrt(4*np.pi) # to have CGS units consistent with ECsim
    data['Ex'].append(Ex)
    data['Ey'].append(Ey)
    data['Ez'].append(Ez)
    
    
    Jtot = md.load_field('Jtot')[:,indexmin:-indexmax,indexmin:-indexmax]*.01/(4*np.pi) #/np.sqrt(4*np.pi) # to have CGS units consistent with ECsim
    
    
    for spec in data['rho'].keys():
        Ji = md.load_field('Ji')[:,indexmin:-indexmax,indexmin:-indexmax]*.01/(4*np.pi) #/np.sqrt(4*np.pi) 
        density = md.load_field('density')[indexmin:-indexmax,indexmin:-indexmax]/(4*np.pi)  # to have normalization expected in ECsim
        if spec == 'e':
            J = Jtot - Ji
            rho = -density
        else:
            J = Ji
            rho = density
        data['rho'][spec].append(rho)
        Vx, Vy, Vz = J / rho
        data['Vx'][spec].append(Vx)
        data['Vy'][spec].append(Vy)
        data['Vz'][spec].append(Vz)
        data['Jx'][spec].append(J[0])
        data['Jy'][spec].append(J[1])
        data['Jz'][spec].append(J[2])

        if spec == 'i':
            Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = md.load_field('stress_s0')[:,indexmin:-indexmax,indexmin:-indexmax]*(.01)**2/(4*np.pi) # to have CGS units consistent with ECsim
            data['Pxx'][spec].append(Pxx- density * Vx**2)
            data['Pyy'][spec].append(Pyy- density * Vy**2)
            data['Pzz'][spec].append(Pzz- density * Vz**2)
            data['Pxy'][spec].append(Pxy- density * Vx * Vy)
            data['Pyx'][spec].append(Pxy- density * Vx * Vy)
            data['Pxz'][spec].append(Pxz- density * Vx * Vz)
            data['Pzx'][spec].append(Pxz - density * Vx * Vz)
            data['Pyz'][spec].append(Pyz - density * Vy * Vz)
            data['Pzy'][spec].append(Pyz - density * Vy * Vz)
        elif spec == 'e':
            data['Pxx'][spec].append(2* beta0* density**poly_ind/3*(.01)**2/(4*np.pi)) # why do I have to mulitply by two?
            data['Pyy'][spec].append(2* beta0* density**poly_ind/3*(.01)**2/(4*np.pi)) # why do I have to mulitply by two?
            data['Pzz'][spec].append(2* beta0* density**poly_ind/3*(.01)**2/(4*np.pi)) # why do I have to mulitply by two?
            data['Pxy'][spec].append(np.zeros_like(density))
            data['Pyx'][spec].append(np.zeros_like(density))
            data['Pxz'][spec].append(np.zeros_like(density))
            data['Pzx'][spec].append(np.zeros_like(density))
            data['Pyz'][spec].append(np.zeros_like(density))
            data['Pzy'][spec].append(np.zeros_like(density))
        
        
for field in data.keys():
    if field in species_fields:
        for spec in data[field].keys():
            try:
                data[field][spec] = np.array(data[field][spec]).transpose(1, 2, 0)
            except Exception as e:
                print(f'Error for {field} and {spec} : {e}, {len(data[field][spec])}')
    else:
        data[field] = np.array(data[field]).transpose(1, 2, 0)

import matplotlib.animation as animation
import matplotlib.pyplot as plt
fig, axs = plt.subplots(5, 3, figsize=(15, 20))
fields = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez', 'Jx', 'Jy', 'Jz', 'Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pxz', 'Pyz']

# Create the pcolormesh objects ahead of time
pcolormesh_objects = []
for ax, field in zip(axs.flat, fields):
    if field in ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']:
        pcolormesh = ax.pcolormesh(x, y, data[field][..., 0], cmap='seismic', 
                                   vmax=data[field][...,-1].max(), vmin=-data[field][...,-1].max())
    else:
        pcolormesh = ax.pcolormesh(x, y, data[field]['e'][..., 0], cmap='seismic', 
                                   vmax=data[field]['e'][...,-1].max(), vmin=-data[field]['e'][...,-1].max())
    fig.colorbar(pcolormesh, ax=ax)
    ax.set_title(f"{field}, time = {times[0]:.2f}"+r"$\omega_{pi}^{-1}$")
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
        ax.set_title(f"{field}, time = {time:.2f}"+r"$\omega_{pi}^{-1}$")

ani = animation.FuncAnimation(fig, update_frame, frames=range(data['Bx'].shape[-1]), repeat=False)
ani.save('../examples/img/animation_menura.gif', writer='imagemagick')