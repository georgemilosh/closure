import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import src.trainers as tr
import src.read_pic as rp
import os
import src.utilities as ut
import numpy as np
import pickle
import time
start_time = time.time()
# Fields to read.
fields_to_read={"B":True,"B_ext":False,"divB":True,"E":True,"E_ext":False,"rho":True,"J":True,
                "P":True,"PI":False,"Heat_flux":False,"N":False,"Qrem":False}
# Path of the folder containing the .h5 files to read.
if len(sys.argv) > 2:
    experiment = sys.argv[2]
    files_path = sys.argv[1] #"/volume1/scratch/share_dir/ecsim/peppe/" #"/lustre1/project/stg_00032/share_dir/brecht/" # "/users/cpa/francesc/share_dir/SW/data_small/" #"/users/cpa/francesc/share_dir/jincai/dat_FF2D07e/" #="/users/cpa/francesc/share_dir/nn/data/raw_data/"
else:
    print("Usage: python compute_flux.py <path> <experiment>")
    sys.exit(1)

experiments = [f.name for f in os.scandir(files_path) if f.is_dir()]
print(f"{experiments = }")


data, X, Y, qom, times = rp.get_exp_times([experiment], files_path, fields_to_read, 
                                          choose_species=['e','i'], verbose=True, 
                                          choose_times=1, indexing='ij',
                                           filters = None) 
print(f"{data[experiment]['Bx'].shape = }")
print(f"{data[experiment].keys() = }")
filtered = {}
if X.shape[0] == 2048:
    xs = [2048, 1376, 928, 608, 416, 288, 192, 128, 96, 64, 32, 20, 16, 12, 10, 8, 6, 4, 3, 2, 1]
elif X.shape[0] == 512:
    xs = [512, 352, 256, 176, 128, 88, 64, 44, 32, 22, 16, 11, 8, 6, 4, 3, 2, 1]
else:
    raise ValueError(f"Shape {X.shape[0]} treatment not implemented")

filtered['xs'] = xs

for quantity in ['PIuu', 'PIbb', 'Ef_favre', 'PS', '-Ptheta', 'JdotE']:
    filtered[quantity] = {}
    for species in ['e', 'i']:
        filtered[quantity][species] = []
for quantity in ['E2_bar','B2_bar']:
    filtered[quantity] = []

for xi in xs:
    print(f"xi = {xi}")
    ut.scale_filtering(data[experiment], X[:,0], Y[0,:], qom, verbose=False, 
                       filters = {'name': 'uniform_filter', 'size': xi, 'mode' : 'wrap', 'axes': (0,1)})
    filtered['E2_bar'].append(np.mean(data[experiment]['E2_bar'], axis=(0,1)))
    filtered['B2_bar'].append(np.mean(data[experiment]['B2_bar'], axis=(0,1)))
    for quantity in ['PIuu', 'PIbb', 'Ef_favre', 'PS', '-Ptheta', 'JdotE']:
        for species in ['e', 'i']:
            filtered[quantity][species].append(np.mean(data[experiment][quantity][species], axis=(0,1)))
for quantity in ['PIuu', 'PIbb', 'Ef_favre', 'PS', '-Ptheta', 'JdotE']:
    for species in ['e', 'i']:
        filtered[quantity][species] = np.array(filtered[quantity][species])
for quantity in ['E2_bar', 'B2_bar']:
    filtered[quantity] = np.array(filtered[quantity])
data[experiment]['P'] = {}
for species in ['e', 'i']:
    data[experiment]['P'][species]=(data[experiment]['Pxx'][species]+data[experiment]['Pyy'][species]+data[experiment]['Pzz'][species])/3
filtered['Ethi_i'] = 3*np.mean(data[experiment]['P']['i'], axis=(0,1))/2
filtered['Ethi_e'] = 3*np.mean(data[experiment]['P']['e'], axis=(0,1))/2

with open(f'{files_path}/{experiment}/filtered_quantities.pkl', 'wb') as f:
    pickle.dump(filtered, f)


end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
