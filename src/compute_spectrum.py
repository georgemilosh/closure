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


def vector_spectrum_2D(field, x, y):
    if len(X.shape) == 1:
        Lx = X[-1] - X[0]
        Ly = Y[-1] - Y[0]
        x = X
        y = Y
    elif len(X.shape) == 2:
        Lx = X[-1,0] - X[0,0]
        Ly = Y[0,-1] - Y[0,0]
        x = X[:,0]
        y = Y[0,:]
    else:
        raise ValueError("X and Y must be 1D or 2D arrays")
    nxc=len(x)
    nyc=len(y)
    # Repeated boundaries must be excluded according to the definition of the FFT.
    field_x_ft=np.fft.rfft2(field[0])
    field_y_ft=np.fft.rfft2(field[1])
    field_z_ft=np.fft.rfft2(field[2])

    # 2D power spectrum.
    spec_2D=(abs(field_x_ft)**2+abs(field_y_ft)**2+abs(field_z_ft)**2)/((nxc*nyc)**2)
    #print(f"{spec_2D.shape = }")
    spec_2D[:,1:-1]*=2 # Some modes are doubled to take into account the redundant ones removed by numpy's rfft.
    try:
        kx=np.fft.fftfreq(nxc-1,x[1]-x[0])*2*np.pi
    except Exception as e:
        print(f"{nxc = }, {x.shape = }")
        raise e
    ky=np.fft.rfftfreq(nyc-1,x[1]-x[0])*2*np.pi
    #print(f"{nxc = }, {nyc = }, {kx= }")
    # The 1D magnetic field energy spectrum is calculated.
    spec_1D=np.zeros(nxc//2+1)

    for iy in range(len(ky)):
        for ix in range(len(kx)):
            try:
                index=round( np.sqrt( (Lx*kx[ix]/(2*np.pi))**2+(Ly*ky[iy]/(2*np.pi))**2 ) )
                if index<=(nxc//2):
                    spec_1D[index]+=spec_2D[ix,iy]
            except Exception as e:
                print(f"{index = }, {ix = }, {iy = }, {spec_2D.shape = }, {spec_1D.shape = }")
                raise e

    return ky,spec_1D[:-1]



start_time = time.time()
# Fields to read.
fields_to_read={"B":True,"B_ext":False,"divB":True,"E":True,"E_ext":False,"rho":True,"J":True,
                "P":True,"PI":False,"Heat_flux":False,"N":False,"Qrem":False}
# Path of the folder containing the .h5 files to read.
if len(sys.argv) > 2:
    experiment = sys.argv[2]
    files_path = sys.argv[1] #"/volume1/scratch/share_dir/ecsim/peppe/" #"/lustre1/project/stg_00032/share_dir/brecht/" # "/users/cpa/francesc/share_dir/SW/data_small/" #"/users/cpa/francesc/share_dir/jincai/dat_FF2D07e/" #="/users/cpa/francesc/share_dir/nn/data/raw_data/"
else:
    print("Please provide the experiment name and files_path as command line arguments separated by a space.")
    sys.exit(1)

experiments = [f.name for f in os.scandir(files_path) if f.is_dir()]
print(f"{experiments = }")


data, X, Y, qom, times = rp.get_exp_times([experiment], files_path, fields_to_read, 
                                          choose_species=['e','i'], verbose=True, 
                                          choose_times=1, indexing='ij',
                                           filters = None) 

data = data[experiment]
import src.utilities as ut
ut.get_PS_2D_field(data, X[:,0], Y[0,:]) # indexing ij
print("computed pressure strain")
ut.get_Ohm(data, [-np.inf,1], X[:,0], Y[0,:])
print("computed ohm's law")


Jx = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
Jy = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
Jz = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)
J = np.sqrt(np.mean(Jx**2 + Jy**2 + Jz**2, axis=(0,1)))
Ji = np.sqrt(np.mean(data['Jx']['i']**2 + data['Jy']['i']**2 + data['Jz']['i']**2, axis=(0,1)))

B_spectrum = []
V_spectrum = []
for iteri, time in enumerate(times):
    print(f"computing spectra at {time = }")
    B = np.array([data['Bx'], data['By'], data['Bz']])[...,iteri]
    V = np.array([data['Vx']['i'], data['Vy']['i'], data['Vz']['i']])[...,iteri]
    ky, B_spectrum_ar = vector_spectrum_2D(B, X, Y)
    ky, V_spectrum_ar = vector_spectrum_2D(V, X, Y)
    B_spectrum.append(B_spectrum_ar)
    V_spectrum.append(V_spectrum_ar)
B_spectrum = np.array(B_spectrum)
V_spectrum = np.array(V_spectrum)



for quantity_name in ['E','EMHD_', 'EHall_', 'EP_']:
    ky2,spec_Ohms = ut.vector_spectrum_2D(data[f'{quantity_name}x'], data[f'{quantity_name}y'], data[f'{quantity_name}z'], X, Y)
print("computed spectra")



imin = np.argmax(J)-4
imax = np.argmax(J)+1
thresholds = {}
mean_values = {}
percentiles = {}
thresholds2 = {}
mean_values2 = {}
percentiles2 = {}

for species in ['i', 'e']:
    thresholds[species] = {}
    mean_values[species] = {}
    percentiles[species] = {}
    thresholds2[species] = {}
    mean_values2[species] = {}
    percentiles2[species] = {}
    quantity_name = 'PiD'
    condition_names = ['Qomega', 'QD', 'QJ']
    for condition_name in condition_names:
            print(f"{condition_name = }")
            if condition_name == 'QJ':
                condition = data[condition_name][...,imin:imax] #/ np.sqrt(np.mean(data[condition_name][...,imin:imax] ** 2))
            else:
                condition = data[condition_name][species][...,imin:imax] #/ np.sqrt(np.mean(data[condition_name][species][...,imin:imax] ** 2))
            conditionmax = 8 #np.max(condition)/5
            quantity = data[quantity_name][species][...,imin:imax] #/ np.sqrt(np.mean(data[quantity_name][species][...,imin:imax] ** 2))
            thresholds[species][condition_name] = np.arange(0, conditionmax, conditionmax / 20)
            mean_values[species][condition_name] = [np.mean(quantity[condition > a]) for a in thresholds[species][condition_name]]
            percentiles[species][condition_name] = [np.mean(condition > a) for a in thresholds[species][condition_name]]
    


    quantity_names = ['PiD', 'J*(E+VxB)']
    for quantity_name in quantity_names:
            if quantity_name == 'J*(E+VxB)':
                condition_name = 'QJ'
                condition = np.sqrt(data[condition_name][...,imin:imax]) 
            else:
                condition_name = 'QD'
                condition = np.sqrt(data[condition_name][species])[...,imin:imax] 
            quantity = data[quantity_name][species][...,imin:imax] / np.sqrt(np.mean(data[quantity_name][species][...,imin:imax] ** 2))
            thresholds2[species][quantity_name] = np.arange(0, np.max(condition), np.max(condition) / 100)
            mean_values2[species][quantity_name] = np.array([np.mean(quantity[(condition > a) & (condition < (a + np.max(condition) / 100))]) for a in thresholds2[species][quantity_name]])
            percentiles2[species][quantity_name] = np.array([np.mean(condition > a) for a in thresholds2[species][quantity_name]])

            thresholds2[species][quantity_name] = thresholds2[species][quantity_name][~np.isnan(mean_values2[species][quantity_name])]
            percentiles2[species][quantity_name] = percentiles2[species][quantity_name][~np.isnan(mean_values2[species][quantity_name])]
            mean_values2[species][quantity_name] = mean_values2[species][quantity_name][~np.isnan(mean_values2[species][quantity_name])]
print("computed thresholds")

output_data = {
    'times': times,
    'J': J,
    'Ji': Ji,
    'ky' : ky,
    'B_spectrum' : B_spectrum,
    'V_spectrum' : V_spectrum,
    'ky2' : ky2,
    'spec_Ohms' : spec_Ohms,
    'thresholds' : thresholds,
    'mean_values' : mean_values,
    'percentiles' : percentiles,
    'thresholds2' : thresholds2,
    'mean_values2' : mean_values2,
    'percentiles2' : percentiles2
}

with open(f'{files_path}/{experiment}/spectra.pkl', 'wb') as f:
    pickle.dump(output_data, f)