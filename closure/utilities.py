"""
utlities.py
This module contains utility functions for various tasks such 
as data transformation, loss evaluation, and plotting.
Repo:       closure
Projects:   STRIDE, HELIOSKILL
Author:     George Miloshevich
Date:       2025
License:    MIT License
            
"""

import subprocess

import pandas as pd
try:
    import torch
    from . import trainers as tr
except ImportError:
    print("utilities: PyTorch is not installed. Some functions may not work. Omitting trainer-dependent functions.")
#import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
from . import read_pic as rp
import os
import pickle
import scipy.ndimage as nd
import ast

def alias(*names):
    """
    A decorator that assigns multiple global aliases to a function. It allows conveniently renaming functions without breaking backward compatibility
    Args:
        *names: One or more strings representing the alias names to assign to the decorated function.
    Returns:
        decorator: A decorator that, when applied to a function, adds the function to the global namespace under each specified alias.
    Example:
        @alias('foo', 'bar')
        def my_function():
            pass
        # Now, my_function can be accessed as foo or bar in the global namespace.
    """

    def decorator(func):
        globals_ = globals()
        for name in names:
            globals_[name] = func
        return func
    return decorator

def set_nested_config(*args, **kwargs):  # backward compat
    from closure.config import set_nested_config as _f
    return _f(*args, **kwargs)


def species_to_list(input_list):
    """
    Splits each item in the input_list by '_' if '_' is present in the item.
    
    Args:
        input_list (list): A list of strings.
        
    Returns:
        list: A new list where each item is split by '_' if '_' is present, otherwise the item remains unchanged.

    Example:
        species_to_list(['a', 'b_c', 'd_e_f']) -> ['a', ['b', 'c'], ['d', 'e', 'f']]
    """
    return [item.split('_') if '_' in item else item for item in input_list]


def load_and_compute_difference(file_path):
    """
    Load a pickle file containing the training information from the given file path and compute the difference between 'train+val' and 'train' times.
    Parameters:
    file_path (str): The path to the pickle file.
    Returns:
    dict: A dictionary containing the loaded data with the computed difference between 'train+val' and 'train' times stored in 'val' key.
    """
    
    with open(file_path, 'rb') as file:
        loss_dict = pickle.load(file)
    
    loss_dict['time']['val'] = []
    for train, train_val in zip(loss_dict['time']['train'], loss_dict['time']['train+val']):
        loss_dict['time']['val'].append(train_val - train)
    
    return loss_dict

def append_index_to_duplicates(lst):
    """
    This function takes a list as input and returns a new list where each duplicate string element is appended with its 
    index within its group of duplicates. Non-string elements are left unchanged.

    Parameters:
    lst (list): The input list. It can contain elements of any type.

    Returns:
    list: A new list where each duplicate string is appended with its index within its group of duplicates. 
    Non-string elements are left unchanged.
    """
    count_dict = {}
    result = []
    for i, elem in enumerate(lst):
        if isinstance(elem, str):
            if lst.count(elem) > 1:  # Only count duplicates
                if elem in count_dict:
                    count_dict[elem] += 1
                    result.append(f"{elem}{count_dict[elem]}")
                else:
                    count_dict[elem] = 1
                    result.append(f"{elem}{count_dict[elem]}")
            else:
                result.append(elem)
        else:
            result.append(elem)
    return result

def get_duplicate_indices(lst):
    """
    Returns a dictionary containing the indices of duplicate elements in the given list.

    Parameters:
    lst (list): A list of elements.

    Returns:
    dict: A dictionary where the keys are the duplicate elements and the values are lists of their indices.

    Example:
    >>> get_duplicate_indices([1, 2, 3, 2, 4, 1, 5, 4])
    {1: [0, 5], 2: [1, 3], 4: [4, 7]}
    """
    index_dict = {}
    for i, elem in enumerate(lst):
        if elem is not None:
            if elem in index_dict:
                index_dict[elem].append(i)
            else:
                index_dict[elem] = [i]
    return {key: value for key, value in index_dict.items() if len(value) > 1}



def get_git_revision_hash() -> str:
    """
    Returns the hash of the current Git revision. This function assumes that the Git executable is available in the
    system path.

    Returns:
        str: The hash of the current Git revision.
    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def parse_score(*args, **kwargs):  # backward compat
    from closure.evaluation import parse_score as _f
    return _f(*args, **kwargs)

def compare_runs(*args, **kwargs):  # backward compat
    from closure.evaluation import compare_runs as _f
    return _f(*args, **kwargs)

def compare_metrics(*args, **kwargs):  # backward compat
    from closure.evaluation import compare_metrics as _f
    return _f(*args, **kwargs)

def conserved_quantities(folder, verbose=True):
    """
    Reads ConservedQuantities.txt generated by ECsim containing conserved quantities from a specified folder,
    and returns the data as a pandas DataFrame.
    Args:
        folder (str): The path to the folder containing the CSV file.
        verbose (bool, optional): If True, prints the list of variable names. Defaults to True.
    Returns:
        pandas.DataFrame: A DataFrame containing the data from the CSV file, with columns named appropriately.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is an error reading the file.
    Usage:
        >>> 
            conserved_quantities("path/to/folder")
            import matplotlib.pyplot as plt

            # Create subplots with 3 rows and 3 columns, and adjust the figsize parameter
            fig, axs = plt.subplots(3, 3, figsize=(12, 6))

            # Iterate over the axes and plot the data
            for i, ax in enumerate(axs.flatten()):
                data.iloc[:, i].plot(ax=ax)
                ax.set_ylabel(f'{variables[i+1]}')
                ax.set_xlabel('cycles')

            # Adjust the layout of the subplots
            plt.tight_layout()

            # Show the plot
            plt.show()
        >>>

    """
    file_path = f"{folder}/ConservedQuantities.txt"

    # Define column names based on the provided structure
    column_names = [
        "Cycle",
        "Total internal energy",
        "Variation of total internal energy",
        "Electric energy",
        "Local magnetic energy",
        "Kinetic energy (currently in the domain)",
        "Momentum",
        "Total magnetic energy",
        "Internal magnetic energy",
        "Kinetic energy removed",
        "Electric energy removed",
        "Number of particles of species 0",
        "Total charge of species 0",
        "Kinetic energy of species 0",
        "Number of particles of species 1",
        "Total charge of species 1",
        "Kinetic energy of species 1"
    ]

    # Read the data from the file
    data = pd.read_csv(file_path,  delim_whitespace=True, 
                    comment='#', 
                    header=None)
    # Select only the first len(column_names) columns and assign the names
    data = data.iloc[:, :len(column_names)]
    data.columns = column_names
    if verbose:
        print("variables ", column_names[1:])
    
    return data, column_names

def transform_features(*args, **kwargs):  # backward compat
    from closure.evaluation import transform_features as _f
    return _f(*args, **kwargs)

def transform_targets(*args, **kwargs):  # backward compat
    from closure.evaluation import transform_targets as _f
    return _f(*args, **kwargs)

def compute_loss(*args, **kwargs):  # backward compat
    from closure.evaluation import compute_loss as _f
    return _f(*args, **kwargs)

def evaluate_loss(*args, **kwargs):  # backward compat
    from closure.evaluation import evaluate_loss as _f
    return _f(*args, **kwargs)

def graph_pred_targets(*args, **kwargs):  # backward compat
    from closure.visualization import graph_pred_targets as _f
    return _f(*args, **kwargs)

def pred_ground_targets(*args, **kwargs):  # backward compat
    from closure.evaluation import pred_ground_targets as _f
    return _f(*args, **kwargs)

def plot_pred_targets(*args, **kwargs):  # backward compat
    from closure.visualization import plot_pred_targets as _f
    return _f(*args, **kwargs)

def normalize_input(*args, **kwargs):  # backward compat
    from closure.evaluation import normalize_input as _f
    return _f(*args, **kwargs)

def pred_unnormalize(*args, **kwargs):  # backward compat
    from closure.evaluation import pred_unnormalize as _f
    return _f(*args, **kwargs)

unnormalize_output = pred_unnormalize  # backward compat alias

def prediction2data(*args, **kwargs):  # backward compat
    from closure.evaluation import prediction2data as _f
    return _f(*args, **kwargs)

# The scripts below are adapted from G. Arrò

def scalar_spectrum_2D(field, X, Y):
    """
    Author: Peppe Arrò
    This script calculates the 1D power spectrum for scalar functions
    """
    Lx = X[-1,0]
    Ly = Y[0,-1]
    x = X[:,0]
    y = Y[0,:]
    t = np.arange(field.shape[-1])
    nxc=len(X)-1
    nyc=len(Y)-1
    print(f"{field.shape = }, {nxc = }, {nyc = }, {x.shape = }, {y.shape = }, {t.shape = }, {x[:2] = }, {y[:2] = }")
    # Repeated boundaries must be excluded according to the definition of the FFT.
    field_ft=np.fft.rfft2(field[0:-1,0:-1,:],axes=(0,1))
    print(f"{field_ft.shape = }")
    # 2D power spectrum.
    spec_2D=(abs(field_ft)**2)/((nxc*nyc)**2)
    spec_2D[:,1:-1,:]*=2 # Some modes are doubled to take into account the redundant ones removed by numpy's rfft.
    kx=np.fft.fftfreq(nxc-1,x[1])*2*np.pi
    ky=np.fft.rfftfreq(nyc-1,y[1])*2*np.pi
    print(f"{len(kx) = }, {len(ky) = }")

    # The 1D magnetic field energy spectrum is calculated.
    spec_1D=np.zeros((nxc//2+1,len(t)))

    for iy in range(len(ky)):
        for ix in range(len(kx)):
            try:
                index=round( np.sqrt( (Lx*kx[ix]/(2*np.pi))**2+(Ly*ky[iy]/(2*np.pi))**2 ) )
                if index<=(nxc//2):
                    spec_1D[index,:]+=spec_2D[ix,iy,:]
            except Exception as e:
                print(f"{index = }, {ix = }, {iy = }, {spec_2D.shape = }, {spec_1D.shape = }")
                raise e

    return ky,spec_1D[:-1]

def vector_spectrum_2D(field_x,field_y,field_z, X, Y):
    """
    Author: Peppe Arrò
    This script calculates the 1D power spectrum for vector functions
    """
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
    t = np.arange(field_x.shape[-1])
    nxc=len(X)-1
    nyc=len(Y)-1
    # Repeated boundaries must be excluded according to the definition of the FFT.
    field_x_ft=np.fft.rfft2(field_x[0:-1,0:-1,:],axes=(0,1))
    field_y_ft=np.fft.rfft2(field_y[0:-1,0:-1,:],axes=(0,1))
    field_z_ft=np.fft.rfft2(field_z[0:-1,0:-1,:],axes=(0,1))

    # 2D power spectrum.
    spec_2D=(abs(field_x_ft)**2+abs(field_y_ft)**2+abs(field_z_ft)**2)/((nxc*nyc)**2)
    spec_2D[:,1:-1,:]*=2 # Some modes are doubled to take into account the redundant ones removed by numpy's rfft.
    kx=np.fft.fftfreq(nxc-1,x[1])*2*np.pi
    ky=np.fft.rfftfreq(nyc-1,y[1])*2*np.pi

    # The 1D magnetic field energy spectrum is calculated.
    spec_1D=np.zeros((nxc//2+1,len(t)))

    for iy in range(len(ky)):
        for ix in range(len(kx)):
            try:
                index=round( np.sqrt( (Lx*kx[ix]/(2*np.pi))**2+(Ly*ky[iy]/(2*np.pi))**2 ) )
                if index<=(nxc//2):
                    spec_1D[index,:]+=spec_2D[ix,iy,:]
            except Exception as e:
                print(f"{index = }, {ix = }, {iy = }, {spec_2D.shape = }, {spec_1D.shape = }")
                raise e

    return ky,spec_1D #[:-1]


def get_spectral_index(k,spec,N):
	"""
        Calculate the spectral index by fitting a line to the log-log plot of the given spectrum.
        Parameters:
        k (array-like): The wavenumber array.
        spec (array-like): The spectrum array corresponding to the wavenumbers.
        N (int): The number of points to use in each segment for fitting.
        Returns:
        tuple: A tuple containing:
            - k_red (numpy.ndarray): The reduced wavenumber array, averaged over each segment.
            - slopes (numpy.ndarray): The slopes of the fitted lines, representing the spectral index for each segment.
        """
	from scipy.optimize import curve_fit
	
	def line(x,a,b):
		return a*x+b
	
	X=np.log10(k[1:])
	Y=np.log10(spec[1:])
	
	k_red=[]
	slopes=[]
	#print(k.shape,len(k)//N)
	for i in range(len(k)//N):
		#if i == 0:
		#	print(X[i*N:(i+1)*N].shape, Y[i*N:(i+1)*N].shape)
		p,_=curve_fit(line,X[i*N:(i+1)*N],Y[i*N:(i+1)*N],sigma=Y[i*N:(i+1)*N])
		k_red.append(np.mean(k[i*N+1:(i+1)*N+1]))
		slopes.append(p[0])
	
	return np.array(k_red), np.array(slopes)



def code2alfven(data, X, Y, times, B0x, nb):
    "Rescale code units to Alfven units, using the normalisation  given by B0x and nb"
    VA = B0x/np.sqrt(nb)
    J0 = nb*VA
    p0 = nb*VA**2
    E0 = VA*B0x
    for field_name in ['Bx', 'By', 'Bz']:
        try:
            data[field_name] = data[field_name]/B0x
        except:
            print(f"{field_name} not in data")
    data['Bmagn'] = data['Bmagn']/B0x
    for field_name in ['Ex', 'Ey', 'Ez','EPx', 'EPy', 'EPz','EHallx', 'EHally', 'EHallz','Ohmresx', 'Ohmresy', 'Ohmresz']:
        try:
            data[field_name] = data[field_name]/E0
        except:
            print(f"{field_name} not in data")
    data['Emagn'] = data['Emagn']/E0
    for field_name in ['Jx', 'Jy', 'Jz', 'Jmagn']:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec]/J0
        except:
            print(f"{field_name} not in data")
    for field_name in ['Jtotx', 'Jtoty', 'Jtotz']:
        try:
            data[field_name] = data[field_name]/J0
        except:
            print(f"{field_name} not in data")
    for field_name in ['rho']:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec]/nb
        except:
            print(f"{field_name} not in data")
    for field_name in ['Vx', 'Vy', 'Vz']:
        try:             
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec]/VA
        except:
            print(f"{field_name} not in data")
    for field_name in ['Pxx', 'Pxy', 'Pxz', 'Pyx', 'Pyy', 'Pyz', 'Pzx', 'Pzy', 'Pzz', 'Ppar', 'Pperp']:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec]/p0  
        except:
            print(f"{field_name} not in data")
    for field_name in ['qx', 'qy', 'qz','EFx', 'EFy', 'EFz']:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec]/(p0*VA)
                
        except:
            print(f"{field_name} not in data")
    for field_name in ['gyro_radius']:
        try:
            for spec in data[field_name].keys():
                data[field_name][spec] = data[field_name][spec]/(VA/B0x)
        except:
            print(f"{field_name} not in data")

    return X*np.sqrt(nb), Y*np.sqrt(nb), [t*B0x for t in times]

def do_dot(fx,fy,fz,gx,gy,gz):
	return fx*gx+fy*gy+fz*gz
	
def do_cross(fx,fy,fz,gx,gy,gz):
	return fy*gz-fz*gy, fz*gx-fx*gz, fx*gy-fy*gx	

def get_PS_3D_field(data, x, y, z):
    """
    Get the pressure-strain term and theta
    """
    data['QJ'] = {}
    data['Qomega'] = {}
    data['QD'] = {}
    data['PiD'] = {}
    data['Ptheta'] = {}
    data['PS'] = {}
    data['theta'] = {}
    data['Dxx'] = {}
    data['Dyy'] = {}
    data['Dzz'] = {}
    data['Dxy'] = {}
    data['Dxz'] = {}
    data['Dyz'] = {}
    data['Ppar'] = {}
    data['Pperp'] = {}
    data['P'] = {}
    data['J*(E+VxB)'] = {}
    data['Jtotx'] = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
    data['Jtoty'] = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
    data['Jtotz'] = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)
    E = np.array([data['Ex'], data['Ey'], data['Ez']]).transpose(1,2,3,4,0)
    B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,4,0)
    J2 = data['Jtotx']**2 + data['Jtoty']**2 + data['Jtotz']**2
    data['QJ'] = 0.25*J2/np.mean(J2, axis=(0,1))
    for species in data['rho'].keys():
        J = np.array([data['Jx'][species], data['Jy'][species], data['Jz'][species]]).transpose(1,2,3,4,0)
        V = np.array([data['Vx'][species], data['Vy'][species], data['Vz'][species]]).transpose(1,2,3,4,0)
        data['J*(E+VxB)'][species] = np.sum(J*(E + np.cross(V, B)),axis=-1)
        uxx = np.gradient(data['Vx'][species],x, axis=0, edge_order=2)
        uxy = np.gradient(data['Vx'][species],y, axis=1, edge_order=2)
        uyx = np.gradient(data['Vy'][species],x, axis=0, edge_order=2)
        uyy = np.gradient(data['Vy'][species],y, axis=1, edge_order=2)
        uzx = np.gradient(data['Vz'][species],x, axis=0, edge_order=2)
        uzy = np.gradient(data['Vz'][species],y, axis=1, edge_order=2)
        uxz = np.gradient(data['Vx'][species],z, axis=2, edge_order=2)
        uyz = np.gradient(data['Vy'][species],z, axis=2, edge_order=2)
        uzz = np.gradient(data['Vz'][species],z, axis=2, edge_order=2)
        omega2 = (uzy-uyz)**2 + (uxz-uzx)**2 + (uyx-uxy)**2
        data['Qomega'][species] = 0.25*omega2/np.mean(omega2, axis=(0,1,2))
        data['P'][species]=(data['Pxx'][species]+\
                                data['Pyy'][species]+\
                                    data['Pzz'][species])/3
        data['Ppar'][species] = (data['Pxx'][species]*data['Bx']**2 + data['Pyy'][species]*data['By']**2  + data['Pzz'][species]*data['Bz']**2 + \
                                        2*data['Pxy'][species]*data['Bx']*data['By']+2*data['Pxz'][species]*data['Bx']*data['Bz'] + \
                                            2*data['Pyz'][species]*data['By']*data['Bz'])/(data['By']**2+data['Bx']**2+data['Bz']**2)
        data['Pperp'][species] = (data['Pxx'][species] + data['Pyy'][species] + data['Pzz'][species] - data['Ppar'][species])/2
        data['theta'][species]=uxx+uyy+uzz
        data['PS'][species]=-data['Pxx'][species]*uxx-\
            data['Pxy'][species]*uxy-data['Pxy'][species]*uyx-\
                data['Pyy'][species]*uyy-data['Pxz'][species]*uzx-\
                    data['Pyz'][species]*uzy-data['Pxz'][species]*uxz-\
                        data['Pyz'][species]*uyz-data['Pzz'][species]*uzz
        data['Ptheta'][species]=data['P'][species]*data['theta'][species]
        data['Dxx'][species] = uxx - data['theta'][species]/3
        data['Dyy'][species] = uyy - data['theta'][species]/3
        data['Dzz'][species] = uzz - data['theta'][species]/3
        data['Dxy'][species] = (uxy + uyx)/2
        data['Dxz'][species] = (uxz + uzx)/2
        data['Dyz'][species] = (uyz + uzy)/2
        Dsum = data['Dxx'][species]**2 + data['Dyy'][species]**2 + data['Dzz'][species]**2 +\
            2*(data['Dxy'][species]**2 + data['Dxz'][species]**2 + data['Dyz'][species]**2) 
        data['QD'][species] = 0.25*Dsum/np.mean(Dsum, axis=(0,1,2))
        # Using PiD = - (Pij - Pdelta_ij)Dij
        data['PiD'][species]=-(data['Pxx'][species]-data['P'][species])*(uxx-data['theta'][species]/3)-\
                (data['Pyy'][species]-data['P'][species])*(uyy-data['theta'][species]/3)-\
                        (data['Pzz'][species]-data['P'][species])*(uzz-data['theta'][species]/3)-\
                                data['Pxy'][species]*(uyx+uxy)-\
                                    data['Pxz'][species]*(uzx+uxz)-\
                                        data['Pyz'][species]*(uzy+uyz)

def get_PS_2D_field(data, x, y):
    """
    Get the pressure-strain term and theta
    """
    data['QJ'] = {}
    data['Qomega'] = {}
    data['QD'] = {}
    data['PiD'] = {}
    data['Ptheta'] = {}
    data['PS'] = {}
    data['theta'] = {}
    data['Dxx'] = {}
    data['Dyy'] = {}
    data['Dzz'] = {}
    data['Dxy'] = {}
    data['Dxz'] = {}
    data['Dyz'] = {}
    data['Ppar'] = {}
    data['Pperp'] = {}
    data['P'] = {}
    data['J*(E+VxB)'] = {}
    data['Jtotx'] = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
    data['Jtoty'] = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
    data['Jtotz'] = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)
    E = np.array([data['Ex'], data['Ey'], data['Ez']]).transpose(1,2,3,0)
    B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,0)
    J2 = data['Jtotx']**2 + data['Jtoty']**2 + data['Jtotz']**2
    data['QJ'] = 0.25*J2/np.mean(J2, axis=(0,1))
    for species in data['rho'].keys():
        J = np.array([data['Jx'][species], data['Jy'][species], data['Jz'][species]]).transpose(1,2,3,0)
        V = np.array([data['Vx'][species], data['Vy'][species], data['Vz'][species]]).transpose(1,2,3,0)
        data['J*(E+VxB)'][species] = np.sum(J*(E + np.cross(V, B)),axis=-1)
        uxx = np.gradient(data['Vx'][species],x, axis=0, edge_order=2)
        uxy = np.gradient(data['Vx'][species],y, axis=1, edge_order=2)
        uyx = np.gradient(data['Vy'][species],x, axis=0, edge_order=2)
        uyy = np.gradient(data['Vy'][species],y, axis=1, edge_order=2)
        uzx = np.gradient(data['Vz'][species],x, axis=0, edge_order=2)
        uzy = np.gradient(data['Vz'][species],y, axis=1, edge_order=2)
        omega2 = (uzy)**2 + (-uzx)**2 + (uyx-uxy)**2
        data['Qomega'][species] = 0.25*omega2/np.mean(omega2, axis=(0,1))
        data['P'][species]=(data['Pxx'][species]+\
                                data['Pyy'][species]+\
                                    data['Pzz'][species])/3
        data['Ppar'][species] = (data['Pxx'][species]*data['Bx']**2 + data['Pyy'][species]*data['By']**2  + data['Pzz'][species]*data['Bz']**2 + \
                                        2*data['Pxy'][species]*data['Bx']*data['By']+2*data['Pxz'][species]*data['Bx']*data['Bz'] + \
                                            2*data['Pyz'][species]*data['By']*data['Bz'])/(data['By']**2+data['Bx']**2+data['Bz']**2)
        data['Pperp'][species] = (data['Pxx'][species] + data['Pyy'][species] + data['Pzz'][species] - data['Ppar'][species])/2
        data['theta'][species]=uxx+uyy
        data['PS'][species]=-data['Pxx'][species]*uxx-\
            data['Pxy'][species]*uxy-data['Pxy'][species]*uyx-\
                data['Pyy'][species]*uyy-data['Pxz'][species]*uzx-\
                    data['Pyz'][species]*uzy
        data['Ptheta'][species]=data['P'][species]*data['theta'][species]
        data['Dxx'][species] = uxx - data['theta'][species]/3
        data['Dyy'][species] = uyy - data['theta'][species]/3
        data['Dzz'][species] = -data['theta'][species]/3
        data['Dxy'][species] = (uxy + uyx)/2
        data['Dxz'][species] = uzx/2
        data['Dyz'][species] = uzy/2
        Dsum = data['Dxx'][species]**2 + data['Dyy'][species]**2 + data['Dzz'][species]**2 +\
            2*(data['Dxy'][species]**2 + data['Dxz'][species]**2 + data['Dyz'][species]**2) 
        data['QD'][species] = 0.25*Dsum/np.mean(Dsum, axis=(0,1))
        # Using PiD = - (Pij - Pdelta_ij)Dij
        data['PiD'][species]=-(data['Pxx'][species]-data['P'][species])*\
            (uxx-data['theta'][species]/3)-\
                (data['Pyy'][species]-data['P'][species])*\
                    (uyy-data['theta'][species]/3)-\
                        (data['Pzz'][species]-data['P'][species])*\
                            (-data['theta'][species]/3)-\
                                data['Pxy'][species]*(uyx+uxy)-\
                                    data['Pxz'][species]*(uzx)-\
                                        data['Pyz'][species]*(uzy)

def get_PS_2D(data, x, y):
    """
    Get the pressure-strain term and theta
    """
    for experiment in data.keys():
        get_PS_2D_field(data[experiment], x, y)

def apply_filter(field, density=None, filters = {'name': 'uniform_filter', 'size': 3, 'mode' : 'wrap', 'axes': (0,1)}):
    """
    Apply a specified filter to a given field, optionally using a density field for weighted filtering.
    Parameters:
    -----------
    field : numpy.ndarray
        The input field to which the filter will be applied.
    density : numpy.ndarray, optional
        The density field used for weighted filtering. If provided, it must have the same shape as `field` or be broadcastable to the shape of `field`.
    filters : dict, optional. A dictionary specifying the filter parameters. 
    Returns:
    --------
    numpy.ndarray
        The filtered field. If `density` is provided, the result is a density-weighted filtered field.
    Notes:
    ------
    - The filter function is dynamically retrieved from the `nd` module using the name provided in the `filters` dictionary.
    - If `density` is provided, the function performs a density-weighted filtering.
    - The filtering is applied only to the spatial dimensions specified by the `axes` parameter.
    """

    filters_copy = filters.copy()
    if not isinstance(filters_copy, dict):
        filters_object = getattr(nd, filters_copy)
    else:
        filters_name = filters_copy.pop("name", None)
        filters_object = getattr(nd, filters_name)
        filter_kwargs = filters_copy
        if isinstance(filter_kwargs['axes'], list):
            filter_kwargs['axes'] = tuple(filter_kwargs['axes'])
        if filter_kwargs['axes'] is None or filter_kwargs['axes'] != (0,1):
            print("Filtering targets should be aplied to only spatial dimensions")
    #print(filter_kwargs)
    if density is not None:
        if field.shape == density.shape:
            return filters_object(field*density, **filter_kwargs)/ filters_object(density, **filter_kwargs)
        else: #try to broadcast density assuming that field has one extra axes
            return filters_object(field*density[...,np.newaxis], **filter_kwargs)/ filters_object(density[...,np.newaxis], **filter_kwargs)
    else:
        return filters_object(field, **filter_kwargs)

def scale_filtering(data, x, y, qom, verbose=False,
                    filters = {'name': 'uniform_filter', 'size': 100, 'mode' : 'wrap', 'axes': (0,1)}):
    """
    Applies filters to the input data and computes following fitlered quantities
        E2_bar, B2_bar, Ef_favre, PIuu, PIbb, PS, -Ptheta, and JdotE
        which are appended to the dictionary `data`. Not that this will overwrite any existing keys 
        in `data` with the same names.
    Parameters:
    data (dict): A dictionary containing the experimental data. 
    Returns:
    None: The function modifies the input data dictionary in place by adding filtered and derived quantities.
    Notes:
    - The function computes filtered versions of the magnetic and electric fields.
    - It computes Favre-averaged quantities for velocity components.
    - It calculates the energy densities (kinetic and magnetic).
    - It computes various pressure and interaction terms based on the filtered data.
    - The function assumes the existence of certain keys and structures within the input data dictionary.
    """
    auxiliary = {} 
    for fields in ['Bx', 'By', 'Bz', 'Ex','Ey', 'Ez']:
        auxiliary[f"{fields}_bar"] = apply_filter(data[fields], filters = filters)
      
    for fields in ['Vx', 'Vy', 'Vz', 'Bx', 'By', 'Bz', 'Ex','Ey', 'Ez']:
        auxiliary[f"{fields}_favre"] = {}
        
    data['E2_bar'] = (auxiliary['Ex_bar']**2 + auxiliary['Ey_bar']**2 + auxiliary['Ez_bar']**2)/(8*np.pi)
    data['B2_bar'] = (auxiliary['Bx_bar']**2 + auxiliary['By_bar']**2 + auxiliary['Bz_bar']**2)/(8*np.pi)
    data['Ef_favre'] = {}
    data['PIuu'] = {}
    data['PIbb'] = {}
    data['PS'] = {}
    data['-Ptheta'] = {}
    data['JdotE'] = {}
    auxiliary['rho_bar'] = {}
    B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,0)
    E_bar = np.array([auxiliary['Ex_bar'], auxiliary['Ey_bar'], auxiliary['Ez_bar']]).transpose(1,2,3,0)
    for i, species in enumerate(data['rho'].keys()):
        for fields in ['Vx', 'Vy', 'Vz']:
            auxiliary[f"{fields}_favre"][species] = apply_filter(data[fields][species], density=data['rho'][species], filters = filters)
        for fields in ['Bx', 'By', 'Bz', 'Ex','Ey', 'Ez']:
            auxiliary[f"{fields}_favre"][species] = apply_filter(data[fields], density=data['rho'][species], filters = filters)
        auxiliary['rho_bar'][species] = apply_filter(data['rho'][species], filters = filters) # charge density
        data['Ef_favre'][species] = 0.5*auxiliary['rho_bar'][species]*(auxiliary['Vx_favre'][species]**2 + auxiliary['Vy_favre'][species]**2 + auxiliary['Vz_favre'][species]**2)/qom[i]
        B_favre = np.array([auxiliary['Bx_favre'][species], auxiliary['By_favre'][species], auxiliary['Bz_favre'][species]]).transpose(1,2,3,0)
        E_favre = np.array([auxiliary['Ex_favre'][species], auxiliary['Ey_favre'][species], auxiliary['Ez_favre'][species]]).transpose(1,2,3,0)
        tau_e = E_favre - E_bar
        if verbose:
            print(f"{species = }")
        V_favre = np.array([auxiliary['Vx_favre'][species], auxiliary['Vy_favre'][species], auxiliary['Vz_favre'][species]]).transpose(1,2,3,0)
        data['PIbb'][species] = -auxiliary['rho_bar'][species]*np.sum(tau_e*V_favre, axis=-1)
        data['JdotE'][species] = +auxiliary['rho_bar'][species]*np.sum(E_favre*V_favre, axis=-1)

        V = np.array([data['Vx'][species], data['Vy'][species], data['Vz'][species]]).transpose(1,2,3,0)
        tau_b = apply_filter(np.cross(V, B), density=data['rho'][species], filters = filters) - np.cross(V_favre, B_favre)
        dV_favre = {}
        for component in ['x', 'y', 'z']:
            dV_favre[f"{component}x"] = np.gradient(auxiliary[f'V{component}_favre'][species],x, axis=0, edge_order=2)
            dV_favre[f"{component}y"] = np.gradient(auxiliary[f'V{component}_favre'][species],y, axis=1, edge_order=2)
        data['-Ptheta'][species] = 0
        for component in ['x', 'y', 'z']: #calculating trace
            data['-Ptheta'][species] += apply_filter(data[f'P{component}{component}'][species], filters = filters)
        data['-Ptheta'][species] *= -(dV_favre['xx']+dV_favre['yy'])/3 # divergence of velocity times pressure trace
        data['PIuu'][species] = 0 # equation (21) of Matthaeus, W. H.; Yang, Y.; Wan, M.; Parashar, T. N.; Bandyopadhyay, R.; Chasapis, A.; Pezzi, O.; Valentini, F. Pathways to Dissipation in Weakly Collisional Plasmas. ApJ 2020, 891 (1), 101. https://doi.org/10.3847/1538-4357/ab6d6a. See also Yang, Y.; Matthaeus, W. H.; Roy, S.; Roytershteyn, V.; Parashar, T. N.; Bandyopadhyay, R.; Wan, M. Pressure–Strain Interaction as the Energy Dissipation Estimate in Collisionless Plasma. ApJ 2022, 929 (2), 142. https://doi.org/10.3847/1538-4357/ac5d3e.
        data['PS'][species] = 0
        for component1, component2 in zip(['x', 'x', 'y' ,'y', 'z', 'z'], ['x', 'y', 'x', 'y', 'x', 'y']):
            Pbar = apply_filter(data[f'P{component1}{component2}'][species], filters = filters)
            if verbose:
                print(f"adding: Pbar{component1}{component2} * nabla dVfavre_{component1}d{component2}")
            data['PS'][species] += - Pbar*dV_favre[f"{component1}{component2}"]

            tauu = apply_filter(data[f'V{component1}'][species]*data[f'V{component2}'][species], \
                            density=data['rho'][species], filters = filters) - \
                            auxiliary[f'V{component1}_favre'][species]*auxiliary[f'V{component2}_favre'][species]

            data['PIuu'][species] += - auxiliary['rho_bar'][species]*tauu*dV_favre[f"{component1}{component2}"]/qom[i]
            
        data['PIuu'][species] += - auxiliary['rho_bar'][species]*np.sum(tau_b*V_favre, axis=-1)


def get_T(data, qom):
    """
    Get T, T_perp, T_par
    """
    data['T'] = {}
    data['T_par'] = {}
    data['T_perp'] = {}
    data['beta_par'] = {}
    bx=data['Bx']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
    by=data['By']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
    bz=data['Bz']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
    for i, species in enumerate(data['rho'].keys()):
        data['T'][species]=(data['Pxx'][species]+\
                                data['Pyy'][species]+\
                                    data['Pzz'][species])/(3*data['rho'][species]*np.sign(qom[i]))
        data['T_par'][species]=(data['Pxx'][species]*bx**2+\
            data['Pyy'][species]*by**2+data['Pzz'][species]*bz**2+\
                2*(data['Pxy'][species]*bx*by+data['Pxz'][species]*bx*bz+\
                    data['Pyz'][species]*by*bz))/(data['rho'][species]*np.sign(qom[i]))
        data['T_perp'][species]=(3*data['T'][species]-data['T_par'][species])/2
        data['beta_par'][species] = 8*np.pi*data['T_par'][species]*(data['rho'][species]*np.sign(qom[i]))/(data['Bx']**2 + data['By']**2 + data['Bz']**2)

def get_agyrotropy(data):
    """
    Compute agyrotropy for all species
    """
    data['agyrotropy'] = {}
    for species in data['rho'].keys():
        bx=data['Bx']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
        by=data['By']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
        bz=data['Bz']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
        I1=data['Pxx'][species]+data['Pyy'][species]+data['Pzz'][species]
        I2=data['Pxx'][species]*data['Pyy'][species]+\
            data['Pxx'][species]*data['Pzz'][species]+\
                data['Pyy'][species]*data['Pzz'][species]-\
                    (data['Pxy'][species]**2+data['Pxz'][species]**2+\
    data['Pyz'][species]**2)
        P_par=data['Pxx'][species]*bx**2+data['Pyy'][species]*by**2+\
            data['Pzz'][species]*bz**2+2*(data['Pxy'][species]*bx*by+\
                                            data['Pxz'][species]*bx*bz+\
                                            data['Pyz'][species]*by*bz)
        data['agyrotropy'][species]=1-4*I2/((I1-P_par)*(I1+3*P_par))

def highdiff(data, dx, dy, coeff = None, axis=0, **kwargs):
    """
    Compute the 4th-order central finite difference derivative for a 2D array 
    along either the x or y axis.

    Parameters:
        data (ndarray): Input 2D or higher-dimensional array.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        coeff (ndarray): Coefficients for the finite difference scheme.
            Default is 4th-order central difference coefficients.
        axis (str): Axis along which to compute the derivative (0 or 1).

    Returns:
        ndarray: The derivative along the specified axis.
    """
    # 4th-order finite difference coefficients
    if coeff is None:
        coeff = np.array([-1, 8, 0, -8, 1]) / 12.0
    
    if axis == 0:
        # Compute derivative along the x-axis
        dx_kernel = coeff.reshape((-1,) + (1,) * (data.ndim - 1))  # generalizing reshape (-1,1) for higher dimensions
        return nd.convolve(data, dx_kernel, output=float, **kwargs) / dx
    elif axis == 1:
        # Compute derivative along the y-axis
        dy_kernel = coeff.reshape((1, -1) + (1,) * (data.ndim - 2))   # generalizing reshape (1,-1) for higher dimensions
        return nd.convolve(data, dy_kernel, output=float, **kwargs) / dy
    else:
        raise ValueError("Invalid axis. Use 0 or 1.")

def get_Ohm(data,qom, x,y, coeff=None, small=1e-10):
    """
    E_Ohm = - V x B + J x B / ne - grad P_e / ne
    Compute the electric field and other derived quantities based on the input data.
    This function calculates the electric field, ExB/B^2, EHall, EMHD, and other quantities
    using the provided data dictionary. It also computes the pressure gradient and other
    relevant quantities based on the input data.
    """
    B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,0)
    E = np.array([data['Ex'], data['Ey'], data['Ez']]).transpose(1,2,3,0)
    data['ExB/B^2'] = np.cross(E,B)/(data['Bx']**2+data['By']**2+data['Bz']**2)[...,np.newaxis]
    data['Jtotx'] = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
    data['Jtoty'] = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
    data['Jtotz'] = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)
    J = np.array([data['Jtotx'], data['Jtoty'], data['Jtotz']]).transpose(1,2,3,0)
    data['EHallx'], data['EHally'], data['EHallz'] = (np.cross(J,B)/(-data['rho']['e']+small)[...,np.newaxis]).transpose(3,0,1,2)
    norm = 0
    data['uCMx'] = 0
    data['uCMy'] = 0
    data['uCMz'] = 0
    for i, species in enumerate(data['rho'].keys()):
        data['uCMx'] += (data['rho'][species]/qom[i])*data['Vx'][species]
        data['uCMy'] += (data['rho'][species]/qom[i])*data['Vy'][species]
        data['uCMz'] += (data['rho'][species]/qom[i])*data['Vz'][species]
        norm += data['rho'][species]/qom[i]
    data['uCMx'] /= norm
    data['uCMy'] /= norm
    data['uCMz'] /= norm
    uCM = np.array([data['uCMx'], data['uCMy'], data['uCMz']]).transpose(1,2,3,0)
    data['EMHDx'], data['EMHDy'], data['EMHDz'] = - np.cross(uCM,B).transpose(3,0,1,2)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    #data['EP_x'] = (np.gradient(data['Pxx']['e'],x,axis=0,edge_order=2)+np.gradient(data['Pxy']['e'],y,axis=1,edge_order=2))/data['rho']['e']
    #data['EP_y'] = (np.gradient(data['Pxy']['e'],x,axis=0,edge_order=2)+np.gradient(data['Pyy']['e'],y,axis=1,edge_order=2))/data['rho']['e']
    #data['EP_z'] = (np.gradient(data['Pxz']['e'],x,axis=0,edge_order=2)+np.gradient(data['Pyz']['e'],y,axis=1,edge_order=2))/data['rho']['e']
    data['EPx'] = -(highdiff(data['Pxx']['e'], dx, dy, coeff=coeff, axis=0, mode='wrap') + highdiff(data['Pxy']['e'], dx, dy, coeff=coeff, axis=1, mode='wrap'))/(-data['rho']['e']+small) # density in ECsim is negative (electron charge density)
    data['EPy'] = -(highdiff(data['Pxy']['e'], dx, dy, coeff=coeff, axis=0, mode='wrap') + highdiff(data['Pyy']['e'], dx, dy, coeff=coeff, axis=1, mode='wrap'))/(-data['rho']['e']+small) # density in ECsim is negative (electron charge density)
    data['EPz'] = -(highdiff(data['Pxz']['e'], dx, dy, coeff=coeff, axis=0, mode='wrap') + highdiff(data['Pyz']['e'], dx, dy, coeff=coeff, axis=1, mode='wrap'))/(-data['rho']['e']+small) # density in ECsim is negative (electron charge density)

    data['mVgradVx/e'] = highdiff(data['Vx']['e'], dx, dy, coeff=coeff, axis=0, mode='wrap')*data['Vx']['e']/qom[0] + \
                                highdiff(data['Vx']['e'], dx, dy, coeff=coeff, axis=1, mode='wrap')*data['Vy']['e']/qom[0]
    data['mVgradVy/e'] = highdiff(data['Vy']['e'], dx, dy, coeff=coeff, axis=0, mode='wrap')*data['Vx']['e']/qom[0] + \
                                highdiff(data['Vy']['e'], dx, dy, coeff=coeff, axis=1, mode='wrap')*data['Vy']['e']/qom[0]
    data['mVgradVz/e'] = highdiff(data['Vz']['e'], dx, dy, coeff=coeff, axis=0, mode='wrap')*data['Vx']['e']/qom[0] + \
                                highdiff(data['Vz']['e'], dx, dy, coeff=coeff, axis=1, mode='wrap')*data['Vy']['e']/qom[0]
    
def get_J_perp(data, x,y, coeff=None):
    """
    Calculate the perpendicular current contribution from pressure gradients and curvature
    """
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,0)
    data['gradPperpx'] = highdiff(data['Pperp']['e'], dx, dy, coeff=coeff, axis=0, mode='wrap')
    data['gradPperpy'] = highdiff(data['Pperp']['e'], dx, dy, coeff=coeff, axis=1, mode='wrap')
    data['gradPperpz'] = np.zeros_like(data['gradPperpx'])
    gradPperp = np.array([data['gradPperpx'], data['gradPperpy'], data['gradPperpz']]).transpose(1,2,3,0)
    data['cross(B,DPperp)/B^2'] = np.cross(B, gradPperp)/np.sum(B**2, axis=3, keepdims=True)
    data['b'] = B / np.sqrt(np.sum(B**2, axis=3, keepdims=True))
    print(f"data['b'] shape: {data['b'].shape}")
    data['b*Db'] = data['b'][...,0,np.newaxis]*highdiff(data['b'], dx, dy, coeff=coeff, axis=0, mode='wrap') + \
                data['b'][...,1,np.newaxis]*highdiff(data['b'], dx, dy, coeff=coeff, axis=1, mode='wrap')
    data['(Ppar - Pperp) cros(B, b*Db)/B^2'] = (data['Ppar']['e'] - data['Pperp']['e'])[...,np.newaxis]*np.cross(B, data['b*Db'])/np.sum(B**2, axis=3, keepdims=True)

def get_Az(x,y,data):
    """
    Compute the vector potential component Az based on the input magnetic field components Bx and By.
    This function calculates the Az component of the vector potential using the provided
    magnetic field data (Bx and By) and spatial coordinates (x and y). The calculation
    is performed using numerical integration along the x and y axes.
    Parameters:
    -----------
    x : numpy.ndarray
        1D array representing the x-coordinates of the grid points.
    y : numpy.ndarray
        1D array representing the y-coordinates of the grid points.
    data : dict
        Dictionary containing the magnetic field components:
        - 'Bx': 3D numpy array representing the x-component of the magnetic field.
        - 'By': 3D numpy array representing the y-component of the magnetic field.
    Modifies:
    ---------
    data : dict
        Adds a new key 'Az' to the input dictionary, which contains the computed
        3D numpy array of the Az component of the vector potential.
    Notes:
    ------
    - The function assumes that the input magnetic field components ('Bx' and 'By')
        are defined on a regular grid.
    - The grid spacing is computed as the difference between consecutive elements
        in the x and y arrays (dx and dy).
    - The integration is performed using a trapezoidal rule along the respective axes.
    Example:
    --------
    >>> ut.get_Az(X[:,0],Y[0,:],data)
    >>> print(data['Az'])  # Access the computed Az component
    """
    
    Nx=data['Bx'].shape[0]
    Ny=data['Bx'].shape[1]
    Nz=data['Bx'].shape[2]
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    f=np.zeros((Nx,Ny,Nz))
    g=np.zeros((Nx,Ny,Nz))
    
    for iy in range(1,Ny):
        g[:,iy,:]=g[:,iy-1,:]+(data['Bx'][:,iy-1,:]+data['Bx'][:,iy,:])*dy/2
        
    for iy in range(0,Ny):
        for ix in range(1,Nx):
            f[ix,iy,:]=f[ix-1,iy,:]-(data['By'][ix-1,0,:]+data['By'][ix,0,:])*dx/2    
    data['Az'] = f+g

    



def get_W(data):
	"""
	Get W
	"""
	for experiment in data.keys():
		data[experiment]['W'] = {}
		for species in data[experiment]['rho'].keys():
			data[experiment]['W'][species] = do_dot(data[experiment]['Ex'],data[experiment]['Ey'],data[experiment]['Ez'],\
						data[experiment]['Jx'][species],data[experiment]['Jy'][species],data[experiment]['Jz'][species])

def get_D(data):
	"""
	Get D
	"""
	for experiment in data.keys():
		data[experiment]['D'] = {}
		for species in data[experiment]['rho'].keys():
			data[experiment]['D'][species] = do_dot(data[experiment]['Jx'][species],data[experiment]['Jy'][species],data[experiment]['Jz'][species],\
						data[experiment]['Jx'][species],data[experiment]['Jy'][species],data[experiment]['Jz'][species])


