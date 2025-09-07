import numpy as np
import os
import re
from . import utilities as ut
import scipy.ndimage as nd
import pickle

import logging
logger = logging.getLogger(__name__)

# Define global default values
DEFAULT_CHOOSE_X = None
DEFAULT_CHOOSE_Y = None
DEFAULT_CHOOSE_Z = None
DEFAULT_INDEXING = 'ij'
DEFAULT_VERBOSE = False


def read_fieldname(files_path,filenames,fieldname,choose_x=DEFAULT_CHOOSE_X, choose_y=DEFAULT_CHOOSE_Y, 
                   choose_z=DEFAULT_CHOOSE_Z, indexing=DEFAULT_INDEXING, verbose=DEFAULT_VERBOSE, filters=None):
    """
    Read a specific field from multiple files and return a subset of the field.

    Parameters:
    - files_path (str): The path to the directory containing the files.
    - filenames (list): A list of filenames to read from.
    - fieldname (str): The name of the field to read.
    - choose_x (list, optional): A list specifying the range of indices to select along the x-axis. Defaults to None.
    - choose_y (list, optional): A list specifying the range of indices to select along the y-axis. Defaults to None.
    - choose_z (list, optional): A list specifying the range of indices to select along the z-axis. Defaults to None.
    - indexing (string, defaults to 'ij'): A flag indicating how to transpose the field. If 'ij', the field is 
        transposed to have the x-axis as the first index, the y-axis as the second index, and the z-axis as the third index. 
        If 'xy', the field is transposed to have the x-axis as the second index, the y-axis as the first index, and the z-axis as the third index.
    - verbose (bool, optional): A flag indicating whether to logger.info debug information.
    - filters (dict, optional): A dictionary containing the name of the filter to apply and the arguments to pass to the filter.
        Usage: filters = {'name': 'gaussian_filter', 'sigma': 1, 'axes': (1,2)}
                filters = [{'name': 'gaussian_filter', 'sigma': 1, 'axes': (1,2)},
                           {'name': 'zoom', 'zoom': (0.25, 0.25), 'mode' : 'grid-wrap'}]

    Returns:
    - numpy.ndarray: A subset of the field, with the z-dimension removed.

    """
    field = []
    if not isinstance(filenames, list):
        filenames = [filenames]
    for filename in filenames:
        try:
            if filename.endswith(".h5"):
                import h5py
                with h5py.File(files_path + filename, "r") as n:
                    field.append(np.array(n[f"/Step#0/Block/{fieldname}/0"]))
            elif filename.endswith(".h5.pkl"):
                with open(files_path + filename, "rb") as n:
                    field.append(pickle.load(n)[fieldname])
            else:
                raise FileNotFoundError(f"Neither {filename} nor {filename}.pkl found in {files_path}")
            if choose_x is None:
                choose_x = [0,field[-1].shape[2]-1]
            if choose_y is None:
                choose_y = [0,field[-1].shape[1]-1]
            if choose_z is None:
                choose_z = [0,field[-1].shape[0]]
        except Exception as e:
            if verbose:
                logger.warning(f"Failed to read {fieldname} from {filename} using path {files_path}")
            raise e
    if indexing == 'ij': 
        a = np.transpose(np.array(field), (3, 2, 1, 0))[choose_x[0]:choose_x[1],choose_y[0]:choose_y[1],choose_z[0]:choose_z[1],:]
    elif indexing == 'xy':
        a = np.transpose(np.array(field), (2, 3, 1, 0))[choose_y[0]:choose_y[1],choose_x[0]:choose_x[1],choose_z[0]:choose_z[1],:]
    else:
        a = np.array(field)[choose_x[0]:choose_x[1],choose_y[0]:choose_y[1],choose_z[0]:choose_z[1],:]
    if a.shape[2] <= 2: #  if z axis has length 2 or less it should be dropped
        a = np.squeeze(a[...,0,:])
    else:
        a = np.squeeze(a) # transposing means swapping x and y and remove z
    #logger.info(f"{filename}, {fieldname}, {a.shape = }")
    if filters is not None:
        if not isinstance(filters, list):
            filters = [filters]
        for filteri in filters: # apply all filters in succession
            if verbose:
                logger.info(f"Filtering {fieldname} from {filename} with {filteri['name']}")
            filters_copy = filteri.copy()
            filters_name = filters_copy.pop("name", None)
            filters_object = getattr(nd, filters_name)
            filter_kwargs = filters_copy
            for _, kwarg in filter_kwargs.items():
                if  isinstance(kwarg, list):
                    kwarg = tuple(kwarg)  #  configs usually provide lists, but we need tuples
            a = filters_object(a, **filter_kwargs)
            if verbose:
                logger.info(f"Resulting shape {a.shape}")
    return a

def apply_filters(field, filters, fieldname=None, filename=None, verbose=DEFAULT_VERBOSE):
    """
    Apply a sequence of scipy.ndimage filters to a numpy array.

    Parameters:
    - field (np.ndarray): The array to filter.
    - fieldname (str, optional): Name of the field (for logging).
    - filename (str, optional): Name of the file (for logging).
    - verbose (bool): Whether to log filter application.
    - filters (dict, optional): A dictionary containing the name of the filter to apply and the arguments to pass to the filter.
        Usage: filters = {'name': 'gaussian_filter', 'sigma': 1, 'axes': (0,1)}
                filters = [{'name': 'gaussian_filter', 'sigma': 1, 'axes': (0,1)},
                           {'name': 'zoom', 'zoom': (0.25, 0.25), 'mode' : 'grid-wrap'}]
    Example usage:
    Bz_filtered = data['Bz'] - rp.apply_filters(data['Bz'], filters=[{'name': 'gaussian_filter', 'sigma': 10, 'axes': (0,1)}])
    TODO: merge with function read_fieldname

    Returns:
    - np.ndarray: The filtered array.
    """
    if filters is None:
        return field
    if not isinstance(filters, list):
        filters = [filters]
    a = field
    for filteri in filters:
        if verbose and fieldname is not None and filename is not None:
            logger.info(f"Filtering {fieldname} from {filename} with {filteri['name']}")
        filters_copy = filteri.copy()
        filters_name = filters_copy.pop("name", None)
        filters_object = getattr(nd, filters_name)
        # Convert list arguments to tuples for axes, etc.
        for k, v in filters_copy.items():
            if isinstance(v, list):
                filters_copy[k] = tuple(v)
        a = filters_object(a, **filters_copy)
        if verbose and fieldname is not None and filename is not None:
            logger.info(f"Resulting shape {a.shape}")
    return a


def build_XY(files_path,choose_x=DEFAULT_CHOOSE_X, choose_y=DEFAULT_CHOOSE_Y, choose_z=DEFAULT_CHOOSE_Z, indexing=DEFAULT_INDEXING):
    # Read qom, Lx, Ly, Lz, nxc, nyc, nzc and dt from the SimulationData.txt file.
    try:
        f=open(files_path+"SimulationData.txt","r")
    except Exception:
        # Remove the last folder from files_path
        files_path = os.path.dirname(os.path.normpath(files_path)) + os.sep
        f = open(files_path + "SimulationData.txt", "r")
    content=f.readlines()
    f.close()
    # TODO: deal with qom in more serious way, so that it is readed from the correct folder if the filenames are from different folders
    qom=[]
    for n in content:
        if "QOM" in n:
            qom.append(float(re.split("=",re.sub(" |\n","",n))[-1]))
        if "x-Length" in n:
            Lx=float(re.split("=",re.sub(" |\n","",n))[1])
        if "y-Length" in n:
            Ly=float(re.split("=",re.sub(" |\n","",n))[1])
        if "z-Length" in n:
            Lz=float(re.split("=",re.sub(" |\n","",n))[1])
        if "Number of cells (x)" in n:
            nxc=int(re.split("=",re.sub(" |\n","",n))[1])
        if "Number of cells (y)" in n:
            nyc=int(re.split("=",re.sub(" |\n","",n))[1])
        if "Number of cells (z)" in n:
            nzc=int(re.split("=",re.sub(" |\n","",n))[1])
        if "Time step" in n:
            dt=float(re.split("=",re.sub(" |\n","",n))[1])

    # The x, y and z axes are set.
    x=np.linspace(0,Lx,nxc+1)
    y=np.linspace(0,Ly,nyc+1)
    z=np.linspace(0,Lz,nzc+1)
    
    if choose_x is None:
        choose_x = [0,nxc]
    if choose_y is None:
        choose_y = [0,nyc]
    if choose_z is None:
        choose_z = [0,nzc]       
    if isinstance(choose_x[0],list):
        if isinstance(choose_y[0],list):
            raise ValueError("choose_x and choose_y must be of the same type")
        X = []
        Y = []
        if nzc > 1:
            Z = []
        for i in range(len(choose_x)): # deal with the situation where the user wants to extract multiple regions
            assert len(choose_x) == len(choose_y), "choose_x and choose_y must have the same length"
            if nzc > 1:
                assert len(choose_x) == len(choose_z), "choose_x and choose_y must have the same length"
                X_i, Y_i, Z_i = np.meshgrid(x[choose_x[i][0]:choose_x[i][1]], y[choose_y[i][0]:choose_y[i][1]], z[choose_z[i][0]:choose_z[i][1]], indexing=indexing)
                X.append(X_i)
                Y.append(Y_i)
                Z.append(Z_i)
            else:
                X_i, Y_i = np.meshgrid(x[choose_x[i][0]:choose_x[i][1]], y[choose_y[i][0]:choose_y[i][1]], indexing=indexing)
                X.append(X_i)
                Y.append(Y_i)
        X = np.concatenate(X,axis=1)
        Y = np.concatenate(Y,axis=1) 
        if nzc > 1:
            Z = np.concatenate(Z,axis=1)
    else:
        if nzc > 1:
            X, Y, Z = np.meshgrid(x[choose_x[0]:choose_x[1]], y[choose_y[0]:choose_y[1]], z[choose_z[0]:choose_z[1]], indexing=indexing)
        else:
            X, Y = np.meshgrid(x[choose_x[0]:choose_x[1]], y[choose_y[0]:choose_y[1]], indexing=indexing)
    
    if nzc > 1:
        return X, Y, Z
    else:
        return X, Y


def read_features_targets(files_path, filenames, fields_to_read=None, request_features = None, request_targets = None, 
               choose_species=None,choose_x=DEFAULT_CHOOSE_X, choose_y=DEFAULT_CHOOSE_Y, choose_z=DEFAULT_CHOOSE_Z, features_dtype = np.float32, targets_dtype = np.float32,  verbose=DEFAULT_VERBOSE):
    """
    Reads and extracts features and targets from simulation data files.
        # Read qom, Lx, Ly, Lz, nxc, nyc, nzc and dt from the SimulationData.txt file.

    Parameters:
        files_path (str): The path to the directory containing the simulation data files.
        filenames (list): A list of filenames to read from.
        fields_to_read (list, optional): A list of fields to read from the files. If None, all fields will be read.
        request_features (list, optional): A list of features to extract from the fields. If None, all fields will be considered as features.
        request_targets (list, optional): A list of targets to extract from the fields. If None, all fields will be considered as targets.
        choose_species (str, optional): The species to choose from the fields. 
        choose_x (tuple, optional): The range of x-coordinates to choose from. If None, all x-coordinates will be considered.
        choose_y (tuple, optional): The range of y-coordinates to choose from. If None, all y-coordinates will be considered.
        choose_z (tuple, optional): The range of z-coordinates to choose from. If None, all z-coordinates will be considered.
        features_dtype (dtype, optional): The data type to use for the extracted features.
        targets_dtype (dtype, optional): The data type to use for the extracted targets.
        verbose (bool, optional): Whether to print verbose output during the extraction process.

    Returns:
        features (ndarray): An array containing the extracted features.
        targets (ndarray): An array containing the extracted targets.
    """
    try: # looks in the specific folder, or in the root:
        f=open(files_path+filenames[0].rsplit("/",1)[0]+"/SimulationData.txt","r")
    except Exception:
        f=open(files_path+"SimulationData.txt","r")
   
    content=f.readlines()
    f.close()

    qom=[]
    for n in content:
        if "QOM" in n:
            qom.append(float(re.split("=",re.sub(" |\n","",n))[-1]))
        if "x-Length" in n:
            Lx=float(re.split("=",re.sub(" |\n","",n))[1])
        if "y-Length" in n:
            Ly=float(re.split("=",re.sub(" |\n","",n))[1])
        #if "z-Length" in n:
        #    Lz=float(re.split("=",re.sub(" |\n","",n))[1])
        if "Number of cells (x)" in n:
            nxc=int(re.split("=",re.sub(" |\n","",n))[1])
        if "Number of cells (y)" in n:
            nyc=int(re.split("=",re.sub(" |\n","",n))[1])
        if "Number of cells (z)" in n:
            nzc=int(re.split("=",re.sub(" |\n","",n))[1])
        if "Time step" in n:
            dt=float(re.split("=",re.sub(" |\n","",n))[1])

    if choose_x is not None and isinstance(choose_x[0],list):
        if not isinstance(choose_y[0],list):
            raise ValueError("choose_x and choose_y must be of the same type")
        features = []
        targets = []
        if choose_z is None:
            choose_z = [None]*len(choose_x)
        for i in range(len(choose_x)): # deal with the situation where the user wants to extract multiple regions
            assert len(choose_x) == len(choose_y), "choose_x and choose_y must have the same length"
            
            features.append(read_files(files_path, filenames, fields_to_read, qom, features_dtype, 
                          extract_fields=ut.species_to_list(request_features), choose_species=choose_species, 
                          choose_x=choose_x[i], choose_y=choose_y[i], choose_z=choose_z[i], verbose=verbose))
            if verbose:
                logger.info(f"{features[-1].shape =}")
            
            targets.append(read_files(files_path, filenames, fields_to_read, qom, targets_dtype,
                            extract_fields=ut.species_to_list(request_targets), choose_species=choose_species, 
                            choose_x=choose_x[i], choose_y=choose_y[i], choose_z=choose_z[i], verbose=verbose)) 
            if verbose:
                logger.info(f"{targets[-1].shape =}")
        features = np.concatenate(features,axis=2)
        targets = np.concatenate(targets,axis=2) 
    else:
        features = read_files(files_path, filenames, fields_to_read, qom, features_dtype, 
                            extract_fields=ut.species_to_list(request_features), choose_species=choose_species, 
                            choose_x=choose_x, choose_y=choose_y, choose_z=choose_z, verbose=verbose)
        targets = read_files(files_path, filenames, fields_to_read, qom, targets_dtype, 
                            extract_fields=ut.species_to_list(request_targets), choose_species=choose_species, 
                            choose_x=choose_x, choose_y=choose_y, choose_z=choose_z, verbose=verbose)

    return features, targets

def read_files(files_path, filenames, fields_to_read, qom, dtype, extract_fields=None, choose_species=None, choose_x=DEFAULT_CHOOSE_X, 
               choose_y=DEFAULT_CHOOSE_Y, choose_z=DEFAULT_CHOOSE_Z, verbose=DEFAULT_VERBOSE):
    out2 = []
    for filename in filenames:
        out = []
        data = read_data(files_path,filename,fields_to_read,qom,
                                    choose_species=choose_species,choose_x=choose_x,choose_y=choose_y,choose_z=choose_z, verbose=verbose)
        # TODO: Introduce something that check that the input of extract_fields is correct, e.g. `Jx` does not exist
        for extract_field_index in extract_fields:
            if isinstance(extract_field_index, list):
                try:
                    out.append(data[extract_field_index[0]][extract_field_index[1]])
                except Exception as e:
                    logger.info(f"Failed to extract {extract_field_index = }")
                    logger.info(f"Available fields are {data[extract_field_index[0]] = }")
                    logger.info(f"Available data keys are {data.keys() = }")
                    raise e
            else:
                #logger.info(data[extract_field_index])
                out.append(data[extract_field_index])
        out2.append(np.array(out))
    return np.array(out2, dtype=dtype).transpose(0,2,3,1)  # we want to have the time as the first index, then x, then y, then the field

def read_data(files_path, filenames, fields_to_read, qom, choose_species=None, choose_x=DEFAULT_CHOOSE_X, choose_y=DEFAULT_CHOOSE_Y, 
              choose_z=DEFAULT_CHOOSE_Z, verbose=DEFAULT_VERBOSE, small=1e-10, **kwargs):
    """
    Reads and processes data from files.

    Parameters:
    - files_path (str): The path to the files.
    - filenames (list): A list of filenames to read.
    - fields_to_read (dict): A dictionary indicating which fields to read.
    - qom (list): A list of charge-to-mass ratios for each species.
    - choose_species (list): A list of species to choose.
    - choose_x (float): The x-coordinates to choose.
    - choose_y (float): The y-coordinates to choose.
    - choose_z (float): The z-coordinates to choose.
    - verbose (bool): A flag indicating whether to logger.info debug information.
    - small (float): A small number to avoid division by zero, e.g. Jx/rho

    Returns:
    - data (dict): A dictionary containing the processed data.

    Names of fields:
    - Bx, By, Bz: The magnetic field components.
    - Ex, Ey, Ez: The electric field components.
    - Bx_ext, By_ext, Bz_ext: The external magnetic field components.
    - divB: The divergence of the magnetic field.
    - rho: The charge density.
    - N: The number of particles per cell in the particle in cell simulation.
    - Qrem: The remaining charge in the particle in cell simulation.???????????
    - Jx, Jy, Jz: The current density components.
    - Pxx, Pxy, Pxz, Pyy, Pyz, Pzz: The pressure tensor components.
    - PIxx, PIxy, PIxz, PIyy, PIyz, PIzz: The stress tensor components.
    - Ppar, Pperp: The parallel and perpendicular pressure.
    - q: The heat flux.
    
    """
    #choose_species_new = ut.append_index_to_duplicates(choose_species) 
    #dublicatespecies = ut.get_duplicate_indices(choose_species)
    data = {}
    # The magnetic and electric field is read.
    for fields in ['B', 'E']:
        if fields_to_read[fields]:
            if verbose:
                logger.info(f"loading {fields}")
            for component in ['x','y','z']:
                data[f'{fields}{component}'] = read_fieldname(files_path,filenames,f"{fields}{component}",choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
            try:    
                data[f'{fields}magn'] = np.sqrt(data[f'{fields}x']**2 + data[f'{fields}y']**2 + data[f'{fields}z']**2)
            except Exception as e:
                logger.info(f"{fields}magn failed")
                raise e
        if fields_to_read[f"{fields}_ext"]:
            for component in ['x','y','z']:
                data[f'B{component}_ext'] = read_fieldname(files_path,filenames,f"{fields}{component}_ext",choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
    # The divergence of B is read.
    if fields_to_read["divB"]:
        if verbose:
                logger.info(f"loading divB")
        data['divB'] = read_fieldname(files_path,filenames,'divB',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
    for fields in ['rho', 'N', 'Qrem']:
        if fields_to_read[fields]:
            if verbose:
                logger.info(f"loading {fields}")
            data[fields] = {}
            for i, species in enumerate(choose_species): # Care must be taken that these the only species and they are actually correctly labeled
                if species is not None:
                    if species in data[fields]: # we sum over identical species
                        data[fields][species] += read_fieldname(files_path,filenames,fields+f'_{i}',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
                    else:
                        data[fields][species] = read_fieldname(files_path,filenames,f'{fields}_{i}',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)


    if fields_to_read["J"]:
        data['Jx'], data['Jy'], data['Jz'] = {}, {}, {}
        if fields_to_read['rho']:
            data['Vx'], data['Vy'], data['Vz'] = {}, {}, {}
        if verbose:
            logger.info(f"loading J")
        for component in ['x','y','z']:
            for i, species in enumerate(choose_species):
                if species is not None:
                    if species in data[f'J{component}']: # we sum over identical species
                        data[f'J{component}'][species] += read_fieldname(files_path,filenames,f'J{component}_{i}',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
                    else:
                        data[f'J{component}'][species] = read_fieldname(files_path,filenames,f'J{component}_{i}',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
            if fields_to_read['rho']:
                for species in data[f'J{component}'].keys():
                    data[f'V{component}'][species] = data[f'J{component}'][species]/(data['rho'][species]+small)
        data['Jmagn'] = {}
        data['Jtotx'] = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
        data['Jtoty'] = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
        data['Jtotz'] = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)
        if 'Vx' in data.keys():
            data['Vmagn'] = {}
        for species in data[f'J{component}'].keys():
            if species is not None:
                data['Jmagn'][species] = np.sqrt(data['Jx'][species]**2 + data['Jy'][species]**2 + data['Jz'][species]**2)
                if 'Vx' in data.keys():
                    data['Vmagn'][species] = np.sqrt(data['Vx'][species]**2 + data['Vy'][species]**2 + data['Vz'][species]**2)
                

    # The diagonal and offdiagonal part of the pressure is calculated (to do so you need to read rho and J first).
    if fields_to_read["P"] or fields_to_read["PI"]:
        if verbose:
            logger.info(f"loading P and/or PI")
        for component_1 in ['x','y','z']:
            for component_2 in ['x','y','z']:
                data[f'PI{component_1}{component_2}'] = {}
                data[f'P{component_1}{component_2}'] = {}

                for i, species in enumerate(choose_species):
                    if species is not None:
                        try:
                            if species in data[f'PI{component_1}{component_2}']:
                                data[f'PI{component_1}{component_2}'][species] += read_fieldname(files_path,filenames,f'P{component_1}{component_2}_{i}',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
                            else:
                                data[f'PI{component_1}{component_2}'][species] = read_fieldname(files_path,filenames,f'P{component_1}{component_2}_{i}',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
                        except:
                            if verbose:
                                logger.info(f'Component P{component_1}{component_2} for species {species} missing because tensor is symmetric')
                for species in data[f'PI{component_1}{component_2}']: # because now the number of species has potentially changed
                    i = choose_species.index(species)
                    data[f'P{component_1}{component_2}'][species]  = (data[f'PI{component_1}{component_2}'][species] - \
                                data[f'J{component_1}'][species]*data[f'J{component_2}'][species]/(data[f'rho'][species]+small))/qom[i]

                if not fields_to_read["P"]:
                    del data[f'P{component_1}{component_2}']
                if not fields_to_read["PI"]:
                    del data[f'PI{component_1}{component_2}']  
                       
        if fields_to_read["PI"]:
            for species in data[f'PI{component_1}{component_2}']:
                if species in data['PIxy']:
                    data['PIyx'][species] = data['PIxy'][species]
                if species in data['PIxz']:
                    data['PIzx'][species] = data['PIxz'][species]
                if species in data['PIyz']:
                    data['PIzy'][species] = data['PIyz'][species]
        if fields_to_read["P"]:
            data['Ppar'], data['Pperp'] = {}, {}
            for species in data[f'P{component_1}{component_2}']:
                if species in data['Pxy']:
                    data['Pyx'][species] = data['Pxy'][species]
                if species in data['Pxz']:
                    data['Pzx'][species] = data['Pxz'][species]
                if species in data['Pyz']:
                    data['Pzy'][species] = data['Pyz'][species]
                if verbose:
                    logger.info(f"loading Ppar and Pperp")
                try:
                    data['Ppar'][species] = (data['Pxx'][species]*data['Bx']**2 + data['Pyy'][species]*data['By']**2  + data['Pzz'][species]*data['Bz']**2 + \
                                        2*data['Pxy'][species]*data['Bx']*data['By']+2*data['Pxz'][species]*data['Bx']*data['Bz'] + \
                                            2*data['Pyz'][species]*data['By']*data['Bz'])/(data['By']**2+data['Bx']**2+data['Bz']**2)
                except Exception as e:
                    logger.warning(f"Failed to calculate Ppar for {species} likely due to missing fields, see: {e}")
                try:
                    data['Pperp'][species] = (data['Pxx'][species] + data['Pyy'][species] + data['Pzz'][species] - data['Ppar'][species])/2
                except Exception as e:
                    logger.warning(f"Failed to calculate Pperp for {species} likely due to missing fields, see: {e}")
        if "gyro_radius" in fields_to_read and fields_to_read["gyro_radius"]:
            try:
                data['gyro_radius'] = {}
                for species in data['rho']:
                    i = choose_species.index(species)
                    p = data['Pxx'][species]+data['Pyy'][species]+data['Pzz'][species]
                    vth=np.sqrt(np.abs(p/(data['rho'][species]+small)*qom[i]))
                    data['gyro_radius'][species] = np.abs(vth/(qom[i]*data['Bmagn']))
            except Exception as e:
                logger.warning(f"Failed to calculate gyro_radius, see: {e}")

    # The heat flux is calculated (to do so you need to read rho, J and P first).
    if fields_to_read["Heat_flux"]:
        if verbose:
            logger.info(f"loading q")
        for component in ['x','y','z']:
            data[f'EF{component}'] = {}
            for i, species in enumerate(choose_species):
                if species is not None:
                    if species in data[f'EF{component}']:
                        data[f'EF{component}'][species] += read_fieldname(files_path,filenames,f'EF{component}_{i}',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
                    else:
                        data[f'EF{component}'][species] = read_fieldname(files_path,filenames,f'EF{component}_{i}',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
            #logger.info(f"{data[f'EF{component}'].keys() = }")
            try:
                data[f'q{component}'] = {}
                for species in data[f'EF{component}'].keys():
                    i = choose_species.index(species)
                    data[f'q{component}'][species] =  data[f'EF{component}'][species] - \
                        (data['Jx'][species]**2+data['Jy'][species]**2+data['Jz'][species]**2)*data[f'J{component}'][species]/(2*qom[i]*data[f'rho'][species]**2+small) - \
                        (data['Pxx'][species] + data[f'Pyy'][species] + data[f'Pzz'][species])*data[f'J{component}'][species]/(2*data['rho'][species]+small) - \
                        (data['Jx'][species]*data[f'Px{component}'][species] + data['Jy'][species]*data[f'Py{component}'][species] + data['Jz'][species]*data[f'Pz{component}'][species])/(data['rho'][species]+small)
            except Exception as e:
                logger.warning(f"Failed to calculate q{component} see: {e}")
                #logger.info(f"{data[f'q{component}'].keys() = }")
            if 'EF' not in fields_to_read or not fields_to_read['EF']:
                del data[f'EF{component}']
    return data

def get_exp_times(experiments, files_path, fields_to_read, choose_species=None, choose_times=None,choose_x=DEFAULT_CHOOSE_X, choose_y=DEFAULT_CHOOSE_Y, choose_z=DEFAULT_CHOOSE_Z, 
                  verbose=DEFAULT_VERBOSE, **kwargs):
    """
    Retrieves data from experiments and returns the data structure stored as a dictionary along with the corresponding meshgrid.

    Parameters:
    - experiments (list): A list of experiment names. Each experiment is a directory containing the experiment files.
    - files_path (str): The path to the directory containing the experiment files. The experiment directories are subdirectories of this directory.
    - fields_to_read (list of bools): A list of bools corresponding to the condition of whether or not to read a specific
                field names to read from the files. 
    - qom (array of floats): The charge-to-mass ratio in the PIC units for each species.
    - choose_species (list): A list of species indices to choose.  # the ones which have directive None will be ignored, the ones which have same name will be summed over
    - choose_times (list): A list of time indices to choose. If None is given, all times are chosen. If an integer is given, all times before that one are ignored.
         list specific timeshots are chosen,  i.e. [0, 1, 5], otherwise choose_times = None means take all times
    - choose_x (list): A list specifying the range of x indices to choose. If None is given, the whole range is chosen.
    - choose_y (list): A list specifying the range of y indices to choose. If None is given, the whole range is chosen.
    - choose_z (list): A list specifying the range of z indices to choose. If None is given, the whole range is chosen.
    - verbose (bool): A flag indicating whether to logger.info debug information when reading data such as which fields are being imported.

    Returns:
    - data (dict): A dictionary containing the retrieved data for each experiment.
    - X (ndarray): The meshgrid of x values.
    - Y (ndarray): The meshgrid of y values.
    - (optional) Z (ndarray): The meshgrid of z values.
    - qom (list): A list of charge-to-mass ratios for each species.
    - times (list): A list of times corresponding to the data.
    
    Example:
    >>>
        choose_species = ['e1',None,'e2',None] # the ones which have directive None will be ignored, the ones which have same name will be summed over
    """    
    # Read qom, Lx, Ly, Lz, nxc, nyc, nzc and dt from the SimulationData.txt file.
    
    data = {}
    for experiment in experiments:
        logger.info(f" reading {files_path}/{experiment}/SimulationData.txt")
        f=open(f"{files_path}/{experiment}/SimulationData.txt","r")
        content=f.readlines()
        f.close()

        qom=[]
        for n in content:
            if "QOM" in n:
                qom.append(float(re.split("=",re.sub(" |\n","",n))[-1]))
            if "x-Length" in n:
                Lx=float(re.split("=",re.sub(" |\n","",n))[1])
            if "y-Length" in n:
                Ly=float(re.split("=",re.sub(" |\n","",n))[1])
            if "z-Length" in n:
                Lz=float(re.split("=",re.sub(" |\n","",n))[1])
            if "Number of cells (x)" in n:
                nxc=int(re.split("=",re.sub(" |\n","",n))[1])
            if "Number of cells (y)" in n:
                nyc=int(re.split("=",re.sub(" |\n","",n))[1])
            if "Number of cells (z)" in n:
                nzc=int(re.split("=",re.sub(" |\n","",n))[1])
            if "Time step" in n:
                dt=float(re.split("=",re.sub(" |\n","",n))[1])
        logger.info(f"{Lx = }, {Ly = }, {nxc = }, {nyc = }")
        # The x, y and z axes are set.
        x=np.linspace(0,Lx,nxc+1)
        y=np.linspace(0,Ly,nyc+1)
        z=np.linspace(0,Lz,nzc+1)    
        #compute dx and dy to be used for the gradients computation
        #dx = Lx/nxc
        #dy = Ly/nyc
        # sorted(os.listdir()) creates a sorted list containing the .h5 filenames, os.listdir() alone would put them in random order.
        filenames = sorted([n for n in os.listdir(f"{files_path}{experiment}") if "-Fields_" in n and (n.endswith(".pkl") or n.endswith(".h5"))])
        if choose_times is None:
            selected_filenames = filenames
        elif isinstance(choose_times, int):
            selected_filenames = filenames[choose_times:]
        else:
            try:
                selected_filenames = [filenames[i] for i in choose_times]
            except Exception as e:
                logger.info(f"Inconsistent size: {len(filenames) = }  {len(choose_times) = }")
                raise e
        try:
            times = [int(n[-13:-7] if n.endswith(".h5.pkl") else n[-9:-3]) * dt for n in selected_filenames]
        except Exception as e:
            logger.info(f"Failed to extract times from {n = }")
            logger.info(f"{selected_filenames=}")
            raise e
        #logger.info(times)
        data[experiment] = read_data(f"{files_path}{experiment}/",selected_filenames,fields_to_read,qom,
                                     choose_species=choose_species,choose_x=choose_x,choose_y=choose_y,choose_z=choose_z,verbose=verbose, **kwargs)
        if choose_x is None:
            choose_x = [0,x.shape[0]-1]
        if choose_y is None:
            choose_y = [0,y.shape[0]-1]
        if choose_z is None:
            choose_z = [0,z.shape[0]-1]
        if verbose:
            logger.info(f"{choose_x = }, {choose_y = }, {choose_z = }, {choose_times =}")
        if nzc == 1:
            X, Y = np.meshgrid(x[choose_x[0]:choose_x[1]], y[choose_y[0]:choose_y[1]], indexing=kwargs.get('indexing',DEFAULT_INDEXING))
        else:
            X, Y, Z = np.meshgrid(x[choose_x[0]:choose_x[1]], y[choose_y[0]:choose_y[1]], z[choose_z[0]:choose_z[1]], indexing=kwargs.get('indexing',DEFAULT_INDEXING))
    if nzc == 1:
        return data, X, Y, qom, times
    else:
        return data, X, Y, Z, qom, times
    
def get_experiments(*args, **kwargs):
    """
    A wrapper function for get_exp_times that does not return times for backward compatibility.
    """
    logger.warning("get_experiments is deprecated, use get_exp_times instead")
    return get_exp_times(*args, **kwargs)[:-1]