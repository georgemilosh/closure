import h5py
import numpy as np
import os
import logging
import re
import utilities as ut
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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

def read_fieldname(files_path,filenames,fieldname,choose_x=None, choose_y=None):
    """
    Read a specific field from multiple files and return a subset of the field.

    Parameters:
    - files_path (str): The path to the directory containing the files.
    - filenames (list): A list of filenames to read from.
    - fieldname (str): The name of the field to read.
    - choose_x (list, optional): A list specifying the range of indices to select along the x-axis. Defaults to None.
    - choose_y (list, optional): A list specifying the range of indices to select along the y-axis. Defaults to None.

    Returns:
    - numpy.ndarray: A subset of the field, with the z-dimension removed.

    """
    field = []
    if not isinstance(filenames, list):
        filenames = [filenames]
    for filename in filenames:
        with h5py.File(files_path+filename,"r") as n:
            field.append(np.array(n[f"/Step#0/Block/{fieldname}/0"]))
            if choose_x is None:
                choose_x = [0,field[-1].shape[2]-1]
            if choose_y is None:
                choose_y = [0,field[-1].shape[1]-1]
    
    return np.squeeze(np.transpose(np.array(field))[choose_x[0]:choose_x[1],choose_y[0]:choose_y[1],0,:]) # transposing means swapping x and y and remove z


def build_XY(files_path,choose_x=None, choose_y=None):
    # Read qom, Lx, Ly, Lz, nxc, nyc, nzc and dt from the SimulationData.txt file.
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

    # The x, y and z axes are set.
    x=np.linspace(0,Lx,nxc+1)
    y=np.linspace(0,Ly,nyc+1)
    
    
    X, Y = np.meshgrid(x[choose_x[0]:choose_x[1]], y[choose_y[0]:choose_y[1]], indexing='ij')

    return X, Y


def read_features_targets(files_path, filenames, fields_to_read=None, request_features = None, request_targets = None, 
               choose_species=None,choose_x=None, choose_y=None, feature_dtype = np.float32, target_dtype = np.float32,  verbose=False):
    # Read qom, Lx, Ly, Lz, nxc, nyc, nzc and dt from the SimulationData.txt file.
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

    # The x, y and z axes are set.
    x=np.linspace(0,Lx,nxc+1)
    y=np.linspace(0,Ly,nyc+1)
    
    #extract_fields = request_features + request_targets
    features = read_files(files_path, filenames, fields_to_read, qom, feature_dtype, 
                          extract_fields=ut.species_to_list(request_features), choose_species=choose_species, 
                          choose_x=choose_x, choose_y=choose_y, verbose=verbose)
    targets = read_files(files_path, filenames, fields_to_read, qom, target_dtype, 
                         extract_fields=ut.species_to_list(request_targets), choose_species=choose_species, 
                         choose_x=choose_x, choose_y=choose_y, verbose=verbose)
    
    X, Y = np.meshgrid(x[choose_x[0]:choose_x[1]], y[choose_y[0]:choose_y[1]], indexing='ij')

    return features, targets

def read_files(files_path, filenames, fields_to_read, qom, dtype, extract_fields=None, choose_species=None, choose_x=None, choose_y=None, verbose=False):
    out2 = []
    for filename in filenames:
        out = []
        data = read_data(files_path,filename,fields_to_read,qom,
                                    choose_species=choose_species,choose_x=choose_x,choose_y=choose_y,verbose=verbose)
        
        for extract_field_index in extract_fields:
            if isinstance(extract_field_index, list):
                #logger.info(data[extract_field_index[0]][extract_field_index[1]])
                out.append(data[extract_field_index[0]][extract_field_index[1]])
            else:
                #logger.info(data[extract_field_index])
                out.append(data[extract_field_index])
        out2.append(np.array(out))
    return np.array(out2, dtype=dtype).transpose(0,2,3,1)  # we want to have the time as the first index, then x, then y, then the field

def read_data(files_path, filenames, fields_to_read, qom, choose_species=None, choose_x=None, choose_y=None, verbose=False):
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
    - verbose (bool): A flag indicating whether to logger.info debug information.

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
    choose_species_new = append_index_to_duplicates(choose_species) 
    dublicatespecies = get_duplicate_indices(choose_species)
    data = {}
    small = 1e-12
    # The magnetic and electric field is read.
    for fields in ['B', 'E']:
        if fields_to_read[fields]:
            if verbose:
                logger.info(f"loading {fields}")
            for component in ['x','y','z']:
                data[f'{fields}{component}'] = read_fieldname(files_path,filenames,f"{fields}{component}",choose_x,choose_y)
            try:    
                data[f'{fields}magn'] = np.sqrt(data[f'{fields}x']**2 + data[f'{fields}y']**2 + data[f'{fields}z']**2)
            except Exception as e:
                logger.info(f"{fields}magn failed")
                raise e
        if fields_to_read[f"{fields}_ext"]:
            for component in ['x','y','z']:
                data[f'B{component}_ext'] = read_fieldname(files_path,filenames,f"{fields}{component}_ext",choose_x,choose_y)
    # The divergence of B is read.
    if fields_to_read["divB"]:
        if verbose:
                logger.info(f"loading divB")
        data['divB'] = read_fieldname(files_path,filenames,'divB')
    for fields in ['rho', 'N', 'Qrem']:
        if fields_to_read[fields]:
            if verbose:
                logger.info(f"loading {fields}")
            data[fields] = {}
            for i, species in enumerate(choose_species_new): # Care must be taken that these the only species and they are actually correctly labeled
                if species is not None:
                    data[fields][species] = read_fieldname(files_path,filenames,f'{fields}_{i}',choose_x,choose_y)


    if fields_to_read["J"]:
        data['Jx'], data['Jy'], data['Jz'] = {}, {}, {}
        data['Vx'], data['Vy'], data['Vz'] = {}, {}, {}
        if verbose:
            logger.info(f"loading J")
        for component in ['x','y','z']:
            for i, species in enumerate(choose_species_new):
                if species is not None:
                    data[f'J{component}'][species] = read_fieldname(files_path,filenames,f'J{component}_{i}',choose_x,choose_y)
                    data[f'V{component}'][species] = data[f'J{component}'][species]/(data['rho'][species]+1e-12)
        data['Jmagn'], data['Vmagn'] = {}, {}
        for i, species in enumerate(choose_species_new):
            if species is not None:
                data['Jmagn'][species] = np.sqrt(data['Jx'][species]**2 + data['Jy'][species]**2 + data['Jz'][species]**2)
                data['Vmagn'][species] = np.sqrt(data['Vx'][species]**2 + data['Vy'][species]**2 + data['Vz'][species]**2)
                

    # The diagonal and offdiagonal part of the pressure is calculated (to do so you need to read rho and J first).
    if fields_to_read["P"] or fields_to_read["PI"]:
        if verbose:
            logger.info(f"loading P and/or PI")
        for component_1 in ['x','y','z']:
            for component_2 in ['x','y','z']:
                if fields_to_read["P"]:
                    data[f'P{component_1}{component_2}'] = {}
                if fields_to_read["PI"]:
                    data[f'PI{component_1}{component_2}'] = {}
                for i, species in enumerate(choose_species_new):
                    if species is not None:
                        try:
                            PI = read_fieldname(files_path,filenames,f'P{component_1}{component_2}_{i}',choose_x,choose_y)
                            if fields_to_read["PI"]:
                                data[f'PI{component_1}{component_2}'][species] = PI
                            if fields_to_read["P"]:
                                data[f'P{component_1}{component_2}'][species] = (PI - data[f'J{component_1}'][species]*data[f'J{component_2}'][species]/(data[f'rho'][species]+small))/qom[i]
                                #logger.info(f"{component_1 = }, {component_2}, {i = }, {species = }, {(data[f'P{component_1}{component_2}'][species]).shape = }")
                        except:
                            if verbose:
                                logger.info(f'Component P{component_1 = }{component_2 =} for species {species} missing because tensor is symmetric')
                        
                       
        if fields_to_read["PI"]:
            for i, species in enumerate(choose_species_new):
                if species is not None:
                    data['PIyx'][species] = data['PIxy'][species]
                    data['PIzx'][species] = data['PIxz'][species]
                    data['PIzy'][species] = data['PIyz'][species]
        if fields_to_read["P"]:
            data['Ppar'], data['Pperp'] = {}, {}
            for i, species in enumerate(choose_species_new):
                if species is not None:
                    data['Pyx'][species] = data['Pxy'][species]
                    data['Pzx'][species] = data['Pxz'][species]
                    data['Pzy'][species] = data['Pyz'][species]
                    if verbose:
                        logger.info(f"loading Ppar and Pperp")
                    data['Ppar'][species] = (data['Pxx'][species]*data['Bx']**2 + data['Pyy'][species]*data['By']**2  + data['Pzz'][species]*data['Bz']**2 + \
                                            2*data['Pxy'][species]*data['Bx']*data['By']+2*data['Pxz'][species]*data['Bx']*data['Bz'] + \
                                                2*data['Pyz'][species]*data['By']*data['Bz'])/(data['By']**2+data['Bx']**2+data['Bz']**2)
                    data['Pperp'][species] = (data['Pxx'][species] + data['Pyy'][species] + data['Pzz'][species] - data['Ppar'][species])/2

    # The heat flux is calculated (to do so you need to read rho, J and P first).
    if fields_to_read["Heat_flux"]:
        if verbose:
            logger.info(f"loading q")
        for component in ['x','y','z']:
            data[f'q{component}'] = {}
            for i, species in enumerate(choose_species_new):
                if species is not None:
                    data[f'q{component}'][species] = read_fieldname(files_path,filenames,f'EF{component}_{i}',choose_x,choose_y) - \
                        (data['Jx'][species]**2+data['Jy'][species]**2+data['Jz'][species]**2)*data[f'J{component}'][species]/(2*qom[i]*data[f'rho'][species]+small) - \
                        (data['Pxx'][species] + data[f'Pyy'][species] + data[f'Pzz'][species])*data[f'J{component}'][species]/(2*data['rho'][species]+small) - \
                        (data['Jx'][species]*data[f'Px{component}'][species] + data['Jy'][species]*data[f'Py{component}'][species] + data['Jz'][species]*data[f'Pz{component}'][species])/(data['rho'][species]+small)
    # Treat dublicate species:
    for fields in data.keys():
        if fields not in ['Bmagn','Emagn','Bx','By','Bz','Ex','Ey','Ez','Bx_ext','By_ext','Bz_ext','divB']:
            for dublicatespecie, indices in dublicatespecies.items():
                try:
                    data[fields][dublicatespecie] = np.sum([data[fields][choose_species_new[i]] for i in indices], axis=0)
                except Exception as e:
                    logger.info(f"{indices = }, {choose_species_new = }, {i = }, {dublicatespecie = }, {fields = }")
                    logger.info(f"{data[fields] = }")
                    raise e
                for index in indices: # and remove these indices
                    del data[fields][choose_species_new[index]]
        
    return data

def get_experiments(experiments, files_path, fields_to_read, choose_species=None, choose_times=None,choose_x=None, choose_y=None, verbose=False):
    """
    Retrieves data from experiments and returns the data structure stored as a dictionary along with the corresponding meshgrid.

    Parameters:
    - experiments (list): A list of experiment names. Each experiment is a directory containing the experiment files.
    - files_path (str): The path to the directory containing the experiment files. The experiment directories are subdirectories of this directory.
    - fields_to_read (list of bools): A list of bools corresponding to the condition of whether or not to read a specific
                field names to read from the files. 
    - qom (array of floats): The charge-to-mass ratio in the PIC units for each species.
    - choose_species (list): A list of species indices to choose.  # the ones which have directive None will be ignored, the ones which have same name will be summed over
    - choose_times (list): A list of time indices to choose. If None is given, all times are chosen. If list specific timeshots are chosen, 
                i.e. [0, 1, 5], otherwise choose_times = None means take all times
    - choose_x (list): A list specifying the range of x indices to choose. If None is given, the whole range is chosen.
    - choose_y (list): A list specifying the range of y indices to choose. If None is given, the whole range is chosen.
    - verbose (bool): A flag indicating whether to logger.info debug information when reading data such as which fields are being imported.

    Returns:
    - data (dict): A dictionary containing the retrieved data for each experiment.
    - X (ndarray): The meshgrid of x values.
    - Y (ndarray): The meshgrid of y values.
    
    Example:
    >>>
        choose_species = ['e1',None,'e2',None] # the ones which have directive None will be ignored, the ones which have same name will be summed over
    """    
    # Read qom, Lx, Ly, Lz, nxc, nyc, nzc and dt from the SimulationData.txt file.
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

    # The x, y and z axes are set.
    x=np.linspace(0,Lx,nxc+1)
    y=np.linspace(0,Ly,nyc+1)
    #z=np.linspace(0,Lz,nzc+1)    
    #compute dx and dy to be used for the gradients computation
    #dx = Lx/nxc
    #dy = Ly/nyc
    data = {}
    for experiment in experiments:
        # sorted(os.listdir()) creates a sorted list containing the .h5 filenames, os.listdir() alone would put them in random order.
        filenames=sorted([n for n in os.listdir(f"{files_path}{experiment}") if "Fields" in n])
        if choose_times is not None:
            selected_filenames = [filenames[i] for i in choose_times]
        else:
            selected_filenames = filenames
        times=[int(n[-9:-3])*dt for n in filenames]  # the last 6 characters of the filename are the time in units of dt.
        logger.info(times)
        data[experiment] = read_data(f"{files_path}{experiment}/",selected_filenames,fields_to_read,qom,
                                     choose_species=choose_species,choose_x=choose_x,choose_y=choose_y,verbose=verbose)
        if choose_x is None:
            choose_x = [0,x.shape[0]]
        if choose_y is None:
            choose_y = [0,y.shape[0]]
        logger.info(choose_x, choose_y)
        X, Y = np.meshgrid(x[choose_x[0]:choose_x[1]], y[choose_y[0]:choose_y[1]], indexing='ij')
    return data, X, Y, qom