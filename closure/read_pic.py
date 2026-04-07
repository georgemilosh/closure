from __future__ import annotations

import glob
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as nd

from .config import load_paths
from .plasma import do_cross, highdiff
from .utilities import species_to_list

import logging
logger = logging.getLogger(__name__)

__all__ = [
    "apply_filters",
    "build_XY",
    "convert_ipic3d_to_ecsim_h5",
    "ecsim_available_run_info",
    "find_field_in_hdf5",
    "get_exp_times",
    "get_experiments",
    "get_saved_iterations",
    "ipic3D_available_cycles",
    "parse_simulation_data",
    "read_data",
    "read_data_ipic3d",
    "read_features_targets",
    "read_fieldname",
    "read_ipic3d_field",
]

# Define global default values
DEFAULT_CHOOSE_X = None
DEFAULT_CHOOSE_Y = None
DEFAULT_CHOOSE_Z = None
DEFAULT_INDEXING = 'ij'
DEFAULT_VERBOSE = False


def _resolve_files_path(files_path: str | os.PathLike[str] | None) -> str:
    """Resolve a data path, falling back to ``paths.yaml`` when omitted."""
    if files_path is None:
        files_path = load_paths().get("data_dir", "./data")
    return str(Path(files_path).expanduser())

def get_saved_iterations(files_path, experiment, choose_times=None):
    """Return sorted saved field iterations and corresponding simulation times.
    Example:
        saved_iterations, saved_times  = rp.get_saved_iterations(files_path, experiment)
        saved_iterations.index(17100)
    """
    files_path = _resolve_files_path(files_path)
    exp_path = os.path.join(files_path, experiment)
    parser = parse_simulation_data(exp_path)
    
    filenames = _collect_experiment_filenames(f"{files_path}/{experiment}")
    field_files = _select_filenames_by_time(filenames, choose_times)

    saved_iterations = sorted({
        int(match.group(1))
        for name in field_files
        for match in [re.search(r"(\d+)\.h5(?:\.pkl)?$", name)]
        if match
    })
    saved_times = [parser['dt']*iteration for iteration in saved_iterations]

    return saved_iterations, saved_times    


def _extract_fields_and_species_from_names(names):
    """Parse field names and species indices from dataset/key names.

    Species-dependent fields are commonly stored as e.g. ``Jx_0`` or
    ``rho_1`` in ECSIM-style outputs.
    """
    fields = set()
    species_indices = set()
    for name in names:
        match = re.match(r"(.+)_([0-9]+)$", str(name))
        if match:
            fields.add(match.group(1))
            species_indices.add(int(match.group(2)))
        else:
            fields.add(str(name))
    return fields, species_indices


def _is_empty_data_value(value):
    """Return whether a loaded data value should be considered empty."""
    if value is None:
        return True
    if isinstance(value, dict):
        return len(value) == 0
    if isinstance(value, (list, tuple, set, str)):
        return len(value) == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    return False


def _collect_empty_data_keys(data):
    """Collect empty top-level keys and nested empty dict entries."""
    empty_keys = [key for key, value in data.items() if _is_empty_data_value(value)]
    empty_nested_keys = {
        key: [subkey for subkey, subvalue in value.items() if _is_empty_data_value(subvalue)]
        for key, value in data.items()
        if isinstance(value, dict)
    }
    empty_nested_keys = {key: subkeys for key, subkeys in empty_nested_keys.items() if subkeys}
    return empty_keys, empty_nested_keys


def ecsim_available_run_info(files_path, experiment=None):
    """Return available cycles, fields, and species indices for an ECSIM run.

    Parameters:
    - files_path (str): Base data directory, or the run directory itself.
    - experiment (str, optional): Run subdirectory name under ``files_path``.

    Returns:
    - dict with keys:
        - ``cycles`` (list[int]): Sorted available iterations/cycles.
        - ``fields`` (list[str]): Sorted available field names.
        - ``species_indices`` (list[int]): Sorted available species indices.
        - ``qom`` (list[float]): Charge-to-mass ratios from SimulationData.txt.
    """
    files_path = _resolve_files_path(files_path)
    run_path = os.path.join(files_path, experiment) if experiment else files_path
    if not os.path.isdir(run_path):
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    filenames = _collect_experiment_filenames(run_path)
    if not filenames:
        raise FileNotFoundError(f"No ECSIM field files found in {run_path}")

    sim_data = parse_simulation_data(run_path)
    qom = sim_data.get("qom", [])

    cycles = sorted({
        int(match.group(1))
        for name in filenames
        for match in [re.search(r"(\d+)(?:\.h5(?:\.pkl)?|\.npz)?$", str(name))]
        if match
    })

    sample = filenames[0]
    fields = set()
    species_indices = set()

    if str(sample).endswith(".h5"):
        import h5py

        with h5py.File(os.path.join(run_path, sample), "r") as h5f:
            if "Step#0" in h5f and "Block" in h5f["Step#0"]:
                names = list(h5f["Step#0"]["Block"].keys())
                fields, species_indices = _extract_fields_and_species_from_names(names)
            elif "Fields" in h5f:
                names = list(h5f["Fields"].keys())
                fields, species_indices = _extract_fields_and_species_from_names(names)
            else:
                names = list(h5f.keys())
                fields, species_indices = _extract_fields_and_species_from_names(names)
    elif str(sample).endswith(".npz"):
        with np.load(os.path.join(run_path, sample)) as npzf:
            names = list(npzf.keys())
            fields, species_indices = _extract_fields_and_species_from_names(names)
    elif str(sample).endswith(".h5.pkl"):
        with open(os.path.join(run_path, sample), "rb") as pklf:
            data = pickle.load(pklf)
        names = list(data.keys())
        fields, species_indices = _extract_fields_and_species_from_names(names)

    return {
        "cycles": cycles,
        "fields": sorted(fields),
        "species_indices": sorted(species_indices),
        "qom": qom,
    }

def ipic3D_available_cycles(files_path):
    """
    Identify available cycles in iPiC3D HDF5 files.

    Returns:
    - list: A sorted list of available cycle identifiers.
    """
    files_path = _resolve_files_path(files_path)
    all_hdf_files = sorted(glob.glob(os.path.join(files_path, "proc*.hdf")))
    sim_data = parse_simulation_data(files_path)
    dt = sim_data['dt']
    sample_file = all_hdf_files[0]
    import h5py
    with h5py.File(sample_file, "r") as f:
        try:
            time_cycles = list(f['fields']['Bx'].keys())
        except KeyError:
            logger.info(f"available groups in file: {list(f.keys())} ")
            logger.info(f"available fields: {list(f['fields'].keys())} ")
            raise KeyError(f"Could not find 'Bx' field in file {sample_file}")
    cycles = [int(re.search(r'cycle_(\d+)', cycle).group(1)) for cycle in time_cycles]
    cycles = sorted(cycles)
    times = [np.round(cycle*dt, decimals=8) for cycle in cycles]
    cycles = [int(cycle) for cycle in cycles]
    return cycles, times

def find_field_in_hdf5(f, path_prefix, field_name, time_cycle):
    """
    Find a field in HDF5 file with case-insensitive matching.
    
    Returns the correct field name and data, or raises KeyError if not found.
    """
    try:
        # Try exact match first
        return np.array(f[f"{path_prefix}/{field_name}/{time_cycle}"])[:-1, :-1, :-1]
    except KeyError:
        # Try case-insensitive match
        available_fields = list(f[path_prefix].keys())
        field_name_lower = field_name.lower()
        
        for available_field in available_fields:
            if available_field.lower() == field_name_lower:
                return np.array(f[f"{path_prefix}/{available_field}/{time_cycle}"])[:-1, :-1, :-1]
        
        # No match found
        raise KeyError(f"Field '{field_name}' not found. Available: {available_fields}")

def read_ipic3d_field(files_path, cycles, fieldname, choose_x=DEFAULT_CHOOSE_X, choose_y=DEFAULT_CHOOSE_Y, 
                      choose_z=DEFAULT_CHOOSE_Z, indexing=DEFAULT_INDEXING, verbose=DEFAULT_VERBOSE):
    """
    Read a specific field from multiple iPiC3D HDF5 files and return a subset of the field.

    Parameters:
    - files_path (str): The path to the directory containing the files.
    - cycles (list): A list of cycle identifiers to read from.
    - fieldname (str): The name of the field to read.
    - choose_x (list, optional): A list specifying the range of indices to select along the x-axis. Defaults to None.
    - choose_y (list, optional): A list specifying the range of indices to select along the y-axis. Defaults to None.
    - choose_z (list, optional): A list specifying the range of indices to select along the z-axis. Defaults to None.
    - indexing (string, defaults to 'ij'): A flag indicating how to transpose the field. If 'ij', the field is 
        transposed to have the x-axis as the first index, the y-axis as the second index, and the z-axis as the third index. 
        If 'xy', the field is transposed to have the x-axis as the second index, the y-axis as the first index, and the z-axis as the third index.
    - verbose (bool, optional): A flag indicating whether to logger.info debug information.

    Returns:
    - numpy.ndarray: A subset of the field, with the z-dimension removed.

    """
    files_path = _resolve_files_path(files_path)
    sim_data = parse_simulation_data(files_path)
    nxc = sim_data['nxc']
    nyc = sim_data['nyc']
    nzc = sim_data['nzc']

    if choose_x is None:
        choose_x = [0, nxc]
    if choose_y is None:
        choose_y = [0, nyc]
    if choose_z is None:
        choose_z = [0, nzc]
    if verbose:
        logger.info(f"{choose_x = }, {choose_y = }, {choose_z = }")
    if "_" in fieldname:
        field_name, species = fieldname.split("_", 1)  # splits at first underscore
    else:
        field_name, species = fieldname, None            # or handle however you need
    requests = [{
        'output_name': fieldname,
        'path_prefix': f"moments/species_{species}" if species is not None else 'fields',
        'field_name': field_name,
    }]
    field_times = _read_ipic3d_cycles(
        files_path,
        cycles,
        requests,
        choose_x=choose_x,
        choose_y=choose_y,
        choose_z=choose_z,
        indexing=indexing,
        skip_missing=False,
        verbose=verbose,
    )[fieldname]
    if verbose:
        logger.info(f"Extracted {field_times.shape = }")
    return field_times


def read_data_ipic3d(files_path, cycles, fields_to_read, qom=None, choose_species=None, choose_x=DEFAULT_CHOOSE_X, choose_y=DEFAULT_CHOOSE_Y, 
              choose_z=DEFAULT_CHOOSE_Z, verbose=DEFAULT_VERBOSE, small=1e-10, **kwargs):
    """
    Reads and processes data from files in iPiC3D hdf5 format.

    Parameters:
    - files_path (str): The path to the files.
    - cycles (list): A list of cycle numbers to read.
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
    files_path = _resolve_files_path(files_path)
    if choose_species is None:
        choose_species = _detect_ipic3d_species(files_path)

    sim_data = parse_simulation_data(files_path)
    if qom is None:
        qom = sim_data['qom']

    indexing = kwargs.get('indexing', DEFAULT_INDEXING)
    X, Y = build_XY(files_path, choose_x=choose_x, choose_y=choose_y, choose_z=choose_z, indexing=indexing)
    data = {}

    species_index = {
        species: i
        for i, species in enumerate(choose_species)
        if species is not None and species not in choose_species[:i]
    }

    def qom_for_species(species):
        return qom[species_index[species]]

    def accumulate_species_field(field_dict, output_name, species_alias):
        value = raw_fields.get(output_name)
        if value is None or species_alias is None:
            return
        if species_alias in field_dict:
            field_dict[species_alias] += value
        else:
            field_dict[species_alias] = value

    requests = _build_ipic3d_analysis_requests(fields_to_read, choose_species)
    raw_fields = {}
    if requests:
        available = _inspect_ipic3d_available_fields(files_path)
        requests = _filter_ipic3d_requests_by_availability(requests, available, verbose=verbose)
        raw_fields = _read_ipic3d_cycles(
            files_path,
            cycles,
            requests,
            choose_x=choose_x,
            choose_y=choose_y,
            choose_z=choose_z,
            indexing=indexing,
            skip_missing=True,
            verbose=verbose,
        )

    for fields in ['B', 'E']:
        if fields_to_read.get(fields, False):
            if verbose:
                logger.info(f"loading {fields}")
            for component in ['x', 'y', 'z']:
                key = f'{fields}{component}'
                if key in raw_fields:
                    data[key] = raw_fields[key]
            try:
                data[f'{fields}magn'] = np.sqrt(data[f'{fields}x']**2 + data[f'{fields}y']**2 + data[f'{fields}z']**2)
            except Exception as e:
                logger.warning(f"Failed to calculate {fields}magn, see: {e}")
        if fields_to_read.get(f"{fields}_ext", False):
            for component in ['x', 'y', 'z']:
                key = f'{fields}{component}_ext'
                if key in raw_fields:
                    data[key] = raw_fields[key]

    if fields_to_read.get("divB", False):
        if verbose:
            logger.info("loading divB")
        if 'divB' in raw_fields:
            data['divB'] = raw_fields['divB']

    for fields in ['rho', 'N', 'Qrem']:
        if fields_to_read.get(fields, False):
            if verbose:
                logger.info(f"loading {fields}")
            data[fields] = {}
            for i, species in enumerate(choose_species):
                accumulate_species_field(data[fields], f'{fields}_{i}', species)

    if fields_to_read.get("J", False):
        data['Jx'], data['Jy'], data['Jz'] = {}, {}, {}
        if fields_to_read.get('rho', False):
            data['Vx'], data['Vy'], data['Vz'] = {}, {}, {}
        if verbose:
            logger.info("loading J")
        for component in ['x', 'y', 'z']:
            field_name = f'J{component}'
            for i, species in enumerate(choose_species):
                accumulate_species_field(data[field_name], f'{field_name}_{i}', species)
            if fields_to_read.get('rho', False):
                for species in data[field_name].keys():
                    species_qom = qom_for_species(species)
                    data[f'V{component}'][species] = data[field_name][species] / (
                        data['rho'][species] + small * np.sign(species_qom)
                    )
        data['Jmagn'] = {}
        data['Jtotx'] = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
        data['Jtoty'] = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
        data['Jtotz'] = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)
        if 'Vx' in data:
            data['Vmagn'] = {}
        for species in data['Jx'].keys():
            data['Jmagn'][species] = np.sqrt(data['Jx'][species]**2 + data['Jy'][species]**2 + data['Jz'][species]**2)
            if 'Vx' in data:
                data['Vmagn'][species] = np.sqrt(data['Vx'][species]**2 + data['Vy'][species]**2 + data['Vz'][species]**2)

    if fields_to_read.get("P", False) or fields_to_read.get("PI", False):
        if verbose:
            logger.info("loading P and/or PI")
        for component_1 in ['x', 'y', 'z']:
            for component_2 in ['x', 'y', 'z']:
                data[f'PI{component_1}{component_2}'] = {}
                data[f'P{component_1}{component_2}'] = {}
                for i, species in enumerate(choose_species):
                    accumulate_species_field(data[f'PI{component_1}{component_2}'], f'P{component_1}{component_2}_{i}', species)
                for species in data[f'PI{component_1}{component_2}']:
                    species_qom = qom_for_species(species)
                    data[f'P{component_1}{component_2}'][species] = (
                        data[f'PI{component_1}{component_2}'][species]
                        - data[f'J{component_1}'][species] * data[f'J{component_2}'][species] / (
                            data['rho'][species] + small * np.sign(species_qom)
                        )
                    ) / species_qom
                if not fields_to_read.get("P", False):
                    del data[f'P{component_1}{component_2}']
                if not fields_to_read.get("PI", False):
                    del data[f'PI{component_1}{component_2}']
        if fields_to_read.get("PI", False):
            for species in data['PIxx']:
                if species in data['PIxy']:
                    data['PIyx'][species] = data['PIxy'][species]
                if species in data['PIxz']:
                    data['PIzx'][species] = data['PIxz'][species]
                if species in data['PIyz']:
                    data['PIzy'][species] = data['PIyz'][species]
        if fields_to_read.get("P", False):
            data['Ppar'], data['Pperp'] = {}, {}
            for species in data['Pxx']:
                if species in data['Pxy']:
                    data['Pyx'][species] = data['Pxy'][species]
                if species in data['Pxz']:
                    data['Pzx'][species] = data['Pxz'][species]
                if species in data['Pyz']:
                    data['Pzy'][species] = data['Pyz'][species]
                if verbose:
                    logger.info("loading Ppar and Pperp")
                try:
                    data['Ppar'][species] = (
                        data['Pxx'][species] * data['Bx']**2
                        + data['Pyy'][species] * data['By']**2
                        + data['Pzz'][species] * data['Bz']**2
                        + 2 * data['Pxy'][species] * data['Bx'] * data['By']
                        + 2 * data['Pxz'][species] * data['Bx'] * data['Bz']
                        + 2 * data['Pyz'][species] * data['By'] * data['Bz']
                    ) / (data['By']**2 + data['Bx']**2 + data['Bz']**2)
                except Exception as e:
                    logger.warning(f"Failed to calculate Ppar for {species = } likely due to missing fields, see: {e}")
                try:
                    data['Pperp'][species] = (data['Pxx'][species] + data['Pyy'][species] + data['Pzz'][species] - data['Ppar'][species]) / 2
                except Exception as e:
                    logger.warning(f"Failed to calculate Pperp for {species} likely due to missing fields, see: {e}")
        if fields_to_read.get("gyro_radius", False):
            try:
                data['gyro_radius'] = {}
                for species in data['rho']:
                    species_qom = qom_for_species(species)
                    vth = np.sqrt(np.abs(species_qom * data['Pperp'][species] / (np.abs(data['rho'][species]) + small)))
                    data['gyro_radius'][species] = np.abs(vth / (species_qom * data['Bmagn']))
            except Exception as e:
                logger.warning(f"Failed to calculate gyro_radius, see: {e}")
    if "divP" in fields_to_read and fields_to_read["divP"] or "Ohmres" in fields_to_read and fields_to_read["Ohmres"]:
        if verbose:
            logger.info(f"computing divP and or Ohmres")

        dx = X[1,0]-X[0,0]
        dy = Y[0,1]-Y[0,0]

        if not 'e' in choose_species:
            raise ValueError(f"Calculating divP_e or Ohmres without electron species cannot be done")

        data['EPx'] = -(highdiff(data['Pxx']['e'], dx, dy, axis=0, mode='wrap') + highdiff(data['Pxy']['e'], dx, dy, axis=1, mode='wrap'))/(-data['rho']['e']) # density in ECsim is negative (electron charge density)
        data['EPy'] = -(highdiff(data['Pxy']['e'], dx, dy, axis=0, mode='wrap') + highdiff(data['Pyy']['e'], dx, dy, axis=1, mode='wrap'))/(-data['rho']['e']) # density in ECsim is negative (electron charge density)
        data['EPz'] = -(highdiff(data['Pxz']['e'], dx, dy, axis=0, mode='wrap') + highdiff(data['Pyz']['e'], dx, dy, axis=1, mode='wrap'))/(-data['rho']['e']) # density in ECsim is negative (electron charge density)
        
        if "Ohmres" in fields_to_read and fields_to_read["Ohmres"]:
            #logger.info(f"{data['Bx'].shape = }")
            #B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,0)
            #E = np.array([data['Ex'], data['Ey'], data['Ez']]).transpose(1,2,3,0)
            Jtotx = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
            Jtoty = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
            Jtotz = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)

            # = np.array([Jtotx, Jtoty, Jtotz]).transpose(1,2,3,0)
            data['EHallx'], data['EHally'], data['EHallz'] = do_cross(Jtotx,Jtoty,Jtotz,data['Bx'],data['By'],data['Bz'])/(-data['rho']['e']) # EHx,EHy,EHz=do_cross(Jx,Jy,Jz,Bx,By,Bz)/(-rho_0)
            norm = 0
            uCMx = 0
            uCMy = 0
            uCMz = 0
            for species in data['rho'].keys():
                species_qom = qom_for_species(species)
                uCMx += (data['rho'][species]/species_qom)*data['Vx'][species]
                uCMy += (data['rho'][species]/species_qom)*data['Vy'][species]
                uCMz += (data['rho'][species]/species_qom)*data['Vz'][species]
                norm += data['rho'][species]/species_qom
            uCMx /= norm
            uCMy /= norm
            uCMz /= norm
            data['EMHDx'], data['EMHDy'], data['EMHDz'] = do_cross(uCMx,uCMy,uCMz,data['Bx'],data['By'],data['Bz']) # TODO: fix sign, should be minus
            # data['EMHDx'], data['EMHDy'], data['EMHDz'] = -do_cross(uCMx,uCMy,uCMz,data['Bx'],data['By'],data['Bz']) 
            data['Ohmresx'] = data['Ex'] + data['EMHDx'] - data['EHallx'] - data['EPx']
            data['Ohmresy'] = data['Ey'] + data['EMHDy'] - data['EHally'] - data['EPy']
            data['Ohmresz'] = data['Ez'] + data['EMHDz'] - data['EHally'] - data['EPz']
    # The heat flux is calculated (to do so you need to read rho, J and P first).
    if fields_to_read.get("Heat_flux", False):
        if verbose:
            logger.info(f"loading q")
        for component in ['x','y','z']:
            data[f'EF{component}'] = {}
            for i, species in enumerate(choose_species):
                accumulate_species_field(data[f'EF{component}'], f'EF{component}_{i}', species)
            try:
                data[f'q{component}'] = {}
                for species in data[f'EF{component}'].keys():
                    species_qom = qom_for_species(species)
                    data[f'q{component}'][species] =  data[f'EF{component}'][species] - \
                        (data['Jx'][species]**2+data['Jy'][species]**2+data['Jz'][species]**2)*data[f'J{component}'][species]/(2*species_qom*data[f'rho'][species]**2+small*np.sign(species_qom)) - \
                        (data['Pxx'][species] + data[f'Pyy'][species] + data[f'Pzz'][species])*data[f'J{component}'][species]/(2*data['rho'][species]+small*np.sign(species_qom)) - \
                        (data['Jx'][species]*data[f'Px{component}'][species] + data['Jy'][species]*data[f'Py{component}'][species] + data['Jz'][species]*data[f'Pz{component}'][species])/(data['rho'][species]+small*np.sign(species_qom))
            except Exception as e:
                logger.warning(f"Failed to calculate q{component} see: {e}")
            if not fields_to_read.get('EF', False):
                del data[f'EF{component}']
    return data   

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
        current_file_path = os.path.join(files_path, filename)
        try:
            if filename.endswith(".h5"):
                import h5py
                with h5py.File(current_file_path, "r") as n:
                    try:
                        temp = np.array(n[f"/Step#0/Block/{fieldname}/0"])
                    except Exception as e:
                        available_fields = []
                        try:
                            if "/Step#0/Block" in n:
                                available_fields = list(n["/Step#0/Block"].keys())
                            else:
                                available_fields = list(n.keys())
                        except Exception:
                            available_fields = []
                        logger.error(
                            f"Unable to open {fieldname = } from {files_path = } of {filename = }. "
                            f"Available fields: {available_fields}"
                        )
                        raise e
            elif filename.endswith(".h5.pkl"):
                with open(os.path.join(files_path, filename), "rb") as n:
                    temp = pickle.load(n)[fieldname]
            elif filename.endswith(".npz"):
                with np.load(os.path.join(files_path, filename)) as n:
                    temp = n[fieldname]
            else:
                # Assuming that string of integers was passed
                import h5py
                iteration = int(filename)
                # Split fieldname to handle cases like Jx_0 (field Jx, species 0)
                fieldname_parts = fieldname.split('_')
                base_fieldname = fieldname_parts[0]
                species_id = fieldname_parts[1] if len(fieldname_parts) > 1 else None
                try:
                    if base_fieldname in ['Bx', 'By', 'Bz']:
                        filepath = f"{files_path}/Fields_{iteration:05d}/B_{iteration:05d}.h5"
                        with h5py.File(filepath, "r") as n:
                            temp = np.array(n['Fields'][fieldname])
                    elif base_fieldname in ['Ex', 'Ey', 'Ez']:
                        filepath = f"{files_path}/Fields_{iteration:05d}/E_{iteration:05d}.h5"
                        with h5py.File(filepath, "r") as n:
                            temp = np.array(n['Fields'][fieldname])
                    elif base_fieldname in ['Jx','Jy','Jz']:
                        filepath = f"{files_path}/Moments_{iteration:05d}/J_species_{species_id}_{iteration:05d}.h5"
                        with h5py.File(filepath, "r") as n:
                            temp = np.array(n['Moments'][f'species_{species_id}'][base_fieldname])
                    elif base_fieldname in ['Pxx','Pxy','Pxz','Pyy','Pyz','Pzz']:
                        filepath = f"{files_path}/Moments_{iteration:05d}/Pressure_species_{species_id}_{iteration:05d}.h5"
                        #print(base_fieldname.swapcase())
                        with h5py.File(filepath, "r") as n:
                            temp = np.array(n['Moments'][f'species_{species_id}'][base_fieldname.swapcase()])
                    elif base_fieldname in ['rho']:
                        filepath = f"{files_path}/Moments_{iteration:05d}/rho_species_{species_id}_{iteration:05d}.h5"
                        with h5py.File(filepath, "r") as n:
                            temp = np.array(n['Moments'][f'species_{species_id}'][base_fieldname])
                    elif base_fieldname in ['EFx','EFy','EFz']:
                        filepath = f"{files_path}/Moments_{iteration:05d}/E_flux_species_{species_id}_{iteration:05d}.h5"
                        with h5py.File(filepath, "r") as n:
                            temp = np.array(n['Moments'][f'species_{species_id}'][base_fieldname])
                    elif base_fieldname in ['Qxxxs','Qxxys','Qxxzs','Qxxyy','Qxxyz','Qxxzz','Qyyys','Qyyzs','Qyyyy','Qyyzz','Qzzzs','Qzzzz']:
                        filepath = f"{files_path}/Moments_{iteration:05d}/H_flux_species_{species_id}_{iteration:05d}.h5"
                        with h5py.File(filepath, "r") as n:
                            temp = np.array(n['Moments'][f'species_{species_id}'][base_fieldname])
                except Exception as e:
                    logger.error(f"Unable to open {filepath = } for {fieldname = }")
                    raise e
                # Permute from (x, y, z) to (z, y, x)
                temp = np.transpose(temp, (2, 1, 0))
            #print(f"{temp.shape = }")
            # Slicing if needed, if not specified we take the whole range (or 0,1 if the dimension is of size 1)
            if choose_x is None:
                if temp.shape[2] > 1:
                    choose_x = [0,temp.shape[2]-1]
                else:
                    choose_x = [0,1]
            if choose_y is None:
                if temp.shape[1] > 1:
                    choose_y = [0,temp.shape[1]-1]
                else:
                    choose_y = [0,1]
            if choose_z is None:
                if temp.shape[0] > 1:
                    choose_z = [0,temp.shape[0]-1]
                else:
                    choose_z = [0,1]
            #logger.warning(f"{choose_x = }, {choose_y = }, {choose_z = } with shape {temp.shape}")
            if indexing == 'ij':
                temp = np.transpose(temp[choose_z[0]:choose_z[1], choose_y[0]:choose_y[1], choose_x[0]:choose_x[1]], (2, 1, 0))  # to (z,y,x)
            elif indexing == 'xy':
                temp = np.transpose(temp[choose_z[0]:choose_z[1], choose_y[0]:choose_y[1], choose_x[0]:choose_x[1]], (1, 0, 2))  # to (y,x,z)
            #logger.warning(f"Read {fieldname} from {filename} with shape {temp.shape} before squeeze")
            temp = temp.squeeze()
            #temp = temp.reshape([d for d in temp.shape if d != 0]) # remove dimensions of size 0
            field.append(temp)
        except Exception as e:
            if isinstance(e, PermissionError):
                logger.error(
                    f"Permission denied while reading {fieldname} from {current_file_path}. "
                    "Check file/folder ACLs and group permissions."
                )
            else:
                logger.warning(f"Failed to read {fieldname} from {filename} using path {files_path}: {e}")
            #logger.warning(f"{temp.shape = }")
            raise e
    a = np.moveaxis(np.array(field), 0, -1)
    a = apply_filters(a, filters, fieldname=fieldname, filename=filenames, verbose=verbose)
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

def parse_simulation_data(files_path):
    """
    Parse SimulationData.txt file that can be in either old or new format.
    
    Old format uses key patterns like "x-Length" and "Number of cells (x)"
    New format uses "Simulation domain" and "Grid resolution" with comma-separated values
    
    Returns:
    - dict: A dictionary containing Lx, Ly, Lz, nxc, nyc, nzc, dt, and qom values
    """
    files_path = _resolve_files_path(files_path)
    try:
        sim_path = files_path
        if os.path.isdir(files_path):
            sim_path = os.path.join(files_path, "SimulationData.txt")
        elif os.path.basename(files_path) == "SimulationData.txt":
            sim_path = files_path
        else:
            sim_path = os.path.join(os.path.dirname(os.path.normpath(files_path)), "SimulationData.txt")
    except Exception as e:
        logger.error(f"Error determining simulation data path: {files_path = }")
        raise e

    try:
        f = open(sim_path, "r")
    except Exception:
        # Remove the last folder from files_path and try again
        try:
            fallback_path = os.path.join(os.path.dirname(os.path.normpath(files_path)), "SimulationData.txt")
            f = open(fallback_path, "r")
        except Exception as e:
            logger.error(f"Failed to open SimulationData.txt at {files_path = } and fallback path {fallback_path = }")
            raise e
    
    content = f.readlines()
    f.close()
    
    # Initialize variables
    result = {
        'Lx': None, 'Ly': None, 'Lz': None,
        'nxc': None, 'nyc': None, 'nzc': None,
        'dt': None, 'qom': []
    }
    
    # Try to detect format by checking for key indicators
    content_str = ''.join(content)
    is_new_format = "Simulation domain" in content_str and "Grid resolution" in content_str
    
    if is_new_format:
        # Parse new format
        for line in content:
            line_clean = line.strip()
            
            # Parse Simulation domain (e.g., "Simulation domain = 30 x 30 x 1")
            if line_clean.startswith("Simulation domain"):
                parts = line_clean.split("=")[1].strip().split("x")
                result['Lx'] = float(parts[0].strip())
                result['Ly'] = float(parts[1].strip())
                result['Lz'] = float(parts[2].strip())
            
            # Parse Grid resolution (e.g., "Grid resolution = 100 x 100 x 1")
            elif line_clean.startswith("Grid resolution"):
                parts = line_clean.split("=")[1].strip().split("x")
                result['nxc'] = int(parts[0].strip())
                result['nyc'] = int(parts[1].strip())
                result['nzc'] = int(parts[2].strip())
            
            # Parse Time step size (e.g., "Time step size (dt) = 0.125")
            elif "Time step size" in line_clean:
                result['dt'] = float(line_clean.split("=")[1].strip())
            
            # Parse Charge-to-mass ratio
            elif "Charge-to-mass ratio" in line_clean:
                qom_val = float(line_clean.split("=")[1].strip())
                result['qom'].append(qom_val)
    else:
        # Parse old format
        for n in content:
            if "QOM" in n:
                result['qom'].append(float(re.split("=", re.sub(" |\n", "", n))[-1]))
            if "x-Length" in n:
                result['Lx'] = float(re.split("=", re.sub(" |\n", "", n))[1])
            if "y-Length" in n:
                result['Ly'] = float(re.split("=", re.sub(" |\n", "", n))[1])
            if "z-Length" in n:
                result['Lz'] = float(re.split("=", re.sub(" |\n", "", n))[1])
            if "Number of cells (x)" in n:
                result['nxc'] = int(re.split("=", re.sub(" |\n", "", n))[1])
            if "Number of cells (y)" in n:
                result['nyc'] = int(re.split("=", re.sub(" |\n", "", n))[1])
            if "Number of cells (z)" in n:
                result['nzc'] = int(re.split("=", re.sub(" |\n", "", n))[1])
            if "Time step" in n:
                result['dt'] = float(re.split("=", re.sub(" |\n", "", n))[1])
    
    # Fallback: if qom is still empty, try to read from qom[%d] = value format
    if not result['qom']:
        for line in content:
            # Match lines like "qom[%d] = -64" or "qom[0] = 1.5"
            if re.match(r'qom\[', line, re.IGNORECASE):
                try:
                    qom_val = float(re.split("=", re.sub(" |\n", "", line))[-1])
                    result['qom'].append(qom_val)
                except (ValueError, IndexError):
                    pass
    
    return result


def build_XY(files_path, choose_x=DEFAULT_CHOOSE_X, choose_y=DEFAULT_CHOOSE_Y, 
             choose_z=DEFAULT_CHOOSE_Z, indexing=DEFAULT_INDEXING):
    """
    Read grid parameters from SimulationData.txt and build coordinate meshgrids.
    Supports both old and new SimulationData.txt formats.
    """
    files_path = _resolve_files_path(files_path)
    sim_data = parse_simulation_data(files_path)
    
    Lx = sim_data['Lx']
    Ly = sim_data['Ly']
    Lz = sim_data['Lz']
    nxc = sim_data['nxc']
    nyc = sim_data['nyc']
    nzc = sim_data['nzc']
    #qom = sim_data['qom']
    
    # The x, y and z axes are set.
    x = np.linspace(0, Lx, nxc + 1)
    y = np.linspace(0, Ly, nyc + 1)
    z = np.linspace(0, Lz, nzc + 1)
    
    if choose_x is None:
        choose_x = [0, nxc]
    if choose_y is None:
        choose_y = [0, nyc]
    if choose_z is None:
        choose_z = [0, nzc]
    
    if isinstance(choose_x[0], list):
        if isinstance(choose_y[0], list):
            raise ValueError("choose_x and choose_y must be of the same type")
        X = []
        Y = []
        if nzc > 1:
            Z = []
        for i in range(len(choose_x)):  # deal with the situation where the user wants to extract multiple regions
            assert len(choose_x) == len(choose_y), "choose_x and choose_y must have the same length"
            if nzc > 1:
                assert len(choose_x) == len(choose_z), "choose_x and choose_y must have the same length"
                X_i, Y_i, Z_i = np.meshgrid(
                    x[choose_x[i][0]:choose_x[i][1]], 
                    y[choose_y[i][0]:choose_y[i][1]], 
                    z[choose_z[i][0]:choose_z[i][1]], 
                    indexing=indexing
                )
                X.append(X_i)
                Y.append(Y_i)
                Z.append(Z_i)
            else:
                X_i, Y_i = np.meshgrid(
                    x[choose_x[i][0]:choose_x[i][1]], 
                    y[choose_y[i][0]:choose_y[i][1]], 
                    indexing=indexing
                )
                X.append(X_i)
                Y.append(Y_i)
        X = np.concatenate(X, axis=1)
        Y = np.concatenate(Y, axis=1)
        if nzc > 1:
            Z = np.concatenate(Z, axis=1)
    else:
        if nzc > 1:
            X, Y, Z = np.meshgrid(
                x[choose_x[0]:choose_x[1]], 
                y[choose_y[0]:choose_y[1]], 
                z[choose_z[0]:choose_z[1]], 
                indexing=indexing
            )
        else:
            X, Y = np.meshgrid(
                x[choose_x[0]:choose_x[1]], 
                y[choose_y[0]:choose_y[1]], 
                indexing=indexing
            )
    
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
    files_path = _resolve_files_path(files_path)
    # Determine the correct path for SimulationData.txt
    # Try looking in the subdirectory of the first filename, or fall back to files_path
    lookup_path = files_path
    try:
        lookup_path = os.path.join(files_path, filenames[0].rsplit("/", 1)[0]) + os.sep
    except Exception:
        lookup_path = files_path
    
    # Parse simulation data using the new unified parser
    sim_data = parse_simulation_data(lookup_path)
    qom = sim_data['qom']

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
                          extract_fields=species_to_list(request_features), choose_species=choose_species, 
                          choose_x=choose_x[i], choose_y=choose_y[i], choose_z=choose_z[i], verbose=verbose))
            if verbose:
                logger.info(f"{features[-1].shape =}")
            
            targets.append(read_files(files_path, filenames, fields_to_read, qom, targets_dtype,
                            extract_fields=species_to_list(request_targets), choose_species=choose_species, 
                            choose_x=choose_x[i], choose_y=choose_y[i], choose_z=choose_z[i], verbose=verbose)) 
            if verbose:
                logger.info(f"{targets[-1].shape =}")
        features = np.concatenate(features,axis=2)
        targets = np.concatenate(targets,axis=2) 
    else:
        features = read_files(files_path, filenames, fields_to_read, qom, features_dtype, 
                            extract_fields=species_to_list(request_features), choose_species=choose_species, 
                            choose_x=choose_x, choose_y=choose_y, choose_z=choose_z, verbose=verbose)
        targets = read_files(files_path, filenames, fields_to_read, qom, targets_dtype, 
                            extract_fields=species_to_list(request_targets), choose_species=choose_species, 
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
                    logger.info(f"Attempting to read {filename = }")
                    logger.info(f"Available data keys are {data.keys() = }")
                    logger.info(f"Attempting to extract {extract_field_index = } which should a list of length 2,")
                    logger.info(f"where the first element is the field name and the second element is the species name")
                    logger.warning(f"The extracted field is {data[extract_field_index[0]] = }")
                    raise e
            else:
                #logger.info(data[extract_field_index])
                try:
                    out.append(data[extract_field_index])
                except Exception as e:
                    logger.info(f"Attempting to read {filename = }")
                    logger.info(f"Available data keys are {data.keys() = }")
                    logger.info(f"Attempting to extract {extract_field_index = } which should be a field name")
                    raise e
            outshape = out[-1].shape
        out2.append(np.array(out))
    try:
        out2 = np.array(out2, dtype=dtype)
        if len(out2.shape) > 4:
            logger.warning(f"read_data: prior to appending {outshape = }")
            logger.warning(f"Output shape {out2.shape} has more than 4 dimensions, we try fixing by removing single-dimensional entries")
            out2 = out2[...,0].squeeze()
            logger.warning(f"Output shape after fixing {out2.shape}")
        # Add logging before transpose
        logger.info(f"About to transpose array with shape {out2.shape}")
        
        # Only transpose if we have 4 dimensions
        if len(out2.shape) == 4:
            out2 = out2.transpose(0,2,3,1)
        else:
            logger.error(f"Expected 4D array for transpose, got shape {out2.shape}")
            raise ValueError(f"Cannot transpose array with shape {out2.shape} - expected 4 dimensions, got {len(out2.shape)}")
            
    except Exception as e:
        logger.error(f"Failed to process array. Current shape: {np.array(out2).shape if isinstance(out2, list) else out2.shape}")
        logger.error(f"dtype: {dtype}, outshape from last file: {outshape}")
        raise e
        
    return out2  # we want to have the time as the first index, then x, then y, then the field


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
    - divB: The divergence of the magnetic field. DEPRECATED and absent in new ipic3d version
    - rho: The charge density.
    - N: The number of particles per cell in the particle in cell simulation. DEPRECATED and absent in new ipic3d version
    - Qrem: The remaining charge in the particle in cell simulation. DEPRECATED and absent in new ipic3d version
    - Jx, Jy, Jz: The current density components.
    - Pxx, Pxy, Pxz, Pyy, Pyz, Pzz: The pressure tensor components.
    - PIxx, PIxy, PIxz, PIyy, PIyz, PIzz: The stress tensor components.
    - Ppar, Pperp: The parallel and perpendicular pressure.
    - q: The heat flux.
    
    """
    files_path = _resolve_files_path(files_path)
    #logger.info(f"{files_path = }")
    #logger.info(f"{filenames = }")
    try:
        first = filenames[0] if isinstance(filenames, list) else filenames
        if "/" in first:
            folder_path = os.path.join(files_path, first.rsplit("/", 1)[0])
        else:
            folder_path = files_path
    except Exception as e:
        logger.error(f"Error determining folder path from files_path and filenames: {files_path = }, {filenames = }")
        raise e
    #logger.info(f"Reading data from folder: {folder_path}")
    X, Y = build_XY(folder_path,choose_x=DEFAULT_CHOOSE_X, choose_y=DEFAULT_CHOOSE_Y, choose_z=DEFAULT_CHOOSE_Z, indexing=DEFAULT_INDEXING)
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
        if fields in fields_to_read and fields_to_read[fields]:
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
                    data[f'V{component}'][species] = data[f'J{component}'][species]/(data['rho'][species]+small*np.sign(qom[i]))
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
                            if component_1 <= component_2: # we only read the upper triangular part of the pressure tensor, because it is symmetric, and we will fill the lower triangular part later
                                PIread = read_fieldname(files_path,filenames,f'P{component_1}{component_2}_{i}',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
                            #else:
                            #    PIread = read_fieldname(files_path,filenames,f'PI{component_2}{component_1}_{i}',choose_x,choose_y,choose_z,verbose=verbose, **kwargs)
                            if species in data[f'PI{component_1}{component_2}']:
                                data[f'PI{component_1}{component_2}'][species] += PIread
                            else:
                                data[f'PI{component_1}{component_2}'][species] = PIread
                        except:
                            if verbose:
                                logger.info(f'Component P{component_1}{component_2} for species {species} missing because tensor is symmetric')
                for species in data[f'PI{component_1}{component_2}']: # because now the number of species has potentially changed
                    i = choose_species.index(species)
                    data[f'P{component_1}{component_2}'][species]  = (data[f'PI{component_1}{component_2}'][species] - \
                                data[f'J{component_1}'][species]*data[f'J{component_2}'][species]/(data[f'rho'][species]+small*np.sign(qom[i])))/qom[i]

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
                    logger.warning(f"Failed to calculate Ppar for {species = } likely due to missing fields, see: {e}")
                try:
                    data['Pperp'][species] = (data['Pxx'][species] + data['Pyy'][species] + data['Pzz'][species] - data['Ppar'][species])/2
                except Exception as e:
                    logger.warning(f"Failed to calculate Pperp for {species} likely due to missing fields, see: {e}")
        if "gyro_radius" in fields_to_read and fields_to_read["gyro_radius"]:
            try:
                data['gyro_radius'] = {}
                for species in data['rho']:
                    i = choose_species.index(species)
                    #p = data['Pxx'][species]+data['Pyy'][species]+data['Pzz'][species]
                    vth=np.sqrt(np.abs(qom[i]*data['Pperp'][species]/(np.abs(data['rho'][species])+small)))
                    data['gyro_radius'][species] = np.abs(vth/(qom[i]*data['Bmagn']))
            except Exception as e:
                logger.warning(f"Failed to calculate gyro_radius, see: {e}")
    if "divP" in fields_to_read and fields_to_read["divP"] or "Ohmres" in fields_to_read and fields_to_read["Ohmres"]:
        if verbose:
            logger.info(f"computing divP and or Ohmres")

        dx = X[1,0]-X[0,0]
        dy = Y[0,1]-Y[0,0]

        if not 'e' in choose_species:
            raise ValueError(f"Calculating divP_e or Ohmres without electron species cannot be done")
        
        

        data['EPx'] = -(highdiff(data['Pxx']['e'], dx, dy, axis=0, mode='wrap') + highdiff(data['Pxy']['e'], dx, dy, axis=1, mode='wrap'))/(-data['rho']['e']) # density in ECsim is negative (electron charge density)
        data['EPy'] = -(highdiff(data['Pxy']['e'], dx, dy, axis=0, mode='wrap') + highdiff(data['Pyy']['e'], dx, dy, axis=1, mode='wrap'))/(-data['rho']['e']) # density in ECsim is negative (electron charge density)
        data['EPz'] = -(highdiff(data['Pxz']['e'], dx, dy, axis=0, mode='wrap') + highdiff(data['Pyz']['e'], dx, dy, axis=1, mode='wrap'))/(-data['rho']['e']) # density in ECsim is negative (electron charge density)
        
        if "Ohmres" in fields_to_read and fields_to_read["Ohmres"]:
            #logger.info(f"{data['Bx'].shape = }")
            #B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,0)
            #E = np.array([data['Ex'], data['Ey'], data['Ez']]).transpose(1,2,3,0)
            Jtotx = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
            Jtoty = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
            Jtotz = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)

            # = np.array([Jtotx, Jtoty, Jtotz]).transpose(1,2,3,0)
            data['EHallx'], data['EHally'], data['EHallz'] = do_cross(Jtotx,Jtoty,Jtotz,data['Bx'],data['By'],data['Bz'])/(-data['rho']['e']) # EHx,EHy,EHz=do_cross(Jx,Jy,Jz,Bx,By,Bz)/(-rho_0)
            norm = 0
            uCMx = 0
            uCMy = 0
            uCMz = 0
            for i, species in enumerate(data['rho'].keys()):
                uCMx += (data['rho'][species]/qom[i])*data['Vx'][species]
                uCMy += (data['rho'][species]/qom[i])*data['Vy'][species]
                uCMz += (data['rho'][species]/qom[i])*data['Vz'][species]
                norm += data['rho'][species]/qom[i]
            uCMx /= norm
            uCMy /= norm
            uCMz /= norm
            data['EMHDx'], data['EMHDy'], data['EMHDz'] = do_cross(uCMx,uCMy,uCMz,data['Bx'],data['By'],data['Bz']) # TODO: fix sign, should be minus
            # data['EMHDx'], data['EMHDy'], data['EMHDz'] = -do_cross(uCMx,uCMy,uCMz,data['Bx'],data['By'],data['Bz']) 
            data['Ohmresx'] = data['Ex'] + data['EMHDx'] - data['EHallx'] - data['EPx']
            data['Ohmresy'] = data['Ey'] + data['EMHDy'] - data['EHally'] - data['EPy']
            data['Ohmresz'] = data['Ez'] + data['EMHDz'] - data['EHally'] - data['EPz']
        

            

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
                        (data['Jx'][species]**2+data['Jy'][species]**2+data['Jz'][species]**2)*data[f'J{component}'][species]/(2*qom[i]*data[f'rho'][species]**2+small*np.sign(qom[i])) - \
                        (data['Pxx'][species] + data[f'Pyy'][species] + data[f'Pzz'][species])*data[f'J{component}'][species]/(2*data['rho'][species]+small*np.sign(qom[i])) - \
                        (data['Jx'][species]*data[f'Px{component}'][species] + data['Jy'][species]*data[f'Py{component}'][species] + data['Jz'][species]*data[f'Pz{component}'][species])/(data['rho'][species]+small*np.sign(qom[i]))
            except Exception as e:
                logger.warning(f"Failed to calculate q{component} see: {e}")
                #logger.info(f"{data[f'q{component}'].keys() = }")
            if 'EF' not in fields_to_read or not fields_to_read['EF']:
                del data[f'EF{component}']
    empty_keys, empty_nested_keys = _collect_empty_data_keys(data)
    if empty_keys:
        logger.warning(f"read_data empty top-level keys: {empty_keys}")
    if empty_nested_keys:
        logger.warning(f"read_data empty nested keys: {empty_nested_keys}")
    return data


def _collect_experiment_filenames(experiment_dir):
    """
    Collect sorted field filenames for an experiment directory.

    Supports both legacy ECSIM-style files (e.g. *-Fields_XXXXX.h5/.pkl/.npz)
    and iPiC3D-like directory naming where iteration appears as Fields_<iter>.
    """
    filenames = sorted([
        n for n in os.listdir(experiment_dir)
        if "-Fields_" in n and (n.endswith(".pkl") or n.endswith(".h5") or n.endswith(".npz"))
    ])
    if filenames == []:
        filenames = sorted([
            re.search(r'Fields_(\d+)', n).group(1)
            for n in os.listdir(experiment_dir)
            if re.search(r'Fields_(\d+)', n)
        ])
    return filenames


def _select_filenames_by_time(filenames, choose_times):
    """Select filenames according to choose_times semantics used in get_exp_times."""
    if choose_times is None:
        return filenames
    if isinstance(choose_times, int):
        return filenames[choose_times:]
    try:
        return [filenames[i] for i in choose_times]
    except Exception as e:
        logger.info(f"Inconsistent size: {len(filenames) = }  {len(choose_times) = }")
        raise e


def _extract_times_from_filenames(selected_filenames, dt):
    """Extract physical times from selected filenames/iterations."""
    times = []
    for n in selected_filenames:
        n_str = str(n)
        if n_str.endswith(".h5.pkl"):
            time_token = n_str[-13:-7]
        elif n_str.endswith(".npz"):
            time_token = n_str[-9:-4]
        elif n_str.endswith(".h5"):
            time_token = n_str[-9:-3]
        else:  # deal with new ipic3d version where the time is directly after "Fields_"
            match = re.search(r'Fields_(\d+)', n_str)
            time_token = match.group(1) if match else int(n_str)
        times.append(int(time_token) * dt)
    return times

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
    
    files_path = _resolve_files_path(files_path)
    data = {}
    for experiment in experiments:
        logger.info(f" reading {files_path}/{experiment}/SimulationData.txt")
        
        # Parse simulation data using the new unified parser
        experiment_path = os.path.join(files_path, experiment)
        sim_data = parse_simulation_data(experiment_path)
        
        qom = sim_data['qom']
        Lx = sim_data['Lx']
        Ly = sim_data['Ly']
        Lz = sim_data['Lz']
        nxc = sim_data['nxc']
        nyc = sim_data['nyc']
        nzc = sim_data['nzc']
        dt = sim_data['dt']
        
        logger.info(f"{Lx = }, {Ly = }, {nxc = }, {nyc = }")
        # The x, y and z axes are set.
        x = np.linspace(0, Lx, nxc + 1)
        y = np.linspace(0, Ly, nyc + 1)
        z = np.linspace(0, Lz, nzc + 1)    
        #compute dx and dy to be used for the gradients computation
        #dx = Lx/nxc
        #dy = Ly/nyc
        # sorted(os.listdir()) creates a sorted list containing the .h5 filenames, os.listdir() alone would put them in random order.
        filenames = _collect_experiment_filenames(os.path.join(files_path, experiment))
        selected_filenames = _select_filenames_by_time(filenames, choose_times)
        try:
            logger.info(f"selected_filenames = {selected_filenames}")
            times = _extract_times_from_filenames(selected_filenames, dt)
        except Exception as e:
            logger.info(f"Failed to extract times from {selected_filenames = }")
            logger.info(f"{selected_filenames=}")
            raise e
        #logger.info(times)
        #logger.info(f"Selected filenames for experiment {experiment}: {selected_filenames}")
        data[experiment] = read_data(os.path.join(files_path, experiment) + os.sep,selected_filenames,fields_to_read,qom,
                                     choose_species=choose_species,choose_x=choose_x,choose_y=choose_y,choose_z=choose_z,verbose=verbose, **kwargs)
        if choose_x is None:
            choose_x = [0, x.shape[0] - 1]
        if choose_y is None:
            choose_y = [0, y.shape[0] - 1]
        if choose_z is None:
            choose_z = [0, z.shape[0] - 1]
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


def _detect_ipic3d_species(files_path, sample_file=None):
    """
    Detect species labels from iPiC3D HDF5 files.

    Returns a list of species suffixes (e.g., ["0", "1"]).
    """
    files_path = _resolve_files_path(files_path)
    all_hdf_files = sorted(glob.glob(os.path.join(files_path, "proc*.hdf")))
    if not all_hdf_files:
        raise FileNotFoundError(f"No proc*.hdf files found in {files_path}")
    sample_file = sample_file or all_hdf_files[0]
    import h5py
    with h5py.File(sample_file, "r") as f:
        if "moments" not in f:
            return []
        species_groups = [k for k in f["moments"].keys() if k.startswith("species_")]
    species = [k.split("species_", 1)[1] for k in species_groups]
    # Sort numerically if possible, otherwise lexicographically
    try:
        species_sorted = sorted(species, key=lambda s: int(s))
    except ValueError:
        species_sorted = sorted(species)
    return species_sorted


def _expand_to_ecsim(field_xy, nxc, nyc, nzc):
    """
    Convert a 2D field with shape (nx, ny) into ECSIM format (nz+1, ny+1, nx+1).

    The ECSIM format is (z, y, x). This helper places the provided field into
    the z=0 plane and leaves the final index as a guard (consistent with ECSIM readers
    that slice to -1).
    """
    nx, ny = field_xy.shape
    out = np.zeros((nzc + 1, nyc + 1, nxc + 1), dtype=field_xy.dtype)
    max_x = min(nx, nxc)
    max_y = min(ny, nyc)
    out[0, :max_y, :max_x] = field_xy[:max_x, :max_y].T
    return out


def _default_ipic3d_fields_to_read():
    """
    Default fields dictionary for iPiC3D -> ECSIM conversion.
    """
    return {
        "B": True,
        "B_ext": False,
        "divB": False,
        "E": True,
        "E_ext": False,
        "rho": True,
        "J": True,
        "P": True,
        "PI": False,
        "N": False,
        "Qrem": False,
        "Heat_flux": False,
        "EF": False,
    }


def _get_ipic3d_hdf_files(files_path):
    all_hdf_files = sorted(glob.glob(os.path.join(files_path, "proc*.hdf")))
    if not all_hdf_files:
        raise FileNotFoundError(f"No proc*.hdf files found in {files_path}")
    return all_hdf_files


def _normalize_ipic3d_selection(sim_data, choose_x, choose_y, choose_z):
    nxc = sim_data['nxc']
    nyc = sim_data['nyc']
    nzc = sim_data['nzc']
    if choose_x is None:
        choose_x = [0, nxc]
    if choose_y is None:
        choose_y = [0, nyc]
    if choose_z is None:
        choose_z = [0, nzc]
    return choose_x, choose_y, choose_z


def _inspect_ipic3d_available_fields(files_path, sample_file=None):
    sample_file = sample_file or _get_ipic3d_hdf_files(files_path)[0]
    import h5py

    available = {
        "fields": set(),
        "moments": {},
    }
    with h5py.File(sample_file, "r") as f:
        if "fields" in f:
            available["fields"] = set(f["fields"].keys())
        if "moments" in f:
            for species_group in f["moments"].keys():
                if species_group.startswith("species_"):
                    species = species_group.split("species_", 1)[1]
                    available["moments"][species] = set(f["moments"][species_group].keys())
    return available


def _build_ipic3d_conversion_requests(fields_to_read, choose_species):
    requests = []

    for field_prefix in ["B", "E"]:
        if fields_to_read.get(field_prefix, False):
            for comp in ["x", "y", "z"]:
                requests.append({
                    "output_name": f"{field_prefix}{comp}",
                    "path_prefix": "fields",
                    "field_name": f"{field_prefix}{comp}",
                })

        if fields_to_read.get(f"{field_prefix}_ext", False) and field_prefix == "B":
            for comp in ["x", "y", "z"]:
                requests.append({
                    "output_name": f"{field_prefix}{comp}_ext",
                    "path_prefix": "fields",
                    "field_name": f"{field_prefix}{comp}_ext",
                })

    if fields_to_read.get("divB", False):
        requests.append({
            "output_name": "divB",
            "path_prefix": "fields",
            "field_name": "divB",
        })

    for field_name in ["rho", "N", "Qrem"]:
        if fields_to_read.get(field_name, False):
            for i, species in enumerate(choose_species):
                requests.append({
                    "output_name": f"{field_name}_{i}",
                    "path_prefix": f"moments/species_{species}",
                    "field_name": field_name,
                })

    if fields_to_read.get("J", False):
        for comp in ["x", "y", "z"]:
            for i, species in enumerate(choose_species):
                requests.append({
                    "output_name": f"J{comp}_{i}",
                    "path_prefix": f"moments/species_{species}",
                    "field_name": f"J{comp}",
                })

    if fields_to_read.get("P", False) or fields_to_read.get("PI", False):
        for comp1 in ["x", "y", "z"]:
            for comp2 in ["x", "y", "z"]:
                for i, species in enumerate(choose_species):
                    requests.append({
                        "output_name": f"P{comp1}{comp2}_{i}",
                        "path_prefix": f"moments/species_{species}",
                        "field_name": f"P{comp1}{comp2}",
                    })

    return requests


def _build_ipic3d_analysis_requests(fields_to_read, choose_species):
    requests = []

    for field_prefix in ["B", "E"]:
        if fields_to_read.get(field_prefix, False):
            for comp in ["x", "y", "z"]:
                requests.append({
                    "output_name": f"{field_prefix}{comp}",
                    "path_prefix": "fields",
                    "field_name": f"{field_prefix}{comp}",
                })
        if fields_to_read.get(f"{field_prefix}_ext", False):
            for comp in ["x", "y", "z"]:
                requests.append({
                    "output_name": f"{field_prefix}{comp}_ext",
                    "path_prefix": "fields",
                    "field_name": f"{field_prefix}{comp}_ext",
                })

    if fields_to_read.get("divB", False):
        requests.append({
            "output_name": "divB",
            "path_prefix": "fields",
            "field_name": "divB",
        })

    for field_name in ["rho", "N", "Qrem"]:
        if fields_to_read.get(field_name, False):
            for i, species in enumerate(choose_species):
                if species is not None:
                    requests.append({
                        "output_name": f"{field_name}_{i}",
                        "path_prefix": f"moments/species_{i}",
                        "field_name": field_name,
                    })

    if fields_to_read.get("J", False):
        for comp in ["x", "y", "z"]:
            for i, species in enumerate(choose_species):
                if species is not None:
                    requests.append({
                        "output_name": f"J{comp}_{i}",
                        "path_prefix": f"moments/species_{i}",
                        "field_name": f"J{comp}",
                    })

    if fields_to_read.get("P", False) or fields_to_read.get("PI", False):
        for comp1 in ["x", "y", "z"]:
            for comp2 in ["x", "y", "z"]:
                for i, species in enumerate(choose_species):
                    if species is not None:
                        requests.append({
                            "output_name": f"P{comp1}{comp2}_{i}",
                            "path_prefix": f"moments/species_{i}",
                            "field_name": f"P{comp1}{comp2}",
                        })

    if fields_to_read.get("Heat_flux", False) or fields_to_read.get("EF", False):
        for comp in ["x", "y", "z"]:
            for i, species in enumerate(choose_species):
                if species is not None:
                    requests.append({
                        "output_name": f"EF{comp}_{i}",
                        "path_prefix": f"moments/species_{i}",
                        "field_name": f"EF{comp}",
                    })

    return requests


def _filter_ipic3d_requests_by_availability(requests, available, verbose=DEFAULT_VERBOSE):
    filtered_requests = []
    skipped = []
    for request in requests:
        path_prefix = request["path_prefix"]
        field_name = request["field_name"]
        if path_prefix == "fields":
            is_available = field_name in available["fields"]
        else:
            species = path_prefix.split("species_", 1)[1]
            is_available = field_name in available["moments"].get(species, set())
        if is_available:
            filtered_requests.append(request)
        else:
            skipped.append(request)

    if verbose and skipped:
        skipped_names = [request["output_name"] for request in skipped]
        logger.info(f"Skipping unavailable iPiC3D fields: {skipped_names}")

    return filtered_requests


def _read_ipic3d_cycles(files_path, cycles, requests, choose_x=DEFAULT_CHOOSE_X, choose_y=DEFAULT_CHOOSE_Y,
                        choose_z=DEFAULT_CHOOSE_Z, indexing=DEFAULT_INDEXING, skip_missing=True,
                        verbose=DEFAULT_VERBOSE):
    sim_data = parse_simulation_data(files_path)
    nxc = sim_data['nxc']
    nyc = sim_data['nyc']
    choose_x, choose_y, choose_z = _normalize_ipic3d_selection(sim_data, choose_x, choose_y, choose_z)
    all_hdf_files = _get_ipic3d_hdf_files(files_path)

    results = {request['output_name']: [] for request in requests}
    import h5py

    time_cycles = [f"cycle_{int(cycle)}" for cycle in cycles]
    cycle_fields = [dict() for _ in cycles]
    found_fields = [set() for _ in cycles]

    for file_path in all_hdf_files:
        rank_id = int(os.path.basename(file_path).replace("proc", "").replace(".hdf", ""))
        with h5py.File(file_path, "r") as f:
            topology = f['topology']
            cartesian_coord = topology['cartesian_coord'][()]
            cartesian_rank = topology['cartesian_rank'][()]
            if rank_id != cartesian_rank:
                raise ValueError(
                    f"Rank ID {rank_id} does not match cartesian rank {cartesian_rank} in file {file_path}"
                )

            for cycle_index, time_cycle in enumerate(time_cycles):
                for request in requests:
                    try:
                        field_data = find_field_in_hdf5(
                            f,
                            request['path_prefix'],
                            request['field_name'],
                            time_cycle,
                        )
                    except KeyError as exc:
                        if skip_missing:
                            continue
                        raise KeyError(
                            f"Field '{request['field_name']}' not found in any proc*.hdf for {time_cycle}"
                        ) from exc

                    output_name = request['output_name']
                    found_fields[cycle_index].add(output_name)
                    if output_name not in cycle_fields[cycle_index]:
                        cycle_fields[cycle_index][output_name] = np.zeros((nxc, nyc), dtype=field_data.dtype)

                    x0, y0, z0 = (np.array(field_data.shape) * cartesian_coord).astype(int)
                    nx_local, ny_local, nz_local = field_data.shape
                    if verbose == 'debug':
                        logger.info(f"Rank {rank_id} at position ({x0 = }, {y0 = }, {z0 = }) processing file {file_path}")
                        logger.info(f".   cartesian_coord: {cartesian_coord}")
                        logger.info(f" {nx_local = }, {ny_local = }, {nz_local = }")
                        logger.info(f" writing data to global arrays at indices x: {x0} to {x0 + nx_local}, y: {y0} to {y0 + ny_local}")

                    cycle_fields[cycle_index][output_name][x0:x0 + nx_local, y0:y0 + ny_local] = field_data[:, :, 0]

    for cycle_index, time_cycle in enumerate(time_cycles):
        for request in requests:
            output_name = request['output_name']
            if output_name not in found_fields[cycle_index]:
                if skip_missing:
                    continue
                raise KeyError(f"Field '{output_name}' not found in any proc*.hdf for {time_cycle}")

            sliced_field = cycle_fields[cycle_index][output_name][choose_x[0]:choose_x[1], choose_y[0]:choose_y[1]]
            if indexing == 'xy':
                sliced_field = sliced_field.T
            results[output_name].append(sliced_field)

    out = {}
    for output_name, field_times in results.items():
        if not field_times:
            continue
        field_times = np.array(field_times)
        if indexing in ['ij', 'xy']:
            out[output_name] = np.transpose(field_times, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported indexing: {indexing}")
    return out


def convert_ipic3d_to_ecsim_h5(
    input_folder,
    output_folder,
    cycles=None,
    fields_to_read=None,
    choose_species=None,
    choose_x=DEFAULT_CHOOSE_X,
    choose_y=DEFAULT_CHOOSE_Y,
    choose_z=DEFAULT_CHOOSE_Z,
    indexing=DEFAULT_INDEXING,
    simulation_name="iPIC3D",
    time_digits=6,
    overwrite=False,
    copy_aux_files=True,
    aux_globs=None,
    compression="gzip",
    compression_opts=4,
    allow_3d=False,
    skip_missing=True,
    verbose=DEFAULT_VERBOSE,
):
    """
    Convert iPiC3D HDF5 output (proc*.hdf) to ECSIM-compatible HDF5 files.

    Parameters are provided as kwargs with defaults so the function can be called
    programmatically or via a thin CLI wrapper.
    """
    input_folder = _resolve_files_path(input_folder)
    output_folder = str(Path(output_folder).expanduser())
    if fields_to_read is None:
        fields_to_read = _default_ipic3d_fields_to_read()
    if aux_globs is None:
        aux_globs = ["SimulationData.txt", "*.txt", "*.ini", "*.cfg", "*.json"]

    os.makedirs(output_folder, exist_ok=True)

    if copy_aux_files:
        for pattern in aux_globs:
            for file_path in glob.glob(os.path.join(input_folder, pattern)):
                if os.path.isfile(file_path):
                    dest_path = os.path.join(output_folder, os.path.basename(file_path))
                    if overwrite or not os.path.exists(dest_path):
                        shutil.copy2(file_path, dest_path)

    sim_data = parse_simulation_data(input_folder)
    nxc = sim_data["nxc"]
    nyc = sim_data["nyc"]
    nzc = sim_data["nzc"]
    if nzc is None:
        raise ValueError("SimulationData.txt missing nzc; cannot determine z dimension.")
    if nzc > 1 and not allow_3d:
        raise NotImplementedError("3D conversion is not implemented yet (nzc > 1).")

    if cycles is None:
        cycles, _ = ipic3D_available_cycles(input_folder)
    elif isinstance(cycles, int):
        cycles = [cycles]

    if choose_species is None:
        choose_species = _detect_ipic3d_species(input_folder)

    requests = _build_ipic3d_conversion_requests(fields_to_read, choose_species)
    available = _inspect_ipic3d_available_fields(input_folder)
    requests = _filter_ipic3d_requests_by_availability(requests, available, verbose=verbose)

    import h5py

    for cycle in cycles:
        time_label = f"{int(cycle):0{time_digits}d}"
        out_filename = f"{simulation_name}-Fields_{time_label}.h5"
        out_path = os.path.join(output_folder, out_filename)

        if os.path.exists(out_path) and not overwrite:
            if verbose:
                logger.info(f"Skipping existing file: {out_path}")
            continue

        cycle_fields = _read_ipic3d_cycles(
            input_folder,
            [cycle],
            requests,
            choose_x=choose_x,
            choose_y=choose_y,
            choose_z=choose_z,
            indexing=indexing,
            skip_missing=skip_missing,
            verbose=verbose,
        )
        field_datasets = {
            field_name: _expand_to_ecsim(field_data[:, :, 0], nxc, nyc, nzc)
            for field_name, field_data in cycle_fields.items()
        }

        with h5py.File(out_path, "w") as h5f:
            step_group = h5f.create_group("Step#0")
            block_group = step_group.create_group("Block")
            for field_name, field_data in field_datasets.items():
                field_group = block_group.create_group(field_name)
                field_group.create_dataset(
                    "0",
                    data=field_data,
                    compression=compression,
                    compression_opts=compression_opts,
                )

        if verbose:
            logger.info(f"Wrote {out_path} with {len(field_datasets)} fields")

    return True