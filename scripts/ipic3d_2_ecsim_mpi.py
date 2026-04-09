#!/usr/bin/env python3
"""
MPI converter: iPiC3D proc*.hdf -> one ECSIM-style HDF5 file for one cycle.

Example:
    mpirun -np 8 python ipic3d_to_ecsim_onecycle_mpi.py \
        --input_folder . \
        --output_folder ./converted \
        --cycles 19100 \
        --species 0,1,2,3 \
        --simulation_name iPIC3D

This version is adapted to raw iPiC3D moments stored as:
    /moments/species_X/rho/cycle_...
    /moments/species_X/Jx/cycle_...
    /moments/species_X/pXX/cycle_...
    /moments/species_X/Qxxx/cycle_...
"""

import os
import glob
import argparse
import numpy as np
import h5py
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

from datetime import datetime

startTime = datetime.now()

print("Starting MPI converter at ", startTime)

def parse_cycles_arg(cycles_arg, sample_file):
    if cycles_arg is not None:
        out = []
        for c in cycles_arg.split(","):
            c = c.strip()
            if not c:
                continue
            if c.startswith("cycle_"):
                c = c.split("cycle_", 1)[1]
            out.append(int(c))
        return sorted(out)

    # auto-detect from one sample dataset group
    cycles = set()
    with h5py.File(sample_file, "r") as f:
        # try fields first
        if "fields" in f:
            for field_name in f["fields"].keys():
                for key in f["fields"][field_name].keys():
                    if key.startswith("cycle_"):
                        cycles.add(int(key.split("cycle_", 1)[1]))
                if cycles:
                    break

        # fallback to moments if needed
        if not cycles and "moments" in f:
            for sp in f["moments"].keys():
                if sp.startswith("species_"):
                    for moment_name in f["moments"][sp].keys():
                        for key in f["moments"][sp][moment_name].keys():
                            if key.startswith("cycle_"):
                                cycles.add(int(key.split("cycle_", 1)[1]))
                        if cycles:
                            break
                if cycles:
                    break

    return sorted(cycles)

def parse_species_arg(species_arg, sample_file):
    if species_arg is not None:
        out = []
        for s in species_arg.split(","):
            s = s.strip()
            if not s:
                continue
            if s.startswith("species_"):
                s = s.split("species_", 1)[1]
            out.append(int(s))
        return out

    # auto-detect from sample file
    with h5py.File(sample_file, "r") as f:
        if "moments" not in f:
            return []
        species = []
        for key in f["moments"].keys():
            if key.startswith("species_"):
                species.append(int(key.split("species_", 1)[1]))
        return sorted(species)


def get_dataset_shape_and_dtype(sample_file, dataset_path):
    with h5py.File(sample_file, "r") as f:
        dset = f[dataset_path]
        return dset.shape, dset.dtype


def dataset_exists(h5file, path):
    try:
        h5file[path]
        return True
    except KeyError:
        return False


def build_requests(sample_file, species_list, time_cycle):
    """
    Build conversion requests by inspecting what exists in the sample file.
    """
    requests = []

    field_components = ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]
    pressure_map = {
        "Pxx": "pXX",
        "Pxy": "pXY",
        "Pxz": "pXZ",
        "Pyy": "pYY",
        "Pyz": "pYZ",
        "Pzz": "pZZ",
    }
    heat_flux_names = [
        "Qxxx", "Qxxy", "Qxxz", "Qxyy", "Qxyz",
        "Qxzz", "Qyyy", "Qyyz", "Qyzz", "Qzzz"
    ]

    with h5py.File(sample_file, "r") as f:
        # Fields group
        for name in field_components:
            raw_path = f"fields/{name}/{time_cycle}"
            if dataset_exists(f, raw_path):
                requests.append({
                    "output_name": name,
                    "raw_path_prefix": f"fields/{name}",
                })

        # Species moments
        for species in species_list:
            sp_group = f"moments/species_{species}"

            # rho
            raw_path = f"{sp_group}/rho/{time_cycle}"
            if dataset_exists(f, raw_path):
                requests.append({
                    "output_name": f"rho_{species}",
                    "raw_path_prefix": f"{sp_group}/rho",
                })

            # J
            for comp in ["Jx", "Jy", "Jz"]:
                raw_path = f"{sp_group}/{comp}/{time_cycle}"
                if dataset_exists(f, raw_path):
                    requests.append({
                        "output_name": f"{comp}_{species}",
                        "raw_path_prefix": f"{sp_group}/{comp}",
                    })

            # EF
            for comp in ["EFx", "EFy", "EFz"]:
                raw_path = f"{sp_group}/{comp}/{time_cycle}"
                if dataset_exists(f, raw_path):
                    requests.append({
                        "output_name": f"{comp}_{species}",
                        "raw_path_prefix": f"{sp_group}/{comp}",
                    })

            
            # Pressure tensor (raw iPIC names are pXX, pXY, ...)
            for out_name, raw_name in pressure_map.items():
                raw_path = f"{sp_group}/{raw_name}/{time_cycle}"
                if dataset_exists(f, raw_path):
                    requests.append({
                        "output_name": f"{out_name}_{species}",
                        "raw_path_prefix": f"{sp_group}/{raw_name}",
                    })

            # Heat flux tensor
            for qname in heat_flux_names:
                raw_path = f"{sp_group}/{qname}/{time_cycle}"
                if dataset_exists(f, raw_path):
                    requests.append({
                        "output_name": f"{qname}_{species}",
                        "raw_path_prefix": f"{sp_group}/{qname}",
                    })

    return requests


def get_all_proc_files(input_folder):
    files = sorted(glob.glob(os.path.join(input_folder, "proc*.hdf")))
    if not files:
        raise FileNotFoundError(f"No proc*.hdf files found in {input_folder}")
    return files


def parse_simulation_data(input_folder):
    """
    Read SimulationData.txt from input_folder.

    Expected examples:
        Simulation domain                      = 20 x 20 x 0.1
        Grid resolution                        = 1024 x 1024 x 1
        Time step size (dt)                    = 0.0625
        Charge-to-mass ratio       = -400
    """
    sim_path = os.path.join(input_folder, "SimulationData.txt")

    if not os.path.isfile(sim_path):
        raise FileNotFoundError(f"Could not find SimulationData.txt at: {sim_path}")

    result = {
        "Lx": None, "Ly": None, "Lz": None,
        "nxc": None, "nyc": None, "nzc": None,
        "dt": None, "qom": []
    }

    with open(sim_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Simulation domain = 20 x 20 x 0.1
            if line.startswith("Simulation domain"):
                rhs = line.split("=", 1)[1].strip()
                vals = [v.strip() for v in rhs.split("x")]
                if len(vals) == 3:
                    result["Lx"] = float(vals[0])
                    result["Ly"] = float(vals[1])
                    result["Lz"] = float(vals[2])

            # Grid resolution = 1024 x 1024 x 1
            elif line.startswith("Grid resolution"):
                rhs = line.split("=", 1)[1].strip()
                vals = [v.strip() for v in rhs.split("x")]
                if len(vals) == 3:
                    result["nxc"] = int(vals[0])
                    result["nyc"] = int(vals[1])
                    result["nzc"] = int(vals[2])

            # Time step size (dt) = 0.0625
            elif "Time step size" in line and "(dt)" in line:
                rhs = line.split("=", 1)[1].strip()
                result["dt"] = float(rhs)

            # Charge-to-mass ratio = ...
            elif line.startswith("Charge-to-mass ratio"):
                rhs = line.split("=", 1)[1].strip()
                result["qom"].append(float(rhs))

    missing = [k for k in ["nxc", "nyc", "nzc"] if result[k] is None]
    if missing:
        raise ValueError(f"Missing required keys in {sim_path}: {missing}")

    return result

def infer_global_layout(input_folder, all_hdf_files):
    """
    Infer global layout using:
      - global grid size from SimulationData.txt
      - process grid size from proc*.hdf topology/cartesian_coord
    """
    sim_data = parse_simulation_data(input_folder)

    max_coords = np.array([0, 0, 0], dtype=int)
    for file_path in all_hdf_files:
        with h5py.File(file_path, "r") as f:
            coords = np.array(f["topology/cartesian_coord"][()], dtype=int)
            max_coords = np.maximum(max_coords, coords)

    xlen, ylen, zlen = (max_coords + 1).astype(int)

    nx_global = int(sim_data["nxc"])
    ny_global = int(sim_data["nyc"])
    nz_global = int(sim_data["nzc"])

    if nx_global % xlen != 0 or ny_global % ylen != 0:
        raise ValueError(
            f"Global grid ({nx_global}, {ny_global}) is not divisible by "
            f"process grid ({xlen}, {ylen})."
        )

    return {
        "nx_global": nx_global,
        "ny_global": ny_global,
        "nz_global": nz_global,
        "xlen": int(xlen),
        "ylen": int(ylen),
        "zlen": int(zlen),
    }


def assemble_fields_mpi(all_hdf_files, requests, time_cycle, nx_global, ny_global, xlen, ylen):
    """
    Assemble global 2D fields from proc*.hdf tiles using MPI.

    Placement rule:
      - global tile size is inferred from SimulationData global shape / process grid
      - tile origin comes from cartesian_coord
      - raw local block is cropped to the target tile size if needed
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_files = all_hdf_files[rank::size]

    local_fields = {
        req["output_name"]: np.zeros((nx_global, ny_global), dtype=np.float64)
        for req in requests
    }

    tile_nx = nx_global // xlen
    tile_ny = ny_global // ylen

    for file_path in local_files:
        rank_id = int(os.path.basename(file_path).replace("proc", "").replace(".hdf", ""))

        with h5py.File(file_path, "r") as f:
            topology = f["topology"]
            cartesian_coord = np.array(topology["cartesian_coord"][()], dtype=int)
            cartesian_rank = int(np.array(topology["cartesian_rank"][()]).item())

            if cartesian_rank != rank_id:
                raise ValueError(
                    f"Rank mismatch in {file_path}: filename rank {rank_id}, topology rank {cartesian_rank}"
                )

            cx, cy, cz = cartesian_coord
            x0 = cx * tile_nx
            y0 = cy * tile_ny

            for req in requests:
                full_path = f"{req['raw_path_prefix']}/{time_cycle}"
                if full_path not in f:
                    continue

                field_data = np.array(f[full_path])
                if field_data.ndim != 3:
                    raise ValueError(
                        f"Expected 3D dataset for {full_path}, got shape {field_data.shape}"
                    )

                nx_local, ny_local, nz_local = field_data.shape

                # Crop to the target tile size if raw tiles contain extra cells/ghost cells
                use_nx = min(tile_nx, nx_local)
                use_ny = min(tile_ny, ny_local)

                if x0 + use_nx > nx_global or y0 + use_ny > ny_global:
                    raise ValueError(
                        f"Tile placement exceeds bounds for {req['output_name']} in {file_path}: "
                        f"coord={cartesian_coord}, x0={x0}, y0={y0}, "
                        f"use_nx={use_nx}, use_ny={use_ny}, "
                        f"nx_global={nx_global}, ny_global={ny_global}"
                    )

                local_fields[req["output_name"]][x0:x0 + use_nx, y0:y0 + use_ny] = field_data[:use_nx, :use_ny, 0]

    if rank == 0:
        global_fields = {
            name: np.zeros((nx_global, ny_global), dtype=np.float64)
            for name in local_fields
        }
    else:
        global_fields = {name: None for name in local_fields}

    for name in local_fields:
        comm.Reduce(local_fields[name], global_fields[name], op=MPI.SUM, root=0)

    return global_fields


def write_ecsim_h5(output_path, field_datasets, compression="gzip", compression_opts=4):
    """
    Write fields in the ECSIM-style layout expected by read_fieldname():

        dataset shape = (2, ny+1, nx+1)   # interpreted as (z, y, x)

    Why:
    - read_fieldname() assumes .h5 datasets are ordered as (z, y, x)
    - it uses default slices [0:shape-1], i.e. exclusive upper bounds
    - old ECSIM files therefore use an extra point in x and y, and 2 planes in z

    We store the physical 2D field in plane 0, upper-left block:
        out[0, :ny, :nx] = field_data.T

    Duplicating into plane 1 is optional, but safer for compatibility.
    """
    with h5py.File(output_path, "w") as h5f:
        step_group = h5f.create_group("Step#0")
        block_group = step_group.create_group("Block")

        for field_name, field_data in field_datasets.items():
            arr = np.asarray(field_data)

            # Reduce to 2D if needed
            if arr.ndim == 3:
                if arr.shape[-1] == 1:
                    arr = arr[:, :, 0]
                elif arr.shape[0] == 1:
                    arr = arr[0, :, :]
                else:
                    raise ValueError(
                        f"{field_name}: expected 2D field or singleton 3D field, got shape {arr.shape}"
                    )

            if arr.ndim != 2:
                raise ValueError(f"{field_name}: expected 2D array, got shape {arr.shape}")

            nx, ny = arr.shape

            # ECSIM-compatible on-disk layout: (z, y, x)
            out = np.zeros((2, ny + 1, nx + 1), dtype=arr.dtype)

            # arr is in (x, y); store as (y, x) in plane 0
            out[0, :ny, :nx] = arr.T

            # Duplicate into second plane for compatibility
            out[1, :ny, :nx] = arr.T

            field_group = block_group.create_group(field_name)
            field_group.create_dataset(
                "0",
                data=out,
                compression=compression,
                compression_opts=compression_opts,
            )


def main():
    parser = argparse.ArgumentParser(description="MPI iPiC3D -> ECSIM converter for one cycle.")
    parser.add_argument("--input_folder", required=True, help="Folder containing proc*.hdf")
    parser.add_argument("--output_folder", required=True, help="Folder to write output HDF5")
    parser.add_argument("--cycles", default=None,
                    help="Comma-separated cycle list, e.g. 19100,19200. Default: all available cycles")
    parser.add_argument("--species", default=None, help="Comma-separated species list, e.g. 0,1,2,3")
    parser.add_argument("--simulation_name", default="iPIC3D", help="Output filename prefix")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    os.makedirs(args.output_folder, exist_ok=True)
    all_hdf_files = get_all_proc_files(args.input_folder)
    sample_file = all_hdf_files[0]
    species_list = parse_species_arg(args.species, sample_file)
    cycles = parse_cycles_arg(args.cycles, sample_file)

    if rank == 0:
        print(f"Detected {len(all_hdf_files)} proc*.hdf files")
        print(f"Selected species: {species_list}")
        print(f"Cycles to convert: {cycles}")

    for cycle in cycles:
        time_cycle = f"cycle_{cycle}"

        if rank == 0:
            max_coords = np.array([0, 0, 0], dtype=int)
            for fp in all_hdf_files:
                with h5py.File(fp, "r") as f:
                    cc = np.array(f["topology/cartesian_coord"][()], dtype=int)
                    max_coords = np.maximum(max_coords, cc)
            print("Max cartesian coords:", max_coords)

        if rank == 0:
            print(f"\nProcessing {time_cycle}")

        requests = build_requests(sample_file, species_list, time_cycle)

        if not requests:
            if rank == 0:
                print(f"No readable fields found for {time_cycle}, skipping")
            continue

        if rank == 0:
            print(f"Will convert {len(requests)} fields:")
            for req in requests:
                print(f"  {req['output_name']} <- {req['raw_path_prefix']}/{time_cycle}")

        
        layout = infer_global_layout(args.input_folder, all_hdf_files)

        nx_global = layout["nx_global"]
        ny_global = layout["ny_global"]
        nz_global = layout["nz_global"]

        if rank == 0:
            print(f"Global shape inferred: ({nz_global}, {ny_global}, {nx_global})")
        
        global_fields = assemble_fields_mpi(
            all_hdf_files=all_hdf_files,
            requests=requests,
            time_cycle=time_cycle,
            nx_global=layout["nx_global"],
            ny_global=layout["ny_global"],
            xlen=layout["xlen"],
            ylen=layout["ylen"],
        )

        if rank == 0:
            time_label = f"{cycle:06d}"
            out_filename = f"{args.simulation_name}-Fields_{time_label}.h5"
            out_path = os.path.join(args.output_folder, out_filename)

            if os.path.exists(out_path) and not args.overwrite:
                print(f"Skipping existing file: {out_path}")
                continue

            write_ecsim_h5(out_path, global_fields)

            print(f"Wrote: {out_path}")
            print(f"Number of fields written: {len(global_fields)}")

    if rank == 0:
        print("Complete .....", "Time Elapsed =", datetime.now() - startTime)

if __name__ == "__main__":
    main()