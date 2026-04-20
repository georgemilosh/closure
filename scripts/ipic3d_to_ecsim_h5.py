#!/usr/bin/env python3
"""
Convert iPiC3D HDF5 output (proc*.hdf) to ECSIM-compatible HDF5 files.

Usage:
  python ipic3d_to_ecsim_h5.py --input_folder /path/to/ipic3d --output_folder /path/to/ecsim

Optional overrides:
  --cycles 0,10,20
  --species 0,1
  --simulation_name T2D
  --time_digits 6
  --overwrite
  --verbose
  --set fields_to_read.B=False
  --set fields_to_read.divB=True
"""

import argparse
import glob
import logging
import os
import sys

from closure import read_pic as rp
from closure.utilities import set_nested_config


def _parse_csv_ints(value):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv_strings(value):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Convert iPiC3D HDF5 output to ECSIM-compatible HDF5 files.")
    parser.add_argument("--input_folder", required=True, help="Folder containing proc*.hdf files.")
    parser.add_argument("--output_folder", required=True, help="Folder to write ECSIM HDF5 files.")
    parser.add_argument("--cycles", default=None, help="Comma-separated list of cycles (default: all).")
    parser.add_argument("--species", default=None, help="Comma-separated species suffixes (default: auto-detect).")
    parser.add_argument("--simulation_name", default="iPIC3D", help="Output filename prefix.")
    parser.add_argument("--time_digits", type=int, default=6, help="Zero-padding width for cycle labels.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override kwargs using dotted keys, e.g. --set fields_to_read.B=False",
    )

    args = parser.parse_args()

    kwargs = {
        "simulation_name": args.simulation_name,
        "time_digits": args.time_digits,
        "overwrite": args.overwrite,
        "verbose": args.verbose,
    }

    cycles = _parse_csv_ints(args.cycles)
    species = _parse_csv_strings(args.species)
    if cycles is not None:
        kwargs["cycles"] = cycles
    if species is not None:
        kwargs["choose_species"] = species

    # initialize logger for this CLI wrapper
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # Detect available fields in a sample iPiC3D HDF5 file and prune fields_to_read
    try:
        sample_files = sorted(glob.glob(os.path.join(args.input_folder, "proc*.hdf")))
        if sample_files:
            import h5py

            sample = sample_files[0]
            with h5py.File(sample, "r") as f:
                # start from provided fields_to_read or defaults
                fields_to_read = kwargs.get("fields_to_read") or rp._default_ipic3d_fields_to_read()

                # Inspect available 'fields' group
                if "fields" in f:
                    available_field_keys = [k for k in f["fields"].keys()]
                    if not any(k.lower().startswith("bx") for k in available_field_keys):
                        fields_to_read["B"] = False
                    if not any(k.lower().startswith("ex") for k in available_field_keys):
                        fields_to_read["E"] = False
                    if not any(k.lower() == "divb" for k in available_field_keys):
                        fields_to_read["divB"] = False
                else:
                    for k in ["B", "E", "divB", "B_ext", "E_ext", "EF"]:
                        fields_to_read[k] = False

                # Inspect available 'moments' group for species-specific data
                if "moments" not in f:
                    for k in ["rho", "J", "P", "PI", "N", "Qrem", "Heat_flux"]:
                        fields_to_read[k] = False
                else:
                    species_groups = [k for k in f["moments"].keys() if k.startswith("species_")]
                    if not species_groups:
                        for k in ["rho", "J", "P", "PI"]:
                            fields_to_read[k] = False
                    else:
                        sample_species = species_groups[0]
                        species_keys = [k for k in f["moments"][sample_species].keys()]
                        if "rho" not in species_keys:
                            fields_to_read["rho"] = False
                        if not any(k.lower().startswith("jx") for k in species_keys):
                            fields_to_read["J"] = False
                        if not any(k.startswith("P") for k in species_keys):
                            fields_to_read["P"] = False
                            fields_to_read["PI"] = False

                kwargs["fields_to_read"] = fields_to_read
                if args.verbose:
                    logger.info(f"Using fields_to_read after detection: {fields_to_read}")
    except Exception as e:
        # Non-fatal: log and continue with defaults
        logging.getLogger(__name__).warning(f"Field detection failed, proceeding with defaults: {e}")

    for override in args.set:
        if "=" not in override:
            raise ValueError(f"Invalid --set value: {override}. Expected key=value.")
        key, value = override.split("=", 1)
        set_nested_config(kwargs, key, value)

    rp.convert_ipic3d_to_ecsim_h5(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        **kwargs,
    )


if __name__ == "__main__":
    main()
