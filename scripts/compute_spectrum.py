"""Compute magnetic, velocity, and Ohm-term spectra for one experiment."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from closure import plasma, read_pic as rp
from closure.config import load_paths


FIELDS_TO_READ = {
    "B": True,
    "B_ext": False,
    "divB": True,
    "E": True,
    "E_ext": False,
    "rho": True,
    "J": True,
    "P": True,
    "PI": False,
    "Heat_flux": False,
    "N": False,
    "Qrem": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute spectra for a single experiment")
    parser.add_argument("experiment", type=str, help="Experiment folder name")
    parser.add_argument("--files_path", type=str, default=None, help="Root data directory")
    parser.add_argument("--choose_species", type=str, default="e,i", help="Comma-separated species labels")
    parser.add_argument("--choose_times", type=str, default="1", help="Single int, comma list, or 'None'")
    parser.add_argument("--output", type=str, default=None, help="Optional output pickle path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def parse_list_arg(value: str, dtype=str) -> list:
    return [dtype(item.strip()) for item in value.split(",")]


def parse_choose_times(value: str | None) -> int | list[int] | None:
    if value is None or value.lower() == "none":
        return None
    if "," in value:
        return parse_list_arg(value, dtype=int)
    return int(value)


def resolve_files_path(override: str | None) -> str:
    return override or load_paths().get("data_dir", "./data")


def main() -> None:
    args = parse_args()
    files_path = resolve_files_path(args.files_path)
    choose_species = parse_list_arg(args.choose_species, dtype=str)
    choose_times = parse_choose_times(args.choose_times)

    data, x, y, qom, times = rp.get_exp_times(
        [args.experiment],
        files_path,
        FIELDS_TO_READ,
        choose_species=choose_species,
        verbose=args.verbose,
        choose_times=choose_times,
        indexing="ij",
    )
    data = data[args.experiment]

    plasma.get_PS_2D_field(data, x[:, 0], y[0, :])
    plasma.get_Ohm(data, qom, x[:, 0], y[0, :])

    jx = np.sum([data["Jx"][species] for species in data["Jx"].keys()], axis=0)
    jy = np.sum([data["Jy"][species] for species in data["Jy"].keys()], axis=0)
    jz = np.sum([data["Jz"][species] for species in data["Jz"].keys()], axis=0)
    current = np.sqrt(np.mean(jx**2 + jy**2 + jz**2, axis=(0, 1)))
    ion_current = np.sqrt(np.mean(data["Jx"]["i"]**2 + data["Jy"]["i"]**2 + data["Jz"]["i"]**2, axis=(0, 1)))

    b_spectrum = []
    v_spectrum = []
    for iteri, time in enumerate(times):
        if args.verbose:
            print(f"computing spectra at time={time}")
        b_field = np.array([data["Bx"], data["By"], data["Bz"]])[..., iteri]
        v_field = np.array([data["Vx"]["i"], data["Vy"]["i"], data["Vz"]["i"]])[..., iteri]
        ky, b_spec = plasma.vector_spectrum_2D(b_field[0], b_field[1], b_field[2], x, y)
        _, v_spec = plasma.vector_spectrum_2D(v_field[0], v_field[1], v_field[2], x, y)
        b_spectrum.append(b_spec)
        v_spectrum.append(v_spec)
    b_spectrum = np.array(b_spectrum)
    v_spectrum = np.array(v_spectrum)

    for quantity_name in ["E", "EMHD", "EHall", "EP"]:
        ky2, spec_ohms = plasma.vector_spectrum_2D(
            data[f"{quantity_name}x"],
            data[f"{quantity_name}y"],
            data[f"{quantity_name}z"],
            x,
            y,
        )

    imin = np.argmax(current) - 4
    imax = np.argmax(current) + 1
    thresholds = {}
    mean_values = {}
    percentiles = {}
    thresholds2 = {}
    mean_values2 = {}
    percentiles2 = {}

    for species in ["i", "e"]:
        thresholds[species] = {}
        mean_values[species] = {}
        percentiles[species] = {}
        thresholds2[species] = {}
        mean_values2[species] = {}
        percentiles2[species] = {}
        quantity_name = "PiD"
        condition_names = ["Qomega", "QD", "QJ"]
        for condition_name in condition_names:
            condition = data[condition_name][..., imin:imax] if condition_name == "QJ" else data[condition_name][species][..., imin:imax]
            condition_max = 8
            quantity = data[quantity_name][species][..., imin:imax]
            thresholds[species][condition_name] = np.arange(0, condition_max, condition_max / 20)
            mean_values[species][condition_name] = [np.mean(quantity[condition > a]) for a in thresholds[species][condition_name]]
            percentiles[species][condition_name] = [np.mean(condition > a) for a in thresholds[species][condition_name]]

        for quantity_name in ["PiD", "J*(E+VxB)"]:
            if quantity_name == "J*(E+VxB)":
                condition_name = "QJ"
                condition = np.sqrt(data[condition_name][..., imin:imax])
            else:
                condition_name = "QD"
                condition = np.sqrt(data[condition_name][species])[..., imin:imax]
            quantity = data[quantity_name][species][..., imin:imax] / np.sqrt(np.mean(data[quantity_name][species][..., imin:imax] ** 2))
            thresholds2[species][quantity_name] = np.arange(0, np.max(condition), np.max(condition) / 100)
            mean_values2[species][quantity_name] = np.array([
                np.mean(quantity[(condition > a) & (condition < (a + np.max(condition) / 100))])
                for a in thresholds2[species][quantity_name]
            ])
            percentiles2[species][quantity_name] = np.array([np.mean(condition > a) for a in thresholds2[species][quantity_name]])
            mask = ~np.isnan(mean_values2[species][quantity_name])
            thresholds2[species][quantity_name] = thresholds2[species][quantity_name][mask]
            percentiles2[species][quantity_name] = percentiles2[species][quantity_name][mask]
            mean_values2[species][quantity_name] = mean_values2[species][quantity_name][mask]

    output_data = {
        "times": times,
        "J": current,
        "Ji": ion_current,
        "ky": ky,
        "B_spectrum": b_spectrum,
        "V_spectrum": v_spectrum,
        "ky2": ky2,
        "spec_Ohms": spec_ohms,
        "thresholds": thresholds,
        "mean_values": mean_values,
        "percentiles": percentiles,
        "thresholds2": thresholds2,
        "mean_values2": mean_values2,
        "percentiles2": percentiles2,
    }

    output_path = Path(args.output) if args.output else Path(files_path) / args.experiment / "spectra.pkl"
    with open(output_path, "wb") as stream:
        pickle.dump(output_data, stream)


if __name__ == "__main__":
    main()
