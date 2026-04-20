"""Compute filtered flux diagnostics for one experiment."""

from __future__ import annotations

import argparse
import pickle
import time
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
    parser = argparse.ArgumentParser(description="Compute filtered flux diagnostics")
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
    start_time = time.time()
    args = parse_args()
    files_path = resolve_files_path(args.files_path)
    choose_species = parse_list_arg(args.choose_species, dtype=str)
    choose_times = parse_choose_times(args.choose_times)

    data, x, y, qom, _times = rp.get_exp_times(
        [args.experiment],
        files_path,
        FIELDS_TO_READ,
        choose_species=choose_species,
        verbose=args.verbose,
        choose_times=choose_times,
        indexing="ij",
    )
    experiment_data = data[args.experiment]
    filtered: dict[str, object] = {}

    if x.shape[0] == 2048:
        xs = [2048, 1376, 928, 608, 416, 288, 192, 128, 96, 64, 32, 20, 16, 12, 10, 8, 6, 4, 3, 2, 1]
    elif x.shape[0] == 512:
        xs = [512, 352, 256, 176, 128, 88, 64, 44, 32, 22, 16, 11, 8, 6, 4, 3, 2, 1]
    else:
        raise ValueError(f"Shape {x.shape[0]} treatment not implemented")

    filtered["xs"] = xs
    for quantity in ["PIuu", "PIbb", "Ef_favre", "PS", "-Ptheta", "JdotE"]:
        filtered[quantity] = {"e": [], "i": []}
    for quantity in ["E2_bar", "B2_bar"]:
        filtered[quantity] = []

    for xi in xs:
        plasma.scale_filtering(
            experiment_data,
            x[:, 0],
            y[0, :],
            qom,
            verbose=False,
            filters={"name": "uniform_filter", "size": xi, "mode": "wrap", "axes": (0, 1)},
        )
        filtered["E2_bar"].append(np.mean(experiment_data["E2_bar"], axis=(0, 1)))
        filtered["B2_bar"].append(np.mean(experiment_data["B2_bar"], axis=(0, 1)))
        for quantity in ["PIuu", "PIbb", "Ef_favre", "PS", "-Ptheta", "JdotE"]:
            for species in ["e", "i"]:
                filtered[quantity][species].append(np.mean(experiment_data[quantity][species], axis=(0, 1)))

    for quantity in ["PIuu", "PIbb", "Ef_favre", "PS", "-Ptheta", "JdotE"]:
        for species in ["e", "i"]:
            filtered[quantity][species] = np.array(filtered[quantity][species])
    for quantity in ["E2_bar", "B2_bar"]:
        filtered[quantity] = np.array(filtered[quantity])

    experiment_data["P"] = {}
    for species in ["e", "i"]:
        experiment_data["P"][species] = (
            experiment_data["Pxx"][species] + experiment_data["Pyy"][species] + experiment_data["Pzz"][species]
        ) / 3
    filtered["Ethi_i"] = 3 * np.mean(experiment_data["P"]["i"], axis=(0, 1)) / 2
    filtered["Ethi_e"] = 3 * np.mean(experiment_data["P"]["e"], axis=(0, 1)) / 2

    output_path = Path(args.output) if args.output else Path(files_path) / args.experiment / "filtered_quantities.pkl"
    with open(output_path, "wb") as stream:
        pickle.dump(filtered, stream)

    if args.verbose:
        print(f"Execution time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
