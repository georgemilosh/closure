"""Generate field images from experiment data.

The default input root comes from ``paths.yaml`` via ``closure.config.load_paths``.
Use ``--files_path`` to override it for a specific run.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from closure import plasma, read_pic as rp
from closure.config import load_paths


DEFAULT_FIELDS_TO_READ = {
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
    parser = argparse.ArgumentParser(description="Generate field images from experiment data")
    parser.add_argument("experiment", type=str, help="Name of the experiment to analyze")
    parser.add_argument("--files_path", type=str, default=None, help="Root directory containing experiment folders")
    parser.add_argument("--fields", type=str, default="Jz-tot", help="Comma-separated field names to plot")
    parser.add_argument("--field_max", type=float, default=None, help="Maximum absolute color value")
    parser.add_argument("--choose_species", type=str, default="e,i", help="Comma-separated species labels")
    parser.add_argument("--choose_times", type=str, default="1", help="Single int, comma list, or 'None'")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved images")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--gif", action="store_true", help="Also save GIF animations")
    parser.add_argument("--cmap", type=str, default="auto", help="Colormap name or 'auto'")
    parser.add_argument("--choose_x", type=str, default=None, help="X index range as 'start,end'")
    parser.add_argument("--choose_y", type=str, default=None, help="Y index range as 'start,end'")
    return parser.parse_args()


def parse_list_arg(arg_str: str, dtype=str) -> list:
    return [dtype(x.strip()) for x in arg_str.split(",")]


def parse_range_arg(arg_str: str | None) -> list[int] | None:
    if arg_str is None:
        return None
    return [int(x.strip()) for x in arg_str.split(",")]


def resolve_files_path(override: str | None) -> str:
    return override or load_paths().get("data_dir", "./data")


def parse_choose_times(value: str | None) -> int | list[int] | None:
    if value is None or value.lower() == "none":
        return None
    if "," in value:
        return parse_list_arg(value, dtype=int)
    return int(value)


def build_requested_fields(plot_fields: list[str]) -> tuple[list[str], list[str | None]]:
    fields_list = []
    species_list = []
    for field in plot_fields:
        if "_" in field:
            parsed_field, species = field.rsplit("_", 1)
            fields_list.append(parsed_field)
            species_list.append(species)
        else:
            fields_list.append(field)
            species_list.append(None)
    return fields_list, species_list


def main() -> None:
    args = parse_args()
    files_path = resolve_files_path(args.files_path)
    choose_species = parse_list_arg(args.choose_species, dtype=str)
    choose_times = parse_choose_times(args.choose_times)
    choose_x = parse_range_arg(args.choose_x)
    choose_y = parse_range_arg(args.choose_y)
    plot_fields = parse_list_arg(args.fields, dtype=str)
    fields_list, species_list = build_requested_fields(plot_fields)

    data, x, y, qom, times = rp.get_exp_times(
        [args.experiment],
        files_path,
        DEFAULT_FIELDS_TO_READ,
        choose_species=choose_species,
        choose_times=choose_times,
        choose_x=choose_x,
        choose_y=choose_y,
        verbose=args.verbose,
    )
    data = data[args.experiment]

    plasma.get_Ohm(data, qom, x[:, 0], y[0, :])
    if "e" in data["Jz"] and "i" in data["Jz"]:
        data["Jz-tot"] = data["Jz"]["e"] + data["Jz"]["i"]
        data["Jx-tot"] = data["Jx"]["e"] + data["Jx"]["i"]
        data["Jy-tot"] = data["Jy"]["e"] + data["Jy"]["i"]

    for plot_field, species in zip(fields_list, species_list):
        frames_dir = Path(files_path) / args.experiment / "plots" / f"{args.experiment}_frames" / plot_field
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = "viridis" if args.cmap == "auto" and plot_field in ["rho", "Pxx", "Pyy", "Pzz"] else args.cmap
        if cmap == "auto":
            cmap = "seismic"

        field_data = data[plot_field] if species is None else data[plot_field][species]
        finite_data = field_data[np.isfinite(field_data)]
        if finite_data.size == 0:
            plt.close(fig)
            continue

        if args.field_max is None:
            field_min = np.nanmin(finite_data) / 4
            field_max = np.nanmax(finite_data) / 4
        else:
            field_max = args.field_max
            field_min = -args.field_max
        vlimit = max(-field_min, field_max)
        n_frames = field_data.shape[2]

        for frame in range(n_frames):
            ax.clear()
            plotted = field_data[:, :, frame]
            if cmap == "seismic":
                cax = ax.pcolormesh(x, y, plotted, vmin=-vlimit, vmax=vlimit, cmap=cmap)
            else:
                cax = ax.pcolormesh(x, y, np.abs(plotted), cmap=cmap, vmin=0, vmax=vlimit)
            fig.colorbar(cax)
            title_prefix = plot_field if species is None else f"{plot_field}, {species}"
            ax.set_title(f"{title_prefix}, run {args.experiment}, time = {times[frame]:.2f}" + r"$\Omega_{ci}^{-1}$")
            frame_name = f"frame_{frame:04d}.png" if species is None else f"{species}_frame_{frame:04d}.png"
            fig.savefig(frames_dir / frame_name, dpi=args.dpi, bbox_inches="tight")
            fig.clf()
            ax = fig.add_subplot(111)

        if args.gif:
            gif_fig, gif_ax = plt.subplots(figsize=(6, 5))
            initial_frame = field_data[:, :, 0]
            if cmap == "seismic":
                gif_cax = gif_ax.pcolormesh(x, y, initial_frame, vmin=-vlimit, vmax=vlimit, cmap=cmap)
            else:
                gif_cax = gif_ax.pcolormesh(x, y, np.abs(initial_frame), cmap=cmap, vmin=0, vmax=vlimit)
            gif_fig.colorbar(gif_cax)

            def update(frame: int):
                frame_data = field_data[:, :, frame]
                if cmap != "seismic":
                    frame_data = np.abs(frame_data)
                gif_cax.set_array(frame_data.ravel())
                return (gif_cax,)

            gif_path = Path(files_path) / args.experiment / "plots" / f"{plot_field}_{args.experiment}_movie.gif"
            animation.FuncAnimation(gif_fig, update, frames=n_frames, blit=True).save(gif_path, dpi=args.dpi)
            plt.close(gif_fig)

        plt.close(fig)


if __name__ == "__main__":
    main()
