"""
Generate field images from experiment data with configurable options.

Usage:
    python compute_field_images.py <experiment_name> [options]
    
    Options:
        --files_path <path>          Path to experiment files (default: /volume1/scratch/share_dir/ecsim/Harris/)
        --fields <list>              Fields to plot, comma-separated (default: Jz-tot)
        --field_max <float>          Maximum field value for color scale (default: None, auto-scale)
        --choose_species <list>      Species to analyze, comma-separated (default: e,i)
        --choose_times <int>         Time index to load: single int (load all up to that), comma-separated list (load specific), or None for all (default: 1)
        --dpi <int>                  DPI for saved images (default: 150)
        --verbose                    Enable verbose output (default: False)
        --cmap <str>                 Colormap to use (default: auto, seismic for signed, viridis for unsigned)
        --choose_x <list>            X range to select, comma-separated (default: None)
        --choose_y <list>            Y range to select, comma-separated (default: None)
        --gif                        Also save a GIF animation for each field (default: False)

Example:
    # Basic usage
    python compute_field_images.py Le0

    # Custom path and fields
    python compute_field_images.py Le0 --files_path /custom/path --fields "Jz-tot,Ex"

    # Set field maximum and DPI
    python compute_field_images.py Le0 --fields Jz_e --field_max 0.5 --dpi 200

    # Multiple fields with verbose output
    python compute_field_images.py Le0 --fields "Jz-tot,Jx-tot,Ez" --verbose

    # Custom species and time selection
    python compute_field_images.py Le0 --choose_species "e1,i1" --choose_times "0,10,20,30"

    # Save GIFs in addition to PNG frames
    python compute_field_images.py Le0 --fields "Jz-tot,Ex" --gif
"""

import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import src.trainers as tr
import src.read_pic as rp
import os
import numpy as np
import shutil
import matplotlib.animation as animation
import src.utilities as ut


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate field images from experiment data with configurable options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required argument
    parser.add_argument(
        'experiment',
        type=str,
        help='Name of the experiment to analyze'
    )
    
    # Optional arguments
    parser.add_argument(
        '--files_path',
        type=str,
        default="/volume1/scratch/share_dir/ecsim/Harris/",
        help='Path to experiment files (default: /volume1/scratch/share_dir/ecsim/Harris/)'
    )
    
    parser.add_argument(
        '--fields',
        type=str,
        default='Jz-tot',
        help='Fields to plot, comma-separated. Can include species suffix like Jz_e (default: Jz-tot)'
    )
    
    parser.add_argument(
        '--field_max',
        type=float,
        default=None,
        help='Maximum field value for color scale (default: None, auto-scale)'
    )
    
    parser.add_argument(
        '--choose_species',
        type=str,
        default='e,i',
        help='Species to analyze, comma-separated (default: e,i)'
    )
    
    parser.add_argument(
        '--choose_times',
        type=str,
        default='1',
        help='Time indices to load: single int (load all up to that), comma-separated list (load specific), or None for all (default: 1)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved images (default: 150)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output (default: False)'
    )

    parser.add_argument(
        '--gif',
        action='store_true',
        help='Save GIF animations for each field (default: False)'
    )
    
    parser.add_argument(
        '--cmap',
        type=str,
        default='auto',
        help='Colormap to use (default: auto, seismic for signed fields, viridis for unsigned)'
    )
    
    parser.add_argument(
        '--choose_x',
        type=str,
        default=None,
        help='X range to select, comma-separated (default: None)'
    )
    
    parser.add_argument(
        '--choose_y',
        type=str,
        default=None,
        help='Y range to select, comma-separated (default: None)'
    )
    
    return parser.parse_args()


def parse_list_arg(arg_str, dtype=str):
    """Parse comma-separated list argument."""
    return [dtype(x.strip()) for x in arg_str.split(',')]


def parse_range_arg(arg_str):
    """Parse comma-separated range argument."""
    return [int(x.strip()) for x in arg_str.split(',')]


def main():
    """Main execution function."""
    args = parse_args()
    
    # Parse arguments
    choose_species = parse_list_arg(args.choose_species, dtype=str)
    choose_x = parse_range_arg(args.choose_x) if args.choose_x is not None else None
    choose_y = parse_range_arg(args.choose_y) if args.choose_y is not None else None
    
    # Parse fields and extract species suffixes
    plot_fields = parse_list_arg(args.fields, dtype=str)
    fields_list = []
    species_list = []
    for field in plot_fields:
        if '_' in field:
            parsed_field, species = field.rsplit('_', 1)
            fields_list.append(parsed_field)
            species_list.append(species)
        else:
            fields_list.append(field)
            species_list.append(None)
    
    # Parse choose_times: can be a single int or comma-separated list
    if args.choose_times is not None and args.choose_times.lower() != 'none':
        if ',' in args.choose_times:
            choose_times = parse_list_arg(args.choose_times, dtype=int)
        else:
            choose_times = int(args.choose_times)
    else:
        choose_times = None
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Experiment: {args.experiment}")
        print(f"  Files path: {args.files_path}")
        print(f"  Fields: {fields_list}")
        print(f"  Species: {species_list}")
        print(f"  Field max: {args.field_max}")
        print(f"  Choose species: {choose_species}")
        print(f"  Choose times: {choose_times}")
        print(f"  DPI: {args.dpi}")
        print(f"  Colormap: {args.cmap}")
        print(f"  X range: {choose_x}")
        print(f"  Y range: {choose_y}")
        print(f"  GIF: {args.gif}")
        print()
    
    # Fields to read
    fields_to_read = {
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
        "Qrem": False
    }
    
    # Load data
    if args.verbose:
        print(f"Loading data for experiment: {args.experiment}")
    
    data, X, Y, qom, times = rp.get_exp_times(
        [args.experiment],
        args.files_path,
        fields_to_read,
        choose_species=choose_species,
        choose_times=choose_times,
        choose_x=choose_x,
        choose_y=choose_y,
        verbose=args.verbose
    )
    
    data = data[args.experiment]
    
    if args.verbose:
        print(f"Data loaded successfully.")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")
        print(f"  Number of time steps: {len(times)}")
        print()
    
    # Compute additional fields
    ut.get_Ohm(data, qom, X[:,0], Y[0,:])
    
    data['Jz-tot'] = data['Jz']['e'] + data['Jz']['i']
    data['Jx-tot'] = data['Jx']['e'] + data['Jx']['i']
    data['Jy-tot'] = data['Jy']['e'] + data['Jy']['i']
    
    # Process each field
    for plot_field, species in zip(fields_list, species_list):
        if args.verbose:
            print(f"Processing field: {plot_field}, species: {species}")
        
        # Create output directory for frames (single folder per field)
        frames_dir = f'{args.files_path}/{args.experiment}/plots/{args.experiment}_frames/{plot_field}'
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Determine colormap
        if args.cmap == 'auto':
            if plot_field in ['rho', 'Pxx', 'Pyy', 'Pzz']:
                cmap = 'viridis'
            else:
                cmap = 'seismic'
        else:
            cmap = args.cmap
        
        # Initialize the plot with the first frame
        if species is None:
            try:
                shape2 = data[plot_field].shape[2]
            except KeyError:
                print(f"Field {plot_field} not found in data.")
                print(f"Available fields: {list(data.keys())}")
                continue
            
            finite_data = data[plot_field][np.isfinite(data[plot_field])]
            
            # Determine field min/max for color scale
            if args.field_max is None:
                field_min = np.nanmin(finite_data) / 4
                field_max = np.nanmax(finite_data) / 4
            else:
                field_max = args.field_max
                field_min = -args.field_max
            vlimit = max(-field_min, field_max)
            
            if args.verbose:
                print(f"  Field shape: {data[plot_field].shape}")
                print(f"  Field min: {field_min:.4e}, max: {field_max:.4e}")
                print(f"  Color limit: {vlimit:.4e}")
            
            # Loop through frames and save each as PNG
            for frame in range(shape2):
                ax.clear()
                if cmap == 'seismic':
                    cax = ax.pcolormesh(X, Y, data[plot_field][:, :, frame], vmin=-vlimit, vmax=vlimit, cmap=cmap)
                else:
                    cax = ax.pcolormesh(X, Y, np.abs(data[plot_field][:, :, frame]), cmap=cmap, vmin=0, vmax=vlimit)
                fig.colorbar(cax)
                ax.set_title(f'{plot_field}, run {args.experiment}, time = {times[frame]:.2f}' + r"$\Omega_{ci}^{-1}$")
                
                # Save the frame
                frame_path = os.path.join(frames_dir, f'frame_{frame:04d}.png')
                fig.savefig(frame_path, dpi=args.dpi, bbox_inches='tight')
                fig.clf()
                ax = fig.add_subplot(111)
            
            if args.verbose:
                print(f"  Saved {shape2} frames to {frames_dir}")

            if args.gif:
                gif_fig, gif_ax = plt.subplots(figsize=(6, 5))
                initial_frame = data[plot_field][:, :, 0]
                if cmap == 'seismic':
                    gif_cax = gif_ax.pcolormesh(X, Y, initial_frame, vmin=-vlimit, vmax=vlimit, cmap=cmap)
                else:
                    gif_cax = gif_ax.pcolormesh(X, Y, np.abs(initial_frame), cmap=cmap, vmin=0, vmax=vlimit)
                gif_fig.colorbar(gif_cax)
                gif_ax.set_title(f'{plot_field}, run {args.experiment}, time = {times[0]:.2f}' + r"$\Omega_{ci}^{-1}$")

                def update(frame):
                    frame_data = data[plot_field][:, :, frame]
                    if cmap != 'seismic':
                        frame_data = np.abs(frame_data)
                    gif_cax.set_array(frame_data.ravel())
                    gif_ax.set_title(f'{plot_field}, run {args.experiment}, time = {times[frame]:.2f}' + r"$\Omega_{ci}^{-1}$")
                    return gif_cax,

                gif_fig.set_tight_layout(True)
                ani = animation.FuncAnimation(gif_fig, update, frames=shape2, blit=True)
                gif_path = os.path.join(args.files_path, args.experiment, 'plots', f'{plot_field}_{args.experiment}_movie.gif')
                ani.save(gif_path, dpi=args.dpi)
                plt.close(gif_fig)
                if args.verbose:
                    print(f"  Saved GIF to {gif_path}")
        else:
            shape2 = data[plot_field][species].shape[2]
            finite_data = data[plot_field][species][np.isfinite(data[plot_field][species])]
            
            # Determine field min/max for color scale
            if args.field_max is None:
                field_min = np.nanmin(finite_data) / 4
                field_max = np.nanmax(finite_data) / 4
            else:
                field_max = args.field_max
                field_min = -args.field_max
            vlimit = max(-field_min, field_max)
            
            if args.verbose:
                print(f"  Field shape: {data[plot_field][species].shape}")
                print(f"  Field min: {field_min:.4e}, max: {field_max:.4e}")
                print(f"  Color limit: {vlimit:.4e}")
            
            # Loop through frames and save each as PNG
            for frame in range(shape2):
                ax.clear()
                if cmap == 'seismic':
                    cax = ax.pcolormesh(X, Y, data[plot_field][species][:, :, frame], vmin=-vlimit, vmax=vlimit, cmap=cmap)
                else:
                    cax = ax.pcolormesh(X, Y, np.abs(data[plot_field][species][:, :, frame]), cmap=cmap, vmin=0, vmax=vlimit)
                fig.colorbar(cax)
                ax.set_title(f'{plot_field}, {species}, run {args.experiment}, time = {times[frame]:.2f}' + r"$\Omega_{ci}^{-1}$")
                
                # Save the frame
                frame_path = os.path.join(frames_dir, f'{species}_frame_{frame:04d}.png')
                fig.savefig(frame_path, dpi=args.dpi, bbox_inches='tight')
                fig.clf()
                ax = fig.add_subplot(111)
            
            if args.verbose:
                print(f"  Saved {shape2} frames to {frames_dir}")

            if args.gif:
                gif_fig, gif_ax = plt.subplots(figsize=(6, 5))
                initial_frame = data[plot_field][species][:, :, 0]
                if cmap == 'seismic':
                    gif_cax = gif_ax.pcolormesh(X, Y, initial_frame, vmin=-vlimit, vmax=vlimit, cmap=cmap)
                else:
                    gif_cax = gif_ax.pcolormesh(X, Y, np.abs(initial_frame), cmap=cmap, vmin=0, vmax=vlimit)
                gif_fig.colorbar(gif_cax)
                gif_ax.set_title(f'{plot_field}, {species}, run {args.experiment}, time = {times[0]:.2f}' + r"$\Omega_{ci}^{-1}$")

                def update(frame):
                    frame_data = data[plot_field][species][:, :, frame]
                    if cmap != 'seismic':
                        frame_data = np.abs(frame_data)
                    gif_cax.set_array(frame_data.ravel())
                    gif_ax.set_title(f'{plot_field}, {species}, run {args.experiment}, time = {times[frame]:.2f}' + r"$\Omega_{ci}^{-1}$")
                    return gif_cax,

                gif_fig.set_tight_layout(True)
                ani = animation.FuncAnimation(gif_fig, update, frames=shape2, blit=True)
                gif_path = os.path.join(args.files_path, args.experiment, 'plots', f'{plot_field}_{species}_{args.experiment}_movie.gif')
                ani.save(gif_path, dpi=args.dpi)
                plt.close(gif_fig)
                if args.verbose:
                    print(f"  Saved GIF to {gif_path}")
        
        plt.close(fig)
    
    if args.verbose:
        print("\nDone!")


if __name__ == '__main__':
    main()
