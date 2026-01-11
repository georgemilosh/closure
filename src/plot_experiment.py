"""
Plot experiment data with configurable options.

Usage:
    python plot_experiment.py experiment <experiment_name> [options]
    
    Options:
        --files_path <path>          Path to experiment files (default: ./)
        --choose_species <list>      Species to analyze, comma-separated (default: e,i,e,i)
        --choose_times <int>         Time index to plot (default: None)
        --iteration <int> or -i      Iteration index to plot (default: -1, last)
        --species <str> or -s        Species to plot: e or i (default: e)
        --verbose                    Enable verbose output (default: False)
        --fields <list>              Fields to plot, comma-separated (default: Ez,Jx,Jz,Bx,By,rho)
        --output <path>              Save figure to file (default: None, display only)

Example:
    python plot_experiment.py Finelli1 --files_path /path/to/data --iteration 4 --species e

    # Basic usage
    python plot_experiment.py Finelli1

    # Plot iteration 4, species 'i', custom path
    python plot_experiment.py Finelli1 --iteration 4 --species i --files_path /custom/path

    # Plot specific fields with verbose output
    python plot_experiment.py Finelli1 -i 5 -s e --fields "Ez,Bx" --verbose

    # Save to file
    python plot_experiment.py Finelli1 --output plot.png
    
    # Custom species list
    python plot_experiment.py Finelli1 --choose_species "e1,i1,e2,i2"
"""

import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("Agg")
matplotlib.use("TkAgg")   # or "QtAgg" if your Qt stack works

sys.path.append('/dodrio/scratch/projects/2025_065/georgem/2024_109/closure/')
import src.read_pic as rp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot experiment data with configurable options",
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
        default="./",
        help='Path to experiment files (default: ./)'
    )
    
    parser.add_argument(
        '--choose_species',
        type=str,
        default='e,i,e,i',
        help='Species to analyze, comma-separated (default: e,i,e,i)'
    )
    
    parser.add_argument(
        '--choose_times',
        type=str,
        default=None,
        help='Time indices to load: single int (load all up to that), comma-separated list (load specific), or None for all (default: None)'
    )
    
    parser.add_argument(
        '-i', '--iteration',
        type=int,
        default=-1,
        help='Iteration index to plot (default: -1, last)'
    )
    
    parser.add_argument(
        '-s', '--species',
        type=str,
        default='e',
        help='Species to plot (default: e)'
    )
    
    parser.add_argument(
        '--fields',
        type=str,
        default='Ez,Jx,Jz,Bx,By,rho',
        help='Fields to plot, comma-separated (default: Ez,Jx,Jz,Bx,By,rho)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output (default: False)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save figure to file (default: None, display only)'
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
    fields_list = parse_list_arg(args.fields, dtype=str)
    
    # Parse choose_times: can be a single int or comma-separated list
    if args.choose_times is not None:
        if ',' in args.choose_times:
            choose_times = parse_list_arg(args.choose_times, dtype=int)
        else:
            choose_times = int(args.choose_times)
    else:
        choose_times = None
    
    print(f"Parsed choose_times: {choose_times}")

    if args.verbose:
        print(f"Configuration:")
        print(f"  Experiment: {args.experiment}")
        print(f"  Files path: {args.files_path}")
        print(f"  Choose species: {choose_species}")
        print(f"  Choose times: {choose_times}")
        print(f"  Iteration to plot: {args.iteration}")
        print(f"  Species to plot: {args.species}")
        print(f"  Fields: {fields_list}")
        print(f"  X range: {choose_x}")
        print(f"  Y range: {choose_y}")
        print()
    
    # Define fields to read
    fields_to_read = {
        "B": True,
        "B_ext": False,
        "divB": False,
        "E": True,
        "E_ext": False,
        "rho": True,
        "J": True,
        "P": True,
        "PI": True,
        "gyro_radius": True,
        "Heat_flux": False,
        "N": False,
        "Qrem": False,
        "EF": False
    }
    
    # Load data
    if args.verbose:
        print(f"Loading data for experiment: {args.experiment}")
    
    

    try:
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
    except Exception as e:
        files_path = f"{args.files_path}/{args.experiment}/"
        import numpy as np
        cycles, times = np.array(rp.ipic3D_available_cycles(files_path))
        cycles = cycles[choose_times] if choose_times is not None else cycles
        print(f"Loading data using fallback method for cycles: {cycles}")
        data = rp.read_data_ipic3d(files_path, cycles=cycles,
                                   fields_to_read=fields_to_read,
                                   choose_species=choose_species,
                                   choose_x=choose_x,
                                   choose_y=choose_y,
                                   verbose=args.verbose)
    
    
    
    
    if args.verbose:
        print(f"Data loaded successfully.")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")
        print(f"  Number of time steps: {len(times)}")
        print()
    
    # Determine iteration index
    if args.iteration < 0:
        iteration = len(times) + args.iteration
    else:
        iteration = args.iteration
    
    if args.verbose:
        print(f"Plotting iteration {iteration} (time={times[iteration]}) for species '{args.species}'")
        print()
    
    # Create figure
    num_fields = len(fields_list)
    ncols = 2
    nrows = (num_fields + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6 * nrows))
    
    # Handle single row case
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot fields
    for idx, field in enumerate(fields_list):
        row, col = divmod(idx, ncols)
        
        # Get field data
        if isinstance(data[field], dict):
            field_value = data[field][args.species][..., iteration]
        else:
            field_value = data[field][..., iteration]
        
        if args.verbose:
            print(f"Field '{field}': min={field_value.min():.4e}, max={field_value.max():.4e}")
        
        # Determine colormap and normalization
        if field_value.max() * field_value.min() < 0:
            # Signed field
            vmax = max(field_value.max(), -field_value.min())
            vmin = -vmax
            cmap = 'seismic'
        else:
            # Unsigned field
            if field_value.max() <= 0:
                field_value = -field_value
            cmap = 'viridis'
            vmax = field_value.max()
            vmin = 0
        
        # Plot
        im = axes[row, col].pcolormesh(X, Y, field_value, cmap=cmap, vmax=vmax, vmin=vmin)
        axes[row, col].set_title(f"{field}, time={times[iteration]:.4e}, species={args.species}", 
                                 loc='left', pad=5)
        
        # Move tickmarks and tick labels inside
        axes[row, col].tick_params(direction='in', which='both', pad=-15)
        
        # Move title inside using transform with offset
        title = axes[row, col].title
        title.set_position((0.02, 0.98))
        title.set_transform(axes[row, col].transAxes)
        title.set_verticalalignment('top')
        title.set_bbox(dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        #axes[row, col].set_xlabel("X")
        #axes[row, col].set_ylabel("Y")
        fig.colorbar(im, ax=axes[row, col])
    
    # Hide unused subplots
    for idx in range(num_fields, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if args.output:
        if args.verbose:
            print(f"Saving figure to: {args.output}")
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    if args.verbose:
        print("Done!")


if __name__ == '__main__':
    main()
