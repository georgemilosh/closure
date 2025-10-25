"""datasplit.py
This script collects file paths matching a specified pattern from multiple folders and appends them to a CSV file. 
It is intended for use in data preparation workflows where lists of files from different datasets or experiments 
need to be aggregated for further processing.
Usage:
    python datasplit.py folders=[folder1,folder2,...] name=output.csv [root_folder=/path/to/root] [pattern=glob_pattern] [max_number=N] [min_number=M]
Arguments:
    folders       (required): List of folder names or paths to search for files. Can be specified as a Python list 
                             (e.g., [a,b,c]) or as a comma-separated string (e.g., a,b,c).
    name          (required): Output CSV filename.
    root_folder   (optional): Root directory to prepend to each folder path. If not specified, folders are treated as 
                             relative or absolute paths as given.
    pattern       (optional): Glob pattern to match files within each folder. Default is "T2D-Fields_*".
    max_number    (optional): Maximum number allowed between _ and . in filenames. Files with numbers above this 
                             threshold will be ignored. If not specified, no filtering is applied.
    min_number    (optional): Minimum number required between _ and . in filenames. Files with numbers below this 
                             threshold will be ignored. If not specified, no filtering is applied.
Functionality:
    - Parses command-line arguments in key=value format.
    - Validates the existence of specified folders.
    - Globs files matching the given pattern in each folder.
    - Filters out files with numbers between _ and . that exceed the max_number threshold.
    - Filters out files with numbers between _ and . that are below the min_number threshold.
    - Appends the relative or absolute file paths to the specified CSV file, writing a header if the file does not exist.
    - Prints a summary of the files added.
Functions:
    create_files_csv(folders, csv_filename, pattern, root_folder, max_number, min_number):
        Collects files from folders and appends their paths to a CSV file.
    extract_number_from_filename(filename):
        Extracts the number between _ and . from a filename.
    parse_arguments():
        Parses command-line arguments in key=value format.
    parse_folder_list(folder_string):
        Parses a folder list from a string in either Python list or comma-separated format.
    main():
        Main entry point for command-line execution.
Example:
    python datasplit.py folders=[T2D14_filter2,T2D15_filter2] name=train.csv root_folder=/data/ max_number=10000 min_number=5000
    python datasplit.py folders=[DeltaB.1_u.4_filter2,DeltaB.25_u1_filter2,DeltaB.4_u1.6_filter2,Smooth10_filter2,,DeltaB.85_u3.4_filter2,DeltaB1_u4_filter2] name=/dodrio/scratch/projects/2025_012/georgem/ecsim/peppe/sampling/DeltaB_B/DeltaB.75_u3/train.csv root_folder=/dodrio/scratch/projects/2025_012/georgem/ecsim/peppe/ min_number=7500
"""
import glob
import csv
import os
import sys
import ast
import re

def extract_number_from_filename(filename):
    """
    Extract the number between _ and . from a filename.
    
    Args:
        filename: The filename to extract the number from
        
    Returns:
        int: The extracted number, or None if no valid number is found
    """
    # Get just the filename without the path
    basename = os.path.basename(filename)
    
    # Look for pattern: underscore followed by digits followed by dot
    match = re.search(r'_(\d+)\.', basename)
    
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    
    return None

def create_files_csv(folders, csv_filename, pattern="T2D-Fields_*", root_folder="", max_number=None, min_number=None):
    """
    Glob files matching pattern from predefined folders and write to CSV (overwrite if exists).
    
    Args:
        folders: List of folder paths to search
        csv_filename: Output CSV filename
        pattern: File pattern to match (default: "T2D-Fields_*")
        root_folder: Root directory to prepend to folder paths
        max_number: Maximum number allowed between _ and . in filenames (None for no filtering)
        min_number: Minimum number required between _ and . in filenames (None for no filtering)
    """
    
    # Open CSV in write mode (overwrite if exists)
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['filenames'])
        
        total_files_found = 0
        total_files_filtered = 0
        total_files_written = 0
        
        # Process each folder
        for folder in folders:
            # Combine root folder with relative folder path
            if root_folder:
                full_folder_path = os.path.join(root_folder, folder)
            else:
                full_folder_path = folder
            
            # Create full pattern path
            full_pattern = os.path.join(full_folder_path, pattern)
            
            # Glob files matching pattern
            files = glob.glob(full_pattern)
            total_files_found += len(files)
            
            # Sort files for consistent ordering
            files.sort()
            
            # Filter files based on max_number and min_number if specified
            filtered_files = []
            for file_path in files:
                file_number = extract_number_from_filename(file_path)
                
                # Apply max_number filter
                if max_number is not None and file_number is not None and file_number > max_number:
                    total_files_filtered += 1
                    continue
                
                # Apply min_number filter
                if min_number is not None and file_number is not None and file_number < min_number:
                    total_files_filtered += 1
                    continue
                
                filtered_files.append(file_path)
            
            total_files_written += len(filtered_files)
            
            # Write each file to CSV (use relative path from root_folder if specified)
            for file_path in filtered_files:
                if root_folder and file_path.startswith(root_folder):
                    # Store relative path from root_folder
                    relative_path = os.path.relpath(file_path, root_folder)
                    writer.writerow([relative_path])
                else:
                    writer.writerow([file_path])
    
    if max_number is not None or min_number is not None:
        filter_msg = []
        if min_number is not None:
            filter_msg.append(f"numbers < {min_number}")
        if max_number is not None:
            filter_msg.append(f"numbers > {max_number}")
        
        print(f"Created {csv_filename} with {total_files_written} files from {len(folders)} folders")
        print(f"Found {total_files_found} total files, filtered out {total_files_filtered} files with {' and '.join(filter_msg)}")
    else:
        print(f"Created {csv_filename} with {total_files_written} files from {len(folders)} folders")

def parse_arguments():
    """Parse command line arguments in key=value format."""
    args = {}
    
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
        else:
            print(f"Warning: Ignoring invalid argument format: {arg}")
    
    return args

def parse_folder_list(folder_string):
    """Parse folder list from string format [a,b,c] or a,b,c"""
    # Remove brackets if present
    if folder_string.startswith('[') and folder_string.endswith(']'):
        folder_string = folder_string[1:-1]
    
    # Split by comma and strip whitespace
    folders = [f.strip() for f in folder_string.split(',')]
    
    # Remove empty strings
    folders = [f for f in folders if f]
    
    if not folders:
        print(f"Error: No folders found in: {folder_string}")
        print("Use format: folders=[a,b,c] or folders=a,b,c")
        sys.exit(1)
    
    return folders

def main():
    """Main function to handle command line execution."""
    args = parse_arguments()
    
    # Check required arguments
    if 'folders' not in args:
        print("Error: folders parameter is required")
        print("Usage: python datasplit.py folders=[a,b,c] name=train.csv [root_folder=/data/] [pattern=T2D-Fields_*] [max_number=N] [min_number=M]")
        sys.exit(1)
    
    if 'name' not in args:
        print("Error: name parameter is required")
        print("Usage: python datasplit.py folders=[a,b,c] name=train.csv [root_folder=/data/] [pattern=T2D-Fields_*] [max_number=N] [min_number=M]")
        sys.exit(1)
    
    # Parse arguments
    folders = parse_folder_list(args['folders'])
    csv_filename = args['name']
    pattern = args.get('pattern', 'T2D-Fields_*')
    root_folder = args.get('root_folder', '')
    max_number = args.get('max_number', None)
    min_number = args.get('min_number', None)
    
    # Parse max_number if provided
    if max_number is not None:
        try:
            max_number = int(max_number)
            print(f"Filtering files with numbers > {max_number}")
        except ValueError:
            print(f"Error: max_number must be an integer, got: {max_number}")
            sys.exit(1)
    
    # Parse min_number if provided
    if min_number is not None:
        try:
            min_number = int(min_number)
            print(f"Filtering files with numbers < {min_number}")
        except ValueError:
            print(f"Error: min_number must be an integer, got: {min_number}")
            sys.exit(1)
    
    # Clean up root_folder path
    if root_folder:
        root_folder = os.path.abspath(root_folder)
        print(f"Using root folder: {root_folder}")
    
    print(f"Processing folders: {folders}")
    
    # Validate folders exist
    missing_folders = []
    for folder in folders:
        if root_folder:
            full_path = os.path.join(root_folder, folder)
        else:
            full_path = folder
        
        if not os.path.exists(full_path):
            missing_folders.append(full_path)
    
    if missing_folders:
        print(f"Warning: The following folders do not exist: {missing_folders}")
    
    # Run the function
    create_files_csv(folders, csv_filename, pattern, root_folder, max_number, min_number)
    
    # Print summary
    print(f"\nFiles written to {csv_filename}:")
    try:
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            count = 0
            for row in reader:
                count += 1
                if count <= 10:  # Show first 10 files
                    print(f"  {row[0]}")
            if count > 10:
                print(f"  ... and {count - 10} more files")
            print(f"Total: {count} files")
    except FileNotFoundError:
        print("No files were written.")

if __name__ == "__main__":
    main()