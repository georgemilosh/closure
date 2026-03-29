#!/usr/bin/env python3
"""
Convert .h5.pkl files to .npz format for better data sharing and archival.

This script:
- Converts pickle files to NumPy compressed archives (.npz)
- Processes multiple simulation directories
- Writes converted files to new output directories
- Preserves all field data and metadata files
- Provides verification and logging

Usage:
    python convert_pkl_to_npz.py --data_dir /path/to/data --output_dir /path/to/output --pattern "T2D*_filter*"
    python convert_pkl_to_npz.py --data_dir /path/to/data --output_dir /path/to/output --dirs T2D13_filter2 T2D14_filter2
    
Author: George Miloshevich
Date: December 2025
License: MIT
"""

import os
import sys
import pickle
import numpy as np
import argparse
import glob
import logging
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def convert_pkl_to_npz(pkl_path, npz_path=None, verify=True):
    """
    Convert a single .h5.pkl file to .npz format.
    
    Args:
        pkl_path (str): Path to input .pkl file
        npz_path (str, optional): Path to output .npz file. If None, replaces .h5.pkl with .npz
        verify (bool): Whether to verify the conversion by reading back the file
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    if npz_path is None:
        npz_path = pkl_path.replace('.h5.pkl', '.npz')
    
    try:
        # Load pickle file
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if data is a dictionary
        if not isinstance(data, dict):
            logger.warning(f"Expected dictionary in {pkl_path}, got {type(data)}")
            return False
        
        # Save as compressed NPZ
        np.savez_compressed(npz_path, **data)
        
        # Verify if requested
        if verify:
            npz_data = np.load(npz_path)
            
            # Check all keys are present
            if set(data.keys()) != set(npz_data.keys()):
                logger.error(f"Key mismatch in {npz_path}")
                logger.error(f"  Original: {sorted(data.keys())}")
                logger.error(f"  Converted: {sorted(npz_data.keys())}")
                return False
            
            # Check array shapes and values
            for key in data.keys():
                if not np.allclose(data[key], npz_data[key], rtol=1e-7, atol=1e-10):
                    logger.error(f"Data mismatch for key '{key}' in {npz_path}")
                    return False
            
            npz_data.close()
        
        # Log file sizes
        pkl_size = os.path.getsize(pkl_path) / (1024**2)  # MB
        npz_size = os.path.getsize(npz_path) / (1024**2)  # MB
        compression_ratio = (1 - npz_size/pkl_size) * 100
        
        logger.info(f"Converted {os.path.basename(pkl_path)}: "
                   f"{pkl_size:.2f} MB → {npz_size:.2f} MB "
                   f"({compression_ratio:+.1f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert {pkl_path}: {e}")
        return False


def convert_directory(input_dir, output_dir, pattern="*Fields*.h5.pkl", 
                      copy_metadata=True, dry_run=False, verify=True):
    """
    Convert all .h5.pkl files in a directory to a new output directory.
    
    Args:
        input_dir (str): Source directory containing .h5.pkl files
        output_dir (str): Destination directory for .npz files
        pattern (str): Glob pattern for files to convert
        copy_metadata (bool): If True, copy non-.pkl files (txt, etc.) to output
        dry_run (bool): If True, only show what would be converted
        verify (bool): Whether to verify each conversion
        
    Returns:
        tuple: (successful_count, failed_count, total_size_saved)
    """
    pkl_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    if not pkl_files:
        logger.warning(f"No files matching '{pattern}' found in {input_dir}")
        return 0, 0, 0
    
    logger.info(f"Found {len(pkl_files)} .pkl files in {input_dir}")
    
    # Create output directory if it doesn't exist
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    if dry_run:
        logger.info("DRY RUN - no files will be converted")
        logger.info(f"  Would create: {output_dir}")
        for pkl_file in pkl_files:
            npz_name = os.path.basename(pkl_file).replace('.h5.pkl', '.npz')
            logger.info(f"  Would convert: {os.path.basename(pkl_file)} -> {npz_name}")
        return 0, 0, 0
    
    successful = 0
    failed = 0
    total_saved = 0
    
    # Convert .pkl files to .npz
    for pkl_file in tqdm(pkl_files, desc=f"Converting {os.path.basename(input_dir)}"):
        pkl_basename = os.path.basename(pkl_file)
        npz_basename = pkl_basename.replace('.h5.pkl', '.npz')
        npz_file = os.path.join(output_dir, npz_basename)
        
        # Skip if already exists
        if os.path.exists(npz_file):
            logger.info(f"Skipping {pkl_basename} - NPZ already exists")
            continue
        
        pkl_size = os.path.getsize(pkl_file)
        
        if convert_pkl_to_npz(pkl_file, npz_file, verify=verify):
            successful += 1
            npz_size = os.path.getsize(npz_file)
            total_saved += (pkl_size - npz_size)
        else:
            failed += 1
    
    # Copy metadata files (SimulationData.txt, ConservedQuantities.txt, etc.)
    if copy_metadata:
        import shutil
        metadata_patterns = ['*.txt', '*.md', '*.json', '*.yml', '*.yaml']
        copied_files = 0
        
        for pattern in metadata_patterns:
            for metadata_file in glob.glob(os.path.join(input_dir, pattern)):
                dest_file = os.path.join(output_dir, os.path.basename(metadata_file))
                if not os.path.exists(dest_file):
                    shutil.copy2(metadata_file, dest_file)
                    copied_files += 1
        
        if copied_files > 0:
            logger.info(f"Copied {copied_files} metadata files to {output_dir}")
    
    return successful, failed, total_saved


def main():
    parser = argparse.ArgumentParser(
        description='Convert .h5.pkl files to .npz format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all T2D*_filter* directories to zenodo_data/
  python convert_pkl_to_npz.py \\
      --data_dir /volume1/scratch/share_dir/ecsim/peppe \\
      --output_dir /volume1/scratch/share_dir/zenodo_data \\
      --pattern "T2D*_filter*"
  
  # Convert specific directories
  python convert_pkl_to_npz.py \\
      --data_dir /volume1/scratch/share_dir/ecsim/peppe \\
      --output_dir /volume1/scratch/share_dir/zenodo_data \\
      --dirs T2D13_filter2 T2D14_filter2
  
  # Dry run to see what would be converted
  python convert_pkl_to_npz.py \\
      --data_dir /path/to/data \\
      --output_dir /path/to/output \\
      --pattern "T2D*" --dry_run
  
  # Skip copying metadata files (only convert .pkl to .npz)
  python convert_pkl_to_npz.py \\
      --data_dir /path/to/data \\
      --output_dir /path/to/output \\
      --dirs T2D13_filter2 --no_copy_metadata
        """
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Base directory containing source simulation folders')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Base directory for output (converted) files')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pattern', type=str,
                      help='Glob pattern for directories to process (e.g., "T2D*_filter*")')
    group.add_argument('--dirs', nargs='+',
                      help='Specific directory names to process')
    
    parser.add_argument('--file_pattern', type=str, default='*Fields*.h5.pkl',
                       help='Pattern for files within directories (default: *Fields*.h5.pkl)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be converted without actually converting')
    parser.add_argument('--no_copy_metadata', action='store_true',
                       help='Skip copying metadata files (.txt, .json, etc.)')
    parser.add_argument('--no_verify', action='store_true',
                       help='Skip verification of converted files (faster but less safe)')
    
    args = parser.parse_args()
    
    # Get directories to process
    if args.pattern:
        input_directories = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
        input_directories = [d for d in input_directories if os.path.isdir(d)]
    else:
        input_directories = [os.path.join(args.data_dir, d) for d in args.dirs]
        input_directories = [d for d in input_directories if os.path.isdir(d)]
    
    if not input_directories:
        logger.error(f"No directories found matching the criteria in {args.data_dir}")
        return 1
    
    logger.info(f"Processing {len(input_directories)} directories")
    logger.info(f"Input base: {args.data_dir}")
    logger.info(f"Output base: {args.output_dir}")
    
    # Create output base directory
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
    
    total_successful = 0
    total_failed = 0
    total_saved = 0
    
    for input_dir in input_directories:
        # Create corresponding output directory path
        dir_name = os.path.basename(input_dir)
        output_dir = os.path.join(args.output_dir, dir_name)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {input_dir}")
        logger.info(f"Output to: {output_dir}")
        logger.info(f"{'='*60}")
        
        successful, failed, saved = convert_directory(
            input_dir,
            output_dir,
            pattern=args.file_pattern,
            copy_metadata=not args.no_copy_metadata,
            dry_run=args.dry_run,
            verify=not args.no_verify
        )
        
        total_successful += successful
        total_failed += failed
        total_saved += saved
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("CONVERSION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Directories processed: {len(directories)}")
    logger.info(f"Files converted successfully: {total_successful}")
    logger.info(f"Files failed: {total_failed}")
    logger.info(f"Space saved: {total_saved / (1024**2):.2f} MB ({total_saved / (1024**3):.2f} GB)")
    
    if total_failed > 0:
        logger.warning(f"\n{total_failed} files failed to convert. Check conversion.log for details.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())