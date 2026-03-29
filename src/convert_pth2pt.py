#!/usr/bin/env python3
"""
Convert all PyTorch models in subdirectories to TorchScript format.

This script:
1. Scans a directory for all subdirectories containing trained models
2. Attempts to load each model using Trainer.load_run()
3. Converts successfully loaded models to TorchScript format (.pt)
4. Saves scripted models alongside original checkpoints

Usage:
    python convert_pth2pt.py --work_dir /path/to/models
    python convert_pth2pt.py --work_dir /path/to/models --pattern "run_*"

Author: George Miloshevich
Date: December 2025
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch

from . import trainers as tr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('convert_pth2pt.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def convert_model_to_torchscript(trainer, run_name, force=False):
    """
    Load a model from a run directory and convert to TorchScript.
    
    Args:
        trainer: Pre-initialized Trainer object
        run_name (str): Name of the run subdirectory
        force (bool): If True, overwrite existing scripted models
        
    Returns:
        bool: True if successful, False otherwise
    """
    run_path = os.path.join(trainer.work_dir, run_name)
    output_path = os.path.join(run_path, 'scripted_model.pt')
    
    # Check if already exists
    if os.path.exists(output_path) and not force:
        logger.info(f"Skipping {run_name} - scripted model already exists")
        return True
    
    try:
        # Load the run
        logger.info(f"Loading model from {run_name}")
        trainer.load_run(run_name)
        
        # Set to evaluation mode
        trainer.model.model.eval()
        
        # Convert to TorchScript
        logger.info(f"Converting {run_name} to TorchScript")
        scripted_model = torch.jit.script(trainer.model.model)
        
        # Save scripted model
        scripted_model.save(output_path)
        logger.info(f"✓ Saved scripted model to {output_path}")
        
        # Clean up scripted model (but keep trainer)
        del scripted_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except FileNotFoundError as e:
        logger.warning(f"✗ {run_name}: Model file not found - {e}")
        return False
    except RuntimeError as e:
        logger.error(f"✗ {run_name}: Runtime error during conversion - {e}")
        return False
    except Exception as e:
        logger.error(f"✗ {run_name}: Unexpected error - {e}")
        return False


def find_model_directories(work_dir, pattern="*"):
    """
    Find all subdirectories that likely contain trained models.
    
    Args:
        work_dir (str): Base directory to search
        pattern (str): Glob pattern for directory names
        
    Returns:
        list: List of subdirectory names
    """
    work_path = Path(work_dir)
    
    # Find directories matching pattern
    candidate_dirs = sorted([d.name for d in work_path.glob(pattern) if d.is_dir()])
    
    # Filter to those with model files
    model_dirs = []
    for dirname in candidate_dirs:
        dir_path = work_path / dirname
        # Check for common model file patterns
        has_model = any([
            (dir_path / 'model.pth').exists(),
            (dir_path / 'best_model.pth').exists(),
            (dir_path / 'checkpoint.pth').exists(),
            (dir_path / 'config.json').exists()
        ])
        if has_model:
            model_dirs.append(dirname)
    
    return model_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch models to TorchScript format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all models in directory
  python convert_pth2pt.py --work_dir /path/to/models
  
  # Convert only specific pattern
  python convert_pth2pt.py --work_dir /path/to/models --pattern "run_*"
  
  # Force overwrite existing scripted models
  python convert_pth2pt.py --work_dir /path/to/models --force
  
  # Dry run to see what would be converted
  python convert_pth2pt.py --work_dir /path/to/models --dry_run
        """
    )
    
    parser.add_argument('--work_dir', type=str, default='./',
                       help='Base directory containing model subdirectories (default: ./)')
    parser.add_argument('--pattern', type=str, default='*',
                       help='Glob pattern for subdirectory names (default: *)')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing scripted models')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be converted without actually converting')
    parser.add_argument('--runs', nargs='+',
                       help='Specific run names to convert (overrides pattern)')
    
    args = parser.parse_args()
    
    # Get list of runs to process
    if args.runs:
        runs = args.runs
        logger.info(f"Processing {len(runs)} specified runs")
    else:
        runs = find_model_directories(args.work_dir, args.pattern)
        logger.info(f"Found {len(runs)} model directories matching pattern '{args.pattern}'")
    
    if not runs:
        logger.error(f"No model directories found in {args.work_dir}")
        return 1
    
    # Display what will be processed
    logger.info(f"\nRuns to process:")
    for run in runs:
        logger.info(f"  - {run}")
    
    if args.dry_run:
        logger.info("\nDRY RUN - No conversions will be performed")
        return 0
    
    # Initialize trainer ONCE
    logger.info(f"\nInitializing trainer...")
    try:
        trainer = tr.Trainer(
            work_dir=args.work_dir,
            mode_test=True,
            log_name='convert.log',
            log_level='CRITICAL'
        )
        logger.info("✓ Trainer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        return 1
    
    # Convert each run
    logger.info(f"\n{'='*60}")
    logger.info("Starting conversion")
    logger.info(f"{'='*60}\n")
    
    successful = 0
    failed = 0
    
    for i, run in enumerate(runs, 1):
        logger.info(f"[{i}/{len(runs)}] Processing: {run}")
        
        result = convert_model_to_torchscript(
            trainer,
            run,
            force=args.force
        )
        
        if result:
            successful += 1
        else:
            failed += 1
        
        logger.info("")  # Blank line between runs
    
    # Clean up trainer
    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    logger.info(f"{'='*60}")
    logger.info("CONVERSION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total runs processed: {len(runs)}")
    logger.info(f"Successfully converted: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (already exist): {len(runs) - successful - failed}")
    
    if failed > 0:
        logger.warning(f"\n{failed} conversions failed. Check convert_pth2pt.log for details.")
        return 1
    
    logger.info("\n✓ All conversions completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())