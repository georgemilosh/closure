# ...existing code...
#!/usr/bin/env bash                     # Use the env command to locate bash and run this script with it
set -euo pipefail                       # Exit on error (-e), treat unset vars as errors (-u), and fail a pipeline if any command fails (-o pipefail)

BASE_DIR="/volume1/scratch/share_dir/ecsim/peppe/"  # Base directory containing experiment subfolders
SCRIPT1="compute_flux.py"                # Python script to execute for each matching subfolder
SCRIPT2="compute_spectrum.py"             # Another Python script to execute for each matching subfolder

if [[ ! -d "$BASE_DIR" ]]; then         # Check that the base directory exists
  echo "Base directory not found: $BASE_DIR" >&2   # Print error to stderr if directory missing
  exit 1                                # Exit with failure
fi

if ! command -v python &>/dev/null; then  # Verify that 'python' is available in PATH
  echo "python not found in PATH" >&2    # Inform user if python is missing
  exit 1                                 # Exit with failure
fi

shopt -s nullglob                       # Make globs that match nothing expand to empty (avoid literal pattern)
matches=("$BASE_DIR"/*_filter2)         # Collect all paths in BASE_DIR ending with _filter2 into an array
shopt -u nullglob                       # Restore default globbing behavior

if (( ${#matches[@]} == 0 )); then      # Test if no matching directories were found
  echo "No *_filter2 subdirectories found in $BASE_DIR"  # Inform user that nothing matched
  exit 0                                # Exit successfully (nothing to do)
fi

echo "Found ${#matches[@]} folders."    # Report how many matching directories were found

for dir in "${matches[@]}"; do          # Iterate over each matched path
  if [[ -d "$dir" ]]; then              # Ensure the path is actually a directory
    folder_name="$(basename "$dir")"    # Extract just the directory name without its path

    echo "Running: python $SCRIPT1 $BASE_DIR $folder_name"
    if ! python "$SCRIPT1" "$BASE_DIR" "$folder_name"; then
      echo "Error running $SCRIPT1 for $folder_name" >&2
      continue
    fi

    echo "Running: python $SCRIPT2 $BASE_DIR $folder_name"
    if ! python "$SCRIPT2" "$BASE_DIR" "$folder_name"; then
      echo "Error running $SCRIPT2 for $folder_name" >&2
      continue
    fi
  fi                                     # End directory check
done                                     # End loop over matches

echo "All done."                        # Final status message
