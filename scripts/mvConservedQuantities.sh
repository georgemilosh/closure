#!/bin/bash

# Define the source and destination base directories
SOURCE_BASE="hortense:/dodrio/scratch/projects/2024_109/ecsim/peppe"
DEST_BASE="/volume1/scratch/share_dir/ecsim/peppe"

# Define the list of folders
FOLDERS=("T2D10c1_filter2" "T2D10_filter2" "T2D12_filter2" "T2D13_filter2" "T2D14_filter2" "T2D15_filter2" "T2D16_filter2" "data_filter2")

# Loop over each folder and perform the rsync operation
for FOLDER in "${FOLDERS[@]}"; do
    # Construct the source folder name by removing '_filter2' if it exists
    SOURCE_FOLDER="${FOLDER/_filter2/}"
    SOURCE_FILE="${SOURCE_BASE}/${SOURCE_FOLDER}/ConservedQuantities.txt"
    DEST_DIR="${DEST_BASE}/${FOLDER}/"
    
    # Check if the source file exists
    echo rsync -avhP "${SOURCE_FILE}" "${DEST_DIR}"
    rsync -avhP "${SOURCE_FILE}" "${DEST_DIR}"
done