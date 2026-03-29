#!/bin/bash

PATH_NAME=$1
RUN_NAME=$2

echo "Computing field images for run: $RUN_NAME in path: $PATH_NAME"

python /volume1/scratch/georgem/closure/src/compute_field_images.py EPz $PATH_NAME $RUN_NAME 0.0003
python /volume1/scratch/georgem/closure/src/compute_field_images.py EPx $PATH_NAME $RUN_NAME 0.0003
python /volume1/scratch/georgem/closure/src/compute_field_images.py Jx-tot $PATH_NAME $RUN_NAME 0.01
python /volume1/scratch/georgem/closure/src/compute_field_images.py Jz-tot $PATH_NAME $RUN_NAME 0.01
python /volume1/scratch/georgem/closure/src/compute_field_images.py Ex $PATH_NAME $RUN_NAME 0.003
python /volume1/scratch/georgem/closure/src/compute_field_images.py Ez $PATH_NAME $RUN_NAME 0.0005

