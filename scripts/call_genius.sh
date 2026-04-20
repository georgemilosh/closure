#!/bin/bash

##For example, you can call the script like this:
## ./call_genius.sh --NODES=4 --GPUS=2 --CPUS=8
##This command will execute the call_genius.sh script with the values 4 for NODES, 2 for GPUS, and 8 for CPUS.
##Inside the script, the while loop iterates over the command-line arguments using the shift command. The case 
##statement checks each argument and assigns its value to the corresponding variable (NODES, GPUS, or CPUS). 
##If an invalid argument is provided, the script will display an error message and exit with a status code of 1.
##Finally, the script constructs and executes a sbatch command using the values of NODES, GPUS, and CPUS. 
##The command is printed to the console using echo, and then executed using sbatch.

while [[ $# -gt 0 ]]; do
    case "$1" in
        --NODES=*)
            NODES="${1#*=}"
            shift
            ;;
        --GPUS=*)
            GPUS="${1#*=}"
            shift
            ;;
        --CPUS=*)
            CPUS="${1#*=}"
            shift
            ;;
        *)
            echo "Invalid argument: $1"
            exit 1
            ;;
    esac
done

echo "sbatch --job-name=job${NODES}-${GPUS}-${CPUS}- --nodes=${NODES} --gres=gpu:${GPUS} --ntasks-per-node=${GPUS} --cpus-per-task=${CPUS} genius.sh"
sbatch --job-name=job${NODES}-${GPUS}-${CPUS}- --nodes=${NODES} --gres=gpu:${GPUS} --ntasks-per-node=${GPUS} --cpus-per-task=${CPUS} genius.sh 