#!/bin/bash

# Thread limiting for BLAS/OpenMP stuff
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FLEXIBLAS_NUM_THREADS=1

# Create logs directory if it doesn't exist
mkdir -p /home/jkam/mx95_scratch2/jkam/corrqec2_results/logs

# Load module and conda environment
module load miniforge3
conda activate corrqec2-env

# Generate params csv if it doesn't exist
if [ ! -f "params/experiment3_params.csv" ]; then
    echo "Generating params CSV file..."
    python scripts/generate_experiment3_gen_csv.py
else
    echo "Params CSV file already exists. Skipping generation."
fi

# Count the number of lines in the params CSV
N=$(( $(wc -l < "params/experiment3_params.csv") - 1 ))
if [ "$N" -le 0 ]; then
    echo "No rows found in $PARAMS_CSV"
    exit 2
fi

# Set max and concurrency for array job
MAX=$((N - 1))
CONCURRENCY=50

echo "Submitting job array with $N tasks: 0-$MAX (max concurrent: $CONCURRENCY)"
sbatch 
