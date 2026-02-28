#!/bin/bash

# Paths relative to where you run this (repo root)
GEN_SCRIPT="scripts/experiment3_gen_csv.py"
PARAMS_CSV="scripts/experiment3_params.csv"
SBATCH_SCRIPT="slurm/run_experiment3_array.sh"
PROCESSING_SCRIPT="scripts/experiment3_processing.py"

# Create logs directory if it doesn't exist
mkdir -p /home/jkam/mx95_scratch2/jkam/corrqec2_results/logs

# Load module and conda environment
module load miniforge3
conda activate corrqec2-env

# Generate params csv if it doesn't exist
if [ ! -f "$PARAMS_CSV" ]; then
    echo "Generating params CSV file..."
    python "$GEN_SCRIPT"
else
    echo "Params CSV file already exists. Skipping generation."
fi

# Count the number of lines in the params CSV
N=$(( $(wc -l < "$PARAMS_CSV") - 1 ))
if [ "$N" -le 0 ]; then
    echo "No rows found in $PARAMS_CSV"
    exit 2
fi

# Set max and concurrency for array job
MAX=$((N - 1))
CONCURRENCY=50

echo "Submitting job array with $N tasks: 0-$MAX (max concurrent: $CONCURRENCY)"
sbatch --array=0-"$MAX"%$CONCURRENCY \
  --export=ALL,PARAMS_CSV="$PARAMS_CSV" \
  "$SBATCH_SCRIPT"
