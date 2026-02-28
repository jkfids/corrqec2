#!/bin/bash
#SBATCH --account=mx95
#SBATCH --ntasks=1
# SBATCH --array=0-299%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home/jkam/mx95_scratch2/jkam/corrqec2_results/logs/experiment3_%A_%a.out
#SBATCH --error=/home/jkam/mx95_scratch2/jkam/corrqec2_results/logs/experiment3_%A_%a.err

set -euo pipefail

# Prevent numpy/BLAS from using extra threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FLEXIBLAS_NUM_THREADS=1

# Load module and conda environment
module load miniforge3
conda activate corrqec2-env

# Provided by submit_experiment3.sh via --export
: "${PARAMS_CSV:?PARAMS_CSV env var not set}"

TASK_SCRIPT="${SLURM_SUBMIT_DIR}/scripts/experiment3_task.py"

# Each task reads the (task_id)-th data row (skip header)
ROW=$((SLURM_ARRAY_TASK_ID + 2))
LINE=$(sed -n "${ROW}p" "${PARAMS_CSV}")

if [ -z "${LINE}" ]; then
    echo "No params line for task ${SLURM_ARRAY_TASK_ID} (row ${ROW}) in ${PARAMS_CSV}"
    exit 2
fi

# Must match the CSV column order written by experiment3_gen_csv.py:
# distance, rounds, shots, theta, a, b
IFS=',' read -r distance rounds shots theta a b <<< "${LINE}"

python -u "${TASK_SCRIPT}" \
    --distance "${distance}" \
    --rounds "${rounds}" \
    --shots "${shots}" \
    --theta "${theta}" \
    --a "${a}" \
    --b "${b}"