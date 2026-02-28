#!/bin/bash
#SBATCH --account=mx95
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/jkam/mx95_scratch2/jkam/corrqec2_results/logs/experiment3_processing_%j.out
#SBATCH --error=/home/jkam/mx95_scratch2/jkam/corrqec2_results/logs/experiment3_processing_%j.err

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p /home/jkam/mx95_scratch2/jkam/corrqec2_results/logs

# Load module and conda environment
module load miniforge3
conda activate corrqec2-env

python scripts/experiment3_processing.py