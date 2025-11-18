#!/bin/bash
#SBATCH --account=mx95
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/test_sampling_%j.out
#SBATCH --error=logs/test_sampling_%j.err

# ============================================================================
# Environment setup
# ============================================================================

# Load module and conda environment
module load miniforge3
conda activate corrqec2-env

# Print header info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Time started: $(date)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "=========================================="
echo ""

# ============================================================================
# Run the Python script
# ============================================================================

python scripts/test_sampling.py --num-workers $SLURM_CPUS_PER_TASK "$@"

# ============================================================================
# Job finished
# ============================================================================

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="