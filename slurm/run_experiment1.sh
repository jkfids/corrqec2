#!/bin/bash
#SBATCH --account=mx95
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2000
#SBATCH --time=0-00:30:00
#SBATCH --output=/home/jkam/mx95_scratch2/jkam/corrqec2_results/logs/test_sampling_%j.out
#SBATCH --error=/home/jkam/mx95_scratch2/jkam/corrqec2_results/logs/test_sampling_%j.err


# ============================================================================
# Environment setup
# ============================================================================

# Create logs directory if it doesn't exist
mkdir -p /home/jkam/mx95_scratch2/jkam/corrqec2_results/logs

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

python scripts/experiment1.py --num-workers $SLURM_CPUS_PER_TASK "$@"
# python scripts/experiment1.py --num-workers $SLURM_CPUS_PER_TASK --max-shots 1_000_000 --batch-size 10_000

# ============================================================================
# Job finished
# ============================================================================

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="