#!/bin/bash
#SBATCH --account=mx95
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
# SBATCH --mem-per-cpu=8000
#SBATCH --mem=300G
#SBATCH --time=0-08:00:00
#SBATCH --output=/home/jkam/mx95_scratch2/jkam/corrqec2_results/logs/experiment1_%j.out
#SBATCH --error=/home/jkam/mx95_scratch2/jkam/corrqec2_results/logs/experiment1_%j.err


# ============================================================================
# Environment setup
# ============================================================================

# Thread limiting for BLAS/OpenMP stuff
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# FLEXIBLAS is optional; only matters if you’re actually using it:
# export FLEXIBLAS_NUM_THREADS=1

# JAX/XLA behaviour
# export JAX_PLATFORMS=cpu
# export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# export XLA_PYTHON_CLIENT_PREALLOCATE=false


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

# python scripts/experiment1.py --num-workers $SLURM_CPUS_PER_TASK "$@"
python scripts/experiment1.py --num-workers $SLURM_CPUS_PER_TASK --max-shots 10_000_000 --batch-size 10_000

# ============================================================================
# Job finished
# ============================================================================

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="