#!/bin/bash
#SBATCH --job-name=prep_multiplex_rwr
#SBATCH --output=slurm_logs/prep_multiplex_rwr_%j.out
#SBATCH --error=slurm_logs/prep_multiplex_rwr_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=3G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p slurm_logs
mkdir -p multixrank_setup

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load python3/default

# Run the multiplex RWR preparation
# WARNING: This step is memory-intensive due to matrix inversion
# The supra-adjacency matrix size is (n_genes × n_layers)^2
python prepare_mp_rwr.py \
    --multiplex-dir multiplex_interactome \
    --output-dir multixrank_setup \
    --min-edges 5
    # --restart 0.7 \
    # --inter-layer-weight 1.0

# End time
echo "Finished at: $(date)"