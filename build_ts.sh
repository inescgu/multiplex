#!/bin/bash
#SBATCH --job-name=build_ts
#SBATCH --output=slurm_logs/ts_%j.out
#SBATCH --error=slurm_logs/ts_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p multiplex_ts
mkdir -p slurm_logs

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0
pip install --user pandas numpy networkx scipy

# Run the multiplex interactome construction
python build_tabulasapiens_multiplex.py \
    --ppi-dir ./tabula_sapiens \
    --output-dir multiplex_ts \
    --seed-genes ../og_gene_set/seed_genes.csv

# End time
echo "Finished at: $(date)"