#!/bin/bash
#SBATCH --job-name=tsapi
#SBATCH --output=slurm_logs/tsapi_%j.out
#SBATCH --error=slurm_logs/tsapi_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p new_ts
mkdir -p slurm_logs

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.12.3-GCCcore-13.3.0
pip install --user scipy tiledbsoma cellxgene-census

# Run the multiplex interactome construction
python ts_api.py \
    --ppi ../og_gene_set/temp/combined26/full_network.pkl \
    --output-dir new_ts \
    --seed-genes ../og_gene_set/seed_genes.csv\
    --expression-threshold 0.1

# End time
echo "Finished at: $(date)"