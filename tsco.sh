#!/bin/bash
#SBATCH --job-name=tscoex
#SBATCH --output=slurm_logs/tscoex_%j.out
#SBATCH --error=slurm_logs/tscoex_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=130G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p new_ts_coex
mkdir -p slurm_logs

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

module load Python/3.10.8-GCCcore-12.2.0-bare
# python -m venv /lisc/data/scratch/menche/ines/tscoex
source /lisc/data/scratch/menche/ines/tscoex/bin/activate
# pip install --upgrade pip
pip install cellxgene-census

# Run the multiplex interactome construction
python ts_altcoex.py \
    --output-dir new_ts_coex \
    --seed-genes ../og_gene_set/seed_genes.csv\
    --expression-threshold 0.1

# End time
echo "Finished at: $(date)"

# The "stable" release is currently 2025-11-08. Specify 'census_version="2025-11-08"' in future calls to open_soma() to ensure data consistency.