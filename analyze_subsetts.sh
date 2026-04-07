#!/bin/bash
#SBATCH --job-name=analyze_ts
#SBATCH --output=slurm_logs/ats_%j.out
#SBATCH --error=slurm_logs/ats_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=14G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0
pip install --user pandas numpy networkx scipy

# Run the multiplex interactome construction
python analyze_subset_tabulasapiens.py \
    --layers-dir new_ts/layers \
    --celltype-list converted_cells.txt \
    --output-dir subset_analysis \
    --seed-genes-file ../og_gene_set/seed_genes.csv

# End time
echo "Finished at: $(date)"