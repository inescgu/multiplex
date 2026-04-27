#!/bin/bash
#SBATCH --job-name=mc
#SBATCH --output=slurm_logs/mc_%j.out
#SBATCH --error=slurm_logs/mc_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=15G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p comparison_results

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0
pip install --user pandas numpy networkx pickly scipy 


# Run the multiplex interactome construction
python mc_fast.py \
  --tabula-layers ./new_ts/layers \
  --pain-layers ./multiplex_2025hdrg/layers \
  --pain-subset-layers ./subset_analysis/layers \
  --seed-genes ../og_gene_set/seed_genes.csv \
  --output-dir comparison_results

# End time
echo "Finished at: $(date)"