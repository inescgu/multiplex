#!/bin/bash
#SBATCH --job-name=big
#SBATCH --output=slurm_logs/big_%j.out
#SBATCH --error=slurm_logs/big_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p multiplex_2025hdrg
mkdir -p slurm_logs


# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0
pip install --user pandas numpy networkx pickly scipy 


# Run the multiplex interactome construction
python build_master_multiplex.py \
  --ts-layers-dir   ./new_ts/layers \
  --hrdg-layers-dir ./multiplex_2025hdrg/layers \
  --reactome-csv    ./pathway_distance/reactome_interactions_with_pathways.csv \
  --output-dir      integrated_multiplex_output

# End time
echo "Finished at: $(date)"