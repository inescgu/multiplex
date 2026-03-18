#!/bin/bash
#SBATCH --job-name=build_2024
#SBATCH --output=slurm_logs/bm_%j.out
#SBATCH --error=slurm_logs/bm_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p multiplex_2024hdrg
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
python build_24hdrg_multiplex.py \
    --base-ppi ../og_gene_set/temp/combined/full_network.pkl \
    --human-expr-file ./converted_expression/all_tissues_combined_human.csv \
    --output-dir multiplex_2024hdrg \
    --expression-threshold 0.00000001 \
    --coexpr-method pearson

# End time
echo "Finished at: $(date)"