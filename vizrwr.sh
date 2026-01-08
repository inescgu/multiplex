#!/bin/bash
#SBATCH --job-name=vizrwr
#SBATCH --output=slurm_logs/viz_mp_rwr_%j.out
#SBATCH --error=slurm_logs/viz_mp_rwr_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
# mkdir -p slurm_logs
mkdir -p plots

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load python3/default
# pip install --user multixrank 

# Run the multiplex RWR
python viz_rwr.py \
    # --multiplex-dir multixrank_setup \
    # --output-dir plots \
    # --top-n 10
    # --seed-file '../og_gene_set/seed_genes.csv'

# End time
echo "Finished at: $(date)"