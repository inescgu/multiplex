#!/bin/bash
#SBATCH --job-name=run_multiplex_rwr
#SBATCH --output=slurm_logs/run_multiplex_rwr_%j.out
#SBATCH --error=slurm_logs/run_multiplex_rwr_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p slurm_logs
mkdir -p rwr_results

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load python3/default
pip install --user multixrank 

# Run the multiplex RWR
python run_mp_rwr.py \
    --multixrank-dir multixrank_setup \
    --output-dir rwr_results 
    # --seed-file '../og_gene_set/seed_genes.csv'

# End time
echo "Finished at: $(date)"