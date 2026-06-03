#!/bin/bash
#SBATCH --job-name=ortholog_convert
#SBATCH --output=slurm_logs/ortholog_convert_%j.out
#SBATCH --error=slurm_logs/ortholog_convert_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/pathway_distance/

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0

# Activate it
source /lisc/data/scratch/menche/ines/multi_env/bin/activate
pip install requests

# Convert mouse LR pairs to human orthologs
python ortholog_conversion_lr.py \
    --input  lr_pairs.csv \
    --output human_lr_pairs.csv \
    --unmapped unmapped_lr_pairs.csv \
    --delay 0.1

echo "Finished at: $(date)"