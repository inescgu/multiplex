#!/bin/bash
#SBATCH --job-name=bm
#SBATCH --output=slurm_logs/bm_%j.out
#SBATCH --error=slurm_logs/bm_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/


# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.12.3-GCCcore-13.3.0

# Activate it
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate
  

python layer_stats.py \
    --wdir multixrank_input_consolidated \
    --csv  layers_categories.csv \
    --output-dir topology_output \
    --top-n-eigen 5 \
    --exclude-ai "immune tissue" \
    --min-edges 1
  
# End time
echo "Finished at: $(date)"