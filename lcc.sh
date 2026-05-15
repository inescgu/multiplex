#!/bin/bash
#SBATCH --job-name=lcc
#SBATCH --output=slurm_logs/lcc_%j.out
#SBATCH --error=slurm_logs/lcc_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/


# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.14.4-weekly  
# Activate it
source /lisc/data/scratch/menche/ines/general_env/bin/activate
  
# Step 1
python multiplex_lcc.py \
    --multiplex-pkl integrated_multiplex_output/integrated_multiplex.pkl \
    --seed-genes ../og_gene_set/seed_genes.csv \
    --node-registry integrated_multiplex_output/node_registry.csv \

# End time
echo "Finished at: $(date)"