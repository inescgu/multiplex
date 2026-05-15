#!/bin/bash
#SBATCH --job-name=dm
#SBATCH --output=slurm_logs/dm_%j.out
#SBATCH --error=slurm_logs/dm_%j.err
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
module load Python/3.12.3-GCCcore-13.3.0

# Activate it
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate
  
# Step 1
python dm1_build_multixrank_inputs.py \
    --multiplex-pkl integrated_multiplex_output/integrated_multiplex.pkl \
    --seed-csv ../og_gene_set/seed_genes.csv \
    --output-dir multixrank_input

# Step 2
python dm2_run_multixrank.py \
    --wdir multixrank_input \
    --config multixrank_input/config.yml \
    --output-dir multixrank_results

# Step 3
python dm3_build_disease_module.py \
    --scores multixrank_results/gene_rwr_scores.csv \
    --input-dir multixrank_input \
    --output-dir disease_module_output \
    --connectivity all \
    --k 300 \
    --max-bridge 2000

  
# End time
echo "Finished at: $(date)"