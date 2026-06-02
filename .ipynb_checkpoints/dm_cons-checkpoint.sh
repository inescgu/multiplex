#!/bin/bash
#SBATCH --job-name=dm_cons
#SBATCH --output=slurm_logs/dm_cons_%j.out
#SBATCH --error=slurm_logs/dm_cons_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=15G
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

# echo "=========================================="
# echo "Step 1: Build MultiXrank inputs (v2)"
# echo "=========================================="
# python dm1_build_multixrank_inputs_v2.py \
#     --multiplex-pkl integrated_multiplex_output/integrated_multiplex.pkl \
#     --seed-csv ../og_gene_set/seed_genes.csv \
#     --output-dir multixrank_input_v2

# echo ""
# echo "=========================================="
# echo "Step 2: Run MultiXrank (v2)"
# echo "=========================================="
# python dm2_run_multixrank_v2.py \
#     --wdir multixrank_input_consolidated \
#     --config config.yml \
#     --output-dir multixrank_results_v2

echo ""
echo "=========================================="
echo "Step 3: Build disease module"
echo "=========================================="
python dm3_build_disease_module.py \
    --scores multixrank_results_v2/gene_rwr_scores.csv \
    --per-celltype-scores multixrank_results_v2/per_celltype_scores.csv \
    --input-dir multixrank_input_consolidated \
    --output-dir disease_module_output_v2 \
    --connectivity all \
    --k 400 \
    --max-bridge 200 \
    --exclude-seeds   SSTR4
echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
