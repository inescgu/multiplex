#!/bin/bash
#SBATCH --job-name=step7b_lcc
#SBATCH --output=slurm_logs/step7b_lcc_%j.out
#SBATCH --error=slurm_logs/step7b_lcc_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      # match this to the workers you actually want

cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail

echo "Step 7b: Per-cell-type LCC significance analysis"
echo "Host: $(hostname)"
echo "Start: $(date)"

module load Python/3.14.6-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate

python step7b_layer_lcc_analysis.py \
    --wdir 04_multixrank_prep/multixrank_hetero_input_pseudobulk \
    --seed-genes 00_inputs/seed_genes.csv \
    --pool-mode reference \
    --output-dir 07b_layer_lcc_analysis \
    --plots-dir 07b_layer_lcc_analysis/plots \
    --per-component

echo "End: $(date)"