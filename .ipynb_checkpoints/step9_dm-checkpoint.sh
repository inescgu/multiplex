#!/bin/bash
#SBATCH --job-name=step9_dm
#SBATCH --output=slurm_logs/step9_dm_%j.out
#SBATCH --error=slurm_logs/step9_dm_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=35G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 9: Building disease module"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step9_build_disease_module.py \
    --scores 05_multixrank_results/multixrank_results_consolidated/gene_rwr_scores.csv \
    --per-celltype-scores 05_multixrank_results/multixrank_results_consolidated/per_celltype_scores.csv \
    --input-dir 04_multixrank_prep/multixrank_input_consolidated \
    --output-dir 06_disease_module/disease_module_output_consolidated \
    --connectivity all \
    --k 1000 \
    --max-bridge 4000 \
    --exclude-seeds SSTR4 RGS9 NPPC THBS4 IL4 IDO1 HCRT BAMBI SRD5A1 \
    --lcc-per-layer

echo "Finished at: $(date)"