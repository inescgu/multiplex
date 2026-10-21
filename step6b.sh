#!/bin/bash
#SBATCH --job-name=step6b_diag
#SBATCH --output=slurm_logs/step6b_diag_%j.out
#SBATCH --error=slurm_logs/step6b_diag_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=35G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 6b: Investigating disease module structure"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step6b_module_diagnostics.py \
    --output-dir 06_disease_module/disease_module_output_pseudobulk \
    --input-dir  04_multixrank_prep/multixrank_hetero_input_pseudobulk \
    --markers-csv 00_inputs/hdrg_pseudobulk_markers_all.csv \
    --celltype-gene-sets 04_multixrank_prep/multixrank_hetero_input_pseudobulk/celltype_gene_sets.json
    
echo "Finished at: $(date)"