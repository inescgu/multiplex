#!/bin/bash
#SBATCH --job-name=step6e
#SBATCH --output=slurm_logs/step6e_%j.out
#SBATCH --error=slurm_logs/step6e_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/

echo "Export for datadivr"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

python step6e_export_wwiznet.py \
    --module-dir 06_disease_module/disease_module_output \
    --per-celltype-scores 05_multixrank_results/multixrank_hetero_results/per_celltype_scores.csv \
    --provenance 04_multixrank_prep/multixrank_hetero_input/scored_lr_edge_provenance.csv \
    --out 06_disease_module/wwiznet_export/data
echo "Finished at: $(date)"