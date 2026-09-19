#!/bin/bash
#SBATCH --job-name=step12b
#SBATCH --output=slurm_logs/step12b_%j.out
#SBATCH --error=slurm_logs/step12b_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
cd /lisc/home/user/gerardursin/multiplex/

echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate

python step12b_layer_preserving_null.py \
    --module-graph 06_disease_module/disease_module_output/module_graph.pkl \
    --disease-module 06_disease_module/disease_module_output/disease_module.csv \
    --output-dir 12_hub_removal \
    --stratify-by-seed-status \
    --n-null 50 \
    --n-null-cascading 15 \
    --step 1 \
    --cascading-step 10 \
    --null-betweenness-k 200 \
    --cascading-modes degree,betweenness,degree_betweenness,random \
    --n-jobs 4
echo "Finished at: $(date)"