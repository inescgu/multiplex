#!/bin/bash
#SBATCH --job-name=step12a_main
#SBATCH --output=slurm_logs/step12a_main_%j.out
#SBATCH --error=slurm_logs/step12a_main_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail

echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate

python step12a_main.py \
    --module-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
    --disease-module 06_disease_module/disease_module_output_pseudobulk/disease_module.csv \
    --background-graph ../og_gene_set/temp/combined26/full_network.pkl \
    --output-dir 12_hub_removal \
    --n-random-trials 100 \
    --n-null-networks 50 \
    --n-null-cascading 15 \
    --n-permutations 2000 \
    --step 1 \
    --cascading-step 10 \
    --null-betweenness-k 200 \
    --cascading-modes degree,betweenness,degree_betweenness,random \
    --n-jobs 4 \
    --run-enrichment \
    --ablation-exclude-celltypes global_ppi \
    --skip-null --skip-permutation
echo "Finished at: $(date)"