#!/bin/bash
#SBATCH --job-name=step12_hub_removal
#SBATCH --output=slurm_logs/step12_hub_removal_%j.out
#SBATCH --error=slurm_logs/step12_hub_removal_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
cd /lisc/home/user/gerardursin/multiplex/
mkdir -p slurm_logs
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate
pip install gseapy
 
# Runs after step6_build_disease_module.py has produced:
#   06_disease_module/disease_module_output/module_graph.pkl   (nodes: "<celltype>::GENE")
#   06_disease_module/disease_module_output/disease_module.csv (node/gene/cell_type/is_seed/...)
#
# --background-graph MUST be a BARE gene-symbol PPI (e.g. graph_lcc.pkl from the
# single-layer og_gene_set pipeline) -- fill in the correct path below.

python step12_hub_removal_analysis.py \
    --module-graph 06_disease_module/disease_module_output/module_graph.pkl \
    --disease-module 06_disease_module/disease_module_output/disease_module.csv \
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
    --run-enrichment
echo "Finished at: $(date)"