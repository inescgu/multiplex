#!/bin/bash
#SBATCH --job-name=step12g_removed
#SBATCH --output=slurm_logs/step12g_%j.out
#SBATCH --error=slurm_logs/step12g_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate

# Add --betweenness-k N / change --cascading-step if the main run used non-default values.
python step12g_removed_genes.py \
    --module-graph 06_disease_module/disease_module_output/module_graph.pkl \
    --disease-module 06_disease_module/disease_module_output/disease_module.csv \
    --output-dir 12_hub_removal \
    --modes degree,betweenness,degree_betweenness \
    --cascading --cascading-step 10 --module-background