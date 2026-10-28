#!/bin/bash
#SBATCH --job-name=step12d_tail
#SBATCH --output=slurm_logs/step12d_tail_%j.out
#SBATCH --error=slurm_logs/step12d_tail_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail

echo "Step 12d: permutation test + fc-degree-attack enrichment"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"
module load Python/3.12.3-GCCcore-13.3.0   # general_env is built on 3.12.3
source /lisc/data/scratch/menche/ines/general_env/bin/activate

python step12d_tail.py \
    --module-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
    --background-graph ../og_gene_set/temp/combined26/full_network.pkl \
    --output-dir 12_hub_removal \
    --n-permutations 2000
echo "Finished at: $(date)"