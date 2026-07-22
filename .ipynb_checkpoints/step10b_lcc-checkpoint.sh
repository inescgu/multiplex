#!/bin/bash
#SBATCH --job-name=step10b_lcc
#SBATCH --output=slurm_logs/step10b_lcc_%j.out
#SBATCH --error=slurm_logs/step10b_lcc_%j.err
#SBATCH --time=36:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      # match this to n_workers you actually want
cd /lisc/home/user/gerardursin/multiplex/
echo "Running Layer LCC Analysis"
echo "Host: $(hostname)"
echo "Start: $(date)"
module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate
python step10b_layer_lcc_analysis.py \
    --multiplex-pkl 03_integrated/integrated_multiplex_output_LR/integrated_multiplex.pkl \
    --node-registry 03_integrated/integrated_multiplex_output_LR/node_registry.csv \
    --seed-genes 00_inputs/seed_genes.csv \
    --output-dir 10b_layer_lcc_analysis \
    --plots-dir 10b_layer_lcc_analysis/plots \
    --per-component
echo "End: $(date)"
