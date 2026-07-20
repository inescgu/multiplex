#!/bin/bash
#SBATCH --job-name=step13_viz_umap
#SBATCH --output=slurm_logs/step13_viz_umap_%j.out
#SBATCH --error=slurm_logs/step13_viz_umap_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/
echo "Step 13: Disease module visualization (UMAP, final)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate
pip install -q umap-learn
mkdir -p 08_dm_visualization

# UMAP version: slower but produces prettier layout
# Expected runtime: 30 minutes - 2 hours (depending on graph size)
python step13_viz_dm_fixed.py \
    --graph 03_integrated/integrated_multiplex_output/integrated_multiplex.pkl \
    --dm-csv 06_disease_module/disease_module_output_consolidated/disease_module.csv \
    --seed-csv 00_inputs/seed_genes.csv \
    --out-dir 08_dm_visualization \
    --use-umap

echo "Finished at: $(date)"