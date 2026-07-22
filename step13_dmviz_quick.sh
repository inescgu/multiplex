#!/bin/bash
#SBATCH --job-name=step13_viz
#SBATCH --output=slurm_logs/step13_viz_%j.out
#SBATCH --error=slurm_logs/step13_viz_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#!/bin/bash
# Quick iteration version: uses spring layout instead of UMAP
# Expected runtime: 2-5 minutes
set -e

cd /lisc/home/user/gerardursin/multiplex/

echo "========================================"
echo "Step 13: QUICK VIZ (spring layout)"
echo "========================================"
echo "Expected runtime: 2-5 minutes"
echo ""

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

mkdir -p 08_dm_visualization

python step13_viz_dm.py \
    --graph 03_integrated/integrated_multiplex_output_LR/integrated_multiplex.pkl \
    --dm-csv 06_disease_module/disease_module_output_consolidated/disease_module.csv \
    --seed-csv 00_inputs/seed_genes.csv \
    --out-dir 08_dm_visualization

echo ""
echo "✓ Quick viz complete!"
echo "Output: 08_dm_visualization/disease_module_visualization.png"
echo ""
echo "To use UMAP instead (slower, prettier), add: --use-umap"