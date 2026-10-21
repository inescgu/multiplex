#!/bin/bash
#SBATCH --job-name=step7c_connectivity
#SBATCH --output=slurm_logs/step7c_connectivity_%j.out
#SBATCH --error=slurm_logs/step7c_connectivity_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail

echo "Step 7c: Inter-cell-type connectivity analysis (merged old step10 pathway-connectivity + step14 interlayer-edges)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

# --no-plots gives you the CSVs + text report in a single fast pass;
# drop the flag once you want the DRG-hierarchy heatmap PNGs too.
python step7c_interlayer_connectivity_analysis.py \
    --wdir 04_multixrank_prep/multixrank_hetero_input_pseudobulk \
    --lr-csv pathway_prep/lr_pairs_with_pathways.csv \
    --reactome-csv pathway_prep/reactome_interactions_with_pathways_v2.csv \
    --output-dir 07_analysis/interlayer_connectivity \
    2>&1 | tee slurm_logs/step7c_connectivity_$(date +%s).log

echo "Finished at: $(date)"