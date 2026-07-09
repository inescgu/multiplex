#!/bin/bash
#SBATCH --job-name=step10_analysis
#SBATCH --output=slurm_logs/step10_analysis_%j.out
#SBATCH --error=slurm_logs/step10_analysis_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=15G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 10: Running topology and pathway analysis"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

# Create output directory
mkdir -p 07_analysis/topology_output
mkdir -p 07_analysis/pathway_analysis

echo ""
echo "=========================================="
echo "Pathway connection analysis"
echo "=========================================="
python step10_pathway_connection_analysis.py \
  --wdir 04_multixrank_prep/multixrank_input_consolidated \
  --reactome-csv pathway_prep/reactome_interactions_with_pathways_v2.csv \
  --uniprot-symbol-csv 00_inputs/uniprot_to_symbol.csv \
  --lr-csv pathway_prep/human_lr_pairs.csv \
  --exclude-ai "immune tissue" "immune" "nervous" \
  --csv layers_categories.csv \
  --lr-pathways-csv pathway_prep/lr_pairs_with_pathways.csv \
  --output-dir 07_analysis/pathway_analysis

echo ""
echo "=========================================="
echo "Layer topology statistics"
echo "=========================================="
python step10_layer_stats.py \
    --wdir 04_multixrank_prep/multixrank_input_consolidated \
    --csv layers_categories.csv \
    --output-dir 07_analysis/topology_output \
    --top-n-eigen 5 \
    --exclude-ai "immune tissue" "immune" "nervous" \
    --min-edges 1

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="