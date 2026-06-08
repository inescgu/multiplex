#!/bin/bash
#SBATCH --job-name=conn
#SBATCH --output=slurm_logs/conn_%j.out
#SBATCH --error=slurm_logs/conn_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/


# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.12.3-GCCcore-13.3.0

# Activate it
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

# python pathway_connection_analysis.py \
#     --wdir multixrank_input_consolidated \
#     --lr-csv ./pathway_distance/human_lr_pairs.csv \
#     --output-dir connectivity_output

python pathway_connection_analysis.py \
  --wdir multixrank_input_consolidated \
  --reactome-csv ./pathway_distance/reactome_interactions_with_pathways_v2.csv \
  --uniprot-symbol-csv ../og_gene_set/ppi_chloe/d_uniprot_to_symbol_20251120.csv \
  --lr-csv ./pathway_distance/human_lr_pairs.csv \
  --exclude-ai "immune tissue" \
  --csv layers_categories.csv
  
# python consolidate_layers.py \
#     --wdir multixrank_input_consolidated \
#     --output-wdir multixrank_input_consolidated \
#     --exclude-ai    "immune tissue" 
  
# End time
echo "Finished at: $(date)"