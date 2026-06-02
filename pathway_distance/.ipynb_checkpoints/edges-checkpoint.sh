#!/bin/bash
#SBATCH --job-name=edge
#SBATCH --output=slurm_logs/edge_%j.out
#SBATCH --error=slurm_logs/edge_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/pathway_distance/


# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0

# Activate it
source /lisc/data/scratch/menche/ines/multi_env/bin/activate

# understand edges
python edge_audit.py \
  --reactome-csv  reactome_interactions_with_pathways.csv \
  --lr-csv        lr_pairs.csv \
  --kegg-csv      kegg_pathways_with_names_symbols.csv \
  --layers-dir    layers_uniprot/ \
  --uniprot-map   ../../og_gene_set/ppi_chloe/d_uniprot_to_symbol_20251120.csv \
  --output-dir    audit_output/