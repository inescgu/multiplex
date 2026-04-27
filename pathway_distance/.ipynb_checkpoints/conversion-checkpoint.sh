#!/bin/bash
#SBATCH --job-name=conv
#SBATCH --output=slurm_logs/conv_%j.out
#SBATCH --error=slurm_logs/conv_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/og_gene_set/pathway_distance/

# Create necessary directories
mkdir -p slurm_logs
mkdir -p output

# Load modules
module load python3/default

# Run conversion
python uniprot_to_symbols.py kegg_signaling_pathways_with_names.csv \
  --output kegg_signaling_pathways_with_names_symbols.csv \
  --rate-limit 0.7

echo "Finished at: $(date)"