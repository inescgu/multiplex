#!/bin/bash
#SBATCH --job-name=pathway_dist
#SBATCH --output=slurm_logs/pathway_dist_%j.out
#SBATCH --error=slurm_logs/pathway_dist_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/og_gene_set/pathway_distance/

# Create necessary directories
mkdir -p slurm_logs
mkdir -p pathway_output

# Load modules
module load python3/default

# Run pathway distance analysis
python pathway_distance.py \
    --module ../output/combined/rwr/disease_module.pkl \
    --graph ../output/combined/rwr/module_graph.pkl \
    --pathways kegg_pathways_with_names_symbols.csv \
    --gene-id-column gene_symbol \
    --clusters ../output/combined/disease_module_hidef/disease_communities.csv \
    --output-dir pathway_output

echo "Finished at: $(date)"