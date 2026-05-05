#!/bin/bash
#SBATCH --job-name=big
#SBATCH --output=slurm_logs/big_%j.out
#SBATCH --error=slurm_logs/big_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/


# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0

# Activate it
source /lisc/data/scratch/menche/ines/multi_env/bin/activate

# Run the multiplex interactome construction
python build_master_multiplex.py \
  --ts-layers-dir   ./new_ts/layers \
  --hrdg-layers-dir ./multiplex_2025hdrg/layers \
  --reactome-csv    ./pathway_distance/reactome_interactions_with_pathways.csv \
  --lr-csv          ./pathway_distance/lr_pairs.csv \
  --output-dir      integrated_multiplex_output\
  --uniprot-map ../og_gene_set/ppi_chloe/d_uniprot_to_symbol_20251120.csv \
  --max-pathway-size   300 \
  --pathway-chunk-size 50 \
  
# End time
echo "Finished at: $(date)"