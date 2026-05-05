#!/bin/bash
#SBATCH --job-name=buildcoex
#SBATCH --output=slurm_logs/bcoex_%j.out
#SBATCH --error=slurm_logs/bcoex_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p multiplex_2025hdrg_coex


# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0
source /lisc/data/scratch/menche/ines/multi_env/bin/activate

pip install --user pandas numpy networkx pickly scipy 


# Run the multiplex interactome construction
python build_25hdrg_multiplex.py \
    --markers-file ../R/convert_hdrg_human_2025/cell_type_markers.csv \
    --base-ppi ../og_gene_set/temp/combined26/full_network.pkl \
    --output-dir multiplex_2025hdrg \
    --seed-genes-file ../og_gene_set/seed_genes.csv

# End time
echo "Finished at: $(date)"