#!/bin/bash
#SBATCH --job-name=vizedge
#SBATCH --output=slurm_logs/vizedge_%j.out
#SBATCH --error=slurm_logs/vizedge_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=45G
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

pip install upsetplot matplotlib seaborn matplotlib-venn


# Run the multiplex interactome construction
python visualize_edge_overlaps.py \
  --multiplex-pkl ./integrated_multiplex_output/integrated_multiplex.pkl \
  --registry-csv ./integrated_multiplex_output/node_registry.csv \
  --output-dir plots
  
# End time
echo "Finished at: $(date)"