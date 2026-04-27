#!/bin/bash
#SBATCH --job-name=venn
#SBATCH --output=slurm_logs/venn_%j.out
#SBATCH --error=slurm_logs/venn_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=15G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

mkdir -p comparison_results

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0


# Run the multiplex interactome construction
python venn_multiplex_comparison.py \
  --tabula-layers ./new_ts/layers \
  --pain-layers ./multiplex_2025hdrg/layers \
  --pain-subset-layers ./subset_analysis/layers \
  --output-dir comparison_results

# End time
echo "Finished at: $(date)"