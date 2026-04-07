#!/bin/bash
#SBATCH --job-name=comviz
#SBATCH --output=slurm_logs/com_vmulti_%j.out
#SBATCH --error=slurm_logs/com_vmulti_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=5G
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


# Run the multiplex interactome construction
python viz_multi_comp.py \
  --tabula-layers ./multiplex_ts/layers \
  --pain-layers ./multiplex_2025hdrg/layers \
  --pain-subset-layers ./subset_analysis/layers \
  --seed-genes ../og_gene_set/seed_genes.csv \
  --output-dir comparison_results

# End time
echo "Finished at: $(date)"