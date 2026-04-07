#!/bin/bash
#SBATCH --job-name=comprwr
#SBATCH --output=slurm_logs/com_rwr_%j.out
#SBATCH --error=slurm_logs/com_rwr_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p comparison_rwr_ranking

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0

# compares the rwr rankings
python compare_multiplex_rankings.py \
  --ss-dir ./rwr_ss_results \
  --ts-dir ./rwr_ts_results \
  --atlas-dir ./rwr_2025_results \
  --output-dir comparison_rwr_ranking \
  --k 200

# End time
echo "Finished at: $(date)"