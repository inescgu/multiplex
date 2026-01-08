#!/bin/bash
#SBATCH --job-name=convert_orthologs
#SBATCH --output=slurm_logs/convert_%j.out
#SBATCH --error=slurm_logs/convert_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

# Create necessary directories
mkdir -p converted_expression
mkdir -p conversion_report
mkdir -p slurm_logs

# Print job information
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.13.5-GCCcore-14.3.0
pip install --user pandas numpy mygene

# Run the conversion
python convert_orthologs.py
#python test.py 

# End time
echo "Finished at: $(date)"