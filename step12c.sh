#!/bin/bash
#SBATCH --job-name=plothub_removal
#SBATCH --output=slurm_logs/plot_hub_removal_%j.out
#SBATCH --error=slurm_logs/plot_hub_removal_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/
mkdir -p slurm_logs

PPI=combined26

echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate

# Runs after hub_removal_analysis.py has produced:
#   output/${PPI}/hub_removal/observed_vs_null_comparison.csv
INPUT_CSV=12_hub_removal/hub_removal/observed_vs_null_comparison.csv
PLOT_DIR=12_hub_removal/hub_removal/


python step12c_plothubremoval.py \
    ${INPUT_CSV} \
    ${PLOT_DIR}

echo "Finished at: $(date)"