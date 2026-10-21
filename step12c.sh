#!/bin/bash
#SBATCH --job-name=plothub_removal
#SBATCH --output=slurm_logs/plot_hub_removal_%j.out
#SBATCH --error=slurm_logs/plot_hub_removal_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/


echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate

HUB_REMOVAL_DIR=12_hub_removal/hub_removal
PLOT_DIR=12_hub_removal/plots

mkdir -p "${PLOT_DIR}"


python step12c_plothubremoval.py \
    ${HUB_REMOVAL_DIR} \
    ${PLOT_DIR}

echo "Finished at: $(date)"