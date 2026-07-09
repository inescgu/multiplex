#!/bin/bash
#SBATCH --job-name=step2b_pathways
#SBATCH --output=slurm_logs/step2b_pathways_%j.out
#SBATCH --error=slurm_logs/step2b_pathways_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 2b: Preparing pathway and LR reference data"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python pathway_prep/build_pathway_data.py

echo "Finished at: $(date)"