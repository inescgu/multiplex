#!/bin/bash
#SBATCH --job-name=multiplex_lcc
#SBATCH --output=slurm_logs/multiplex_lcc_%j.out
#SBATCH --error=slurm_logs/multiplex_lcc_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

mkdir -p slurm_logs
mkdir -p output/lcc

echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load python3/default

python mp_lcc.py

echo "Finished at: $(date)"