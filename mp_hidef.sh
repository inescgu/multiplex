#!/bin/bash
#SBATCH --job-name=multiplex_cluster
#SBATCH --output=slurm_logs/multiplex_cluster_%j.out
#SBATCH --error=slurm_logs/multiplex_cluster_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

mkdir -p slurm_logs
mkdir -p output/hidef
mkdir -p plots/hidef

echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load python3/default

# Install python-louvain if not already installed
pip install --user python-louvain scikit-learn

# Run HiDeF clustering on multiplex disease module
# Note: This uses the disease module from multiplex RWR
python mp_hidef.py \
    --multiplex-dir multiplex_interactome \
    --disease-module output/multiplex_disease_module.csv \
    --output-dir output/hidef \
    --plots-dir plots/hidef \
    --min-resolution 0.1 \
    --max-resolution 2.0 \
    --resolution-steps 20 \
    --min-persistence 5

echo "Finished at: $(date)"