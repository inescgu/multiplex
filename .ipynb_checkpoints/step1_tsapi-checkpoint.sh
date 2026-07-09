#!/bin/bash
#SBATCH --job-name=step1_tsapi
#SBATCH --output=slurm_logs/step1_tsapi_%j.out
#SBATCH --error=slurm_logs/step1_tsapi_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 1: Building Tabula Sapiens multiplex"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

pip install scipy tiledbsoma cellxgene-census

python step1_ts_api.py \
    --ppi 00_inputs/ppi_network.pkl \
    --output-dir 01_ts_multiplex/new_ts \
    --seed-genes 00_inputs/seed_genes.csv \
    --expression-threshold 0.1 \
    --census-version "2025-11-08"

echo "Finished at: $(date)"