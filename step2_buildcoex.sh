#!/bin/bash
#SBATCH --job-name=step2_hdrg
#SBATCH --output=slurm_logs/step2_hdrg_%j.out
#SBATCH --error=slurm_logs/step2_hdrg_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 2: Building HDRG multiplex"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.13.5-GCCcore-14.3.0
source /lisc/data/scratch/menche/ines/multi_env/bin/activate

python step2_build_25hdrg_multiplex.py \
    --markers-file 00_inputs/hdrg_markers.csv \
    --base-ppi 00_inputs/ppi_network.pkl \
    --output-dir 02_hdrg_multiplex/multiplex_2025hdrg_filtered \
    --seed-genes-file 00_inputs/seed_genes.csv

echo "Finished at: $(date)"