#!/bin/bash
#SBATCH --job-name=step1_celltype_ppi
#SBATCH --output=slurm_logs/step1_celltype_ppi_%j.out
#SBATCH --error=slurm_logs/step1_celltype_ppi_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 1: Building per-cell-type PPI multiplex layers (heterogeneous structure)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step1_build_celltype_ppi.py \
    --hdrg-layers-dir 02_hdrg_multiplex/multiplex_2025hdrg_filtered/layers \
    --output-wdir 04_multixrank_prep/multixrank_hetero_input \
    --layers-csv neurons_only.csv \
    --merge-by none \
    --base-ppi 00_inputs/ppi_network.pkl   # uncomment to add a fully-connected
                                          # "global_ppi" pseudo-cell-type multiplex

echo "Finished at: $(date)"