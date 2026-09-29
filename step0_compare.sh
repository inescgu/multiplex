#!/bin/bash
#SBATCH --job-name=step0_compare
#SBATCH --output=slurm_logs/step0_compare_%j.out
#SBATCH --error=slurm_logs/step0_compare_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
echo "Step 0b: Running NeuronChat (R)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

python step0_compare_multiplex_layers.py \
    --old-dir 02_hdrg_multiplex/multiplex_2025hdrg \
    --new-dir 02_hdrg_multiplex/multiplex_2025hdrg_filtered \
    --output-dir 02_hdrg_multiplex/layer_comparison
echo "Finished at: $(date)"