#!/bin/bash
#SBATCH --job-name=step7d_seeds
#SBATCH --output=slurm_logs/step7d_seeds_%j.out
#SBATCH --error=slurm_logs/step7d_seeds_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 7d: Seed profiles - can by run anytime after step 1"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

 
python step7d_seed_layer_profile.py \
    --wdir 04_multixrank_prep/multixrank_hetero_input \
    --output-dir 07d_seed_layer_profile \
    --markers-csv ../R/convert_hdrg_human_2025/cell_type_markers.csv
 
echo "Finished at: $(date)"
 