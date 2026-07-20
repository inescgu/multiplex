#!/bin/bash
#SBATCH --job-name=step4_mxr_inputs
#SBATCH --output=slurm_logs/step4_mxr_inputs_%j.out
#SBATCH --error=slurm_logs/step4_mxr_inputs_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=90G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 4: Building MultiXrank inputs"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step4_build_multixrank_inputs.py \
    --multiplex-pkl 03_integrated/integrated_multiplex_output_LR/integrated_multiplex.pkl \
    --seed-csv 00_inputs/seed_genes.csv \
    --output-dir 04_multixrank_prep/multixrank_input \
    --restart 0.7 \
    --delta 0.5

echo "Finished at: $(date)"