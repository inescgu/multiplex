#!/bin/bash
#SBATCH --job-name=step14_perm
#SBATCH --output=slurm_logs/step14_perm_%j.out
#SBATCH --error=slurm_logs/step14_perm_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 14: Permutation test for multiplex disease module"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

mkdir -p slurm_logs
mkdir -p 06_disease_module/permutation_test

python step14_permutation_test_multiplex.py \
    --disease-module-dir 06_disease_module \
    --restart 0.7 \
    --scaling 0 \
    --k 200 \
    --n-perms 1000 \
    --bin-size 5 \
    --seed 42 \
    --output-dir 06_disease_module

echo "Finished at: $(date)"