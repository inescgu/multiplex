#!/bin/bash
#SBATCH --job-name=step12_mp
#SBATCH --output=slurm_logs/step12_mp_%j.out
#SBATCH --error=slurm_logs/step12_mp_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 12: Marchenko-Pastur analysis"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

mkdir -p 08_marchenko_pastur

python step12_multiplex_marchenko_pastur.py \
    --graph 03_integrated/integrated_multiplex_output/integrated_multiplex.pkl \
    --registry 03_integrated/integrated_multiplex_output/node_registry.csv \
    --out-dir 08_marchenko_pastur \
    --sample-p 0.4 \
    --omega 1.0 \
    --n-eigs 400 \
    --min-neg-layers 2 \
    --seed 42

echo "Finished at: $(date)"
