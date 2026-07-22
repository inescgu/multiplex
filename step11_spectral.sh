#!/bin/bash
#SBATCH --job-name=step11_spectral
#SBATCH --output=slurm_logs/step11_spectral_%j.out
#SBATCH --error=slurm_logs/step11_spectral_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT  # Get warnings

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 11: Spectral clustering"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate
pip install scikit-learn
mkdir -p 08_spectral_analysis

python step11_multiplex_spectral_clustering.py \
    --graph 03_integrated/integrated_multiplex_output_LR/integrated_multiplex.pkl \
    --registry 03_integrated/integrated_multiplex_output_LR/node_registry.csv \
    --out-dir 08_spectral_analysis \
    --mode sos \
    --sample-p 0.4 \
    --K 12 \
    --k-min 4 \
    --k-max 16 \
    --seed 42

echo "Finished at: $(date)"