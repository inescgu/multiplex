#!/bin/bash
#SBATCH --job-name=step3_config_seeds
#SBATCH --output=slurm_logs/step3_config_seeds_%j.out
#SBATCH --error=slurm_logs/step3_config_seeds_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 3: Writing config.yml, filtering seeds, building reference graphs"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

# Verify prerequisites
if [ ! -f 04_multixrank_prep/multixrank_hetero_input_pseudobulk/bipartite_manifest.json ]; then
    echo "ERROR: bipartite_manifest.json not found! Run step 2 first."
    exit 1
fi

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step3_build_config_seeds.py \
    --wdir 04_multixrank_prep/multixrank_hetero_input_pseudobulk \
    --seed-csv 00_inputs/seed_genes.csv \
    --r 0.7 \
    --self-loops 0
#   --lamb-json 00_inputs/lamb_overrides.json   # uncomment to bias specific
#                                                # cell-type-to-cell-type jumps

echo "Finished at: $(date)"