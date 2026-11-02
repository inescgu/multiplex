#!/bin/bash
#SBATCH --job-name=step5_preflight
#SBATCH --output=slurm_logs/step5_preflight_%j.out
#SBATCH --error=slurm_logs/step5_preflight_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail

echo "Step 5: Preflight seed reachability check"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step5_preflight_seed_reachability.py \
    --ref-graph 04_multixrank_prep/multixrank_hetero_input_pseudobulk/gene_reference_graph.pkl \
    --seeds 04_multixrank_prep/multixrank_hetero_input_pseudobulk/seeds/seeds.txt \
    --lcc-per-layer \
    --exclude-ppi-seeds \
    --write-excluded 04_multixrank_prep/multixrank_hetero_input_pseudobulk/seeds/excluded_seeds.txt

echo "Finished at: $(date)"