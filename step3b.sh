#!/bin/bash
#SBATCH --job-name=step3b
#SBATCH --output=slurm_logs/step3b_%j.out
#SBATCH --error=slurm_logs/step3b_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail

echo "Step 3b: Building LR (+ Reactome) bipartite edges between cell types,"
echo "        plus identity bipartite edges connecting global_ppi to everything"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step3b_namespace_nodes.py \
    --wdir 04_multixrank_prep/multixrank_hetero_input_pseudobulk \
    --exclude-ppi-seeds
echo "Finished at: $(date)"