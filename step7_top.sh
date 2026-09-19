#!/bin/bash
#SBATCH --job-name=step7_topology
#SBATCH --output=slurm_logs/step7_topology_%j.out
#SBATCH --error=slurm_logs/step7_topology_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=15G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 7: Per-cell-type topology statistics"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

mkdir -p 07_analysis/topology_output

python step7_layer_topology_stats.py \
    --wdir 04_multixrank_prep/multixrank_hetero_input \
    --csv layers_categories.csv \
    --output-dir 07_analysis/topology_output \
    --top-n-eigen 5 \
    --exclude-ai "immune tissue" "immune" "nervous" \
    --min-edges 1
echo "Finished at: $(date)"