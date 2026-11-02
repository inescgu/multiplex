#!/bin/bash
#SBATCH --job-name=step11h
#SBATCH --output=slurm_logs/step11h_%j.out
#SBATCH --error=slurm_logs/step11h_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail
source ./run_env.sh

echo "Step 11h: Persistence tier analysis"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

snapshot "$RUN_DIR/persistence_tiers"

python step11h_persistence_tier_analysis.py \
    --cluster-summary "$RUN_DIR/characterized/heterogeneous_hidef_faithful_cpm_cluster_summary.csv" \
    --n-partitions 200 \
    --persistence-basis percentile \
    --tier-percentiles 90 75 50 \
    --output-dir "$RUN_DIR/persistence_tiers"
echo "Finished at: $(date)"