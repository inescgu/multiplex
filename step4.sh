#!/bin/bash
#SBATCH --job-name=step4_multixrank
#SBATCH --output=slurm_logs/step4_multixrank_%j.out
#SBATCH --error=slurm_logs/step4_multixrank_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 4: Running MultiXrank (heterogeneous multiplex + bipartite)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step4_run_multixrank.py \
    --wdir 04_multixrank_prep/multixrank_hetero_input \
    --config config.yml \
    --output-dir 05_multixrank_results/multixrank_hetero_results \
    --aggregation max

echo "Finished at: $(date)"