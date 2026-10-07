#!/bin/bash
#SBATCH --job-name=step12c_agg
#SBATCH --output=slurm_logs/step12c_agg_%j.out
#SBATCH --error=slurm_logs/step12c_agg_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/

echo "Step 12c: aggregate null replicates"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"
module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate

python step12c_aggregate_null.py \
    --output-dir 12_hub_removal \
    --expected ${N_NULL:-50}
echo "Finished at: $(date)"