#!/bin/bash
#SBATCH --job-name=step12f_agg
#SBATCH --output=slurm_logs/step12f_agg_%j.out
#SBATCH --error=slurm_logs/step12f_agg_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/

echo "Step 12f aggregate"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"
module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate

python step12f_aggregate_matched.py --output-dir 12_hub_removal
echo "Finished at: $(date)"