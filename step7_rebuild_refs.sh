#!/bin/bash
#SBATCH --job-name=step7_rebuild
#SBATCH --output=slurm_logs/step7_rebuild_%j.out
#SBATCH --error=slurm_logs/step7_rebuild_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 7: Rebuilding reference graphs"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step7_rebuild_reference_graphs.py \
    --wdir 04_multixrank_prep/multixrank_input_consolidated

echo "Finished at: $(date)"