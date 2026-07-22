#!/bin/bash
#SBATCH --job-name=step9b
#SBATCH --output=slurm_logs/step9b_%j.out
#SBATCH --error=slurm_logs/step9b_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=35G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 9: Investigating Disease Module Structurecd "
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step9b_module_diagnostics.py \
      --output-dir 06_disease_module/disease_module_output_consolidated \
      --input-dir  04_multixrank_prep/multixrank_input_consolidated

echo "Finished at: $(date)"
