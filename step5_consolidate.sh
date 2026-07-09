#!/bin/bash
#SBATCH --job-name=step5_consol
#SBATCH --output=slurm_logs/step5_consol_%j.out
#SBATCH --error=slurm_logs/step5_consol_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 5: Consolidating layers"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

# ✓ Verify prerequisites
if [ ! -f 04_multixrank_prep/multixrank_input/seeds/seeds.txt ]; then
    echo "ERROR: seeds.txt not found! Run step 4 first."
    exit 1
fi

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step5_consolidate_layers.py \
    --wdir 04_multixrank_prep/multixrank_input \
    --output-wdir 04_multixrank_prep/multixrank_input_consolidated \
    --csv neurons_only.csv \
    --merge-by none

echo "Finished at: $(date)"