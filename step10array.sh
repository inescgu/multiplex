#!/bin/bash
#SBATCH --job-name=step10_perm
#SBATCH --output=slurm_logs/step10_perm_%A_%a.out
#SBATCH --error=slurm_logs/step10_perm_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-9

cd /lisc/home/user/gerardursin/multiplex/
echo "Step 10: Permutation test, array task ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

OUTDIR=06_disease_module/perm_array/task_${SLURM_ARRAY_TASK_ID}
mkdir -p "$OUTDIR"

python step10_permutation_test_multiplex.py \
    --disease-module-dir 06_disease_module/disease_module_output \
    --reference-graph 04_multixrank_prep/multixrank_hetero_input/gene_reference_graph.pkl \
    --lcc-per-layer \
    --restart 0.7 \
    --scaling 0 \
    --k 200 \
    --n-perms 100 \
    --bin-size 5 \
    --seed $((42 + SLURM_ARRAY_TASK_ID)) \
    --output-dir "$OUTDIR"
echo "Finished at: $(date)"