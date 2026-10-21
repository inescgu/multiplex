#!/bin/bash
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# Job name and array range are set by submit_step12e.sh (-J step12e_<variant>, --array=...).
# VARIANT must be passed in: observed | naive | layer | layer_seed
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail

if [ -z "${VARIANT}" ]; then echo "ERROR: VARIANT not set"; exit 1; fi

echo "Step 12e ${VARIANT} replicate ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"
module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate

python step12e_matched_replicate.py \
    --module-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
    --disease-module 06_disease_module/disease_module_output_pseudobulk/disease_module.csv \
    --output-dir 12_hub_removal \
    --variant ${VARIANT} \
    --step 10 \
    --cascading-step 10 \
    --betweenness-k 200 \
    --n-cascading 15
echo "Finished at: $(date)"