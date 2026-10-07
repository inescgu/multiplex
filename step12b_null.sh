#!/bin/bash
#SBATCH --job-name=step12b_null
#SBATCH --output=slurm_logs/step12b_null_%A_%a.out
#SBATCH --error=slurm_logs/step12b_null_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# Array range is set by submit_step12_split.sh (--array=0-N-1%10)
cd /lisc/home/user/gerardursin/multiplex/

echo "Step 12b null replicate ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"
module load Python/3.14.4-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate

python step12b_null_replicate.py \
    --module-graph 06_disease_module/disease_module_output/module_graph.pkl \
    --output-dir 12_hub_removal \
    --step 10 \
    --cascading-step 10 \
    --betweenness-k 200 \
    --n-null-cascading 15 \
    --cascading-modes degree,betweenness,degree_betweenness
echo "Finished at: $(date)"