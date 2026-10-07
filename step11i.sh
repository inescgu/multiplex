#!/bin/bash
#SBATCH --job-name=step11i
#SBATCH --output=slurm_logs/step11i_%j.out
#SBATCH --error=slurm_logs/step11i_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/
source ./run_env.sh

: "${STAGE:?STAGE must be set (true|null|loo|aggregate)}"
: "${OUT:?OUT must be set}"

echo "Step 11i [$STAGE]  host=$(hostname)  job=${SLURM_JOB_ID}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
set -eo pipefail

TASK=${SLURM_ARRAY_TASK_ID:-0}
TASK_DIR=$(printf "task_%03d" "$TASK")

COMMON=(
  --reference-graph 04_multixrank_prep/multixrank_hetero_input/gene_reference_graph.pkl
  --disease-module-dir 06_disease_module
  --partition-type CPM
  --gamma-mode calibrated
  --max-resolution 0.9 --n-resolutions 40 --n-restarts 5
  --minsize "$MINSIZE" --jaccard "$JACCARD" --chi "$CHI" --consensus-p "$CONSENSUS_P"
  --neighbors "$NEIGHBORS"
  --cap-factor 1.5 --min-lcc-factor 0.5
  --perm-timeout "${PERM_TIMEOUT:-0}"
)

case "$STAGE" in
  true)
    set +e
    guard "$OUT";    echo "guard rc=$?"
    snapshot "$OUT"; echo "snapshot rc=$?"
    set -e
    mkdir -p "$OUT/true"
    python -u step11i_cluster_level_null_model.py --stage true \
      "${COMMON[@]}" --output-dir "$OUT/true"
    ;;
  null)
    python -u step11i_cluster_level_null_model.py --stage null \
      --task-id "$TASK" --n-tasks "$N_TASKS" --n-perms "$N_PERMS" \
      "${COMMON[@]}" --output-dir "$OUT/null/$TASK_DIR"
    ;;
  loo)
    python -u step11i_cluster_level_null_model.py --stage loo \
      --task-id "$TASK" --n-tasks "$N_TASKS" --true-dir "$OUT/true" \
      "${COMMON[@]}" --output-dir "$OUT/loo/$TASK_DIR"
    ;;
  aggregate)
    python -u step11i_aggregate.py --root "$OUT" --output-dir "$OUT/aggregate" \
      --expected-perms "${EXPECTED_PERMS:-0}"
    ;;
  *)
    echo "Unknown STAGE=$STAGE"; exit 2 ;;
esac

echo "Finished at: $(date)"
