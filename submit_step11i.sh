#!/bin/bash
# Usage (run from the multiplex/ directory on the login node):
#   bash submit_step11i.sh pilot                       # 1 task x 3 perms, to time MultiXrank + check the builder
#   bash submit_step11i.sh full                        # 20 tasks x 10 perms = 200 nulls
#   N_TASKS=25 N_PERMS=20 TIME=08:00:00 bash submit_step11i.sh full     # 500 nulls
#   RUN_LOO=1 N_LOO_TASKS=10 bash submit_step11i.sh full
#
# Overridable env vars: OUT, MEM, TIME, N_TASKS, N_PERMS, MAX_CONC, RUN_LOO,
#                       N_LOO_TASKS, LOO_TIME, PERM_TIMEOUT, MX_WDIR, MX_CONFIG
# Total nulls = N_TASKS x N_PERMS.  Always use a fresh OUT when you change N_PERMS.
set -euo pipefail

MODE=${1:-pilot}
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p slurm_logs

case "$MODE" in
  pilot)
    OUT=${OUT:-results/11i_null/pilot_mx_$STAMP}
    N_TASKS=1; N_PERMS=3
    MEM=${MEM:-32G}; TIME=${TIME:-03:00:00}
    RUN_LOO=0
    ;;
  full)
    N_TASKS=${N_TASKS:-20}; N_PERMS=${N_PERMS:-10}
    OUT=${OUT:-results/11i_null/nperms$((N_TASKS * N_PERMS))_mx_$STAMP}
    MEM=${MEM:-32G}; TIME=${TIME:-06:00:00}   # replace with pilot-derived values
    RUN_LOO=${RUN_LOO:-0}
    ;;
  *) echo "mode must be pilot|full"; exit 1 ;;
esac
MAX_CONC=${MAX_CONC:-20}
N_LOO_TASKS=${N_LOO_TASKS:-10}
LOO_TIME=${LOO_TIME:-$TIME}
EXPECTED=$((N_TASKS * N_PERMS))

echo "Mode=$MODE OUT=$OUT tasks=$N_TASKS perms/task=$N_PERMS (total $EXPECTED) mem=$MEM time=$TIME"

# 1) true-module stage (also does guard + snapshot of $OUT)
J_TRUE=$(sbatch --parsable --job-name=11i_true --time=01:30:00 --mem="$MEM" \
  --output=slurm_logs/11i_true_%j.out --error=slurm_logs/11i_true_%j.err \
  --export=ALL,STAGE=true,OUT="$OUT" step11i.sh)
echo "true       : $J_TRUE"

# 2) null array (waits for the true stage so guard/snapshot see a clean $OUT)
J_NULL=$(sbatch --parsable --job-name=11i_null --time="$TIME" --mem="$MEM" \
  --array=0-$((N_TASKS - 1))%"$MAX_CONC" --dependency=afterok:"$J_TRUE" \
  --output=slurm_logs/11i_null_%A_%a.out --error=slurm_logs/11i_null_%A_%a.err \
  --export=ALL,STAGE=null,OUT="$OUT",N_TASKS="$N_TASKS",N_PERMS="$N_PERMS" step11i.sh)
echo "null array : $J_NULL"
DEPS="afterany:$J_NULL"

# 3) optional leave-one-seed-out array
if [ "$RUN_LOO" = "1" ]; then
  J_LOO=$(sbatch --parsable --job-name=11i_loo --time="$LOO_TIME" --mem="$MEM" \
    --array=0-$((N_LOO_TASKS - 1))%"$MAX_CONC" --dependency=afterok:"$J_TRUE" \
    --output=slurm_logs/11i_loo_%A_%a.out --error=slurm_logs/11i_loo_%A_%a.err \
    --export=ALL,STAGE=loo,OUT="$OUT",N_TASKS="$N_LOO_TASKS" step11i.sh)
  echo "loo array  : $J_LOO"
  DEPS="$DEPS:$J_LOO"
fi

# 4) aggregate (runs even if some tasks failed, so partial results are summarised)
J_AGG=$(sbatch --parsable --job-name=11i_agg --time=00:30:00 --mem=8G \
  --dependency="$DEPS" \
  --output=slurm_logs/11i_agg_%j.out --error=slurm_logs/11i_agg_%j.err \
  --export=ALL,STAGE=aggregate,OUT="$OUT",EXPECTED_PERMS="$EXPECTED" step11i.sh)
echo "aggregate  : $J_AGG"
echo
echo "Watch:   squeue -u \$USER ; tail -f slurm_logs/11i_null_${J_NULL}_0.out"
echo "Results: $OUT/aggregate/   (budget suggestion is printed in slurm_logs/11i_agg_${J_AGG}.out)"