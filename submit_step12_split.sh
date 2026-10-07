#!/bin/bash
# Driver -- run with:  bash submit_step12_split.sh [N_NULL=50]
# (NOT with sbatch: it calls sbatch itself on the three *.sh job files.)
set -euo pipefail
cd /lisc/home/user/gerardursin/multiplex/
mkdir -p slurm_logs

N=${1:-50}

ARR=$(sbatch --parsable --array=0-$((N-1))%10 step12b_null.sh)
echo "null array (B):  $ARR"

TAIL=$(sbatch --parsable step12d_tail.sh)
echo "tail (D):        $TAIL"

AGG=$(sbatch --parsable --dependency=afterany:$ARR --export=ALL,N_NULL=$N step12c_aggregate.sh)
echo "aggregate (C):   $AGG (runs after array $ARR ends)"