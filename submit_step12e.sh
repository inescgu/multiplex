#!/bin/bash
# Submits all four variants as arrays, then the aggregation job after they all finish.
# Test timing first with:  sbatch -J step12e_test --array=0 --export=ALL,VARIANT=layer step12e_matched.sh
mkdir -p slurm_logs

submit() {  # $1=variant  $2=array range
    sbatch --parsable -J step12e_$1 --array=$2 --export=ALL,VARIANT=$1 step12e_matched.sh
}

J_OBS=$(submit observed   0-9)
J_NAI=$(submit naive      0-49%10)
J_LAY=$(submit layer      0-49%10)
J_SED=$(submit layer_seed 0-49%10)
echo "Submitted: observed=$J_OBS naive=$J_NAI layer=$J_LAY layer_seed=$J_SED"

# afterany (not afterok) so a single failed replicate doesn't block aggregation;
# the aggregator reports how many reps of each variant it found.
J_AGG=$(sbatch --parsable --dependency=afterany:${J_OBS}:${J_NAI}:${J_LAY}:${J_SED} step12f_aggregate.sh)
echo "Aggregation job $J_AGG will run after all arrays end"