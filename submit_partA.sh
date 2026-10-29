#!/bin/bash
# Usage (login node, from multiplex/):  bash submit_partA.sh
# Submits Part A (multiplex build -> disease module -> layer analyses, steps 1-7)
# with afterok dependencies, and records the job IDs Part B depends on in
# slurm_logs/partA_jobs.txt for submit_partB.sh. Step 7b (~17 h, nothing in
# Part B reads it) is submitted but not recorded.
# Run with bash, not sbatch: this script calls sbatch itself.
set -euo pipefail
cd /lisc/home/user/gerardursin/multiplex/

WDIR=04_multixrank_prep/multixrank_hetero_input_pseudobulk
if [ -e "$WDIR" ]; then
    echo "$WDIR exists -- move it aside first so step 1 starts from an empty folder"; exit 1
fi

mkdir -p slurm_logs
sub() { local dep=$1; shift; sbatch --parsable ${dep:+--dependency=afterok:$dep} "$@"; }
IDS=()
add() { IDS+=("$1"); echo "$1"; }

J=$(sub ""  step2_buildcoex.sh); add $J
J=$(sub $J  step1.sh);           add $J
J=$(sub $J  step2.sh);           add $J
J=$(sub $J  step3.sh);           add $J
J=$(sub $J  step3b.sh);          add $J
J=$(sub $J  step5.sh);           add $J
J=$(sub $J  step4.sh);           add $J
J6=$(sub $J step6.sh);           add $J6
for s in step6b step6c step6d step6e step7_top step7c step7d step7dn; do add "$(sub $J6 $s.sh)"; done
J7B=$(sub $J6 step7b.sh)

printf "%s\n" "${IDS[@]}" > slurm_logs/partA_jobs.txt
echo
echo "Part A submitted (${#IDS[@]} jobs recorded in slurm_logs/partA_jobs.txt; step7b = $J7B, not recorded)."
echo "When they're done:  bash submit_partB.sh"
