#!/bin/bash
# Usage (login node, from multiplex/):  bash submit_partA.sh
# Submits Part A (multiplex build -> disease module -> layer analyses, steps 1-7)
# with afterok dependencies, and records the job IDs Part B depends on in
# slurm_logs/partA_jobs.txt for submit_partB.sh. Step 7b (~17 h, nothing in
# Part B reads it) is submitted but not recorded.
# Step 5 writes the unreachable seeds to <wdir>/seeds/excluded_seeds.txt and
# step 6 (and later 10, 11i) read them, so the chain needs no manual pause.
#
# Restart partway through the chain, keeping the earlier outputs:
#   FROM=step5 bash submit_partA.sh      # reruns 5 -> 4 -> 6 -> 6b..7dn (not 7b)
#   FROM=step6 bash submit_partA.sh      # reruns 6 -> 6b..7dn (not 7b)
# Run with bash, not sbatch: this script calls sbatch itself.
set -euo pipefail
cd /lisc/home/user/gerardursin/multiplex/

CHAIN=(step2_buildcoex step1 step2 step3 step3b step5 step4 step6)
AFTER6=(step6b step6c step6d step6e step7_top step7c step7d step7dn)
FROM=${FROM:-${CHAIN[0]}}
[[ " ${CHAIN[*]} " == *" $FROM "* ]] || { echo "FROM must be one of: ${CHAIN[*]}"; exit 1; }

WDIR=04_multixrank_prep/multixrank_hetero_input_pseudobulk
if [ "$FROM" = "${CHAIN[0]}" ] && [ -e "$WDIR" ]; then
    echo "$WDIR exists -- move it aside first so step 1 starts from an empty folder"; exit 1
fi

mkdir -p slurm_logs
sub() { local dep=$1; shift; sbatch --parsable ${dep:+--dependency=afterok:$dep} "$@"; }
IDS=()
add() { IDS+=("$1"); echo "$1 $2"; }

J=""; ON=0
for s in "${CHAIN[@]}"; do
    [ "$s" = "$FROM" ] && ON=1
    [ "$ON" -eq 1 ] || continue
    J=$(sub "$J" $s.sh); add $J $s
done
J6=$J
for s in "${AFTER6[@]}"; do add "$(sub $J6 $s.sh)" $s; done
if [ "$FROM" = "${CHAIN[0]}" ]; then J7B=$(sub $J6 step7b.sh); echo "$J7B step7b (not recorded)"; fi

printf "%s\n" "${IDS[@]}" > slurm_logs/partA_jobs.txt
echo "$FROM" > slurm_logs/partA_from.txt   # submit_partB.sh only checks freshness of what this run rebuilt
echo
echo "Part A submitted from $FROM (${#IDS[@]} jobs recorded in slurm_logs/partA_jobs.txt)."
echo "When they're done:  bash submit_partB.sh"
