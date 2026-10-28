#!/bin/bash
# Usage (login node, from multiplex/):  bash submit_partB.sh
# 1. Checks that every Part A job (steps 1-7) COMPLETED, has no Traceback in
#    its log, and that the files Part B reads exist. Refuses to submit if not.
# 2. Submits Part B (steps 8-12) with afterok dependencies inside each branch.
# Part A job IDs: override with PART_A="first last" if you rerun Part A.
# SKIP_IDS="id ..." leaves out Part A jobs nothing in Part B reads (e.g. step7b, ~17 h).
# Run with bash, not sbatch: this script calls sbatch itself.
set -euo pipefail
cd /lisc/home/user/gerardursin/multiplex/

read -r FIRST LAST <<< "${PART_A:-7521343 7521359}"
SKIP=" ${SKIP_IDS:-} "
CHECK=$(for id in $(seq "$FIRST" "$LAST"); do [[ "$SKIP" == *" $id "* ]] || echo "$id"; done)
IDS=$(echo $CHECK | tr ' ' ,)

echo "== Checking Part A jobs $FIRST-$LAST${SKIP_IDS:+ (skipping $SKIP_IDS)}"
RUNNING=$(squeue -u "$USER" -h -o "%F %j %T" | grep -wFf <(echo "$CHECK") || true)
if [ -n "$RUNNING" ]; then echo "Part A is still running:"; echo "$RUNNING"; exit 1; fi

BAD=$(sacct -j "$IDS" -X -n -P -o JobID,JobName,State | grep -v '|COMPLETED$' || true)
if [ -n "$BAD" ]; then echo "Jobs that did not COMPLETE:"; echo "$BAD"; exit 1; fi

ERR=0
for id in $CHECK; do
    for f in slurm_logs/*_"$id".err; do
        [ -e "$f" ] || continue
        if grep -q Traceback "$f"; then echo "Traceback in $f:"; tail -5 "$f"; ERR=1; fi
    done
done
[ "$ERR" -eq 0 ] || exit 1

for f in 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
         06_disease_module/disease_module_output_pseudobulk/disease_module.csv \
         05_multixrank_results/multixrank_pseudo_results/gene_rwr_scores.csv \
         04_multixrank_prep/multixrank_hetero_input_pseudobulk/gene_reference_graph.pkl; do
    if [ ! -s "$f" ]; then echo "Missing or empty: $f"; exit 1; fi
    if [ "$(stat -c %Y "$f")" -lt "$(date -d 2026-10-21 +%s)" ]; then
        echo "Stale (not rebuilt by Part A): $f"; exit 1
    fi
done
echo "Part A looks clean."
echo

mkdir -p slurm_logs
sub() { local dep=$1; shift; sbatch --parsable ${dep:+--dependency=afterok:$dep} "$@"; }

echo "== Spectral (8)"
J8=$(sub "" step8.sh); sub $J8 step8ctd.sh; sub $J8 step8f.sh

echo "== Marchenko-Pastur (9)"
J9=$(sub "" step9.sh); J9b=$(sub $J9 step9b.sh); sub $J9b step9v.sh

echo "== Validation (10)"
NULLS="seeds"     sbatch --parsable --cpus-per-task=25 step10.sh
NULLS="rewired"   sbatch --parsable --cpus-per-task=25 step10.sh
NULLS="neighbors" sbatch --parsable --cpus-per-task=6 --time=02:00:00 step10.sh

echo "== HiDeF (11)"
A=$(sub ""   step11a.sh)
P=$(sub $A   step11preb.sh)
B=$(sub $A   step11b.sh)
C1=$(sub $B  step11c1.sh)
C2=$(sub $C1 step11c2.sh)
sub $C2 step11d.sh; sub $C2 step11k.sh
E=$(sub $C2 step11e.sh)
sub $E step11f.sh; sub $E step11h.sh; sub $E step11j.sh
sub $C1:$P step11g.sh

echo "== Hub removal (12)"
A12=$(sub "" step12a.sh)
NB=$(sbatch --parsable --array=0-49%10 step12b_null.sh)
AG=$(sbatch --parsable --dependency=afterok:$NB --export=ALL,N_NULL=50 step12c_aggregate.sh)
D=$(sub $A12 step12d_tail.sh)
G=$(sub $A12 step12g.sh)
sub $A12:$AG:$D:$G step12c.sh
bash submit_step12e.sh

echo
echo "Submitted. Watch with: squeue -u \$USER"
echo "Afterwards (by hand): submit_step11i.sh pilot/full, phen/ enrichment, then compare_ml_sl.sh -> plot_sl.sh -> step6f.sh"
