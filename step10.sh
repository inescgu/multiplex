#!/bin/bash
#SBATCH --job-name=step10_validate
#SBATCH --output=slurm_logs/step10_validate_%j.out
#SBATCH --error=slurm_logs/step10_validate_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#
# Submit with:  sbatch run_step10.sh      (a .sh file; sbatch reads the #SBATCH lines)
# Defaults above (24 cpus, 64G, 48h) can be overridden on the command line, which beats the header.
# Memory: the earlier 6-worker run peaked at ~9 GB for the whole job, so 24 workers need roughly 35 GB.
# cpus-per-task must fit on ONE node (workers are forked processes). Check:  sinfo -o "%P %c %m %l"
#
#   NULLS="seeds"   sbatch --cpus-per-task=25 run_step10.sh     # 100 perms = exactly 4 rounds of 25
#   NULLS="rewired" sbatch --cpus-per-task=25 run_step10.sh
#   NULLS="neighbors" sbatch --cpus-per-task=6 --time=02:00:00 run_step10.sh
# Pilot (one round only, to read the per-permutation time from the log):
#   NULLS="seeds" EXTRA="--perm-range 0 25" sbatch --cpus-per-task=25 run_step10.sh
# Split one null across several jobs, then merge:
#   NULLS="seeds" EXTRA="--perm-range 0 50"   sbatch ... run_step10.sh
#   NULLS="seeds" EXTRA="--perm-range 50 100" sbatch ... run_step10.sh
#   NULLS="neighbors seeds rewired" EXTRA="--from-partial" sbatch --cpus-per-task=2 --time=00:30:00 run_step10.sh
# Timed out? Resubmit the SAME line: --resume continues from the saved permutations.
# (sbatch exports your environment by default, so NULLS/EXTRA reach this script.)
NULLS=${NULLS:-neighbors}
EXTRA=${EXTRA:-}

cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail
echo "Step 10: validate disease module | nulls: $NULLS | extra: $EXTRA"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID | CPUs: $SLURM_CPUS_PER_TASK"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate
mkdir -p slurm_logs 10_validation
START_MARK=$(mktemp); touch "$START_MARK"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python -c "import multixrank, yaml, pyarrow, matplotlib, networkx" || exit 1

WDIR=04_multixrank_prep/multixrank_hetero_input_pseudobulk
SCORES=05_multixrank_results/multixrank_pseudo_results/gene_rwr_scores.csv

# The step6 flags below must be IDENTICAL to the step6 run that made disease_module.csv
# (both read the excluded seeds from the file step5 writes).
# --resume: every finished permutation is saved immediately (null_<tag>_partial*.csv), so a timeout
# loses nothing; resubmitting the same line continues where it stopped, and reuses identity_check.csv.
python -u step10_validate_module.py \
    --wdir $WDIR \
    --config config.yml \
    --scores $SCORES \
    --disease-module-csv 06_disease_module/disease_module_output_pseudobulk/disease_module.csv \
    --output-dir 10_validation \
    --connectivity all \
    --k 200 \
    --max-bridge 30000 \
    --lcc-per-layer \
    --exclude-ppi-seeds \
    --exclude-seeds $(cat "$WDIR/seeds/excluded_seeds.txt") \
    --ot-data ../og_gene_set/gene_ranking/ot_data \
    --phenotype-csv ../expression_data/human_pain_genes_database.csv \
    --topology \
    --nulls $NULLS \
    --n-perms-neighbors 1000 \
    --n-perms-seeds 100 \
    --n-perms-rewired 100 \
    --resume \
    --n-workers $SLURM_CPUS_PER_TASK \
    $EXTRA

echo "Exit code: $?"

# step10 exits 0 even when every permutation failed (e.g. workers OOM-killed -> BrokenProcessPool),
# so fail the job unless each requested null wrote a fresh summary. Skipped for split --perm-range runs.
if [[ "$EXTRA" != *--perm-range* ]]; then
    for N in $NULLS; do
        S=10_validation/null_${N}_summary.csv
        if [ ! -f "$S" ] || [ "$S" -ot "$START_MARK" ]; then
            echo "ERROR: no new $S -- check the log for failed permutations"; exit 1
        fi
    done
fi
echo "Finished at: $(date)"