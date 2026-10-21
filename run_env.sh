# Sourced by every step11*.sh. Edit HERE to switch runs.
export RUN_DIR=11_hidef_module/results_cpm   # for a new run use e.g. results/11c/neighbors5_chi5_p75
export NEIGHBORS=2 CHI=5 CONSENSUS_P=75 JACCARD=0.75 MINSIZE=4

# copy the submitted sbatch script next to the outputs, named by job id
snapshot() { mkdir -p "$1" && cp "$0" "$1/${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-nojob}.sh"; }
# refuse to run if the path already exists
guard() { if [ -e "$1" ]; then echo "REFUSING: $1 already exists (rename it or change RUN_DIR)"; exit 1; fi; }