#!/bin/bash
#SBATCH --job-name=step10b_funcperm
#SBATCH --output=slurm_logs/step10b_funcperm_%j.out
#SBATCH --error=slurm_logs/step10b_funcperm_%j.err
#SBATCH --time=35:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --array=1
cd /lisc/home/user/gerardursin/multiplex/
# Load any required modules here
module load Python/3.14.7-weekly
source /lisc/data/scratch/menche/ines/general_env/bin/activate
pip install pyarrow
pip install --upgrade pip
mkdir -p slurm_logs

# NOTE: the script now parallelizes permutations across worker PROCESSES
# (concurrent.futures.ProcessPoolExecutor), not via BLAS/OpenMP threading --
# the RWR power iteration is a scipy sparse matvec, which OMP/MKL/OPENBLAS
# threads don't meaningfully speed up. --cpus-per-task above was bumped to
# 16 (adjust to whatever's actually free/fair-share on your partition) and
# --n-workers below is wired to $SLURM_CPUS_PER_TASK so the pool size tracks
# whatever SLURM actually grants. If you don't pass --n-workers at all the
# script auto-detects $SLURM_CPUS_PER_TASK itself, so this is here mainly to
# make the pool size visible/explicit in the job script.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --disease-module-dir points DIRECTLY at the folder containing
# disease_module.csv, same convention as step10_permutation_test.slurm --
# i.e. 06_disease_module/disease_module_output, not its parent. (The script
# itself does NOT hardcode a "disease_module_output" subfolder -- if you've
# edited your local copy to do that, revert it, since it makes this flag
# mean something different here than it does in step10.)
#
# --lcc-per-layer: your step6 run passed --lcc-per-layer, which filters
# gene_reference_graph.pkl BEFORE building the true module -- without the
# same flag here, this script would sample its null seeds/neighbors from a
# larger, unfiltered graph than the one the true module actually came from
# (same fix as step10's slurm).
python step10b_functional_permutation_test_multiplex.py \
    --reference-graph 04_multixrank_prep/multixrank_hetero_input/gene_reference_graph.pkl \
    --disease-module-dir 06_disease_module/ \
    --lcc-per-layer \
    --ot-data ../og_gene_set/gene_ranking/ot_data \
    --phenotype-csv ../expression_data/human_pain_genes_database.csv \
    --scaling 1 \
    --n-perms 200 \
    --n-workers $SLURM_CPUS_PER_TASK \
    --no-go \
    --run-rewired-null \
    --n-perms-rewired 100 \
    --rewire-nswap-factor 10 \
    --run-fixed-seed-null \
    --n-perms-fixed-seed 200 \
    --fixed-seed-null-mode degree_matched
# Print some information about the job
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Finished at: $(date)"
