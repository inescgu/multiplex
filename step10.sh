#!/bin/bash
#SBATCH --job-name=step10_perm
#SBATCH --output=slurm_logs/step10_perm_%j.out
#SBATCH --error=slurm_logs/step10_perm_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
echo "Step 10: Permutation test for disease module"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate
mkdir -p slurm_logs
mkdir -p 06_disease_module/permutation_test

# --exclude-ppi-seeds is already handled correctly: step6 records it in
# disease_module.csv's is_excluded_seed column, and step10 now reads
# is_seed & NOT is_excluded_seed as the true anchor set, so it matches
# automatically -- no separate flag needed here.
#
# NEW: --lcc-per-layer added below. Your step6 run passed --lcc-per-layer,
# which filters gene_reference_graph.pkl BEFORE building the true module --
# without the same flag here, step10 would sample its null seeds/neighbors
# from a larger, unfiltered graph than the one the true module actually
# came from.
python step10_permutation_test_multiplex.py \
    --disease-module-dir 06_disease_module/disease_module_output \
    --reference-graph 04_multixrank_prep/multixrank_hetero_input/gene_reference_graph.pkl \
    --lcc-per-layer \
    --restart 0.7 \
    --scaling 0 \
    --k 200 \
    --n-perms 1000 \
    --bin-size 5 \
    --seed 42 \
    --output-dir 06_disease_module
echo "Finished at: $(date)"
