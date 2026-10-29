#!/bin/bash
#SBATCH --job-name=step9b_mp
#SBATCH --output=slurm_logs/step9b_mp_%j.out
#SBATCH --error=slurm_logs/step9b_mp_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail
echo "Step 9: Marchenko-Pastur analysis"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

# --graph/--registry (old integrated_multiplex.pkl + node_registry.csv) are
# gone -- see step8_spectral_clustering.slurm's comment for why. Point
# --ref-graph at the same gene_reference_graph.pkl used there.
# step9 writes one multiplex_bridges.csv per omega (keep in sync with step9.sh --omega-sweep)
for OM in 0.1 1.0 5.0; do
    echo "--- omega=$OM"
    python step9b_permutation_test_bridges.py \
        --bridges-csv 09_marchenko_pastur/omega_$OM/multiplex_bridges.csv \
        --out-dir 09_marchenko_pastur/omega_$OM/permutation_test \
        --min-celltypes-present 3 \
        --frac-minority-threshold 0.6 \
        --n-perm 10000
done

echo "Finished at: $(date)"
