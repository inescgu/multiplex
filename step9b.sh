#!/bin/bash
#SBATCH --job-name=step9b_mp
#SBATCH --output=slurm_logs/step9b_mp_%j.out
#SBATCH --error=slurm_logs/step9b_mp_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
echo "Step 9: Marchenko-Pastur analysis"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

# --graph/--registry (old integrated_multiplex.pkl + node_registry.csv) are
# gone -- see step8_spectral_clustering.slurm's comment for why. Point
# --ref-graph at the same gene_reference_graph.pkl used there.
python step9b_permutation_test_bridges.py \
    --bridges-csv 09_marchenko_pastur/multiplex_bridges.csv \
    --out-dir 09_marchenko_pastur/permutation_test \
    --min-celltypes-present 3 \
    --frac-minority-threshold 0.6 \
    --n-perm 10000
    
echo "Finished at: $(date)"
