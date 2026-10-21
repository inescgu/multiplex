#!/bin/bash
#SBATCH --job-name=step8_spectral
#SBATCH --output=slurm_logs/step8_spectral_%j.out
#SBATCH --error=slurm_logs/step8_spectral_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10          # <- the ONE place this is hardcoded

cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail

echo "Step 8: Spectral clustering + component diagnostics"
echo "Running on host: $(hostname)"
echo "Requested cores: $SLURM_CPUS_PER_TASK"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate


mkdir -p slurm_logs
mkdir -p 08_spectral_analysis

# BLAS/OpenMP threading — reads the count SLURM already gave us
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

REF_GRAPH=06_disease_module/disease_module_output_pseudobulk/module_graph.pkl

# ------------------------------------------------------------------
# Run 1: full module (global_ppi backbone included)
# ------------------------------------------------------------------
echo ""
echo "=== [1/2] Spectral clustering: module_with_ppi ==="
python step8_multiplex_spectral_clustering.py \
    --ref-graph "$REF_GRAPH" \
    --out-dir 08_spectral_analysis/module_with_ppi \
    --mode sos --sample-p 0.4 --K 12 --k-min 2 --k-max 50 --seed 42 --degree-normalize

echo ""
echo "=== [1/2] Component diagnostic: module_with_ppi ==="
python step8b_component_diagnostic.py \
    --ref-graph "$REF_GRAPH" \
    --clusters-csv 08_spectral_analysis/module_with_ppi/spectral_clusters.csv \
    --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
    --label with_ppi

echo ""
echo "=== [1/2] Embedding diagnostic: module_with_ppi ==="
python step8c_embedding_diagnostic.py \
    --clusters-csv 08_spectral_analysis/module_with_ppi/spectral_clusters.csv \
    --embedding 08_spectral_analysis/module_with_ppi/embedding.npy \
    --eigvals 08_spectral_analysis/module_with_ppi/eigvals_sos.npy \
    --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
    --label with_ppi
# ------------------------------------------------------------------
# Run 2: global_ppi backbone excluded before layer splitting/embedding
# ------------------------------------------------------------------
echo ""
echo "=== [2/2] Spectral clustering: module_no_ppi ==="
python step8_multiplex_spectral_clustering.py \
    --ref-graph "$REF_GRAPH" \
    --out-dir 08_spectral_analysis/module_no_ppi \
    --mode sos --sample-p 0.4 --K 12 --k-min 2 --k-max 50 --seed 42 --degree-normalize \
    --exclude-ppi-layer

echo ""
echo "=== [2/2] Component diagnostic: module_no_ppi ==="
python step8b_component_diagnostic.py \
    --ref-graph "$REF_GRAPH" \
    --exclude-ppi-layer \
    --clusters-csv 08_spectral_analysis/module_no_ppi/spectral_clusters.csv \
    --out-dir 08_spectral_analysis/module_no_ppi/diagnostics \
    --label no_ppi

echo ""
echo "=== [2/2] Embedding diagnostic: module_no_ppi ==="
python step8c_embedding_diagnostic.py \
    --clusters-csv 08_spectral_analysis/module_no_ppi/spectral_clusters.csv \
    --embedding 08_spectral_analysis/module_no_ppi/embedding.npy \
    --eigvals 08_spectral_analysis/module_no_ppi/eigvals_sos.npy \
    --out-dir 08_spectral_analysis/module_no_ppi/diagnostics \
    --label no_ppi

echo ""
echo "Finished at: $(date)"