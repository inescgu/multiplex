#!/bin/bash
#SBATCH --job-name=step8f
#SBATCH --output=slurm_logs/step8f_%j.out
#SBATCH --error=slurm_logs/step8f_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2          # <- the ONE place this is hardcoded

cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail

echo "Step 8: Spectral clustering + component diagnostics"
echo "Running on host: $(hostname)"
echo "Requested cores: $SLURM_CPUS_PER_TASK"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate


# BLAS/OpenMP threading — reads the count SLURM already gave us
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

REF_GRAPH=06_disease_module/disease_module_output_pseudobulk/module_graph.pkl
echo ""
echo "=== Step 8f: module_with_ppi ==="
python step8f_null_models.py \
      --ref-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
      --mode sos --sample-p 0.4 --K 3 --force-k 2 --degree-normalize \
      --full-ref-graph 04_multixrank_prep/multixrank_hetero_input_pseudobulk/gene_reference_graph.pkl \
      --n-permutation 100 --n-population 100 --n-jobs 2 \
      --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
      --label with_ppi

python step8f_null_models.py \
      --ref-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
      --mode sos --sample-p 0.4 --K 3 --force-k 2 --degree-normalize \
      --full-ref-graph 04_multixrank_prep/multixrank_hetero_input_pseudobulk/gene_reference_graph.pkl \
      --n-permutation 100 --n-population 100 --n-jobs 2 \
      --out-dir 08_spectral_analysis/module_no_ppi/diagnostics \
      --exclude-ppi-layer \
      --label no_ppi

echo ""
echo "Finished at: $(date)"