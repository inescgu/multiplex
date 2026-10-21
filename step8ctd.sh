#!/bin/bash
#SBATCH --job-name=step8_spectral
#SBATCH --output=slurm_logs/step8_spectral_%j.out
#SBATCH --error=slurm_logs/step8_spectral_%j.err
#SBATCH --time=5:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         # <- the ONE place this is hardcoded
cd /lisc/home/user/gerardursin/multiplex/
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
echo "=== Step 8d: module_with_ppi ==="
python step8d_duplicate_groups.py \
  --clusters-csv 08_spectral_analysis/module_with_ppi/spectral_clusters.csv \
  --embedding 08_spectral_analysis/module_with_ppi/embedding.npy \
  --ref-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
  --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
  --label with_ppi

echo ""
echo "=== Step 8d: module_no_ppi ==="
python step8d_duplicate_groups.py \
  --clusters-csv 08_spectral_analysis/module_no_ppi/spectral_clusters.csv \
  --embedding 08_spectral_analysis/module_no_ppi/embedding.npy \
  --ref-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
  --exclude-ppi-layer \
  --out-dir 08_spectral_analysis/module_no_ppi/diagnostics \
  --label no_ppi

echo ""
echo "=== Step 8e: module_with_ppi (cluster sizes + forced k=2 only, no single-layer comparison yet) ==="
python step8e_forced_k_and_compare.py \
  --clusters-csv 08_spectral_analysis/module_with_ppi/spectral_clusters.csv \
  --embedding 08_spectral_analysis/module_with_ppi/embedding.npy \
  --force-k 3 \
  --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
  --label with_ppi

echo ""
echo "=== Step 8e: module_no_ppi (cluster sizes + forced k=2 only, no single-layer comparison yet) ==="
python step8e_forced_k_and_compare.py \
  --clusters-csv 08_spectral_analysis/module_no_ppi/spectral_clusters.csv \
  --embedding 08_spectral_analysis/module_no_ppi/embedding.npy \
  --force-k 3 \
  --out-dir 08_spectral_analysis/module_no_ppi/diagnostics \
  --label no_ppi


echo ""
echo "=== Step 8g: module with ppi ==="
# only if this run's clusters-csv came from module_no_ppi
python step8g_seed_core_periphery_null.py \
  --ref-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
  --clusters-csv 08_spectral_analysis/module_with_ppi/spectral_clusters.csv \
  --cluster-col cluster \
  --core-definition composite \
  --seed-genes-csv 00_inputs/seed_genes.csv \
  --full-ref-graph 04_multixrank_prep/multixrank_hetero_input_pseudobulk/gene_reference_graph.pkl \
  --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
  --label with_ppi

echo ""
echo "=== Step 8g: module no ppi ==="
python step8g_seed_core_periphery_null.py \
  --ref-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
  --exclude-ppi-layer \
  --clusters-csv 08_spectral_analysis/module_no_ppi/spectral_clusters.csv \
  --cluster-col cluster \
  --core-definition composite \
  --seed-genes-csv 00_inputs/seed_genes.csv \
  --full-ref-graph 04_multixrank_prep/multixrank_hetero_input_pseudobulk/gene_reference_graph.pkl \
  --out-dir 08_spectral_analysis/module_no_ppi/diagnostics \
  --label no_ppi

echo ""
echo "=== Step 8f: slide figures ==="
python step8h_slide_figures.py \
    --with-ppi-dir 08_spectral_analysis/module_with_ppi \
    --no-ppi-dir 08_spectral_analysis/module_no_ppi \
    --ref-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
    --seed-genes-csv 00_inputs/seed_genes.csv \
    --out-dir 08_spectral_analysis/slide_figures
    
python step8i_cluster_overview_figures.py \
    --with-ppi-dir 08_spectral_analysis/module_with_ppi \
    --no-ppi-dir 08_spectral_analysis/module_no_ppi \
    --seed-genes-csv 00_inputs/seed_genes.csv \
    --out-dir 08_spectral_analysis/slide_figures
    
echo ""
echo "Finished at: $(date)"