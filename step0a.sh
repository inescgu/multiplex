#!/bin/bash
#SBATCH --job-name=step0a_liana
#SBATCH --output=slurm_logs/step0a_liana_%j.out
#SBATCH --error=slurm_logs/step0a_liana_%j.err
#SBATCH --time=18:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/

echo "Step 0a: Running LIANA (R) on the Seurat object — natmi, connectome, logfc, sca, cellphonedb + consensus"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load R
# Same GLIBCXX fix as install_r_pkgs.sh — needed here too, since loading
# CellChat/NeuronChat/liana/presto at runtime hits the same libstdc++
# mismatch installing them did.
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="$GCC_LIBDIR:$LD_LIBRARY_PATH"

# Path matches what you actually ran last (relative to
# /lisc/home/user/gerardursin/multiplex/) — adjust if it moves.
Rscript step0a_run_liana.R \
    --seurat-rds ../R/convert_hdrg_human_2025/seurat_hdrg/seurat_object_preprint_release.Rds \
    --celltype-col Atlas_annotation \
    --assay RNA \
    --resource Consensus \
    --batch-by-pair \
    --out pathway_prep/lr_sources/liana_long.csv

# Note: --batch-by-pair processes cell types two at a time (231 pairs for 22
# types) instead of loading the whole atlas into liana_wrap() at once — this
# is what makes --mem=80G plausible without capping genes or cells. Expect
# a longer wall-clock time than a single whole-atlas call; bump --time above
# if it doesn't finish in 6h.
echo "Finished at: $(date)"