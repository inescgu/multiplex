#!/bin/bash
#SBATCH --job-name=step0b_neuronchat
#SBATCH --output=slurm_logs/step0b_neuronchat_%j.out
#SBATCH --error=slurm_logs/step0b_neuronchat_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail
echo "Step 0b: Running NeuronChat (R)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load R
module load GCC/15.2.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="$GCC_LIBDIR:$LD_LIBRARY_PATH"

# NOTE: needs NeuronChat, Seurat, optparse, dplyr installed in this R lib
Rscript step0b_run_neuronchat.R \
    --seurat-rds ../R/convert_hdrg_human_2025/seurat_hdrg/seurat_object_preprint_release.Rds \
    --celltype-col Atlas_annotation \
    --assay RNA \
    --species human \
    --out pathway_prep/lr_sources/neuronchat_long.csv
echo "Finished at: $(date)"