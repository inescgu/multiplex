#!/bin/bash
#SBATCH --job-name=step0c_umap
#SBATCH --output=slurm_logs/step0c_umap_%j.out
#SBATCH --error=slurm_logs/step0c_umap_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

cd /lisc/home/user/gerardursin/multiplex/
module load Python/3.12.3-GCCcore-13.3.0
module load R   # whatever your cluster's R module is called; Seurat + Matrix must be installed
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

pip install scanpy harmonypy

python step0c_umap_compare.py \
    --seurat-rds ../R/convert_hdrg_human_2025/seurat_hdrg/seurat_object_preprint_release.Rds \
    --census-version "2025-11-08" \
    --output-dir 01_ts_multiplex/step0c_umap