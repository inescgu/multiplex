#!/bin/bash
#SBATCH --job-name=check
#SBATCH --output=slurm_logs/check_%j.out
#SBATCH --error=slurm_logs/check_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

module load Python/3.10.8-GCCcore-12.2.0-bare
# python -m venv /lisc/data/scratch/menche/ines/tscoex
source /lisc/data/scratch/menche/ines/multi_env/bin/activate
# pip install --upgrade pip
# pip install pandas numpy networkx scipy tiledbsoma cellxgene-census

# Run the multiplex interactome construction
python check_paths.py

# End time
echo "Finished at: $(date)"

# The "stable" release is currently 2025-11-08. Specify 'census_version="2025-11-08"' in future calls to open_soma() to ensure data consistency.