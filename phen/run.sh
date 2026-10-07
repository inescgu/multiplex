#!/bin/bash
#SBATCH --job-name=phen
#SBATCH --output=slurm_logs/phen_%j.out
#SBATCH --error=slurm_logs/phen_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/phen/

echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
cp pipeline_config.py "output/combined26_multiplex/pipeline_config_$(date +%Y%m%d_%H%M).py" 2>/dev/null || true
python run_multiplex_enrichment_pipeline.py
echo "Finished at: $(date)"