#!/bin/bash
#SBATCH --job-name=phen_viz
#SBATCH --output=slurm_logs/phen_viz_%j.out
#SBATCH --error=slurm_logs/phen_viz_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/phen/
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
python run_multiplex_visualizations.py
echo "Finished at: $(date)"
 