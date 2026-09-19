#!/bin/bash
#SBATCH --job-name=step11g
#SBATCH --output=slurm_logs/step11g_%j.out
#SBATCH --error=slurm_logs/step11g_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
echo "Step 11g"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

python step11g_presentation_plots.py \
    --consensus 11_hidef_results_faithful_cpm/consensus.pkl \
    --wide-scan-csv 11_hidef_hetero/resolution_profile.csv \
    --fine-scan-csv 11_hidef_hetero/resolution_profile_cpm_finescan.csv \
    --partition-type CPM \
    --production-min-resolution 1e-5 \
    --production-max-resolution 0.9 \
    --total-nodes 140139 \
    --output-dir presentation_figures
echo "Finished at: $(date)"