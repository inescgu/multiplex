#!/bin/bash
#SBATCH --job-name=step11g
#SBATCH --output=slurm_logs/step11g_%j.out
#SBATCH --error=slurm_logs/step11g_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
source ./run_env.sh

echo "Step 11g: Presentation plots"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

python step11g_presentation_plots.py \
    --consensus "$RUN_DIR/consensus.pkl" \
    --wide-scan-csv 11_hidef_module/resolution_profile.csv \
    --partition-type CPM \
    --production-min-resolution 8.34e-04 --production-max-resolution 0.9 \
    --total-nodes 9940 \
    --output-dir presentation_figures
echo "Finished at: $(date)"