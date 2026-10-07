#!/bin/bash
#SBATCH --job-name=step11preb_resolution_sweep
#SBATCH --output=slurm_logs/step11preb_resolution_sweep_%j.out
#SBATCH --error=slurm_logs/step11preb_resolution_sweep_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/

echo "Step 11preb: CPM resolution diagnostic"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate


python step11preb_diagnostic.py \
        --supra-graph 11_hidef_module/supra_graph.pkl \
        --output-prefix 11_hidef_module/resolution_profile \
        --min-resolution 8.34e-04 \
        --max-resolution 1.0 \
        --n-resolutions 20 \
        --partition-types CPM
echo "Finished at: $(date)"