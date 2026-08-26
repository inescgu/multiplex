#!/bin/bash
#SBATCH --job-name=step11b_resolution_sweep
#SBATCH --output=slurm_logs/step11b_resolution_sweep_%j.out
#SBATCH --error=slurm_logs/step11b_resolution_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/

echo "Step 11b: Multi-resolution (+coupling) partition sweep over the heterogeneous supra-graph"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

# NOTE: this is the heaviest step -- n_resolutions x n_omegas x n_restarts full Leiden
# runs on the whole supra-graph. --time/--mem above are a starting guess; increase for
# a wider gamma grid, multiple omegas, or restarts > 1, and dial down for a first
# quick sanity-check pass (e.g. --n-resolutions 15 --omegas 1.0) before committing to
# the full sweep. If you do need multiple omegas, consider splitting them across a
# SLURM job array instead of one long serial job (one array task per omega value,
# each writing its own ensemble file, concatenated before step 3c).
python step11b_resolution_sweep.py \
    --supra-graph 11_hidef_hetero/supra_graph.pkl \
    --output 11_hidef_hetero/ensemble.pkl \
    --min-resolution 0.01 \
    --max-resolution 10.0 \
    --n-resolutions 30 \
    --omegas 1.0 \
    --n-restarts 1 \
    --partition-type RB
echo "Finished at: $(date)"