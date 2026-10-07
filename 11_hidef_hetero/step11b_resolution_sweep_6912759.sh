#!/bin/bash
#SBATCH --job-name=step11b_resolution_sweep
#SBATCH --output=slurm_logs/step11b_resolution_sweep_%j.out
#SBATCH --error=slurm_logs/step11b_resolution_sweep_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
source ./run_env.sh

echo "Step 11b: Multi-resolution (+coupling) partition sweep over the heterogeneous supra-graph"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

guard 11_hidef_hetero/ensemble.pkl      # everything downstream depends on this file
snapshot 11_hidef_hetero

# CHANGED from the original RB-based sweep, based on the resolution_profile diagnostics:
#   - RB (--min-resolution 0.01 --max-resolution 10.0) sat almost entirely inside a flat
#     plateau (9,030 -> 9,183 communities, a 1.7% change) -- not a useful ensemble for
#     persistence, which is why persistence collapsed to ~max for 99.5% of clusters.
#   - CPM shows real, well-resolved variation across roughly 1e-5 to 0.94 (9,034 -> 99,444
#     communities), then collapses hard to full singleton fragmentation (140,139 = every
#     node isolated) exactly at gamma=1.0 -- confirmed by a fine-grained scan between 0.3
#     and 1.0. --max-resolution 0.9 below stays comfortably clear of that cliff while still
#     covering essentially the whole useful range.
#   - --n-resolutions bumped from 30 to 40: the useful range is now ~5 orders of magnitude
#     wide (vs. ~3 for the original RB sweep), so more points are needed to keep roughly
#     the same sampling density per decade.
# --n-restarts is still 1, same as your original conservative choice -- bump this up
# (at proportional runtime cost) if you want restart-based robustness signal for step 11c.
python step11b_resolution_sweep.py \
    --supra-graph 11_hidef_hetero/supra_graph.pkl \
    --output 11_hidef_hetero/ensemble.pkl \
    --min-resolution 1e-5 \
    --max-resolution 0.9 \
    --n-resolutions 40 \
    --omegas 1.0 \
    --n-restarts 1 \
    --partition-type CPM
echo "Finished at: $(date)"

# The guard means a rerun refuses until you mv the old ensemble.pkl aside, e.g. mv ensemble.pkl ensemble_CPM_1e-5_0.9_$(date +%Y%m%d).pkl.