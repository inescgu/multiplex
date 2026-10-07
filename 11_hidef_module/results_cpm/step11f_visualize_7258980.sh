#!/bin/bash
#SBATCH --job-name=step11f_visualize
#SBATCH --output=slurm_logs/step11f_visualize_%j.out
#SBATCH --error=slurm_logs/step11f_visualize_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
source ./run_env.sh

echo "Step 11f: Visualizing persistent heterogeneous HiDeF clusters (supra-graph overview, composition heatmap, per-cluster zooms)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
# NOTE: --time/--mem above assume no cached layout exists yet, so the DrL layout
# of the full supra-graph has to be computed from scratch -- this is the slow part
# and the main reason for the long walltime/high mem here. Once
# 11_hidef_hetero/supra_graph_layout.pkl exists, reruns (e.g. after re-running
# step 11e with different thresholds) load the cached layout and finish in
# minutes -- you can drop --time/--mem way down for those.
snapshot "$RUN_DIR"

python step11f_visualize_clusters.py \
    --supra-graph 11_hidef_module/supra_graph.pkl \
    --char-prefix "$RUN_DIR/characterized/heterogeneous_hidef_faithful_cpm" \
    --seed-genes 00_inputs/seed_genes.csv \
    --output-dir "$RUN_DIR" \
    --layout-cache 11_hidef_module/supra_graph_layout.pkl \
    --layout-algo drl --top-n 40 --min-persistence-pct 75
echo "Finished at: $(date)"