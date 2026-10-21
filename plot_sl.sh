#!/bin/bash
#SBATCH --job-name=plot_sl_decomp
#SBATCH --output=slurm_logs/plot_sl_decomp_%j.out
#SBATCH --error=slurm_logs/plot_sl_decomp_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail
source ./run_env.sh          # provides $RUN_DIR

echo "Plotting single-layer module decomposition into multiplex neuron-type sub-modules"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate   # hidef env: igraph, networkx, pandas, matplotlib

# fail early with a clear message if something the script needs is missing from the env
python -c "import igraph, networkx, pandas, matplotlib; print('igraph', igraph.__version__)" || {
    echo "ERROR: hidef_env is missing one of igraph/networkx/pandas/matplotlib"; exit 1; }

SL=../og_gene_set
OUT="$RUN_DIR/compare_singlelayer"      # same folder compare_ml_sl.sbatch wrote matched_pairs.csv to
SL_CLUSTER=${SL_CLUSTER:-6}             # override at submit time: sbatch --export=ALL,SL_CLUSTER=29 plot_sl_decomposition.sbatch

[ -f "$OUT/matched_pairs.csv" ] || { echo "ERROR: $OUT/matched_pairs.csv not found -- run compare_ml_sl.sbatch first"; exit 1; }
snapshot "$OUT/figures_SL${SL_CLUSTER}"

python plot_sl_module_decomposition.py \
    --sl-cluster "$SL_CLUSTER" \
    --matched-pairs "$OUT/matched_pairs.csv" \
    --ml-nodes "$RUN_DIR/heterogeneous_hidef_faithful_cpm.nodes" \
    --ml-edges "$RUN_DIR/heterogeneous_hidef_faithful_cpm.edges" \
    --supra-graph 11_hidef_module/supra_graph.pkl \
    --sl-communities $SL/output/combined26/hidef/hidef_communities.pkl \
    --sl-network $SL/output/combined26/rwr/module_graph.pkl \
    --highlight OPRM1 KCNJ6 \
    --frames \
    --steps bubbles none OPRM1 OPRM1,KCNJ6 \
    --max-partner-labels 30 \
    --out-dir "$OUT/figures_SL${SL_CLUSTER}"
# optional extras (append before --out-dir):
#   --steps none OPRM1 OPRM1,KCNJ6     animation frames (default shown); comma = highlight together,
#   --steps bubbles none OPRM1 ...     'bubbles' = empty circles only, as an extra opening frame
#   --no-svg                           skip the editable SVG copies of the frames
#   --members shared            only SL-module genes inside the bubbles (less busy)
#   --link-all-shared           dashed arcs for every recurring gene
#   --include Cluster3-24 Cluster2-51 ...   pick sub-modules by hand
#   --exclude ClusterX-Y        drop specific sub-modules
echo "Finished at: $(date)"
