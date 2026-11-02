#!/bin/bash
#SBATCH --job-name=step6f
#SBATCH --output=slurm_logs/stpe6f_%j.out
#SBATCH --error=slurm_logs/step6f_%j.err
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
SL_PPI=${SL_PPI:-combined26}     # single-layer network; ppi20261028 once og_gene_set is rerun on it
OUT="$RUN_DIR/compare_singlelayer"      # same folder compare_ml_sl.sbatch wrote matched_pairs.csv to


python step6f_export_sl_decomposition_wwiznet.py \
    --sl-cluster 6 \
    --matched-pairs "$RUN_DIR/compare_singlelayer/matched_pairs.csv" \
    --ml-nodes "$RUN_DIR/heterogeneous_hidef_faithful_cpm.nodes" \
    --ml-edges "$RUN_DIR/heterogeneous_hidef_faithful_cpm.edges" \
    --supra-graph 11_hidef_module/supra_graph.pkl \
    --sl-communities $SL/output/${SL_PPI}/hidef/hidef_communities.pkl \
    --sl-network $SL/output/${SL_PPI}/rwr/module_graph.pkl \
    --highlight OPRM1 COMT KCNJ6 \
    --out "$RUN_DIR/compare_singlelayer/wwiznet_SL6" \
    --animate "$RUN_DIR/compare_singlelayer/wwiznet_SL6/SL6_decomposition.gif"

echo "Finished at: $(date)"
