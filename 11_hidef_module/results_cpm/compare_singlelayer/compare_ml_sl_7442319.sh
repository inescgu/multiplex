#!/bin/bash
#SBATCH --job-name=compare_ml_sl
#SBATCH --output=slurm_logs/compare_ml_sl_%j.out
#SBATCH --error=slurm_logs/compare_ml_sl_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
source ./run_env.sh          # provides $RUN_DIR

echo "Comparing high-persistence multiplex vs single-layer HiDeF clusters"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

SL=../og_gene_set
OUT="$RUN_DIR/compare_singlelayer"
snapshot "$OUT"

# CHECK THESE TWO against multiplex/phen/pipeline_config.py:
#   --ml-enrich-dir   = phen/<OUTPUT_DIR>/<PPI_SOURCE>
#   --ml-cluster-type = <CLUSTER_TYPE>   (left as '*' it globs, and warns if >1 dir matches)
python compare_multiplex_vs_singlelayer.py \
    --ml-nodes "$RUN_DIR/heterogeneous_hidef_faithful_cpm.nodes" \
    --ml-supra-graph 11_hidef_module/supra_graph.pkl \
    --ml-enrich-dir phen/output/combined26_multiplex \
    --ml-cluster-type '*' \
    --sl-communities $SL/output/combined26/hidef/hidef_communities.pkl \
    --sl-network $SL/output/combined26/rwr/module_graph.pkl \
    --sl-enrich-dir $SL/phen/output/combined26 \
    --sl-cluster-type hidef_from_comparison \
    --persistence-quantile 0.75 \
    --min-size 4 \
    --enrich-p 0.05 --fdr 0.05 \
    --min-enrich-cosine 0.3 --min-shared-terms 2 \
    --output-dir "$OUT"
echo "Finished at: $(date)"