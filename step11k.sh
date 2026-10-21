#!/bin/bash
#SBATCH --job-name=step11k
#SBATCH --output=slurm_logs/step11k_%j.out
#SBATCH --error=slurm_logs/step11k_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail
source ./run_env.sh
echo "Step 11k: DAG nesting"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
OUT="$RUN_DIR/dag_persist8_$(date +%Y%m%d)"
guard "$OUT"
snapshot "$OUT"

python step11k_dag_nesting_analysis.py \
        --nodes-file "$RUN_DIR/heterogeneous_hidef_faithful_cpm.nodes" \
        --edges-file "$RUN_DIR/heterogeneous_hidef_faithful_cpm.edges" \
        --output-prefix "$OUT/dag" \
        --enrichment-csv phen/cluster_enrichment_long.csv \
        --enrichment-label-col analysis --enrichment-pval-col adjusted_p_value
echo "Finished at: $(date)"