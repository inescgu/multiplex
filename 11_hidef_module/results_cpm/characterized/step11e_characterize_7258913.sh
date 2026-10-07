#!/bin/bash
#SBATCH --job-name=step11e_characterize
#SBATCH --output=slurm_logs/step11e_characterize_%j.out
#SBATCH --error=slurm_logs/step11e_characterize_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
source ./run_env.sh

echo "Step 11e: Characterizing persistent heterogeneous HiDeF clusters (cell-type composition, conserved genes, edge-type breakdown)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
# NOTE: --nodes-file points at step 11c Phase 2's real hidef_finder output.
# If you only have the recovered communities pickle from step 11d's changelog
# instead, swap --nodes-file for:
#   --communities-pkl 11_hidef_results_fixed/heterogeneous_hidef_communities_FIXED.pkl
snapshot "$RUN_DIR/characterized"

python step11e_characterize_clusters.py \
    --supra-graph 11_hidef_module/supra_graph.pkl \
    --nodes-file "$RUN_DIR/heterogeneous_hidef_faithful_cpm.nodes" \
    --seed-genes 00_inputs/seed_genes.csv \
    --min-size 2 \
    --output-prefix "$RUN_DIR/characterized/heterogeneous_hidef_faithful_cpm"
echo "Finished at: $(date)"