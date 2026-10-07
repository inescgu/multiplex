#!/bin/bash
#SBATCH --job-name=step11d_crosscheck_infomap
#SBATCH --output=slurm_logs/step11d_crosscheck_infomap_%j.out
#SBATCH --error=slurm_logs/step11d_crosscheck_infomap_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
source ./run_env.sh

echo "Step 11d: Cross-checking persistent communities against Infomap on the supra-graph"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

# Uses the patched step11d_validation.py: reads .nodes directly via --nodes-file,
# no separately-recovered communities pickle needed (that recovery step was tied
# to the old RB run and would otherwise silently go stale here).
snapshot "$RUN_DIR"

python step11d_validation.py \
    --supra-graph 11_hidef_module/supra_graph.pkl \
    --ensemble 11_hidef_module/ensemble.pkl \
    --nodes-file "$RUN_DIR/heterogeneous_hidef_faithful_cpm.nodes" \
    --min-size 4 \
    --output "$RUN_DIR/infomap_crosscheck.json"
echo "Finished at: $(date)"