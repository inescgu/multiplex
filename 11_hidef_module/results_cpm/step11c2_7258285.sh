#!/bin/bash
#SBATCH --job-name=step11c2
#SBATCH --output=slurm_logs/step11c2_%j.out
#SBATCH --error=slurm_logs/step11c2_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
cd /lisc/home/user/gerardursin/multiplex/
source ./run_env.sh

echo "Step 11c phase 2: weave"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"

module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
guard "$RUN_DIR/heterogeneous_hidef_faithful_cpm.nodes"
snapshot "$RUN_DIR"

python step11c_phase2_weave.py \
    --consensus "$RUN_DIR/consensus.pkl" \
    --output-prefix "$RUN_DIR/heterogeneous_hidef_faithful_cpm" \
    --cutoff $JACCARD
echo "Finished at: $(date)"