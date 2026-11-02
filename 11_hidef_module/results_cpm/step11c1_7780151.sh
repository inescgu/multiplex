#!/bin/bash
#SBATCH --job-name=step11c1
#SBATCH --output=slurm_logs/step11c1_%j.out
#SBATCH --error=slurm_logs/step11c1_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=55G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex
set -eo pipefail
source ./run_env.sh

echo "Step 11c phase 1: windowed consensus"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
guard "$RUN_DIR/consensus.pkl"
snapshot "$RUN_DIR"

python step11c_phase1_consensus.py \
    --ensemble 11_hidef_module/ensemble.pkl \
    --output "$RUN_DIR/consensus.pkl" \
    --minsize $MINSIZE --jaccard $JACCARD --chi $CHI \
    --consensus-p $CONSENSUS_P --neighbors $NEIGHBORS
echo "Finished at: $(date)"