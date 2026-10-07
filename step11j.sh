#!/bin/bash
#SBATCH --job-name=step11j
#SBATCH --output=slurm_logs/step11j_%j.out
#SBATCH --error=slurm_logs/step11j_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
source ./run_env.sh
echo "Step 11j: Persistence confounds"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

snapshot "$RUN_DIR/characterized"

python step11j_persistence_confounds.py \
        --cluster-summary "$RUN_DIR/characterized/heterogeneous_hidef_faithful_cpm_cluster_summary.csv" \
        --output-prefix "$RUN_DIR/characterized/heterogeneous_hidef_faithful_cpm"
echo "Finished at: $(date)"