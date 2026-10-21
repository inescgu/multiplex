#!/bin/bash
#SBATCH --job-name=step11c
#SBATCH --output=slurm_logs/step11c_%j.out
#SBATCH --error=slurm_logs/step11c_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=70G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail
source ./run_env.sh

echo "Step 11c sweep: --neighbors sensitivity"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

OUT=11c_neighbors_sweep/ref10_$(date +%Y%m%d)
guard "$OUT"
snapshot "$OUT"

python step11c_neighbors_sweep.py \
        --ensemble 11_hidef_module/ensemble.pkl \
        --neighbor-values 1 2 3 4 5 7 10 \
        --reference-neighbors 5 \
        --minsize $MINSIZE --jaccard $JACCARD --chi $CHI --consensus-p $CONSENSUS_P \
        --output-dir "$OUT"
echo "Finished at: $(date)"