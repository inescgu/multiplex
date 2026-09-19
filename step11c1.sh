#!/bin/bash
#SBATCH --job-name=step11c1
#SBATCH --output=slurm_logs/step11c1_%j.out
#SBATCH --error=slurm_logs/step11c1_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=55G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
echo "Step 11c: Weaving partition ensemble into a HiDeF persistent hierarchy (hidef.weaver)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
python step11c_phase1_consensus.py \
    --ensemble 11_hidef_hetero/ensemble.pkl \
    --output 11_hidef_results_faithful_cpm/consensus.pkl \
    --minsize 4 \
    --jaccard 0.75 \
    --chi 5 \
    --consensus-p 75 \
    --neighbors 3

echo "Finished at: $(date)"