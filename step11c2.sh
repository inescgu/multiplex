#!/bin/bash
#SBATCH --job-name=step11c2
#SBATCH --output=slurm_logs/step11c2_%j.out
#SBATCH --error=slurm_logs/step11c2_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
cd /lisc/home/user/gerardursin/multiplex/
echo "Step 11c: Weaving partition ensemble into a HiDeF persistent hierarchy (hidef.weaver)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
python step11c_phase2_weave.py \
    --consensus 11_hidef_results_faithful_cpm/consensus.pkl \
    --output-prefix 11_hidef_results_faithful_cpm/heterogeneous_hidef_faithful_cpm \
    --cutoff 0.75
echo "Finished at: $(date)"