#!/bin/bash
#SBATCH --job-name=step11c_weave_hierarchy
#SBATCH --output=slurm_logs/step11c_weave_hierarchy_%j.out
#SBATCH --error=slurm_logs/step11c_weave_hierarchy_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=130G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
echo "Step 11c: Weaving partition ensemble into a HiDeF persistent hierarchy (hidef.weaver)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate
python step11c_weave.py \
    --ensemble 11_hidef_hetero/ensemble.pkl \
    --output-dir 11_hidef_results \
    --cutoff 0.75 \
    --min-cluster-size 10 \
    --max-size 4680
echo "Finished at: $(date)"