#!/bin/bash
#SBATCH --job-name=step11c
#SBATCH --output=slurm_logs/step11c_%j.out
#SBATCH --error=slurm_logs/step11c_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/

echo "Step 11b: Multi-resolution (+coupling) partition sweep over the heterogeneous supra-graph"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate


python step11c_neighbors_sweep.py \
        --ensemble 11_hidef_hetero/ensemble.pkl \
        --neighbor-values 2 3 5 10 15 20 \
        --reference-neighbors 10 \
        --minsize 4 --jaccard 0.75 --chi 5 --consensus-p 75 \
        --output-dir 11_hidef_neighbors_sweep
echo "Finished at: $(date)"