#!/bin/bash
#SBATCH --job-name=step11i
#SBATCH --output=slurm_logs/step11i_%j.out
#SBATCH --error=slurm_logs/step11i_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
echo "Step 11g"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

python step11i_cluster_level_null_model.py \
        --reference-graph gene_reference_graph.pkl \
        --disease-module-dir 06_disease_module \
        --n-perms 50 \
        --min-resolution 0.01 --max-resolution 10 --n-resolutions 30 \
        --neighbors 3 \
        --output-dir 11_hidef_null_models

echo "Finished at: $(date)"