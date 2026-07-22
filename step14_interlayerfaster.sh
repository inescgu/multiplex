#!/bin/bash
#SBATCH --job-name=step14_interlayer
#SBATCH --output=slurm_logs/step14_interlayer_%j.out
#SBATCH --error=slurm_logs/step14_interlayer_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 14: Analyzing inter-layer edges"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

# --no-plots gives you the CSVs + text report in a single fast pass over the
# graph (seconds to a couple minutes instead of hours). Drop the flag once
# you want the PNG plots too - they add matplotlib/seaborn overhead but the
# core analysis (edge counts, annotations, density) is unaffected either way.
python step14_interlayerfaster.py \
    --multiplex-pkl 03_integrated/integrated_multiplex_output_LR/integrated_multiplex.pkl \
    --node-registry-csv 03_integrated/integrated_multiplex_output_LR/node_registry.csv \
    --exclude-sources ts \
    --cell-types-csv neurons_only.csv \
    --drg-hierarchy-only \
    --output-dir 03_integrated/interlayer_analysis_neurons \
    2>&1 | tee slurm_logs/step14_interlayer_$(date +%s).log

echo "Finished at: $(date)"