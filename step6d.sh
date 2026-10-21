#!/bin/bash
#SBATCH --job-name=step6d
#SBATCH --output=slurm_logs/step6d_%j.out
#SBATCH --error=slurm_logs/step6d_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail

echo "Export for datadivr"
echo "Running on host: $(hostname)"; echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

python step6d_export_context_for_datadivr.py \
  --ref-graph 04_multixrank_prep/multixrank_hetero_input_pseudobulk/gene_reference_graph.pkl \
  --module-dir 06_disease_module/disease_module_output_pseudobulk \
  --out 06_disease_module/datadivr \
  --scores-csv 05_multixrank_results/multixrank_pseudo_results/gene_rwr_scores.csv
echo "Finished at: $(date)"