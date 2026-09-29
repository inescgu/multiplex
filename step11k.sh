#!/bin/bash
#SBATCH --job-name=step11k
#SBATCH --output=slurm_logs/step11k_%j.out
#SBATCH --error=slurm_logs/step11k_%j.err
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

python step11k_dag_nesting_analysis.py \
        --nodes-file heterogeneous_hidef_faithful_cpm.nodes \
        --edges-file heterogeneous_hidef_faithful_cpm.edges \
        --output-prefix characterized/heterogeneous_hidef_dag \
        --enrichment-csv output/combined26_multiplex/.../combined_gene_results_multiplex.tsv \
        --enrichment-label-col top_cell_type --enrichment-pval-col adjusted_p_value

echo "Finished at: $(date)"