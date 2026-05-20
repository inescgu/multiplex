#!/bin/bash
#SBATCH --job-name=mp
#SBATCH --output=slurm_logs/mp_%j.out
#SBATCH --error=slurm_logs/mp_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1

echo "Started at: $(date)"
# Change to your working directory
cd /lisc/home/user/gerardursin/multiplex/

# Add current directory to Python path
#export PYTHONPATH="/lisc/home/user/gerardursin/og_gene_set:$PYTHONPATH"

# Load any required modules here
module load Python/3.14.4-weekly  
source /lisc/data/scratch/menche/ines/general_env/bin/activate

# Run the job
echo "Running HiDeF per layer clustering..."
python multiplex_spectral_clustering.py \
    --graph    integrated_multiplex_output/integrated_multiplex.pkl \
    --registry integrated_multiplex_output/node_registry.csv \
    --out-dir   ./marchenko/ \
    --sample-p 0.4 --omega 1.0 --n-eigs 400 --min-neg-layers 2
    
echo "Finished at: $(date)"
#    --graph_pkl disease_module_output/module_graph.pkl \
