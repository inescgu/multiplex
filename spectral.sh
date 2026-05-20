#!/bin/bash
#SBATCH --job-name=spec
#SBATCH --output=slurm_logs/spec_%j.out
#SBATCH --error=slurm_logs/spec_%j.err
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
    --out-dir   ./spectral/ \
    --scheme       hybrid \
    --mode     sos  --sample-p 0.4 --k-max 25
    
echo "Finished at: $(date)"
#    --graph_pkl disease_module_output/module_graph.pkl \
