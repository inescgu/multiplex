#!/bin/bash
#SBATCH --job-name=chidef
#SBATCH --output=slurm_logs/chidef_%j.out
#SBATCH --error=slurm_logs/chidef_%j.err
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
module load Python/3.13.5-GCCcore-14.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

# Run the job
echo "Running HiDeF clustering..."
python comparable_hidef_multiplex.py \
    --graph_pkl integrated_multiplex_output/integrated_multiplex.pkl \
    --registry_csv integrated_multiplex_output/node_registry.csv
    
echo "Finished at: $(date)"
#    --graph_pkl disease_module_output/module_graph.pkl \
