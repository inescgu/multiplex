#!/bin/bash
#SBATCH --job-name=str
#SBATCH --output=slurm_logs/str_%j.out
#SBATCH --error=slurm_logs/str_%j.err
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
echo "Running first core periph..."
python multiplex_core_periphery.py \
    --multiplex-pkl integrated_multiplex_output/integrated_multiplex.pkl \
    --output-dir    mpnsm_output_full \
    --intra-edge-types ppi \
    --weighted-virtual \
    --seed-genes    ../og_gene_set/seed_genes.csv \
    --mw-alternative less \
    --alpha 10 --p 1.4 --q 1.4 --max-iter 300

python multiplex_core_periphery.py \
    --multiplex-pkl integrated_multiplex_output/integrated_multiplex.pkl \
    --output-dir    mpnsm_output_intra \
    --no-virtual-pathway --no-virtual-lr
    
echo "Finished at: $(date)"
#    --graph_pkl disease_module_output/module_graph.pkl \
