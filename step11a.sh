#!/bin/bash
#SBATCH --job-name=step11a_supra_graph
#SBATCH --output=slurm_logs/step11a_supra_graph_%j.out
#SBATCH --error=slurm_logs/step11a_supra_graph_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
source ./run_env.sh

echo "Step 11a: Building heterogeneous supra-graph (all cell types + bipartite edges, no flattening)"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

snapshot 11_hidef_module

python step11a_supragraph.py \
    --wdir 04_multixrank_prep/multixrank_hetero_input \
    --module-csv 06_disease_module/disease_module_output/disease_module.csv \
    --output 11_hidef_module/supra_graph.pkl
    
echo "Finished at: $(date)"