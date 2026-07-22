#!/bin/bash
#SBATCH --job-name=pre9
#SBATCH --output=slurm_logs/pre_step9_%j.out
#SBATCH --error=slurm_logs/pre_step9_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Pre Step 9 Connectivity Check"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python pre_step9check.py \
    --ref-graph 04_multixrank_prep/multixrank_input_consolidated/gene_reference_graph.pkl \
    --seeds 04_multixrank_prep/multixrank_input_consolidated/seeds/seeds.txt \
    --lcc-per-layer
    
echo "Finished at: $(date)"