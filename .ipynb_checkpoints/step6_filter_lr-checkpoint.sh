#!/bin/bash
#SBATCH --job-name=step6_extract
#SBATCH --output=slurm_logs/step6_extract_%j.out
#SBATCH --error=slurm_logs/step6_extract_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 6: Extract LR + Reactome from CSV"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step6_filter_lr_pairs.py \
    --ppi-pkl 00_inputs/ppi_network.pkl \
    --integrated-multiplex 03_integrated/integrated_multiplex_output_LR/integrated_multiplex.pkl \
    --wdir 04_multixrank_prep/multixrank_input_consolidated \
    --validate-against-ppi
    
echo "Finished at: $(date)"