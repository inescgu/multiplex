#!/bin/bash
#SBATCH --job-name=step9v
#SBATCH --output=slurm_logs/step9v_%j.out
#SBATCH --error=slurm_logs/step9v_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail
echo "Step 9: Marchenko-Pastur Viz"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

# --graph/--registry (old integrated_multiplex.pkl + node_registry.csv) are
# gone -- see step8_spectral_clustering.slurm's comment for why. Point
# --ref-graph at the same gene_reference_graph.pkl used there.
python step9viz_mp_visuals.py \
    --mp-dir 09_marchenko_pastur \
    --omegas 0.1 1.0 5.0 \
    --gene ADCYAP1 \
    --top-n 30 \
    --out-dir 09_marchenko_pastur/slide_figs
    
echo "Finished at: $(date)"
