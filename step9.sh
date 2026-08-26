#!/bin/bash
#SBATCH --job-name=step9_mp
#SBATCH --output=slurm_logs/step9_mp_%j.out
#SBATCH --error=slurm_logs/step9_mp_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
cd /lisc/home/user/gerardursin/multiplex/
echo "Step 9: Marchenko-Pastur analysis"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate
mkdir -p slurm_logs

# --graph/--registry (old integrated_multiplex.pkl + node_registry.csv) are
# gone -- see step8_spectral_clustering.slurm's comment for why. Point
# --ref-graph at the same gene_reference_graph.pkl used there.
python step9_multiplex_marchenko_pastur.py \
    --module-graph 06_disease_module/disease_module_output/module_graph.pkl \
    --out-dir 09_marchenko_pastur \
    --omega-sweep 0.1 1.0 5.0 \
    --min-celltypes-present 3 \
    --frac-minority-threshold 0.6 \
    --n-perm 10000
    
echo "Finished at: $(date)"
