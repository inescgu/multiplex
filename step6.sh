#!/bin/bash
#SBATCH --job-name=step6_dm
#SBATCH --output=slurm_logs/step6_dm_%j.out
#SBATCH --error=slurm_logs/step6_dm_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Step 6: Building disease module"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step6_build_disease_module.py \
    --scores 05_multixrank_results/multixrank_hetero_results/gene_rwr_scores.csv \
    --per-celltype-scores 05_multixrank_results/multixrank_hetero_results/per_celltype_scores.csv \
    --input-dir 04_multixrank_prep/multixrank_hetero_input \
    --output-dir 06_disease_module/disease_module_output \
    --connectivity all \
    --k 200 \
    --max-bridge 30000 \
    --lcc-per-layer \
    --exclude-ppi-seeds \
    --exclude-seeds A-PEP.NTRK3_S100A16::TRPV1 A-PROPR.HAPLN4::TRPV1 ATF3::CNR1 ATF3::HTR2A AB-LTMR.CCKAR::SYN2 AB-LTMR.CCKAR::TRPV1 AB-LTMR.NSG2::TRPV1 C-NP.MRGPRX1_GFRA2::GLA C-NP.MRGPRX1_GFRA2::GRM4 C-NP.MRGPRX1_GFRA2::KCNQ2 C-NP.MRGPRX1_MRGPRX4::BDKRB2 C-NP.MRGPRX1_MRGPRX4::GLA C-NP.MRGPRX1_MRGPRX4::GRM4 C-NP.SST::BDKRB2 C-PEP.ADORA2B::GRM4 C-PEP.TAC1_CACNG5::TRPV1 C-PEP.TAC1_CHRNA3::TRPV1 C-THERMO.TRPM8::KCNQ2

echo "Finished at: $(date)"