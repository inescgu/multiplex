#!/bin/bash
#SBATCH --job-name=dm_cons
#SBATCH --output=slurm_logs/dm_cons_%j.out
#SBATCH --error=slurm_logs/dm_cons_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=35G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/

echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

# echo "=========================================="
# echo "Step 0a: Consolidate layers"
# echo "=========================================="
# python consolidate_layers.py \
#     --wdir          multixrank_input_consolidated \
#     --output-wdir   multixrank_input_consolidated
# echo ""

# echo "=========================================="
# echo "Step 0b: Filter LR pairs"
# echo "=========================================="
# python filter_lr_pairs.py \
#     --lr-pairs      pathway_distance/human_lr_pairs.csv \
#     --wdir          multixrank_input_consolidated \
#     --output        multixrank_input_consolidated/multiplex/gene_multiplex/ligand_receptor_layer.tsv \
#     --ligand-col    human_ligand \
#     --receptor-col  human_receptor \
#     --sep           $'\t'
# echo ""

echo "=========================================="
echo "Step 0c: Rebuild reference graphs"        # ← NEW
echo "=========================================="
python rebuild_reference_graphs.py \
    --wdir          multixrank_input_consolidated \
    --original-pkl  integrated_multiplex_output/integrated_multiplex.pkl
echo ""

# echo "=========================================="
# echo "Step 2: Run MultiXrank"
# echo "=========================================="
# python dm2_run_multixrank_v2.py \
#     --wdir          multixrank_input_consolidated \
#     --config        config.yml \
#     --output-dir    multixrank_results_consolidated
# echo ""

# echo "=========================================="
# echo "Step 3: Build disease module"
# echo "=========================================="
# python dm3_build_disease_module.py \
#     --scores                multixrank_results_consolidated/gene_rwr_scores.csv \
#     --per-celltype-scores   multixrank_results_consolidated/per_celltype_scores.csv \
#     --input-dir             multixrank_input_consolidated \
#     --output-dir            disease_module_output_consolidated \
#     --connectivity          all \
#     --k                     400 \
#     --max-bridge            400 \
# #     --exclude-seeds         SSTR4
# echo ""

# echo "=========================================="
# echo "Finished at: $(date)"
# echo "=========================================="