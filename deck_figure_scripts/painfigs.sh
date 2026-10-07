#!/bin/bash
#SBATCH --job-name=painfig
#SBATCH --output=slurm_logs/pf_%j.out
#SBATCH --error=slurm_logs/pf_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

BASE=/lisc/home/user/gerardursin/multiplex
cd $BASE/deck_figure_scripts

echo "Pain figs"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
module load Python/3.10.4-GCCcore-11.3.0
source /lisc/data/scratch/menche/ines/hidef_env/bin/activate

OUT=deck_figures

# Slide 12, left: category-level bars (step 7)
python fig_step7_topology.py \
    --step7-dir $BASE/07_analysis/topology_output \
    --out-dir $OUT \
    --figsize 5.9 1.95 \
    --rename "unknown=Global PPI (bridge)" "Non-classified=ATF3 (injured)"
#   add --scatter for fig12c_size_vs_density.png (needs 01_table_layer_metrics.csv)

# Slide 12, right: fiber-type density heatmap (step 7c)
python fig_step7c_density.py \
    --step7c-dir $BASE/07_analysis/interlayer_connectivity \
    --out-dir $OUT \
    --figsize 3.2 2.9

# Slide 15: the nulls (step 10b)
python fig_step10b_nulls.py \
    --step10b-dir $BASE/06_disease_module/functional_permutation_test \
    --out-dir $OUT

# Slides 17-19: neighbour profiles (step 7d, neighbor-mode run)
python fig_step7d_neighbors.py \
    --step7d-dir $BASE/07d_module_neighbors \
    --genes-file gene_evidence.csv \
    --out-dir $OUT

# LCC significance per cell type (step 7b)
python fig_step7b_lcc.py \
    --step7b-dir $BASE/07b_layer_lcc_analysis \
    --out-dir $OUT \
    --n-sims 1000
#   -> fig_lcc_by_celltype.png  (rename with --name to match the slide)

python fig_step12_enrichment.py grid \
    --table ../phen/output/combined26_multiplex/cluster_enrichment_long.csv \
    --tier 1 --max-clusters 20 --out figs

python fig_step12_enrichment.py gene \
    --tsv ../phen/combined_gene_results_multiplex_significant_only.tsv --out figs