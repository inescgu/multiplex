#!/bin/bash
#SBATCH --job-name=step2_bipartite
#SBATCH --output=slurm_logs/step2_bipartite_%j.out
#SBATCH --error=slurm_logs/step2_bipartite_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail
echo "Step 2: Building LR (+ Reactome, + scored LIANA/NeuronChat) bipartite edges"
echo "        between cell types, plus identity bipartite edges connecting"
echo "        global_ppi to everything"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate

python step2_build_bipartite_edges.py \
    --wdir 04_multixrank_prep/multixrank_hetero_input_pseudobulk \
    --lr-csv pathway_prep/lr_pairs_with_pathways.csv \
    --scored-lr-csv pathway_prep/lr_sources/liana_consensus_top.csv pathway_prep/lr_sources/neuronchat_long.csv \
    --ppi-pkl 00_inputs/ppi_network.pkl \
    --global-ppi-name global_ppi \
    --exclude-identity-from-charts

# --scored-lr-csv adds the new directional, cell-type-specific LR edges from
# step0a (LIANA) and step0b (NeuronChat), on top of your existing flat
# --lr-csv resource. Multi-subunit complexes (e.g. "ITGA4_ITGB1") are split
# into individual gene subunits automatically; pass --no-split-complexes to
# turn that off.
#
# To also add Reactome-pathway-sharing edges between cell types, add:
#   --reactome-csv pathway_prep/reactome_interactions_with_pathways_v2.csv
#   --uniprot-map 00_inputs/uniprot_to_symbol.csv
#
# --global-ppi-name global_ppi is the default, spelled out here so it's
# obvious this wires step 1's --base-ppi layer to every other cell type via
# identity edges. Pass --no-global-ppi-bipartite if you ran step 1 without
# --base-ppi, or want global_ppi left unconnected.

echo "Finished at: $(date)"