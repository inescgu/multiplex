#!/bin/bash
#SBATCH --job-name=build_ppi
#SBATCH --output=slurm_logs/build_ppi_%j.out
#SBATCH --error=slurm_logs/build_ppi_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Builds a fresh consensus PPI with ../PPI_builder (settings in its config.py:
# all 5 sources, min_sources 1, LCC) and converts it to the pipeline's pickle
# format as 00_inputs/ppi_network_<date>.pkl. Does NOT swap it in -- check the
# comparison at the end of the log, then:
#     ln -sfn ppi_network_<date>.pkl 00_inputs/ppi_network.pkl
cd /lisc/home/user/gerardursin/multiplex/
set -eo pipefail
echo "Building consensus PPI with PPI_builder"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

module load Python/3.12.3-GCCcore-13.3.0
source "$HOME/.local/bin/env"            # uv

cd ../PPI_builder
echo "PPI_builder commit: $(git rev-parse --short HEAD)"
uv sync
MARK=$(mktemp); touch "$MARK"
uv run python main.py --fetch --generate-mappings

# main.py logs per-source errors instead of failing -- stop here if any happened
ERRS=$(find reports -name '*.log' -newer "$MARK" -exec grep -H "ERROR" {} + || true)
if [ -n "$ERRS" ]; then echo "$ERRS"; echo "PPI_builder reported errors (above) -- not converting."; exit 1; fi

TSV=$(ls -t outputs/consensus_ppi_symbol_combined_lcc_*.tsv | head -1)
D=$(basename "$TSV" .tsv | grep -oE '[0-9]{8}$')
echo "Built: $TSV"

cd ../multiplex
source /lisc/data/scratch/menche/ines/multixrank_env/bin/activate
python ppi_tsv_to_pkl.py --tsv "../PPI_builder/$TSV" \
    --out "00_inputs/ppi_network_$D.pkl" \
    --compare 00_inputs/ppi_network.pkl \
    --seeds 00_inputs/seed_genes.csv

echo
echo "To use it:  ln -sfn ppi_network_$D.pkl 00_inputs/ppi_network.pkl"
echo "Finished at: $(date)"
