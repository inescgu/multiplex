#!/usr/bin/env python3
"""
run_multiplex_enrichment_pipeline.py

End-to-end driver: runs all five enrichment testers against the
heterogeneous multiplex HiDeF output (steps 11a-11c), annotates results
with cell-type facets, combines everything with ResultsCombiner. Produces
the enrichment CSVs -- plotting is handled separately by
run_multiplex_visualizations.py (your `phen_viz` sbatch job), so this script
stays focused on analysis and doesn't spend this job's runtime re-rendering
plots every time.

It assumes the analyzer classes (ProteomicsAnalyzer, PhenotypesAnalyzer,
NeuropathicPainAnalyzer, SpatialTranscriptomicsAnalyzer,
OMIMPhenotypicSeriesAnalyzer) and common_utils.py are importable from your
existing scripts -- adjust the imports below to match wherever those
files live.
"""
import os
import pandas as pd

# --- your existing analyzer classes (unmodified) ---
from analyze_proteomics import ProteomicsAnalyzer
from analyze_phenotypes import PhenotypesAnalyzer
from analyze_neuropathic_pain import NeuropathicPainAnalyzer
from analyze_spatial_transcriptomics import SpatialTranscriptomicsAnalyzer
# from analyze_omim_ps import OMIMPhenotypicSeriesAnalyzer  # wherever this class lives

# --- new modules from this conversation ---
from hidef_multiplex_adapter import load_multiplex_clusters, patch_analyzer_for_multiplex
from hidef_multiplex_facets import compute_cluster_facets, facet_all_analysis_outputs
from results_combiner_multiplex import ResultsCombiner


# ============================================================================
# CONFIG -- edit this block
# ============================================================================

PPI_SOURCE = 'combined26_multiplex'      # distinct from your monoplex 'combined26'
CLUSTER_TYPE = 'hidef_from_comparison'   # kept the same so file-naming patterns still match;
                                          # PPI_SOURCE alone separates the two pipelines' folders
OUTPUT_DIR = './output'
PERSISTENCE_THRESHOLD = 8                # floor for which clusters get enrichment-tested at
                                          # all in this run. NOTE: run_multiplex_visualizations.py
                                          # reads this same value separately (as
                                          # ANALYZER_PERSISTENCE_THRESHOLD) for its "top clusters"
                                          # plot -- keep the two in sync if you change this.

NODES_FILE = '../11_hidef_results_faithful_cpm/heterogeneous_hidef_faithful_cpm.nodes'
# ^ MUST match NODES_FILE in run_multiplex_visualizations.py. Cluster IDs and
# persistence values in the output CSVs are tied to whichever .nodes file was
# in effect when this script last ran -- if you change this, rerun this whole
# pipeline before trusting anything the visualization script plots.
SUPRA_GRAPH_PKL = '../11_hidef_hetero/supra_graph.pkl'
SEED_GENES_FILE = '../00_inputs/seed_genes.csv'

# Data files for each analysis type -- same files you already use.
PROTEOMICS_FILE = '../../expression_data/matrix-for-limma2.csv'
PROTEOMICS_METADATA_FILE = '../../expression_data/colData-for-limma.csv'
PHENOTYPES_FILE = '../../expression_data/human_pain_genes_database.csv'
NEUROPATHIC_PAIN_FILE = '../../expression_data/neuropathic_pain_drg2.csv'
SPATIAL_FILE = '../../expression_data/spatial_transcriptomics_drg.csv'
OMIM_PS_FILE = '../../phenotypic_series/omim_phenotypic_series_enrichment.csv'


# ============================================================================
# STEP 1 -- load clusters once, build the facet table once
# ============================================================================

print("=" * 80)
print("STEP 1: loading multiplex clusters + computing cell-type facets")
print("=" * 80)

seed_genes = set(pd.read_csv(SEED_GENES_FILE)["gene"].astype(str)) if SEED_GENES_FILE else None

clusters, cluster_seed_mapping, extra = load_multiplex_clusters(
    NODES_FILE,
    seed_genes=seed_genes,
    persistence_threshold=PERSISTENCE_THRESHOLD,
)
facets_df = compute_cluster_facets(clusters, extra)
facets_out_path = os.path.join(OUTPUT_DIR, PPI_SOURCE, 'celltype_facets.csv')
os.makedirs(os.path.dirname(facets_out_path), exist_ok=True)
facets_df.to_csv(facets_out_path, index=False)
print(f"Saved facet table -> {facets_out_path}\n")


# ============================================================================
# STEP 2 -- run each analyzer, patched to read the multiplex clustering
# ============================================================================

print("=" * 80)
print("STEP 2: running enrichment analyzers")
print("=" * 80)


def _run(analyzer):
    patch_analyzer_for_multiplex(
        analyzer,
        nodes_file=NODES_FILE,
        supra_graph_pkl=SUPRA_GRAPH_PKL,
        seed_genes_file=SEED_GENES_FILE,
        persistence_threshold=PERSISTENCE_THRESHOLD,
    )
    ok = analyzer.run_analysis()
    print(f"  -> {'OK' if ok else 'FAILED'}\n")
    return ok


proteomics = ProteomicsAnalyzer(
    ppi_source=PPI_SOURCE, proteomics_file=PROTEOMICS_FILE,
    input_dir=OUTPUT_DIR, output_dir=OUTPUT_DIR, cluster_type=CLUSTER_TYPE,
    metadata_file=PROTEOMICS_METADATA_FILE, persistence_threshold=PERSISTENCE_THRESHOLD,
)
_run(proteomics)

phenotypes = PhenotypesAnalyzer(
    ppi_source=PPI_SOURCE, expression_file=PHENOTYPES_FILE,
    input_dir=OUTPUT_DIR, output_dir=OUTPUT_DIR, cluster_type=CLUSTER_TYPE,
    persistence_threshold=PERSISTENCE_THRESHOLD,
)
_run(phenotypes)

neuropathic_pain = NeuropathicPainAnalyzer(
    ppi_source=PPI_SOURCE, expression_file=NEUROPATHIC_PAIN_FILE,
    input_dir=OUTPUT_DIR, output_dir=OUTPUT_DIR, cluster_type=CLUSTER_TYPE,
    persistence_threshold=PERSISTENCE_THRESHOLD,
)
_run(neuropathic_pain)

spatial = SpatialTranscriptomicsAnalyzer(
    ppi_source=PPI_SOURCE, expression_file=SPATIAL_FILE,
    input_dir=OUTPUT_DIR, output_dir=OUTPUT_DIR,
    persistence_threshold=PERSISTENCE_THRESHOLD,
)
_run(spatial)

# omim = OMIMPhenotypicSeriesAnalyzer(
#     ppi_source=PPI_SOURCE, input_dir=OUTPUT_DIR, output_dir=OUTPUT_DIR,
#     omim_ps_file=OMIM_PS_FILE, persistence_threshold=PERSISTENCE_THRESHOLD,
# )
# _run(omim)

# NOTE: orphanet isn't in the documents shared in this conversation, but if
# you have an OrphanetEnrichrAnalyzer with the same load_hidef_clusters_from_
# comparison() + run_analysis() shape, patch and run it exactly the same way.


# ============================================================================
# STEP 3 -- annotate every analyzer's output with cell-type facets
# ============================================================================

print("=" * 80)
print("STEP 3: annotating outputs with cell-type facets")
print("=" * 80)

facet_all_analysis_outputs(
    output_dir=OUTPUT_DIR, ppi_source=PPI_SOURCE, facets_df=facets_df, extra=extra,
    analysis_types=['proteomics', 'neuropathic_pain', 'phenotypes', 'spatial', 'omim', 'orphanet'],  # add 'omim', 'orphanet' once run
    cluster_type=CLUSTER_TYPE,
)


# ============================================================================
# STEP 4 -- combine
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: combining results across analyses")
print("=" * 80)

combiner = ResultsCombiner(output_dir=OUTPUT_DIR, ppi_source=PPI_SOURCE, cluster_type=CLUSTER_TYPE)
if combiner.process_all_analyses():
    combiner.save_combined_results('combined_gene_results_multiplex.tsv', p_value_threshold=0.05)


print("\nDone with analysis. Run run_multiplex_visualizations.py (separately,")
print("via your phen_viz sbatch job) to generate plots from these results.")