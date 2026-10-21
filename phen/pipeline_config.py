"""
pipeline_config.py

Single source of truth for paths and thresholds used across every analyzer
and validation script in the heterogeneous (multiplex) pipeline.

Before this file existed, NODES_FILE / SUPRA_GRAPH_PKL / SEED_GENES_FILE
were hardcoded, independently, inside:
  - analyze_proteomics.py       (run_analysis)
  - analyze_phenotypes.py       (run_analysis)
  - analyze_neuropathic_pain.py (run_analysis)
  - run_multiplex_enrichment_pipeline.py (module-level CONFIG block)
If you changed the persistence threshold or reran step 11c with a new
--neighbors value, you had to remember to edit all four in sync (the
driver script even has a comment warning about this). Now: edit ONCE here.

Every script in this bundle (fix1..fix6, the DAG analysis, the loader)
imports from this file. Update the paths below to match your actual
directory layout, then never touch them again in the individual scripts.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent   # multiplex/
# ---------------------------------------------------------------------
# Heterogeneous (multiplex) pipeline
# ---------------------------------------------------------------------
PPI_SOURCE = "combined26_multiplex"
CLUSTER_TYPE = "hidef_from_comparison"

# The persistence threshold used to decide which HiDeF clusters are even
# considered "real" for downstream enrichment / validation. Keep this the
# SAME value everywhere -- it silently changes which clusters exist.
PERSISTENCE_THRESHOLD = 8

# ---------------------------------------------------------------------
# Multiple-testing correction (used by multiple_testing.py -> every analyzer)
# ---------------------------------------------------------------------
# One method, one alpha, for ALL enrichment analyses. Options:
#   "fdr_bh"      Benjamini-Hochberg FDR (default; standard for enrichment)
#   "fdr_by"      Benjamini-Yekutieli FDR (valid under arbitrary dependence)
#   "bonferroni"  family-wise error, most conservative
#   "holm"        family-wise error, uniformly more powerful than Bonferroni
MULTIPLE_TESTING_METHOD = "fdr_bh"
ALPHA = 0.05

# hidef_finder Phase-2 weave output (step11c_phase2_weave.py)
NODES_FILE = str(ROOT / "11_hidef_module" / "results_cpm" / "heterogeneous_hidef_faithful_cpm.nodes")
EDGES_FILE = str(ROOT / "11_hidef_module" / "results_cpm"  / "heterogeneous_hidef_faithful_cpm.edges")

# step11a output (supra-graph) and step11b output (partition ensemble)
SUPRA_GRAPH_PKL = str(ROOT / "11_hidef_module" / "supra_graph.pkl")
ENSEMBLE_PKL = str(ROOT / "11_hidef_module" / "ensemble.pkl")

# step6_build_disease_module.py output dir (must contain disease_module.csv)
DISEASE_MODULE_DIR = str(ROOT / "06_disease_module")

# step2.5 output -- full namespaced reference network ("<celltype>::GENE")
REFERENCE_GRAPH_PKL = str(ROOT / "04_multixrank_prep" / "multixrank_hetero_input_pseudobulk" / "gene_reference_graph.pkl")
SEED_GENES_FILE = str(ROOT / "00_inputs" / "seed_genes.csv")

# The --neighbors value actually used for the .nodes/.edges files above.
# Record it here so every downstream script that reports on "the current
# result" can print it, instead of you having to remember which sbatch
# script produced which output.
HIDEF_NEIGHBORS_USED = 7

OUTPUT_DIR = str(HERE / "output")
# ---------------------------------------------------------------------
# Single-layer (monoplex) pipeline -- fill in per PPI source
# ---------------------------------------------------------------------
MONOPLEX_HIDEF_COMMUNITIES_PKL = str(ROOT /"hidef_results" / "hidef_communities.pkl")
MONOPLEX_DISEASE_MODULE_DIR = str(ROOT / "temp")          # contains graph_lcc.pkl etc.
MONOPLEX_RWR_OUTPUT_DIR = str(ROOT / "output")            # rwr/disease_module.csv etc.


def loader_kwargs():
    """Common kwargs for hidef_multiplex_adapter.load_multiplex_clusters /
    patch_analyzer_for_multiplex -- every caller passes the same thing, so
    build it in one place."""
    return dict(
        nodes_file=NODES_FILE,
        supra_graph_pkl=SUPRA_GRAPH_PKL,
        seed_genes_file=SEED_GENES_FILE,
        persistence_threshold=PERSISTENCE_THRESHOLD,
    )


def check_paths(verbose=True):
    """Sanity check: warn about any configured path that doesn't exist yet,
    rather than failing deep inside some analyzer with a confusing error."""
    paths = {
        "NODES_FILE": NODES_FILE,
        "EDGES_FILE": EDGES_FILE,
        "SUPRA_GRAPH_PKL": SUPRA_GRAPH_PKL,
        "ENSEMBLE_PKL": ENSEMBLE_PKL,
        "DISEASE_MODULE_DIR": DISEASE_MODULE_DIR,
        "REFERENCE_GRAPH_PKL": REFERENCE_GRAPH_PKL,
        "SEED_GENES_FILE": SEED_GENES_FILE,
    }
    missing = [k for k, v in paths.items() if not Path(v).exists()]
    if verbose:
        for k, v in paths.items():
            flag = "OK " if Path(v).exists() else "MISSING"
            print(f"  [{flag}] {k}: {v}")
    return missing


if __name__ == "__main__":
    print("pipeline_config paths:")
    missing = check_paths()
    if missing:
        print(f"\n{len(missing)} path(s) missing -- edit pipeline_config.py before running "
              f"any downstream script: {missing}")