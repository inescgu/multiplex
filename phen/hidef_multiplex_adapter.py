"""
hidef_multiplex_adapter.py

ONE canonical implementation of:
  - load_multiplex_clusters()          -- parse a hidef_finder .nodes file
  - patch_analyzer_for_multiplex()     -- bolt multiplex clusters onto an
                                           analyzer that only knows the old
                                           monoplex load_hidef_clusters_from_
                                           comparison() interface
  - MultiplexClusterLoaderMixin        -- inherit this instead of copy-
                                           pasting load_hidef_clusters_
                                           multiplex() into every analyzer

BEFORE: analyze_proteomics.py, analyze_phenotypes.py, and
analyze_neuropathic_pain.py each defined their own byte-identical
load_hidef_clusters_multiplex() method. That's a landmine -- you already
found one bug in this exact code path (the namespace mismatch documented
in step6's docstring); if you ever fix a second one, you now have to
remember to patch it in three places, and it's easy to patch two and ship
the third silently wrong.

AFTER: every analyzer either
  (a) inherits MultiplexClusterLoaderMixin and calls
      self.load_hidef_clusters_multiplex()  [no args -- reads
      pipeline_config.py], or
  (b) for analyzers you don't want to touch (e.g. SpatialTranscriptomics
      Analyzer, which still has the OLD load_hidef_clusters_from_
      comparison() method reading the monoplex hidef_communities.pkl),
      gets patched in-place with patch_analyzer_for_multiplex(), exactly
      as run_multiplex_enrichment_pipeline.py already assumes exists.

USAGE (new analyzer code)
--------------------------
    from hidef_multiplex_adapter import MultiplexClusterLoaderMixin

    class ProteomicsAnalyzer(MultiplexClusterLoaderMixin):
        ...
        def run_analysis(self):
            if not self.load_hidef_clusters_multiplex():   # <- no args now
                return False
            ...

USAGE (patch an analyzer you don't want to edit)
--------------------------------------------------
    from hidef_multiplex_adapter import patch_analyzer_for_multiplex
    import pipeline_config as cfg

    analyzer = SpatialTranscriptomicsAnalyzer(...)
    patch_analyzer_for_multiplex(analyzer, **cfg.loader_kwargs())
    analyzer.run_analysis()
"""
import pickle
import re
from collections import Counter

import pandas as pd


def _gene_of(node: str) -> str:
    return node.rsplit("::", 1)[-1] if "::" in node else node


def _celltype_of(node: str) -> str:
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


def load_multiplex_clusters(nodes_file, node_names=None, seed_genes=None,
                             persistence_threshold=5, min_size=2):
    """
    Parse a hidef_finder .nodes file (community, size, members, persistence
    -- tab-separated, no header) into the same three objects every analyzer
    expects: self.clusters, self.cluster_seed_mapping, self._multiplex_extra.

    `members` tokens may be either "cell_type::gene" strings directly, or
    integer indices into `node_names` (the supra-graph's vertex order) --
    both are handled, same sniffing logic as step11e_characterize_clusters.py's
    parse_nodes_file(). Gene identity is DE-NAMESPACED (deduplicated across
    cell types) here, because every downstream enrichment analyzer (proteomics,
    phenotypes, neuropathic pain, spatial) tests against gene-level annotation
    data (protein symbols, phenotype gene lists) -- a namespaced
    "Ab-LTMR::TRPV1" node has to become plain "TRPV1" before it can be
    compared against expression_data.index or gene_phenotypes.keys().

    Returns
    -------
    clusters : dict  cluster_name -> sorted list of unique bare gene symbols
    cluster_seed_mapping : dict  cluster_name -> set of seed genes present
    extra : dict with per-cluster diagnostics you'd otherwise have to
            recompute later: 'persistence', 'namespaced_members' (the
            original, non-deduplicated "ct::gene" list, useful if you need
            cell-type composition), 'n_cell_types'.
    """
    df = pd.read_csv(nodes_file, sep="\t", header=None,
                      names=["cluster_id", "size", "members", "persistence"],
                      dtype={"members": str})

    clusters, cluster_seed_mapping = {}, {}
    extra = {"persistence": {}, "namespaced_members": {}, "n_cell_types": {}}
    n_skipped_persistence, n_skipped_empty, n_skipped_small = 0, 0, 0

    for _, row in df.iterrows():
        if pd.isna(row["persistence"]) or float(row["persistence"]) < persistence_threshold:
            n_skipped_persistence += 1
            continue

        raw_field = row["members"]
        if pd.isna(raw_field) or not str(raw_field).strip():
            n_skipped_empty += 1
            continue
        raw = [tok for tok in re.split(r"[,\s]+", str(raw_field).strip()) if tok]
        if not raw:
            n_skipped_empty += 1
            continue

        looks_like_indices = all(tok.lstrip("-").isdigit() for tok in raw[:5])
        if looks_like_indices and node_names is not None:
            try:
                members = [node_names[int(tok)] for tok in raw]
            except (IndexError, ValueError):
                members = raw
        else:
            members = raw

        if len(members) < min_size:
            n_skipped_small += 1
            continue

        genes = sorted({_gene_of(m) for m in members})
        cluster_name = f"HiDeF_Cluster_{row['cluster_id']}"

        clusters[cluster_name] = genes
        cluster_seed_mapping[cluster_name] = (set(genes) & seed_genes) if seed_genes else set()
        extra["persistence"][cluster_name] = float(row["persistence"])
        extra["namespaced_members"][cluster_name] = members
        extra["n_cell_types"][cluster_name] = len({_celltype_of(m) for m in members if "::" in m})

    print(f"load_multiplex_clusters({nodes_file}): kept {len(clusters)} clusters "
          f"(persistence >= {persistence_threshold}); skipped "
          f"{n_skipped_persistence} below threshold, {n_skipped_empty} empty, "
          f"{n_skipped_small} below min_size={min_size}")

    return clusters, cluster_seed_mapping, extra


def patch_analyzer_for_multiplex(analyzer, nodes_file, supra_graph_pkl=None,
                                  seed_genes_file=None, persistence_threshold=5):
    """
    Bolt multiplex clusters onto an analyzer instance that only implements
    the OLD monoplex-era load_hidef_clusters_from_comparison() interface
    (e.g. SpatialTranscriptomicsAnalyzer as currently written). Sets
    analyzer.clusters / .cluster_seed_mapping / .seed_genes / ._multiplex_extra
    directly and monkeypatches the load method to a no-op that returns True,
    so analyzer.run_analysis() proceeds without modification.
    """
    node_names = None
    if supra_graph_pkl:
        with open(supra_graph_pkl, "rb") as f:
            g = pickle.load(f)
        node_names = g.vs["name"]

    seed_genes = None
    if seed_genes_file:
        seed_genes = set(pd.read_csv(seed_genes_file)["gene"].astype(str))

    clusters, cluster_seed_mapping, extra = load_multiplex_clusters(
        nodes_file, node_names=node_names, seed_genes=seed_genes,
        persistence_threshold=persistence_threshold,
    )

    analyzer.clusters = clusters
    analyzer.cluster_seed_mapping = cluster_seed_mapping
    analyzer._multiplex_extra = extra
    analyzer.seed_genes = (
        set().union(*cluster_seed_mapping.values()) if cluster_seed_mapping else set()
    )

    def _already_loaded(*args, **kwargs):
        return bool(analyzer.clusters)

    analyzer.load_hidef_clusters_from_comparison = _already_loaded
    analyzer.load_hidef_clusters_multiplex = _already_loaded
    return analyzer


class MultiplexClusterLoaderMixin:
    """
    Inherit this in ProteomicsAnalyzer / PhenotypesAnalyzer /
    NeuropathicPainAnalyzer instead of each defining its own copy of
    load_hidef_clusters_multiplex(). Requires self.persistence_threshold
    to already be set (every analyzer's __init__ already does this).

    Old call site (inside run_analysis, x3, hardcoded paths):
        if not self.load_hidef_clusters_multiplex(
            nodes_file='../11_hidef_results_faithful_cpm/heterogeneous_hidef_faithful_cpm.nodes',
            supra_graph_pkl='../11_hidef_hetero/supra_graph.pkl',
            seed_genes_file='../00_inputs/seed_genes.csv'):

    New call site:
        if not self.load_hidef_clusters_multiplex():
    """

    def load_hidef_clusters_multiplex(self, nodes_file=None, supra_graph_pkl=None,
                                       seed_genes_file=None):
        import pipeline_config as cfg
        kw = cfg.loader_kwargs()
        nodes_file = nodes_file or kw["nodes_file"]
        supra_graph_pkl = supra_graph_pkl or kw["supra_graph_pkl"]
        seed_genes_file = seed_genes_file or kw["seed_genes_file"]
        persistence_threshold = getattr(self, "persistence_threshold", kw["persistence_threshold"])

        node_names = None
        if supra_graph_pkl:
            with open(supra_graph_pkl, "rb") as f:
                g = pickle.load(f)
            node_names = g.vs["name"]

        seed_genes = None
        if seed_genes_file:
            seed_genes = set(pd.read_csv(seed_genes_file)["gene"].astype(str))

        self.clusters, self.cluster_seed_mapping, self._multiplex_extra = load_multiplex_clusters(
            nodes_file, node_names=node_names, seed_genes=seed_genes,
            persistence_threshold=persistence_threshold,
        )
        self.seed_genes = (
            set().union(*self.cluster_seed_mapping.values()) if self.cluster_seed_mapping else set()
        )

        if self.clusters:
            sizes = [len(g) for g in self.clusters.values()]
            print(f"Loaded {len(self.clusters)} multiplex clusters "
                  f"(gene-set sizes {min(sizes)}-{max(sizes)}, avg {sum(sizes)/len(sizes):.1f})")
        return bool(self.clusters)