#!/usr/bin/env python3
"""
hidef_multiplex_adapter.py

Adapter that lets the existing single-layer enrichment testers
(analyze_proteomics.py, analyze_phenotypes.py, analyze_neuropathic_pain.py,
analyze_spatial_transcriptomics_hidef_v2.py, OMIMPhenotypicSeriesAnalyzer)
run UNMODIFIED against the heterogeneous multiplex HiDeF output (steps
11a-11c) instead of the old single-layer hidef_communities.pkl.

WHY THIS EXISTS
---------------
Every analyzer's load_hidef_clusters_from_comparison() ultimately just
needs to populate two attributes:

    self.clusters             : {cluster_name: [gene, gene, ...]}
    self.cluster_seed_mapping : {cluster_name: {seed_gene, ...}}

Everything downstream (Fisher's/hypergeometric tests, plotting, the
ResultsCombiner, the lollipop/heatmap scripts) only ever touches genes by
plain symbol against gene-indexed resources (expression matrices, OMIM/
Orphanet gene lists, phenotype gene sets) that know nothing about cell
type. So this adapter's job is narrow: read the heterogeneous .nodes
file, strip every "<cell_type>::GENE" member down to GENE, DEDUPLICATE
within each cluster (a gene expressed in 3 cell types inside one cluster
must count once, not 3x -- otherwise every downstream Fisher's/
hypergeometric test's denominator gets silently inflated and p-values
shrink for a reason that has nothing to do with real enrichment), and
hand back exactly the two dicts above -- plus a side-channel of
cell-type-resolved detail for anyone who wants it later without
re-parsing the .nodes file.

USAGE
-----
Drop this file next to your analyzer scripts. In each analyzer class, add
ONE new method that calls load_multiplex_clusters() instead of
load_hidef_clusters_from_comparison() -- see the patch example at the
bottom of this file (identical pattern applies to all five analyzers).
"""
import re
from collections import defaultdict

import pandas as pd


def _parse_member_token(token: str):
    """Split a 'cell_type::gene' token on the LAST '::', not the first.

    Some upstream data doubles the cell-type prefix -- e.g. the supra-graph
    itself can contain vertex names like
    'global_ppi::global_ppi::A1BG' instead of 'global_ppi::A1BG' -- because
    the per-cell-type gene identifiers fed into step 11a's own
    f"{ct}::{gene}" node-naming line were ALREADY namespaced before that
    line ran, so step 11a's naming doubled it. This is a real upstream
    data issue (check whether your ppi_layer.tsv / bipartite edge files'
    gene columns already contain "cell_type::" before step 11a runs), not
    something this adapter can fix at the source -- but it can recover
    the correct gene regardless of how many times the prefix got
    duplicated: gene symbols never legitimately contain '::', so whatever
    comes after the LAST '::' is always the real gene. The recovered
    cell-type label is then collapsed if it's an exact self-repeated
    'X::X' pattern, purely for readability in `extra['genes_by_celltype']`
    etc. -- it doesn't affect which gene gets extracted.

    Falls back to (None, token) for anything without '::' at all --
    callers should treat ct is None as 'unresolved, drop it'."""
    if "::" not in token:
        return None, token
    ct, gene = token.rsplit("::", 1)
    parts = ct.split("::")
    if len(parts) == 2 and parts[0] == parts[1]:
        ct = parts[0]
    return ct, gene


def load_multiplex_clusters(
    nodes_file,
    node_names=None,
    seed_genes=None,
    persistence_threshold=5,
    min_size_after_dedup=2,
    cluster_name_prefix="HiDeF_Cluster_",
):
    """
    Read a hidef_finder .nodes file from the heterogeneous pipeline
    (step 11c Phase 2 output) and return clusters in the exact shape the
    existing single-layer analyzers expect.

    Parameters
    ----------
    nodes_file : str or Path
        The <prefix>.nodes file from step 11c Phase 2.
    node_names : list[str], optional
        g.vs["name"] from the step 11a supra-graph pickle. Only needed if
        the .nodes file's members column contains integer vertex indices
        instead of "cell_type::gene" strings (same ambiguity step 11e's
        parser handles). Pass it whenever you have the supra-graph pickle
        handy -- otherwise index-encoded rows are dropped rather than
        guessed at.
    seed_genes : set[str], optional
        Flat set of seed gene symbols (e.g. the same seed_genes.csv used
        upstream in step 11a/11e). If omitted, cluster_seed_mapping falls
        back to "every distinct gene in the cluster is its own seed" --
        the same fallback the old loader used
        (`seed_genes = comm_data.get('seed_genes', members)`).
    persistence_threshold : int
        Same semantics as every analyzer's existing --persistence_threshold.
    min_size_after_dedup : int
        Drop clusters whose DISTINCT gene count (after collapsing
        cell-type duplicates) falls below this. Checked AFTER dedup, not
        against the raw multiplex member count -- a cluster with 6
        members that are really just 2 genes recurring across 3 cell
        types is a 2-gene cluster for a flat, cell-type-agnostic test.
    cluster_name_prefix : str
        Kept as "HiDeF_Cluster_" by default because downstream code (the
        ResultsCombiner, the lollipop/heatmap scripts) strips this exact
        prefix to recover the raw cluster id. Change it only if you also
        update those strip calls.

    Returns
    -------
    clusters : dict[str, list[str]]
        cluster_name -> deduplicated flat gene list (drop-in replacement
        for the old pickle's 'members' list).
    cluster_seed_mapping : dict[str, set[str]]
        cluster_name -> seed genes present in that cluster.
    extra : dict
        Side-channel detail NOT consumed by the existing analyzers --
        join it back onto results afterwards instead of re-parsing the
        .nodes file a second time:
          - 'genes_by_celltype'   : {cluster_name: {cell_type: [genes]}}
          - 'n_cell_types'        : {cluster_name: int}
          - 'dominant_cell_type'  : {cluster_name: str}
          - 'conserved_genes'     : {cluster_name: {gene: [cell_types]}}
          - 'persistence'         : {cluster_name: float}
          - 'raw_members'         : {cluster_name: [full "ct::gene" tokens]}
    """
    df = pd.read_csv(
        nodes_file, sep="\t", header=None,
        names=["cluster_id", "size", "members", "persistence"],
        dtype={"members": str},
    )

    clusters, cluster_seed_mapping = {}, {}
    genes_by_celltype, n_cell_types, dominant_cell_type = {}, {}, {}
    conserved_genes, persistence_map, raw_members_map = {}, {}, {}

    n_filtered_persistence = 0
    n_filtered_size = 0
    n_skipped_unresolved = 0

    for _, row in df.iterrows():
        persistence = float(row["persistence"])
        if persistence < persistence_threshold:
            n_filtered_persistence += 1
            continue

        raw_field = row["members"]
        if pd.isna(raw_field) or not str(raw_field).strip():
            continue
        raw_tokens = [t for t in re.split(r"[,\s]+", str(raw_field).strip()) if t]
        if not raw_tokens:
            continue

        # Same index-vs-name sniffing step 11e's parser uses.
        looks_like_indices = all(t.lstrip("-").isdigit() for t in raw_tokens[:5])
        if looks_like_indices:
            if node_names is None:
                n_skipped_unresolved += 1
                continue  # can't resolve indices without the supra-graph's node list
            try:
                members = [node_names[int(t)] for t in raw_tokens]
            except (IndexError, ValueError):
                n_skipped_unresolved += 1
                continue
        else:
            members = raw_tokens

        # Strip "cell_type::gene" -> gene, and dedup.
        by_ct = defaultdict(set)
        gene_to_cts = defaultdict(set)
        for tok in members:
            ct, gene = _parse_member_token(tok)
            if ct is None:
                continue
            by_ct[ct].add(gene)
            gene_to_cts[gene].add(ct)

        # Sanity check only -- _parse_member_token's rsplit-on-last-'::'
        # already handles arbitrarily duplicated cell-type prefixes, so
        # this should never fire in normal operation. If it does, it means
        # a "gene" contains '::' for some OTHER reason (a genuinely
        # malformed row, a gene symbol that itself contains '::', etc.) --
        # worth investigating rather than silently shipping a broken token
        # into every downstream enrichment test.
        if any("::" in g for g in gene_to_cts):
            n_bad = sum(1 for g in gene_to_cts if "::" in g)
            print(f"  ⚠ cluster {row['cluster_id']}: {n_bad} member(s) still contained "
                  f"'::' after parsing -- forcing a second strip pass. This means some "
                  f"tokens in this row didn't match the expected 'cell_type::gene' "
                  f"format on the first pass; investigate the raw .nodes row for this "
                  f"cluster_id if this keeps happening.")
            fixed_gene_to_cts = defaultdict(set)
            for g, cts in gene_to_cts.items():
                g2 = g.split("::")[-1] if "::" in g else g
                fixed_gene_to_cts[g2] |= cts
            gene_to_cts = fixed_gene_to_cts

        distinct_genes = sorted(gene_to_cts)
        if len(distinct_genes) < min_size_after_dedup:
            n_filtered_size += 1
            continue

        cluster_id = str(row["cluster_id"])
        cluster_name = f"{cluster_name_prefix}{cluster_id}"

        clusters[cluster_name] = distinct_genes

        if seed_genes is not None:
            cluster_seed_mapping[cluster_name] = set(distinct_genes) & set(seed_genes)
        else:
            cluster_seed_mapping[cluster_name] = set(distinct_genes)

        genes_by_celltype[cluster_name] = {ct: sorted(g) for ct, g in by_ct.items()}
        n_cell_types[cluster_name] = len(by_ct)
        dominant_cell_type[cluster_name] = max(by_ct, key=lambda ct: len(by_ct[ct]))
        conserved_genes[cluster_name] = {
            g: sorted(cts) for g, cts in gene_to_cts.items() if len(cts) >= 2
        }
        persistence_map[cluster_name] = persistence
        raw_members_map[cluster_name] = members

    print(f"[multiplex adapter] {len(clusters):,} clusters kept from {nodes_file} "
          f"(dropped {n_filtered_persistence:,} below persistence={persistence_threshold}, "
          f"{n_filtered_size:,} below min_size_after_dedup={min_size_after_dedup}, "
          f"{n_skipped_unresolved:,} with unresolved index-form members)")

    extra = {
        "genes_by_celltype": genes_by_celltype,
        "n_cell_types": n_cell_types,
        "dominant_cell_type": dominant_cell_type,
        "conserved_genes": conserved_genes,
        "persistence": persistence_map,
        "raw_members": raw_members_map,
    }
    return clusters, cluster_seed_mapping, extra


# ============================================================================
# NO-SOURCE-EDIT PATCH: monkey-patch the instance, not the class
# ============================================================================

def patch_analyzer_for_multiplex(analyzer, nodes_file, supra_graph_pkl=None,
                                  seed_genes_file=None, persistence_threshold=None,
                                  min_size_after_dedup=2):
    """
    Monkey-patch an already-constructed analyzer instance (Proteomics/
    Phenotypes/NeuropathicPain/SpatialTranscriptomics/
    OMIMPhenotypicSeriesAnalyzer) so its existing run_analysis() call to
    load_hidef_clusters_from_comparison() transparently loads the
    heterogeneous multiplex .nodes file instead of the old
    hidef_communities.pkl -- with NO edits to any of the analyzer scripts.

    Why this works: every analyzer's run_analysis() does, as its first
    step,
        if not self.load_hidef_clusters_from_comparison():
            return False
    and only ever touches self.clusters / self.cluster_seed_mapping /
    self.seed_genes afterward. Rebinding load_hidef_clusters_from_comparison
    on the INSTANCE (not the class) is enough.

    IMPORTANT: `ppi_source` / `input_dir` / `cluster_type` passed to the
    analyzer's constructor are NOT used anywhere to locate PPI or cluster
    data -- they are purely folder-naming labels for
    output_dir/<ppi_source>/<analysis>_<cluster_type>/. Point THIS
    function's nodes_file / supra_graph_pkl / seed_genes_file at your
    actual multiplex outputs; the constructor args can stay whatever you
    want your output folders called -- just pick a ppi_source/cluster_type
    that's DIFFERENT from your monoplex run's, or you'll silently
    overwrite/interleave the two pipelines' results in the same folders.

    Parameters
    ----------
    analyzer : an already-constructed analyzer instance
    nodes_file : path to the heterogeneous pipeline's .nodes file
        (step 11c Phase 2 output)
    supra_graph_pkl : path to the step 11a supra-graph pickle, only needed
        if your .nodes file's members are integer indices rather than
        "cell_type::gene" strings
    seed_genes_file : CSV with a 'gene' column -- the same seed gene list
        used upstream (e.g. in step 11a/11e). If omitted, every distinct
        gene in a cluster is treated as its own seed (matches the old
        loader's fallback behavior for the monoplex pickle).
    persistence_threshold : overrides analyzer.persistence_threshold if
        given (see the recalibration note above -- your multiplex sweep
        almost certainly has a different partition count than your
        monoplex HiDeF run, so the same numeric threshold means something
        different).
    """
    import types

    node_names = None
    if supra_graph_pkl:
        import pickle
        with open(supra_graph_pkl, "rb") as f:
            g = pickle.load(f)
        node_names = g.vs["name"]

    seed_genes = None
    if seed_genes_file:
        seed_genes = set(pd.read_csv(seed_genes_file)["gene"].astype(str))

    pthresh = (persistence_threshold if persistence_threshold is not None
               else analyzer.persistence_threshold)

    def _load_multiplex(self):
        self.clusters, self.cluster_seed_mapping, self._multiplex_extra = load_multiplex_clusters(
            nodes_file, node_names=node_names, seed_genes=seed_genes,
            persistence_threshold=pthresh, min_size_after_dedup=min_size_after_dedup,
        )
        self.seed_genes = (
            set().union(*self.cluster_seed_mapping.values())
            if self.cluster_seed_mapping else set()
        )
        if self.clusters:
            sizes = [len(v) for v in self.clusters.values()]
            print(f"Loaded {len(self.clusters)} multiplex clusters "
                  f"(deduped gene-set sizes {min(sizes)}-{max(sizes)}, "
                  f"avg {sum(sizes)/len(sizes):.1f})")
        return bool(self.clusters)

    analyzer.load_hidef_clusters_from_comparison = types.MethodType(_load_multiplex, analyzer)
    return analyzer