#!/usr/bin/env python3
"""
hidef_multiplex_facets.py

Extension on top of hidef_multiplex_adapter.py: turns the per-cluster
cell-type detail already sitting in `extra` (from load_multiplex_clusters)
into facet columns you can actually use -- on the enrichment testers'
output CSVs, and in the heatmap/lollipop plots.

WHY A SEPARATE, POST-HOC STEP
------------------------------
The analyzers (OMIMPhenotypicSeriesAnalyzer, ProteomicsAnalyzer, etc.)
only ever compute a statistic per (cluster, term) pair. Cell-type
composition is a property of the CLUSTER, not of any one enrichment test,
so it doesn't belong inside analyze_enrichment() -- it belongs bolted on
afterwards, once, and reused by every analysis type and every plot. Doing
it this way also means you never have to touch the analyzer classes again
if you add a 6th analysis type later.

Two use cases are covered:
  1. `annotate_summary_csv` / `annotate_gene_csv` -- add facet columns to
     an analyzer's *_summarized_results.csv and *_gene_level_results.csv
     in place (writes a sibling *_with_celltype_facets.csv, originals
     untouched so ResultsCombiner keeps working off the unmodified files
     unless you point it at the new ones).
  2. `facet_all_analysis_outputs` -- do (1) for every analysis type in one
     call, using the same directory-naming convention ResultsCombiner
     already relies on.
  3. Filter/grouping helpers (`filter_to_crosscelltype`,
     `celltype_color_map`, `add_celltype_facet_to_plot_df`) for use
     directly inside create_enrichment_heatmap /
     create_persistence_lollipop_plot.

USAGE
-----
    from hidef_multiplex_adapter import load_multiplex_clusters
    from hidef_multiplex_facets import compute_cluster_facets, facet_all_analysis_outputs

    clusters, cluster_seed_mapping, extra = load_multiplex_clusters(
        nodes_file='11_hidef_results_faithful/heterogeneous_hidef_faithful.nodes',
        node_names=node_names, seed_genes=seed_genes, persistence_threshold=5,
    )
    facets_df = compute_cluster_facets(clusters, extra)
    facets_df.to_csv('output/combined26/celltype_facets.csv')  # keep for later

    facet_all_analysis_outputs(
        output_dir='./output', ppi_source='combined26', facets_df=facets_df,
        analysis_types=['neuropathic_pain', 'omim', 'orphanet', 'phenotypes',
                         'proteomics', 'spatial'],
    )
"""
import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================================
# 1. CLUSTER-LEVEL FACET TABLE
# ============================================================================

def compute_cluster_facets(clusters: dict, extra: dict) -> pd.DataFrame:
    """One row per cluster: how many cell types it spans, which one
    dominates, how much of it is conserved (same gene, multiple cell
    types). All computed off `extra` from load_multiplex_clusters -- no
    need to re-touch the .nodes file or the supra-graph.

    Columns
    -------
    cluster_id              : matches the 'cluster_id' column every
                               analyzer's *_summarized_results.csv /
                               *_gene_level_results.csv already uses
    n_cell_types             : distinct cell types contributing genes
    dominant_cell_type       : cell type contributing the most distinct genes
    dominant_cell_type_frac  : that count / n_distinct_genes
    n_distinct_genes         : size of the deduplicated gene set (matches
                                len(clusters[cluster_id]))
    n_conserved_genes        : genes shared by >=2 cell types in this cluster
    conserved_gene_frac      : n_conserved_genes / n_distinct_genes
    is_cross_celltype        : n_cell_types >= 2 (convenience bool)
    persistence               : carried over from `extra['persistence']`
    """
    rows = []
    genes_by_ct = extra["genes_by_celltype"]
    conserved = extra["conserved_genes"]
    persistence = extra.get("persistence", {})

    for cluster_id, gene_list in clusters.items():
        n_distinct = len(gene_list)
        by_ct = genes_by_ct.get(cluster_id, {})
        n_ct = len(by_ct)
        if n_ct == 0:
            continue
        dom_ct = max(by_ct, key=lambda ct: len(by_ct[ct]))
        dom_frac = len(by_ct[dom_ct]) / n_distinct if n_distinct else 0.0
        n_cons = len(conserved.get(cluster_id, {}))
        cons_frac = n_cons / n_distinct if n_distinct else 0.0

        rows.append({
            "cluster_id": cluster_id,
            "n_cell_types": n_ct,
            "dominant_cell_type": dom_ct,
            "dominant_cell_type_frac": round(dom_frac, 4),
            "n_distinct_genes": n_distinct,
            "n_conserved_genes": n_cons,
            "conserved_gene_frac": round(cons_frac, 4),
            "is_cross_celltype": n_ct >= 2,
            "persistence": persistence.get(cluster_id, np.nan),
        })

    return pd.DataFrame(rows)


def invert_gene_celltypes(extra: dict) -> dict:
    """cluster_id -> {gene: [cell_types]} for EVERY gene (not just
    conserved ones) -- built from extra['genes_by_celltype']. Used to
    annotate gene-level result rows with which cell type(s) that specific
    gene came from within that specific cluster."""
    out = {}
    for cluster_id, by_ct in extra["genes_by_celltype"].items():
        gene_to_cts = defaultdict(list)
        for ct, genes in by_ct.items():
            for g in genes:
                gene_to_cts[g].append(ct)
        out[cluster_id] = {g: sorted(cts) for g, cts in gene_to_cts.items()}
    return out


# ============================================================================
# 2. ANNOTATE EXISTING ANALYZER OUTPUT CSVs
# ============================================================================

def annotate_summary_csv(summary_csv_path, facets_df, cluster_id_col="cluster_id",
                          out_suffix="_with_celltype_facets"):
    """Merge cell-type facet columns onto an analyzer's
    *_summarized_results.csv. Writes a sibling file; original untouched."""
    if not os.path.exists(summary_csv_path):
        print(f"  [skip] not found: {summary_csv_path}")
        return None

    df = pd.read_csv(summary_csv_path)
    if cluster_id_col not in df.columns:
        print(f"  [skip] no '{cluster_id_col}' column in {summary_csv_path}")
        return None

    merged = df.merge(
        facets_df.drop(columns=["persistence"], errors="ignore"),
        on=cluster_id_col, how="left",
    )
    n_unmatched = merged["n_cell_types"].isna().sum()
    if n_unmatched:
        print(f"  ⚠ {n_unmatched}/{len(merged)} rows in {os.path.basename(summary_csv_path)} "
              f"had no matching facet row (cluster_id mismatch? check naming/prefix)")

    out_path = summary_csv_path.replace(".csv", f"{out_suffix}.csv")
    merged.to_csv(out_path, index=False)
    print(f"  ✓ {out_path}  ({len(merged)} rows, {n_unmatched} unmatched)")
    return merged


def annotate_gene_csv(gene_csv_path, extra, cluster_id_col="cluster_id", gene_col="gene",
                       out_suffix="_with_celltype_facets"):
    """Merge per-gene cell-type provenance onto an analyzer's
    *_gene_level_results.csv: which cell type(s) that gene token came
    from within that specific cluster, and whether it's conserved there."""
    if not os.path.exists(gene_csv_path):
        print(f"  [skip] not found: {gene_csv_path}")
        return None

    df = pd.read_csv(gene_csv_path)
    if cluster_id_col not in df.columns or gene_col not in df.columns:
        print(f"  [skip] missing '{cluster_id_col}'/'{gene_col}' columns in {gene_csv_path}")
        return None

    gene_ct_map = invert_gene_celltypes(extra)
    conserved = extra["conserved_genes"]

    def lookup_cts(row):
        cts = gene_ct_map.get(row[cluster_id_col], {}).get(row[gene_col])
        return ";".join(cts) if cts else ""

    def lookup_conserved(row):
        return row[gene_col] in conserved.get(row[cluster_id_col], {})

    df["cell_types_in_cluster"] = df.apply(lookup_cts, axis=1)
    df["n_cell_types_for_gene"] = df["cell_types_in_cluster"].apply(
        lambda s: len(s.split(";")) if s else 0
    )
    df["is_conserved_in_cluster"] = df.apply(lookup_conserved, axis=1)

    out_path = gene_csv_path.replace(".csv", f"{out_suffix}.csv")
    df.to_csv(out_path, index=False)
    n_unresolved = (df["n_cell_types_for_gene"] == 0).sum()
    print(f"  ✓ {out_path}  ({len(df)} rows, {n_unresolved} genes not found in facet map)")
    return df


# ============================================================================
# 3. DO IT FOR EVERY ANALYSIS TYPE AT ONCE
# ============================================================================

# Same file-naming convention ResultsCombiner already uses.
_SUMMARY_FILE_MAP = {
    "proteomics": "proteomics_summarized_results.csv",
    "neuropathic_pain": "neuropathic_pain_summarized_results.csv",
    "orphanet": "orphanet_enrichr_summarized_results.csv",
    "omim": "omim_ps_summarized_results.csv",
    "phenotypes": "phenotype_summarized_results.csv",
    "spatial": "spatial_transcriptomics_hidef_comparison_summarized_results.csv",
}
_GENE_FILE_MAP = {
    "proteomics": "proteomics_gene_level_results.csv",
    "neuropathic_pain": "neuropathic_pain_gene_level_results.csv",
    "orphanet": "orphanet_enrichr_gene_level_results.csv",
    "omim": "omim_ps_gene_level_results.csv",
    "phenotypes": "phenotype_gene_level_results.csv",
    "spatial": "spatial_transcriptomics_hidef_comparison_gene_level_results.csv",
}
_DIR_NAME_MAP = {
    "proteomics": "proteomics_analysis",
    "neuropathic_pain": "neuropathic_pain_analysis",
    "orphanet": "orphanet_enrichr_analysis",
    "omim": "omim_ps_analysis",
    "phenotypes": "phenotypes_analysis",
    "spatial": "spatial_analysis",
}


def facet_all_analysis_outputs(output_dir, ppi_source, facets_df, extra,
                                analysis_types=None, cluster_type="hidef_from_comparison"):
    """Annotate every analyzer's summary + gene-level CSV with cell-type
    facets in one call. Mirrors ResultsCombiner.find_analysis_directories()
    so it finds the same directories your analyzers already wrote."""
    analysis_types = analysis_types or list(_SUMMARY_FILE_MAP)
    base_path = os.path.join(output_dir, ppi_source)

    for analysis_type in analysis_types:
        dir_pattern = _DIR_NAME_MAP.get(analysis_type)
        if dir_pattern is None:
            print(f"[{analysis_type}] unknown analysis type, skipping")
            continue
        matching_dirs = glob.glob(os.path.join(base_path, f"{dir_pattern}_{cluster_type}*"))
        if not matching_dirs:
            print(f"[{analysis_type}] no output directory found, skipping")
            continue
        analysis_dir = matching_dirs[0]
        print(f"\n[{analysis_type}] {analysis_dir}")

        summary_file = _SUMMARY_FILE_MAP.get(analysis_type)
        if summary_file:
            annotate_summary_csv(os.path.join(analysis_dir, summary_file), facets_df)

        gene_file = _GENE_FILE_MAP.get(analysis_type)
        if gene_file:
            annotate_gene_csv(os.path.join(analysis_dir, gene_file), extra)


# ============================================================================
# 4. HELPERS FOR THE HEATMAP / LOLLIPOP PLOTS
# ============================================================================

def filter_to_crosscelltype(df, facets_df, min_cell_types=2, cluster_id_col="cluster_id"):
    """Keep only rows whose cluster spans >= min_cell_types cell types --
    drop this into create_enrichment_heatmap / the lollipop script right
    after the existing significance/size filters, to focus either plot on
    genuinely cross-cell-type communication modules rather than
    single-cell-type PPI blocks that happen to score well."""
    keep_ids = set(facets_df.loc[facets_df["n_cell_types"] >= min_cell_types, "cluster_id"])
    before = len(df)
    out = df[df[cluster_id_col].isin(keep_ids)].copy()
    print(f"Cross-cell-type filter (>= {min_cell_types} cell types): {before} -> {len(out)} rows")
    return out


def celltype_color_map(facets_df, palette=None):
    """Consistent color per dominant cell type, for coloring cluster
    x-axis labels/ticks in the heatmap or lollipop plot so the reader can
    see at a glance which cell type each cluster is anchored in.
    Returns {cell_type: hex_color}."""
    import matplotlib.pyplot as plt

    cell_types = sorted(facets_df["dominant_cell_type"].dropna().unique())
    if palette is None:
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i / max(len(cell_types) - 1, 1)) for i in range(len(cell_types))]
    return dict(zip(cell_types, palette))


def add_celltype_facet_to_plot_df(plot_df, facets_df, cluster_id_col="cluster_id"):
    """Merge n_cell_types / dominant_cell_type / is_cross_celltype onto a
    plotting dataframe (e.g. the `plot_df` built inside
    create_persistence_lollipop_plot, or `heatmap_df` inside
    create_enrichment_heatmap) so you can facet the figure -- e.g. split
    into two panels (single-cell-type vs cross-cell-type modules), or set
    marker style by is_cross_celltype without a second lookup pass."""
    facet_cols = ["cluster_id", "n_cell_types", "dominant_cell_type", "is_cross_celltype"]
    return plot_df.merge(facets_df[facet_cols], on=cluster_id_col, how="left")