#!/usr/bin/env python3
"""
Build Multiplex Interactome from Cell-Type Marker Genes (2025 Hrdg atlas)
Using CO-EXPRESSION to define edges (NOT PPI).

This script creates a multiplex network where:
- Each layer represents a cell type
- Nodes are marker genes for that cell type that are present in the expression matrix
- Edges are co-expression links (Pearson r >= threshold) between marker genes,
  computed from a provided expression matrix (cells × genes)
- Seed genes are overlaid and tracked across layers

NOTE: Because edges are derived from co-expression rather than a PPI, you must
supply an expression matrix alongside the marker gene list.  The expression
matrix should be a CSV (or pickle of a pd.DataFrame) with:
  - rows = cells / samples
  - columns = gene symbols (uppercase)
Use --expr-matrix to pass this file.  Supported formats:
  .csv / .tsv  → read with pandas
  .pkl / .pickle → unpickled pd.DataFrame
"""

import os
import sys
import argparse
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
import time
import json


# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================

DEFAULT_CORR_THRESHOLD = 0.3   # Pearson r threshold for a co-expression edge


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_celltype_markers(markers_file, p_val_threshold=0.05, log2fc_threshold=0.0):
    """
    Load cell-type marker genes from CSV file.

    Parameters
    ----------
    markers_file      : path to CSV with columns including gene and cell_type
    p_val_threshold   : (unused by default; filtering commented out as in original)
    log2fc_threshold  : (unused by default)

    Returns
    -------
    celltype_markers : dict  cell_type → list of marker gene symbols (uppercase)
    markers_df       : pd.DataFrame (full, unfiltered)
    """
    print(f"Loading cell-type markers from {markers_file}...")

    markers_df = pd.read_csv(markers_file)

    print(f"  Loaded {len(markers_df)} total marker entries")
    print(f"  Columns: {list(markers_df.columns)}")

    gene_col = None
    celltype_col = None

    for col in ['gene', 'Gene', 'gene_id', 'Gene_ID', 'symbol', 'Symbol']:
        if col in markers_df.columns:
            gene_col = col
            break

    for col in ['cell_type', 'Cell_Type', 'celltype', 'CellType', 'cluster']:
        if col in markers_df.columns:
            celltype_col = col
            break

    if gene_col is None or celltype_col is None:
        print(f"  ERROR: Could not identify gene or cell_type columns")
        print(f"  Available columns: {list(markers_df.columns)}")
        return None, None

    print(f"  Using gene column: '{gene_col}'")
    print(f"  Using cell_type column: '{celltype_col}'")

    # Convert gene names to uppercase
    markers_df[gene_col] = markers_df[gene_col].astype(str).str.upper()

    # Group by cell type
    celltype_markers = {}
    for cell_type in markers_df[celltype_col].unique():
        genes = markers_df[markers_df[celltype_col] == cell_type][gene_col].tolist()
        genes = [g.strip() for g in genes if g.strip()]
        celltype_markers[cell_type] = genes
        print(f"  {cell_type}: {len(genes)} marker genes")

    return celltype_markers, markers_df


def load_expression_matrix(expr_file: str) -> pd.DataFrame:
    """
    Load an expression matrix (cells × genes).

    Supported formats
    -----------------
    .csv / .tsv  : pandas read_csv (auto-detects separator)
    .pkl / .pickle : unpickled pd.DataFrame

    Returns
    -------
    pd.DataFrame with gene symbols (uppercase) as columns
    """
    print(f"\nLoading expression matrix from {expr_file} ...")

    ext = Path(expr_file).suffix.lower()
    if ext in (".pkl", ".pickle"):
        with open(expr_file, "rb") as f:
            df = pickle.load(f)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame in pickle, got {type(df)}")
    elif ext in (".csv",):
        df = pd.read_csv(expr_file, index_col=0)
    elif ext in (".tsv",):
        df = pd.read_csv(expr_file, sep="\t", index_col=0)
    else:
        # Try CSV as fallback
        df = pd.read_csv(expr_file, index_col=0)

    # Normalise column names to uppercase gene symbols
    df.columns = [str(c).strip().upper() for c in df.columns]

    print(f"  Expression matrix loaded: {df.shape[0]:,} cells × {df.shape[1]:,} genes")
    return df


def load_seed_genes(seed_genes_file: str) -> set:
    if not os.path.exists(seed_genes_file):
        print(f"Seed genes file not found: {seed_genes_file}")
        return set()
    try:
        seed_df = pd.read_csv(seed_genes_file)
        gene_col = None
        for col in ['gene', 'Gene', 'gene_id', 'Gene_ID', 'symbol', 'Symbol', 'SYMBOL']:
            if col in seed_df.columns:
                gene_col = col
                break
        if gene_col is None:
            gene_col = seed_df.columns[0]
        seed_genes = set(seed_df[gene_col].dropna().astype(str).str.strip().str.upper())
        print(f"Loaded {len(seed_genes)} seed genes from {seed_genes_file}")
        return seed_genes
    except Exception as e:
        print(f"Error loading seed genes: {e}")
        return set()


# ============================================================================
# CO-EXPRESSION LAYER BUILDER
# ============================================================================

def build_coexpression_layer(
    cell_type: str,
    marker_genes: list,
    expr_df: pd.DataFrame,
    seed_genes: set = None,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    corr_method: str = "pearson",
) -> nx.Graph:
    """
    Build a co-expression graph for one cell type from the expression matrix.

    Only marker genes that are present as columns in expr_df are used.
    Edges are added between gene pairs whose Pearson (or Spearman) correlation
    coefficient meets or exceeds corr_threshold.

    Parameters
    ----------
    cell_type      : name of the cell type (used for logging and node attributes)
    marker_genes   : list of gene symbols for this cell type
    expr_df        : pd.DataFrame (cells × genes), all cell types combined
    seed_genes     : optional set of seed genes to annotate
    corr_threshold : minimum r value to add an edge
    corr_method    : 'pearson' or 'spearman'

    Returns
    -------
    nx.Graph
    """
    print(f"\nBuilding co-expression layer for: {cell_type}")

    marker_set = set(str(g).upper() for g in marker_genes)
    available = [g for g in marker_set if g in expr_df.columns]

    print(f"  Marker genes: {len(marker_set)}")
    print(f"  Available in expression matrix: {len(available)}")

    if len(available) < 2:
        print(f"  ❌  Fewer than 2 marker genes found in expression matrix — empty layer")
        return nx.Graph()

    # Subset expression matrix to this cell type's marker genes
    sub_expr = expr_df[available]

    # Compute correlation matrix
    corr = sub_expr.corr(method=corr_method)

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(available)

    n_edges = 0
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            r = corr.iloc[i, j]
            if pd.notna(r) and r >= corr_threshold:
                G.add_edge(available[i], available[j], weight=float(r))
                n_edges += 1

    # Add node attributes
    for gene in G.nodes():
        G.nodes[gene]['cell_type'] = cell_type
        G.nodes[gene]['is_marker'] = True
        G.nodes[gene]['is_seed'] = bool(seed_genes and gene in seed_genes)

    density = nx.density(G)
    avg_deg = 2 * n_edges / len(available) if available else 0
    print(f"  Nodes: {G.number_of_nodes():,} | Edges: {n_edges:,} | "
          f"density={density:.4f} | avg_degree={avg_deg:.2f}")

    if seed_genes:
        seeds_in = seed_genes.intersection(set(G.nodes()))
        print(f"  Seed genes in layer: {len(seeds_in)} / {len(seed_genes)} "
              f"({len(seeds_in)/len(seed_genes)*100:.1f}%)")
        if len(seeds_in) > 1:
            seed_subgraph = G.subgraph(seeds_in)
            n_comp = nx.number_connected_components(seed_subgraph)
            print(f"  Seed genes form {n_comp} connected component(s)")

    return G


# ============================================================================
# MAIN BUILD FUNCTION
# ============================================================================

def build_multiplex_from_markers(
    markers_file,
    expr_matrix_file,
    output_dir,
    p_val_threshold=0.05,
    log2fc_threshold=0.0,
    corr_threshold=DEFAULT_CORR_THRESHOLD,
    corr_method="pearson",
    seed_genes_file=None,
):
    """
    Build multiplex co-expression interactome from cell-type marker genes.

    Parameters
    ----------
    markers_file     : path to cell-type markers CSV
    expr_matrix_file : path to expression matrix (cells × genes)
    output_dir       : output directory
    p_val_threshold  : p-value threshold for markers (currently unused)
    log2fc_threshold : log2FC threshold for markers (currently unused)
    corr_threshold   : Pearson r threshold for co-expression edges
    corr_method      : 'pearson' or 'spearman'
    seed_genes_file  : optional path to seed genes CSV
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    report = {
        'summary': {},
        'per_celltype': {},
        'seed_gene_analysis': {'per_celltype': {}, 'summary': {}}
    }

    # ============================================================================
    # LOAD DATA
    # ============================================================================
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    celltype_markers, markers_df = load_celltype_markers(
        markers_file,
        p_val_threshold=p_val_threshold,
        log2fc_threshold=log2fc_threshold,
    )
    if celltype_markers is None:
        print("ERROR: Failed to load markers")
        return None, None

    expr_df = load_expression_matrix(expr_matrix_file)

    seed_genes = set()
    if seed_genes_file:
        print()
        seed_genes = load_seed_genes(seed_genes_file)

    # ============================================================================
    # VERIFICATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("GENE ID VERIFICATION")
    print("=" * 80)

    all_marker_genes = set()
    for genes in celltype_markers.values():
        all_marker_genes.update(g.upper() for g in genes)

    expr_genes = set(expr_df.columns)

    print(f"\n1. Gene ID Format Check:")
    print(f"   Total unique marker genes: {len(all_marker_genes)}")
    print(f"   Genes in expression matrix: {len(expr_genes)}")

    overlap = all_marker_genes.intersection(expr_genes)
    overlap_pct = len(overlap) / len(all_marker_genes) * 100 if all_marker_genes else 0

    print(f"\n2. Marker → Expression Matrix Overlap:")
    print(f"   Genes in both: {len(overlap)} ({overlap_pct:.1f}%)")
    if overlap_pct < 50:
        print("   ⚠️  WARNING: Low overlap — check gene ID format compatibility.")
    elif overlap_pct < 80:
        print("   ⚠️  NOTICE: Moderate overlap.")
    else:
        print("   ✓ Good overlap.")

    if seed_genes:
        print(f"\n3. Seed Gene Verification:")
        seeds_in_markers = seed_genes.intersection(all_marker_genes)
        seeds_in_expr = seed_genes.intersection(expr_genes)
        seeds_in_both = seed_genes.intersection(overlap)
        print(f"   Total seed genes: {len(seed_genes)}")
        print(f"   Seeds in marker genes: {len(seeds_in_markers)} ({len(seeds_in_markers)/len(seed_genes)*100:.1f}%)")
        print(f"   Seeds in expression matrix: {len(seeds_in_expr)} ({len(seeds_in_expr)/len(seed_genes)*100:.1f}%)")
        print(f"   Seeds in BOTH: {len(seeds_in_both)} ({len(seeds_in_both)/len(seed_genes)*100:.1f}%)")

        report['seed_gene_analysis']['summary']['total_seed_genes'] = len(seed_genes)
        report['seed_gene_analysis']['summary']['seed_genes_in_markers'] = len(seeds_in_markers)
        report['seed_gene_analysis']['summary']['seed_genes_in_expr'] = len(seeds_in_expr)
        report['seed_gene_analysis']['summary']['seed_genes_in_both_before'] = len(seeds_in_both)

    print("=" * 80 + "\n")

    # ============================================================================
    # BUILD MULTIPLEX LAYERS
    # ============================================================================
    print("\n" + "=" * 80)
    print("BUILDING CO-EXPRESSION MULTIPLEX LAYERS")
    print("=" * 80)

    layers = {}
    layer_stats = []

    for cell_type in sorted(celltype_markers.keys()):
        marker_genes = celltype_markers[cell_type]

        layer_graph = build_coexpression_layer(
            cell_type=cell_type,
            marker_genes=marker_genes,
            expr_df=expr_df,
            seed_genes=seed_genes,
            corr_threshold=corr_threshold,
            corr_method=corr_method,
        )

        layers[cell_type] = layer_graph

        n_nodes = layer_graph.number_of_nodes()
        n_edges = layer_graph.number_of_edges()
        density = nx.density(layer_graph) if n_nodes > 0 else 0
        avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0

        marker_set = set(str(g).upper() for g in marker_genes)
        markers_in_expr = marker_set.intersection(expr_genes)

        seeds_in_layer = seed_genes.intersection(set(layer_graph.nodes())) if seed_genes else set()

        seed_genes_connected = False
        n_seed_components = 0
        if len(seeds_in_layer) > 1:
            seed_subgraph = layer_graph.subgraph(seeds_in_layer)
            n_seed_components = nx.number_connected_components(seed_subgraph)
            seed_genes_connected = (n_seed_components == 1)
        elif len(seeds_in_layer) == 1:
            seed_genes_connected = True
            n_seed_components = 1

        stat = {
            'cell_type': cell_type,
            'n_marker_genes': len(marker_set),
            'n_markers_in_expr': len(markers_in_expr),
            'n_nodes_in_layer': n_nodes,
            'n_edges_in_layer': n_edges,
            'density': density,
            'avg_degree': avg_degree,
            'n_seed_genes': len(seeds_in_layer),
            'seed_gene_pct': len(seeds_in_layer) / len(seed_genes) * 100 if seed_genes else 0,
            'seed_genes_connected': seed_genes_connected,
            'n_seed_components': n_seed_components,
        }
        layer_stats.append(stat)

        report['per_celltype'][cell_type] = {
            'marker_genes': len(marker_set),
            'markers_in_expr': len(markers_in_expr),
            'nodes_in_layer': n_nodes,
            'edges_in_layer': n_edges,
            'density': density,
            'avg_degree': avg_degree,
        }

        if seed_genes:
            report['seed_gene_analysis']['per_celltype'][cell_type] = {
                'seed_genes_in_layer': len(seeds_in_layer),
                'seed_gene_pct_of_total': len(seeds_in_layer) / len(seed_genes) * 100 if seed_genes else 0,
                'seed_genes_connected': seed_genes_connected,
                'n_seed_components': n_seed_components,
            }

    # ============================================================================
    # MULTIPLEX-LEVEL STATISTICS
    # ============================================================================
    print("\n" + "=" * 80)
    print("MULTIPLEX-LEVEL STATISTICS")
    print("=" * 80)

    all_layer_nodes = set().union(*[set(g.nodes()) for g in layers.values() if g.number_of_nodes() > 0])
    all_layer_edges = sum(g.number_of_edges() for g in layers.values())

    print(f"Total unique genes in multiplex: {len(all_layer_nodes)}")
    print(f"Total edges in multiplex: {all_layer_edges}")
    print(f"Number of layers: {len(layers)}")

    report['summary']['n_cell_types'] = len(layers)
    report['summary']['marker_genes_total'] = len(all_marker_genes)
    report['summary']['marker_genes_in_expr'] = len(overlap)
    report['summary']['genes_in_multiplex'] = len(all_layer_nodes)
    report['summary']['edges_in_multiplex'] = all_layer_edges
    report['summary']['corr_threshold'] = corr_threshold
    report['summary']['corr_method'] = corr_method

    if seed_genes:
        seeds_in_multiplex = seed_genes.intersection(all_layer_nodes)
        recovery_pct = len(seeds_in_multiplex) / len(seed_genes) * 100 if seed_genes else 0
        report['seed_gene_analysis']['summary']['seed_genes_in_multiplex'] = len(seeds_in_multiplex)
        report['seed_gene_analysis']['summary']['seed_genes_lost'] = len(seed_genes) - len(seeds_in_multiplex)
        report['seed_gene_analysis']['summary']['seed_gene_recovery_pct'] = recovery_pct
        print(f"\nSeed genes in multiplex: {len(seeds_in_multiplex)} / {len(seed_genes)} ({recovery_pct:.1f}%)")

    # ============================================================================
    # SAVE OUTPUTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    layers_dir = os.path.join(output_dir, 'layers')
    os.makedirs(layers_dir, exist_ok=True)

    for cell_type, layer_graph in layers.items():
        safe_cell_type = cell_type.replace('/', '_')
        layer_file = os.path.join(layers_dir, f"{safe_cell_type}_layer.pkl")
        with open(layer_file, 'wb') as f:
            pickle.dump(layer_graph, f, protocol=4)

    print(f"Saved {len(layers)} layer networks to {layers_dir}")

    stats_df = pd.DataFrame(layer_stats)
    stats_file = os.path.join(output_dir, 'layer_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved layer statistics to {stats_file}")

    if markers_df is not None:
        markers_df.to_csv(os.path.join(output_dir, 'markers_with_layers.csv'), index=False)

    # Report
    report_file = os.path.join(output_dir, 'multiplex_conversion_report.txt')
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTIPLEX CO-EXPRESSION INTERACTOME FROM CELL-TYPE MARKERS\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of cell types: {report['summary']['n_cell_types']}\n")
        f.write(f"Correlation method: {corr_method}\n")
        f.write(f"Correlation threshold: {corr_threshold}\n\n")

        f.write("MARKER GENE STATISTICS\n")
        f.write(f"  Total marker genes: {report['summary']['marker_genes_total']:,}\n")
        f.write(f"  Marker genes in expression matrix: {report['summary']['marker_genes_in_expr']:,}\n")
        f.write(f"  Genes in multiplex: {report['summary']['genes_in_multiplex']:,}\n")
        f.write(f"  Total edges in multiplex: {report['summary']['edges_in_multiplex']:,}\n\n")

        f.write("PER CELL TYPE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Cell Type':<40} {'Markers':<12} {'In Expr':<12} {'In Layer':<12} {'Edges':<10}\n")
        f.write("-" * 80 + "\n")
        for cell_type in sorted(report['per_celltype'].keys()):
            ct = report['per_celltype'][cell_type]
            f.write(f"{cell_type:<40} {ct['marker_genes']:<12} {ct['markers_in_expr']:<12} "
                    f"{ct['nodes_in_layer']:<12} {ct['edges_in_layer']:<10}\n")

        if seed_genes:
            sg_summary = report['seed_gene_analysis']['summary']
            f.write("\n" + "=" * 80 + "\n")
            f.write("SEED GENE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total seed genes: {sg_summary['total_seed_genes']}\n")
            f.write(f"Seed genes in multiplex: {sg_summary['seed_genes_in_multiplex']}\n")
            f.write(f"Seed genes lost: {sg_summary['seed_genes_lost']}\n")
            f.write(f"Seed gene recovery: {sg_summary['seed_gene_recovery_pct']:.1f}%\n\n")

            f.write(f"{'Cell Type':<40} {'In Layer':<15} {'% of Total':<15} {'Connected':<12}\n")
            f.write("-" * 80 + "\n")
            for cell_type in sorted(report['seed_gene_analysis']['per_celltype'].keys()):
                sg_ct = report['seed_gene_analysis']['per_celltype'][cell_type]
                f.write(f"{cell_type:<40} {sg_ct['seed_genes_in_layer']:<15} "
                        f"{sg_ct['seed_gene_pct_of_total']:.1f}%{'':<10} "
                        f"{str(sg_ct['seed_genes_connected']):<12}\n")

    print(f"Saved report to {report_file}")

    json_report_file = os.path.join(output_dir, 'multiplex_conversion_report.json')
    with open(json_report_file, 'w') as f:
        json.dump(report, f, indent=2)

    metadata = {
        'n_layers': len(layers),
        'n_total_genes': len(all_layer_nodes),
        'n_total_edges': all_layer_edges,
        'corr_threshold': corr_threshold,
        'corr_method': corr_method,
        'layer_stats': layer_stats,
        'report': report,
    }
    with open(os.path.join(output_dir, 'multiplex_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f, protocol=4)

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("MULTIPLEX CO-EXPRESSION INTERACTOME CONSTRUCTION COMPLETE")
    print("=" * 80)
    print(f"Total layers: {len(layers)}")
    print(f"Total unique genes: {len(all_layer_nodes)}")
    print(f"Total edges: {all_layer_edges}")
    print(f"Correlation method: {corr_method} | threshold: {corr_threshold}")
    if seed_genes:
        print(f"Seed gene recovery: {report['seed_gene_analysis']['summary']['seed_gene_recovery_pct']:.1f}%")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    return layers, metadata


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build multiplex co-expression interactome from cell-type marker genes. "
            "Edges are based on gene co-expression in the supplied expression matrix, NOT PPI."
        )
    )
    parser.add_argument('--markers-file', type=str, required=True,
                        help='Path to cell-type markers CSV file')
    parser.add_argument('--expr-matrix', type=str, required=True,
                        help='Path to expression matrix (cells × genes). '
                             'Supported: .csv, .tsv, .pkl (pd.DataFrame)')
    parser.add_argument('--output-dir', type=str, default='multiplex_coexpr_markers',
                        help='Output directory for multiplex network')
    parser.add_argument('--corr-threshold', type=float, default=DEFAULT_CORR_THRESHOLD,
                        help=f'Pearson r threshold for co-expression edge (default: {DEFAULT_CORR_THRESHOLD})')
    parser.add_argument('--corr-method', default='pearson', choices=['pearson', 'spearman'],
                        help='Correlation method (default: pearson)')
    parser.add_argument('--p-val-threshold', type=float, default=0.05,
                        help='P-value threshold for marker significance (currently unused)')
    parser.add_argument('--log2fc-threshold', type=float, default=0.0,
                        help='Log2 fold change threshold for markers (currently unused)')
    parser.add_argument('--seed-genes-file', type=str, default=None,
                        help='Path to seed genes CSV file (optional)')

    args = parser.parse_args()

    layers, metadata = build_multiplex_from_markers(
        markers_file=args.markers_file,
        expr_matrix_file=args.expr_matrix,
        output_dir=args.output_dir,
        p_val_threshold=args.p_val_threshold,
        log2fc_threshold=args.log2fc_threshold,
        corr_threshold=args.corr_threshold,
        corr_method=args.corr_method,
        seed_genes_file=args.seed_genes_file,
    )

    if layers is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())