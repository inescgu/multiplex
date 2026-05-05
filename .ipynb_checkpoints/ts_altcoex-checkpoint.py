#!/usr/bin/env python3
"""
Build Multiplex Interactome from Tabula Sapiens v2 via CELLxGENE Census API
Using CO-EXPRESSION to define edges (NOT PPI).

Strategy:
  1. Pull cell-type expression data from Tabula Sapiens via Census API
  2. Identify expressed genes per cell type (mean count > threshold)
  3. For each cell type, build a per-cell expression matrix for expressed genes
  4. Compute pairwise Pearson correlations and threshold to form edges
  5. Save outputs

Dependencies:
    pip install cellxgene-census tiledbsoma networkx pandas numpy scipy
"""

import os
import sys
import argparse
import time
import json
import pickle
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import cellxgene_census
import tiledbsoma as soma

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================

TABULA_SAPIENS_COLLECTION = "Tabula Sapiens"
DEFAULT_EXPRESSION_THRESHOLD = 0.1   # mean raw count to call a gene "expressed"
DEFAULT_CORR_THRESHOLD       = 0.3   # Pearson r threshold for a co-expression edge
MIN_CELLS_PER_CELLTYPE       = 50    # drop cell types with fewer cells than this


# ============================================================================
# STEP 1: FETCH TABULA SAPIENS EXPRESSION VIA CENSUS API
#         Returns both expressed-gene sets AND per-cell expression matrices
# ============================================================================

def fetch_tabula_sapiens_expression(
    expression_threshold: float = DEFAULT_EXPRESSION_THRESHOLD,
    min_cells: int = MIN_CELLS_PER_CELLTYPE,
    census_version: str = "stable",
    gene_limit: int = None,          # optional cap on tracked genes (memory guard)
) -> tuple[dict, dict]:
    """
    Query the CELLxGENE Census for Tabula Sapiens v2 cells, stream the
    expression matrix, and return:
      - expressed_genes  : cell_type → set of expressed gene symbols
      - expr_matrices    : cell_type → pd.DataFrame (cells × genes, raw counts)

    Parameters
    ----------
    expression_threshold : mean raw count to call a gene "expressed"
    min_cells            : minimum cells to include a cell type
    census_version       : Census release tag ("stable" or e.g. "2024-07-01")
    gene_limit           : if set, only track the top-N most variable genes
                           (reduces memory; set to e.g. 5000 for large datasets)

    Returns
    -------
    (expressed_genes dict, expr_matrices dict)
    """
    print(f"\nOpening CELLxGENE Census (version='{census_version}') ...")

    with cellxgene_census.open_soma(census_version=census_version) as census:

        # ------------------------------------------------------------------
        # 1a. Identify Tabula Sapiens dataset IDs
        # ------------------------------------------------------------------
        print("  Querying datasets table for Tabula Sapiens ...")
        datasets = (
            census["census_info"]["datasets"]
            .read(column_names=["dataset_id", "collection_name", "dataset_title"])
            .concat()
            .to_pandas()
        )
        ts_datasets = datasets[
            datasets["collection_name"].str.contains(
                TABULA_SAPIENS_COLLECTION, case=False, na=False
            )
        ]
        if ts_datasets.empty:
            raise RuntimeError(
                f"No datasets found matching '{TABULA_SAPIENS_COLLECTION}'. "
                "Check the collection_name available in your Census version."
            )

        ts_ids = ts_datasets["dataset_id"].tolist()
        print(f"  Found {len(ts_ids)} Tabula Sapiens dataset(s):")
        for _, row in ts_datasets.iterrows():
            print(f"    • {row['dataset_title']}  [{row['dataset_id']}]")

        # ------------------------------------------------------------------
        # 1b. Fetch cell metadata
        # ------------------------------------------------------------------
        obs_filter = (
            f"dataset_id in {ts_ids} and "
            f"is_primary_data == True and "
            f"disease == 'normal'"
        )

        print("\n  Fetching cell metadata ...")
        obs_df = cellxgene_census.get_obs(
            census,
            "homo_sapiens",
            value_filter=obs_filter,
            column_names=["soma_joinid", "cell_type"],
        )
        print(f"  {len(obs_df):,} cells retrieved")

        ct_counts = obs_df["cell_type"].value_counts()
        valid_cts = ct_counts[ct_counts >= min_cells].index.tolist()
        n_dropped = len(ct_counts) - len(valid_cts)
        print(
            f"  {len(valid_cts)} cell types with ≥ {min_cells} cells "
            f"(dropped {n_dropped} with fewer)"
        )

        obs_df = obs_df[obs_df["cell_type"].isin(valid_cts)].copy()
        cell_to_ct = dict(zip(obs_df["soma_joinid"], obs_df["cell_type"]))
        valid_cell_ids = set(cell_to_ct.keys())

        # ------------------------------------------------------------------
        # 1c. Build gene joinid → symbol map (all human genes)
        # ------------------------------------------------------------------
        print("\n  Fetching gene metadata ...")
        var_df = cellxgene_census.get_var(
            census,
            "homo_sapiens",
            column_names=["soma_joinid", "feature_name"],
        )
        var_df["feature_name_upper"] = var_df["feature_name"].str.upper()
        gene_joinid_to_symbol = dict(
            zip(var_df["soma_joinid"], var_df["feature_name_upper"])
        )
        all_gene_joinids = set(gene_joinid_to_symbol.keys())

        if gene_limit:
            print(f"  gene_limit={gene_limit}: will restrict to top-{gene_limit} "
                  "genes by total expression after first pass.")

        print(f"  {len(all_gene_joinids):,} total Census genes will be tracked")

        # ------------------------------------------------------------------
        # 1d. Stream expression matrix
        #     Accumulate: (a) per-cell raw counts (for correlation matrix)
        #                 (b) gene sums (for mean-expression filter)
        # ------------------------------------------------------------------
        print("\n  Streaming expression matrix (this may take several minutes) ...")

        # ct → soma_joinid(cell) → gene_joinid → count
        # We store as ct → gene_joinid → {cell_id: count}  (sparse dict-of-dicts)
        ct_cell_gene: dict[str, dict[int, dict[int, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        ct_gene_sum: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

        query = census["census_data"]["homo_sapiens"].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=obs_filter),
        )

        n_batches = 0
        n_values_kept = 0

        for batch in query.X("raw").tables():
            batch_dict = batch.to_pydict()
            cell_ids = batch_dict["soma_dim_0"]
            gene_ids = batch_dict["soma_dim_1"]
            counts   = batch_dict["soma_data"]

            for cell_id, gene_id, count in zip(cell_ids, gene_ids, counts):
                if cell_id not in valid_cell_ids:
                    continue
                if gene_id not in all_gene_joinids:
                    continue
                ct = cell_to_ct[cell_id]
                ct_gene_sum[ct][gene_id] += count
                ct_cell_gene[ct][cell_id][gene_id] = count
                n_values_kept += 1

            n_batches += 1
            if n_batches % 200 == 0:
                print(f"    ... {n_batches} batches | {n_values_kept:,} values kept")

        print(f"  Streaming complete: {n_batches} batches | {n_values_kept:,} values kept")

    # ------------------------------------------------------------------
    # 1e. Compute mean expression and identify expressed genes per cell type
    # ------------------------------------------------------------------
    print(f"\n  Applying expression threshold (mean count > {expression_threshold}) ...")
    expressed_genes: dict[str, set] = {}

    for ct in sorted(valid_cts):
        n_cells = int(ct_counts[ct])
        gene_sums = ct_gene_sum.get(ct, {})
        expressed = set()
        for gene_joinid, total in gene_sums.items():
            if (total / n_cells) > expression_threshold:
                expressed.add(gene_joinid)   # store joinid for matrix building
        expressed_genes[ct] = expressed
        print(f"    {ct}: {len(expressed):,} expressed genes ({n_cells:,} cells)")

    # ------------------------------------------------------------------
    # 1f. Build dense expression DataFrames per cell type
    #     Rows = cells, Columns = expressed gene symbols
    # ------------------------------------------------------------------
    print("\n  Building per-cell-type expression matrices ...")
    expr_matrices: dict[str, pd.DataFrame] = {}

    for ct in sorted(valid_cts):
        expressed_joinids = expressed_genes[ct]

        if len(expressed_joinids) < 2:
            print(f"    {ct}: too few expressed genes — skipped")
            continue

        # Optional: if gene_limit is set, keep only top-N by total expression
        if gene_limit and len(expressed_joinids) > gene_limit:
            top_genes = sorted(
                expressed_joinids,
                key=lambda g: ct_gene_sum[ct].get(g, 0),
                reverse=True,
            )[:gene_limit]
            expressed_joinids = set(top_genes)

        # cell_id → gene_joinid → count  (only expressed genes)
        cell_gene_data = ct_cell_gene.get(ct, {})

        # Collect all cell ids that have any data for this cell type
        cell_ids_for_ct = list(cell_gene_data.keys())
        gene_joinids_list = sorted(expressed_joinids)
        gene_symbols = [gene_joinid_to_symbol[g] for g in gene_joinids_list]

        # Build dense matrix (cells × genes), zero-filling sparse entries
        matrix = np.zeros((len(cell_ids_for_ct), len(gene_joinids_list)), dtype=np.float32)
        gene_idx = {g: i for i, g in enumerate(gene_joinids_list)}

        for row_i, cell_id in enumerate(cell_ids_for_ct):
            for gene_id, count in cell_gene_data[cell_id].items():
                if gene_id in gene_idx:
                    matrix[row_i, gene_idx[gene_id]] = count

        df = pd.DataFrame(matrix, columns=gene_symbols)
        expr_matrices[ct] = df

        print(f"    {ct}: matrix {df.shape[0]} cells × {df.shape[1]} genes")

    # Convert expressed_genes from joinid sets → symbol sets for downstream use
    expressed_genes_symbols: dict[str, set] = {
        ct: {gene_joinid_to_symbol[g] for g in joinids}
        for ct, joinids in expressed_genes.items()
        if ct in expr_matrices
    }

    return expressed_genes_symbols, expr_matrices


# ============================================================================
# STEP 2: BUILD CO-EXPRESSION LAYERS
# ============================================================================

def build_coexpression_layer(
    cell_type: str,
    expr_df: pd.DataFrame,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    corr_method: str = "pearson",
) -> nx.Graph:
    """
    Build a co-expression graph for one cell type.

    Parameters
    ----------
    cell_type      : label for logging
    expr_df        : pd.DataFrame (cells × genes), raw or normalised counts
    corr_threshold : minimum |r| to add an edge
    corr_method    : 'pearson' or 'spearman'

    Returns
    -------
    nx.Graph with gene symbols as nodes and correlation weight on edges
    """
    genes = list(expr_df.columns)

    if len(genes) < 2:
        print(f"  {cell_type}: fewer than 2 genes — empty layer returned")
        return nx.Graph()

    # Compute full correlation matrix
    corr = expr_df.corr(method=corr_method)

    G = nx.Graph()
    G.add_nodes_from(genes)

    n_edges = 0
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            r = corr.iloc[i, j]
            if pd.notna(r) and r >= corr_threshold:
                G.add_edge(genes[i], genes[j], weight=float(r))
                n_edges += 1

    density = nx.density(G)
    avg_deg = 2 * n_edges / len(genes) if genes else 0
    print(
        f"  {cell_type}: {len(genes):,} nodes | {n_edges:,} edges | "
        f"density={density:.4f} | avg_degree={avg_deg:.2f}"
    )
    return G


def build_layers(
    expr_matrices: dict,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    corr_method: str = "pearson",
) -> dict:
    """
    Build one co-expression layer per cell type.

    Parameters
    ----------
    expr_matrices  : cell_type → pd.DataFrame (cells × genes)
    corr_threshold : Pearson r threshold for edges
    corr_method    : 'pearson' or 'spearman'

    Returns
    -------
    dict cell_type → nx.Graph
    """
    print("\n" + "=" * 70)
    print("BUILDING CO-EXPRESSION LAYERS")
    print("=" * 70)

    layers = {}
    for ct in sorted(expr_matrices.keys()):
        layer = build_coexpression_layer(
            ct, expr_matrices[ct], corr_threshold, corr_method
        )
        if layer.number_of_nodes() > 0:
            layers[ct] = layer
        else:
            print(f"  {ct}: EMPTY — skipped")

    print(f"\nTotal layers built: {len(layers)}")
    return layers


# ============================================================================
# STEP 3: SAVE OUTPUTS
# ============================================================================

def save_outputs(
    layers: dict,
    expressed_genes: dict,
    seed_genes: set,
    output_dir: str,
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)
    layers_dir = os.path.join(output_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    layer_stats = []
    all_genes: set = set()
    all_edges = 0

    for ct, layer in layers.items():
        with open(os.path.join(layers_dir, f"{ct}_layer.pkl"), "wb") as f:
            pickle.dump(layer, f, protocol=4)

        n_nodes = layer.number_of_nodes()
        n_edges = layer.number_of_edges()
        all_genes.update(layer.nodes())
        all_edges += n_edges

        seeds_in = seed_genes.intersection(layer.nodes()) if seed_genes else set()
        seed_components = 0
        seed_connected = False
        if len(seeds_in) > 1:
            sg = layer.subgraph(seeds_in)
            seed_components = nx.number_connected_components(sg)
            seed_connected = seed_components == 1
        elif len(seeds_in) == 1:
            seed_components = 1
            seed_connected = True

        layer_stats.append({
            "cell_type": ct,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": nx.density(layer),
            "avg_degree": 2 * n_edges / n_nodes if n_nodes else 0,
            "n_expressed_genes": len(expressed_genes.get(ct, set())),
            "n_seed_genes": len(seeds_in),
            "seed_gene_pct": len(seeds_in) / len(seed_genes) * 100 if seed_genes else 0,
            "seed_genes_connected": seed_connected,
            "n_seed_components": seed_components,
        })

    stats_df = pd.DataFrame(layer_stats)
    stats_df.to_csv(os.path.join(output_dir, "layer_statistics.csv"), index=False)

    metadata = {
        "n_cell_types": len(layers),
        "n_total_genes": len(all_genes),
        "n_total_edges": all_edges,
        "n_seed_genes": len(seed_genes),
        "layer_stats": layer_stats,
    }
    with open(os.path.join(output_dir, "multiplex_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f, protocol=4)

    json_summary = {k: v for k, v in metadata.items() if k != "layer_stats"}
    with open(os.path.join(output_dir, "multiplex_summary.json"), "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"\nSaved {len(layers)} layers  →  {layers_dir}")
    print(f"Statistics CSV  →  {output_dir}/layer_statistics.csv")
    print(f"Metadata pickle →  {output_dir}/multiplex_metadata.pkl")
    print(f"JSON summary    →  {output_dir}/multiplex_summary.json")
    print(f"\nMultiplex totals: {len(all_genes):,} unique genes | {all_edges:,} edges")

    return stats_df


# ============================================================================
# SEED GENE LOADER
# ============================================================================

def load_seed_genes(seed_genes_file: str) -> set:
    if not os.path.exists(seed_genes_file):
        print(f"Seed genes file not found: {seed_genes_file}")
        return set()
    try:
        df = pd.read_csv(seed_genes_file)
        col = next(
            (c for c in df.columns if c.lower() in ["gene", "symbol", "gene_id", "hgnc_symbol"]),
            df.columns[0],
        )
        genes = set(df[col].dropna().astype(str).str.strip().str.upper())
        print(f"Loaded {len(genes):,} seed genes from {seed_genes_file}")
        return genes
    except Exception as e:
        print(f"Error loading seed genes: {e}")
        return set()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build multiplex co-expression interactome from Tabula Sapiens v2 "
            "(Census API). Edges are based on gene co-expression, NOT PPI."
        )
    )
    parser.add_argument(
        "--output-dir", default="multiplex_coexpr_census",
        help="Output directory",
    )
    parser.add_argument(
        "--seed-genes", default=None,
        help="Optional CSV with a seed gene column",
    )
    parser.add_argument(
        "--expression-threshold", type=float, default=DEFAULT_EXPRESSION_THRESHOLD,
        help=f"Mean raw count threshold for 'expressed' (default: {DEFAULT_EXPRESSION_THRESHOLD})",
    )
    parser.add_argument(
        "--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD,
        help=f"Pearson r threshold for a co-expression edge (default: {DEFAULT_CORR_THRESHOLD})",
    )
    parser.add_argument(
        "--corr-method", default="pearson", choices=["pearson", "spearman"],
        help="Correlation method for co-expression edges (default: pearson)",
    )
    parser.add_argument(
        "--min-cells", type=int, default=MIN_CELLS_PER_CELLTYPE,
        help=f"Min cells per cell type (default: {MIN_CELLS_PER_CELLTYPE})",
    )
    parser.add_argument(
        "--census-version", default="stable",
        help="Census version tag, e.g. 'stable' or '2024-07-01'",
    )
    parser.add_argument(
        "--gene-limit", type=int, default=None,
        help="Optionally cap genes per cell type to top-N by total expression "
             "(reduces memory; e.g. --gene-limit 5000)",
    )
    args = parser.parse_args()

    start = time.time()

    seed_genes = set()
    if args.seed_genes:
        seed_genes = load_seed_genes(args.seed_genes)

    # Step 1: Fetch expression data from Census
    expressed_genes, expr_matrices = fetch_tabula_sapiens_expression(
        expression_threshold=args.expression_threshold,
        min_cells=args.min_cells,
        census_version=args.census_version,
        gene_limit=args.gene_limit,
    )

    # Step 2: Build co-expression layers
    layers = build_layers(
        expr_matrices,
        corr_threshold=args.corr_threshold,
        corr_method=args.corr_method,
    )

    # Step 3: Save
    save_outputs(layers, expressed_genes, seed_genes, args.output_dir)

    print(f"\nFinished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    sys.exit(main())