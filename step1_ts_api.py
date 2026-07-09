#!/usr/bin/env python3
"""
Build Multiplex Interactome from Tabula Sapiens v2 via CELLxGENE Census API
Using a custom pre-built reference PPI (pickled NetworkX graph).

Strategy:
  1. Load your custom PPI from a pickle file
  2. Pull cell-type expression data from Tabula Sapiens via Census API
  3. Identify expressed genes per cell type (mean count > threshold)
  4. Intersect expressed genes with your PPI to build one layer per cell type
  5. Save outputs in the same format as the original file-based script

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
MIN_CELLS_PER_CELLTYPE = 50          # drop cell types with fewer cells than this


# ============================================================================
# STEP 1: LOAD YOUR CUSTOM PPI
# ============================================================================

def load_custom_ppi(ppi_path: str) -> nx.Graph:
    """
    Load a pre-built PPI from a pickle file.

    Expects a NetworkX Graph with gene symbol strings as nodes (uppercase).

    Parameters
    ----------
    ppi_path : path to pickled NetworkX graph

    Returns
    -------
    nx.Graph
    """
    print(f"\nLoading custom PPI from {ppi_path} ...")

    if not os.path.exists(ppi_path):
        raise FileNotFoundError(f"PPI file not found: {ppi_path}")

    with open(ppi_path, "rb") as f:
        G = pickle.load(f)

    if not isinstance(G, nx.Graph):
        raise TypeError(f"Expected a NetworkX Graph, got {type(G)}")

    # Normalise node labels to uppercase strings (in case they aren't already)
    if not nx.is_empty(G):
        sample_node = next(iter(G.nodes()))
        if not isinstance(sample_node, str) or sample_node != sample_node.upper():
            print("  Normalising node labels to uppercase strings ...")
            mapping = {n: str(n).strip().upper() for n in G.nodes()}
            G = nx.relabel_nodes(G, mapping)

    print(f"  Custom PPI loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Basic sanity check
    n_components = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)
    print(f"  Connected components: {n_components:,} | Largest CC: {len(largest_cc):,} nodes")

    return G


# ============================================================================
# STEP 2: FETCH TABULA SAPIENS EXPRESSION VIA CENSUS API
# ============================================================================

def fetch_tabula_sapiens_expressed_genes(
    ppi_genes: set,
    expression_threshold: float = DEFAULT_EXPRESSION_THRESHOLD,
    min_cells: int = MIN_CELLS_PER_CELLTYPE,
    census_version: str = "stable",
) -> dict:
    """
    Query the CELLxGENE Census for Tabula Sapiens v2 cells, stream the
    expression matrix, and return expressed genes per cell type.

    Only genes that are also present in your PPI are tracked, which
    dramatically reduces memory usage during streaming.

    Parameters
    ----------
    ppi_genes : set of gene symbols in your reference PPI (used to pre-filter)
    expression_threshold : mean raw count threshold to call a gene "expressed"
    min_cells : minimum cells required to include a cell type as a layer
    census_version : Census release tag ("stable" or e.g. "2024-07-01")

    Returns
    -------
    dict mapping cell_type (str) → set of expressed gene symbols (uppercase)
    """
    print(f"\nOpening CELLxGENE Census (version='{census_version}') ...")

    with cellxgene_census.open_soma(census_version=census_version) as census:

        # ------------------------------------------------------------------
        # 2a. Identify Tabula Sapiens dataset IDs in this Census release
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
        # 2b. Fetch cell metadata (soma_joinid + cell_type) for TS cells only
        # ------------------------------------------------------------------
        # Census value_filter syntax is Python-like; lists must be plain Python lists
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

        # Drop cell types below the minimum cell count
        ct_counts = obs_df["cell_type"].value_counts()
        valid_cts = ct_counts[ct_counts >= min_cells].index.tolist()
        n_dropped = len(ct_counts) - len(valid_cts)
        print(
            f"  {len(valid_cts)} cell types with ≥ {min_cells} cells "
            f"(dropped {n_dropped} with fewer)"
        )

        obs_df = obs_df[obs_df["cell_type"].isin(valid_cts)].copy()
        cell_to_ct = dict(zip(obs_df["soma_joinid"], obs_df["cell_type"]))

        # ------------------------------------------------------------------
        # 2c. Build a gene joinid → symbol map, restricted to PPI genes
        #     This means we only accumulate counts for genes we can actually use
        # ------------------------------------------------------------------
        print("\n  Fetching gene metadata ...")
        var_df = cellxgene_census.get_var(
            census,
            "homo_sapiens",
            column_names=["soma_joinid", "feature_name"],
        )
        # Only keep genes that exist in your PPI (saves memory during streaming)
        var_df["feature_name_upper"] = var_df["feature_name"].str.upper()
        var_df_ppi = var_df[var_df["feature_name_upper"].isin(ppi_genes)]
        gene_joinid_to_symbol = dict(
            zip(var_df_ppi["soma_joinid"], var_df_ppi["feature_name_upper"])
        )
        ppi_gene_joinids = set(gene_joinid_to_symbol.keys())

        print(
            f"  {len(var_df):,} total Census genes | "
            f"{len(ppi_gene_joinids):,} overlap with your PPI"
        )

        # ------------------------------------------------------------------
        # 2d. Stream expression matrix, accumulate sums per (cell_type, gene)
        # ------------------------------------------------------------------
        print("\n  Streaming expression matrix (this may take several minutes) ...")
        print(f"  Tracking {len(ppi_gene_joinids):,} PPI genes across {len(valid_cts)} cell types\n")

        # ct → gene_joinid → cumulative raw count sum
        ct_gene_sum: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

        query = census["census_data"]["homo_sapiens"].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=obs_filter),
        )

        n_batches = 0
        n_values_kept = 0

        for batch in query.X("raw").tables():
            batch_dict = batch.to_pydict()
            cell_ids = batch_dict["soma_dim_0"]   # cell soma_joinid
            gene_ids = batch_dict["soma_dim_1"]   # gene soma_joinid
            counts   = batch_dict["soma_data"]    # raw count (sparse: only non-zero)

            for cell_id, gene_id, count in zip(cell_ids, gene_ids, counts):
                if gene_id not in ppi_gene_joinids:
                    continue                          # skip genes not in your PPI
                ct = cell_to_ct.get(cell_id)
                if ct is None:
                    continue                          # skip cells outside valid set
                ct_gene_sum[ct][gene_id] += count
                n_values_kept += 1

            n_batches += 1
            if n_batches % 200 == 0:
                print(f"    ... {n_batches} batches processed, {n_values_kept:,} values kept")

        print(f"  Streaming complete: {n_batches} batches | {n_values_kept:,} values kept")

    # ------------------------------------------------------------------
    # 2e. Compute mean expression and apply threshold
    # ------------------------------------------------------------------
    print(f"\n  Applying expression threshold (mean count > {expression_threshold}) ...")
    celltype_expressed_genes = {}

    for ct in sorted(valid_cts):
        n_cells = int(ct_counts[ct])
        gene_sums = ct_gene_sum.get(ct, {})
        expressed = set()
        for gene_joinid, total in gene_sums.items():
            if (total / n_cells) > expression_threshold:
                symbol = gene_joinid_to_symbol[gene_joinid]
                expressed.add(symbol)
        celltype_expressed_genes[ct] = expressed
        print(f"    {ct}: {len(expressed):,} expressed PPI-overlapping genes ({n_cells:,} cells)")

    return celltype_expressed_genes


# ============================================================================
# STEP 3: BUILD MULTIPLEX LAYERS
# ============================================================================

def build_layers(
    celltype_expressed_genes: dict,
    G_ref: nx.Graph,
) -> dict:
    """
    Restrict the reference PPI to expressed genes for each cell type.

    Parameters
    ----------
    celltype_expressed_genes : dict cell_type → set of expressed gene symbols
    G_ref : your custom PPI (gene symbol nodes)

    Returns
    -------
    dict cell_type → nx.Graph (cell-type-specific PPI layer)
    """
    print("\n" + "=" * 70)
    print("BUILDING MULTIPLEX LAYERS")
    print("=" * 70)

    layers = {}
    ref_genes = set(G_ref.nodes())

    for ct in sorted(celltype_expressed_genes.keys()):
        expressed = celltype_expressed_genes[ct]
        node_set = expressed.intersection(ref_genes)
        layer = G_ref.subgraph(node_set).copy()

        n_nodes = layer.number_of_nodes()
        n_edges = layer.number_of_edges()

        if n_nodes == 0:
            print(f"  {ct}: EMPTY after PPI intersection — skipped")
            continue

        layers[ct] = layer
        density = nx.density(layer)
        avg_deg = 2 * n_edges / n_nodes
        print(
            f"  {ct}: {n_nodes:,} nodes | {n_edges:,} edges | "
            f"density={density:.4f} | avg_degree={avg_deg:.2f}"
        )

    print(f"\nTotal layers built: {len(layers)}")
    return layers


# ============================================================================
# STEP 4: SAVE OUTPUTS  (identical format to the original file-based script)
# ============================================================================

def save_outputs(
    layers: dict,
    celltype_expressed_genes: dict,
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

        # Save individual layer pickle
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
            "n_expressed_genes": len(celltype_expressed_genes.get(ct, set())),
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
# SEED GENE LOADER (unchanged from original)
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
            "Build multiplex interactome from Tabula Sapiens v2 (Census API) "
            "using a custom pickled PPI."
        )
    )
    parser.add_argument(
        "--ppi", default="../temp/combined26/full_network.pkl",
        help="Path to your pickled NetworkX PPI (default: ../temp/combined26/full_network.pkl)",
    )
    parser.add_argument(
        "--output-dir", default="multiplex_interactome_census",
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
        "--min-cells", type=int, default=MIN_CELLS_PER_CELLTYPE,
        help=f"Min cells per cell type (default: {MIN_CELLS_PER_CELLTYPE})",
    )
    parser.add_argument(
        "--census-version", default="stable",
        help="Census version tag, e.g. 'stable' or '2024-07-01'",
    )
    args = parser.parse_args()

    start = time.time()

    # Load seed genes (optional)
    seed_genes = set()
    if args.seed_genes:
        seed_genes = load_seed_genes(args.seed_genes)

    # Step 1: Your custom PPI
    G_ref = load_custom_ppi(args.ppi)
    ppi_genes = set(G_ref.nodes())

    # Step 2: Tabula Sapiens expression per cell type
    celltype_expressed = fetch_tabula_sapiens_expressed_genes(
        ppi_genes=ppi_genes,
        expression_threshold=args.expression_threshold,
        min_cells=args.min_cells,
        census_version=args.census_version,
    )

    # Step 3: Build cell-type-specific layers
    layers = build_layers(celltype_expressed, G_ref)

    # Step 4: Save
    save_outputs(layers, celltype_expressed, seed_genes, args.output_dir)

    print(f"\nFinished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    sys.exit(main())