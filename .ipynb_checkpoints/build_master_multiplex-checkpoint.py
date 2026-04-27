#!/usr/bin/env python3
"""
Build Integrated Multiplex Interactome
=======================================
Combines Tabula Sapiens (Census API) and Human Dorsal Root Ganglia (HRDG)
multiplex layers into a single unified multiplex graph where:

  • Intra-layer edges  : PPI interactions within each cell-type layer (unchanged)
  • Inter-layer edges  : Same gene present in two different layers (unchanged)
  • Pathway edges      : NEW — nodes in *different* layers linked if they share
                         a Reactome signalling pathway

Node keys: (source, cell_type, gene)
  source ∈ {"tabula_sapiens", "hrdg"}

Input:
  reactome_interactions_with_pathways.csv — pre-built file with columns:
    uniprot1, uniprot2, pathway_protein1, pathway_protein2, common_pathway, ...

Dependencies:
    pip install networkx pandas numpy requests pickle5
"""

import os
import json
import pickle
import time
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================================
# SECTION 1 — LAYER LOADING
# ============================================================================

def load_layers_from_folder(layers_dir: str, source_tag: str) -> dict:
    """
    Load all *_layer.pkl files from a folder and tag them with a source label.

    Returns
    -------
    dict  (source_tag, cell_type) -> nx.Graph   (nodes are gene symbols / UniProt IDs)
    """
    layers_dir = Path(layers_dir)
    pkl_files  = sorted(layers_dir.glob("*_layer.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No *_layer.pkl files found in {layers_dir}")

    print(f"[{source_tag}] Found {len(pkl_files)} layer file(s)")
    layers = {}

    for fpath in pkl_files:
        cell_type = fpath.stem.replace("_layer", "")
        with open(fpath, "rb") as f:
            g = pickle.load(f)
        # Normalise node labels to uppercase strings
        mapping = {n: str(n).strip().upper() for n in g.nodes()}
        g = nx.relabel_nodes(g, mapping)
        layers[(source_tag, cell_type)] = g
        print(f"  {source_tag} | {cell_type}: "
              f"{g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    print(f"[{source_tag}] Loaded {len(layers)} layers\n")
    return layers


# ============================================================================
# SECTION 2 — PATHWAY INDEX FROM PRE-BUILT CSV
# ============================================================================

def build_pathway_index_from_csv(csv_path: str) -> tuple[dict, dict]:
    """
    Build gene ↔ pathway indices directly from reactome_interactions_with_pathways.csv.

    The CSV must contain at minimum:
      - pathway_protein1  : gene symbol for interactor 1
      - pathway_protein2  : gene symbol for interactor 2
      - common_pathway    : shared Reactome pathway name (NaN if none)

    Returns
    -------
    gene_to_pathways : dict  gene_symbol (upper) -> set of pathway names
    pathway_to_genes : dict  pathway_name        -> set of gene_symbol (upper)
    """
    print(f"Loading Reactome interaction/pathway data from {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  {len(df):,} rows loaded, columns: {df.columns.tolist()}\n")

    # Only rows with a shared pathway are relevant for pathway edges
    pathway_df = df.dropna(subset=["common_pathway"]).copy()
    print(f"  {len(pathway_df):,} rows have a common_pathway annotation")

    gene_to_pathways: dict[str, set] = defaultdict(set)
    pathway_to_genes: dict[str, set] = defaultdict(set)

    for _, row in pathway_df.iterrows():
        sym1    = str(row["pathway_protein1"]).strip().upper()
        sym2    = str(row["pathway_protein2"]).strip().upper()
        pathway = str(row["common_pathway"]).strip()

        if not pathway or pathway.lower() == "nan":
            continue

        for sym in (sym1, sym2):
            if sym and sym != "NAN":
                gene_to_pathways[sym].add(pathway)
                pathway_to_genes[pathway].add(sym)

    print(f"  {len(gene_to_pathways):,} genes with pathway annotations")
    print(f"  {len(pathway_to_genes):,} distinct pathways\n")
    return dict(gene_to_pathways), dict(pathway_to_genes)


# ============================================================================
# SECTION 3 — MULTIPLEX ASSEMBLY
# ============================================================================

def build_integrated_multiplex(
    ts_layers:     dict,
    hrdg_layers:   dict,
    gene_to_pathways: dict,
    pathway_to_genes: dict,
    add_pathway_intralayer: bool = False,
) -> nx.Graph:
    """
    Assemble the full integrated multiplex graph.

    Node key : (source, cell_type, gene)   — all strings, gene is uppercase
    Edge types (stored in edge attribute 'edge_type'):
      'ppi'           — original intra-layer PPI edge
      'interlayer'    — same gene, different layer (any source combination)
      'pathway'       — different gene OR different layer, shared Reactome pathway
    """
    all_layers = {**ts_layers, **hrdg_layers}

    G = nx.Graph()
    G.graph["description"] = "Integrated TS + HRDG multiplex with pathway inter-layer edges"

    # -----------------------------------------------------------------------
    # 3a. Add all intra-layer PPI nodes + edges
    # -----------------------------------------------------------------------
    print("Adding intra-layer PPI nodes and edges ...")
    n_ppi_edges = 0

    for (source, cell_type), layer in all_layers.items():
        for gene in layer.nodes():
            node_key = (source, cell_type, gene)
            attrs    = dict(layer.nodes[gene])
            attrs.update(source=source, cell_type=cell_type, gene=gene)
            G.add_node(node_key, **attrs)

        for u, v, edata in layer.edges(data=True):
            src_u = (source, cell_type, u)
            src_v = (source, cell_type, v)
            G.add_edge(src_u, src_v, edge_type="ppi",
                       source=source, cell_type=cell_type, **edata)
            n_ppi_edges += 1

    print(f"  Nodes so far : {G.number_of_nodes():,}")
    print(f"  PPI edges    : {n_ppi_edges:,}\n")

    # -----------------------------------------------------------------------
    # 3b. Inter-layer edges — same gene, different (source, cell_type)
    # -----------------------------------------------------------------------
    print("Adding inter-layer edges (same gene across layers) ...")

    gene_to_layers: dict[str, list] = defaultdict(list)
    for (source, cell_type), layer in all_layers.items():
        for gene in layer.nodes():
            gene_to_layers[gene].append((source, cell_type))

    n_interlayer = 0
    for gene, layer_keys in gene_to_layers.items():
        if len(layer_keys) < 2:
            continue
        for i in range(len(layer_keys)):
            for j in range(i + 1, len(layer_keys)):
                src_i, ct_i = layer_keys[i]
                src_j, ct_j = layer_keys[j]
                node_i = (src_i, ct_i, gene)
                node_j = (src_j, ct_j, gene)
                if G.has_node(node_i) and G.has_node(node_j):
                    G.add_edge(node_i, node_j,
                               edge_type="interlayer",
                               gene=gene,
                               source_i=src_i, source_j=src_j,
                               cell_type_i=ct_i, cell_type_j=ct_j)
                    n_interlayer += 1

    print(f"  Inter-layer edges added : {n_interlayer:,}\n")

    # -----------------------------------------------------------------------
    # 3c. Pathway edges — connect nodes in different layers sharing a pathway
    # -----------------------------------------------------------------------
    print("Adding pathway-based inter-layer edges ...")
    print(f"  (add_pathway_intralayer = {add_pathway_intralayer})")

    n_pathway_edges = 0
    pathway_edge_set: set = set()

    for pathway, genes_in_path in pathway_to_genes.items():
        # Collect all graph nodes whose gene is in this pathway
        relevant_nodes = [
            n for n in G.nodes()
            if n[2] in genes_in_path        # n = (source, cell_type, gene)
        ]

        if len(relevant_nodes) < 2:
            continue

        for i in range(len(relevant_nodes)):
            for j in range(i + 1, len(relevant_nodes)):
                ni = relevant_nodes[i]   # (source_i, ct_i, gene_i)
                nj = relevant_nodes[j]   # (source_j, ct_j, gene_j)

                same_layer = (ni[0] == nj[0]) and (ni[1] == nj[1])
                if same_layer and not add_pathway_intralayer:
                    continue

                if G.has_edge(ni, nj):
                    existing_type = G[ni][nj].get("edge_type", "")
                    if existing_type in ("ppi", "interlayer"):
                        if "pathways" not in G[ni][nj]:
                            G[ni][nj]["pathways"] = set()
                        G[ni][nj]["pathways"].add(pathway)
                        continue

                edge_key = (min(ni, nj), max(ni, nj))
                if edge_key in pathway_edge_set:
                    if "pathways" not in G[ni][nj]:
                        G[ni][nj]["pathways"] = set()
                    G[ni][nj]["pathways"].add(pathway)
                    continue

                pathway_edge_set.add(edge_key)
                G.add_edge(ni, nj,
                           edge_type="pathway",
                           pathway=pathway,
                           pathways={pathway},
                           source_i=ni[0], source_j=nj[0],
                           cell_type_i=ni[1], cell_type_j=nj[1],
                           gene_i=ni[2], gene_j=nj[2])
                n_pathway_edges += 1

    print(f"  Pathway edges added : {n_pathway_edges:,}\n")

    # -----------------------------------------------------------------------
    # 3d. Summary
    # -----------------------------------------------------------------------
    edge_types = defaultdict(int)
    for _, _, d in G.edges(data=True):
        edge_types[d.get("edge_type", "unknown")] += 1

    print("=" * 60)
    print("INTEGRATED MULTIPLEX SUMMARY")
    print("=" * 60)
    print(f"  Total nodes         : {G.number_of_nodes():,}")
    print(f"  Total edges         : {G.number_of_edges():,}")
    for etype, count in sorted(edge_types.items()):
        print(f"    {etype:<18}: {count:,}")
    print("=" * 60 + "\n")

    return G


# ============================================================================
# SECTION 4 — SAVE & STATISTICS
# ============================================================================

def save_multiplex(G: nx.Graph, output_dir: str) -> None:
    """Save the integrated multiplex and a human-readable summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    graph_path = out / "integrated_multiplex.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f, protocol=4)
    print(f"Saved multiplex graph → {graph_path}")

    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "source_node"   : str(u),
            "target_node"   : str(v),
            "edge_type"     : d.get("edge_type", ""),
            "source_layer_u": u[0],
            "cell_type_u"   : u[1],
            "gene_u"        : u[2],
            "source_layer_v": v[0],
            "cell_type_v"   : v[1],
            "gene_v"        : v[2],
            "pathway"       : str(d.get("pathway", "")),
        })

    edges_df = pd.DataFrame(rows)
    edges_path = out / "integrated_multiplex_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    print(f"Saved edge list      → {edges_path}")

    edge_type_counts = edges_df["edge_type"].value_counts().to_dict()
    summary = {
        "n_nodes"          : G.number_of_nodes(),
        "n_edges"          : G.number_of_edges(),
        "edge_type_counts" : edge_type_counts,
        "n_layers"         : len(set(
            (d["source"], d["cell_type"])
            for _, d in G.nodes(data=True)
            if "source" in d and "cell_type" in d
        )),
    }
    summary_path = out / "integrated_multiplex_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON   → {summary_path}\n")


def layer_statistics(G: nx.Graph) -> pd.DataFrame:
    """Return a DataFrame with per-layer node / edge counts by edge type."""
    rows = []

    layer_nodes: dict[tuple, set] = defaultdict(set)
    for node, d in G.nodes(data=True):
        key = (d.get("source", node[0]), d.get("cell_type", node[1]))
        layer_nodes[key].add(node)

    for (source, ct), node_set in sorted(layer_nodes.items()):
        sub = G.subgraph(node_set)
        n_ppi = sum(1 for _, _, d in sub.edges(data=True) if d.get("edge_type") == "ppi")

        n_inter = sum(
            1 for u, v, d in G.edges(node_set, data=True)
            if d.get("edge_type") == "interlayer"
        )
        n_pathway = sum(
            1 for u, v, d in G.edges(node_set, data=True)
            if d.get("edge_type") == "pathway"
        )

        rows.append({
            "source"        : source,
            "cell_type"     : ct,
            "n_nodes"       : len(node_set),
            "n_ppi_edges"   : n_ppi,
            "n_interlayer"  : n_inter,
            "n_pathway"     : n_pathway,
        })

    return pd.DataFrame(rows)


# ============================================================================
# SECTION 5 — MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build integrated TS + HRDG multiplex with pathway inter-layer edges"
    )
    parser.add_argument(
        "--ts-layers-dir",   required=True,
        help="Folder containing Tabula Sapiens *_layer.pkl files"
    )
    parser.add_argument(
        "--hrdg-layers-dir", required=True,
        help="Folder containing HRDG *_layer.pkl files"
    )
    parser.add_argument(
        "--reactome-csv",    required=True,
        help="Path to reactome_interactions_with_pathways.csv"
    )
    parser.add_argument(
        "--output-dir",      default="integrated_multiplex_output",
        help="Output directory (default: integrated_multiplex_output)"
    )
    parser.add_argument(
        "--add-pathway-intralayer", action="store_true",
        help="Also add pathway edges within the same layer (off by default)"
    )
    args = parser.parse_args()

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load layers
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: LOADING LAYERS")
    print("=" * 60)
    ts_layers   = load_layers_from_folder(args.ts_layers_dir,   "tabula_sapiens")
    hrdg_layers = load_layers_from_folder(args.hrdg_layers_dir, "hrdg")

    # ------------------------------------------------------------------
    # 2. Build pathway index from pre-built CSV
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: PATHWAY INDEX FROM CSV")
    print("=" * 60)
    gene_to_pathways, pathway_to_genes = build_pathway_index_from_csv(args.reactome_csv)

    # ------------------------------------------------------------------
    # 3. Build integrated multiplex
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: ASSEMBLING MULTIPLEX")
    print("=" * 60)
    G = build_integrated_multiplex(
        ts_layers, hrdg_layers,
        gene_to_pathways, pathway_to_genes,
        add_pathway_intralayer=args.add_pathway_intralayer,
    )

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: SAVING OUTPUTS")
    print("=" * 60)
    save_multiplex(G, args.output_dir)

    stats_df = layer_statistics(G)
    stats_path = os.path.join(args.output_dir, "layer_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved layer stats    → {stats_path}")
    print("\nPer-layer overview:")
    print(stats_df.to_string(index=False))

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()