#!/usr/bin/env python3
"""
rebuild_reference_graphs.py
============================
After consolidate_layers.py filters the working directory, rebuild:
  1. integrated_multiplex.pkl        (filtered multiplex graph)
  2. gene_reference_graph.pkl        (all edges in surviving layers)
  3. gene_ppi_reference_graph.pkl    (PPI-only edges)

These must reflect ONLY the surviving layers, otherwise dm3 can
find paths through nodes/edges that were removed by consolidation.

Usage
-----
python rebuild_reference_graphs.py \
    --wdir  multixrank_input_consolidated \
    --original-pkl  integrated_multiplex_output/integrated_multiplex.pkl
"""

import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import networkx as nx
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Load surviving layers from config
# ---------------------------------------------------------------------------

def load_surviving_layers(wdir: Path) -> dict[str, pd.DataFrame]:
    """
    Read config.yml and load every layer TSV that still exists.
    Returns stem -> DataFrame(g1, g2)
    """
    config_path = wdir / "config.yml"
    with open(config_path) as f:
        import yaml
        cfg = yaml.safe_load(f)

    layer_paths: list[str] = cfg["multiplex"]["gene_multiplex"]["layers"]
    layers: dict[str, pd.DataFrame] = {}

    print(f"Loading {len(layer_paths)} surviving layer(s) …")
    for rel_path in sorted(layer_paths):
        abs_path = wdir / rel_path
        stem = Path(rel_path).stem
        if not abs_path.exists():
            print(f"  ⚠  Missing: {rel_path} — skipped")
            continue
        df = pd.read_csv(
            abs_path, sep="\t", header=None,
            names=["g1", "g2"], dtype=str,
        )
        df["g1"] = df["g1"].str.strip().str.upper()
        df["g2"] = df["g2"].str.strip().str.upper()
        layers[stem] = df
        print(f"  {stem:<55} {len(df):>8,} edges")

    print(f"\nTotal surviving layers: {len(layers)}\n")
    return layers


# ---------------------------------------------------------------------------
# Build gene universe
# ---------------------------------------------------------------------------

def build_gene_universe(layers: dict[str, pd.DataFrame]) -> set[str]:
    universe: set[str] = set()
    for df in layers.values():
        universe.update(df["g1"])
        universe.update(df["g2"])
    print(f"Gene universe (consolidated): {len(universe):,} unique genes\n")
    return universe


# ---------------------------------------------------------------------------
# Classify layers using layer_info.json
# ---------------------------------------------------------------------------

def classify_layers(wdir: Path, layer_stems: set[str]) -> tuple[set[str], set[str]]:
    """
    Returns (ppi_stems, inter_stems) based on layer_info.json.
    Falls back to name heuristics if layer_info.json is absent.
    """
    import json
    layer_info_path = wdir / "layer_info.json"

    if layer_info_path.exists():
        with open(layer_info_path) as f:
            layer_info = json.load(f)

        ppi_stems   = {s for s in layer_stems
                       if layer_info.get(s, {}).get("edge_type") == "ppi"}
        inter_stems = {s for s in layer_stems
                       if layer_info.get(s, {}).get("type") in
                       ("inter", "special")}
        print(f"Layer classification (from layer_info.json):")
    else:
        # Heuristic fallback
        ppi_stems   = {s for s in layer_stems
                       if "inter" not in s and "ligand" not in s
                       and "pathway" not in s and "kegg" not in s
                       and "reactome" not in s}
        inter_stems = layer_stems - ppi_stems
        print(f"Layer classification (heuristic — no layer_info.json):")

    print(f"  PPI layers   : {len(ppi_stems)}")
    print(f"  Inter layers : {len(inter_stems)}")
    print()
    return ppi_stems, inter_stems


# ---------------------------------------------------------------------------
# Build reference graphs from surviving TSVs
# (does NOT need the original pkl at all)
# ---------------------------------------------------------------------------

def build_reference_graphs(
    layers: dict[str, pd.DataFrame],
    ppi_stems: set[str],
    inter_stems: set[str],
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build two graphs directly from the surviving TSV files:

    gene_reference_graph     — all edges (ppi + inter)
    gene_ppi_reference_graph — PPI edges only

    Edge attributes on gene_reference_graph:
      edge_type  : 'ppi', 'inter', or 'both'
      cell_types : set of cell-type stems where this is a PPI edge
    """
    ref   = nx.Graph()
    ppi   = nx.Graph()

    def ensure(G, a, b):
        if not G.has_edge(a, b):
            G.add_edge(a, b,
                       cell_types=set(),
                       inter_cell_types=set(),
                       edge_type=None)

    for stem, df in layers.items():
        is_ppi   = stem in ppi_stems
        is_inter = stem in inter_stems

        for _, row in df.iterrows():
            a, b = sorted([row["g1"], row["g2"]])
            if a == b:
                continue  # skip self-loops

            # --- full reference graph ---
            ensure(ref, a, b)
            if is_ppi:
                ref[a][b]["cell_types"].add(stem)
                ref[a][b]["edge_type"] = (
                    "both" if ref[a][b]["edge_type"] == "inter" else "ppi"
                )
            elif is_inter:
                ref[a][b]["inter_cell_types"].add(stem)
                ref[a][b]["edge_type"] = (
                    "both" if ref[a][b]["edge_type"] == "ppi" else "inter"
                )

            # --- PPI-only graph ---
            if is_ppi:
                ppi.add_edge(a, b)

    print(f"gene_reference_graph     : "
          f"{ref.number_of_nodes():,} nodes, "
          f"{ref.number_of_edges():,} edges")
    print(f"gene_ppi_reference_graph : "
          f"{ppi.number_of_nodes():,} nodes, "
          f"{ppi.number_of_edges():,} edges")
    print()
    return ref, ppi


# ---------------------------------------------------------------------------
# Optionally filter the original multiplex pkl
# (needed only if downstream code reads the pkl directly)
# ---------------------------------------------------------------------------

def filter_multiplex_pkl(
    original_pkl: Path,
    gene_universe: set[str],
    output_path: Path,
) -> nx.Graph:
    """
    Load the original multiplex pkl and remove any node whose gene symbol
    is NOT in gene_universe (i.e. it belonged to a removed layer).
    """
    print(f"Loading original multiplex pkl: {original_pkl} …")
    with open(original_pkl, "rb") as f:
        G = pickle.load(f)
    print(f"  Original: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    # Nodes to keep: gene symbol in universe
    keep = [
        n for n, d in G.nodes(data=True)
        if str(d.get("gene", "")).strip().upper() in gene_universe
    ]
    G_filtered = G.subgraph(keep).copy()

    print(f"  Filtered: {G_filtered.number_of_nodes():,} nodes, "
          f"{G_filtered.number_of_edges():,} edges")
    print(f"  Removed : "
          f"{G.number_of_nodes() - G_filtered.number_of_nodes():,} nodes\n")

    with open(output_path, "wb") as f:
        pickle.dump(G_filtered, f, protocol=4)
    print(f"Saved filtered multiplex pkl → {output_path}\n")
    return G_filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild reference graphs after consolidate_layers.py"
    )
    parser.add_argument(
        "--wdir", required=True,
        help="Consolidated MultiXrank working directory",
    )
    parser.add_argument(
        "--original-pkl", default=None,
        help=(
            "Path to the original integrated_multiplex.pkl. "
            "If provided, a filtered copy is saved to --wdir. "
            "Optional — only needed if downstream code reads the pkl directly."
        ),
    )
    args = parser.parse_args()

    wdir = Path(args.wdir)

    # 1. Load surviving layers
    layers = load_surviving_layers(wdir)
    gene_universe = build_gene_universe(layers)

    # 2. Classify layers
    ppi_stems, inter_stems = classify_layers(wdir, set(layers.keys()))

    # 3. Rebuild reference graphs
    ref_graph, ppi_graph = build_reference_graphs(layers, ppi_stems, inter_stems)

    ref_path = wdir / "gene_reference_graph.pkl"
    ppi_path = wdir / "gene_ppi_reference_graph.pkl"

    with open(ref_path, "wb") as f:
        pickle.dump(ref_graph, f, protocol=4)
    print(f"Saved gene_reference_graph     → {ref_path}")

    with open(ppi_path, "wb") as f:
        pickle.dump(ppi_graph, f, protocol=4)
    print(f"Saved gene_ppi_reference_graph → {ppi_path}")
    print()

    # 4. Optionally filter the multiplex pkl
    if args.original_pkl:
        filtered_pkl_path = wdir / "integrated_multiplex_filtered.pkl"
        filter_multiplex_pkl(
            Path(args.original_pkl),
            gene_universe,
            filtered_pkl_path,
        )
    else:
        print("--original-pkl not provided — skipping multiplex pkl update.")
        print("  (This is fine as long as dm2/dm3 do not read the pkl directly)\n")

    print("=" * 60)
    print("Reference graphs rebuilt successfully")
    print("=" * 60)
    print(f"  gene_reference_graph     → {ref_path}")
    print(f"  gene_ppi_reference_graph → {ppi_path}")
    if args.original_pkl:
        print(f"  filtered multiplex pkl   → "
              f"{wdir / 'integrated_multiplex_filtered.pkl'}")
    print()
    print("Next: run dm2 then dm3")
    print("=" * 60)


if __name__ == "__main__":
    main()