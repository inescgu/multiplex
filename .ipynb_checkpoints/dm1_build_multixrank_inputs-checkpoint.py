#!/usr/bin/env python3
"""
Build MultiXrank Inputs
========================
Decomposes integrated_multiplex.pkl into MultiXrank's expected directory
structure (Option B: layers = edge types).

Outputs
-------
<output-dir>/
├── multiplex/
│   └── gene_multiplex/
│       ├── ppi_layer.tsv
│       ├── pathway_layer.tsv
│       └── lr_layer.tsv
├── seeds/
│   └── seeds.txt
├── config.yml
└── gene_reference_graph.pkl   # for downstream connectivity checks
"""

import os
import argparse
import pickle
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd
import networkx as nx
import yaml


def load_multiplex(pkl_path: str) -> nx.Graph:
    print(f"Loading integrated multiplex from {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")
    return G


def project_to_gene_level(G: nx.Graph) -> tuple[dict, dict]:
    """
    Returns
    -------
    edges_by_type : dict[str, set[tuple[str,str]]]
        edge_type -> set of (geneA, geneB) tuples, gene-symbol level
    gene_universe : set[str]
        all gene symbols appearing in any layer
    """
    edges_by_type = defaultdict(set)
    gene_universe = set()

    for nid, d in G.nodes(data=True):
        gene_universe.add(d["gene"])

    for u, v, d in G.edges(data=True):
        gu = G.nodes[u]["gene"]
        gv = G.nodes[v]["gene"]
        if gu == gv:
            continue  # skip self-loops at gene level
        etype = d.get("edge_type", "unknown")
        a, b = sorted([gu, gv])
        edges_by_type[etype].add((a, b))

    print("Gene-level projection:")
    print(f"  Total unique genes: {len(gene_universe):,}")
    for etype, edges in edges_by_type.items():
        print(f"  {etype:<20} {len(edges):>10,} unique gene-pair edges")
    print()

    return dict(edges_by_type), gene_universe


def write_layer_tsv(edges: set, path: Path, gene_universe: set) -> int:
    """Write edge list as TSV (gene1<TAB>gene2). Returns edge count written."""
    valid = [(a, b) for (a, b) in edges if a in gene_universe and b in gene_universe]
    df = pd.DataFrame(valid, columns=["gene1", "gene2"])
    df.to_csv(path, sep="\t", index=False, header=False)
    return len(valid)


def build_gene_reference_graph(edges_by_type: dict) -> nx.Graph:
    """Union graph across all edge types — for connectivity checks downstream."""
    H = nx.Graph()
    for etype, edges in edges_by_type.items():
        for a, b in edges:
            if H.has_edge(a, b):
                H[a][b]["edge_types"].add(etype)
            else:
                H.add_edge(a, b, edge_types={etype})
    return H

def save_edge_type_metadata(edges_by_type: dict, output_path: Path) -> None:
    """Save a mapping of edge types for reference by downstream scripts."""
    metadata = {
        "edge_types": list(edges_by_type.keys()),
        "edge_counts": {etype: len(edges) for etype, edges in edges_by_type.items()},
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

def write_seeds(seed_csv: str, output_path: Path, gene_universe: set) -> list:
    """Read seed_genes.csv and write filtered seed list."""
    df = pd.read_csv(seed_csv)
    if "gene" not in df.columns:
        raise ValueError(f"Expected 'gene' column in {seed_csv}, found {list(df.columns)}")
    seeds = [str(g).strip().upper() for g in df["gene"].dropna()]
    seeds_in = [s for s in seeds if s in gene_universe]
    seeds_out = [s for s in seeds if s not in gene_universe]

    print(f"Seeds: {len(seeds)} provided, {len(seeds_in)} in multiplex, {len(seeds_out)} missing")
    if seeds_out:
        print(f"  Missing seeds (first 10): {seeds_out[:10]}")

    with open(output_path, "w") as f:
        for g in seeds_in:
            f.write(g + "\n")
    print(f"  Saved {len(seeds_in)} seeds → {output_path}\n")

    return seeds_in


def write_config(
    config_path: Path,
    multiplex_dir: Path,
    seeds_path: Path,
    layer_files: dict,
    restart: float,
    delta: float,
) -> None:
    """
    Write MultiXrank config.yml.
    Layer paths are RELATIVE to the wdir passed to Multixrank(wdir=...).
    """
    n_layers = len(layer_files)
    lamb_value = round(1.0 / n_layers, 6)

    config = {
        "seed": str(seeds_path),
        "self_loops": 0,
        "r": restart,
        "multiplex": {
            "gene_multiplex": {
                "layers": [str(p) for p in layer_files.values()],
                "graph_type": ["00"] * n_layers,  # 00 = undirected, unweighted
                "delta": delta,
                "tau": [lamb_value] * n_layers,
            }
        },
    }

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved MultiXrank config → {config_path}")
    print(f"  restart (r)   : {restart}")
    print(f"  delta         : {delta}")
    print(f"  tau per layer : {lamb_value} ({n_layers} layers)\n")


def main():
    parser = argparse.ArgumentParser(description="Build MultiXrank inputs from integrated multiplex")
    parser.add_argument("--multiplex-pkl", required=True,
                        help="Path to integrated_multiplex.pkl")
    parser.add_argument("--seed-csv", required=True,
                        help="Path to seed_genes.csv (must have 'gene' column)")
    parser.add_argument("--output-dir", default="multixrank_input",
                        help="Output directory")
    parser.add_argument("--restart", type=float, default=0.7,
                        help="Restart probability r (default 0.7)")
    parser.add_argument("--delta", type=float, default=0.5,
                        help="Inter-layer jump probability delta (default 0.5)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    multiplex_dir = out / "multiplex" / "gene_multiplex"
    seeds_dir = out / "seeds"
    multiplex_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + project to gene level
    G = load_multiplex(args.multiplex_pkl)
    edges_by_type, gene_universe = project_to_gene_level(G)

    # 2. Write 3 layer files (always create all three, even if empty)
    layer_files = {}
    expected_layers = ["ppi", "pathway", "ligand_receptor"]
    for etype in expected_layers:
        edges = edges_by_type.get(etype, set())
        # MultiXrank expects paths relative to wdir
        rel_path = Path("multiplex") / "gene_multiplex" / f"{etype}_layer.tsv"
        abs_path = out / rel_path
        n = write_layer_tsv(edges, abs_path, gene_universe)
        print(f"  Wrote {etype:<20} layer: {n:,} edges → {abs_path}")
        layer_files[etype] = rel_path
    print()
    
    # After writing layer files, add:
    metadata_path = out / "edge_type_metadata.json"
    save_edge_type_metadata(edges_by_type, metadata_path)
    print(f"Saved edge type metadata → {metadata_path}\n")

    # 3. Build gene-level reference graph (for downstream connectivity)
    ref_graph = build_gene_reference_graph(edges_by_type)
    ref_path = out / "gene_reference_graph.pkl"
    with open(ref_path, "wb") as f:
        pickle.dump(ref_graph, f, protocol=4)
    print(f"Saved gene reference graph → {ref_path}")
    print(f"  {ref_graph.number_of_nodes():,} nodes, {ref_graph.number_of_edges():,} edges\n")

    # 4. Build PPI-only reference (for --connectivity ppi mode later)
    ppi_only = nx.Graph()
    for a, b in edges_by_type.get("ppi", set()):
        ppi_only.add_edge(a, b)
    ppi_path = out / "gene_ppi_reference_graph.pkl"
    with open(ppi_path, "wb") as f:
        pickle.dump(ppi_only, f, protocol=4)
    print(f"Saved PPI-only reference graph → {ppi_path}")
    print(f"  {ppi_only.number_of_nodes():,} nodes, {ppi_only.number_of_edges():,} edges\n")

    # 5. Write seeds
    seeds_rel_path = Path("seeds") / "seeds.txt"
    seeds_abs_path = out / seeds_rel_path
    seeds_in = write_seeds(args.seed_csv, seeds_abs_path, gene_universe)

    # 6. Write config
    config_path = out / "config.yml"
    write_config(config_path, multiplex_dir, seeds_rel_path, layer_files,
                 args.restart, args.delta)

    print("=" * 60)
    print("MultiXrank inputs ready")
    print("=" * 60)
    print(f"Working directory for MultiXrank: {out.resolve()}")
    print(f"Config file: {config_path}")
    print(f"Run script 2 with: --wdir {out} --config config.yml")
    print("=" * 60)


if __name__ == "__main__":
    main()