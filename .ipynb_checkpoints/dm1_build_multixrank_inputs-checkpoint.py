#!/usr/bin/env python3
"""
Build MultiXrank Inputs (Cell-Type Layers)
==========================================
Decomposes a cell-type-specific multiplex into MultiXrank's expected directory
structure. Each cell type becomes one layer in the gene_multiplex.

Node attributes required in the input graph
--------------------------------------------
  gene      : gene symbol (str)
  cell_type : cell type label (str)   ← replaces edge_type logic

Edge attributes used (optional)
--------------------------------
  edge_type : kept for reference/metadata, NOT used for layer splitting

Outputs
-------
<output-dir>/
├── multiplex/
│   └── gene_multiplex/
│       ├── <cell_type_1>_layer.tsv
│       ├── <cell_type_2>_layer.tsv
│       └── ...
├── seeds/
│   └── seeds.txt
├── config.yml
├── cell_type_metadata.json
├── gene_reference_graph.pkl   # union across all cell-type layers
└── gene_ppi_reference_graph.pkl  # PPI-only edges (if edge_type == 'ppi')
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_multiplex(pkl_path: str) -> nx.Graph:
    print(f"Loading multiplex from {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")
    return G


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_by_cell_type(G: nx.Graph) -> tuple[dict, set]:
    """
    Split the multiplex into per-cell-type gene-level edge sets.

    Strategy
    --------
    An edge (u, v) is assigned to a cell-type layer when BOTH endpoints share
    the same cell_type attribute.  Cross-cell-type edges are collected under
    the special key "__intercellular__" so they are not silently dropped.

    Returns
    -------
    edges_by_celltype : dict[str, set[tuple[str, str]]]
        cell_type (or "__intercellular__") -> set of sorted (geneA, geneB) pairs
    gene_universe : set[str]
        all gene symbols across all nodes
    """
    edges_by_celltype: dict[str, set] = defaultdict(set)
    gene_universe: set[str] = set()

    for nid, d in G.nodes(data=True):
        gene = d.get("gene")
        if gene:
            gene_universe.add(str(gene).strip().upper())

    missing_attr: list = []
    for u, v, d in G.edges(data=True):
        du = G.nodes[u]
        dv = G.nodes[v]

        gu = str(du.get("gene", "")).strip().upper()
        gv = str(dv.get("gene", "")).strip().upper()
        if not gu or not gv or gu == gv:
            continue  # skip self-loops or missing gene symbols

        ct_u = du.get("cell_type")
        ct_v = dv.get("cell_type")

        if ct_u is None or ct_v is None:
            missing_attr.append((u, v))
            continue

        a, b = sorted([gu, gv])
        if ct_u == ct_v:
            edges_by_celltype[str(ct_u)].add((a, b))
        else:
            edges_by_celltype["__intercellular__"].add((a, b))

    if missing_attr:
        print(f"  ⚠ {len(missing_attr):,} edges skipped (missing cell_type attribute)")

    print("Cell-type layer projection:")
    print(f"  Total unique genes : {len(gene_universe):,}")
    for ct, edges in sorted(edges_by_celltype.items()):
        label = ct if ct != "__intercellular__" else "(intercellular — not written as a layer)"
        print(f"  {label:<40} {len(edges):>10,} unique gene-pair edges")
    print()

    return dict(edges_by_celltype), gene_universe


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_layer_tsv(edges: set, path: Path, gene_universe: set) -> int:
    """Write edge list as two-column TSV (no header). Returns edge count."""
    valid = [(a, b) for (a, b) in edges if a in gene_universe and b in gene_universe]
    if not valid:
        # MultiXrank requires non-empty layer files; write a placeholder self-loop
        # on the first gene so the run doesn't crash, then warn.
        print(f"    ⚠ Layer {path.name} is EMPTY — writing placeholder self-loop.")
        placeholder = next(iter(gene_universe)) if gene_universe else "PLACEHOLDER"
        valid = [(placeholder, placeholder)]
    df = pd.DataFrame(valid, columns=["gene1", "gene2"])
    df.to_csv(path, sep="\t", index=False, header=False)
    return len(valid)


def save_cell_type_metadata(edges_by_celltype: dict, output_path: Path) -> None:
    """Persist layer names and sizes for downstream scripts."""
    # Exclude the intercellular pseudo-layer from MultiXrank metadata
    layers = {ct: len(e) for ct, e in edges_by_celltype.items()
              if ct != "__intercellular__"}
    metadata = {
        "cell_types": sorted(layers.keys()),
        "edge_counts": layers,
        "intercellular_edges": len(edges_by_celltype.get("__intercellular__", set())),
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved cell-type metadata → {output_path}\n")


def build_gene_reference_graph(edges_by_celltype: dict) -> nx.Graph:
    """Union graph across all cell-type layers (excluding intercellular)."""
    H = nx.Graph()
    for ct, edges in edges_by_celltype.items():
        if ct == "__intercellular__":
            continue
        for a, b in edges:
            if H.has_edge(a, b):
                H[a][b]["cell_types"].add(ct)
            else:
                H.add_edge(a, b, cell_types={ct})
    return H


def write_seeds(seed_csv: str, output_path: Path, gene_universe: set) -> list:
    df = pd.read_csv(seed_csv)
    if "gene" not in df.columns:
        raise ValueError(f"Expected 'gene' column in {seed_csv}, found {list(df.columns)}")
    seeds = [str(g).strip().upper() for g in df["gene"].dropna()]
    seeds_in = [s for s in seeds if s in gene_universe]
    seeds_out = [s for s in seeds if s not in gene_universe]

    print(f"Seeds: {len(seeds)} provided, {len(seeds_in)} in multiplex, "
          f"{len(seeds_out)} missing")
    if seeds_out:
        print(f"  Missing seeds (first 10): {seeds_out[:10]}")

    with open(output_path, "w") as f:
        for g in seeds_in:
            f.write(g + "\n")
    print(f"  Saved {len(seeds_in)} seeds → {output_path}\n")
    return seeds_in


def write_config(
    config_path: Path,
    seeds_rel_path: Path,
    layer_rel_paths: dict,   # cell_type -> Path (relative to wdir)
    restart: float,
    delta: float,
) -> None:
    n_layers = len(layer_rel_paths)
    tau_value = round(1.0 / n_layers, 6)

    config = {
        "seed": str(seeds_rel_path),
        "self_loops": 0,
        "r": restart,
        "multiplex": {
            "gene_multiplex": {
                "layers": [str(p) for p in layer_rel_paths.values()],
                # layer names stored as a comment-friendly list in metadata, not in config
                "graph_type": ["00"] * n_layers,   # undirected, unweighted
                "delta": delta,
                "tau": [tau_value] * n_layers,
            }
        },
    }

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved MultiXrank config → {config_path}")
    print(f"  restart (r)      : {restart}")
    print(f"  delta            : {delta}")
    print(f"  tau per layer    : {tau_value}  ({n_layers} cell-type layers)")
    print(f"  Layers           : {sorted(layer_rel_paths.keys())}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build MultiXrank inputs from a cell-type-specific multiplex"
    )
    parser.add_argument("--multiplex-pkl", required=True,
                        help="Path to the cell-type multiplex .pkl")
    parser.add_argument("--seed-csv", required=True,
                        help="CSV with a 'gene' column of seed gene symbols")
    parser.add_argument("--output-dir", default="multixrank_input",
                        help="Output directory (default: multixrank_input)")
    parser.add_argument("--restart", type=float, default=0.7,
                        help="Restart probability r (default 0.7)")
    parser.add_argument("--delta", type=float, default=0.5,
                        help="Inter-layer jump probability delta (default 0.5)")
    parser.add_argument("--include-intercellular", action="store_true",
                        help="Add intercellular edges as an extra layer (default: off)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    multiplex_dir = out / "multiplex" / "gene_multiplex"
    seeds_dir = out / "seeds"
    multiplex_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & project
    G = load_multiplex(args.multiplex_pkl)
    edges_by_celltype, gene_universe = project_by_cell_type(G)

    # 2. Decide which layers go into MultiXrank
    layer_cell_types = sorted(
        ct for ct in edges_by_celltype if ct != "__intercellular__"
    )
    if args.include_intercellular and "__intercellular__" in edges_by_celltype:
        layer_cell_types.append("__intercellular__")
        print("Including intercellular edges as an extra MultiXrank layer.\n")

    # 3. Write layer TSVs
    layer_rel_paths: dict[str, Path] = {}
    for ct in layer_cell_types:
        safe_name = ct.replace(" ", "_").replace("/", "-")
        rel_path = Path("multiplex") / "gene_multiplex" / f"{safe_name}_layer.tsv"
        abs_path = out / rel_path
        n = write_layer_tsv(edges_by_celltype[ct], abs_path, gene_universe)
        print(f"  Wrote layer [{ct}]: {n:,} edges → {abs_path}")
        layer_rel_paths[ct] = rel_path
    print()

    # 4. Metadata
    save_cell_type_metadata(edges_by_celltype, out / "cell_type_metadata.json")

    # 5. Reference graphs
    ref_graph = build_gene_reference_graph(edges_by_celltype)
    ref_path = out / "gene_reference_graph.pkl"
    with open(ref_path, "wb") as f:
        pickle.dump(ref_graph, f, protocol=4)
    print(f"Saved gene reference graph → {ref_path}")
    print(f"  {ref_graph.number_of_nodes():,} nodes, {ref_graph.number_of_edges():,} edges\n")

    # PPI-only reference (backwards-compatible; requires edge_type attr on edges)
    ppi_edges: set = set()
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "ppi":
            gu = str(G.nodes[u].get("gene", "")).strip().upper()
            gv = str(G.nodes[v].get("gene", "")).strip().upper()
            if gu and gv and gu != gv:
                ppi_edges.add(tuple(sorted([gu, gv])))
    ppi_only = nx.Graph()
    for a, b in ppi_edges:
        ppi_only.add_edge(a, b)
    ppi_path = out / "gene_ppi_reference_graph.pkl"
    with open(ppi_path, "wb") as f:
        pickle.dump(ppi_only, f, protocol=4)
    print(f"Saved PPI-only reference graph → {ppi_path}")
    print(f"  {ppi_only.number_of_nodes():,} nodes, {ppi_only.number_of_edges():,} edges\n")

    # 6. Seeds
    seeds_rel_path = Path("seeds") / "seeds.txt"
    write_seeds(args.seed_csv, out / seeds_rel_path, gene_universe)

    # 7. Config
    write_config(out / "config.yml", seeds_rel_path, layer_rel_paths,
                 args.restart, args.delta)

    print("=" * 60)
    print("MultiXrank inputs ready")
    print("=" * 60)
    print(f"Working directory : {out.resolve()}")
    print(f"Layers            : {sorted(layer_rel_paths.keys())}")
    print(f"Run script 2 with : --wdir {out} --config config.yml")
    print("=" * 60)


if __name__ == "__main__":
    main()