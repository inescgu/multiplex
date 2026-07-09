#!/usr/bin/env python3
"""
Build MultiXrank Inputs (Edge-Driven Inter-Layer Architecture)
================================================================
Decomposes a cell-type-specific multiplex into MultiXrank's expected directory
structure with EXPLICIT inter-layer edges.

Layer Types
-----------
1. INTRA-LAYER (within cell type):
   - PPI edges only
   - File: <cell_type>_ppi_layer.tsv
2. INTER-LAYER (between cell types):
   - Ligand-receptor, KEGG, Reactome edges
   - File: <cell_type_1>_<cell_type_2>_inter_layer.tsv
   - Undirected: both (A,B) and (B,A) included

Outputs
-------
<output-dir>/
├── multiplex/
│   └── gene_multiplex/
│       ├── <cell_type_1>_ppi_layer.tsv
│       ├── <cell_type_2>_ppi_layer.tsv
│       ├── <cell_type_1>_<cell_type_2>_inter_layer.tsv
│       └── ...
├── seeds/
│   └── seeds.txt
├── config.yml
├── cell_type_metadata.json
├── layer_info.json
├── gene_reference_graph.pkl
└── gene_ppi_reference_graph.pkl
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
# Projection: Edge-Type Aware
# ---------------------------------------------------------------------------
def project_by_edge_type(G: nx.Graph) -> tuple[dict, dict, set]:
    """
    Split edges by BOTH cell-type AND edge-type.
    
    Returns
    -------
    intra_edges : dict[str, set[tuple[str, str]]]
        cell_type -> set of (geneA, geneB) PPI pairs within that cell type
    inter_edges : dict[tuple[str, str], set[tuple[str, str]]]
        (cell_type_1, cell_type_2) -> set of (geneA, geneB) inter-layer pairs
    gene_universe : set[str]
        all gene symbols
    """
    intra_edges: dict[str, set] = defaultdict(set)
    inter_edges: dict[tuple, set] = defaultdict(set)
    gene_universe: set[str] = set()

    # Collect all genes
    for nid, d in G.nodes(data=True):
        gene = d.get("gene")
        if gene:
            gene_universe.add(str(gene).strip().upper())

    # Process edges
    missing_attr: list = []
    edge_type_counts = defaultdict(int)
    
    for u, v, d in G.edges(data=True):
        du = G.nodes[u]
        dv = G.nodes[v]
        gu = str(du.get("gene", "")).strip().upper()
        gv = str(dv.get("gene", "")).strip().upper()
        
        if not gu or not gv or gu == gv:
            continue  # skip self-loops or missing gene symbols
        
        ct_u = du.get("cell_type")
        ct_v = dv.get("cell_type")
        edge_type = d.get("edge_type", "unknown")
        
        if ct_u is None or ct_v is None:
            missing_attr.append((u, v))
            continue
        
        a, b = sorted([gu, gv])
        edge_type_counts[edge_type] += 1
        
        if ct_u == ct_v:
            # Intra-layer: only PPI
            if edge_type == "ppi":
                intra_edges[str(ct_u)].add((a, b))
        else:
            # Inter-layer: ligand-receptor, KEGG, Reactome
            if edge_type in ["ligand_receptor", "pathway", "reactome"]:
                ct_pair = tuple(sorted([str(ct_u), str(ct_v)]))
                inter_edges[ct_pair].add((a, b))

    if missing_attr:
        print(f"  ⚠ {len(missing_attr):,} edges skipped (missing cell_type attribute)")
    
    print("Edge-type projection:")
    for et, count in sorted(edge_type_counts.items()):
        print(f"  {et:<20} {count:>10,} edges")
    print()
    
    print("Intra-layer (PPI within cell type):")
    print(f"  Total unique genes : {len(gene_universe):,}")
    for ct, edges in sorted(intra_edges.items()):
        print(f"  {ct:<40} {len(edges):>10,} PPI edges")
    print()
    
    print("Inter-layer (between cell types):")
    for (ct1, ct2), edges in sorted(inter_edges.items()):
        print(f"  {ct1} ↔ {ct2:<30} {len(edges):>10,} inter-layer edges")
    print()
    
    return dict(intra_edges), dict(inter_edges), gene_universe

# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_layer_tsv(edges: set, path: Path, gene_universe: set) -> int:
    """Write edge list as two-column TSV (no header). Returns edge count."""
    valid = [(a, b) for (a, b) in edges if a in gene_universe and b in gene_universe]
    if not valid:
        print(f"    ⚠ Layer {path.name} is EMPTY — writing placeholder self-loop.")
        placeholder = next(iter(gene_universe)) if gene_universe else "PLACEHOLDER"
        valid = [(placeholder, placeholder)]
    df = pd.DataFrame(valid, columns=["gene1", "gene2"])
    df.to_csv(path, sep="\t", index=False, header=False)
    return len(valid)

def sanitize_name(name: str) -> str:
    """Convert cell type name to safe filename."""
    return name.replace(" ", "_").replace("/", "-").replace(",", "")

def save_cell_type_metadata(intra_edges: dict, inter_edges: dict, output_path: Path) -> None:
    """Persist layer names and sizes for downstream scripts."""
    intra_layers = {ct: len(e) for ct, e in intra_edges.items()}
    inter_layers = {f"{ct1}_{ct2}": len(e) for (ct1, ct2), e in inter_edges.items()}
    metadata = {
        "cell_types": sorted(intra_edges.keys()),
        "intra_layer_edges": intra_layers,
        "inter_layer_edges": inter_layers,
        "total_intra_edges": sum(intra_layers.values()),
        "total_inter_edges": sum(inter_layers.values()),
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved cell-type metadata → {output_path}\n")

def save_layer_info(intra_edges: dict, inter_edges: dict, output_path: Path) -> None:
    """Map layer filenames to their types (intra vs inter)."""
    layer_info = {}
    
    # Intra-layers (single cell type)
    for ct in intra_edges.keys():
        safe_name = sanitize_name(ct)
        layer_name = f"{safe_name}_ppi_layer"
        layer_info[layer_name] = {
            "type": "intra",
            "cell_types": [ct],
            "edge_type": "ppi"
        }
    
    # Inter-layers (between cell types)
    for (ct1, ct2) in inter_edges.keys():
        safe_ct1 = sanitize_name(ct1)
        safe_ct2 = sanitize_name(ct2)
        layer_name = f"{safe_ct1}_{safe_ct2}_inter_layer"
        layer_info[layer_name] = {
            "type": "inter",
            "cell_types": [ct1, ct2],
            "edge_type": "inter-layer"
        }
    
    with open(output_path, "w") as f:
        json.dump(layer_info, f, indent=2)
    print(f"Saved layer info → {output_path}\n")

def build_gene_reference_graph(intra_edges: dict, inter_edges: dict) -> nx.Graph:
    """Union graph across all edges."""
    H = nx.Graph()

    def ensure_edge_attrs(u, v):
        if not H.has_edge(u, v):
            H.add_edge(u, v, cell_types=set(), inter_cell_types=set(), edge_type=None)

    # Add intra-layer (PPI) edges
    for ct, edges in intra_edges.items():
        for a, b in edges:
            ensure_edge_attrs(a, b)
            H[a][b]["cell_types"].add(ct)
            H[a][b]["edge_type"] = "ppi"

    # Add inter-layer edges
    for (ct1, ct2), edges in inter_edges.items():
        for a, b in edges:
            ensure_edge_attrs(a, b)
            H[a][b]["inter_cell_types"].add((ct1, ct2))
            if H[a][b]["edge_type"] == "ppi":
                H[a][b]["edge_type"] = "both"
            else:
                H[a][b]["edge_type"] = "inter"

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
    intra_layer_paths: list,
    inter_layer_paths: list,
    restart: float,
    delta: float,
) -> None:
    """Write MultiXrank config with both intra and inter layers."""
    all_layers = intra_layer_paths + inter_layer_paths
    n_layers = len(all_layers)
    
    if n_layers == 0:
        print("ERROR: No layers to write to config!")
        return
    
    tau_value = round(1.0 / n_layers, 6)
    config = {
        "seed": str(seeds_rel_path),
        "self_loops": 0,
        "r": restart,
        "multiplex": {
            "gene_multiplex": {
                "layers": [str(p) for p in sorted(all_layers)],
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
    print(f"  tau per layer    : {tau_value}  ({n_layers} total layers)")
    print(f"  Intra-layers     : {len(intra_layer_paths)} (PPI within cell types)")
    print(f"  Inter-layers     : {len(inter_layer_paths)} (between cell types)")
    print(f"  Total layers     : {n_layers}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build MultiXrank inputs with edge-driven inter-layer architecture"
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
    args = parser.parse_args()

    out = Path(args.output_dir)
    multiplex_dir = out / "multiplex" / "gene_multiplex"
    seeds_dir = out / "seeds"
    multiplex_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & project
    G = load_multiplex(args.multiplex_pkl)
    intra_edges, inter_edges, gene_universe = project_by_edge_type(G)

    # 2. Write intra-layer TSVs (PPI within single cell types)
    print("Writing intra-layer files (PPI within single cell types)...")
    intra_layer_paths: list = []
    for ct in sorted(intra_edges.keys()):
        safe_name = sanitize_name(ct)
        rel_path = Path("multiplex") / "gene_multiplex" / f"{safe_name}_ppi_layer.tsv"
        abs_path = out / rel_path
        n = write_layer_tsv(intra_edges[ct], abs_path, gene_universe)
        print(f"  [{ct}]: {n:,} PPI edges → {rel_path}")
        intra_layer_paths.append(rel_path)
    print()

    # 3. Write inter-layer TSVs (between cell types)
    print("Writing inter-layer files (between cell types)...")
    inter_layer_paths: list = []
    for (ct1, ct2) in sorted(inter_edges.keys()):
        safe_ct1 = sanitize_name(ct1)
        safe_ct2 = sanitize_name(ct2)
        rel_path = Path("multiplex") / "gene_multiplex" / f"{safe_ct1}_{safe_ct2}_inter_layer.tsv"
        abs_path = out / rel_path
        n = write_layer_tsv(inter_edges[(ct1, ct2)], abs_path, gene_universe)
        print(f"  [{ct1} ↔ {ct2}]: {n:,} edges → {rel_path}")
        inter_layer_paths.append(rel_path)
    print()

    # 4. Metadata
    save_cell_type_metadata(intra_edges, inter_edges, out / "cell_type_metadata.json")
    save_layer_info(intra_edges, inter_edges, out / "layer_info.json")

    # 5. Reference graphs
    ref_graph = build_gene_reference_graph(intra_edges, inter_edges)
    ref_path = out / "gene_reference_graph.pkl"
    with open(ref_path, "wb") as f:
        pickle.dump(ref_graph, f, protocol=4)
    print(f"Saved gene reference graph → {ref_path}")
    print(f"  {ref_graph.number_of_nodes():,} nodes, {ref_graph.number_of_edges():,} edges\n")

    # PPI-only reference
    ppi_only = nx.Graph()
    all_ppi_edges = set().union(*intra_edges.values()) if intra_edges else set()
    for a, b in all_ppi_edges:
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
    write_config(out / "config.yml", seeds_rel_path, intra_layer_paths, inter_layer_paths,
                 args.restart, args.delta)

    print("=" * 60)
    print("MultiXrank inputs ready (edge-driven inter-layer)")
    print("=" * 60)
    print(f"Working directory : {out.resolve()}")
    print(f"Intra-layers      : {len(intra_layer_paths)}")
    print(f"Inter-layers      : {len(inter_layer_paths)}")
    print(f"Total layers      : {len(intra_layer_paths) + len(inter_layer_paths)}")
    print("=" * 60)

if __name__ == "__main__":
    main()