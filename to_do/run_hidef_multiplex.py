#!/usr/bin/env python3
"""
Run HiDeF clustering on an integrated multiplex network (multilayer mode).

Uses HiDeF's native multiplex support: pass a list of igraph.Graph objects
sharing a common, index-aligned vertex set. Internally HiDeF routes to
leidenalg/louvain optimise_partition_multiplex with per-layer weights.
"""

import os
import sys
import json
import pickle
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig

# ---- HiDeF imports (same pattern as your single-layer script) ----
try:
    from hidef.logger import PackageLogger
    LOGGER = PackageLogger('HiDeF_Multiplex')
except ImportError:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    LOGGER = logging.getLogger('HiDeF_Multiplex')

try:
    from hidef.hidef_finder import *
    import hidef.hidef_finder as hidef_finder
    hidef_finder.LOGGER = LOGGER
except ImportError:
    print("ERROR: could not import hidef_finder"); sys.exit(1)


# ============================================================================
# 1. Load multiplex + registry
# ============================================================================

def load_multiplex(graph_pkl: str, registry_csv: str):
    print(f"Loading multiplex graph: {graph_pkl}")
    with open(graph_pkl, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"Loading node registry: {registry_csv}")
    reg = pd.read_csv(registry_csv).set_index("node_id")
    # columns: source, cell_type, gene
    print(f"  {len(reg):,} registry entries")
    return G, reg


# ============================================================================
# 2. Decompose into layers
# ============================================================================

def assign_layer_key(u_attr, v_attr, edata, scheme: str):
    """Return the layer key for a given edge under the chosen scheme."""
    et = edata.get("edge_type", "unknown")

    if scheme == "edge_type":
        return et

    if scheme == "source":
        # intra-source vs cross-source layers, separated by edge type
        s = tuple(sorted({u_attr["source"], v_attr["source"]}))
        return (et, s)

    if scheme == "source_celltype":
        # one layer per (source, cell_type) for PPI; pathway/LR each their own
        if et == "ppi":
            return ("ppi", u_attr["source"], u_attr["cell_type"])
        return (et,)

    if scheme == "hybrid":
        # PPI per layer, pathway split by DB, LR as a single layer
        if et == "ppi":
            return ("ppi", u_attr["source"], u_attr["cell_type"])
        if et == "pathway":
            return ("pathway", edata.get("pathway_db", "unknown"))
        if et == "ligand_receptor":
            return ("lr",)
        return (et,)

    raise ValueError(f"Unknown scheme: {scheme}")


def decompose_to_layers(G: nx.Graph, scheme: str):
    """Return dict: layer_key -> list of (u, v) edges."""
    layers = defaultdict(list)
    for u, v, d in G.edges(data=True):
        key = assign_layer_key(G.nodes[u], G.nodes[v], d, scheme)
        layers[key].append((u, v))
    print(f"\nDecomposition '{scheme}' produced {len(layers)} layers:")
    for k, es in sorted(layers.items(), key=lambda kv: -len(kv[1])):
        print(f"  {str(k):<60} {len(es):>12,} edges")
    return dict(layers)


# ============================================================================
# 3. Build vertex-aligned igraphs
# ============================================================================

def build_aligned_igraphs(layers: dict, n_nodes: int, min_layer_edges: int = 1):
    """
    Build one igraph.Graph per layer, all with exactly `n_nodes` vertices,
    indexed 0..n_nodes-1 to match the NodeRegistry integer IDs.

    Layers with fewer than `min_layer_edges` are dropped (would contribute
    nothing in multiplex optimisation but slow things down).
    """
    aligned, kept_keys = [], []
    for key, edges in layers.items():
        if len(edges) < min_layer_edges:
            print(f"  Skipping tiny layer {key} ({len(edges)} edges)")
            continue
        g = ig.Graph(n=n_nodes, directed=False)
        g.add_edges(edges)
        g.simplify(multiple=False)        # HiDeF does this too; safe to pre-empt
        g.vs["name"] = list(range(n_nodes))   # numeric names == registry IDs
        aligned.append(g)
        kept_keys.append(key)
        print(f"  Layer {str(key):<55}  V={g.vcount():,}  E={g.ecount():,}")
    return aligned, kept_keys


# ============================================================================
# 4. Run HiDeF in multiplex mode
# ============================================================================

def run_hidef_multiplex(
    Gs, layer_keys,
    layer_weights=None,
    jaccard_threshold=0.75,
    min_resolution=0.01,
    max_resolution=10.0,
    min_persistence=5,
    alg="leiden",
    numthreads=None,
    consensus_p=75,
    max_community_size=None,
):
    print("\n" + "="*70)
    print(f"RUNNING HIDEF IN MULTIPLEX MODE ({len(Gs)} layers)")
    print("="*70)

    if layer_weights is None:
        layer_weights = [1.0] * len(Gs)
    assert len(layer_weights) == len(Gs)
    print("Layer weights:")
    for k, w in zip(layer_keys, layer_weights):
        print(f"  {str(k):<55}  w = {w}")

    cluG = hidef_finder.run(
        Gs,
        jaccard=jaccard_threshold,
        sample=0.9,
        minres=min_resolution,
        maxres=max_resolution,
        alg=alg,
        maxn=None,
        density=0.1,
        neighbors=10,
        numthreads=numthreads,
        layer_weights=layer_weights,         # <-- key multiplex arg
    )
    print(f"Cluster graph: {len(cluG.nodes()):,} clusters")

    print("Computing consensus...")
    collapsed_w_len = hidef_finder.consensus(
        cluG, k=min_persistence, f=1.0, p=consensus_p
    )

    communities = {}
    for cid, (binary_vec, persistence) in enumerate(collapsed_w_len):
        member_ids = np.where(binary_vec)[0].tolist()
        if max_community_size is not None and len(member_ids) > max_community_size:
            continue
        communities[cid] = {
            "member_node_ids": member_ids,
            "size": len(member_ids),
            "persistence": int(persistence),
        }

    print(f"Found {len(communities):,} persistent communities")
    return cluG, communities


# ============================================================================
# 5. Annotate communities using the registry
# ============================================================================

def annotate_communities(communities, registry_df):
    """Attach gene symbols + per-layer composition stats to each community."""
    annotated = {}
    for cid, c in communities.items():
        ids = c["member_node_ids"]
        rows = registry_df.loc[ids]
        sources    = Counter(rows["source"])
        celltypes  = Counter(zip(rows["source"], rows["cell_type"]))
        genes      = sorted(set(rows["gene"]))

        annotated[cid] = {
            "size": c["size"],
            "persistence": c["persistence"],
            "n_unique_genes": len(genes),
            "n_sources": len(sources),
            "n_layers_spanned": len(celltypes),
            "source_breakdown":   dict(sources),
            "layer_breakdown":    {f"{s}|{ct}": n for (s, ct), n in celltypes.items()},
            "unique_genes": genes,
            "members": [
                {"node_id": int(i),
                 "source": rows.loc[i, "source"],
                 "cell_type": rows.loc[i, "cell_type"],
                 "gene": rows.loc[i, "gene"]}
                for i in ids
            ],
        }
    return annotated


# ============================================================================
# 6. Save outputs
# ============================================================================

def save_outputs(communities, cluG, layer_keys, layer_weights, params, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # full pickle
    with open(os.path.join(out_dir, "hidef_multiplex_communities.pkl"), "wb") as f:
        pickle.dump(communities, f)

    # cluster graph (optional, can be heavy)
    with open(os.path.join(out_dir, "hidef_multiplex_clusterG.pkl"), "wb") as f:
        pickle.dump(cluG, f)

    # JSON summary (no member lists)
    summary = {
        str(cid): {k: v for k, v in c.items()
                   if k not in ("members", "unique_genes")}
        for cid, c in communities.items()
    }
    with open(os.path.join(out_dir, "hidef_multiplex_summary.json"), "w") as f:
        json.dump({
            "parameters": params,
            "layers": [str(k) for k in layer_keys],
            "layer_weights": layer_weights,
            "communities": summary,
        }, f, indent=2)

    # flat CSV: one row per (community, member)
    rows = []
    for cid, c in communities.items():
        for m in c["members"]:
            rows.append({
                "community_id": cid,
                "size": c["size"],
                "persistence": c["persistence"],
                "n_layers_spanned": c["n_layers_spanned"],
                **m,
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "hidef_multiplex_members.csv"), index=False
    )

    # per-community composition CSV
    comp_rows = []
    for cid, c in communities.items():
        row = {"community_id": cid, "size": c["size"],
               "persistence": c["persistence"],
               "n_unique_genes": c["n_unique_genes"],
               "n_layers_spanned": c["n_layers_spanned"]}
        row.update({f"layer::{k}": v for k, v in c["layer_breakdown"].items()})
        row.update({f"source::{k}": v for k, v in c["source_breakdown"].items()})
        comp_rows.append(row)
    pd.DataFrame(comp_rows).fillna(0).to_csv(
        os.path.join(out_dir, "hidef_multiplex_composition.csv"), index=False
    )

    print(f"\n✓ Outputs written to {out_dir}")


# ============================================================================
# 7. Main
# ============================================================================

def parse_layer_weights(spec, layer_keys):
    """spec is either None, a single float, or 'key1=2.0,key2=1.0,...'."""
    if spec is None:
        return [1.0] * len(layer_keys)
    if "=" not in spec:
        try:
            w = float(spec)
            return [w] * len(layer_keys)
        except ValueError:
            pass
    weights = [1.0] * len(layer_keys)
    overrides = {}
    for tok in spec.split(","):
        k, v = tok.split("=")
        overrides[k.strip()] = float(v)
    for i, key in enumerate(layer_keys):
        sk = str(key)
        if sk in overrides:
            weights[i] = overrides[sk]
    return weights


def main():
    p = argparse.ArgumentParser(description="HiDeF on integrated multiplex network")
    p.add_argument("--graph_pkl",    required=True,
                   help="integrated_multiplex.pkl (nx.Graph with int node IDs)")
    p.add_argument("--registry_csv", required=True,
                   help="node_registry.csv (node_id, source, cell_type, gene)")
    p.add_argument("--output_dir",   default="./hidef_multiplex_results")
    p.add_argument("--scheme",       default="hybrid",
                   choices=["edge_type", "source", "source_celltype", "hybrid"],
                   help="how to slice the multiplex into HiDeF layers")
    p.add_argument("--layer_weights", default=None,
                   help="float, or 'layer_key=w,layer_key=w,...'")
    p.add_argument("--min_layer_edges", type=int, default=1)
    p.add_argument("--jaccard_threshold", type=float, default=0.75)
    p.add_argument("--min_resolution",    type=float, default=0.01)
    p.add_argument("--max_resolution",    type=float, default=10.0)
    p.add_argument("--min_persistence",   type=int,   default=5)
    p.add_argument("--consensus_p",       type=int,   default=75)
    p.add_argument("--algorithm",  default="leiden", choices=["leiden", "louvain"])
    p.add_argument("--numthreads", type=int, default=None)
    p.add_argument("--max_community_size", type=int, default=None,
                   help="drop communities larger than this (mirrors your 1330 cap)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load
    G_mx, registry = load_multiplex(args.graph_pkl, args.registry_csv)
    n_nodes = registry.index.max() + 1
    # Sanity: registry IDs must match graph node IDs
    missing = set(G_mx.nodes()) - set(registry.index)
    if missing:
        print(f"WARNING: {len(missing)} graph nodes missing from registry")

    # 2. Decompose
    layers = decompose_to_layers(G_mx, args.scheme)

    # 3. Align
    print("\nBuilding vertex-aligned igraphs...")
    Gs, layer_keys = build_aligned_igraphs(layers, n_nodes, args.min_layer_edges)
    if not Gs:
        print("ERROR: no usable layers"); sys.exit(1)

    # 4. Weights
    weights = parse_layer_weights(args.layer_weights, layer_keys)

    # 5. Run HiDeF
    cluG, communities = run_hidef_multiplex(
        Gs, layer_keys,
        layer_weights=weights,
        jaccard_threshold=args.jaccard_threshold,
        min_resolution=args.min_resolution,
        max_resolution=args.max_resolution,
        min_persistence=args.min_persistence,
        alg=args.algorithm,
        numthreads=args.numthreads,
        consensus_p=args.consensus_p,
        max_community_size=args.max_community_size,
    )

    # 6. Annotate + save
    annotated = annotate_communities(communities, registry)
    params = vars(args).copy()
    save_outputs(annotated, cluG, layer_keys, weights, params, args.output_dir)

    # quick summary
    sizes = [c["size"] for c in annotated.values()]
    spans = [c["n_layers_spanned"] for c in annotated.values()]
    if sizes:
        print(f"\n  Community sizes: {min(sizes)}–{max(sizes)} (mean {np.mean(sizes):.1f})")
        print(f"  Layers spanned : {min(spans)}–{max(spans)} (mean {np.mean(spans):.2f})")
    print("\nDone.")


if __name__ == "__main__":
    main()