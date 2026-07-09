#!/usr/bin/env python3
"""
Per-layer HiDeF baseline for the integrated multiplex network.

For each layer (sliced by the same scheme as run_hidef_multiplex.py),
run HiDeF independently on the induced subgraph, then reconcile
communities across layers via:
  1. Gene-level Jaccard similarity (collapsing node_id -> gene symbol)
  2. Greedy meta-community assembly over a similarity threshold
  3. Cross-layer co-occurrence matrix

Outputs are directly comparable to the multiplex run.
"""

import os
import sys
import json
import pickle
import argparse
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig

# ---- HiDeF ----
try:
    from hidef.logger import PackageLogger
    LOGGER = PackageLogger('HiDeF_PerLayer')
except ImportError:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    LOGGER = logging.getLogger('HiDeF_PerLayer')

try:
    from hidef.hidef_finder import *
    import hidef.hidef_finder as hidef_finder
    hidef_finder.LOGGER = LOGGER
except ImportError:
    print("ERROR: could not import hidef_finder"); sys.exit(1)


# ============================================================================
# 1. Load + decompose (reuses the same scheme logic as the multiplex script)
# ============================================================================

def load_multiplex(graph_pkl, registry_csv):
    with open(graph_pkl, "rb") as f:
        G = pickle.load(f)
    reg = pd.read_csv(registry_csv).set_index("node_id")
    print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Loaded registry: {len(reg):,} entries")
    return G, reg


def assign_layer_key(u_attr, v_attr, edata, scheme):
    """Not needed anymore — decompose_to_layers handles it."""
    pass


def decompose_to_layers(G, scheme):
    """
    Decompose gene-level reference graph by edge type.
    G should have edge_types attribute on each edge (from script 1).
    """
    layers = defaultdict(list)
    
    for u, v, d in G.edges(data=True):
        # Get edge types (could be tuple or set depending on how it was saved)
        etypes = d.get("edge_types", ("unknown",))
        if isinstance(etypes, str):
            etypes = (etypes,)
        
        # Each edge belongs to all its types
        for etype in etypes:
            layers[etype].append((u, v))
    
    return dict(layers)


# ============================================================================
# 2. Per-layer HiDeF
# ============================================================================

def build_layer_igraph(edges):
    """Build an igraph induced by these edges (compact local indexing)."""
    nodes = sorted({n for e in edges for n in e})
    nid_to_idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), directed=False)
    g.add_edges([(nid_to_idx[u], nid_to_idx[v]) for u, v in edges])
    g.simplify(multiple=False)
    g.vs["name"] = nodes
    return g, nodes


def hidef_on_layer(edges, params):
    g, node_ids = build_layer_igraph(edges)
    if g.ecount() == 0 or g.vcount() < 3:
        return [], node_ids
    cluG = hidef_finder.run(
        [g],
        jaccard=params["jaccard_threshold"],
        sample=0.9,
        minres=params["min_resolution"],
        maxres=params["max_resolution"],
        alg=params["algorithm"],
        maxn=None,
        density=0.1,
        neighbors=10,
        numthreads=params["numthreads"],
    )
    collapsed = hidef_finder.consensus(
        cluG, k=params["min_persistence"], f=1.0, p=params["consensus_p"]
    )
    communities = []
    for binary_vec, persistence in collapsed:
        member_ids = [node_ids[i] for i in np.where(binary_vec)[0]]
        if params["max_community_size"] and len(member_ids) > params["max_community_size"]:
            continue
        communities.append({
            "member_node_ids": member_ids,
            "size": len(member_ids),
            "persistence": int(persistence),
        })
    return communities, node_ids


# ============================================================================
# 3. Reconciliation across layers
# ============================================================================

def to_gene_sets(communities, registry):
    """Collapse node_id members to gene-symbol sets (loses cell-type identity)."""
    out = []
    for c in communities:
        genes = set(registry.loc[c["member_node_ids"], "gene"])
        out.append(genes)
    return out


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def reconcile(per_layer_comms, registry, sim_threshold=0.5):
    """
    Build a meta-graph where nodes = (layer_key, local_comm_idx) and
    edges connect community pairs with gene-Jaccard >= sim_threshold.
    Connected components -> meta-communities.
    """
    # flat list with provenance
    flat = []                             # (layer_key, local_idx, gene_set, comm)
    for layer_key, comms in per_layer_comms.items():
        gene_sets = to_gene_sets(comms, registry)
        for i, (gs, c) in enumerate(zip(gene_sets, comms)):
            flat.append((layer_key, i, gs, c))

    meta = nx.Graph()
    for idx, (lk, i, gs, c) in enumerate(flat):
        meta.add_node(idx, layer=lk, local_idx=i,
                      size=c["size"], persistence=c["persistence"],
                      gene_set=gs)

    # pairwise — quadratic but fine for typical HiDeF output sizes
    print(f"Reconciling {len(flat):,} communities pairwise...")
    for (i, a), (j, b) in combinations(enumerate(flat), 2):
        s = jaccard(a[2], b[2])
        if s >= sim_threshold:
            meta.add_edge(i, j, jaccard=s)

    # meta-communities = connected components
    meta_communities = []
    for cc in nx.connected_components(meta):
        member_records = []
        gene_union = set()
        for idx in cc:
            d = meta.nodes[idx]
            member_records.append({
                "layer": str(d["layer"]),
                "local_idx": d["local_idx"],
                "size": d["size"],
                "persistence": d["persistence"],
            })
            gene_union |= d["gene_set"]
        layers_spanned = {r["layer"] for r in member_records}
        meta_communities.append({
            "n_component_communities": len(cc),
            "layers_spanned": sorted(layers_spanned),
            "n_layers_spanned": len(layers_spanned),
            "gene_union_size": len(gene_union),
            "genes": sorted(gene_union),
            "components": member_records,
        })
    meta_communities.sort(key=lambda m: -m["gene_union_size"])
    return meta, meta_communities


def cooccurrence_matrix(per_layer_comms, registry):
    """
    Gene-level co-occurrence: for each gene, which layers placed it
    in *some* community? Returns (genes x layers) binary DataFrame.
    """
    layer_keys = list(per_layer_comms.keys())
    gene_to_layers = defaultdict(set)
    for lk, comms in per_layer_comms.items():
        for c in comms:
            for g in set(registry.loc[c["member_node_ids"], "gene"]):
                gene_to_layers[g].add(lk)
    genes = sorted(gene_to_layers.keys())
    mat = pd.DataFrame(0, index=genes, columns=[str(k) for k in layer_keys])
    for g, lks in gene_to_layers.items():
        for lk in lks:
            mat.at[g, str(lk)] = 1
    return mat


# ============================================================================
# 4. Save
# ============================================================================

def save_per_layer(per_layer_comms, registry, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for lk, comms in per_layer_comms.items():
        for i, c in enumerate(comms):
            recs = registry.loc[c["member_node_ids"]]
            for nid, r in recs.iterrows():
                rows.append({
                    "layer": str(lk),
                    "local_comm_idx": i,
                    "size": c["size"],
                    "persistence": c["persistence"],
                    "node_id": nid,
                    "source": r["source"],
                    "cell_type": r["cell_type"],
                    "gene": r["gene"],
                })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "per_layer_communities_members.csv"), index=False
    )
    with open(os.path.join(out_dir, "per_layer_communities.pkl"), "wb") as f:
        pickle.dump(per_layer_comms, f)


def save_reconciliation(meta_graph, meta_communities, cooc_mat, out_dir):
    with open(os.path.join(out_dir, "meta_graph.pkl"), "wb") as f:
        pickle.dump(meta_graph, f)

    flat_rows = []
    for mid, m in enumerate(meta_communities):
        for comp in m["components"]:
            flat_rows.append({
                "meta_id": mid,
                "n_layers_spanned": m["n_layers_spanned"],
                "gene_union_size": m["gene_union_size"],
                **comp,
            })
    pd.DataFrame(flat_rows).to_csv(
        os.path.join(out_dir, "meta_communities_components.csv"), index=False
    )

    summary = []
    for mid, m in enumerate(meta_communities):
        summary.append({
            "meta_id": mid,
            "n_component_communities": m["n_component_communities"],
            "n_layers_spanned": m["n_layers_spanned"],
            "layers_spanned": "|".join(m["layers_spanned"]),
            "gene_union_size": m["gene_union_size"],
        })
    pd.DataFrame(summary).to_csv(
        os.path.join(out_dir, "meta_communities_summary.csv"), index=False
    )

    # full gene lists
    with open(os.path.join(out_dir, "meta_communities.json"), "w") as f:
        json.dump([{k: v for k, v in m.items() if k != "components"} | {
            "components": m["components"]
        } for m in meta_communities], f, indent=2)

    cooc_mat.to_csv(os.path.join(out_dir, "gene_layer_cooccurrence.csv"))
    # quick stat: how many layers each gene appears in
    gene_layer_counts = cooc_mat.sum(axis=1).sort_values(ascending=False)
    gene_layer_counts.to_csv(
        os.path.join(out_dir, "gene_n_layers.csv"),
        header=["n_layers_with_community"]
    )


# ============================================================================
# 5. Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Per-layer HiDeF baseline + reconciliation")
    p.add_argument("--graph_pkl",    required=True)
    p.add_argument("--registry_csv", required=True)
    p.add_argument("--output_dir",   default="./hidef_perlayer_results")
    p.add_argument("--scheme",       default="hybrid",
                   choices=["edge_type", "source", "source_celltype", "hybrid"])
    p.add_argument("--min_layer_edges", type=int, default=10)
    p.add_argument("--sim_threshold", type=float, default=0.5,
                   help="gene-Jaccard threshold to link communities across layers")

    # HiDeF params (same as multiplex script)
    p.add_argument("--jaccard_threshold", type=float, default=0.75)
    p.add_argument("--min_resolution",    type=float, default=0.01)
    p.add_argument("--max_resolution",    type=float, default=10.0)
    p.add_argument("--min_persistence",   type=int,   default=5)
    p.add_argument("--consensus_p",       type=int,   default=75)
    p.add_argument("--algorithm",  default="leiden", choices=["leiden", "louvain"])
    p.add_argument("--numthreads", type=int, default=None)
    p.add_argument("--max_community_size", type=int, default=1848)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load + decompose
    G_mx, registry = load_multiplex(args.graph_pkl, args.registry_csv)
    layers = decompose_to_layers(G_mx, args.scheme)
    print(f"\n{len(layers)} layers found: {list(layers.keys())}")
    
    edge_types = {}
    for u, v, d in G_mx.edges(data=True):
        et = d.get("edge_type", "unknown")
        edge_types[et] = edge_types.get(et, 0) + 1
    print(f"Edge type distribution: {edge_types}")

    # 2. Run HiDeF per layer
    hidef_params = {
        "jaccard_threshold": args.jaccard_threshold,
        "min_resolution":    args.min_resolution,
        "max_resolution":    args.max_resolution,
        "min_persistence":   args.min_persistence,
        "consensus_p":       args.consensus_p,
        "algorithm":         args.algorithm,
        "numthreads":        args.numthreads,
        "max_community_size":args.max_community_size,
    }

    per_layer_comms = {}
    for lk, edges in layers.items():
        if len(edges) < args.min_layer_edges:
            print(f"  Skipping {lk}: {len(edges)} edges (< {args.min_layer_edges})")
            continue
        print(f"\n--- Layer {lk}: {len(edges):,} edges ---")
        comms, _ = hidef_on_layer(edges, hidef_params)
        print(f"    → {len(comms)} persistent communities")
        per_layer_comms[lk] = comms

    save_per_layer(per_layer_comms, registry, args.output_dir)

    # 3. Reconcile
    print("\n" + "="*70)
    print("RECONCILING COMMUNITIES ACROSS LAYERS")
    print("="*70)
    meta_graph, meta_communities = reconcile(
        per_layer_comms, registry, sim_threshold=args.sim_threshold
    )
    print(f"  → {len(meta_communities)} meta-communities")

    cooc_mat = cooccurrence_matrix(per_layer_comms, registry)
    save_reconciliation(meta_graph, meta_communities, cooc_mat, args.output_dir)

    # quick stats
    spans = [m["n_layers_spanned"] for m in meta_communities]
    sizes = [m["gene_union_size"] for m in meta_communities]
    if spans:
        print(f"\n  Meta layers spanned: {min(spans)}–{max(spans)} (mean {np.mean(spans):.2f})")
        print(f"  Gene-union sizes   : {min(sizes)}–{max(sizes)} (mean {np.mean(sizes):.1f})")
        n_multi = sum(1 for s in spans if s > 1)
        print(f"  Multi-layer meta-communities: {n_multi}/{len(spans)}")

    print(f"\n✓ Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()