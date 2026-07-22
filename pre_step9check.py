#!/usr/bin/env python3
"""
Preflight: Seed Reachability Check
-----------------------------------
Quickly checks, BEFORE running step9_build_disease_module.py, whether each
seed gene sits in the same connected component of the reference graph as
the bulk of the other seeds. Seeds that are isolated (or stuck in a small
component with no path to the main seed cluster) will make
build_module()'s expand-and-check loop burn through k+max_bridge
connectivity checks before the LCC fallback drops them anyway. This script
finds them up front, cheaply, with one connected_components() call.

Usage:
    python preflight_seed_reachability.py \
        --ref-graph 04_multixrank_prep/multixrank_input_consolidated/gene_reference_graph.pkl \
        --seeds     04_multixrank_prep/multixrank_input_consolidated/seeds/seeds.txt \
        [--lcc-per-layer]   # match this flag to whatever you'll pass to step9

Prints a component breakdown for all seeds and a ready-to-paste
--exclude-seeds list for anything not in the main seed component.
"""
import argparse
import pickle
from pathlib import Path
import networkx as nx


def filter_to_lcc_per_layer(H):
    """Mirrors step9's --lcc-per-layer behavior so the preflight check
    matches whatever graph build_module will actually see."""
    layers = set()
    for _, _, d in H.edges(data=True):
        layers |= d.get("cell_types", set())
    if not layers:
        return H
    kept_nodes = set()
    for layer in sorted(layers):
        layer_edges = [(u, v) for u, v, d in H.edges(data=True)
                        if layer in d.get("cell_types", set())]
        layer_nodes = set(n for e in layer_edges for n in e)
        Lsub = H.subgraph(layer_nodes)
        if Lsub.number_of_nodes() == 0:
            continue
        lcc = max(nx.connected_components(Lsub), key=len)
        kept_nodes |= lcc
    return H.subgraph(kept_nodes).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-graph", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--lcc-per-layer", action="store_true",
                     help="Apply the same per-layer LCC filter step9 would use.")
    args = ap.parse_args()

    print(f"Loading reference graph from {args.ref_graph} ...")
    with open(args.ref_graph, "rb") as f:
        H = pickle.load(f)
    mapping = {n: str(n).strip().upper() for n in H.nodes()}
    H = nx.relabel_nodes(H, mapping)
    print(f"  {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

    if args.lcc_per_layer:
        print("\nApplying --lcc-per-layer filter (matching step9) ...")
        H = filter_to_lcc_per_layer(H)
        print(f"  After filter: {H.number_of_nodes():,} nodes, "
              f"{H.number_of_edges():,} edges")

    print(f"\nLoading seeds from {args.seeds} ...")
    with open(args.seeds) as f:
        seeds = [line.strip().upper() for line in f if line.strip()]
    print(f"  {len(seeds)} seed genes")

    seeds_missing = [s for s in seeds if s not in H.nodes()]
    seeds_in_h = [s for s in seeds if s in H.nodes()]
    if seeds_missing:
        print(f"\n  {len(seeds_missing)} seed(s) not present in graph at all:")
        print(f"    {seeds_missing}")

    # Whole-graph component membership (not just seed-induced subgraph) —
    # this is what actually determines whether build_module's bridging
    # loop can ever connect two seeds via ranked genes.
    print("\nComputing connected components of full reference graph ...")
    components = list(nx.connected_components(H))
    components.sort(key=len, reverse=True)
    print(f"  {len(components)} component(s) total, "
          f"largest = {len(components[0]):,} nodes")

    node_to_comp = {}
    for i, comp in enumerate(components):
        for n in comp:
            node_to_comp[n] = i

    seed_comp_ids = {}
    for s in seeds_in_h:
        seed_comp_ids[s] = node_to_comp[s]

    from collections import Counter
    comp_counts = Counter(seed_comp_ids.values())
    main_comp_id = comp_counts.most_common(1)[0][0]
    main_comp_seed_count = comp_counts[main_comp_id]

    print(f"\nSeed distribution across components:")
    for comp_id, n in sorted(comp_counts.items(), key=lambda x: -x[1]):
        tag = "  <-- MAIN cluster (majority of seeds)" if comp_id == main_comp_id else "  <-- UNREACHABLE from main seed cluster"
        seeds_here = sorted(s for s, c in seed_comp_ids.items() if c == comp_id)
        print(f"  Component #{comp_id} (size {len(components[comp_id]):,} nodes): "
              f"{n} seed(s){tag}")
        print(f"    {seeds_here}")

    unreachable = sorted(
        s for s, c in seed_comp_ids.items() if c != main_comp_id
    )
    unreachable += seeds_missing  # not in graph at all = also unreachable

    print("\n" + "=" * 70)
    if unreachable:
        print(f"⚠  {len(unreachable)} seed(s) cannot reach the main seed "
              f"cluster ({main_comp_seed_count}/{len(seeds_in_h)} seeds).")
        print("Paste this into your sbatch script:\n")
        print("    --exclude-seeds " + " ".join(unreachable) + " \\")
    else:
        print("✓ All seeds present in the graph share the same component. "
              "No exclusions needed on reachability grounds.")
    print("=" * 70)


if __name__ == "__main__":
    main()