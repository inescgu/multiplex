#!/usr/bin/env python3
"""
Step 5 (replaces pre_step9check.py / preflight_seed_reachability.py):
Preflight seed reachability check, run BEFORE step 6 (build_disease_module).

No structural changes needed here vs. the old version — the reference
graph's 'cell_types' edge attribute means the exact same thing it always
did (which cell type(s) a PPI edge belongs to), it just holds plain
cell-type names now instead of "<name>_ppi_layer" stems, since there's no
more stem/registry indirection anywhere in the new pipeline. So
--lcc-per-layer works identically.

Quickly checks whether each seed gene sits in the same connected component
of the reference graph as the bulk of the other seeds, BEFORE running
step6_build_disease_module.py. Seeds that are isolated (or stuck in a small
component with no path to the main seed cluster) will make
build_module()'s expand-and-check loop burn through k+max_bridge
connectivity checks before the LCC fallback drops them anyway — this finds
them up front, cheaply, with one connected_components() call.

NEW: --exclude-celltypes / --exclude-ppi-layer
-------------------------------------------------
step1 (--base-ppi) can add a comprehensive, non-cell-type-specific PPI
network as its own pseudo-cell-type multiplex, namespaced "global_ppi::GENE"
just like every real cell type. If you want to check seed reachability
using ONLY the neuron-specific layers -- i.e. as if that backbone network
were never added -- pass either:

    --exclude-celltypes global_ppi
    --exclude-ppi-layer                 # shorthand for the line above

Either flag removes every node whose "<celltype>::" prefix matches
BEFORE the LCC-per-layer filter and BEFORE component analysis, so the
whole check (and the --exclude-seeds suggestion it prints) reflects the
neurons-only graph, not the backbone-augmented one.

NEW: --exclude-seed-celltypes / --exclude-ppi-seeds
-------------------------------------------------------
Lighter-touch sibling of the above, matching step6's option of the same
name: KEEPS the cell type's nodes in the graph (so it's still available
for connectivity/bridging) but drops any of its seeds from the seed list
before checking reachability -- i.e. this reflects a "full" run where
global_ppi is allowed to help connect things but never anchors the
module itself.

    --exclude-seed-celltypes global_ppi
    --exclude-ppi-seeds                 # shorthand for the line above

Usage:
    python step5_preflight_seed_check.py \
        --ref-graph 04_multixrank_prep/multixrank_hetero_input_pseudobulk/gene_reference_graph.pkl \
        --seeds     04_multixrank_prep/multixrank_hetero_input_pseudobulk/seeds/seeds.txt \
        [--lcc-per-layer] \
        [--exclude-ppi-layer]   # match this flag to whatever you'll pass to step6
"""
import argparse
import pickle
from pathlib import Path
import networkx as nx


def filter_to_lcc_per_layer(H):
    """Mirrors step6's --lcc-per-layer behavior so the preflight check
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


def _celltype_of(node: str) -> str:
    """Cell-type prefix from a namespaced (and, in this script, already
    fully-uppercased) node string, or '' if the node has no '::' in it."""
    return node.rsplit("::", 1)[0] if "::" in node else ""


def filter_out_celltypes(H, excluded_celltypes):
    """Remove every node whose '<celltype>::' prefix is in
    excluded_celltypes, entirely -- node, its edges, everything. This is
    how you get a "neurons only" view of a graph that also has a
    non-cell-type-specific backbone (e.g. 'global_ppi') mixed in: the
    backbone nodes are namespaced exactly like real cell types
    ("global_ppi::GENE"), so dropping the namespace drops the whole layer
    cleanly, including its bipartite identity edges to everything else.

    Nodes in H at this point are already fully uppercased (see main()),
    so excluded_celltypes is matched case-insensitively by uppercasing it
    here too.
    """
    if not excluded_celltypes:
        return H
    excluded_set = {c.strip().upper() for c in excluded_celltypes}
    to_remove = [n for n in H.nodes() if _celltype_of(n) in excluded_set]
    if not to_remove:
        print(f"  [exclude-celltypes] None of {sorted(excluded_set)} found in graph "
              f"(checked against node prefixes) — nothing removed.")
        return H
    remove_set = set(to_remove)
    kept = [n for n in H.nodes() if n not in remove_set]
    H_filtered = H.subgraph(kept).copy()
    print(f"  [exclude-celltypes] Removed cell type(s) {sorted(excluded_set)}: "
          f"{len(to_remove):,} nodes dropped -> {H_filtered.number_of_nodes():,} nodes, "
          f"{H_filtered.number_of_edges():,} edges remain "
          f"(was {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges)")
    return H_filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-graph", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--lcc-per-layer", action="store_true",
                     help="Apply the same per-layer LCC filter step6 would use.")
    ap.add_argument("--exclude-celltypes", nargs="+", default=[], metavar="CELLTYPE",
                     help="Cell type(s) (the '<celltype>::' namespace prefix) to drop "
                          "entirely from the reference graph before any analysis, e.g. "
                          "--exclude-celltypes global_ppi. Case-insensitive.")
    ap.add_argument("--exclude-ppi-layer", action="store_true",
                     help="Shorthand for --exclude-celltypes global_ppi -- drops the "
                          "base/comprehensive PPI pseudo-cell-type added in step1 via "
                          "--base-ppi, leaving only the real (e.g. neuron) cell-type layers.")
    ap.add_argument("--exclude-seed-celltypes", nargs="+", default=[], metavar="CELLTYPE",
                     help="Cell type(s) whose seeds should be dropped from the SEED list only "
                          "before checking reachability -- the cell type's nodes stay in the "
                          "graph. Matches step6's option of the same name. Example: "
                          "--exclude-seed-celltypes global_ppi. Case-insensitive.")
    ap.add_argument("--exclude-ppi-seeds", action="store_true",
                     help="Shorthand for --exclude-seed-celltypes global_ppi.")
    args = ap.parse_args()

    exclude_celltypes = list(args.exclude_celltypes)
    if args.exclude_ppi_layer and "global_ppi" not in {c.lower() for c in exclude_celltypes}:
        exclude_celltypes.append("global_ppi")

    exclude_seed_celltypes = list(args.exclude_seed_celltypes)
    if args.exclude_ppi_seeds and "global_ppi" not in {c.lower() for c in exclude_seed_celltypes}:
        exclude_seed_celltypes.append("global_ppi")

    print(f"Loading reference graph from {args.ref_graph} ...")
    with open(args.ref_graph, "rb") as f:
        H = pickle.load(f)
    mapping = {n: str(n).strip().upper() for n in H.nodes()}
    H = nx.relabel_nodes(H, mapping)
    print(f"  {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

    if exclude_celltypes:
        print(f"\nExcluding cell type(s) {exclude_celltypes} from reference graph ...")
        H = filter_out_celltypes(H, exclude_celltypes)
        print()

    if args.lcc_per_layer:
        print("\nApplying --lcc-per-layer filter (matching step6) ...")
        H = filter_to_lcc_per_layer(H)
        print(f"  After filter: {H.number_of_nodes():,} nodes, "
              f"{H.number_of_edges():,} edges")

    print(f"\nLoading seeds from {args.seeds} ...")
    with open(args.seeds) as f:
        seeds = [line.strip().upper() for line in f if line.strip()]
    print(f"  {len(seeds)} seed genes")

    if exclude_seed_celltypes:
        seed_ct_excl_upper = {c.strip().upper() for c in exclude_seed_celltypes}
        implicit_excluded_seeds = [s for s in seeds if _celltype_of(s) in seed_ct_excl_upper]
        if implicit_excluded_seeds:
            print(f"  [exclude-seed-celltypes] Dropping {len(implicit_excluded_seeds)} seed(s) "
                  f"namespaced under {sorted(seed_ct_excl_upper)} from the SEED list only "
                  f"(graph nodes for that cell type are kept): {implicit_excluded_seeds}")
            seeds = [s for s in seeds if s not in set(implicit_excluded_seeds)]
        else:
            print(f"  [exclude-seed-celltypes] No seeds found under {sorted(seed_ct_excl_upper)} "
                  f"-- nothing to drop.")

    if exclude_celltypes:
        excl_set = {c.strip().upper() for c in exclude_celltypes}
        seeds_in_excluded_ct = [s for s in seeds if _celltype_of(s) in excl_set]
        if seeds_in_excluded_ct:
            print(f"  ⚠ {len(seeds_in_excluded_ct)} seed(s) belong to an excluded cell type "
                  f"and will be reported as missing from the (now-filtered) graph:")
            print(f"    {seeds_in_excluded_ct}")

    seeds_missing = [s for s in seeds if s not in H.nodes()]
    seeds_in_h = [s for s in seeds if s in H.nodes()]
    if seeds_missing:
        print(f"\n  {len(seeds_missing)} seed(s) not present in graph at all:")
        print(f"    {seeds_missing}")

    print("\nComputing connected components of full reference graph ...")
    components = list(nx.connected_components(H))
    components.sort(key=len, reverse=True)
    print(f"  {len(components)} component(s) total, "
          f"largest = {len(components[0]):,} nodes")

    node_to_comp = {}
    for i, comp in enumerate(components):
        for n in comp:
            node_to_comp[n] = i

    seed_comp_ids = {s: node_to_comp[s] for s in seeds_in_h}

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

    unreachable = sorted(s for s, c in seed_comp_ids.items() if c != main_comp_id)
    unreachable += seeds_missing

    print("\n" + "=" * 70)
    if exclude_celltypes:
        print(f"(Reference graph analyzed with cell type(s) excluded: {exclude_celltypes})")
    if exclude_seed_celltypes:
        print(f"(Seeds from cell type(s) {exclude_seed_celltypes} dropped from the seed list only; "
              f"those nodes remain in the graph)")
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