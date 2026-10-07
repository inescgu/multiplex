#!/usr/bin/env python3
"""
Step 11c-faithful, Phase 1 (v2 -- WINDOWED neighbor graph): build the REAL
hidef_finder.py ClusterGraph and run its ACTUAL consensus()/persistence
calculation against your step 11b partition ensemble.

WHY THIS VERSION EXISTS
------------------------
The original step11c_phase1_consensus.py deliberately used a COMPLETE
resolution graph (every partition treated as a "neighbor" of every other
partition), documented as a departure from real hidef_finder.py at the
time, "for not much practical gain over inventing a 3D neighbor rule."

That turned out to be wrong in a way that matters: real HiDeF persistence
measures how many NEARBY resolutions a cluster survives across -- it's a
local, chain-like notion (this is exactly what hidef_finder.run()'s own
`neighbors` parameter controls, e.g. neighbors=10 in the single-layer
script). Comparing every partition against every other partition erases
that locality: once two partitions share a Jaccard-similar cluster,
transitivity through the complete graph tends to chain nearly every
surviving consensus cluster into the same (or nearly the same) connected-
component size -- and len_component (= persistence) is exactly that
component size. The practical symptom is a near-constant persistence
value across almost all clusters, usually close to n_partitions, instead
of a real distribution.

THE FIX
-------
Sort partitions by gamma WITHIN each omega group (a restart at the same
gamma is treated as adjacent too), and link each partition only to its
--neighbors nearest partitions in that ordering -- a direct analogue of
hidef_finder's own log-distance neighbor chain, extended to handle
multiple omega values and restarts. Different omega groups are NOT
cross-linked by default (--link-omega enables that, see --help) so a
persistence value stays interpretable as "survived across this many
nearby gamma steps at this coupling strength," same intuition as your
prior monoplex HiDeF run.

Everything else -- Cluster.calculate_similarity, jaccard_matrix (used
inside add_clusters), k-clique percolation, consensus's collapse/
persistence-counting logic -- is still hidef_finder's own code, untouched.
Only the shape of the resolution graph fed into it changes.

COST NOTE: this only touches Phase 1 (fast). You do NOT need to re-run
step 11b's resolution sweep. You DO need to re-run Phase 2's weave()
afterwards to get corrected .nodes/.edges/.gml output -- but check the
new persistence distribution out of THIS script first (it's cheap) before
committing to the slow multi-day Phase 2 rerun.

USAGE
-----
    python step11c_phase1_consensus.py \
        --ensemble 11_hidef_hetero/ensemble.pkl \
        --output 11_hidef_results_faithful/consensus_windowed.pkl \
        --minsize 4 \
        --jaccard 0.75 \
        --chi 5 \
        --consensus-p 75 \
        --neighbors 10
"""
import argparse
import pickle
from pathlib import Path
from provenance import write_manifest
import numpy as np
import scipy as sp
import networkx as nx

# Reuse hidef_finder's own classes/functions -- do not reimplement.
from hidef.hidef_finder import Cluster, ClusterGraph, consensus


def load_ensemble(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["partitions"], data["node_names"], data["cell_type"], data["gene"]


def membership_array_to_csr(membership, minsize=4):
    """Same as the original script: drop clusters smaller than minsize,
    sort remaining clusters by size descending -- this ordering matters,
    it's what hidef_finder itself does."""
    membership = np.asarray(membership, dtype=int)
    ids, counts = np.unique(membership, return_counts=True)
    keep = [(cid, cnt) for cid, cnt in zip(ids, counts) if cnt >= minsize]
    keep.sort(key=lambda x: x[1], reverse=True)

    row, col = [], []
    for i, (cid, _) in enumerate(keep):
        members = np.where(membership == cid)[0]
        row.extend([i] * len(members))
        col.extend(members.tolist())
    row = np.array(row, dtype=int)
    col = np.array(col, dtype=int)
    data = np.ones_like(row, dtype=int)
    C = sp.sparse.coo_matrix((data, (row, col)), shape=(len(keep), len(membership)))
    return C.tocsr()


def build_windowed_resolution_graph(partitions, n_nodes, minsize, jaccard_threshold,
                                     neighbors=10, link_omega=False):
    """Build the ClusterGraph with a genuine neighbor-windowed resolution
    graph: partitions are grouped by omega, sorted by (gamma, restart)
    within each group, and each one is linked only to its `neighbors`
    nearest partitions in that ordering. cluG.add_clusters() itself -- the
    Jaccard comparison and edge-building logic -- is still hidef_finder's
    own unmodified code; only which pairs get compared changes.

    IMPORTANT: hidef_finder.add_clusters() does NOT use the node key you
    give resolution_graph directly -- internally it recomputes the id as
    '{:.4f}'.format(surrogate_res) and looks THAT up. So every node name
    here must equal '{:.4f}'.format(<the surrogate_res passed for it>),
    exactly like the original (complete-graph) script did with
    f"{i/100.0:.4f}". Descriptive names (e.g. "o1_r0003") will KeyError.
    """
    cluG = ClusterGraph()
    cluG.graph["sim_threshold"] = jaccard_threshold
    cluG.graph["num_leaves"] = n_nodes

    by_omega = {}
    for i, p in enumerate(partitions):
        by_omega.setdefault(p["omega"], []).append(i)

    resolution_graph = nx.Graph()
    resname_of = {}          # orig_i -> node key string, e.g. "0.0300"
    surrogate_res_of = {}    # orig_i -> the float that produced that key
    ordered_indices = []     # processing order: grouped by omega, sorted by (gamma, restart)

    global_rank = 0
    for omega in sorted(by_omega):
        idxs_sorted = sorted(by_omega[omega],
                              key=lambda i: (partitions[i]["gamma"], partitions[i]["restart"]))
        group_names = []
        for orig_i in idxs_sorted:
            surrogate_res = global_rank / 100.0
            resname = f"{surrogate_res:.4f}"
            resname_of[orig_i] = resname
            surrogate_res_of[orig_i] = surrogate_res
            group_names.append(resname)
            resolution_graph.add_node(resname, padded=True)
            ordered_indices.append(orig_i)
            global_rank += 1
        # replace the "window-based neighbor linking" block in
        # build_windowed_resolution_graph (11c), and the equivalent loop in
        # run_phase1 (neighbors sweep) and windowed_consensus (11i)
        gammas_sorted = sorted({partitions[i]["gamma"] for i in idxs_sorted})
        grank = {gm: k for k, gm in enumerate(gammas_sorted)}
        ranks = [grank[partitions[i]["gamma"]] for i in idxs_sorted]
        for a in range(len(group_names)):
            for b in range(a + 1, len(group_names)):
                if ranks[b] - ranks[a] > neighbors:
                    break
                resolution_graph.add_edge(group_names[a], group_names[b])

    if link_omega and len(by_omega) > 1:
        # Optional: also link partitions with the SAME (gamma, restart) pair
        # across adjacent omega values, so persistence can additionally
        # reflect robustness across nearby coupling strengths. Off by
        # default -- turn on only if you deliberately want omega treated
        # as a second smooth resolution-like axis.
        omega_sorted = sorted(by_omega)
        for oa, ob in zip(omega_sorted[:-1], omega_sorted[1:]):
            keyed_a = {(partitions[i]["gamma"], partitions[i]["restart"]): resname_of[i]
                       for i in by_omega[oa]}
            keyed_b = {(partitions[i]["gamma"], partitions[i]["restart"]): resname_of[i]
                       for i in by_omega[ob]}
            for key in set(keyed_a) & set(keyed_b):
                resolution_graph.add_edge(keyed_a[key], keyed_b[key])

    for step_i, orig_i in enumerate(ordered_indices, 1):
        p = partitions[orig_i]
        resname = resname_of[orig_i]
        surrogate_res = surrogate_res_of[orig_i]
        mat = membership_array_to_csr(p["membership"], minsize=minsize)
        resolution_graph.nodes[resname]["matrix"] = mat
        cluG.add_clusters(resolution_graph, surrogate_res)
        if step_i % 10 == 0 or step_i == len(partitions):
            print(f"  [{step_i}/{len(partitions)}] partition (gamma={p['gamma']}, "
                  f"omega={p['omega']}, restart={p['restart']}): {mat.shape[0]:,} clusters "
                  f"(size >= {minsize}) added -- cluG now has "
                  f"{cluG.number_of_nodes():,} nodes, {cluG.number_of_edges():,} edges")

    return cluG


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", required=True, help="Pickle from step 11b")
    ap.add_argument("--output", required=True, help="Where to save the consensus pickle")
    ap.add_argument("--minsize", type=int, default=4,
                     help="hidef_finder's own hardcoded default in partition_to_membership_matrix "
                          "is 4 (default here). Override for continuity with earlier results if needed.")
    ap.add_argument("--jaccard", type=float, default=0.75,
                     help="hidef_finder's --t: Jaccard/containment threshold for linking clusters "
                          "across partitions (default 0.75, hidef_finder's own default)")
    ap.add_argument("--chi", type=int, default=5,
                     help="hidef_finder's --k: minimum connected-component size AND the k in "
                          "k-clique percolation (default 5, hidef_finder's own default)")
    ap.add_argument("--consensus-p", type=int, default=75,
                     help="hidef_finder's --p: consensus threshold (percent) for which nodes "
                          "survive collapse_cluster_graph (default 75, hidef_finder's own default)")
    ap.add_argument("--neighbors", type=int, default=10,
                     help="Window size in GAMMA STEPS: each partition is compared against all "
                          "partitions (incl. restarts) within this many gamma steps in the same "
                          "omega group.")
    ap.add_argument("--link-omega", action="store_true",
                     help="Also link matching (gamma, restart) partitions across adjacent omega "
                          "values, treating omega as a second smooth resolution-like axis. "
                          "Off by default -- each omega group's persistence stays independently "
                          "interpretable.")
    args = ap.parse_args()
    write_manifest(Path(args.output).parent, args, inputs=[args.ensemble])
    
    partitions, node_names, cell_types, genes = load_ensemble(Path(args.ensemble))
    n_nodes = len(node_names)
    print(f"Loaded ensemble: {len(partitions)} partitions, {n_nodes} nodes")
    print(f"Omega values present: {sorted({p['omega'] for p in partitions})}")
    print(f"Neighbor window: {args.neighbors}  |  link_omega: {args.link_omega}\n")

    print("Building windowed-neighbor ClusterGraph ...")
    cluG = build_windowed_resolution_graph(
        partitions, n_nodes, args.minsize, args.jaccard,
        neighbors=args.neighbors, link_omega=args.link_omega,
    )
    print(f"\nClusterGraph complete: {cluG.number_of_nodes():,} cluster-nodes, "
          f"{cluG.number_of_edges():,} edges\n")

    print(f"Running consensus() (chi={args.chi}, p={args.consensus_p}) ...")
    cluG_collapsed_w_len = consensus(cluG, k=args.chi, f=1.0, p=args.consensus_p)
    print(f"consensus() produced {len(cluG_collapsed_w_len):,} consensus clusters\n")

    sizes = [int(np.sum(c[0])) for c in cluG_collapsed_w_len]
    persistences = [c[1] for c in cluG_collapsed_w_len]
    if sizes:
        print(f"Consensus cluster size stats: min={min(sizes)} max={max(sizes)} "
              f"mean={np.mean(sizes):.1f}")
        print(f"Persistence stats: min={min(persistences)} max={max(persistences)} "
              f"mean={np.mean(persistences):.1f}  distinct values={len(set(persistences)):,}  "
              f"(max possible = {len(partitions)} partitions)")
        if len(set(persistences)) <= 2:
            print("\n⚠ persistence is still (near-)constant -- if you were expecting this fix to "
                  "resolve that, try a smaller --neighbors window, or double-check --jaccard/--chi "
                  "aren't so loose that most partitions link to each other anyway.")
    else:
        print("⚠ consensus() produced ZERO clusters -- check --chi/--consensus-p/--minsize/"
              "--neighbors before proceeding to Phase 2.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "cluG_collapsed_w_len": cluG_collapsed_w_len,
            "node_names": node_names,
            "n_partitions": len(partitions),
            "args": vars(args),
        }, f)
    print(f"\n✓ Saved -> {out_path}")
    print("\nNext: sanity-check the persistence distribution above (or reload this pickle "
          "and histogram len_component values) BEFORE committing to the slow Phase 2 weave rerun.")


if __name__ == "__main__":
    main()