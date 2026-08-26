#!/usr/bin/env python3
"""
Step 3c (a.k.a. your "step11c"): Weave the partition ensemble from step 3b
into a HiDeF-style persistent hierarchy over the full heterogeneous
supra-graph, using HiDeF's own weaver module (hidef.weaver.Weaver) -- the
same persistence/containment machinery your monoplex HiDeF run used, just
fed a partition ensemble generated on the multi-cell-type supra-graph
instead of a single homogeneous PPI graph.

CHANGELOG vs. the version that errored with
"Weaver.__init__() got an unexpected keyword argument 'terminals'"
--------------------------------------------------------------------
Confirmed directly from the HiDeF README / readthedocs tutorial
("Organize a list of clusters into a DAG"):

    from hidef import weaver
    wv = weaver.Weaver()
    H = wv.weave(P, cutoff=1.0)

Two things were wrong in the previous version, not just the missing
`terminals` kwarg:

  1. Weaver() takes NO constructor arguments in the documented usage --
     there is no way to hand it node labels directly. Node identity has
     to be tracked externally, by keeping every cluster's bit positions
     in the SAME node order you used to build them (exactly node_names
     from step 3b's ensemble pickle).

  2. `P` is NOT a list of partitions (one community-id-per-node array per
     resolution) -- it is a flat list of INDIVIDUAL CLUSTERS, one
     bit-string of length n_nodes per cluster ('1' if that node is a
     member of that cluster, '0' otherwise). A single partition with 12
     communities contributes 12 separate strings to P, not one. This
     script now explicitly flattens step 3b's partitions into clusters
     before calling weave() -- see partitions_to_cluster_strings().

hidef.weaver.weave() is documented as being agnostic to how its input
clusters were produced ("the construction of DAG is independent of the
upstream clustering algorithm"), so this remains a legitimate, supported
use of the library even though the clusters came from a heterogeneous,
weighted, multi-etype supra-graph rather than hidef_finder's single-edge-
list `--g` path.

NOTE ON SCALE: building one full-length bit-string per cluster is exactly
what the HiDeF tutorial shows, but it was only ever illustrated on toy
examples (8 data points). At ~140k nodes and potentially thousands of
clusters across 30 partitions, this can mean several hundred MB to a few
GB of plain Python strings. The script prints a size estimate before
calling weave() so you can see this coming; if it turns out to be too
slow/memory-hungry in practice, the proven-scalable alternative is your
monoplex script's own hidef_finder.run()/hidef_finder.consensus() path,
which handles this internally more efficiently -- worth checking whether
hidef_finder exposes a way to build its internal community graph from an
externally supplied partition ensemble, if this raw weaver path becomes a
bottleneck on the full 140k-node graph.

NOTE ON API VERSION (still applies): hidef.weaver's exact attribute names
for the woven hierarchy have shifted slightly across releases. This
script tries the common wv.hier networkx-DiGraph interface used in the
tutorials (nodes carry an 'index' attribute -- positions into node_names
-- and usually 'persistence'), and if your installed version differs,
extract_communities() prints the Weaver object's actual attributes and
the hierarchy's node-attribute keys so you can adjust, rather than
silently producing wrong output.
"""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from hidef import weaver as hidef_weaver
except ImportError:
    print("ERROR: could not import hidef.weaver -- pip install hidef")
    raise

import weaver_fast_patch  # monkeypatches Weaver._build to be fast

def load_ensemble(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["partitions"], data["node_names"], data["cell_type"], data["gene"]


def partitions_to_cluster_strings(partitions, n_nodes, min_cluster_size=1, dedup=True):
    """Flatten step 3b's partition ensemble (one community-id-per-node
    array per (gamma, omega, restart)) into the flat list of individual
    cluster bit-strings that hidef.weaver.Weaver().weave() actually wants
    -- see module docstring.

    weave()'s cost scales roughly as O(n_clusters^2) (pairwise Jaccard/
    containment across every cluster), so this now does two things to
    keep n_clusters under control instead of passing every raw cluster
    straight through:

    1. --min-cluster-size drops clusters below a member-count threshold.
       At high resolution these are typically abundant and near-useless
       for persistence (a 2-node "cluster" persisting across resolutions
       tells you little).

    2. Deduplication (on by default): if the SAME exact set of nodes shows
       up as a cluster in multiple partitions -- very common for
       disconnected/weakly-connected fragments that no resolution change
       can split or merge -- only one copy is kept. Each unique cluster's
       `n_resolutions_seen` count (how many of the input partitions
       produced it) is tracked and returned, which is itself a useful
       persistence-like signal you can inspect even before running
       weave()'s own containment analysis.

    A near-flat community count across your resolution sweep (as you're
    seeing: ~9,030 -> 9,186 across gamma=0.01 to gamma=10) is a strong
    sign that dedup will help a lot here -- most of that stability likely
    comes from many small clusters being identical at every resolution.
    """
    raw_bits_by_hash = {}  # bitstring -> {"bits": array, "n_resolutions_seen": int}
    n_raw = 0
    n_dropped_small = 0

    for p in partitions:
        membership = np.asarray(p["membership"], dtype=int)
        if len(membership) != n_nodes:
            raise ValueError(f"partition (gamma={p['gamma']}, omega={p['omega']}, "
                              f"restart={p['restart']}) has {len(membership)} entries, "
                              f"expected {n_nodes} -- ensemble and node_names are out of sync")
        ids, counts = np.unique(membership, return_counts=True)
        for comm_id, size in zip(ids, counts):
            n_raw += 1
            if size < min_cluster_size:
                n_dropped_small += 1
                continue
            bits = (membership == comm_id).astype(np.uint8)
            key = bits.tobytes() if dedup else (bits.tobytes(), n_raw)  # unique key per cluster if not deduping
            entry = raw_bits_by_hash.get(key)
            if entry is None:
                raw_bits_by_hash[key] = {"bits": bits, "n_resolutions_seen": 1}
            else:
                entry["n_resolutions_seen"] += 1

    unique_clusters = list(raw_bits_by_hash.values())
    P = ["".join(map(str, c["bits"].tolist())) for c in unique_clusters]
    prevalence = [c["n_resolutions_seen"] for c in unique_clusters]

    n_after_size_filter = n_raw - n_dropped_small
    est_before_gb = 8 * n_raw ** 2 / 1e9
    est_after_size_gb = 8 * n_after_size_filter ** 2 / 1e9
    est_after_dedup_gb = 8 * len(P) ** 2 / 1e9

    print(f"Raw clusters across {len(partitions)} partitions: {n_raw:,}")
    print(f"  dropped by --min-cluster-size {min_cluster_size}: {n_dropped_small:,} "
          f"-> {n_after_size_filter:,} remain")
    print(f"  after deduplication: {len(P):,} unique clusters "
          f"({'dedup enabled' if dedup else 'dedup disabled'})")
    print(f"Estimated dense O(n^2) pairwise-matrix memory: "
          f"{est_before_gb:.1f} GB (raw) -> {est_after_size_gb:.1f} GB (size-filtered) "
          f"-> {est_after_dedup_gb:.1f} GB (size-filtered + deduped)")
    if est_after_dedup_gb > 50:
        print("  ⚠ still large -- consider raising --min-cluster-size further, or see the "
              "module docstring's NOTE ON SCALE")
    return P, prevalence


def weave_hierarchy(P, cutoff: float):
    print(f"Weaving {len(P):,} clusters (cutoff={cutoff}) ...")
    wv = hidef_weaver.Weaver()
    wv.weave(P, cutoff=cutoff)
    print("Weave complete.")
    return wv


def extract_communities(wv, node_names, max_size: int):
    hier = getattr(wv, "hier", None)
    if hier is None:
        print("⚠ wv.hier not found on this hidef version. Attributes available on Weaver:")
        print("   ", [a for a in dir(wv) if not a.startswith("_")])
        raise AttributeError("Adapt extract_communities() to your installed hidef's Weaver API "
                              "(see printed attribute list above).")

    L = getattr(wv, "_L_clusters", None)
    if L is None:
        print("⚠ wv._L_clusters not found -- did you apply the weaver_fast_patch fix? "
              "Falling back to wv._assignment, which may be WRONG for internal nodes.")
        L = wv._assignment

    sample_node = next(iter(hier.nodes(data=True)))
    print(f"  Example hierarchy node: {sample_node[0]!r} -> attrs: {list(sample_node[1].keys())}")

    communities = {}
    for comm_id, attrs in hier.nodes(data=True):
        idx = attrs.get("index")
        if idx is not None:
            # idx is the cluster's ROW INDEX into the assignment matrix L,
            # not a list of member indices -- look membership up via L[idx]
            row = np.asarray(L[idx]).astype(bool)
            members = [node_names[j] for j in np.nonzero(row)[0]]
        else:
            # terminal/leaf node = a single original data point;
            # comm_id itself is that node's position in node_names
            members = attrs.get("members")
            if members is None and isinstance(comm_id, (int, np.integer)):
                members = [node_names[comm_id]]

        if not members:
            continue
        if len(members) > max_size:
            continue

        by_ct = defaultdict(list)
        for m in members:
            ct, gene = m.split("::", 1)
            by_ct[ct].append(gene)

        communities[str(comm_id)] = {
            "members": members,
            "size": len(members),
            "persistence": attrs.get("persistence", None),
            "n_cell_types_spanned": len(by_ct),
            "members_by_cell_type": dict(by_ct),
        }
    return communities


def save_results(communities, hier, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "heterogeneous_hidef_communities.pkl", "wb") as f:
        pickle.dump(communities, f)
    print(f"✓ Saved communities pickle -> {output_dir / 'heterogeneous_hidef_communities.pkl'}")

    summary = {
        cid: {
            "size": c["size"],
            "persistence": c["persistence"],
            "n_cell_types_spanned": c["n_cell_types_spanned"],
            "cell_types": sorted(c["members_by_cell_type"].keys()),
        }
        for cid, c in communities.items()
    }
    with open(output_dir / "heterogeneous_hidef_communities_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary JSON -> {output_dir / 'heterogeneous_hidef_communities_summary.json'}")

    if hier is not None:
        try:
            import networkx as nx
            nx.write_gml(hier, output_dir / "heterogeneous_hidef_hierarchy.gml", stringizer=str)
            print(f"✓ Saved hierarchy DAG -> {output_dir / 'heterogeneous_hidef_hierarchy.gml'}")
        except Exception as e:
            print(f"⚠ Could not save hierarchy GML ({e}); communities pickle/JSON are still complete.")

    sizes = [c["size"] for c in communities.values()]
    spans = [c["n_cell_types_spanned"] for c in communities.values()]
    with open(output_dir / "heterogeneous_hidef_summary.txt", "w") as f:
        f.write("HETEROGENEOUS HIDEF (SUPRA-GRAPH) CLUSTERING RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of communities: {len(communities)}\n")
        if sizes:
            f.write(f"Community sizes: {min(sizes)} - {max(sizes)} (avg {np.mean(sizes):.1f})\n")
            f.write(f"Cell types spanned per community: {min(spans)} - {max(spans)} "
                    f"(avg {np.mean(spans):.1f})\n")
            n_cross = sum(1 for s in spans if s > 1)
            f.write(f"Cross-cell-type communities (span > 1 cell type): {n_cross} / {len(communities)}\n\n")
        f.write("Top 10 communities by size:\n")
        top10 = sorted(communities.items(), key=lambda x: x[1]["size"], reverse=True)[:10]
        for i, (cid, c) in enumerate(top10, 1):
            f.write(f"  {i}. Community {cid}: {c['size']} genes across "
                     f"{c['n_cell_types_spanned']} cell type(s) "
                     f"({', '.join(sorted(c['members_by_cell_type'].keys()))}), "
                     f"persistence={c['persistence']}\n")
    print(f"✓ Saved text summary -> {output_dir / 'heterogeneous_hidef_summary.txt'}")


def main():
    ap = argparse.ArgumentParser(description="Step 3c: weave step 3b's partition ensemble into a "
                                               "persistent hierarchy with hidef.weaver")
    ap.add_argument("--ensemble", required=True, help="Pickle from step 3b")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cutoff", type=float, default=0.75,
                     help="Jaccard/containment threshold (HiDeF default tau=sigma=0.75)")
    ap.add_argument("--max-size", type=int, default=1330,
                     help="Drop communities larger than this (matches the cutoff used in your "
                          "monoplex hidef script)")
    ap.add_argument("--min-cluster-size", type=int, default=5,
                     help="Drop clusters with fewer than this many members before weave() -- "
                          "the main lever for controlling weave()'s O(n_clusters^2) cost. Your "
                          "diagnostic showed ~40%% of raw clusters have 2-4 members; start here "
                          "and raise it further (e.g. 10) if the printed post-filter estimate is "
                          "still too large for your available memory.")
    ap.add_argument("--no-dedup", action="store_true",
                     help="Disable deduplication of identical clusters across partitions "
                          "(dedup is on by default and is essentially free -- only disable this "
                          "for debugging).")
    args = ap.parse_args()

    partitions, node_names, cell_types, genes = load_ensemble(Path(args.ensemble))
    print(f"Loaded ensemble: {len(partitions)} partitions, {len(node_names)} nodes\n")

    P, prevalence = partitions_to_cluster_strings(
        partitions, len(node_names),
        min_cluster_size=args.min_cluster_size, dedup=not args.no_dedup,
    )
    wv = weave_hierarchy(P, args.cutoff)
    communities = extract_communities(wv, node_names, args.max_size)
    print(f"\nFound {len(communities)} persistent communities (<= {args.max_size} members)\n")

    save_results(communities, getattr(wv, "hier", None), Path(args.output_dir))

    print("\n" + "=" * 60)
    print("Step 3c complete")
    print("=" * 60)


if __name__ == "__main__":
    main()