#!/usr/bin/env python3
"""
step11c_neighbors_sweep.py

Addresses the --neighbors 3 question directly. --neighbors controls how
many nearby resolutions each partition is compared against when building
the windowed resolution graph (step11c_phase1_consensus_windowed.py). It's
a free parameter that materially changes which clusters look "persistent":
- too LARGE (your original complete-graph version was effectively
  neighbors=infinity) -> transitivity chains nearly every cluster into one
  giant component -> persistence collapses to a near-constant value at/near
  n_partitions for almost everything (exactly the failure mode you
  documented and fixed).
- too SMALL -> each partition only compared to 1-2 neighbors -> not enough
  overlap to detect real cross-resolution persistence -> spuriously LOW,
  noisy persistence values, and a hierarchy that's mostly artifacts of
  which nearby gamma happened to land where.

You dropped from 10 to 3 by feel. This script reruns Phase 1 (cheap -- no
weave, no genome-scale reclustering) at several --neighbors values against
your EXISTING ensemble.pkl, and reports:
  1. How much the persistence DISTRIBUTION shape changes (n_clusters,
     max persistence, fraction of clusters at max persistence).
  2. How much cluster IDENTITY changes: for each neighbors setting, match
     clusters between it and a reference setting (--reference-neighbors,
     default 10) by max gene-set Jaccard, then Spearman-correlate their
     persistence values. High correlation across a wide range of
     --neighbors values means your headline result isn't an artifact of
     this one arbitrary choice; low correlation means it is, and you need
     to report a sweep (or pick a principled criterion, e.g. the smallest
     --neighbors that stabilizes the persistence distribution) rather than
     the value that "looked right."

Reuses the exact windowed-graph-building and consensus() logic from
step11c_phase1_consensus_windowed.py (hidef_finder's own Cluster/
ClusterGraph/consensus underneath, untouched).

USAGE
-----
    python step11c_neighbors_sweep.py \
        --ensemble 11_hidef_hetero/ensemble.pkl \
        --neighbor-values 2 3 5 10 15 20 \
        --reference-neighbors 10 \
        --minsize 4 --jaccard 0.75 --chi 5 --consensus-p 75 \
        --output-dir 11_hidef_neighbors_sweep
"""
import argparse
import pickle
from pathlib import Path
from provenance import write_manifest
import numpy as np
import pandas as pd
import scipy as sp
import networkx as nx
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hidef.hidef_finder import Cluster, ClusterGraph, consensus


def load_ensemble(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["partitions"], data["node_names"]


def membership_array_to_csr(membership, minsize=4):
    membership = np.asarray(membership, dtype=int)
    ids, counts = np.unique(membership, return_counts=True)
    keep = [(cid, cnt) for cid, cnt in zip(ids, counts) if cnt >= minsize]
    keep.sort(key=lambda x: x[1], reverse=True)
    row, col = [], []
    for i, (cid, _) in enumerate(keep):
        members = np.where(membership == cid)[0]
        row.extend([i] * len(members))
        col.extend(members.tolist())
    data = np.ones(len(row), dtype=int)
    return sp.sparse.coo_matrix((data, (row, col)), shape=(len(keep), len(membership))).tocsr()


def run_phase1(partitions, node_names, minsize, jaccard, chi, consensus_p, neighbors):
    """Single-omega windowed Phase 1 (matches step11c's logic when all
    partitions share omega=1, your `--omegas 1.0` default): sort partitions
    by (gamma, restart), link each to its `neighbors` nearest, run
    hidef_finder's own consensus() on the result."""
    n_nodes = len(node_names)
    cluG = ClusterGraph()
    cluG.graph["sim_threshold"] = jaccard
    cluG.graph["num_leaves"] = n_nodes

    idxs_sorted = sorted(range(len(partitions)),
                          key=lambda i: (partitions[i]["gamma"], partitions[i]["restart"]))
    resolution_graph = nx.Graph()
    resname_of = {}
    for rank, orig_i in enumerate(idxs_sorted):
        resname = f"{rank / 100.0:.4f}"
        resname_of[orig_i] = resname
        resolution_graph.add_node(resname, padded=True)
    names_in_order = [resname_of[i] for i in idxs_sorted]
    gammas_sorted = sorted({partitions[i]["gamma"] for i in idxs_sorted})
    grank = {gm: k for k, gm in enumerate(gammas_sorted)}
    ranks = [grank[partitions[i]["gamma"]] for i in idxs_sorted]
    for a in range(len(names_in_order)):
        for b in range(a + 1, len(names_in_order)):
            if ranks[b] - ranks[a] > neighbors:
                break
            resolution_graph.add_edge(names_in_order[a], names_in_order[b])

    for orig_i in idxs_sorted:
        p = partitions[orig_i]
        resname = resname_of[orig_i]
        mat = membership_array_to_csr(p["membership"], minsize=minsize)
        resolution_graph.nodes[resname]["matrix"] = mat
        cluG.add_clusters(resolution_graph, float(resname))

    return consensus(cluG, k=chi, f=1.0, p=consensus_p)


def clusters_from_consensus(cluG_collapsed_w_len, node_names):
    clusters = {}
    persistence = {}
    for i, (bool_vec, persist) in enumerate(cluG_collapsed_w_len):
        genes = {node_names[j] for j in np.where(bool_vec.astype(bool))[0]}
        name = f"C{i}"
        clusters[name] = genes
        persistence[name] = float(persist)
    return clusters, persistence


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def match_and_correlate(ref_clusters, ref_persist, other_clusters, other_persist, min_overlap=3):
    rows = []
    for rname, rgenes in ref_clusters.items():
        best_j, best_o = 0.0, None
        for oname, ogenes in other_clusters.items():
            if len(rgenes & ogenes) < min_overlap:
                continue
            j = jaccard(rgenes, ogenes)
            if j > best_j:
                best_j, best_o = j, oname
        if best_o is not None:
            rows.append({"ref_cluster": rname, "ref_persistence": ref_persist[rname],
                          "match_persistence": other_persist[best_o], "jaccard": best_j})
    df = pd.DataFrame(rows)
    if len(df) >= 5:
        rho, p = stats.spearmanr(df["ref_persistence"], df["match_persistence"])
    else:
        rho, p = np.nan, np.nan
    return df, rho, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--neighbor-values", type=int, nargs="+", required=True)
    ap.add_argument("--reference-neighbors", type=int, default=10)
    ap.add_argument("--minsize", type=int, default=4)
    ap.add_argument("--jaccard", type=float, default=0.75)
    ap.add_argument("--chi", type=int, default=5)
    ap.add_argument("--consensus-p", type=int, default=75)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    write_manifest(args.output_dir, args, inputs=[args.ensemble])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading ensemble ...")
    partitions, node_names = load_ensemble(args.ensemble)
    print(f"  {len(partitions)} partitions, {len(node_names)} nodes")

    neighbor_values = sorted(set(args.neighbor_values) | {args.reference_neighbors})
    results = {}
    dist_rows = []
    for nb in neighbor_values:
        print(f"\nRunning Phase 1 consensus at --neighbors {nb} ...")
        cluG_collapsed_w_len = run_phase1(partitions, node_names, args.minsize, args.jaccard,
                                           args.chi, args.consensus_p, nb)
        clusters, persist = clusters_from_consensus(cluG_collapsed_w_len, node_names)
        results[nb] = (clusters, persist)
        if persist:
            pvals = np.array(list(persist.values()))
            pmax = pvals.max()
            dist_rows.append({
                "neighbors": nb, "n_clusters": len(persist),
                "max_persistence": float(pmax),
                "frac_at_max": float((pvals == pmax).mean()),
                "n_distinct_persistence_values": int(len(set(pvals.tolist()))),
                "mean_persistence": float(pvals.mean()),
            })
        else:
            dist_rows.append({"neighbors": nb, "n_clusters": 0, "max_persistence": np.nan,
                               "frac_at_max": np.nan, "n_distinct_persistence_values": 0,
                               "mean_persistence": np.nan})
        print(f"  -> {len(persist)} consensus clusters")

    dist_df = pd.DataFrame(dist_rows).sort_values("neighbors")
    dist_df.to_csv(out / "neighbors_sweep_distribution_shape.csv", index=False)
    print("\nPersistence-distribution shape across --neighbors values:")
    print(dist_df.to_string(index=False))

    ref_clusters, ref_persist = results[args.reference_neighbors]
    concord_rows = []
    for nb in neighbor_values:
        if nb == args.reference_neighbors:
            continue
        other_clusters, other_persist = results[nb]
        match_df, rho, p = match_and_correlate(ref_clusters, ref_persist, other_clusters, other_persist)
        match_df.to_csv(out / f"neighbors_{nb}_vs_ref{args.reference_neighbors}_matches.csv", index=False)
        concord_rows.append({"neighbors": nb, "reference_neighbors": args.reference_neighbors,
                              "n_matched": len(match_df), "spearman_rho": rho, "p_value": p})
        print(f"  neighbors={nb} vs reference={args.reference_neighbors}: "
              f"matched {len(match_df)} clusters, rho={rho if pd.isna(rho) else round(rho,3)}")

    concord_df = pd.DataFrame(concord_rows)
    concord_df.to_csv(out / "neighbors_sweep_concordance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(dist_df["neighbors"], dist_df["n_distinct_persistence_values"], "o-", label="distinct persistence values")
    axes[0].plot(dist_df["neighbors"], dist_df["n_clusters"], "s--", label="n_clusters", alpha=0.6)
    axes[0].set_xlabel("--neighbors")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Distribution shape vs. --neighbors")
    axes[0].grid(alpha=0.3)

    axes[1].plot(concord_df["neighbors"], concord_df["spearman_rho"], "o-", color="crimson")
    axes[1].axhline(0.7, color="gray", linestyle="--", linewidth=1, label="rho=0.7 (rough stability bar)")
    axes[1].set_xlabel("--neighbors")
    axes[1].set_ylabel(f"Spearman rho vs. reference (neighbors={args.reference_neighbors})")
    axes[1].set_title("Cluster-identity concordance vs. --neighbors")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "neighbors_sweep_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved everything to {out}")
    print("\nHow to use this: pick the SMALLEST --neighbors where "
          "n_distinct_persistence_values has stopped changing much AND concordance vs. "
          "larger --neighbors values is still high (rho > ~0.7). Report that value and "
          "this sweep in your methods, rather than reporting whichever number 'looked "
          "like it worked' -- a reviewer who knows hidef internals will ask.")


if __name__ == "__main__":
    main()