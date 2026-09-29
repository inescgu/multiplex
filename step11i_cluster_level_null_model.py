#!/usr/bin/env python3
"""
step11i_cluster_level_null_model.py

FIX 3 (proper null) + FIX 6 (leave-one-out robustness) -- addresses the gap
you spotted yourself: step10/step10b's permutation tests validate the
disease MODULE (is it more connected / more functionally enriched than
degree-matched random seed neighborhoods?) but never re-run HiDeF on the
null modules. That means you currently have no evidence that the
persistence HIERARCHY itself -- broad high-persistence clusters nesting
disease-specific low-persistence ones -- is anything other than what you'd
get from clustering ANY RWR-expanded neighborhood in this network,
regardless of which genes you started from.

This script closes that gap directly:

  NULL A (structural, matches step10's Null 1 logic): degree-matched random
      seeds -> sparse RWR -> null module -> actually re-cluster the null
      module's own induced subgraph with the SAME windowed-HiDeF recipe
      (Leiden resolution sweep + windowed consensus, using hidef_finder's
      own Cluster/ClusterGraph/consensus -- not a reimplementation) -> record
      hierarchy-shape summary statistics -> repeat N times -> compare the
      TRUE module's own hierarchy-shape statistics against this null
      distribution.

  LEAVE-ONE-SEED-OUT (Fix 6): for each true seed gene, drop only that one
      seed, rebuild the module via RWR from the remaining seeds, re-cluster
      it the same way, and measure how much the resulting top-persistence
      ("broadest") cluster's gene membership shifts (Jaccard) relative to
      the full-seed-set result. If persistence rank is highly sensitive to
      any single seed, your hierarchy is fragile and "persistence" is partly
      measuring curation choices, not robust network structure.

NOTE ON COST: re-clustering (Leiden sweep x gammas x restarts, per
permutation) is much more expensive than the topology-only nulls in
step10/step10b. This script runs the sweep on the NULL MODULE'S OWN induced
subgraph (typically hundreds of nodes), not the full genome-scale network,
which is what makes repeating it O(100) times per run practical. Start with
--n-perms 30-50 to sanity check before committing to more.

Requires: leidenalg, python-igraph, hidef (for Cluster/ClusterGraph/consensus).

USAGE
-----
    python step11i_cluster_level_null_model.py \
        --reference-graph gene_reference_graph.pkl \
        --disease-module-dir 06_disease_module \
        --n-perms 50 \
        --min-resolution 0.01 --max-resolution 10 --n-resolutions 30 \
        --neighbors 3 \
        --output-dir 11i_cluster_level_null
"""
import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hidef.hidef_finder import Cluster, ClusterGraph, consensus


# ─────────────────────────────────────────────────────────────────────────
# Reference-graph RWR (self-contained; mirrors step10/step10b's sparse RWR)
# ─────────────────────────────────────────────────────────────────────────

def build_transition_matrix(G, nodelist):
    idx_of = {n: i for i, n in enumerate(nodelist)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, format="csr", dtype=np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_safe = np.where(deg == 0, 1.0, deg)
    D_inv = sparse.diags(1.0 / deg_safe)
    M = (A @ D_inv).tocsr()
    return M, idx_of, deg


def rwr_power_iteration(M, p0, restart, max_iter=200, tol=1e-9):
    p = p0.copy()
    for _ in range(max_iter):
        p_new = (1 - restart) * (M @ p) + restart * p0
        if np.abs(p_new - p).sum() < tol:
            return p_new
        p = p_new
    return p


def run_rwr_module(G_ref, nodelist, idx_of, M, deg, seed_nodes, restart, k=200, scaling=0):
    n = len(nodelist)
    seeds_in_graph = [g for g in seed_nodes if g in idx_of]
    if not seeds_in_graph:
        return None
    p0 = np.zeros(n)
    for node in seeds_in_graph:
        i = idx_of[node]
        p0[i] = np.sqrt(deg[i]) if (scaling == 1 and deg[i] > 0) else 1.0
    p0 = p0 / p0.sum()
    pinf = rwr_power_iteration(M, p0, restart)

    ranked_idx = np.argsort(pinf)[::-1]
    seeds_set = set(seeds_in_graph)
    ranked_nodes = [nodelist[i] for i in ranked_idx if nodelist[i] not in seeds_set]

    module = list(seeds_set) + ranked_nodes[:k]
    module_set = set(module)
    extra_idx = k
    while extra_idx < len(ranked_nodes):
        sg = G_ref.subgraph(module)
        if nx.is_connected(sg):
            break
        batch = ranked_nodes[extra_idx: extra_idx + 10]
        if not batch:
            break
        for g in batch:
            if g not in module_set:
                module.append(g)
                module_set.add(g)
        extra_idx += 10
    return module


def build_degree_bins(G, bin_size=5):
    bins = defaultdict(list)
    for node, deg in G.degree():
        bins[(deg // bin_size) * bin_size].append(node)
    return bins


def sample_degree_matched_seeds(true_seeds, G, degree_bins, rng, bin_size=5):
    sampled = set()
    for seed in true_seeds:
        deg = G.degree(seed)
        bin_key = (deg // bin_size) * bin_size
        candidates = [n for n in degree_bins[bin_key] if n not in sampled and n not in true_seeds]
        if not candidates:
            for offset in [bin_size, -bin_size, 2 * bin_size, -2 * bin_size]:
                candidates = [n for n in degree_bins.get(bin_key + offset, [])
                              if n not in sampled and n not in true_seeds]
                if candidates:
                    break
        if candidates:
            sampled.add(rng.choice(candidates))
    return list(sampled)


# ─────────────────────────────────────────────────────────────────────────
# Re-clustering a module's own induced subgraph with windowed HiDeF
# (reuses hidef_finder's own Cluster/ClusterGraph/consensus, same as
# step11c_phase1_consensus_windowed.py -- only the resolution-graph
# construction is reimplemented here, condensed to a single-graph,
# single-omega case)
# ─────────────────────────────────────────────────────────────────────────

def nx_to_igraph(G_sub):
    nodes = sorted(G_sub.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.add_edges([(idx[u], idx[v]) for u, v in G_sub.edges()])
    g.vs["name"] = nodes
    return g, nodes


def leiden_sweep(g_ig, gammas, n_restarts, base_seed, partition_type="RB"):
    PartitionClass = {"RB": la.RBConfigurationVertexPartition,
                       "CPM": la.CPMVertexPartition}[partition_type]
    partitions = []
    run_i = 0
    for gamma in gammas:
        for r in range(n_restarts):
            run_i += 1
            part = la.find_partition(g_ig, PartitionClass, resolution_parameter=gamma,
                                      seed=base_seed + run_i, n_iterations=-1)
            partitions.append({"membership": list(part.membership), "gamma": float(gamma),
                                "restart": r})
    return partitions


def membership_to_csr(membership, minsize=4):
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
    return sparse.coo_matrix((data, (row, col)), shape=(len(keep), len(membership))).tocsr()


def windowed_consensus(partitions, n_nodes, minsize, jaccard_thr, chi, consensus_p, neighbors):
    """Single-omega version of step11c's windowed resolution graph: sort
    partitions by (gamma, restart), link each to its `neighbors` nearest
    neighbors in that order, feed hidef_finder's own consensus() the result."""
    cluG = ClusterGraph()
    cluG.graph["sim_threshold"] = jaccard_thr
    cluG.graph["num_leaves"] = n_nodes

    ordered = sorted(range(len(partitions)),
                      key=lambda i: (partitions[i]["gamma"], partitions[i]["restart"]))
    resolution_graph = nx.Graph()
    resname_of = {}
    for rank, orig_i in enumerate(ordered):
        resname = f"{rank / 100.0:.4f}"
        resname_of[orig_i] = resname
        resolution_graph.add_node(resname, padded=True)
    names_in_order = [resname_of[i] for i in ordered]
    for a in range(len(names_in_order)):
        for b in range(a + 1, min(a + 1 + neighbors, len(names_in_order))):
            resolution_graph.add_edge(names_in_order[a], names_in_order[b])

    for orig_i in ordered:
        p = partitions[orig_i]
        resname = resname_of[orig_i]
        mat = membership_to_csr(p["membership"], minsize=minsize)
        resolution_graph.nodes[resname]["matrix"] = mat
        cluG.add_clusters(resolution_graph, rank_of_resname(resname))

    cluG_collapsed_w_len = consensus(cluG, k=chi, f=1.0, p=consensus_p)
    return cluG_collapsed_w_len


def rank_of_resname(resname):
    return float(resname)


def summarize_hierarchy(cluG_collapsed_w_len, n_nodes_in_module, n_partitions):
    """Compact set of "hierarchy shape" statistics comparable across
    modules of different sizes. This is what gets compared true-vs-null."""
    if not cluG_collapsed_w_len:
        return {"n_clusters": 0, "max_persistence": 0, "frac_at_max_persistence": 0.0,
                "n_distinct_persistence_values": 0, "top_cluster_frac_of_module": 0.0,
                "mean_persistence": 0.0}
    sizes = np.array([int(np.sum(c[0])) for c in cluG_collapsed_w_len])
    persistences = np.array([c[1] for c in cluG_collapsed_w_len])
    pmax = persistences.max()
    top_idx = np.argmax(persistences)
    return {
        "n_clusters": len(cluG_collapsed_w_len),
        "max_persistence": float(pmax),
        "frac_at_max_persistence": float((persistences == pmax).mean()),
        "n_distinct_persistence_values": int(len(set(persistences.tolist()))),
        "top_cluster_frac_of_module": float(sizes[top_idx] / n_nodes_in_module) if n_nodes_in_module else 0.0,
        "mean_persistence": float(persistences.mean()),
    }


def recluster_and_summarize(module_nodes, G_ref, gammas, n_restarts, neighbors,
                             minsize, jaccard_thr, chi, consensus_p, base_seed,
                             partition_type="RB"):
    sub = G_ref.subgraph(module_nodes).copy()
    if sub.number_of_nodes() < minsize * 2:
        return None, None
    g_ig, nodes = nx_to_igraph(sub)
    partitions = leiden_sweep(g_ig, gammas, n_restarts, base_seed, partition_type)
    cluG_collapsed_w_len = windowed_consensus(
        partitions, len(nodes), minsize, jaccard_thr, chi, consensus_p, neighbors)
    summary = summarize_hierarchy(cluG_collapsed_w_len, len(nodes), len(partitions))
    top_cluster_genes = None
    if cluG_collapsed_w_len:
        top_idx = int(np.argmax([c[1] for c in cluG_collapsed_w_len]))
        mask = cluG_collapsed_w_len[top_idx][0].astype(bool)
        top_cluster_genes = {nodes[i] for i in np.where(mask)[0]}
    return summary, top_cluster_genes


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-graph", required=True)
    ap.add_argument("--disease-module-dir", required=True)
    ap.add_argument("--restart", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--n-perms", type=int, default=50)
    ap.add_argument("--bin-size", type=int, default=5)
    ap.add_argument("--min-resolution", type=float, default=0.01)
    ap.add_argument("--max-resolution", type=float, default=10.0)
    ap.add_argument("--n-resolutions", type=int, default=30)
    ap.add_argument("--n-restarts", type=int, default=1)
    ap.add_argument("--neighbors", type=int, default=3,
                     help="MUST match whatever --neighbors you used for the real "
                          "heterogeneous_hidef_faithful*.nodes result, or this isn't "
                          "an apples-to-apples comparison.")
    ap.add_argument("--minsize", type=int, default=4)
    ap.add_argument("--jaccard", type=float, default=0.75)
    ap.add_argument("--chi", type=int, default=5)
    ap.add_argument("--consensus-p", type=int, default=75)
    ap.add_argument("--partition-type", choices=["RB", "CPM"], default="RB")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-leave-one-out", action="store_true",
                     help="Also run Fix 6: drop each true seed one at a time, rebuild + "
                          "recluster, measure top-cluster Jaccard drift.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gammas = np.geomspace(args.min_resolution, args.max_resolution, args.n_resolutions)

    print("Loading reference graph ...")
    with open(args.reference_graph, "rb") as f:
        G_ref = pickle.load(f)
    G_ref = nx.relabel_nodes(G_ref, {n: str(n).strip() for n in G_ref.nodes()})
    nodelist = sorted(G_ref.nodes())
    M, idx_of, deg = build_transition_matrix(G_ref, nodelist)
    print(f"  {G_ref.number_of_nodes():,} nodes, {G_ref.number_of_edges():,} edges")

    dm_dir = Path(args.disease_module_dir)
    module_df = pd.read_csv(dm_dir / "disease_module.csv")
    module_df["node"] = module_df["node"].astype(str).str.strip()
    true_module = module_df["node"].tolist()
    if "is_excluded_seed" in module_df.columns:
        anchor_mask = module_df["is_seed"] & (~module_df["is_excluded_seed"].astype(bool))
    else:
        anchor_mask = module_df["is_seed"]
    true_seeds = [g for g in module_df.loc[anchor_mask, "node"].tolist() if g in idx_of]
    print(f"True module: {len(true_module)} nodes, {len(true_seeds)} anchoring seeds")

    # ── TRUE module: re-cluster its own induced subgraph ───────────────────
    print("\nRe-clustering the TRUE module's own induced subgraph ...")
    true_summary, true_top_genes = recluster_and_summarize(
        true_module, G_ref, gammas, args.n_restarts, args.neighbors,
        args.minsize, args.jaccard, args.chi, args.consensus_p,
        base_seed=args.seed, partition_type=args.partition_type)
    if true_summary is None:
        raise SystemExit("True module too small to re-cluster with current --minsize.")
    print("  True module hierarchy summary:", true_summary)

    # ── NULL A: degree-matched random seeds -> RWR -> module -> recluster ──
    print(f"\nRunning {args.n_perms} null permutations (structural null, Fix 3) ...")
    degree_bins = build_degree_bins(G_ref, bin_size=args.bin_size)
    null_summaries = []
    t0 = time.time()
    for i in range(args.n_perms):
        rand_seeds = sample_degree_matched_seeds(true_seeds, G_ref, degree_bins, rng,
                                                  bin_size=args.bin_size)
        if len(rand_seeds) < max(1, len(true_seeds) // 2):
            continue
        rand_module = run_rwr_module(G_ref, nodelist, idx_of, M, deg, rand_seeds,
                                      args.restart, k=args.k)
        if rand_module is None:
            continue
        summ, _ = recluster_and_summarize(
            rand_module, G_ref, gammas, args.n_restarts, args.neighbors,
            args.minsize, args.jaccard, args.chi, args.consensus_p,
            base_seed=args.seed + 1000 + i, partition_type=args.partition_type)
        if summ is not None:
            null_summaries.append(summ)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{args.n_perms}] elapsed {elapsed:.0f}s")

    print(f"\nCompleted {len(null_summaries)} valid null permutations")
    null_df = pd.DataFrame(null_summaries)
    null_df.to_csv(out / "null_hierarchy_summaries.csv", index=False)

    # ── p-values / z-scores for each hierarchy-shape statistic ─────────────
    higher_is_better = {"max_persistence", "n_distinct_persistence_values", "mean_persistence"}
    rows = []
    for metric, true_val in true_summary.items():
        if metric not in null_df.columns or null_df.empty:
            continue
        null_vals = null_df[metric].to_numpy()
        mu, sigma = null_vals.mean(), null_vals.std()
        z = (true_val - mu) / sigma if sigma > 0 else np.nan
        if metric in higher_is_better:
            p = float(np.mean(null_vals >= true_val))
        else:
            p = float(np.mean(null_vals <= true_val))
        rows.append({"metric": metric, "true_value": true_val, "null_mean": mu,
                      "null_std": sigma, "z_score": z, "p_value": p})
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out / "step11i_null_vs_true_results.csv", index=False)
    print("\n" + "=" * 70)
    print("FIX 3 RESULTS: true module's hierarchy shape vs. degree-matched null")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("\nInterpretation: if max_persistence / n_distinct_persistence_values / "
          "top_cluster_frac_of_module are NOT significantly different from the null "
          "(p not small, |z| not large), then a random RWR neighborhood in this network "
          "produces a similarly-shaped persistence hierarchy by default -- your broad/"
          "specific structure would not be evidence of pain-specific biology, just of "
          "how Leiden+RWR behave on this graph's topology.")

    # ── plots ──
    fig, axes = plt.subplots(1, len(true_summary), figsize=(4.5 * len(true_summary), 4))
    for ax, (metric, true_val) in zip(np.atleast_1d(axes), true_summary.items()):
        if metric not in null_df.columns or null_df.empty:
            continue
        ax.hist(null_df[metric], bins=20, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(true_val, color="crimson", linewidth=2, label="true module")
        ax.set_title(metric)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out / "step11i_null_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── FIX 6: leave-one-seed-out ────────────────────────────────────────
    if args.run_leave_one_out:
        print(f"\nRunning leave-one-seed-out (Fix 6) across {len(true_seeds)} seeds ...")
        loo_rows = []
        for j, dropped_seed in enumerate(true_seeds):
            reduced_seeds = [s for s in true_seeds if s != dropped_seed]
            reduced_module = run_rwr_module(G_ref, nodelist, idx_of, M, deg, reduced_seeds,
                                             args.restart, k=args.k)
            if reduced_module is None:
                continue
            summ, top_genes = recluster_and_summarize(
                reduced_module, G_ref, gammas, args.n_restarts, args.neighbors,
                args.minsize, args.jaccard, args.chi, args.consensus_p,
                base_seed=args.seed + 5000 + j, partition_type=args.partition_type)
            if summ is None:
                continue
            jac = jaccard(true_top_genes or set(), top_genes or set())
            loo_rows.append({"dropped_seed": dropped_seed, "top_cluster_jaccard_vs_full": jac,
                              **{f"loo_{k}": v for k, v in summ.items()}})
            if (j + 1) % 5 == 0:
                print(f"  [{j+1}/{len(true_seeds)}] dropped {dropped_seed}: "
                      f"top-cluster Jaccard vs full-seed result = {jac:.3f}")

        loo_df = pd.DataFrame(loo_rows).sort_values("top_cluster_jaccard_vs_full")
        loo_df.to_csv(out / "step11i_leave_one_out.csv", index=False)
        print(f"\nSaved leave-one-out results -> {out / 'step11i_leave_one_out.csv'}")
        print(f"Top-cluster Jaccard vs. full-seed result: "
              f"mean={loo_df['top_cluster_jaccard_vs_full'].mean():.3f}, "
              f"min={loo_df['top_cluster_jaccard_vs_full'].min():.3f} "
              f"(seed: {loo_df.iloc[0]['dropped_seed']})")
        print("Interpretation: a low Jaccard for any single dropped seed means your "
              "'broadest' cluster's identity hinges on that one gene being in the seed "
              "list -- worth flagging explicitly if it's a recently-added or "
              "less-confident seed.")

        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(loo_df))))
        ax.barh(loo_df["dropped_seed"], loo_df["top_cluster_jaccard_vs_full"], color="#4C72B0")
        ax.set_xlabel("Top-cluster Jaccard vs. full-seed-set result")
        ax.set_title("Leave-one-seed-out sensitivity of the broadest cluster")
        plt.tight_layout()
        fig.savefig(out / "step11i_leave_one_out.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nAll outputs saved to: {out}")


if __name__ == "__main__":
    main()