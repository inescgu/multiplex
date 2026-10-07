#!/usr/bin/env python3
"""
step11i_cluster_level_null_model.py  (parallel / resumable version)

Cluster-level null model (Fix 3) + leave-one-seed-out robustness (Fix 6),
split into independent STAGES so they can run as SLURM job arrays:

  --stage true   Re-cluster the true module (as read from disease_module.csv)
                 AND a "rebuilt" true module (same seeds -> same RWR/cap code
                 path as the nulls). Writes true_summary.json. Run once.
  --stage null   Run --n-perms degree-matched null permutations for task
                 --task-id. Global perm index = task_id * n_perms + i, and each
                 perm's RNG depends only on (seed, perm_idx), so results do not
                 depend on how perms are split across tasks. Appends one row
                 per perm to null_perms.csv (resumable).
  --stage loo    Leave-one-seed-out for seeds[task_id::n_tasks]. Needs --true-dir.

Aggregation (p-values, plots, budget suggestion) is in step11i_aggregate.py.

CHANGES vs. the single-process version
--------------------------------------
1. run_rwr_module(): connectivity-extension loop is capped (--cap-factor x true
   module size). If still disconnected at the cap, the module becomes its
   largest connected component (LCC). Perms whose LCC is < --min-lcc-factor x
   true size are skipped and logged (status column), not silently dropped.
2. CPM gamma grid is calibrated PER MODULE (--gamma-mode calibrated): probe a
   coarse gamma ladder, find where the module splits from 1 community into >=2
   (lo) and where fewer than half the nodes sit in communities >= minsize (hi),
   then geomspace(lo, hi, n_resolutions). Every module (true or null) is swept
   from its own root to its own leaves with the same number of partitions, so
   persistence values are comparable. --gamma-mode fixed restores old behaviour.
3. "Top" cluster = max persistence, ties broken by size (argmax on ties was
   arbitrary before, which would make LOO Jaccards noisy).
4. Per-perm CSV append, timestamps, flushed logs, RSS/peak-RSS, ETA, optional
   per-perm timeout, deterministic per-perm RNG.
5. LOO is compared to a REBUILT full-seed module (same code path), not only to
   the module read from file, so RWR-parameter differences are not mistaken
   for seed sensitivity. Both Jaccards are reported.
"""
import argparse
import gc
import json
import math
import os
import pickle
import resource
import signal
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from provenance import write_manifest
from hidef.hidef_finder import Cluster, ClusterGraph, consensus  # noqa: F401

SUMMARY_KEYS = ["n_clusters", "max_persistence", "frac_at_max_persistence",
                "n_distinct_persistence_values", "top_cluster_frac_of_module",
                "mean_persistence"]


# ─────────────────────────────────────────────────────────────────────────
# Logging / resource helpers
# ─────────────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb():
    """(current RSS, peak RSS) in GiB."""
    cur = float("nan")
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    cur = int(line.split()[1]) / 1024 ** 2  # kB -> GiB
                    break
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2  # kB -> GiB
    return cur, peak


class PermTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise PermTimeout()


signal.signal(signal.SIGALRM, _alarm_handler)


def append_row(path, row, cols):
    df = pd.DataFrame([row]).reindex(columns=cols)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def to_jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return sorted(o)
    return str(o)


# ─────────────────────────────────────────────────────────────────────────
# Reference-graph RWR
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


def run_rwr_module(G_ref, nodelist, idx_of, M, deg, seed_nodes, restart, k=200,
                   scaling=0, max_size=None):
    """RWR -> seeds + top-k -> extend 10 nodes at a time until connected.

    NEW: the extension stops once len(module) >= max_size (if given). If the
    module is still disconnected, it is reduced to its largest connected
    component. Returns a dict (or None if no seed is in the graph).
    """
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
    cap = max(max_size, len(module)) if max_size else None
    extra_idx = k
    connected, capped = False, False
    while True:
        sg = G_ref.subgraph(module)
        if nx.is_connected(sg):
            connected = True
            break
        if extra_idx >= len(ranked_nodes):
            break
        if cap is not None and len(module) >= cap:
            capped = True
            break
        for g in ranked_nodes[extra_idx: extra_idx + 10]:
            if g not in module_set:
                module.append(g)
                module_set.add(g)
        extra_idx += 10

    raw_size = len(module)
    if connected:
        final = module
    else:
        lcc = max(nx.connected_components(G_ref.subgraph(module)), key=len)
        final = [m for m in module if m in lcc]
    return {"module": final, "raw_size": raw_size, "connected_raw": connected,
            "capped": capped, "n_seeds_in_final": len(seeds_set & set(final))}


def build_degree_bins(G, bin_size=5):
    bins = defaultdict(list)
    for node, deg in G.degree():
        bins[(deg // bin_size) * bin_size].append(node)
    return bins


def sample_degree_matched_seeds(true_seeds, G, degree_bins, rng, bin_size=5):
    true_set = set(true_seeds)
    sampled = set()
    for seed in true_seeds:
        deg = G.degree(seed)
        bin_key = (deg // bin_size) * bin_size
        candidates = [n for n in degree_bins[bin_key] if n not in sampled and n not in true_set]
        if not candidates:
            for offset in [bin_size, -bin_size, 2 * bin_size, -2 * bin_size]:
                candidates = [n for n in degree_bins.get(bin_key + offset, [])
                              if n not in sampled and n not in true_set]
                if candidates:
                    break
        if candidates:
            sampled.add(str(rng.choice(candidates)))
    return list(sampled)


# ─────────────────────────────────────────────────────────────────────────
# Windowed HiDeF re-clustering of a module's own induced subgraph
# ─────────────────────────────────────────────────────────────────────────

def nx_to_igraph(G_sub):
    nodes = sorted(G_sub.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.add_edges([(idx[u], idx[v]) for u, v in G_sub.edges()])
    g.vs["name"] = nodes
    return g, nodes


PARTITION_CLASS = {"RB": la.RBConfigurationVertexPartition,
                   "CPM": la.CPMVertexPartition}


def leiden_sweep(g_ig, gammas, n_restarts, base_seed, partition_type="RB"):
    PartitionClass = PARTITION_CLASS[partition_type]
    partitions = []
    run_i = 0
    for gamma in gammas:
        for r in range(n_restarts):
            run_i += 1
            part = la.find_partition(g_ig, PartitionClass, resolution_parameter=float(gamma),
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


def rank_of_resname(resname):
    return float(resname)


def windowed_consensus(partitions, n_nodes, minsize, jaccard_thr, chi, consensus_p, neighbors):
    """Single-omega windowed resolution graph -> hidef_finder.consensus()."""
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
    gammas_sorted = sorted({partitions[i]["gamma"] for i in ordered})
    grank = {gm: k for k, gm in enumerate(gammas_sorted)}
    ranks = [grank[partitions[i]["gamma"]] for i in ordered]
    for a in range(len(names_in_order)):
        for b in range(a + 1, len(names_in_order)):
            if ranks[b] - ranks[a] > neighbors:
                break
            resolution_graph.add_edge(names_in_order[a], names_in_order[b])

    for orig_i in ordered:
        p = partitions[orig_i]
        resname = resname_of[orig_i]
        mat = membership_to_csr(p["membership"], minsize=minsize)
        resolution_graph.nodes[resname]["matrix"] = mat
        cluG.add_clusters(resolution_graph, rank_of_resname(resname))

    return consensus(cluG, k=chi, f=1.0, p=consensus_p)


def top_cluster_index(sizes, persistences):
    """Max persistence; ties broken by larger size (deterministic)."""
    return int(np.lexsort((sizes, persistences))[-1])


def summarize_hierarchy(cluG_collapsed_w_len, n_nodes_in_module):
    if not cluG_collapsed_w_len:
        return {k: 0 for k in SUMMARY_KEYS}
    sizes = np.array([int(np.sum(c[0])) for c in cluG_collapsed_w_len])
    persistences = np.array([c[1] for c in cluG_collapsed_w_len])
    pmax = persistences.max()
    top_idx = top_cluster_index(sizes, persistences)
    return {
        "n_clusters": len(cluG_collapsed_w_len),
        "max_persistence": float(pmax),
        "frac_at_max_persistence": float((persistences == pmax).mean()),
        "n_distinct_persistence_values": int(len(set(persistences.tolist()))),
        "top_cluster_frac_of_module": float(sizes[top_idx] / n_nodes_in_module) if n_nodes_in_module else 0.0,
        "mean_persistence": float(persistences.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────
# Per-module gamma grid
# ─────────────────────────────────────────────────────────────────────────

def module_gamma_grid(g_ig, n, max_res, lo_mult=0.1):
    """Density-based fallback (old heuristic)."""
    nv = g_ig.vcount()
    dens = 2 * g_ig.ecount() / (nv * (nv - 1)) if nv > 1 else 0.01
    lo = max(dens * lo_mult, 1e-4)
    hi = min(max_res, 0.9)
    if lo >= hi:
        lo = hi / 100
    return np.geomspace(lo, hi, n)


def calibrate_gamma_grid(g_ig, n_grid, minsize, partition_type, probe_lo, probe_hi,
                         seed, n_probe=30, coverage_floor=0.5):
    """Find the gamma window over which THIS module goes from root to leaves.

    lo  = last probe gamma before the module first splits into >=2 communities
          of size >= minsize (the "root" end).
    hi  = first probe gamma at which < coverage_floor of nodes sit in
          communities of size >= minsize (the "leaf" end).
    Returns (grid or None, probe_rows). None => no split found (use fallback).
    """
    nv = g_ig.vcount()
    PartitionClass = PARTITION_CLASS[partition_type]
    probes = np.geomspace(probe_lo, probe_hi, n_probe)
    rows = []
    for j, gm in enumerate(probes):
        part = la.find_partition(g_ig, PartitionClass, resolution_parameter=float(gm),
                                 seed=int(seed + j), n_iterations=-1)
        sizes = np.bincount(part.membership)
        big = sizes[sizes >= minsize]
        rows.append({"gamma": float(gm), "n_comm_total": int(len(sizes)),
                     "n_comm_ge_minsize": int(len(big)),
                     "coverage": float(big.sum() / nv)})
    first_multi = next((j for j, r in enumerate(rows) if r["n_comm_ge_minsize"] >= 2), None)
    if first_multi is None:
        return None, rows
    lo_idx = max(first_multi - 1, 0)
    hi_idx = next((j for j in range(first_multi, len(rows))
                   if rows[j]["coverage"] < coverage_floor), len(rows) - 1)
    hi_idx = min(max(hi_idx, lo_idx + 1), len(rows) - 1)
    return np.geomspace(probes[lo_idx], probes[hi_idx], n_grid), rows


def get_gamma_grid(g_ig, args, seed):
    fixed = np.geomspace(args.min_resolution, args.max_resolution, args.n_resolutions)
    if args.gamma_mode == "fixed":
        return fixed, [], "fixed"
    if args.partition_type == "CPM":
        plo, phi = 1e-4, min(args.max_resolution, 1.0)
    else:
        plo, phi = args.min_resolution / 10.0, args.max_resolution * 3.0
    grid, probe_rows = calibrate_gamma_grid(g_ig, args.n_resolutions, args.minsize,
                                            args.partition_type, plo, phi, seed)
    if grid is None:
        if args.partition_type == "CPM":
            return module_gamma_grid(g_ig, args.n_resolutions, args.max_resolution), \
                probe_rows, "fallback_density"
        return fixed, probe_rows, "fallback_fixed"
    return grid, probe_rows, "calibrated"


# ─────────────────────────────────────────────────────────────────────────
# Re-cluster one module
# ─────────────────────────────────────────────────────────────────────────

def recluster_module(module_nodes, G_ref, args, base_seed):
    sub = G_ref.subgraph(module_nodes).copy()
    n_nodes = sub.number_of_nodes()
    if n_nodes < args.minsize * 2:
        return None
    g_ig, nodes = nx_to_igraph(sub)

    t0 = time.time()
    gammas, probe_rows, gamma_status = get_gamma_grid(g_ig, args, seed=base_seed + 900)
    t1 = time.time()
    partitions = leiden_sweep(g_ig, gammas, args.n_restarts, base_seed, args.partition_type)
    t2 = time.time()
    cluG = windowed_consensus(partitions, n_nodes, args.minsize, args.jaccard,
                              args.chi, args.consensus_p, args.neighbors)
    t3 = time.time()

    summary = summarize_hierarchy(cluG, n_nodes)
    top_genes = None
    if cluG:
        sizes = np.array([int(np.sum(c[0])) for c in cluG])
        pers = np.array([c[1] for c in cluG])
        mask = cluG[top_cluster_index(sizes, pers)][0].astype(bool)
        top_genes = {nodes[i] for i in np.where(mask)[0]}
    return {"summary": summary, "top_genes": top_genes, "n_nodes": n_nodes,
            "n_edges": sub.number_of_edges(), "gamma_lo": float(gammas[0]),
            "gamma_hi": float(gammas[-1]), "gamma_status": gamma_status,
            "n_partitions": len(partitions), "probe_rows": probe_rows,
            "t_calibrate_s": t1 - t0, "t_leiden_s": t2 - t1, "t_consensus_s": t3 - t2}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def res_to_json(res, module_nodes=None):
    d = {k: v for k, v in res.items() if k not in ("probe_rows", "top_genes")}
    d["top_genes"] = sorted(res["top_genes"]) if res["top_genes"] else []
    if module_nodes is not None:
        d["module_nodes"] = sorted(module_nodes)
    return d


# ─────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────

def run_true_stage(args, G_ref, nodelist, idx_of, M, deg, true_seeds, true_module,
                   max_size, out):
    log("TRUE stage: re-clustering module read from disease_module.csv ...")
    res_file = recluster_module(true_module, G_ref, args, base_seed=args.seed)
    if res_file is None:
        raise SystemExit("True module too small to re-cluster with current --minsize.")
    log(f"  file module: {res_file['n_nodes']} nodes, {res_file['n_edges']} edges, "
        f"gamma [{res_file['gamma_lo']:.4g}, {res_file['gamma_hi']:.4g}] ({res_file['gamma_status']}), "
        f"{res_file['n_partitions']} partitions")
    log(f"  summary: {res_file['summary']}")
    log(f"  timings: calibrate {res_file['t_calibrate_s']:.1f}s, leiden {res_file['t_leiden_s']:.1f}s, "
        f"consensus {res_file['t_consensus_s']:.1f}s | RSS {rss_gb()[0]:.2f} GiB, peak {rss_gb()[1]:.2f} GiB")
    pd.DataFrame(res_file["probe_rows"]).to_csv(out / "gamma_probe_curve_file.csv", index=False)

    log("TRUE stage: rebuilding module from the same seeds with this script's RWR/cap code ...")
    rb = run_rwr_module(G_ref, nodelist, idx_of, M, deg, true_seeds, args.restart,
                        k=args.k, max_size=max_size)
    out_json = {"n_true_module": len(true_module), "n_true_seeds": len(true_seeds),
                "seeds": true_seeds, "max_size_cap": max_size}
    out_json["file"] = res_to_json(res_file, true_module)
    if rb is None:
        log("  WARNING: rebuild failed (no seeds in graph)")
    else:
        jac_mod = jaccard(true_module, rb["module"])
        log(f"  rebuilt module: raw {rb['raw_size']} -> final {len(rb['module'])} nodes "
            f"(connected_raw={rb['connected_raw']}, capped={rb['capped']}); "
            f"Jaccard with file module = {jac_mod:.3f}")
        if jac_mod < 0.8:
            log("  WARNING: rebuilt module differs a lot from the file module. Check that "
                "--restart / --k match what step06 used; nulls are built with THIS code path.")
        res_rb = recluster_module(rb["module"], G_ref, args, base_seed=args.seed + 1)
        if res_rb is not None:
            out_json["rebuilt"] = res_to_json(res_rb, rb["module"])
            out_json["rebuilt"].update({"module_size_raw": rb["raw_size"],
                                        "connected_raw": rb["connected_raw"],
                                        "capped": rb["capped"],
                                        "jaccard_with_file_module": jac_mod})
            pd.DataFrame(res_rb["probe_rows"]).to_csv(out / "gamma_probe_curve_rebuilt.csv", index=False)
            log(f"  rebuilt summary: {res_rb['summary']}")
    with open(out / "true_summary.json", "w") as f:
        json.dump(out_json, f, indent=1, default=to_jsonable)
    log(f"Wrote {out / 'true_summary.json'}")


NULL_COLS = (["perm_idx", "task_id", "status", "n_seeds_sampled", "module_size_raw",
              "module_size_final", "connected_raw", "capped", "n_nodes", "n_edges",
              "gamma_lo", "gamma_hi", "gamma_status", "n_partitions",
              "t_rwr_s", "t_calibrate_s", "t_leiden_s", "t_consensus_s", "t_total_s",
              "rss_gb", "peak_rss_gb"] + SUMMARY_KEYS)


def run_null_stage(args, G_ref, nodelist, idx_of, M, deg, true_seeds, true_module,
                   max_size, out):
    csv_path = out / "null_perms.csv"
    done = set()
    if csv_path.exists() and not args.no_resume:
        done = set(pd.read_csv(csv_path)["perm_idx"].astype(int))
        log(f"Resuming: {len(done)} perms already in {csv_path.name}")
    min_lcc = int(args.min_lcc_factor * len(true_module))
    degree_bins = build_degree_bins(G_ref, bin_size=args.bin_size)
    log(f"NULL stage task {args.task_id}/{args.n_tasks}: {args.n_perms} perms "
        f"(global idx {args.task_id * args.n_perms}..{(args.task_id + 1) * args.n_perms - 1}); "
        f"module size cap {max_size}, min LCC {min_lcc}")

    t_start = time.time()
    n_done_now, t_sum = 0, 0.0
    for i in range(args.n_perms):
        perm_idx = args.task_id * args.n_perms + i
        if perm_idx in done:
            continue
        row = {"perm_idx": perm_idx, "task_id": args.task_id, "status": "error"}
        t_perm = time.time()
        res = None
        try:
            if args.perm_timeout > 0:
                signal.alarm(int(args.perm_timeout))
            rng = np.random.default_rng([args.seed, perm_idx])
            rand_seeds = sample_degree_matched_seeds(true_seeds, G_ref, degree_bins, rng,
                                                     bin_size=args.bin_size)
            row["n_seeds_sampled"] = len(rand_seeds)
            if len(rand_seeds) < max(1, len(true_seeds) // 2):
                row["status"] = "skip_too_few_seeds"
            else:
                t = time.time()
                rm = run_rwr_module(G_ref, nodelist, idx_of, M, deg, rand_seeds,
                                    args.restart, k=args.k, max_size=max_size)
                row["t_rwr_s"] = time.time() - t
                if rm is None:
                    row["status"] = "skip_no_module"
                else:
                    row.update(module_size_raw=rm["raw_size"], connected_raw=rm["connected_raw"],
                               capped=rm["capped"], module_size_final=len(rm["module"]))
                    if len(rm["module"]) < min_lcc:
                        row["status"] = "skip_small_lcc"
                    else:
                        res = recluster_module(rm["module"], G_ref, args,
                                               base_seed=args.seed + 1_000_000 + perm_idx * 1000)
                        if res is None:
                            row["status"] = "skip_too_small"
                        else:
                            row.update(res["summary"])
                            row.update({k: res[k] for k in
                                        ("n_nodes", "n_edges", "gamma_lo", "gamma_hi", "gamma_status",
                                         "n_partitions", "t_calibrate_s", "t_leiden_s", "t_consensus_s")})
                            row["status"] = "ok"
        except PermTimeout:
            row["status"] = "timeout"
        except MemoryError:
            row["status"] = "memoryerror"
        finally:
            signal.alarm(0)

        row["t_total_s"] = time.time() - t_perm
        row["rss_gb"], row["peak_rss_gb"] = rss_gb()
        append_row(csv_path, row, NULL_COLS)

        n_done_now += 1
        t_sum += row["t_total_s"]
        remaining = sum(1 for j in range(i + 1, args.n_perms)
                        if args.task_id * args.n_perms + j not in done)
        eta = (t_sum / n_done_now) * remaining
        if row["status"] == "ok":
            log(f"perm {perm_idx} [{i + 1}/{args.n_perms}] ok | module raw {row['module_size_raw']}"
                f"->{row['module_size_final']} (conn_raw={row['connected_raw']}, capped={row['capped']}) | "
                f"clusters {row['n_clusters']}, maxpers {row['max_persistence']:.3g} | "
                f"gamma [{row['gamma_lo']:.3g},{row['gamma_hi']:.3g}] {row['gamma_status']} | "
                f"{row['t_total_s']:.0f}s (rwr {row.get('t_rwr_s', 0):.1f}, cal {row['t_calibrate_s']:.1f}, "
                f"leiden {row['t_leiden_s']:.1f}, cons {row['t_consensus_s']:.1f}) | "
                f"RSS {row['rss_gb']:.2f} GiB, peak {row['peak_rss_gb']:.2f} GiB | ETA {eta / 60:.1f} min")
        else:
            log(f"perm {perm_idx} [{i + 1}/{args.n_perms}] {row['status']} | module raw "
                f"{row.get('module_size_raw')}->{row.get('module_size_final')} | "
                f"{row['t_total_s']:.0f}s | peak {row['peak_rss_gb']:.2f} GiB")
        del res
        gc.collect()

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        log("Status counts: " + ", ".join(f"{k}={v}" for k, v in df["status"].value_counts().items()))
    log(f"NULL task {args.task_id} finished in {(time.time() - t_start) / 60:.1f} min")


LOO_COLS = (["dropped_seed", "seed_idx", "status", "module_size_raw", "module_size_final",
             "capped", "module_jaccard_vs_rebuilt", "top_cluster_jaccard_vs_rebuilt",
             "top_cluster_jaccard_vs_file", "n_nodes", "gamma_lo", "gamma_hi",
             "t_total_s", "rss_gb", "peak_rss_gb"] + SUMMARY_KEYS)


def run_loo_stage(args, G_ref, nodelist, idx_of, M, deg, true_seeds, true_module,
                  max_size, out):
    tj_path = Path(args.true_dir) / "true_summary.json"
    if not tj_path.exists():
        raise SystemExit(f"{tj_path} not found; run --stage true first.")
    with open(tj_path) as f:
        tj = json.load(f)
    if "rebuilt" not in tj:
        raise SystemExit("true_summary.json has no 'rebuilt' entry; cannot do LOO.")
    ref_top = set(tj["rebuilt"]["top_genes"])
    ref_mod = set(tj["rebuilt"]["module_nodes"])
    file_top = set(tj["file"]["top_genes"])

    csv_path = out / "loo.csv"
    done = set()
    if csv_path.exists() and not args.no_resume:
        done = set(pd.read_csv(csv_path)["dropped_seed"].astype(str))
    mine = [(j, s) for j, s in enumerate(true_seeds) if j % args.n_tasks == args.task_id]
    log(f"LOO stage task {args.task_id}/{args.n_tasks}: {len(mine)} of {len(true_seeds)} seeds")

    for n_i, (j, dropped) in enumerate(mine):
        if dropped in done:
            continue
        row = {"dropped_seed": dropped, "seed_idx": j, "status": "error"}
        t0 = time.time()
        try:
            if args.perm_timeout > 0:
                signal.alarm(int(args.perm_timeout))
            reduced = [s for s in true_seeds if s != dropped]
            rm = run_rwr_module(G_ref, nodelist, idx_of, M, deg, reduced, args.restart,
                                k=args.k, max_size=max_size)
            if rm is None:
                row["status"] = "skip_no_module"
            else:
                row.update(module_size_raw=rm["raw_size"], module_size_final=len(rm["module"]),
                           capped=rm["capped"],
                           module_jaccard_vs_rebuilt=jaccard(ref_mod, rm["module"]))
                res = recluster_module(rm["module"], G_ref, args,
                                       base_seed=args.seed + 2_000_000 + j * 1000)
                if res is None:
                    row["status"] = "skip_too_small"
                else:
                    tg = res["top_genes"] or set()
                    row.update(res["summary"])
                    row.update(n_nodes=res["n_nodes"], gamma_lo=res["gamma_lo"],
                               gamma_hi=res["gamma_hi"],
                               top_cluster_jaccard_vs_rebuilt=jaccard(ref_top, tg),
                               top_cluster_jaccard_vs_file=jaccard(file_top, tg))
                    row["status"] = "ok"
        except PermTimeout:
            row["status"] = "timeout"
        finally:
            signal.alarm(0)
        row["t_total_s"] = time.time() - t0
        row["rss_gb"], row["peak_rss_gb"] = rss_gb()
        append_row(csv_path, row, LOO_COLS)
        log(f"[{n_i + 1}/{len(mine)}] dropped {dropped}: {row['status']} | "
            f"module Jaccard {row.get('module_jaccard_vs_rebuilt', float('nan')):.3f} | "
            f"top-cluster Jaccard vs rebuilt {row.get('top_cluster_jaccard_vs_rebuilt', float('nan')):.3f} | "
            f"{row['t_total_s']:.0f}s | peak {row['peak_rss_gb']:.2f} GiB")
        gc.collect()
    log("LOO task finished")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["true", "null", "loo"], required=True)
    ap.add_argument("--task-id", type=int, default=0, help="SLURM_ARRAY_TASK_ID")
    ap.add_argument("--n-tasks", type=int, default=1, help="number of array tasks (loo striding)")
    ap.add_argument("--true-dir", default=None, help="dir containing true_summary.json (loo)")
    ap.add_argument("--reference-graph", required=True)
    ap.add_argument("--disease-module-dir", required=True)
    ap.add_argument("--restart", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--n-perms", type=int, default=50, help="perms PER TASK (null stage)")
    ap.add_argument("--bin-size", type=int, default=5)
    ap.add_argument("--min-resolution", type=float, default=0.01)
    ap.add_argument("--max-resolution", type=float, default=10.0)
    ap.add_argument("--n-resolutions", type=int, default=30)
    ap.add_argument("--n-restarts", type=int, default=1)
    ap.add_argument("--neighbors", type=int, default=3,
                    help="MUST match the --neighbors of the real heterogeneous_hidef run.")
    ap.add_argument("--minsize", type=int, default=4)
    ap.add_argument("--jaccard", type=float, default=0.75)
    ap.add_argument("--chi", type=int, default=5)
    ap.add_argument("--consensus-p", type=int, default=75)
    ap.add_argument("--partition-type", choices=["RB", "CPM"], default="RB")
    ap.add_argument("--gamma-mode", choices=["calibrated", "fixed"], default="calibrated",
                    help="calibrated: per-module gamma window; fixed: geomspace(min,max).")
    ap.add_argument("--cap-factor", type=float, default=1.5,
                    help="stop extending a module for connectivity at cap-factor x true module size")
    ap.add_argument("--min-lcc-factor", type=float, default=0.5,
                    help="skip null perms whose final module is < this x true module size")
    ap.add_argument("--perm-timeout", type=int, default=0,
                    help="seconds per perm/LOO item before it is abandoned (0 = off)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir, args, inputs=[args.reference_graph])

    log(f"step11i stage={args.stage} task={args.task_id}/{args.n_tasks} host={socket.gethostname()} "
        f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')} ARRAY_TASK={os.environ.get('SLURM_ARRAY_TASK_ID')}")
    log(f"args: {vars(args)}")

    t0 = time.time()
    log("Loading reference graph ...")
    with open(args.reference_graph, "rb") as f:
        G_ref = pickle.load(f)
    G_ref = nx.relabel_nodes(G_ref, {n: str(n).strip() for n in G_ref.nodes()})
    nodelist = sorted(G_ref.nodes())
    log(f"  {G_ref.number_of_nodes():,} nodes, {G_ref.number_of_edges():,} edges "
        f"({time.time() - t0:.0f}s) | RSS {rss_gb()[0]:.2f} GiB")
    M, idx_of, deg = build_transition_matrix(G_ref, nodelist)
    log(f"  transition matrix built | RSS {rss_gb()[0]:.2f} GiB, peak {rss_gb()[1]:.2f} GiB "
        f"(this is the per-task baseline; --mem must exceed it plus per-perm growth)")

    dm_dir = Path(args.disease_module_dir)
    module_df = pd.read_csv(dm_dir / "disease_module_output" / "disease_module.csv")
    module_df["node"] = module_df["node"].astype(str).str.strip()
    true_module = module_df["node"].tolist()
    if "is_excluded_seed" in module_df.columns:
        anchor_mask = module_df["is_seed"] & (~module_df["is_excluded_seed"].astype(bool))
    else:
        anchor_mask = module_df["is_seed"]
    true_seeds = [g for g in module_df.loc[anchor_mask, "node"].tolist() if g in idx_of]
    max_size = int(math.ceil(args.cap_factor * len(true_module)))
    log(f"True module: {len(true_module)} nodes, {len(true_seeds)} anchoring seeds; "
        f"null module cap = {max_size}")

    stage_fn = {"true": run_true_stage, "null": run_null_stage, "loo": run_loo_stage}[args.stage]
    stage_fn(args, G_ref, nodelist, idx_of, M, deg, true_seeds, true_module, max_size, out)
    log(f"Done. Total wall time {(time.time() - t0) / 60:.1f} min; peak RSS {rss_gb()[1]:.2f} GiB")


if __name__ == "__main__":
    main()