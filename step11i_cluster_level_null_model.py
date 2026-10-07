#!/usr/bin/env python3
"""
step11i_cluster_level_null_model.py  (parallel / resumable, MultiXrank version)

Stages (run as SLURM jobs / job arrays):
  --stage true   Re-cluster the true module (disease_module.csv). With
                 --module-builder multixrank it also checks that MultiXrank on
                 the true seeds reproduces that module (Jaccard) -- the sanity
                 check that the null builder is faithful. Run once.
  --stage null   --n-perms degree/cell-type-matched null permutations for task --task-id.
  --stage loo    Leave-one-seed-out for seeds[task_id::n_tasks]. Needs --true-dir.
Aggregation lives in step11i_aggregate.py.

NULL MODULES (--module-builder multixrank, the default)
-------------------------------------------------------
For each null permutation: sample degree + cell-type matched seeds, run the REAL
MultiXrank (same config as step 4) from those seeds, rank "celltype::GENE" by
score, then add genes one at a time until the induced subgraph is connected
(step 6's loop, via union-find). Then re-cluster. --module-builder rwr keeps the
old plain-RWR proxy.
"""
import argparse
import gc
import json
import math
import os
import pickle
import resource
import shutil
import signal
import socket
import sys
import tempfile
import time
import yaml
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
                    cur = int(line.split()[1]) / 1024 ** 2
                    break
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
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
# Node-name helpers (identical to step6) and graph preprocessing
# ─────────────────────────────────────────────────────────────────────────

def _norm_node(s):
    s = s.strip()
    if "::" in s:
        ct, gene = s.rsplit("::", 1)
        return f"{ct}::{gene.strip().upper()}"
    return s.upper()


def _celltype_of(node):
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


def filter_to_lcc_per_layer(H):
    """Copy of step6's filter: keep nodes in the LCC of >= 1 cell-type layer."""
    layers = set()
    for _, _, d in H.edges(data=True):
        layers |= d.get("cell_types", set())
    if not layers:
        log("  [lcc-per-layer] No 'cell_types' edge attribute found -- skipping.")
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
        log(f"  [lcc-per-layer] '{layer}': {Lsub.number_of_nodes():,} -> {len(lcc):,} nodes")
        kept_nodes |= lcc
    H_filtered = H.subgraph(kept_nodes).copy()
    log(f"  [lcc-per-layer] total {H.number_of_nodes():,} -> {H_filtered.number_of_nodes():,} nodes, "
        f"{H.number_of_edges():,} -> {H_filtered.number_of_edges():,} edges")
    return H_filtered


# ─────────────────────────────────────────────────────────────────────────
# MultiXrank ranker (one fresh MultiXrank run per call, same config as step 4)
# ─────────────────────────────────────────────────────────────────────────

def make_mx_ranker(wdir, config_name, tmp_root):
    """Returns ranker(seed_nodes) -> list of 'celltype::GENE' sorted by MultiXrank
    score (descending, score > 0). Each call builds a scratch wdir that symlinks
    everything in the real MultiXrank wdir except the seed file, which is replaced
    by the given seeds. Same config.yml => same restart/lambda/eta as step 4."""
    import re
    import multixrank

    wdir = Path(wdir).resolve()
    cfg_text = (wdir / config_name).read_text()
    m = re.search(r"^\s*seed\s*:\s*[\"']?([^\"'\s#]+)", cfg_text, re.M)
    seed_rel = Path(m.group(1) if m else "seeds/seeds.txt")
    seed_top = seed_rel.parts[0]
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    log(f"MultiXrank ranker: wdir={wdir} config={config_name} seed file={seed_rel} scratch={tmp_root}")

    def ranker(seed_nodes):
        d = Path(tempfile.mkdtemp(prefix="mx_", dir=str(tmp_root)))
        try:
            for item in wdir.iterdir():
                if item.name == seed_top:
                    continue
                (d / item.name).symlink_to(item)
            seed_file = d / seed_rel
            seed_file.parent.mkdir(parents=True, exist_ok=True)
            seed_file.write_text("\n".join(seed_nodes) + "\n")
            mxr = multixrank.Multixrank(config=str(d / config_name), wdir=str(d))
            res = mxr.random_walk_rank()
        finally:
            shutil.rmtree(d, ignore_errors=True)

        if isinstance(res, list):
            raw = pd.concat(res, ignore_index=True)
        elif isinstance(res, dict):
            raw = pd.concat([v for v in res.values() if isinstance(v, pd.DataFrame)], ignore_index=True)
        else:
            raw = res.copy()
        raw.columns = [str(c).lower().strip() for c in raw.columns]
        raw["score"] = pd.to_numeric(raw["score"], errors="coerce").fillna(0.0)
        raw = raw[raw["score"] > 0]
        # same namespace as step 6: "<multiplex>::GENE"
        gene = raw["node"].astype(str).str.strip().str.rsplit("::", n=1).str[-1].str.upper()
        name = raw["multiplex"].astype(str) + "::" + gene
        s = (pd.Series(raw["score"].to_numpy(), index=name.to_numpy())
             .groupby(level=0).max().sort_values(ascending=False))
        out_list = s.index.tolist()
        # release the multiplex objects MultiXrank built; give freed heap back to the OS
        del mxr, res, raw, gene, name, s
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        return out_list

    return ranker


# ─────────────────────────────────────────────────────────────────────────
# RWR / MultiXrank + step6-style greedy module construction
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


def greedy_connect(G, seeds, ranked_iter, max_added):
    """step6's build_module loop, made cheap with union-find."""
    parent = {}
    in_module = set()
    state = {"n_comp": 0}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def add(v):
        parent[v] = v
        in_module.add(v)
        state["n_comp"] += 1
        for u in G[v]:
            if u != v and u in in_module:
                ra, rb = find(v), find(u)
                if ra != rb:
                    parent[ra] = rb
                    state["n_comp"] -= 1

    order = []
    for s in seeds:
        add(s)
        order.append(s)
    if state["n_comp"] == 1:
        return order, True, 0
    added = 0
    for g in ranked_iter:
        if added >= max_added:
            break
        add(g)
        order.append(g)
        added += 1
        if state["n_comp"] == 1:
            return order, True, added
    return order, False, added


def run_rwr_module(G_ref, nodelist, idx_of, M, deg, seed_nodes, restart,
                   max_bridge=30000, exclude_nodes=frozenset(), max_size=None,
                   policy="skip", scaling=0, mx_ranker=None):
    """Score genes from seeds (MultiXrank if mx_ranker given, else plain RWR),
    then greedy-connect (step6 procedure)."""
    n = len(nodelist)
    seeds_in_graph = [g for g in seed_nodes if g in idx_of]
    if not seeds_in_graph:
        return None
    seeds_set = set(seeds_in_graph)

    if mx_ranker is not None:
        ranked_list = mx_ranker(seeds_in_graph)

        def ranked_iter():
            for nd in ranked_list:
                if nd in seeds_set or nd in exclude_nodes or nd not in idx_of:
                    continue
                yield nd
    else:
        p0 = np.zeros(n)
        for node in seeds_in_graph:
            i = idx_of[node]
            p0[i] = np.sqrt(deg[i]) if (scaling == 1 and deg[i] > 0) else 1.0
        p0 = p0 / p0.sum()
        pinf = rwr_power_iteration(M, p0, restart)
        ranked_idx = np.argsort(pinf)[::-1]

        def ranked_iter():
            for i in ranked_idx:
                if pinf[i] <= 0:
                    return
                nd = nodelist[i]
                if nd in seeds_set or nd in exclude_nodes:
                    continue
                yield nd

    order, connected, added = greedy_connect(G_ref, seeds_in_graph, ranked_iter(), max_bridge)
    out = {"module": None, "raw_size": len(order), "connected_raw": connected,
           "n_bridges": added if connected else None, "capped": False, "oversize": False}
    module = order
    if max_size is not None and len(order) > max_size:
        out["capped"] = True
        if policy == "skip":
            out["oversize"] = True
            return out
        module = order[:max_size]
    if not (connected and not out["capped"]):
        lcc = max(nx.connected_components(G_ref.subgraph(module)), key=len)
        module = [m for m in module if m in lcc]
    out["module"] = module
    return out


# ─────────────────────────────────────────────────────────────────────────
# Degree- (and cell-type-) matched null seeds
# ─────────────────────────────────────────────────────────────────────────

def build_degree_bins(G, bin_size=5, match_celltype=True, excluded_cts=frozenset()):
    bins = defaultdict(list)
    for node, d in G.degree():
        ct = _celltype_of(node)
        if ct.upper() in excluded_cts:
            continue
        bins[(ct if match_celltype else None, (d // bin_size) * bin_size)].append(node)
    return bins


def sample_degree_matched_seeds(true_seeds, G, degree_bins, rng, bin_size=5, match_celltype=True):
    true_set = set(true_seeds)
    sampled = set()
    for seed in true_seeds:
        d = G.degree(seed)
        ct = _celltype_of(seed) if match_celltype else None
        bin_key = (d // bin_size) * bin_size
        candidates = [n for n in degree_bins.get((ct, bin_key), [])
                      if n not in sampled and n not in true_set]
        if not candidates:
            for offset in [bin_size, -bin_size, 2 * bin_size, -2 * bin_size]:
                candidates = [n for n in degree_bins.get((ct, bin_key + offset), [])
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
    nv = g_ig.vcount()
    dens = 2 * g_ig.ecount() / (nv * (nv - 1)) if nv > 1 else 0.01
    lo = max(dens * lo_mult, 1e-4)
    hi = min(max_res, 0.9)
    if lo >= hi:
        lo = hi / 100
    return np.geomspace(lo, hi, n)


def calibrate_gamma_grid(g_ig, n_grid, minsize, partition_type, probe_lo, probe_hi,
                         seed, n_probe=30, coverage_floor=0.8):
    """lo = last probe before the module first splits into >=2 communities of
    size >= minsize; hi = first probe where < coverage_floor of nodes sit in
    communities >= minsize. Returns (grid or None, probe_rows)."""
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
                                            args.partition_type, plo, phi, seed,
                                            coverage_floor=args.coverage_floor)
    if grid is None:
        if args.partition_type == "CPM":
            return module_gamma_grid(g_ig, args.n_resolutions, args.max_resolution), \
                probe_rows, "fallback_density"
        return fixed, probe_rows, "fallback_fixed"
    return grid, probe_rows, "calibrated"


# ─────────────────────────────────────────────────────────────────────────
# Re-cluster one module
# ─────────────────────────────────────────────────────────────────────────

def recluster_module(module_nodes, G_ref, args, base_seed, ref_sets=None):
    """ref_sets: optional {name: set(genes)}. For each, 'best_match' reports the
    max Jaccard between that set and ANY cluster in the consensus hierarchy
    (rank-free stability measure used by leave-one-out)."""
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

    best_match = {}
    if ref_sets:
        for name, rs in ref_sets.items():
            ref_mask = np.fromiter((nd in rs for nd in nodes), dtype=bool, count=len(nodes))
            rn = int(ref_mask.sum())
            b = 0.0
            for c in (cluG or []):
                m = np.asarray(c[0]).astype(bool).ravel()
                inter = int(np.count_nonzero(m & ref_mask))
                union = int(m.sum()) + rn - inter
                if union:
                    b = max(b, inter / union)
            best_match[name] = b

    return {"summary": summary, "top_genes": top_genes, "n_nodes": n_nodes,
            "n_edges": sub.number_of_edges(), "gamma_lo": float(gammas[0]),
            "gamma_hi": float(gammas[-1]), "gamma_status": gamma_status,
            "n_partitions": len(partitions), "probe_rows": probe_rows,
            "best_match": best_match,
            "t_calibrate_s": t1 - t0, "t_leiden_s": t2 - t1, "t_consensus_s": t3 - t2}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def res_to_json(res, module_nodes=None):
    d = {k: v for k, v in res.items() if k not in ("probe_rows", "top_genes", "best_match")}
    d["top_genes"] = sorted(res["top_genes"]) if res["top_genes"] else []
    if module_nodes is not None:
        d["module_nodes"] = sorted(module_nodes)
    return d


# ─────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────

def run_true_stage(args, G_ref, nodelist, idx_of, M, deg, true_seeds, true_module,
                   max_size, excl_nodes, out):
    hard_cap = 3 * max_size

    # ---- MultiXrank mode: cluster the file module + sanity-check the builder ----
    if args.module_builder == "multixrank":
        log("TRUE stage: checking that MultiXrank on the true seeds reproduces the file module ...")
        t = time.time()
        rb = run_rwr_module(G_ref, nodelist, idx_of, M, deg, true_seeds, args.restart,
                            max_bridge=args.max_bridge, exclude_nodes=excl_nodes,
                            max_size=hard_cap, policy="lcc", mx_ranker=args.mx_ranker)
        mx_check = None
        if rb is not None and rb["module"] is not None:
            jac = jaccard(true_module, rb["module"])
            mx_check = {"module_size": len(rb["module"]), "n_bridges": rb["n_bridges"],
                        "connected_raw": rb["connected_raw"], "jaccard_with_file_module": jac,
                        "seconds": time.time() - t}
            log(f"  MultiXrank module from true seeds: {len(rb['module'])} nodes "
                f"(n_bridges={rb['n_bridges']}); file module {len(true_module)} nodes; "
                f"Jaccard = {jac:.3f}  ({time.time() - t:.0f}s)")
            if jac < 0.8:
                log("  WARNING: Jaccard < 0.8 -- the null builder does not reproduce your step-6 module. "
                    "Check --mx-config / seeds file before trusting the nulls.")
        else:
            log("  WARNING: MultiXrank check failed (no module)")

        log("TRUE stage: re-clustering the module read from disease_module.csv ...")
        res_file = recluster_module(true_module, G_ref, args, base_seed=args.seed)
        if res_file is None:
            raise SystemExit("True module too small to re-cluster with current --minsize.")
        log(f"  file module: {res_file['n_nodes']} nodes, {res_file['n_edges']} edges, "
            f"gamma [{res_file['gamma_lo']:.4g}, {res_file['gamma_hi']:.4g}] ({res_file['gamma_status']}), "
            f"{res_file['n_partitions']} partitions")
        log(f"  summary: {res_file['summary']}")
        pd.DataFrame(res_file["probe_rows"]).to_csv(out / "gamma_probe_curve_file.csv", index=False)
        out_json = {"n_true_module": len(true_module), "n_true_seeds": len(true_seeds),
                    "seeds": true_seeds, "max_size_cap": max_size,
                    "file": res_to_json(res_file, true_module), "mx_check": mx_check}
        with open(out / "true_summary.json", "w") as f:
            json.dump(out_json, f, indent=1, default=to_jsonable)
        log(f"Wrote {out / 'true_summary.json'}")
        return

    # ---- old RWR-proxy mode ----
    log("TRUE stage: restart scan (RWR proxy vs. file module) ...")
    scan_rows = []
    for r in [float(x) for x in args.restart_scan.split(",") if x.strip()]:
        rr = run_rwr_module(G_ref, nodelist, idx_of, M, deg, true_seeds, r,
                            max_bridge=args.max_bridge, exclude_nodes=excl_nodes,
                            max_size=hard_cap, policy="lcc")
        if rr is None or rr["module"] is None:
            continue
        row = {"restart": r, "module_size": len(rr["module"]), "raw_size": rr["raw_size"],
               "connected_raw": rr["connected_raw"], "n_bridges": rr["n_bridges"],
               "jaccard_with_file_module": jaccard(true_module, rr["module"])}
        scan_rows.append(row)
        log(f"  restart {r:.2f}: module {row['module_size']} nodes (n_bridges={row['n_bridges']}), "
            f"Jaccard with file module = {row['jaccard_with_file_module']:.3f}")
    if scan_rows:
        scan_df = pd.DataFrame(scan_rows)
        scan_df.to_csv(out / "restart_scan.csv", index=False)

    log("TRUE stage: re-clustering the module read from disease_module.csv ...")
    res_file = recluster_module(true_module, G_ref, args, base_seed=args.seed)
    if res_file is None:
        raise SystemExit("True module too small to re-cluster with current --minsize.")
    log(f"  summary: {res_file['summary']}")
    pd.DataFrame(res_file["probe_rows"]).to_csv(out / "gamma_probe_curve_file.csv", index=False)

    rb = run_rwr_module(G_ref, nodelist, idx_of, M, deg, true_seeds, args.restart,
                        max_bridge=args.max_bridge, exclude_nodes=excl_nodes,
                        max_size=hard_cap, policy="lcc")
    out_json = {"n_true_module": len(true_module), "n_true_seeds": len(true_seeds),
                "seeds": true_seeds, "max_size_cap": max_size}
    out_json["file"] = res_to_json(res_file, true_module)
    if rb is not None and rb["module"] is not None:
        jac_mod = jaccard(true_module, rb["module"])
        res_rb = recluster_module(rb["module"], G_ref, args, base_seed=args.seed + 1)
        if res_rb is not None:
            out_json["rebuilt"] = res_to_json(res_rb, rb["module"])
            out_json["rebuilt"].update({"module_size_raw": rb["raw_size"],
                                        "connected_raw": rb["connected_raw"],
                                        "n_bridges": rb["n_bridges"], "capped": rb["capped"],
                                        "jaccard_with_file_module": jac_mod})
    with open(out / "true_summary.json", "w") as f:
        json.dump(out_json, f, indent=1, default=to_jsonable)
    log(f"Wrote {out / 'true_summary.json'}")


NULL_COLS = (["perm_idx", "task_id", "status", "n_seeds_sampled", "module_size_raw",
              "module_size_final", "connected_raw", "n_bridges", "capped", "n_nodes", "n_edges",
              "gamma_lo", "gamma_hi", "gamma_status", "n_partitions",
              "t_rwr_s", "t_calibrate_s", "t_leiden_s", "t_consensus_s", "t_total_s",
              "rss_gb", "peak_rss_gb"] + SUMMARY_KEYS)


def run_null_stage(args, G_ref, nodelist, idx_of, M, deg, true_seeds, true_module,
                   max_size, excl_nodes, out):
    csv_path = out / "null_perms.csv"
    done = set()
    if csv_path.exists() and not args.no_resume:
        done = set(pd.read_csv(csv_path)["perm_idx"].astype(int))
        log(f"Resuming: {len(done)} perms already in {csv_path.name}")
    min_size = int(args.min_lcc_factor * len(true_module))
    excl_cts = {c.strip().upper() for c in args.exclude_seed_celltypes}
    degree_bins = build_degree_bins(G_ref, bin_size=args.bin_size,
                                    match_celltype=args.match_celltype, excluded_cts=excl_cts)
    log(f"NULL stage task {args.task_id}/{args.n_tasks}: {args.n_perms} perms "
        f"(global idx {args.task_id * args.n_perms}..{(args.task_id + 1) * args.n_perms - 1}); "
        f"builder={args.module_builder}; module size window [{min_size}, {max_size}], "
        f"oversize policy={args.oversize_policy}, match_celltype={args.match_celltype}, "
        f"seed pool excludes {sorted(excl_cts) or 'nothing'}")

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
                                                     bin_size=args.bin_size,
                                                     match_celltype=args.match_celltype)
            row["n_seeds_sampled"] = len(rand_seeds)
            if len(rand_seeds) < max(1, len(true_seeds) // 2):
                row["status"] = "skip_too_few_seeds"
            else:
                t = time.time()
                rm = run_rwr_module(G_ref, nodelist, idx_of, M, deg, rand_seeds, args.restart,
                                    max_bridge=args.max_bridge, exclude_nodes=excl_nodes,
                                    max_size=max_size, policy=args.oversize_policy,
                                    mx_ranker=args.mx_ranker)
                row["t_rwr_s"] = time.time() - t
                if rm is None:
                    row["status"] = "skip_no_module"
                else:
                    row.update(module_size_raw=rm["raw_size"], connected_raw=rm["connected_raw"],
                               n_bridges=rm["n_bridges"], capped=rm["capped"])
                    if rm["oversize"]:
                        row["status"] = "skip_oversize"
                    else:
                        row["module_size_final"] = len(rm["module"])
                        if len(rm["module"]) < min_size:
                            row["status"] = "skip_small"
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
            log(f"perm {perm_idx} [{i + 1}/{args.n_perms}] ok | seeds {row['n_seeds_sampled']} -> "
                f"module {row['module_size_final']} (bridges to connect: {row['n_bridges']}) | "
                f"clusters {row['n_clusters']}, maxpers {row['max_persistence']:.3g} | "
                f"gamma [{row['gamma_lo']:.3g},{row['gamma_hi']:.3g}] {row['gamma_status']} | "
                f"{row['t_total_s']:.0f}s (builder {row.get('t_rwr_s', 0):.1f}, cal {row['t_calibrate_s']:.1f}, "
                f"leiden {row['t_leiden_s']:.1f}, cons {row['t_consensus_s']:.1f}) | "
                f"RSS {row['rss_gb']:.2f} GiB, peak {row['peak_rss_gb']:.2f} GiB | ETA {eta / 60:.1f} min")
        else:
            log(f"perm {perm_idx} [{i + 1}/{args.n_perms}] {row['status']} | module raw "
                f"{row.get('module_size_raw')} (connected={row.get('connected_raw')}, "
                f"bridges={row.get('n_bridges')}) | {row['t_total_s']:.0f}s | peak {row['peak_rss_gb']:.2f} GiB")
        del res
        gc.collect()

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        log("Status counts: " + ", ".join(f"{k}={v}" for k, v in df["status"].value_counts().items()))
    log(f"NULL task {args.task_id} finished in {(time.time() - t_start) / 60:.1f} min")


LOO_COLS = (["dropped_seed", "seed_idx", "status", "module_size_raw", "module_size_final",
             "capped", "n_bridges", "module_jaccard_vs_rebuilt",
             "best_match_jaccard_vs_rebuilt", "best_match_jaccard_vs_file",
             "top_cluster_jaccard_vs_rebuilt", "top_cluster_jaccard_vs_file",
             "n_nodes", "gamma_lo", "gamma_hi", "t_total_s", "rss_gb", "peak_rss_gb"]
            + SUMMARY_KEYS)


def run_loo_stage(args, G_ref, nodelist, idx_of, M, deg, true_seeds, true_module,
                  max_size, excl_nodes, out):
    tj_path = Path(args.true_dir) / "true_summary.json"
    if not tj_path.exists():
        raise SystemExit(f"{tj_path} not found; run --stage true first.")
    with open(tj_path) as f:
        tj = json.load(f)
    # With MultiXrank there is no 'rebuilt' module: the file module IS the reference.
    refkey = "rebuilt" if "rebuilt" in tj else "file"
    ref_top = set(tj[refkey]["top_genes"])
    ref_mod = set(tj[refkey]["module_nodes"])
    file_top = set(tj["file"]["top_genes"])
    hard_cap = 3 * max_size

    csv_path = out / "loo.csv"
    done = set()
    if csv_path.exists() and not args.no_resume:
        done = set(pd.read_csv(csv_path)["dropped_seed"].astype(str))
    mine = [(j, s) for j, s in enumerate(true_seeds) if j % args.n_tasks == args.task_id]
    log(f"LOO stage task {args.task_id}/{args.n_tasks}: {len(mine)} of {len(true_seeds)} seeds "
        f"(reference = {refkey})")

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
                                max_bridge=args.max_bridge, exclude_nodes=excl_nodes,
                                max_size=hard_cap, policy="lcc", mx_ranker=args.mx_ranker)
            if rm is None or rm["module"] is None:
                row["status"] = "skip_no_module"
            else:
                row.update(module_size_raw=rm["raw_size"], module_size_final=len(rm["module"]),
                           capped=rm["capped"], n_bridges=rm["n_bridges"],
                           module_jaccard_vs_rebuilt=jaccard(ref_mod, rm["module"]))
                res = recluster_module(rm["module"], G_ref, args,
                                       base_seed=args.seed + 2_000_000 + j * 1000,
                                       ref_sets={"rebuilt": ref_top, "file": file_top})
                if res is None:
                    row["status"] = "skip_too_small"
                else:
                    tg = res["top_genes"] or set()
                    row.update(res["summary"])
                    row.update(n_nodes=res["n_nodes"], gamma_lo=res["gamma_lo"],
                               gamma_hi=res["gamma_hi"],
                               best_match_jaccard_vs_rebuilt=res["best_match"].get("rebuilt"),
                               best_match_jaccard_vs_file=res["best_match"].get("file"),
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
        nan = float("nan")
        log(f"[{n_i + 1}/{len(mine)}] dropped {dropped}: {row['status']} | "
            f"module Jaccard {row.get('module_jaccard_vs_rebuilt', nan):.3f} | "
            f"best-match Jaccard of ref top cluster {row.get('best_match_jaccard_vs_rebuilt') or nan:.3f} | "
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
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--true-dir", default=None)
    ap.add_argument("--reference-graph", required=True)
    ap.add_argument("--disease-module-dir", required=True)
    # --- module builder
    ap.add_argument("--module-builder", choices=["multixrank", "rwr"], default="multixrank",
                    help="How null (and LOO) modules are scored: real MultiXrank (default) or the plain-RWR proxy.")
    ap.add_argument("--mx-wdir", default=None,
                    help="MultiXrank working dir (the one used in step 4). Default: folder of --reference-graph.")
    ap.add_argument("--mx-config", default="config.yml", help="MultiXrank config inside --mx-wdir.")
    ap.add_argument("--mx-tmp", default=None, help="Scratch dir for per-permutation MultiXrank runs.")
    # --- step6-fidelity options
    ap.add_argument("--lcc-per-layer", action="store_true",
                    help="Apply step6's --lcc-per-layer filter to the reference graph (use if step6 did).")
    ap.add_argument("--exclude-seeds", nargs="+", default=[], metavar="NODE",
                    help="Same list as step6 --exclude-seeds; kept out of the ranking pool.")
    ap.add_argument("--exclude-seed-celltypes", nargs="+", default=["global_ppi"], metavar="CELLTYPE",
                    help="Cell types never sampled as null seeds (still in graph). Default: global_ppi.")
    ap.add_argument("--no-match-celltype", dest="match_celltype", action="store_false",
                    help="Match null seeds on degree only, not on (cell-type layer, degree).")
    ap.add_argument("--max-bridge", type=int, default=30000,
                    help="Max genes added while connecting seeds (step6 --max-bridge).")
    ap.add_argument("--restart", type=float, default=0.7,
                    help="Only used by --module-builder rwr (MultiXrank takes it from its config).")
    ap.add_argument("--restart-scan", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                    help="rwr mode, true stage: restart values to compare against the file module.")
    # --- null design
    ap.add_argument("--n-perms", type=int, default=50, help="perms PER TASK (null stage)")
    ap.add_argument("--bin-size", type=int, default=5)
    ap.add_argument("--cap-factor", type=float, default=1.5,
                    help="null module size cap = cap-factor x true module size")
    ap.add_argument("--oversize-policy", choices=["skip", "lcc"], default="skip",
                    help="null modules above the cap: skip them (logged) or truncate to cap + LCC")
    ap.add_argument("--min-lcc-factor", type=float, default=0.5,
                    help="skip null modules smaller than this x true module size")
    ap.add_argument("--perm-timeout", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    # --- clustering recipe (must match the real hierarchy run)
    ap.add_argument("--min-resolution", type=float, default=0.01)
    ap.add_argument("--max-resolution", type=float, default=10.0)
    ap.add_argument("--n-resolutions", type=int, default=30)
    ap.add_argument("--n-restarts", type=int, default=1)
    ap.add_argument("--neighbors", type=int, default=3)
    ap.add_argument("--minsize", type=int, default=4)
    ap.add_argument("--jaccard", type=float, default=0.75)
    ap.add_argument("--chi", type=int, default=5)
    ap.add_argument("--consensus-p", type=int, default=75)
    ap.add_argument("--partition-type", choices=["RB", "CPM"], default="RB")
    ap.add_argument("--gamma-mode", choices=["calibrated", "fixed"], default="calibrated")
    ap.add_argument("--coverage-floor", type=float, default=0.8,
                    help="calibrated grid ends where < this fraction of nodes sit in communities >= minsize")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    ap.set_defaults(match_celltype=True)
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
        G_raw = pickle.load(f)
    G_ref = nx.relabel_nodes(G_raw, {n: _norm_node(str(n)) for n in G_raw.nodes()})
    del G_raw
    gc.collect()
    log(f"  {G_ref.number_of_nodes():,} nodes, {G_ref.number_of_edges():,} edges "
        f"({time.time() - t0:.0f}s) | RSS {rss_gb()[0]:.2f} GiB")
    if args.lcc_per_layer:
        log("Applying --lcc-per-layer (as in step6) ...")
        G_ref = filter_to_lcc_per_layer(G_ref)
        gc.collect()
    nodelist = sorted(G_ref.nodes())
    M, idx_of, deg = build_transition_matrix(G_ref, nodelist)
    log(f"  transition matrix built | RSS {rss_gb()[0]:.2f} GiB, peak {rss_gb()[1]:.2f} GiB "
        f"(per-task baseline; --mem must exceed it plus per-perm growth)")

    args.mx_ranker = None
    if args.module_builder == "multixrank":
        mx_wdir = args.mx_wdir or str(Path(args.reference_graph).resolve().parent)
        mx_tmp = args.mx_tmp or os.path.join(tempfile.gettempdir(), f"mx11i_{os.environ.get('SLURM_JOB_ID', 'local')}_{args.task_id}")
        args.mx_ranker = make_mx_ranker(mx_wdir, args.mx_config, mx_tmp)

    dm_dir = Path(args.disease_module_dir)
    module_df = pd.read_csv(dm_dir / "disease_module_output" / "disease_module.csv")
    module_df["node"] = module_df["node"].astype(str).str.strip()
    true_module = module_df["node"].tolist()
    n_missing = sum(1 for n in true_module if n not in idx_of)
    if n_missing:
        log(f"  WARNING: {n_missing}/{len(true_module)} file-module nodes are NOT in the reference graph "
            f"(check node naming / --lcc-per-layer); they are ignored when re-clustering.")
    if "is_excluded_seed" in module_df.columns:
        anchor_mask = module_df["is_seed"] & (~module_df["is_excluded_seed"].astype(bool))
    else:
        anchor_mask = module_df["is_seed"]
    anchor_all = module_df.loc[anchor_mask, "node"].tolist()
    true_seeds = [g for g in anchor_all if g in idx_of]
    excl_nodes = frozenset(_norm_node(g) for g in args.exclude_seeds)
    max_size = int(math.ceil(args.cap_factor * len(true_module)))
    log(f"True module: {len(true_module)} nodes, {len(true_seeds)} anchoring seeds "
        f"({len(anchor_all) - len(true_seeds)} not in graph); null module cap = {max_size}; "
        f"{len(excl_nodes)} nodes excluded from ranking pool")

    stage_fn = {"true": run_true_stage, "null": run_null_stage, "loo": run_loo_stage}[args.stage]
    stage_fn(args, G_ref, nodelist, idx_of, M, deg, true_seeds, true_module, max_size, excl_nodes, out)
    log(f"Done. Total wall time {(time.time() - t0) / 60:.1f} min; peak RSS {rss_gb()[1]:.2f} GiB")


if __name__ == "__main__":
    main()