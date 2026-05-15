#!/usr/bin/env python3
"""
Multiplex LCC Significance Analysis (Degree-Normalized)
=======================================================
Per-layer or flattened analysis of an integrated multiplex with edge-type
filtering, degree-preserving null, and optional per-component metrics.
"""

import os
import sys
import gc
import time
import pickle
import random
import argparse
import traceback
import warnings
import multiprocessing as mp
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import psutil
    def print_memory_usage():
        p = psutil.Process(os.getpid())
        print(f"Memory usage: {p.memory_info().rss / (1024 * 1024):.2f} MB")
except ImportError:
    def print_memory_usage():
        pass


# ============================================================================
# DEGREE-NORMALIZED LCC SIGNIFICANCE
# ============================================================================
def popDoL(mydict, k, v):
    mydict.setdefault(k, [])
    if v not in mydict[k]:
        mydict[k].append(v)
    return mydict


def size_lcc(G):
    if G.number_of_nodes() == 0:
        return 0
    return max((len(c) for c in nx.connected_components(G)), default=0)


def getDict_degree2nodeNames(G, nodeList=None, numBins=None, binLimit=None):
    deg2nodes = {}
    nodes = nodeList if nodeList is not None else G.nodes()
    nodes = [n for n in nodes if n in G]
    for n in nodes:
        popDoL(deg2nodes, G.degree(n), n)

    if not deg2nodes:
        return {}

    if numBins is not None:
        minDeg = max(min(deg2nodes.keys()), 1)
        maxDeg = max(deg2nodes.keys())
        if maxDeg <= minDeg:
            return deg2nodes
        bw = (np.log10(maxDeg) - np.log10(minDeg)) / numBins
        lb = 10 ** (bw * np.arange(numBins))
        ub = 10 ** (bw * (np.arange(numBins) + 1))
        binned = {}
        for i in range(numBins):
            binned[ub[i]] = []
            for d in deg2nodes:
                if d > lb[i] and d <= ub[i]:
                    binned[ub[i]].extend(deg2nodes[d])
        return binned

    if binLimit is not None:
        ub = sorted(binLimit)
        lb = [-1] + ub[:-1]
        binned = {}
        for i, u in enumerate(ub):
            binned[u] = []
            for d in deg2nodes:
                if d > lb[i] and d <= u:
                    binned[u].extend(deg2nodes[d])
        return binned

    return deg2nodes


def getDict_degree2numNodes(G, **kw):
    return {k: len(v) for k, v in getDict_degree2nodeNames(G, **kw).items()}


def randSubgraph_preserveDegree(pool_graph, target_dist, eval_graph):
    bin_limits = sorted(target_dist.keys())
    bin_dict = getDict_degree2nodeNames(pool_graph, binLimit=bin_limits)
    sampled = []
    for b, n_needed in target_dist.items():
        avail = bin_dict.get(b, [])
        if not avail or n_needed <= 0:
            continue
        k = min(n_needed, len(avail))
        sampled.extend(random.sample(list(avail), k))
    if not sampled:
        return eval_graph.subgraph([])
    sampled_in_eval = [n for n in sampled if n in eval_graph]
    return eval_graph.subgraph(sampled_in_eval)


def sig_lccOfSubgraph(eval_graph, pool_graph, seed_nodes,
                      numBins=10, numSimulations=1000, desc=""):
    seed_nodes = [n for n in seed_nodes if n in eval_graph]
    if not seed_nodes:
        return {"lcc": 0, "pVal": 1.0, "zScore": 0.0,
                "simResults": [], "mean": 0.0, "std": 0.0,
                "n_seeds_in_graph": 0}

    seeds_in_pool = [n for n in seed_nodes if n in pool_graph]
    if not seeds_in_pool:
        return {"lcc": size_lcc(eval_graph.subgraph(seed_nodes)),
                "pVal": 1.0, "zScore": 0.0,
                "simResults": [], "mean": 0.0, "std": 0.0,
                "n_seeds_in_graph": len(seed_nodes)}

    target_dist = getDict_degree2numNodes(pool_graph, nodeList=seeds_in_pool,
                                          numBins=numBins)
    obs_lcc = size_lcc(eval_graph.subgraph(seed_nodes))

    sim_sizes = []
    for _ in tqdm(range(numSimulations), desc=desc or "LCC sims"):
        gsim = randSubgraph_preserveDegree(pool_graph, target_dist, eval_graph)
        sim_sizes.append(size_lcc(gsim))

    arr = np.array(sim_sizes)
    pVal = float(np.mean(arr >= obs_lcc))
    mu, sd = float(arr.mean()), float(arr.std())
    z = (obs_lcc - mu) / sd if sd > 0 else 0.0
    return {"lcc": obs_lcc, "pVal": pVal, "zScore": z,
            "simResults": sim_sizes, "mean": mu, "std": sd,
            "n_seeds_in_graph": len(seed_nodes)}


# ============================================================================
# PER-COMPONENT METRICS (added back)
# ============================================================================
def _component_metrics(component_nodes, graph, seed_node_set, degrees,
                       component_id, compute_diameter=True):
    nodes = component_nodes if isinstance(component_nodes, set) else set(component_nodes)
    n = len(nodes)
    seed_count = len(seed_node_set & nodes)

    if n <= 1:
        return {
            "component_id": component_id,
            "component_size": n,
            "num_edges": 0,
            "density": 0.0,
            "diameter": 0,
            "seed_count": seed_count,
            "seed_density": seed_count / n if n else 0.0,
        }

    # Edges fully internal to the component (degree-sum overcounts boundary
    # edges only if subgraph is induced; here we use induced subgraph)
    sub = graph.subgraph(nodes)
    n_edges = sub.number_of_edges()
    density = (2 * n_edges) / (n * (n - 1))

    diameter = 0
    if compute_diameter and seed_count > 0 and n < 1000:
        try:
            if n > 100 and hasattr(nx.approximation, "diameter"):
                diameter = nx.approximation.diameter(sub)
            else:
                diameter = nx.diameter(sub)
        except Exception:
            diameter = float("inf")

    return {
        "component_id": component_id,
        "component_size": n,
        "num_edges": n_edges,
        "density": density,
        "diameter": diameter,
        "seed_count": seed_count,
        "seed_density": seed_count / n,
    }


def compute_per_component_metrics(graph_name, eval_graph, seed_nodes,
                                  max_components=5000):
    """Compute per-component metrics on `eval_graph`. Returns dict with:
       - components_df, seed_components_df
       - total_components, components_with_seeds, singleton_count, entropy
       - component_sizes
    """
    if eval_graph.number_of_nodes() == 0:
        return None

    seed_node_set = set(seed_nodes)
    degrees = dict(eval_graph.degree())

    components = list(nx.connected_components(eval_graph))
    components.sort(key=len, reverse=True)
    component_sizes = [len(c) for c in components]
    singleton_count = sum(1 for s in component_sizes if s == 1)

    if len(components) > max_components:
        # only retain the largest for detailed metrics (likely seed-bearing)
        components_for_metrics = components[:max_components]
    else:
        components_for_metrics = components

    rows = []
    seed_rows = []
    for i, comp in enumerate(tqdm(components_for_metrics,
                                  desc=f"Components [{graph_name}]")):
        m = _component_metrics(comp, eval_graph, seed_node_set, degrees,
                               component_id=i)
        m["seed_fraction"] = (m["seed_count"] / len(seed_nodes)
                              if seed_nodes else 0.0)
        rows.append(m)
        if m["seed_count"] > 0:
            seed_rows.append(m)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    seed_df = pd.DataFrame(seed_rows) if seed_rows else pd.DataFrame()

    entropy = 0.0
    if not df.empty and df["seed_count"].sum() > 0:
        p = df["seed_count"] / df["seed_count"].sum()
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p)))

    return {
        "components_df": df,
        "seed_components_df": seed_df,
        "total_components": len(components),
        "components_with_seeds": int((df["seed_count"] > 0).sum()) if not df.empty else 0,
        "singleton_count": singleton_count,
        "component_sizes": component_sizes,
        "entropy": entropy,
    }


def statistical_comparison_components(per_comp_results, out_path):
    """KS + Mann-Whitney across graphs (layers or modes)."""
    names = [n for n, r in per_comp_results.items()
             if r is not None and not r["seed_components_df"].empty]
    with open(out_path, "w") as f:
        f.write("Per-Component Statistical Comparison\n")
        f.write("=" * 60 + "\n\n")

        if len(names) < 2:
            f.write("Need at least 2 graphs with seed-containing components.\n")
            return

        data = {}
        for n in names:
            df = per_comp_results[n]["seed_components_df"]
            data[n] = {
                "seed_count": df["seed_count"].values,
                "component_size": df["component_size"].values,
                "density": df["density"].values,
                "diameter": df["diameter"].replace(
                    [np.inf, -np.inf], np.nan).dropna().values,
            }

        f.write("KS tests (seed_count distributions):\n")
        for a, b in combinations(names, 2):
            d1, d2 = data[a]["seed_count"], data[b]["seed_count"]
            if len(d1) == 0 or len(d2) == 0:
                f.write(f"  {a} vs {b}: empty\n"); continue
            ks, p = stats.ks_2samp(d1, d2)
            f.write(f"  {a} vs {b}: KS={ks:.4f}, p={p:.4g}"
                    f" {'(sig)' if p < 0.05 else ''}\n")

        f.write("\nMann-Whitney U (component sizes, seed-containing):\n")
        for a, b in combinations(names, 2):
            s1, s2 = data[a]["component_size"], data[b]["component_size"]
            if len(s1) == 0 or len(s2) == 0:
                f.write(f"  {a} vs {b}: empty\n"); continue
            try:
                u, p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
                f.write(f"  {a} vs {b}: U={u:.2f}, p={p:.4g}"
                        f" {'(sig)' if p < 0.05 else ''}\n")
            except Exception as e:
                f.write(f"  {a} vs {b}: error {e}\n")
    print(f"Saved per-component statistical comparison → {out_path}")


def save_per_component_outputs(per_comp_results, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # combined component-size distribution
    size_rows = []
    summary_rows = []
    for name, r in per_comp_results.items():
        if r is None:
            continue
        for s in r["component_sizes"]:
            size_rows.append({"graph": name, "component_size": s,
                              "is_singleton": s == 1})
        if r["component_sizes"]:
            summary_rows.append({
                "graph": name,
                "total_components": r["total_components"],
                "singleton_count": r["singleton_count"],
                "singleton_fraction": r["singleton_count"] / r["total_components"],
                "components_with_seeds": r["components_with_seeds"],
                "largest_component": max(r["component_sizes"]),
                "median_component_size": float(np.median(r["component_sizes"])),
                "mean_component_size": float(np.mean(r["component_sizes"])),
                "entropy": r["entropy"],
            })
        # per-graph components CSVs
        if not r["components_df"].empty:
            r["components_df"].to_csv(
                out_dir / f"components_{name}.csv", index=False)
        if not r["seed_components_df"].empty:
            r["seed_components_df"].to_csv(
                out_dir / f"seed_components_{name}.csv", index=False)

    if size_rows:
        pd.DataFrame(size_rows).to_csv(
            out_dir / "component_size_distributions.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            out_dir / "component_size_summary.csv", index=False)
    print(f"Saved per-component outputs → {out_dir}")


def plot_per_component(per_comp_results, plots_dir):
    plots_dir = Path(plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)

    names = list(per_comp_results.keys())
    entropies = [per_comp_results[n]["entropy"] if per_comp_results[n] else 0
                 for n in names]
    if any(entropies):
        plt.figure(figsize=(max(8, len(names) * 0.6), 5))
        plt.bar(names, entropies, color="#1B88A7")
        plt.ylabel("Entropy (bits)")
        plt.title("Seed distribution entropy across components")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "component_entropy.png", dpi=300)
        plt.close()


# ============================================================================
# LOADING
# ============================================================================
def load_multiplex(graph_path, registry_path):
    print(f"Loading multiplex graph from {graph_path} ...")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Loading node registry from {registry_path} ...")
    reg = pd.read_csv(registry_path, index_col="node_id")
    print(f"  {len(reg):,} registry entries")
    return G, reg


def load_seed_genes(path):
    df = pd.read_csv(path)
    if "gene" in df.columns:
        seeds = set(df["gene"].astype(str).str.upper().str.strip())
    else:
        seeds = set(df.iloc[:, 1].astype(str).str.upper().str.strip())
    seeds.discard("NAN")
    print(f"Loaded {len(seeds):,} seed gene symbols")
    return seeds


# ============================================================================
# GRAPH FILTERING
# ============================================================================
def filter_edges_by_type(G, edge_types):
    if edge_types is None:
        return G
    keep = [(u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") in edge_types]
    return G.edge_subgraph(keep).copy()


def build_layer_graph(G, source, cell_type, edge_types):
    layer_nodes = [n for n, d in G.nodes(data=True)
                   if d.get("source") == source and d.get("cell_type") == cell_type]
    sub = G.subgraph(layer_nodes).copy()
    if edge_types is not None:
        sub = filter_edges_by_type(sub, edge_types)
    return sub


def map_seeds_to_node_ids(G, seed_symbols, restrict_nodes=None):
    nodes_iter = restrict_nodes if restrict_nodes is not None else G.nodes()
    out = []
    for n in nodes_iter:
        g = G.nodes[n].get("gene")
        if g and str(g).upper().strip() in seed_symbols:
            out.append(n)
    return out


# ============================================================================
# ANALYSIS DRIVER
# ============================================================================
def analyze_graph(name, eval_graph, pool_graph, seed_symbols,
                  num_bins=10, num_simulations=1000):
    print(f"\n--- Analysing: {name} ---")
    print(f"  eval graph: {eval_graph.number_of_nodes():,} nodes, "
          f"{eval_graph.number_of_edges():,} edges")
    print(f"  pool graph: {pool_graph.number_of_nodes():,} nodes, "
          f"{pool_graph.number_of_edges():,} edges")

    seed_nodes = map_seeds_to_node_ids(eval_graph, seed_symbols)
    print(f"  seeds mapped to {len(seed_nodes):,} node IDs")

    sig = sig_lccOfSubgraph(eval_graph, pool_graph, seed_nodes,
                            numBins=num_bins, numSimulations=num_simulations,
                            desc=f"LCC sims [{name}]")
    print(f"  observed LCC={sig['lcc']}, p={sig['pVal']:.4g}, z={sig['zScore']:.3f}")
    return {
        "name": name,
        "n_nodes": eval_graph.number_of_nodes(),
        "n_edges": eval_graph.number_of_edges(),
        "n_seeds_in_graph": sig["n_seeds_in_graph"],
        "lcc_size": sig["lcc"],
        "lcc_p_value": sig["pVal"],
        "lcc_z_score": sig["zScore"],
        "lcc_random_mean": sig["mean"],
        "lcc_random_std": sig["std"],
        "lcc_significance": sig,
        "seed_nodes": seed_nodes,
        "eval_graph": eval_graph,
    }


# ============================================================================
# PLOTTING (LCC)
# ============================================================================
def plot_lcc_distributions(results, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in results]
    lccs = [r["lcc_size"] for r in results]
    ps = [r["lcc_p_value"] for r in results]

    fig, ax1 = plt.subplots(figsize=(max(8, len(names) * 0.6), 6))
    ax2 = ax1.twinx()
    ax1.bar(names, lccs, color="#1B88A7", alpha=0.85)
    ax1.set_ylabel("LCC size", color="#1B88A7")
    ax2.plot(names, ps, "o-", color="#E95E30")
    ax2.set_ylabel("p-value", color="#E95E30")
    ax2.axhline(0.05, color="#E95E30", ls="--", alpha=0.5)
    ax2.set_ylim(0, max(0.05, max(ps) * 1.1 if ps else 0.05))
    plt.xticks(rotation=60, ha="right")
    ax1.set_title("LCC size & significance (degree-normalized)")
    plt.tight_layout()
    plt.savefig(out_dir / "lcc_summary.png", dpi=300)
    plt.close()

    for r in results:
        sims = r["lcc_significance"]["simResults"]
        if not sims:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(sims, bins=30, alpha=0.75, color="#1B88A7",
                 label="Random (degree-normalized)")
        plt.axvline(r["lcc_size"], color="#C73188", ls="--", lw=2,
                    label=f"Observed LCC = {r['lcc_size']}")
        p = r["lcc_p_value"]
        lbl = (f"p < 0.001\nz = {r['lcc_z_score']:.2f}" if p < 5e-4
               else f"p = {p:.3f}\nz = {r['lcc_z_score']:.2f}")
        plt.title(f"LCC null: {r['name']}")
        plt.xlabel("LCC size"); plt.ylabel("Frequency"); plt.legend()
        plt.text(0.98, 0.95, lbl, transform=plt.gca().transAxes,
                 ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        plt.tight_layout()
        safe = r["name"].replace("/", "_").replace(" ", "_")
        plt.savefig(out_dir / f"lcc_null_{safe}.png", dpi=300)
        plt.close()


def write_summary(results, out_dir):
    out_dir = Path(out_dir)
    rows = [{k: v for k, v in r.items()
             if k not in ("lcc_significance", "seed_nodes", "eval_graph")}
            for r in results]
    pd.DataFrame(rows).to_csv(out_dir / "lcc_summary.csv", index=False)

    with open(out_dir / "lcc_summary.txt", "w") as f:
        f.write("Multiplex LCC Significance (Degree-Normalized)\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"[{r['name']}]\n")
            f.write(f"  nodes={r['n_nodes']:,}  edges={r['n_edges']:,}\n")
            f.write(f"  seeds in graph: {r['n_seeds_in_graph']}\n")
            f.write(f"  observed LCC : {r['lcc_size']}\n")
            f.write(f"  random mean  : {r['lcc_random_mean']:.3f}\n")
            f.write(f"  random std   : {r['lcc_random_std']:.3f}\n")
            f.write(f"  z-score      : {r['lcc_z_score']:.3f}\n")
            f.write(f"  p-value      : {r['lcc_p_value']:.6f}\n")
            f.write(f"  significant  : {'YES' if r['lcc_p_value'] < 0.05 else 'NO'}\n\n")


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Multiplex LCC + per-component analysis")
    ap.add_argument("--multiplex-pkl", required=True)
    ap.add_argument("--node-registry", required=True)
    ap.add_argument("--seed-genes", required=True)
    ap.add_argument("--mode", choices=["per-layer", "flattened"], default="per-layer")
    ap.add_argument("--edge-types", nargs="+", default=["ppi"],
                    choices=["ppi", "pathway", "ligand_receptor"])
    ap.add_argument("--degree-scope", choices=["layer", "multiplex"], default="layer")
    ap.add_argument("--layers", nargs="+", default=None,
                    help="Whitelist as 'source:cell_type' (per-layer mode)")
    ap.add_argument("--num-bins", type=int, default=10)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--output-dir", default="output/multiplex_lcc")
    ap.add_argument("--plots-dir", default="plots/multiplex_lcc")
    ap.add_argument("--per-component-metrics", action="store_true",
                    help="Compute per-component metrics (entropy, KS, sizes, "
                         "density, diameter). Skipped automatically when "
                         "--edge-types is exactly ['ppi'].")
    ap.add_argument("--max-components-per-graph", type=int, default=5000)
    args = ap.parse_args()

    t0 = time.time()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    G_full, _ = load_multiplex(args.multiplex_pkl, args.node_registry)
    seed_symbols = load_seed_genes(args.seed_genes)

    edge_types = set(args.edge_types)
    print(f"\nEdge-type filter: {sorted(edge_types)}")
    print(f"Mode            : {args.mode}")
    print(f"Degree scope    : {args.degree_scope}\n")

    do_components = args.per_component_metrics
    if do_components and edge_types == {"ppi"}:
        print("WARNING: --per-component-metrics requested with PPI-only edges. "
              "Skipping per-component analysis (already done in the PPI script).")
        do_components = False

    multiplex_pool = G_full
    results = []

    if args.mode == "flattened":
        eval_graph = filter_edges_by_type(G_full, edge_types)
        pool_graph = multiplex_pool if args.degree_scope == "multiplex" else eval_graph
        name = f"flattened__{'+'.join(sorted(edge_types))}"
        results.append(analyze_graph(name, eval_graph, pool_graph, seed_symbols,
                                     num_bins=args.num_bins,
                                     num_simulations=args.num_simulations))
    else:
        layer_keys = sorted({(d["source"], d["cell_type"])
                             for _, d in G_full.nodes(data=True)})
        if args.layers:
            wanted = {tuple(s.split(":", 1)) for s in args.layers}
            layer_keys = [lk for lk in layer_keys if lk in wanted]
        print(f"Analysing {len(layer_keys)} layer(s)")

        for src, ct in layer_keys:
            try:
                eg = build_layer_graph(G_full, src, ct, edge_types)
                if eg.number_of_nodes() == 0:
                    print(f"  Skipping {src}/{ct}: empty after filtering")
                    continue
                pg = multiplex_pool if args.degree_scope == "multiplex" else eg
                name = f"{src}__{ct}"
                results.append(analyze_graph(name, eg, pg, seed_symbols,
                                             num_bins=args.num_bins,
                                             num_simulations=args.num_simulations))
                gc.collect(); print_memory_usage()
            except Exception as e:
                print(f"Error on layer {src}/{ct}: {e}")
                traceback.print_exc()

    if not results:
        print("No results produced.")
        return

    write_summary(results, args.output_dir)
    plot_lcc_distributions(results, args.plots_dir)

    # ---- Per-component metrics ----
    if do_components:
        print("\n" + "=" * 60)
        print("Computing per-component metrics")
        print("=" * 60)
        comp_out_dir = Path(args.output_dir) / "components"
        comp_plots_dir = Path(args.plots_dir) / "components"

        per_comp = {}
        for r in results:
            per_comp[r["name"]] = compute_per_component_metrics(
                r["name"], r["eval_graph"], r["seed_nodes"],
                max_components=args.max_components_per_graph)

        save_per_component_outputs(per_comp, comp_out_dir)
        plot_per_component(per_comp, comp_plots_dir)
        statistical_comparison_components(
            per_comp, comp_out_dir / "component_statistical_tests.txt")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.setrecursionlimit(40000)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()