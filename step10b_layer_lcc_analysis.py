#!/usr/bin/env python3
"""
Multiplex Layer LCC Significance Analysis
==========================================
Analyzes each layer of your multiplex to identify which layers have
significant largest connected components (LCC) as a proxy for structural
importance. Uses degree-normalized null model.
Pipeline integration:
  - Loads integrated_multiplex.pkl + node_registry.csv
  - Analyzes per-layer (source + cell_type) LCC significance
  - Reports which layers are structurally important
  - Per-component metrics for understanding layer organization
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
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed  # ← NEW
import multiprocessing as mp  # ← NEW
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
        print(f"  Memory: {p.memory_info().rss / (1024**3):.2f} GB")
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
    """Return size of largest connected component."""
    if G.number_of_nodes() == 0:
        return 0
    return max((len(c) for c in nx.connected_components(G)), default=0)

def getDict_degree2nodeNames(G, nodeList=None, numBins=None, binLimit=None):
    """Map degree → [node names] with optional binning."""
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
    """Sample nodes from pool preserving degree distribution of target."""
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

# ============================================================================
# NEW: ADAPTIVE SIGNIFICANCE TESTING
# ============================================================================
def sig_lccOfSubgraph_adaptive(eval_graph, pool_graph, seed_nodes,
                               numBins=10, numSimulations=1000, desc=""):
    """
    Adaptive significance testing with early stopping.
    - First 200 simulations
    - If p > 0.15, stop (unlikely to be significant)
    - Otherwise continue to full numSimulations
    
    Returns dict with:
      lcc, pVal, zScore, simResults, mean, std, n_seeds_in_graph
    """
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
    
    # Degree distribution of seed nodes in pool
    target_dist = getDict_degree2numNodes(pool_graph, nodeList=seeds_in_pool,
                                          numBins=numBins)
    obs_lcc = size_lcc(eval_graph.subgraph(seed_nodes))
    
    sim_sizes = []
    
    # FIRST PASS: 200 simulations (quick screen)
    for _ in tqdm(range(200), desc=desc or "LCC sims (phase 1)", leave=False):
        gsim = randSubgraph_preserveDegree(pool_graph, target_dist, eval_graph)
        sim_sizes.append(size_lcc(gsim))
    
    # Check interim p-value
    arr_interim = np.array(sim_sizes)
    p_interim = float(np.mean(arr_interim >= obs_lcc))
    
    # EARLY STOPPING: if clearly non-significant, don't waste compute
    if p_interim > 0.15:
        arr = arr_interim
    else:
        # SECOND PASS: continue to full simulations
        for _ in tqdm(range(numSimulations - 200), desc=desc or "LCC sims (phase 2)", leave=False):
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
# PER-COMPONENT METRICS
# ============================================================================
def _component_metrics(component_nodes, graph, seed_node_set, component_id):
    """Compute per-component statistics."""
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
    sub = graph.subgraph(nodes)
    n_edges = sub.number_of_edges()
    density = (2 * n_edges) / (n * (n - 1)) if n > 1 else 0.0
    diameter = 0
    if seed_count > 0 and n < 1000:
        try:
            diameter = nx.diameter(sub) if nx.is_connected(sub) else float("inf")
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

def compute_per_component_metrics(layer_name, eval_graph, seed_nodes,
                                  max_components=5000):
    """Compute per-component metrics on eval_graph."""
    if eval_graph.number_of_nodes() == 0:
        return None
    seed_node_set = set(seed_nodes)
    components = list(nx.connected_components(eval_graph))
    components.sort(key=len, reverse=True)
    component_sizes = [len(c) for c in components]
    singleton_count = sum(1 for s in component_sizes if s == 1)
    components_for_metrics = (components[:max_components]
                              if len(components) > max_components
                              else components)
    rows = []
    seed_rows = []
    for i, comp in enumerate(components_for_metrics):
        m = _component_metrics(comp, eval_graph, seed_node_set, component_id=i)
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

# ============================================================================
# LOADING
# ============================================================================
def load_multiplex(graph_path, registry_path):
    """Load integrated multiplex and node registry."""
    print(f"Loading multiplex from {graph_path} ...")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"Loading node registry from {registry_path} ...")
    reg = pd.read_csv(registry_path, index_col="node_id")
    print(f"  {len(reg):,} registry entries\n")
    return G, reg

def load_seed_genes(path):
    """Load seed genes from CSV."""
    df = pd.read_csv(path)
    if "gene" in df.columns:
        seeds = set(df["gene"].astype(str).str.upper().str.strip())
    else:
        seeds = set(df.iloc[:, 0].astype(str).str.upper().str.strip())
    seeds.discard("NAN")
    seeds.discard("")
    print(f"Loaded {len(seeds):,} seed gene symbols\n")
    return seeds

# ============================================================================
# LAYER EXTRACTION
# ============================================================================
def extract_layer(G, source, cell_type):
    """Extract subgraph for specific source/cell_type layer."""
    layer_nodes = [n for n, d in G.nodes(data=True)
                   if d.get("source") == source and d.get("cell_type") == cell_type]
    if not layer_nodes:
        return nx.Graph()
    return G.subgraph(layer_nodes).copy()

def get_all_layers(G):
    """Get all unique (source, cell_type) pairs."""
    layers = set()
    for n, d in G.nodes(data=True):
        src = d.get("source")
        ct = d.get("cell_type")
        if src and ct:
            layers.add((src, ct))
    return sorted(layers)

def map_seeds_to_node_ids(G, seed_symbols, restrict_nodes=None):
    """Map seed symbols to node IDs in graph."""
    nodes_iter = restrict_nodes if restrict_nodes is not None else G.nodes()
    out = []
    for n in nodes_iter:
        g = G.nodes[n].get("gene")
        if g and str(g).upper().strip() in seed_symbols:
            out.append(n)
    return out

# ============================================================================
# ANALYSIS
# ============================================================================
def analyze_layer(G_full, layer_nodes, layer_name, seed_symbols,
                  num_bins=10, num_simulations=1000):
    """Analyze a single layer for LCC significance."""
    if not layer_nodes or len(layer_nodes) == 0:
        return None

    eval_graph = G_full.subgraph(layer_nodes).copy()
    if eval_graph.number_of_nodes() == 0:
        return None

    # Use full multiplex as pool for degree distribution
    pool_graph = G_full

    # Map seeds to node IDs in this layer
    seed_nodes = map_seeds_to_node_ids(eval_graph, seed_symbols)

    if not seed_nodes:
        return None

    print(f"\n  {layer_name}")
    print(f"    nodes: {eval_graph.number_of_nodes():,} | "
          f"edges: {eval_graph.number_of_edges():,} | "
          f"seeds: {len(seed_nodes)}")

    # LCC significance (UPDATED: uses adaptive version)
    sig = sig_lccOfSubgraph_adaptive(eval_graph, pool_graph, seed_nodes,
                                     numBins=num_bins, numSimulations=num_simulations,
                                     desc=f"LCC [{layer_name}]")

    print(f"    LCC: {sig['lcc']} | p={sig['pVal']:.4g} | "
          f"z={sig['zScore']:.2f} | "
          f"significant: {'YES ✓' if sig['pVal'] < 0.05 else 'no'}")

    return {
        "layer": layer_name,
        "source": layer_name.split("__")[0],
        "cell_type": layer_name.split("__")[1] if "__" in layer_name else "",
        "n_nodes": eval_graph.number_of_nodes(),
        "n_edges": eval_graph.number_of_edges(),
        "n_seeds": sig["n_seeds_in_graph"],
        "lcc_size": sig["lcc"],
        "lcc_p_value": sig["pVal"],
        "lcc_z_score": sig["zScore"],
        "lcc_random_mean": sig["mean"],
        "lcc_random_std": sig["std"],
        "significant_lcc": sig["pVal"] < 0.05,
        "seed_nodes": seed_nodes,
        "eval_graph": eval_graph,
        "simResults": sig["simResults"],
    }

# ============================================================================
# NEW: WRAPPER FOR MULTIPROCESSING
# ============================================================================
def analyze_layer_worker(args):
    """
    Wrapper function for multiprocessing.
    Takes tuple of arguments and runs analyze_layer.
    """
    G_full, layer_nodes, layer_name, seed_symbols, num_bins, num_simulations = args
    try:
        return analyze_layer(G_full, layer_nodes, layer_name, seed_symbols,
                            num_bins=num_bins, num_simulations=num_simulations)
    except Exception as e:
        print(f"ERROR on {layer_name}: {e}")
        traceback.print_exc()
        return None

# ============================================================================
# REPORTING
# ============================================================================
def plot_lcc_results(results, plots_dir):
    """Plot LCC sizes and significance across layers."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_sorted = sorted(results, key=lambda r: r["lcc_p_value"])
    names = [r["cell_type"] if r["cell_type"] else r["source"]
             for r in results_sorted]
    lccs = [r["lcc_size"] for r in results_sorted]
    ps = [r["lcc_p_value"] for r in results_sorted]
    sigs = [r["significant_lcc"] for r in results_sorted]

    colors = ["#2ECC71" if s else "#E74C3C" for s in sigs]

    fig, ax1 = plt.subplots(figsize=(max(12, len(names) * 0.5), 6))
    ax2 = ax1.twinx()

    ax1.bar(range(len(names)), lccs, color=colors, alpha=0.7, label="LCC size")
    ax1.set_ylabel("LCC size", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(lccs) * 1.2 if lccs else 1)

    ax2.plot(range(len(names)), ps, "o-", color="#3498DB", lw=2,
             markersize=6, label="p-value")
    ax2.axhline(0.05, color="#E74C3C", ls="--", lw=2, alpha=0.7,
                label="p=0.05 threshold")
    ax2.set_ylabel("p-value (degree-normalized)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, min(1.0, max(ps) * 1.5 if ps else 0.1))

    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax1.set_xlabel("Layer (cell type)", fontsize=12, fontweight="bold")
    ax1.set_title("LCC Significance by Layer (Structural Importance)",
                  fontsize=14, fontweight="bold")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "lcc_significance_by_layer.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot → {plots_dir / 'lcc_significance_by_layer.png'}")

def write_summary(results, out_dir):
    """Write LCC analysis summary to CSV and TXT."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    rows = [{k: v for k, v in r.items()
             if k not in ("seed_nodes", "eval_graph", "simResults")}
            for r in results]
    df = pd.DataFrame(rows)
    df = df.sort_values("lcc_p_value")
    df.to_csv(out_dir / "lcc_summary.csv", index=False)

    # TXT report
    with open(out_dir / "lcc_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MULTIPLEX LAYER LCC SIGNIFICANCE ANALYSIS (DEGREE-NORMALIZED)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Interpretation:\n")
        f.write("  - LCC: Largest connected component size in seed subgraph\n")
        f.write("  - p-value < 0.05: Layer has SIGNIFICANT structural role\n")
        f.write("  - z-score: How many std devs above random (degree-normalized)\n")
        f.write("  - High LCC + low p-value = layer is structurally important\n\n")

        sig_layers = df[df["significant_lcc"] == True]
        f.write(f"SIGNIFICANT LAYERS (p < 0.05): {len(sig_layers)} / {len(df)}\n")
        f.write("-" * 80 + "\n")
        for _, r in sig_layers.iterrows():
            f.write(f"  [{r['layer']}]\n")
            f.write(f"    Nodes: {r['n_nodes']:,} | Edges: {r['n_edges']:,} | "
                    f"Seeds: {r['n_seeds']}\n")
            f.write(f"    LCC: {r['lcc_size']} (z={r['lcc_z_score']:.2f}, "
                    f"p={r['lcc_p_value']:.6f})\n")
            f.write(f"    Random mean ± std: {r['lcc_random_mean']:.1f} ± {r['lcc_random_std']:.1f}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"NON-SIGNIFICANT LAYERS (p ≥ 0.05): {len(df) - len(sig_layers)} / {len(df)}\n")
        f.write("-" * 80 + "\n")
        nonsig = df[df["significant_lcc"] == False]
        for _, r in nonsig.iterrows():
            f.write(f"  [{r['layer']}] LCC={r['lcc_size']}, p={r['lcc_p_value']:.6f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total layers analyzed: {len(df)}\n")
        f.write(f"Mean LCC size: {df['lcc_size'].mean():.1f}\n")
        f.write(f"Median LCC size: {df['lcc_size'].median():.1f}\n")
        f.write(f"Max LCC: {df['lcc_size'].max()}\n")
        f.write(f"Layers with LCC > 0: {(df['lcc_size'] > 0).sum()}\n")

    print(f"Saved summary → {out_dir / 'lcc_summary.csv'}")
    print(f"Saved report → {out_dir / 'lcc_summary.txt'}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Analyze layer LCC significance in integrated multiplex"
    )
    ap.add_argument("--multiplex-pkl", required=True,
                    help="Path to integrated_multiplex.pkl")
    ap.add_argument("--node-registry", required=True,
                    help="Path to node_registry.csv")
    ap.add_argument("--seed-genes", required=True,
                    help="Path to seed_genes.csv")
    ap.add_argument("--num-bins", type=int, default=10,
                    help="Degree bins for null model (default: 10)")
    ap.add_argument("--num-simulations", type=int, default=1000,
                    help="Number of random simulations (default: 1000)")
    ap.add_argument("--output-dir",
                    default="10b_layer_lcc_analysis",
                    help="Output directory (default: 10b_layer_lcc_analysis)")
    ap.add_argument("--plots-dir",
                    default="10b_layer_lcc_analysis/plots",
                    help="Plots directory (default: 10b_layer_lcc_analysis/plots)")
    ap.add_argument("--per-component", action="store_true",
                    help="Compute per-component metrics (entropy, etc.)")
    args = ap.parse_args()

    t0 = time.time()

    # Setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    # Load
    print("=" * 80)
    print("MULTIPLEX LAYER LCC SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    G_full, reg = load_multiplex(args.multiplex_pkl, args.node_registry)
    seed_symbols = load_seed_genes(args.seed_genes)

    # Extract layers
    print(f"Extracting layers from multiplex...")
    layers = get_all_layers(G_full)
    print(f"Found {len(layers)} layers\n")

    # ========================================================================
    # NEW: PREPARE TASKS FOR PARALLEL EXECUTION
    # ========================================================================
    print(f"Preparing analysis tasks...")
    tasks = []
    for source, cell_type in layers:
        layer_nodes = [n for n, d in G_full.nodes(data=True)
                      if d.get("source") == source and d.get("cell_type") == cell_type]
        if layer_nodes:
            layer_name = f"{source}__{cell_type}"
            tasks.append((G_full, layer_nodes, layer_name, seed_symbols,
                         args.num_bins, args.num_simulations))

    print(f"Prepared {len(tasks)} tasks\n")

    # ========================================================================
    # NEW: PARALLEL EXECUTION
    # ========================================================================
    print(f"Analyzing LCC significance in parallel:")
    results = []
    
    # Determine number of workers (use all but 1 CPU)
    n_workers = min(max(1, mp.cpu_count() - 1), len(tasks))
    print(f"Using {n_workers} workers\n")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(analyze_layer_worker, task): task[2] 
                  for task in tasks}
        
        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Layers analyzed"):
            result = future.result()
            if result:
                results.append(result)
            gc.collect()

    if not results:
        print("\nNo results produced. Check seed/layer matching.")
        return

    # Report
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    write_summary(results, args.output_dir)
    plot_lcc_results(results, args.plots_dir)

    # ========================================================================
    # UPDATED: Per-component metrics (significant layers only)
    # ========================================================================
    if args.per_component:
        print("\nComputing per-component metrics (significant layers only)...")
        comp_dir = Path(args.output_dir) / "components"
        comp_dir.mkdir(exist_ok=True)

        # Only analyze significant layers + top 3 non-significant for comparison
        sig_results = [r for r in results if r["significant_lcc"]]
        nonsig_sample = [r for r in results if not r["significant_lcc"]][:3]
        
        for r in sig_results + nonsig_sample:
            comp = compute_per_component_metrics(
                r["layer"], r["eval_graph"], r["seed_nodes"]
            )
            if comp and not comp["components_df"].empty:
                comp["components_df"].to_csv(
                    comp_dir / f"components_{r['layer']}.csv", index=False
                )
        
        print(f"Saved per-component data → {comp_dir}")

    print(f"\nCompleted in {time.time() - t0:.1f}s\n")

if __name__ == "__main__":
    sys.setrecursionlimit(40000)
    main()