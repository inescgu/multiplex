#!/usr/bin/env python
"""
step14_permutation_test_multiplex.py

Counterfactual permutation test for MULTIPLEX disease module validation.
Generates degree-matched random seed sets, runs MultiXrank-style RWR, 
builds null modules, and compares against the true disease module.
"""
import os
import sys
import argparse
import pickle
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Metric functions (adapted for multiplex)
# ─────────────────────────────────────────────

def edge_density(G, module):
    """Edge density in the aggregated module subgraph."""
    sg = G.subgraph(module)
    n = sg.number_of_nodes()
    e = sg.number_of_edges()
    if n < 2:
        return 0.0
    return e / (n * (n - 1) / 2)

def mean_internal_degree(G, module):
    """Mean degree in aggregated subgraph."""
    sg = G.subgraph(module)
    if sg.number_of_nodes() == 0:
        return 0.0
    return np.mean([d for _, d in sg.degree()])

def conductance_fast(G, module, total_vol):
    """Conductance in aggregated graph."""
    module_set = set(module)
    cut = sum(1 for u in module_set for v in G.neighbors(u) if v not in module_set)
    vol_s = sum(G.degree(u) for u in module_set)
    vol_comp = total_vol - vol_s
    denom = min(vol_s, vol_comp)
    return cut / denom if denom > 0 else 0.0

def mean_rwr_score(module, gene_probs):
    """Mean RWR score across module."""
    scores = [gene_probs.get(g, 0.0) for g in module]
    return np.mean(scores) if scores else 0.0

def jaccard(set_a, set_b):
    """Jaccard similarity."""
    a, b = set(set_a), set(set_b)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0

def compute_all_metrics_fast(G, module, gene_probs, total_vol):
    """Compute all metrics from aggregated graph."""
    module_set = set(module)
    sg = G.subgraph(module)
    n  = sg.number_of_nodes()
    e  = sg.number_of_edges()
    
    # Edge density
    ed = e / (n * (n - 1) / 2) if n >= 2 else 0.0
    
    # Mean internal degree
    degrees = [d for _, d in sg.degree()]
    mid = np.mean(degrees) if degrees else 0.0
    
    # Conductance
    vol_s   = sum(G.degree(u) for u in module_set)
    cut     = sum(1 for u in module_set
                  for v in G.neighbors(u) if v not in module_set)
    vol_comp = total_vol - vol_s
    denom    = min(vol_s, vol_comp)
    cond     = cut / denom if denom > 0 else 0.0
    
    # Mean RWR score
    rwr = np.mean([gene_probs.get(g, 0.0) for g in module])
    
    return {
        "size":              n,
        "edge_density":      ed,
        "mean_internal_deg": mid,
        "conductance":       cond,
        "mean_rwr_score":    rwr,
    }

# ─────────────────────────────────────────────
# Degree-matched random seed sampling
# ─────────────────────────────────────────────

def build_degree_bins(G, bin_size=5):
    """Bin all graph nodes by degree in aggregated graph."""
    bins = defaultdict(list)
    for node, deg in G.degree():
        bin_key = (deg // bin_size) * bin_size
        bins[bin_key].append(node)
    return bins

def sample_degree_matched_seeds(true_seeds_in_graph, G, degree_bins, rng, bin_size=5):
    """Sample random nodes matched to seed degrees."""
    sampled = set()
    for seed in true_seeds_in_graph:
        deg = G.degree(seed)
        bin_key = (deg // bin_size) * bin_size
        candidates = [n for n in degree_bins[bin_key]
                      if n not in sampled and n not in true_seeds_in_graph]
        if not candidates:
            for offset in [bin_size, -bin_size, 2*bin_size, -2*bin_size]:
                candidates = [n for n in degree_bins.get(bin_key + offset, [])
                              if n not in sampled and n not in true_seeds_in_graph]
                if candidates:
                    break
        if candidates:
            chosen = rng.choice(candidates)
            sampled.add(chosen)
    return list(sampled)

# ─────────────────────────────────────────────
# RWR on aggregated/supra graph
# ─────────────────────────────────────────────

def run_rwr_aggregated(G_agg, seed_genes, W, scaling_matrix, scaling=0, k=200):
    """
    Run RWR on the aggregated graph (union of all layers).
    
    Args:
        G_agg: Aggregated multiplex graph (nodes are genes)
        seed_genes: List of seed gene symbols
        W: Transition matrix for aggregated graph
        scaling_matrix: Diagonal scaling matrix (if needed)
        scaling: Whether to apply scaling (0 or 1)
        k: Number of non-seed genes to add to module
    
    Returns:
        module: List of genes in final module
        gene_probs: Dict mapping gene -> RWR probability
    """
    # Map seeds to indices
    node_list = list(G_agg.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    
    seeds_in_graph = [g for g in seed_genes if g in G_agg.nodes()]
    if not seeds_in_graph:
        return None, {}
    
    n = len(node_list)
    p0 = np.zeros(n)
    
    for gene in seeds_in_graph:
        if scaling == 1:
            deg = G_agg.degree(gene)
            p0[node_to_idx[gene]] = np.sqrt(deg) if deg > 0 else 1.0
        else:
            p0[node_to_idx[gene]] = 1.0
    
    # RWR computation
    pinf = W @ p0
    if scaling == 1:
        pinf = scaling_matrix * pinf
    
    n_seeds = len(seeds_in_graph)
    
    # Build ranking
    ranked_idx = np.argsort(pinf)[::-1]
    gene_probs = {}
    ranked_genes = []
    seeds_set = set(seeds_in_graph)
    
    for idx in ranked_idx:
        gene = idx_to_node.get(idx)
        if gene is None:
            continue
        score = pinf[idx] / n_seeds
        gene_probs[gene] = score
        if gene not in seeds_set:
            ranked_genes.append(gene)
    
    # Build module
    module = list(seeds_set) + ranked_genes[:k]
    
    # Ensure connectivity in aggregated graph
    extra_idx = k
    module_set = set(module)
    while extra_idx < len(ranked_genes):
        sg = G_agg.subgraph(module)
        if nx.is_connected(sg):
            break
        batch = ranked_genes[extra_idx: extra_idx + 10]
        if not batch:
            break
        for g in batch:
            if g not in module_set:
                module.append(g)
                module_set.add(g)
        extra_idx += 10
    
    return module, gene_probs

# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_null_distributions(true_metrics, null_metrics_list, pvalues, zscores, output_path):
    """Plot null distributions vs true module metrics."""
    metric_names = list(true_metrics.keys())
    n_metrics = len(metric_names)
    fig = plt.figure(figsize=(5 * n_metrics, 5))
    gs = gridspec.GridSpec(1, n_metrics)
    
    for i, metric in enumerate(metric_names):
        ax = fig.add_subplot(gs[i])
        null_vals = [m[metric] for m in null_metrics_list]
        true_val  = true_metrics[metric]
        
        ax.hist(null_vals, bins=40, color='steelblue', alpha=0.7,
                edgecolor='white', label='Null (random seeds)')
        ax.axvline(true_val, color='crimson', linewidth=2.5,
                   label=f'True module\n(p={pvalues[metric]:.3f}, z={zscores[metric]:.2f})')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count' if i == 0 else '')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Disease Module vs. Degree-Matched Null Distribution (Multiplex)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_path, "permutation_test_null_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved null distribution plot → {out}")

def plot_jaccard_distribution(jaccard_scores, output_path):
    """Plot Jaccard similarity distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(jaccard_scores, bins=40, color='mediumseagreen', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Jaccard Similarity (random module vs. true module)')
    ax.set_ylabel('Count')
    ax.set_title('Overlap of Random Modules with True Disease Module')
    ax.axvline(np.mean(jaccard_scores), color='navy', linestyle='--',
               label=f'Mean = {np.mean(jaccard_scores):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_path, "permutation_test_jaccard.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Jaccard distribution plot → {out}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='step14: Permutation test for multiplex disease module validation')
    parser.add_argument('--disease-module-dir',  type=str,   required=True,
                        help='Directory containing disease module (from step9_build_disease_module.py)')
    parser.add_argument('--restart',             type=float, default=0.7,
                        help='Restart probability for RWR (default: 0.7)')
    parser.add_argument('--scaling',             type=int,   default=0, choices=[0, 1],
                        help='Apply degree scaling in RWR (default: 0)')
    parser.add_argument('--k',                   type=int,   default=200,
                        help='Number of non-seed genes to add (default: 200)')
    parser.add_argument('--n-perms',             type=int,   default=1000,
                        help='Number of permutations (default: 1000)')
    parser.add_argument('--bin-size',            type=int,   default=5,
                        help='Degree bin width for matching (default: 5)')
    parser.add_argument('--seed',                type=int,   default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir',          type=str,   default='06_disease_module',
                        help='Output directory for results (default: 06_disease_module)')
    
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    
    dm_dir = Path(args.disease_module_dir)
    output_path = Path(args.output_dir) / "permutation_test"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("step14: Permutation Test for Multiplex Disease Module")
    print(f"  Disease module dir: {args.disease_module_dir}")
    print(f"  N permutations: {args.n_perms} | k: {args.k}")
    print(f"  Restart: {args.restart} | Scaling: {args.scaling}")
    print(f"{'='*70}\n")
    
    # ── Load multiplex graph ───────────────────────────────────────────────
    print("Loading multiplex graph...")
    module_graph_path = dm_dir / "module_graph.pkl"
    if not module_graph_path.exists():
        print(f"ERROR: {module_graph_path} not found")
        return 1
    
    with open(module_graph_path, 'rb') as f:
        H_multiplex = pickle.load(f)
    
    # For aggregated metrics, create aggregate projection
    G_agg = nx.Graph()
    G_agg.add_nodes_from(H_multiplex.nodes())
    for u, v in H_multiplex.edges():
        G_agg.add_edge(u, v)
    
    print(f"  Multiplex graph: {H_multiplex.number_of_nodes():,} nodes, "
          f"{H_multiplex.number_of_edges():,} edges")
    print(f"  Aggregated graph: {G_agg.number_of_nodes():,} nodes, "
          f"{G_agg.number_of_edges():,} edges")
    
    # ── Load or compute transition matrix ───────────────────────────────────
    W_path = dm_dir / f"transition_matrix_restart_{args.restart}.npy"
    if W_path.exists():
        print(f"Loading transition matrix from {W_path}...")
        W = np.load(W_path)
    else:
        print(f"Computing transition matrix for restart={args.restart}...")
        try:
            from scipy.sparse import csr_matrix, diags
        except ImportError:
            import numpy as np
        
        A = nx.to_scipy_sparse_array(G_agg, format='csr')
        degrees = np.array([G_agg.degree(n) for n in G_agg.nodes()])
        D_inv = diags(1.0 / np.maximum(degrees, 1))
        P = D_inv @ A
        n = G_agg.number_of_nodes()
        I = np.eye(n, format='csr')
        W = (1 - args.restart) * P + args.restart / n * np.ones((n, n))
        W = W.astype(np.float32)
    
    # Scaling matrix
    scaling_matrix = None
    if args.scaling == 1:
        degrees = np.array([G_agg.degree(n) for n in G_agg.nodes()])
        scaling_matrix = 1.0 / np.sqrt(np.maximum(degrees, 1))
    
    # ── Load true disease module ───────────────────────────────────────────
    print("\nLoading true disease module...")
    module_csv_path = dm_dir / "disease_module.csv"
    if not module_csv_path.exists():
        print(f"ERROR: {module_csv_path} not found")
        return 1
    
    module_df = pd.read_csv(module_csv_path)
    true_module = module_df['gene'].tolist()
    true_seeds = module_df[module_df['is_seed'] == True]['gene'].tolist()
    
    print(f"  True module size: {len(true_module)}")
    print(f"  True seeds in module: {len(true_seeds)}")
    
    # Build gene_probs dict for the true module
    true_gene_probs = {}
    if 'rwr_score' in module_df.columns:
        for _, row in module_df.iterrows():
            true_gene_probs[row['gene']] = row['rwr_score']
    else:
        for gene in true_module:
            true_gene_probs[gene] = 1.0 / len(true_module)
    
    # ── Compute true metrics ───────────────────────────────────────────────
    total_vol = sum(d for _, d in G_agg.degree())
    true_metrics = compute_all_metrics_fast(G_agg, true_module, true_gene_probs, total_vol)
    
    print(f"\nTrue module metrics:")
    for k_m, v in true_metrics.items():
        print(f"  {k_m:25s}: {v:.6f}")
    
    # ── Build degree bins ──────────────────────────────────────────────────
    print(f"\nBuilding degree bins (bin_size={args.bin_size})...")
    degree_bins = build_degree_bins(G_agg, bin_size=args.bin_size)
    print(f"  Created {len(degree_bins)} degree bins")
    
    # ── Permutation loop ───────────────────────────────────────────────────
    null_metrics_list = []
    jaccard_scores    = []
    failed_perms      = 0
    
    print(f"\nRunning {args.n_perms} permutations...")
    t0 = time.time()
    
    for i in range(args.n_perms):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (args.n_perms - i - 1) / rate
            print(f"  Permutation {i+1}/{args.n_perms} | "
                  f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s")
        
        # Sample degree-matched random seeds
        rand_seeds = sample_degree_matched_seeds(
            true_seeds, G_agg, degree_bins, rng, bin_size=args.bin_size)
        
        if len(rand_seeds) < max(1, len(true_seeds) // 2):
            failed_perms += 1
            continue
        
        # Run RWR with random seeds
        rand_module, rand_probs = run_rwr_aggregated(
            G_agg, rand_seeds, W, scaling_matrix,
            scaling=args.scaling, k=args.k)
        
        if rand_module is None:
            failed_perms += 1
            continue
        
        # Compute metrics
        null_metrics_list.append(
            compute_all_metrics_fast(G_agg, rand_module, rand_probs, total_vol))
        jaccard_scores.append(jaccard(true_module, rand_module))
    
    print(f"\nCompleted {len(null_metrics_list)} valid permutations "
          f"({failed_perms} failed/skipped)")
    
    if len(null_metrics_list) == 0:
        print("ERROR: No valid permutations completed")
        return 1
    
    # ── Compute p-values and z-scores ──────────────────────────────────────
    higher_is_better = {"edge_density", "mean_internal_deg", "mean_rwr_score"}
    lower_is_better  = {"conductance", "size"}
    
    pvalues = {}
    zscores = {}
    
    for metric in true_metrics:
        null_vals  = np.array([m[metric] for m in null_metrics_list])
        true_val   = true_metrics[metric]
        mu, sigma  = null_vals.mean(), null_vals.std()
        zscores[metric] = (true_val - mu) / sigma if sigma > 0 else np.nan
        
        if metric in higher_is_better:
            pvalues[metric] = np.mean(null_vals >= true_val)
        else:
            pvalues[metric] = np.mean(null_vals <= true_val)
    
    # ── Print results table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Metric':<25} {'True':>10} {'Null Mean':>10} {'Null SD':>10} "
          f"{'Z-score':>10} {'p-value':>10}")
    print(f"{'-'*70}")
    
    for metric in true_metrics:
        null_vals = [m[metric] for m in null_metrics_list]
        print(f"{metric:<25} {true_metrics[metric]:>10.4f} "
              f"{np.mean(null_vals):>10.4f} {np.std(null_vals):>10.4f} "
              f"{zscores[metric]:>10.2f} {pvalues[metric]:>10.4f}")
    
    print(f"{'='*70}")
    print(f"\nJaccard similarity (true vs random modules):")
    print(f"  Mean:   {np.mean(jaccard_scores):.4f}")
    print(f"  Median: {np.median(jaccard_scores):.4f}")
    print(f"  Max:    {np.max(jaccard_scores):.4f}")
    
    # ── Save results ───────────────────────────────────────────────────────
    results_df = pd.DataFrame({
        'metric':     list(true_metrics.keys()),
        'true_value': [true_metrics[m] for m in true_metrics],
        'null_mean':  [np.mean([x[m] for x in null_metrics_list]) for m in true_metrics],
        'null_std':   [np.std( [x[m] for x in null_metrics_list]) for m in true_metrics],
        'z_score':    [zscores[m]  for m in true_metrics],
        'p_value':    [pvalues[m]  for m in true_metrics],
    })
    
    csv_out = output_path / "permutation_test_results.csv"
    results_df.to_csv(csv_out, index=False)
    print(f"\nSaved results → {csv_out}")
    
    jaccard_df = pd.DataFrame({'jaccard': jaccard_scores})
    jaccard_df.to_csv(output_path / "jaccard_scores.csv", index=False)
    print(f"Saved Jaccard scores → {output_path / 'jaccard_scores.csv'}")
    
    null_df = pd.DataFrame(null_metrics_list)
    null_df.to_csv(output_path / "null_module_metrics.csv", index=False)
    print(f"Saved null module metrics → {output_path / 'null_module_metrics.csv'}")
    
    # ── Plots ──────────────────────────────────────────────────────────────
    plot_null_distributions(true_metrics, null_metrics_list, pvalues, zscores, str(output_path))
    plot_jaccard_distribution(jaccard_scores, str(output_path))
    
    print(f"\n{'='*70}")
    print(f"All outputs saved to: {output_path}")
    print(f"{'='*70}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())