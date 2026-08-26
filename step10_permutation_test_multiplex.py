#!/usr/bin/env python
"""
step10_permutation_test_multiplex.py (formerly step14_permutation_test_multiplex.py)

Counterfactual permutation test for disease module validation (heterogeneous
pipeline). Generates degree-matched random seed sets, runs a sparse
iterative RWR, builds null modules, and compares against the true disease
module.

UPDATED for the current namespaced heterogeneous pipeline. Two real bugs
fixed, not just a rename:
---------------------------------------------------------------------------
1. NODE-IDENTITY BUG. The previous version read `module_df['gene']` (bare
   gene symbols) as node IDs and compared them directly against
   `module_graph.pkl`'s nodes. In the current pipeline, module_graph.pkl's
   nodes are namespaced "<celltype>::GENE" strings (written by
   step2.5_namespace_and_seeds.py / step6_build_disease_module.py) -- a bare
   gene symbol never matches, and worse, the same bare gene seeded under two
   different cell types would silently collide into one entry. Fixed by
   using disease_module.csv's `node` column (already namespaced) as the
   node identity everywhere, and `is_seed` for the seed set.

2. NULL MODEL WAS SAMPLED FROM THE DISEASE MODULE ITSELF, NOT THE FULL
   NETWORK. The previous version loaded module_graph.pkl -- which, in the
   current pipeline, IS the final disease module's own small induced
   subgraph (seeds + bridging genes + LCC fallback, typically hundreds of
   nodes) -- and used it as BOTH the degree-matching universe for random
   seeds AND the graph conductance/cut metrics were computed against. Two
   consequences: (a) degree-matched random seeds were only ever drawn from
   nodes already inside the module, which isn't a meaningful null
   ("is the module more connected than a random subset of itself?" instead
   of "...than chance across the whole network?"); (b) conductance
   (cut / min(vol_S, vol_complement)) was computed on a subgraph that BY
   CONSTRUCTION has no edges leaving it, so it was always exactly 0 for
   every permutation, true module included -- a non-informative metric.
   Fixed by adding a required `--reference-graph` argument pointing at
   gene_reference_graph.pkl (or gene_ppi_reference_graph.pkl), the full
   namespaced network from step2.5's output. Degree bins, random seed
   sampling, the local RWR, and all cut/conductance/volume metrics now use
   this full graph; `--disease-module-dir` is now used only for the true
   module's own identity (seeds, node list, and step4's actual RWR scores)
   -- not as a stand-in for the whole network.

3. DENSE O(n^2) TRANSITION MATRIX REPLACED WITH SPARSE ITERATIVE RWR.
   The previous version built (or cached) a dense n x n transition matrix
   and did `pinf = W @ p0` as if that were the converged RWR distribution --
   at genome-scale node counts (tens of thousands of genes x dozens of cell
   types) a dense matrix like that is not just slow, it doesn't fit in
   memory. Replaced with a proper sparse power-iteration RWR (same
   restart-probability formulation MultiXrank itself uses), which is the
   only thing that makes fixing bug #2 practically usable -- sampling from
   the full network only helps if you can actually run RWR on it.

True module metrics still trust step6's own RWR scores where available
(disease_module.csv's `rwr_score` column) rather than recomputing them with
this script's simplified single-graph RWR -- MultiXrank's actual multiplex+
bipartite random walk is the authoritative one. Only the NULL permutations
run this script's own (much cheaper, single-graph) RWR, since re-invoking
full MultiXrank a thousand times per run isn't practical.
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
from scipy.sparse import diags
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Node-name parsing (nodes are "<celltype>::GENE")
# ─────────────────────────────────────────────

def _celltype_of(node: str) -> str:
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


def _gene_of(node: str) -> str:
    return node.rsplit("::", 1)[-1] if "::" in node else node


def filter_to_lcc_per_layer(H):
    """Identical to step6_build_disease_module.py's function of the same
    name. If step6 was run with --lcc-per-layer, it built the true module
    from THIS filtered graph, not the raw gene_reference_graph.pkl -- so the
    null-sampling universe here has to be filtered the same way, or degree
    bins/random seeds/conductance-and-volume totals would all be computed
    against a different (larger, unfiltered) graph than the one the true
    module actually came from. Pass --lcc-per-layer here whenever it was
    passed to step6."""
    layers = set()
    for _, _, d in H.edges(data=True):
        layers |= d.get("cell_types", set())

    if not layers:
        print("  [lcc-per-layer] No 'cell_types' edge attribute found — skipping.")
        return H

    kept_nodes = set()
    for layer in sorted(layers):
        layer_edges = [(u, v) for u, v, d in H.edges(data=True)
                        if layer in d.get("cell_types", set())]
        layer_nodes = set(n for e in layer_edges for n in e)
        Lsub = H.subgraph(layer_nodes)
        if Lsub.number_of_nodes() == 0:
            print(f"  [lcc-per-layer] Cell type '{layer}': empty, skipping.")
            continue
        lcc = max(nx.connected_components(Lsub), key=len)
        print(f"  [lcc-per-layer] Cell type '{layer}': "
              f"{Lsub.number_of_nodes():,} -> {len(lcc):,} nodes "
              f"(dropped {Lsub.number_of_nodes() - len(lcc):,})")
        kept_nodes |= lcc

    H_filtered = H.subgraph(kept_nodes).copy()
    print(f"\n  [lcc-per-layer] Total: {H.number_of_nodes():,} -> "
          f"{H_filtered.number_of_nodes():,} nodes, "
          f"{H.number_of_edges():,} -> {H_filtered.number_of_edges():,} edges")
    return H_filtered


# ─────────────────────────────────────────────
# Metric functions (operate on the FULL reference graph now -- see bug #2
# in the module docstring for why that matters for conductance in particular)
# ─────────────────────────────────────────────

def edge_density(G, module):
    """Edge density among `module`'s nodes, using edges as they exist in G."""
    sg = G.subgraph(module)
    n = sg.number_of_nodes()
    e = sg.number_of_edges()
    if n < 2:
        return 0.0
    return e / (n * (n - 1) / 2)

def mean_internal_degree(G, module):
    """Mean degree in the induced subgraph (internal edges only)."""
    sg = G.subgraph(module)
    if sg.number_of_nodes() == 0:
        return 0.0
    return np.mean([d for _, d in sg.degree()])

def conductance_fast(G, module, total_vol):
    """Conductance relative to the FULL graph G (cut / min(vol_S, vol_comp)).
    Meaningful only when G is the full reference network, not the module's
    own induced subgraph -- see bug #2 above."""
    module_set = set(module)
    cut = sum(1 for u in module_set for v in G.neighbors(u) if v not in module_set)
    vol_s = sum(G.degree(u) for u in module_set)
    vol_comp = total_vol - vol_s
    denom = min(vol_s, vol_comp)
    return cut / denom if denom > 0 else 0.0

def jaccard(set_a, set_b):
    """Jaccard similarity."""
    a, b = set(set_a), set(set_b)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0

def compute_all_metrics_fast(G, module, gene_probs, total_vol):
    """Compute all metrics against the FULL reference graph G."""
    module_set = set(module)
    sg = G.subgraph(module)
    n  = sg.number_of_nodes()
    e  = sg.number_of_edges()

    ed = e / (n * (n - 1) / 2) if n >= 2 else 0.0

    degrees = [d for _, d in sg.degree()]
    mid = np.mean(degrees) if degrees else 0.0

    vol_s   = sum(G.degree(u) for u in module_set)
    cut     = sum(1 for u in module_set
                  for v in G.neighbors(u) if v not in module_set)
    vol_comp = total_vol - vol_s
    denom    = min(vol_s, vol_comp)
    cond     = cut / denom if denom > 0 else 0.0

    rwr = np.mean([gene_probs.get(g, 0.0) for g in module])

    return {
        "size":              n,
        "edge_density":      ed,
        "mean_internal_deg": mid,
        "conductance":       cond,
        "mean_rwr_score":    rwr,
    }

# ─────────────────────────────────────────────
# Degree-matched random seed sampling (now against the full reference graph)
# ─────────────────────────────────────────────

def build_degree_bins(G, bin_size=5):
    """Bin all graph nodes by degree."""
    bins = defaultdict(list)
    for node, deg in G.degree():
        bin_key = (deg // bin_size) * bin_size
        bins[bin_key].append(node)
    return bins

def sample_degree_matched_seeds(true_seeds_in_graph, G, degree_bins, rng, bin_size=5):
    """Sample random nodes matched to seed degrees, drawn from the FULL
    reference graph (not from within the module itself)."""
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
# Sparse iterative RWR on the full reference graph
# ─────────────────────────────────────────────

def build_transition_matrix(G, nodelist):
    """Column-stochastic sparse transition matrix M (each column sums to 1,
    or 0 for isolated nodes): M[:, i] is where a random walker at node i
    goes next. p_{t+1} = (1-restart)*M @ p_t + restart*p0 is the standard
    personalized-PageRank / RWR power-iteration update."""
    idx_of = {n: i for i, n in enumerate(nodelist)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, format="csr", dtype=np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_safe = np.where(deg == 0, 1.0, deg)
    D_inv = diags(1.0 / deg_safe)
    M = (A @ D_inv).tocsr()  # column-stochastic
    return M, idx_of, deg


def rwr_power_iteration(M, p0, restart, max_iter=200, tol=1e-9):
    p = p0.copy()
    for _ in range(max_iter):
        p_new = (1 - restart) * (M @ p) + restart * p0
        if np.abs(p_new - p).sum() < tol:
            return p_new
        p = p_new
    return p


def run_rwr_aggregated(G_ref, nodelist, idx_of, M, deg, seed_nodes, restart,
                        scaling=0, k=200):
    """
    Run sparse RWR on the full reference graph.

    Args:
        G_ref: full reference graph (namespaced "<celltype>::GENE" nodes)
        nodelist, idx_of, M, deg: from build_transition_matrix()
        seed_nodes: list of namespaced seed node IDs
        restart: RWR restart probability
        scaling: if 1, seed initial mass scaled by sqrt(degree)
        k: number of non-seed nodes to add to the module

    Returns:
        module: list of nodes in the final (connectivity-padded) module
        node_probs: dict node -> RWR probability
    """
    n = len(nodelist)
    seeds_in_graph = [g for g in seed_nodes if g in idx_of]
    if not seeds_in_graph:
        return None, {}

    p0 = np.zeros(n)
    for node in seeds_in_graph:
        i = idx_of[node]
        p0[i] = np.sqrt(deg[i]) if (scaling == 1 and deg[i] > 0) else 1.0
    p0 = p0 / p0.sum()

    pinf = rwr_power_iteration(M, p0, restart)

    ranked_idx = np.argsort(pinf)[::-1]
    seeds_set = set(seeds_in_graph)
    node_probs = {}
    ranked_nodes = []
    for i in ranked_idx:
        node = nodelist[i]
        node_probs[node] = float(pinf[i])
        if node not in seeds_set:
            ranked_nodes.append(node)

    module = list(seeds_set) + ranked_nodes[:k]

    # Ensure connectivity within the full reference graph
    extra_idx = k
    module_set = set(module)
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

    return module, node_probs

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
    
    plt.suptitle('Disease Module vs. Degree-Matched Null Distribution (Reference Network)', 
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
        description='step14: Permutation test for disease module validation (heterogeneous pipeline)')
    parser.add_argument('--disease-module-dir',  type=str,   required=True,
                        help='Directory containing disease_module.csv (from '
                             'step6_build_disease_module.py) -- used for the TRUE module\'s '
                             'identity (seeds, nodes, step4\'s own RWR scores) only')
    parser.add_argument('--reference-graph',     type=str,   required=True,
                        help='gene_reference_graph.pkl (or gene_ppi_reference_graph.pkl) from '
                             'step2.5_namespace_and_seeds.py\'s output -- the FULL namespaced '
                             'network used as the null-sampling universe and for all cut/'
                             'conductance/volume metrics (see bug #2 in module docstring)')
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
    parser.add_argument('--lcc-per-layer',       action='store_true',
                        help='Apply the same per-cell-type LCC filter to --reference-graph '
                             'that step6_build_disease_module.py applied before building the '
                             'true module. REQUIRED here if step6 was run with --lcc-per-layer '
                             '-- otherwise the null-sampling universe won\'t match the graph '
                             'space the true module actually came from.')
    
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    
    dm_dir = Path(args.disease_module_dir)
    output_path = Path(args.output_dir) / "permutation_test"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("step14: Permutation Test for Disease Module (heterogeneous pipeline)")
    print(f"  Disease module dir: {args.disease_module_dir}")
    print(f"  Reference graph   : {args.reference_graph}")
    print(f"  N permutations: {args.n_perms} | k: {args.k}")
    print(f"  Restart: {args.restart} | Scaling: {args.scaling}")
    print(f"{'='*70}\n")
    
    # ── Load FULL reference graph (null-sampling universe) ────────────────
    print("Loading full reference graph...")
    with open(args.reference_graph, 'rb') as f:
        G_ref = pickle.load(f)
    G_ref = nx.relabel_nodes(G_ref, {n: str(n).strip() for n in G_ref.nodes()})
    print(f"  Reference graph: {G_ref.number_of_nodes():,} nodes, "
          f"{G_ref.number_of_edges():,} edges")

    if args.lcc_per_layer:
        print("\nApplying --lcc-per-layer filter (matching step6) ...")
        G_ref = filter_to_lcc_per_layer(G_ref)

    nodelist = sorted(G_ref.nodes())
    
    # ── Build sparse transition matrix ─────────────────────────────────────
    print("\nBuilding sparse transition matrix...")
    M, idx_of, deg = build_transition_matrix(G_ref, nodelist)
    print(f"  Transition matrix: {M.shape[0]:,} x {M.shape[1]:,}, nnz={M.nnz:,}")
    
    # ── Load true disease module (namespaced 'node' column, NOT bare 'gene') ──
    print("\nLoading true disease module...")
    module_csv_path = dm_dir / "disease_module.csv"
    if not module_csv_path.exists():
        print(f"ERROR: {module_csv_path} not found")
        return 1
    
    module_df = pd.read_csv(module_csv_path)
    if "node" not in module_df.columns:
        print(f"ERROR: {module_csv_path} has no 'node' column -- this script requires the "
              f"namespaced disease_module.csv written by the current step6_build_disease_module.py")
        return 1
    module_df["node"] = module_df["node"].astype(str).str.strip()

    true_module = module_df["node"].tolist()
    # The true anchoring seed set is is_seed AND NOT is_excluded_seed --
    # is_seed alone only means "was listed in the original seeds.txt", but
    # step6_build_disease_module.py's --exclude-seed-celltypes / --exclude-
    # ppi-seeds (e.g. to keep global_ppi genes available as bridging
    # candidates without letting them anchor the module) drops some of
    # those from the actual anchor set while leaving is_seed=True on the
    # row. Using is_seed alone here would silently anchor the null
    # permutations on a larger, different seed set than step6 actually
    # used to build the true module -- is_excluded_seed is exactly the
    # column that records which ones were dropped.
    if "is_excluded_seed" in module_df.columns:
        anchor_mask = module_df["is_seed"] & (~module_df["is_excluded_seed"].astype(bool))
        n_dropped = int((module_df["is_seed"] & module_df["is_excluded_seed"].astype(bool)).sum())
        if n_dropped:
            print(f"  {n_dropped} seed(s) marked is_excluded_seed=True (e.g. via "
                  f"--exclude-seed-celltypes) -- excluded from the anchor set here too, "
                  f"matching step6's own behavior.")
    else:
        anchor_mask = module_df["is_seed"]
    true_seeds = module_df.loc[anchor_mask, "node"].tolist()

    missing_from_ref = [g for g in true_module if g not in idx_of]
    if missing_from_ref:
        print(f"  ⚠ {len(missing_from_ref)} module node(s) not found in --reference-graph "
              f"(first 10): {missing_from_ref[:10]}")
        print("    (This can happen if --reference-graph doesn't match the wdir step6 was run "
              "against, or if step6 used --connectivity ppi against gene_ppi_reference_graph.pkl "
              "-- pass that file instead if so.)")
    
    print(f"  True module size: {len(true_module)}")
    print(f"  True seeds in module: {len(true_seeds)}")
    
    # Build gene_probs dict for the true module, keyed by NAMESPACED node --
    # bare gene symbols would collide across cell types.
    true_node_probs = {}
    if 'rwr_score' in module_df.columns:
        for _, row in module_df.iterrows():
            true_node_probs[row['node']] = row['rwr_score']
    else:
        for node in true_module:
            true_node_probs[node] = 1.0 / len(true_module)
    
    # ── Compute true metrics (against the FULL reference graph) ────────────
    total_vol = sum(d for _, d in G_ref.degree())
    true_metrics = compute_all_metrics_fast(G_ref, true_module, true_node_probs, total_vol)
    
    print(f"\nTrue module metrics (relative to full reference network):")
    for k_m, v in true_metrics.items():
        print(f"  {k_m:25s}: {v:.6f}")
    
    # ── Build degree bins (from the FULL reference graph) ──────────────────
    print(f"\nBuilding degree bins (bin_size={args.bin_size})...")
    degree_bins = build_degree_bins(G_ref, bin_size=args.bin_size)
    print(f"  Created {len(degree_bins)} degree bins")

    true_seeds_in_ref = [s for s in true_seeds if s in idx_of]
    if len(true_seeds_in_ref) < len(true_seeds):
        print(f"  ⚠ {len(true_seeds) - len(true_seeds_in_ref)} true seed(s) not in "
              f"--reference-graph -- excluded from degree-matched sampling")
    
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
        
        rand_seeds = sample_degree_matched_seeds(
            true_seeds_in_ref, G_ref, degree_bins, rng, bin_size=args.bin_size)
        
        if len(rand_seeds) < max(1, len(true_seeds_in_ref) // 2):
            failed_perms += 1
            continue
        
        rand_module, rand_probs = run_rwr_aggregated(
            G_ref, nodelist, idx_of, M, deg, rand_seeds, args.restart,
            scaling=args.scaling, k=args.k)
        
        if rand_module is None:
            failed_perms += 1
            continue
        
        null_metrics_list.append(
            compute_all_metrics_fast(G_ref, rand_module, rand_probs, total_vol))
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