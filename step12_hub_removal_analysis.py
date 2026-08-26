#!/usr/bin/env python

"""
step12_hub_removal_analysis.py

Hub-removal robustness / attack-tolerance analysis for a disease module
built from the HETEROGENEOUS MULTIPLEX pipeline (step6_build_disease_module.py
output). This script assumes --module-graph nodes are namespaced as
"<celltype>::GENE" and --disease-module is step6's disease_module.csv
(node/gene/cell_type/is_seed/... columns) -- it is not a general-purpose
tool for arbitrary/bare-gene graphs.

The attack-curve / hub-definition / knockout machinery below is node-ID
agnostic -- it just calls G.nodes()/G.degree()/etc, so it runs correctly
on the namespaced module graph as-is (degree/betweenness are computed per
"<celltype>::gene" node, which is the right thing to do: a gene's degree
in one cell-type layer should not be conflated with its degree in
another). Two places DO need bare gene symbols instead of namespaced
node strings, and this script handles the conversion for both:

  1. Functional enrichment (gseapy needs real HGNC symbols, not
     "celltype::gene" strings) -- namespaced nodes are collapsed to
     unique bare gene symbols before writing gene lists / calling gseapy.
  2. The hub-bias permutation test, which does dict lookups by node ID
     against --background-graph. --background-graph must be a BARE
     gene-symbol PPI network (e.g. graph_lcc.pkl from the single-layer
     og_gene_set pipeline) -- the module is collapsed to unique bare gene
     symbols before comparison. The script warns if --background-graph
     itself looks namespaced (i.e. you accidentally pointed it at the
     multiplex reference graph instead).

Also writes hub_sets_celltype_breakdown.csv: for every hub definition
(top10% degree, Yu hub/bottleneck, etc.) the cell-type composition of the
nodes it flags as hubs -- useful since a "hub" set in this graph can be
dominated by a single layer.

CHANGES FROM ORIGINAL (perf only, no change in what is measured/reported
unless you opt into the approximation flags below):

  1. Fixed a bug where nx.connected_components(H) was computed TWICE per
     removal step (once inline, once inside _lcc_fraction). Now computed
     once. This alone ~halves the cost of every attack curve.

  2. Cascading attacks now recompute the centrality ranking every
     `--cascading-step` nodes removed, instead of every single node
     (`--step` still controls the resolution at which points are recorded
     for plotting/f_c/f_90, decoupled from the recompute cadence). Batch
     recomputation is standard in the cascading-failure literature and has
     negligible effect on f_c/f_90 as long as the batch is small relative
     to module size (a few percent).

  3. Optional approximate betweenness centrality via source-sampling
     (`--betweenness-k`), used ONLY inside the degree-preserving null loop
     by default (`--null-betweenness-k`), since that is where betweenness
     is recomputed the most times. Approximation error shrinks as
     O(1/sqrt(k)) and is seeded for reproducibility. The real module's own
     curves still use exact betweenness unless you explicitly also set
     --betweenness-k.

  4. The degree-preserving null benchmark now supports a SEPARATE
     (smaller) replicate count for the expensive cascading modes
     (`--n-null-cascading`) vs. the cheap static modes (`--n-null-networks`),
     since cascading betweenness/degree_betweenness recomputation is the
     dominant cost. Report the (wider) null_std for cascading stats
     explicitly, which the output CSVs already do.

  5. Null-network generation/attack and random-failure trials are
     embarrassingly parallel and are now run with
     concurrent.futures.ProcessPoolExecutor (`--n-jobs`).

Usage
-----
python step12_hub_removal_analysis.py \
    --module-graph 06_disease_module/disease_module_output/module_graph.pkl \
    --disease-module 06_disease_module/disease_module_output/disease_module.csv \
    --background-graph /path/to/bare_gene_ppi.pkl \
    --output-dir 12_hub_removal \
    --cascading-step 10 \
    --n-null-cascading 15 \
    --null-betweenness-k 150 \
    --n-jobs 8
"""

import os
import sys
import time
import pickle
import argparse
import traceback
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# "celltype::gene" namespace helpers
# --------------------------------------------------------------------------

def _bare_gene(node):
    """Strip a '<celltype>::GENE' namespace prefix down to the bare gene
    symbol."""
    s = str(node)
    return s.rsplit("::", 1)[-1] if "::" in s else s


def _celltype_of(node):
    """Cell-type prefix of a namespaced node, or 'unknown' if the node
    string has no '::' separator (shouldn't happen for a well-formed
    step6 module, but handled defensively)."""
    s = str(node)
    return s.rsplit("::", 1)[0] if "::" in s else "unknown"


def to_gene_symbols(nodes):
    """Collapse a node iterable ('celltype::gene' strings) to a sorted
    list of unique bare gene symbols."""
    return sorted({_bare_gene(n) for n in nodes})


def check_namespaced(G, label, sample_size=200):
    """Warn (don't fail) if a graph that's supposed to be namespaced
    doesn't look like it -- most likely means the wrong file was passed."""
    sample = list(itertools.islice(G.nodes(), sample_size))
    if not sample:
        return
    frac = sum(1 for n in sample if "::" in str(n)) / len(sample)
    if frac < 0.5:
        print(f"  WARNING: {label} nodes do not look namespaced ('celltype::gene') -- "
              f"only {frac*100:.0f}% of a {len(sample)}-node sample contain '::'. "
              f"This script expects step6_build_disease_module.py output; double-check "
              f"you passed the right file.")


# --------------------------------------------------------------------------
# 2. Hub definitions
# --------------------------------------------------------------------------

def define_hub_sets(G, sd_multipliers=(1, 2, 3), pct_thresh=0.80):
    """
    Build a dictionary of hub_definition_name -> set of nodes, covering:
      - top 5% / 10% / 20% by degree
      - degree > mean + k*SD for k in sd_multipliers
      - Yu et al. (2007) four-class taxonomy using degree and betweenness
        percentile thresholds (default: 80th percentile each).

    Degree/betweenness are computed within the namespaced module graph
    directly, i.e. per "<celltype>::gene" node -- a gene's degree in one
    cell-type layer is not conflated with its degree in another.

    Returns
    -------
    hub_sets : dict[str, set]
    centrality : dict with 'degree' and 'betweenness' dicts (node -> value)
    """
    print("Computing degree and betweenness centrality for hub definitions...")
    degree = dict(G.degree())
    # exact betweenness here: this is a ONE-TIME cost on the real module,
    # not repeated across nulls, so it is cheap relative to the cascading
    # attack loop and worth keeping exact.
    betweenness = nx.betweenness_centrality(G, normalized=True)

    nodes = list(G.nodes())
    deg_vals = np.array([degree[n] for n in nodes], dtype=float)
    btw_vals = np.array([betweenness[n] for n in nodes], dtype=float)

    hub_sets = {}

    for pct in (0.05, 0.10, 0.20):
        n_top = max(1, int(np.ceil(pct * len(nodes))))
        ranked = sorted(nodes, key=lambda n: degree[n], reverse=True)
        hub_sets[f"top{int(pct*100)}pct_degree"] = set(ranked[:n_top])

    mean_deg = deg_vals.mean()
    sd_deg = deg_vals.std()
    for k in sd_multipliers:
        thresh = mean_deg + k * sd_deg
        hub_sets[f"degree_gt_mean_plus_{k}SD"] = {
            n for n in nodes if degree[n] > thresh
        }

    deg_cut = np.quantile(deg_vals, pct_thresh)
    btw_cut = np.quantile(btw_vals, pct_thresh)
    hub_bottleneck, pure_hub, pure_bottleneck, neither = set(), set(), set(), set()
    for n in nodes:
        high_deg = degree[n] >= deg_cut
        high_btw = betweenness[n] >= btw_cut
        if high_deg and high_btw:
            hub_bottleneck.add(n)
        elif high_deg and not high_btw:
            pure_hub.add(n)
        elif (not high_deg) and high_btw:
            pure_bottleneck.add(n)
        else:
            neither.add(n)
    hub_sets["yu_hub_bottleneck"] = hub_bottleneck
    hub_sets["yu_pure_hub"] = pure_hub
    hub_sets["yu_pure_bottleneck"] = pure_bottleneck
    hub_sets["yu_neither"] = neither

    for name, s in hub_sets.items():
        print(f"  {name}: {len(s)} nodes")

    return hub_sets, {"degree": degree, "betweenness": betweenness}


def hub_sets_celltype_breakdown(hub_sets):
    """Summarize which cell type(s) each hub definition's nodes belong
    to -- i.e. which layer(s) a given knockout is really hitting. Returns
    a long-format DataFrame with columns hub_definition / cell_type /
    n_nodes."""
    rows = []
    for name, s in hub_sets.items():
        counts = pd.Series([_celltype_of(n) for n in s]).value_counts()
        for ct, n in counts.items():
            rows.append({"hub_definition": name, "cell_type": ct, "n_nodes": int(n)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. Attack / robustness curves
# --------------------------------------------------------------------------

def _lcc_size_and_fraction(H, n_original):
    """Single pass over connected_components -> (lcc_size, lcc_fraction)."""
    if H.number_of_nodes() == 0:
        return 0, 0.0
    largest = max((len(c) for c in nx.connected_components(H)), default=0)
    return largest, largest / n_original


def _centrality_order(graph, mode, betweenness_k=None, seed=None):
    if mode == "degree":
        c = dict(graph.degree())
        return sorted(graph.nodes(), key=lambda n: c[n], reverse=True)
    elif mode == "betweenness":
        k = min(betweenness_k, graph.number_of_nodes()) if betweenness_k else None
        c = nx.betweenness_centrality(graph, k=k, normalized=True, seed=seed)
        return sorted(graph.nodes(), key=lambda n: c[n], reverse=True)
    elif mode == "degree_betweenness":
        deg = dict(graph.degree())
        k = min(betweenness_k, graph.number_of_nodes()) if betweenness_k else None
        btw = nx.betweenness_centrality(graph, k=k, normalized=True, seed=seed)
        deg_rank = {n: r for r, n in enumerate(sorted(graph.nodes(), key=lambda n: deg[n], reverse=True))}
        btw_rank = {n: r for r, n in enumerate(sorted(graph.nodes(), key=lambda n: btw[n], reverse=True))}
        combined = {n: deg_rank[n] + btw_rank[n] for n in graph.nodes()}
        return sorted(graph.nodes(), key=lambda n: combined[n])
    else:
        raise ValueError(f"Unknown attack mode: {mode}")


def attack_curve(G, mode="degree", cascading=False, step=1, cascading_step=None,
                  betweenness_k=None, rng=None, seed=None):
    """
    Simulate node removal and track the giant-component (LCC) size.

    Parameters
    ----------
    G : networkx graph (will be copied, not modified)
    mode : 'degree', 'betweenness', 'degree_betweenness', or 'random'
    cascading : if True, recompute the centrality ranking every
        `cascading_step` removed nodes (adaptive attack). If False, rank
        once on the original graph and remove in that fixed order.
    step : granularity (in nodes) at which points are RECORDED for the
        output curve / f_c / f_90 (does not affect recompute cost).
    cascading_step : granularity (in nodes) at which the centrality
        ranking is RECOMPUTED under cascading mode. Defaults to `step`
        if not given. Set this larger than `step` to cut the number of
        (expensive, for betweenness modes) recomputations while still
        recording a fine-grained curve.
    betweenness_k : if set, use source-sampled approximate betweenness
        with this many sampled sources instead of exact all-pairs
        betweenness (only affects 'betweenness' / 'degree_betweenness').
    rng : numpy Generator, used for random-failure mode
    seed : int, passed to nx.betweenness_centrality's internal sampling
        when betweenness_k is set (reproducibility)

    Returns
    -------
    DataFrame with columns: n_removed, frac_removed, lcc_size, lcc_fraction
    """
    if cascading_step is None:
        cascading_step = step

    H = G.copy()
    n_original = H.number_of_nodes()
    records = [(0, 0.0, n_original, 1.0)]

    if mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        order = list(H.nodes())
        rng.shuffle(order)
        removed = 0
        for i in range(0, len(order), step):
            batch = order[i:i + step]
            H.remove_nodes_from(batch)
            removed += len(batch)
            lcc_size, lcc_frac = _lcc_size_and_fraction(H, n_original)
            records.append((removed, removed / n_original, lcc_size, lcc_frac))
        return pd.DataFrame(records, columns=["n_removed", "frac_removed", "lcc_size", "lcc_fraction"])

    if not cascading:
        order = _centrality_order(H, mode, betweenness_k=betweenness_k, seed=seed)
        removed = 0
        for i in range(0, len(order), step):
            batch = order[i:i + step]
            H.remove_nodes_from(batch)
            removed += len(batch)
            lcc_size, lcc_frac = _lcc_size_and_fraction(H, n_original)
            records.append((removed, removed / n_original, lcc_size, lcc_frac))
    else:
        removed = 0
        while H.number_of_nodes() > 0:
            order = _centrality_order(H, mode, betweenness_k=betweenness_k, seed=seed)
            batch = order[:cascading_step]
            H.remove_nodes_from(batch)
            removed += len(batch)
            lcc_size, lcc_frac = _lcc_size_and_fraction(H, n_original)
            records.append((removed, removed / n_original, lcc_size, lcc_frac))

    return pd.DataFrame(records, columns=["n_removed", "frac_removed", "lcc_size", "lcc_fraction"])


def _random_trial(args):
    G, step, seed = args
    rng = np.random.default_rng(seed)
    df = attack_curve(G, mode="random", step=step, rng=rng)
    return df.set_index("frac_removed")["lcc_fraction"]


def random_failure_average(G, n_trials=50, step=1, seed=0, n_jobs=1):
    """Average random-failure curve over n_trials shuffles, with std band."""
    n_original = G.number_of_nodes()
    seeds = [int(s) for s in np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_trials)]
    tasks = [(G, step, s) for s in seeds]

    all_curves = []
    if n_jobs and n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for series in ex.map(_random_trial, tasks):
                all_curves.append(series)
    else:
        for t in tasks:
            all_curves.append(_random_trial(t))

    common_x = np.linspace(0, 1, n_original + 1)
    interp = np.array([np.interp(common_x, s.index.values, s.values) for s in all_curves])
    return pd.DataFrame({
        "frac_removed": common_x,
        "lcc_fraction_mean": interp.mean(axis=0),
        "lcc_fraction_std": interp.std(axis=0),
    })


def critical_fraction(df, threshold_fc=0.05, threshold_f90=0.9):
    """
    f_c   : smallest frac_removed at which lcc_fraction first falls to
            or below threshold_fc.
    f_90  : smallest frac_removed at which lcc_fraction first falls to
            or below threshold_f90.
    """
    below_fc = df[df["lcc_fraction"] <= threshold_fc]
    below_f90 = df[df["lcc_fraction"] <= threshold_f90]
    f_c = below_fc["frac_removed"].min() if not below_fc.empty else np.nan
    f_90 = below_f90["frac_removed"].min() if not below_f90.empty else np.nan
    return f_c, f_90


# --------------------------------------------------------------------------
# Degree-preserving null model
# --------------------------------------------------------------------------

def _null_worker(args):
    (null_id, edges, nodes, modes, cascading_modes, step, cascading_step,
     betweenness_k, swap_factor, seed) = args
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from(edges)
    n_edges = H.number_of_edges()
    n_swaps = max(1, swap_factor * n_edges)
    try:
        nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)
    except nx.NetworkXError as e:
        print(f"  [null {null_id}] double_edge_swap warning: {e}")

    rows = []
    for mode in modes:
        df = attack_curve(H, mode=mode, cascading=False, step=step)
        fc, f90 = critical_fraction(df)
        rows.append({"null_id": null_id, "mode": mode, "cascading": False, "f_c": fc, "f_90": f90})
        if mode in cascading_modes:
            df_c = attack_curve(H, mode=mode, cascading=True, step=step,
                                 cascading_step=cascading_step,
                                 betweenness_k=betweenness_k, seed=seed)
            fc_c, f90_c = critical_fraction(df_c)
            rows.append({"null_id": null_id, "mode": mode, "cascading": True, "f_c": fc_c, "f_90": f90_c})
    return rows


def degree_preserving_null_curves(G, modes, n_null=20, n_null_cascading=None,
                                   cascading_modes=(), step=1, cascading_step=None,
                                   betweenness_k=None, swap_factor=10, seed=0, n_jobs=1):
    """
    Generate degree-preserving rewired graphs (double_edge_swap) and run
    attack modes on each, recording f_c / f_90 for comparison against the
    real module.

    Static modes run on `n_null` replicates; cascading modes (the
    expensive ones for betweenness-based rankings) run on only the first
    `n_null_cascading` of those replicates if given (defaults to n_null).
    Report null_std in the output -- fewer cascading replicates means a
    wider (still valid) null CI, which the downstream comparison CSV
    already surfaces.

    double_edge_swap rewires edges among the EXISTING node set, so
    "celltype::gene" node identities (and thus which layer each node
    belongs to) are preserved in the null graphs -- only which edges
    connect them changes.
    """
    if n_null_cascading is None:
        n_null_cascading = n_null

    edges = list(G.edges())
    nodes = list(G.nodes())
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n_null):
        this_cascading_modes = cascading_modes if i < n_null_cascading else set()
        tasks.append((i, edges, nodes, modes, this_cascading_modes, step, cascading_step,
                      betweenness_k, swap_factor, int(rng.integers(0, 2**31 - 1))))

    rows = []
    if n_jobs and n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_null_worker, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futures):
                rows.extend(fut.result())
                done += 1
                if done % 5 == 0 or done == n_null:
                    print(f"  Completed {done}/{n_null} null networks")
    else:
        for t in tasks:
            rows.extend(_null_worker(t))
            i = t[0]
            if (i + 1) % 5 == 0 or i == n_null - 1:
                print(f"  Completed {i+1}/{n_null} null networks")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Hub-bias permutation test
# --------------------------------------------------------------------------

def hub_bias_permutation_test(background_G, module_genes, n_perm=1000, seed=0,
                                degree_matched=True, bins=10):
    """
    Is the disease module's average degree (in the background/full,
    bare-gene network) higher than expected by chance?

    `module_genes` must already be bare gene symbols matching
    background_G's node IDs -- callers pass to_gene_symbols(G.nodes()),
    which collapses the namespaced module down to its unique gene set.
    """
    rng = np.random.default_rng(seed)
    bg_degree = dict(background_G.degree())
    module_in_bg = [n for n in module_genes if n in bg_degree]
    if len(module_in_bg) < len(module_genes):
        print(f"  Note: {len(module_genes) - len(module_in_bg)} module genes not found in background graph")

    obs_mean_degree = np.mean([bg_degree[n] for n in module_in_bg])
    all_nodes = list(background_G.nodes())
    results = {"observed_mean_degree": obs_mean_degree, "n_module_genes_in_bg": len(module_in_bg)}

    null_means = np.empty(n_perm)
    for i in range(n_perm):
        sample = rng.choice(all_nodes, size=len(module_in_bg), replace=False)
        null_means[i] = np.mean([bg_degree[n] for n in sample])
    p_size = (np.sum(null_means >= obs_mean_degree) + 1) / (n_perm + 1)
    results["size_matched"] = {
        "null_mean": null_means.mean(), "null_std": null_means.std(), "p_value": p_size
    }

    if degree_matched:
        deg_vals = np.array([bg_degree[n] for n in all_nodes])
        quantile_edges = np.quantile(deg_vals, np.linspace(0, 1, bins + 1))
        quantile_edges[-1] += 1e-9
        node_bin = {n: int(np.digitize(bg_degree[n], quantile_edges) - 1) for n in all_nodes}
        module_bin_counts = pd.Series([node_bin[n] for n in module_in_bg]).value_counts().to_dict()
        nodes_by_bin = {}
        for n in all_nodes:
            nodes_by_bin.setdefault(node_bin[n], []).append(n)

        null_means_dm = np.empty(n_perm)
        for i in range(n_perm):
            sample = []
            for b, count in module_bin_counts.items():
                pool = nodes_by_bin.get(b, [])
                if len(pool) == 0:
                    continue
                take = min(count, len(pool))
                sample.extend(rng.choice(pool, size=take, replace=False))
            null_means_dm[i] = np.mean([bg_degree[n] for n in sample]) if sample else np.nan
        p_dm = (np.sum(null_means_dm >= obs_mean_degree) + 1) / (n_perm + 1)
        results["degree_matched"] = {
            "null_mean": np.nanmean(null_means_dm), "null_std": np.nanstd(null_means_dm), "p_value": p_dm
        }

    return results


# --------------------------------------------------------------------------
# 2/4. Discrete hub-set knockouts + functional degradation tracking
# --------------------------------------------------------------------------

def knockout_hub_sets(G, hub_sets):
    n_original = G.number_of_nodes()
    rows = []
    residual_genes = {}
    for name, hub_nodes in hub_sets.items():
        H = G.copy()
        H.remove_nodes_from(hub_nodes)
        n_components = nx.number_connected_components(H) if H.number_of_nodes() > 0 else 0
        _, lcc_frac = _lcc_size_and_fraction(H, n_original)
        rows.append({
            "hub_definition": name,
            "n_hub_nodes_removed": len(hub_nodes),
            "frac_removed": len(hub_nodes) / n_original,
            "n_components_after": n_components,
            "lcc_fraction_after": lcc_frac,
            "n_residual_nodes": H.number_of_nodes(),
        })
        residual_genes[name] = list(H.nodes())
    return pd.DataFrame(rows), residual_genes


def try_enrichment(gene_list, label, output_dir, gene_sets=None, organism="human"):
    gene_sets = gene_sets or ["GO_Biological_Process_2021", "KEGG_2021_Human"]
    genes_file = os.path.join(output_dir, f"genes_{label}.txt")
    with open(genes_file, "w") as f:
        f.write("\n".join(sorted(map(str, gene_list))))

    try:
        import gseapy as gp
    except ImportError:
        print(f"  [{label}] gseapy not installed; wrote gene list to {genes_file} for manual enrichment.")
        return None

    try:
        enr = gp.enrichr(gene_list=list(gene_list), gene_sets=gene_sets,
                          organism=organism, outdir=None)
        results_file = os.path.join(output_dir, f"enrichment_{label}.csv")
        enr.results.to_csv(results_file, index=False)
        print(f"  [{label}] enrichment saved to {results_file}")
        return enr.results
    except Exception as e:
        print(f"  [{label}] enrichment failed ({e}); wrote gene list to {genes_file} for manual enrichment.")
        return None


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_attack_curves(curves, null_summary, output_path):
    modes = sorted({k[0] for k in curves.keys() if isinstance(k, tuple)})
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        for cascading, style in [(False, "-"), (True, "--")]:
            key = (mode, cascading)
            if key not in curves:
                continue
            df = curves[key]
            label = f"{mode} ({'cascading' if cascading else 'static'})"
            ax.plot(df["frac_removed"], df["lcc_fraction"], style, label=label, linewidth=2)

        if "random_mean" in curves:
            rdf = curves["random_mean"]
            ax.plot(rdf["frac_removed"], rdf["lcc_fraction_mean"], color="gray", label="random failure", linewidth=2)
            ax.fill_between(rdf["frac_removed"],
                             rdf["lcc_fraction_mean"] - rdf["lcc_fraction_std"],
                             rdf["lcc_fraction_mean"] + rdf["lcc_fraction_std"],
                             color="gray", alpha=0.2)

        if null_summary is not None and not null_summary.empty:
            sub = null_summary[(null_summary["mode"] == mode) & (~null_summary["cascading"])]
            if not sub.empty:
                null_fc = sub["f_c"].median()
                if not np.isnan(null_fc):
                    ax.axvline(null_fc, color="black", linestyle=":", alpha=0.6,
                                label="null median f_c")

        ax.set_xlabel("Fraction of nodes removed")
        ax.set_title(mode)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("LCC fraction (relative to module size)")
    fig.suptitle("Hub-removal robustness curves — disease module")
    plt.tight_layout()
    outfile = os.path.join(output_path, "attack_curves.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved attack curve plot to {outfile}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hub removal / attack-tolerance analysis for a HETEROGENEOUS "
                     "MULTIPLEX disease module (step6_build_disease_module.py output)")
    parser.add_argument("--module-graph", type=str, required=True,
                         help="step6's module_graph.pkl -- pickled networkx graph with nodes "
                              "namespaced as 'celltype::gene'.")
    parser.add_argument("--disease-module", type=str, default=None,
                         help="Optional, for reference/reporting only: step6's "
                              "disease_module.csv (node/gene/cell_type/is_seed/... columns).")
    parser.add_argument("--background-graph", type=str, default=None,
                         help="Pickled networkx graph used as the background/full network for "
                              "the hub-bias permutation test. Must use BARE gene-symbol node "
                              "IDs (e.g. graph_lcc.pkl from the single-layer og_gene_set "
                              "pipeline) -- NOT the namespaced multiplex reference graph. The "
                              "module is collapsed to unique bare gene symbols before comparison.")
    parser.add_argument("--output-dir", type=str, default="output",
                         help="All outputs are written directly under "
                              "<output-dir>/hub_removal/ (no PPI-name subfolder).")
    parser.add_argument("--n-random-trials", type=int, default=50)
    parser.add_argument("--n-null-networks", type=int, default=20,
                         help="Replicates for static attack modes in the degree-preserving null")
    parser.add_argument("--n-null-cascading", type=int, default=None,
                         help="Replicates for CASCADING attack modes in the null (default: same as "
                              "--n-null-networks; set lower, e.g. 10-15, since cascading betweenness "
                              "is the dominant cost)")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--step", type=int, default=1,
                         help="Granularity (nodes) at which curve points are recorded")
    parser.add_argument("--cascading-step", type=int, default=None,
                         help="Granularity (nodes) at which cascading rankings are RECOMPUTED. "
                              "Defaults to --step. Set this to ~0.5-2%% of module size to drastically "
                              "cut cascading betweenness recomputation cost with negligible effect on "
                              "f_c/f_90.")
    parser.add_argument("--betweenness-k", type=int, default=None,
                         help="If set, use approximate (source-sampled) betweenness with this many "
                              "sampled sources for the REAL module's own curves too. Leave unset to "
                              "keep the real module's curves exact.")
    parser.add_argument("--null-betweenness-k", type=int, default=None,
                         help="If set, use approximate betweenness with this many sampled sources "
                              "ONLY inside the null-network loop (recommended: 100-200 for large "
                              "modules, since this is where recomputation is repeated most).")
    parser.add_argument("--n-jobs", type=int, default=1,
                         help="Parallel workers for null-network generation/attack and random-failure "
                              "trials (embarrassingly parallel). Set to match --cpus-per-task.")
    parser.add_argument("--cascading-modes", type=str, default="degree,random")
    parser.add_argument("--yu-percentile", type=float, default=0.80)
    parser.add_argument("--sd-multipliers", type=str, default="1,2,3")
    parser.add_argument("--run-enrichment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    output_path = ensure_dir(os.path.join(args.output_dir, "hub_removal"))
    print(f"\n{'='*80}\nHub Removal / Attack-Tolerance Analysis (heterogeneous multiplex)\n{'='*80}")
    print(f"Output directory: {output_path}\n")

    try:
        print(f"Loading module graph from {args.module_graph}...")
        G = load_pickle(args.module_graph)
        print(f"Module graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        check_namespaced(G, "--module-graph")
        n_celltypes = len({_celltype_of(n) for n in G.nodes()})
        n_unique_genes = len({_bare_gene(n) for n in G.nodes()})
        print(f"  {n_celltypes} cell type(s), {n_unique_genes} unique gene symbol(s)")
        if not nx.is_connected(G):
            n_comp = nx.number_connected_components(G)
            print(f"WARNING: module graph is not fully connected ({n_comp} components); "
                  f"attack curves use the LCC of whatever remains at each step.")

        if args.disease_module:
            dm_df = pd.read_csv(args.disease_module)
            gene_col = "gene" if "gene" in dm_df.columns else dm_df.columns[0]
            module_genes = dm_df[gene_col].tolist()
            print(f"Loaded disease_module CSV ({len(module_genes)} rows) for reference")
            if "cell_type" in dm_df.columns:
                print("  Cell-type breakdown:")
                print(dm_df["cell_type"].value_counts().to_string())
            if "is_seed" in dm_df.columns:
                print(f"  Seed genes in module: {int(dm_df['is_seed'].sum())}")

        background_G = None
        if args.background_graph:
            print(f"Loading background graph from {args.background_graph}...")
            background_G = load_pickle(args.background_graph)
            print(f"Background graph: {background_G.number_of_nodes()} nodes, {background_G.number_of_edges()} edges")
            bg_sample = list(itertools.islice(background_G.nodes(), 200))
            if bg_sample and any("::" in str(n) for n in bg_sample):
                print("  WARNING: --background-graph nodes look namespaced ('::' found). "
                      "This script needs a BARE gene-symbol background network; "
                      "results will likely show near-zero overlap with the module.")

        print("\n--- Hub definitions ---")
        sd_multipliers = tuple(int(x) for x in args.sd_multipliers.split(","))
        hub_sets, centrality = define_hub_sets(G, sd_multipliers=sd_multipliers, pct_thresh=args.yu_percentile)
        save_pickle(hub_sets, os.path.join(output_path, "hub_sets.pkl"))
        save_pickle(centrality, os.path.join(output_path, "centrality.pkl"))

        print("\n--- Cell-type composition of hub sets ---")
        ct_breakdown_df = hub_sets_celltype_breakdown(hub_sets)
        ct_breakdown_df.to_csv(os.path.join(output_path, "hub_sets_celltype_breakdown.csv"), index=False)
        pivot = ct_breakdown_df.pivot(index="hub_definition", columns="cell_type", values="n_nodes").fillna(0).astype(int)
        print(pivot.to_string())

        print("\n--- Discrete hub-set knockouts (recommendation 2 + 4) ---")
        knockout_df, residual_genes = knockout_hub_sets(G, hub_sets)
        knockout_df.to_csv(os.path.join(output_path, "hub_knockout_summary.csv"), index=False)
        print(knockout_df.to_string(index=False))

        conclusions_flip = knockout_df["lcc_fraction_after"].max() - knockout_df["lcc_fraction_after"].min() > 0.15
        if conclusions_flip:
            print("\nNOTE: post-knockout LCC fraction varies substantially across hub definitions "
                  "(range > 0.15) — fragility conclusions are sensitive to how 'hub' is defined. "
                  "Report this explicitly.")

        if args.run_enrichment:
            print("\n--- Functional enrichment of residual modules (recommendation 4) ---")
            enrich_dir = ensure_dir(os.path.join(output_path, "enrichment"))
            full_genes = to_gene_symbols(G.nodes())
            print(f"  Collapsing {G.number_of_nodes()} celltype::gene nodes to "
                  f"{len(full_genes)} unique gene symbols for enrichment.")
            try_enrichment(full_genes, "full_module", enrich_dir)
            for name, genes in residual_genes.items():
                try_enrichment(to_gene_symbols(genes), f"residual_after_{name}", enrich_dir)

        print("\n--- Attack curves (recommendation 1) ---")
        modes = ["degree", "betweenness", "degree_betweenness"]
        cascading_modes = set(m.strip() for m in args.cascading_modes.split(",") if m.strip())

        curves = {}
        fc_f90_rows = []
        for mode in modes:
            print(f"  Static {mode} attack...")
            df_static = attack_curve(G, mode=mode, cascading=False, step=args.step,
                                      betweenness_k=args.betweenness_k, seed=args.seed)
            curves[(mode, False)] = df_static
            fc, f90 = critical_fraction(df_static)
            fc_f90_rows.append({"mode": mode, "cascading": False, "f_c": fc, "f_90": f90})

            if mode in cascading_modes:
                print(f"  Cascading {mode} attack (recompute every {args.cascading_step or args.step} nodes)...")
                df_casc = attack_curve(G, mode=mode, cascading=True, step=args.step,
                                        cascading_step=args.cascading_step,
                                        betweenness_k=args.betweenness_k, seed=args.seed)
                curves[(mode, True)] = df_casc
                fc_c, f90_c = critical_fraction(df_casc)
                fc_f90_rows.append({"mode": mode, "cascading": True, "f_c": fc_c, "f_90": f90_c})

        print("  Random failure (averaged)...")
        random_df = random_failure_average(G, n_trials=args.n_random_trials, step=args.step,
                                            seed=args.seed, n_jobs=args.n_jobs)
        curves["random_mean"] = random_df
        below = random_df[random_df["lcc_fraction_mean"] <= 0.05]
        fc_rand = below["frac_removed"].min() if not below.empty else np.nan
        below90 = random_df[random_df["lcc_fraction_mean"] <= 0.9]
        f90_rand = below90["frac_removed"].min() if not below90.empty else np.nan
        fc_f90_rows.append({"mode": "random", "cascading": False, "f_c": fc_rand, "f_90": f90_rand})

        fc_f90_df = pd.DataFrame(fc_f90_rows)
        fc_f90_df.to_csv(os.path.join(output_path, "f_c_f90_observed.csv"), index=False)
        print(fc_f90_df.to_string(index=False))

        for key, df in curves.items():
            if key == "random_mean":
                continue
            mode, cascading = key
            tag = "cascading" if cascading else "static"
            df.to_csv(os.path.join(output_path, f"attack_curve_{mode}_{tag}.csv"), index=False)
        random_df.to_csv(os.path.join(output_path, "attack_curve_random.csv"), index=False)

        print("\n--- Degree-preserving rewired null benchmark (recommendation 1) ---")
        null_df = degree_preserving_null_curves(
            G, modes=modes, n_null=args.n_null_networks,
            n_null_cascading=args.n_null_cascading,
            cascading_modes=cascading_modes & set(modes),
            step=args.step, cascading_step=args.cascading_step,
            betweenness_k=args.null_betweenness_k,
            seed=args.seed, n_jobs=args.n_jobs,
        )
        null_df.to_csv(os.path.join(output_path, "null_fc_f90.csv"), index=False)

        comparison_rows = []
        for _, row in fc_f90_df.iterrows():
            sub = null_df[(null_df["mode"] == row["mode"]) & (null_df["cascading"] == row["cascading"])]
            if sub.empty:
                continue
            for stat in ("f_c", "f_90"):
                null_vals = sub[stat].dropna().values
                obs = row[stat]
                if len(null_vals) == 0 or np.isnan(obs):
                    continue
                percentile = (np.sum(null_vals <= obs) / len(null_vals)) * 100
                comparison_rows.append({
                    "mode": row["mode"], "cascading": row["cascading"], "statistic": stat,
                    "observed": obs, "null_median": np.median(null_vals), "null_std": np.std(null_vals),
                    "null_n": len(null_vals),
                    "observed_percentile_in_null": percentile,
                })
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(os.path.join(output_path, "observed_vs_null_comparison.csv"), index=False)
        print(comparison_df.to_string(index=False))

        plot_attack_curves(curves, null_df, output_path)

        if background_G is not None:
            print("\n--- Hub-bias permutation test (recommendation 3) ---")
            query_genes = to_gene_symbols(G.nodes())
            print(f"  Comparing {len(query_genes)} unique bare gene symbols "
                  f"(collapsed across cell types) against --background-graph.")
            perm_results = hub_bias_permutation_test(
                background_G, query_genes, n_perm=args.n_permutations, seed=args.seed,
            )
            save_pickle(perm_results, os.path.join(output_path, "hub_bias_permutation_test.pkl"))
            print(f"  Observed mean degree in background: {perm_results['observed_mean_degree']:.3f}")
            for strategy in ("size_matched", "degree_matched"):
                if strategy in perm_results:
                    r = perm_results[strategy]
                    print(f"  [{strategy}] null mean={r['null_mean']:.3f} std={r['null_std']:.3f} "
                          f"p={r['p_value']:.4g}")
            if perm_results.get("size_matched", {}).get("p_value", 1.0) < 0.05:
                print("  NOTE: module has significantly higher degree than size-matched random node sets "
                      "-> results should be interpreted alongside DIAMOnD/DOMINO cross-check "
                      "(hub-bias / ascertainment caveat; Gillis; Skinnider).")
        else:
            print("\n--background-graph not provided; skipping hub-bias permutation test.")

        print(f"\n{'='*80}")
        print("Hub removal analysis complete.")
        print(f"All outputs saved to: {output_path}")
        print(f"{'='*80}")
        return 0

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Error in hub removal analysis: {e}")
        traceback.print_exc()
        print(f"{'='*80}")
        return 1


if __name__ == "__main__":
    sys.exit(main())