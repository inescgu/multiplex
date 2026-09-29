#!/usr/bin/env python3
"""
Step 7b (replaces step10b_layer_lcc_analysis.py): degree-normalized LCC
significance testing per cell type.

What changed vs. the old version, and why (read this before trusting numbers)
--------------------------------------------------------------------------------
The old version loaded ONE integrated_multiplex.pkl where every (source,
cell_type, gene) triple was its own node, and pathway/LR edges were
annotated directly onto that same graph. Its null model pulled candidate
nodes from a "pool" built from that whole graph's degree sequence — which,
because pathway/LR edges lived on the same graph, meant "degree" included
a gene's cross-cell-type connections, not just its within-cell-type PPI
degree. Sampling was then restricted to nodes that existed in the specific
layer being tested, which — since node IDs were already layer-scoped by
construction — always resolved to nodes native to that layer, evaluated at
their FULL-GRAPH degree rank.

In the new heterogeneous structure there is no single graph with that
property: multiplex/<cell_type>/ppi_layer.tsv graphs are the "eval" graphs
directly, one real nx.Graph per cell type (no slicing needed). For the
"pool" (the degree distribution the null model draws from), this script
uses gene_reference_graph.pkl — the combined graph step 3 already builds
from every cell type's PPI edges plus every bipartite (LR/pathway) edge.
That's the direct structural analogue of the old pool: a gene's degree
there reflects its connectivity across its own cell type's PPI AND its
cross-cell-type bipartite edges. The one real difference: because genes
are no longer namespaced per cell type, EGFR-in-Fibroblast and
EGFR-in-Neuron now contribute to the SAME reference-graph node's degree,
whereas the old node-per-(source,celltype,gene) design kept those degrees
separate. If a handful of promiscuous genes appear in many cell types,
their reference-graph degree will run higher than their old per-layer
degree would have. This is disclosed here rather than silently changing
your p-values without explanation — if it matters for your interpretation,
you can instead pass --pool-mode per-layer to build the null pool from
JUST the eval cell type's own PPI graph (closer to a plain configuration-
model null, losing the "reference includes bipartite reach" property).

NEW: fail-fast seed/layer overlap diagnostic
--------------------------------------------------------
Previously, if --seed-genes didn't actually match any gene names in the
loaded layers (wrong ID type, stray quote/BOM characters from an Excel
export, wrong column, species mismatch, ...), every single cell type would
silently short-circuit through analyze_layer's "if not seed_nodes: return
None" path, and the ONLY sign anything was wrong was the generic
"No results produced. Check seed/cell-type matching." printed after all
tasks finished — which, for a job with --num-simulations in the thousands
and a 36h walltime budget, is a very expensive way to find out your CSV had
a formatting problem. diagnose_seed_overlap() now runs right after seeds
and layers are loaded (before any tasks are dispatched) and reports the
overlap between seed genes and (a) the union of all cell-type layers and
(b) the null-model pool, printing example unmatched names on both sides so
a naming mismatch is visible immediately. If overlap with every layer is
zero, the script exits immediately with that diagnosis instead of running
the full job to arrive at the same "no results" conclusion.

load_seed_genes() is also now a bit more defensive: strips a UTF-8 BOM
(common from Excel-exported CSVs), matches a 'gene' column case-
insensitively, and strips stray quote characters in addition to whitespace.

NEW: DRG major-category coloring on the LCC bar chart
--------------------------------------------------------
plot_lcc_results() previously colored each cell type's bar green/red purely
by significance (p < 0.05). That's now carried by the DRG_HIERARCHY /
CELL_TYPE_TO_CATEGORY mapping (the same one step 7 uses) instead: bars are
colored by major_category (A-fiber / C-fiber / Non-classified /
Non-neuronal / unknown) via the shared _major_category_palette(), and
significance is now shown via bar outline (black edge + a "*" above the
bar) rather than fill color. analyze_layer() now attaches major_category /
minor_category to each result so this is available without any extra
lookups at plot time.

Everything else — the adaptive two-phase permutation test, per-component
metrics, multiprocessing via a degree-only pool object, SLURM-aware worker
count — is unchanged.
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
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# DRG NEURON HIERARCHY DEFINITION (same constant used by step 7 / step 7c —
# keep in sync if you edit cell-type names; consider centralizing in a
# shared module if you maintain all three)
# ============================================================================
DRG_HIERARCHY = {
    "A-fiber": {"color": "#1f77b4", "subcategories": {
        "Proprioceptors": {"marker": "o", "cell_types": ["A-Propr.EPHA3", "A-Propr.HAPLN4", "A-Propr.PCDH8"]},
        "A-beta LTMR": {"marker": "s", "cell_types": ["Ab-LTMR.CCKAR", "Ab-LTMR.ETV1", "Ab-LTMR.LGI2", "Ab-LTMR.NSG2"]},
        "A-LTMR": {"marker": "^", "cell_types": ["A-LTMR.TAC3"]},
        "A-peptidergic": {"marker": "D", "cell_types": ["A-PEP.CHRNA7_SLC18A3", "A-PEP.KIT", "A-PEP.NTRK3_S100A16", "A-PEP.SCGN_ADRA2C"]},
    }},
    "C-fiber": {"color": "#ff7f0e", "subcategories": {
        "C-LTMR": {"marker": "o", "cell_types": ["C-LTMR.CDH9"]},
        "C-peptidergic": {"marker": "D", "cell_types": ["C-PEP.ADORA2B", "C-PEP.TAC1_CACNG5", "C-PEP.TAC1_CHRNA3"]},
        "MRGPRX nociceptors": {"marker": "s", "cell_types": ["C-NP.MRGPRX1_GFRA2", "C-NP.MRGPRX1_MRGPRX4"]},
        "C-thermosensitive": {"marker": "*", "cell_types": ["C-Thermo.RXFP1", "C-Thermo.TRPM8"]},
        "C-SST nociceptor": {"marker": "v", "cell_types": ["C-NP.SST"]},
    }},
    "ATF3": {"color": "#d62728", "subcategories": {
        "ATF3 (injured)": {"marker": "x", "cell_types": ["ATF3"]},
    }},
    "Non-neuronal": {"color": "#2ca02c", "subcategories": {
        "Immune": {"marker": "o", "cell_types": [
            "B cell", "CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell",
            "activated CD4-positive, alpha-beta T cell", "activated CD8-positive, alpha-beta T cell",
            "basophil", "classical monocyte", "gamma-delta T cell", "granulocyte",
            "innate lymphoid cell", "intermediate monocyte", "leukocyte", "macrophage",
            "mast cell", "mature NK T cell", "myeloid cell", "myeloid dendritic cell",
            "myeloid leukocyte", "naive thymus-derived CD4-positive, alpha-beta T cell",
            "naive thymus-derived CD8-positive, alpha-beta T cell", "natural killer cell",
            "neutrophil", "non-classical monocyte", "plasma cell",
            "plasmacytoid dendritic cell", "regulatory T cell",
        ]},
        "Glial/Nervous": {"marker": "s", "cell_types": ["Schwann cell", "enteroglial cell", "microglial cell", "neuron"]},
        "Immune/Hematopoietic": {"marker": "^", "cell_types": [
            "Langerhans cell", "colon macrophage", "common myeloid progenitor",
            "erythroid progenitor cell", "hematopoietic cell", "hematopoietic precursor cell",
            "hematopoietic stem cell", "tissue-resident macrophage",
        ]},
    }},
}

CELL_TYPE_TO_CATEGORY = {}
for _major_cat, _major_data in DRG_HIERARCHY.items():
    for _subcat, _subcat_data in _major_data["subcategories"].items():
        for _cell_type in _subcat_data["cell_types"]:
            CELL_TYPE_TO_CATEGORY[_cell_type] = {
                "major": _major_cat, "minor": _subcat,
                "color": _major_data["color"], "marker": _subcat_data["marker"],
            }


def get_category_from_cell_type(cell_type: str) -> dict:
    if cell_type in CELL_TYPE_TO_CATEGORY:
        return CELL_TYPE_TO_CATEGORY[cell_type]
    return {"major": "unknown", "minor": "unknown", "color": "#808080", "marker": "."}


def _major_category_palette() -> dict:
    return {"A-fiber": "#1f77b4", "C-fiber": "#ff7f0e", "Non-classified": "#d62728",
            "Non-neuronal": "#2ca02c", "unknown": "#999999"}


# ============================================================================
# LIGHTWEIGHT DEGREE-ONLY "GRAPH" FOR THE NULL MODEL POOL
# ============================================================================
class DegreeOnlyGraph:
    __slots__ = ("_deg",)

    def __init__(self, degree_dict):
        self._deg = degree_dict

    def nodes(self):
        return self._deg.keys()

    def degree(self, n):
        return self._deg[n]

    def __contains__(self, n):
        return n in self._deg

    def __len__(self):
        return len(self._deg)


# ============================================================================
# DEGREE-NORMALIZED LCC SIGNIFICANCE (unchanged math)
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

def _gene_of(node: str) -> str:
    """Bare gene symbol from a possibly-namespaced '<celltype>::GENE' node string."""
    return node.rsplit("::", 1)[-1] if "::" in node else node

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


def sig_lccOfSubgraph_adaptive(eval_graph, pool_graph, seed_nodes, numBins=10, numSimulations=1000, desc=""):
    seed_nodes = [n for n in seed_nodes if n in eval_graph]
    if not seed_nodes:
        return {"lcc": 0, "pVal": 1.0, "zScore": 0.0, "simResults": [], "mean": 0.0, "std": 0.0, "n_seeds_in_graph": 0}

    seeds_in_pool = [n for n in seed_nodes if n in pool_graph]
    if not seeds_in_pool:
        return {"lcc": size_lcc(eval_graph.subgraph(seed_nodes)), "pVal": 1.0, "zScore": 0.0,
                "simResults": [], "mean": 0.0, "std": 0.0, "n_seeds_in_graph": len(seed_nodes)}

    target_dist = getDict_degree2numNodes(pool_graph, nodeList=seeds_in_pool, numBins=numBins)
    obs_lcc = size_lcc(eval_graph.subgraph(seed_nodes))

    sim_sizes = []
    for _ in tqdm(range(200), desc=desc or "LCC sims (phase 1)", leave=False):
        gsim = randSubgraph_preserveDegree(pool_graph, target_dist, eval_graph)
        sim_sizes.append(size_lcc(gsim))

    arr_interim = np.array(sim_sizes)
    p_interim = float(np.mean(arr_interim >= obs_lcc))

    if p_interim > 0.15:
        arr = arr_interim
    else:
        for _ in tqdm(range(numSimulations - 200), desc=desc or "LCC sims (phase 2)", leave=False):
            gsim = randSubgraph_preserveDegree(pool_graph, target_dist, eval_graph)
            sim_sizes.append(size_lcc(gsim))
        arr = np.array(sim_sizes)

    pVal = float(np.mean(arr >= obs_lcc))
    mu, sd = float(arr.mean()), float(arr.std())
    z = (obs_lcc - mu) / sd if sd > 0 else 0.0

    return {"lcc": obs_lcc, "pVal": pVal, "zScore": z, "simResults": sim_sizes,
            "mean": mu, "std": sd, "n_seeds_in_graph": len(seed_nodes)}


# ============================================================================
# PER-COMPONENT METRICS (unchanged)
# ============================================================================
def _component_metrics(component_nodes, graph, seed_node_set, component_id):
    nodes = component_nodes if isinstance(component_nodes, set) else set(component_nodes)
    n = len(nodes)
    seed_count = len(seed_node_set & nodes)
    if n <= 1:
        return {"component_id": component_id, "component_size": n, "num_edges": 0, "density": 0.0,
                "diameter": 0, "seed_count": seed_count, "seed_density": seed_count / n if n else 0.0}
    sub = graph.subgraph(nodes)
    n_edges = sub.number_of_edges()
    density = (2 * n_edges) / (n * (n - 1)) if n > 1 else 0.0
    diameter = 0
    if seed_count > 0 and n < 1000:
        try:
            diameter = nx.diameter(sub) if nx.is_connected(sub) else float("inf")
        except Exception:
            diameter = float("inf")
    return {"component_id": component_id, "component_size": n, "num_edges": n_edges, "density": density,
            "diameter": diameter, "seed_count": seed_count, "seed_density": seed_count / n}


def compute_per_component_metrics(layer_name, eval_graph, seed_nodes, max_components=5000):
    if eval_graph.number_of_nodes() == 0:
        return None
    seed_node_set = set(seed_nodes)
    components = list(nx.connected_components(eval_graph))
    components.sort(key=len, reverse=True)
    component_sizes = [len(c) for c in components]
    singleton_count = sum(1 for s in component_sizes if s == 1)
    components_for_metrics = components[:max_components] if len(components) > max_components else components
    rows, seed_rows = [], []
    for i, comp in enumerate(components_for_metrics):
        m = _component_metrics(comp, eval_graph, seed_node_set, component_id=i)
        m["seed_fraction"] = m["seed_count"] / len(seed_nodes) if seed_nodes else 0.0
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
    return {"components_df": df, "seed_components_df": seed_df, "total_components": len(components),
            "components_with_seeds": int((df["seed_count"] > 0).sum()) if not df.empty else 0,
            "singleton_count": singleton_count, "component_sizes": component_sizes, "entropy": entropy}


# ============================================================================
# LOADING — cell-type PPI TSVs directly, no registry, no giant pickle
# ============================================================================
def load_celltype_layers(wdir: Path) -> dict[str, nx.Graph]:
    print(f"Loading per-cell-type PPI layers from {wdir / 'multiplex'} ...")
    layers = {}
    for ct_dir in sorted((wdir / "multiplex").iterdir()):
        if not ct_dir.is_dir():
            continue
        tsv_path = ct_dir / "ppi_layer.tsv"
        if not tsv_path.exists():
            continue
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["g1", "g2"], dtype=str, on_bad_lines="skip")
        G = nx.Graph()
        for a, b in zip(df["g1"], df["g2"]):
            a, b = str(a).strip().upper(), str(b).strip().upper()
            if a and b and a != b:
                G.add_edge(a, b)
        layers[ct_dir.name] = G
        print(f"  {ct_dir.name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"  {len(layers)} cell type(s) loaded\n")
    return layers


def load_pool_degree(wdir: Path, mode: str, layers: dict[str, nx.Graph]) -> dict[str, dict]:
    """Returns {cell_type: degree_dict} for whichever pool each cell type
    should draw its null model from. mode='reference' (default): every cell
    type shares ONE pool built from gene_reference_graph.pkl (PPI + bipartite,
    see module docstring). mode='per-layer': each cell type's own PPI graph
    is its own pool (no cross-cell-type reach)."""
    if mode == "per-layer":
        return {ct: dict(G.degree()) for ct, G in layers.items()}

    ref_path = wdir / "gene_reference_graph.pkl"
    print(f"Loading combined reference graph for null-model pool: {ref_path} ...")
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)
    ref = nx.relabel_nodes(ref, {n: str(n).strip().upper() for n in ref.nodes()})
    degree_dict = dict(ref.degree())
    print(f"  Reference graph: {ref.number_of_nodes():,} nodes "
          f"(shared pool degree dict for every cell type)\n")
    return {ct: degree_dict for ct in layers.keys()}


def load_seed_genes(path):
    # encoding="utf-8-sig" strips a leading BOM if the CSV was exported from
    # Excel — a BOM silently prepended to the very first header/value is a
    # classic source of "my first gene just never matches anything".
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols_lower = {c.strip().lower(): c for c in df.columns}
    col = df[cols_lower["gene"]] if "gene" in cols_lower else df.iloc[:, 0]
    seeds = (
        col.astype(str)
           .str.strip()
           .str.strip("'\"")   # strip stray wrapping quote characters
           .str.upper()
    )
    seeds = set(seeds)
    seeds.discard("NAN"); seeds.discard("")
    print(f"Loaded {len(seeds):,} seed gene symbols")
    print(f"  e.g.: {sorted(seeds)[:10]}\n")
    return seeds


def diagnose_seed_overlap(seed_symbols: set[str], layers: dict[str, nx.Graph],
                           pool_degree_by_ct: dict[str, dict]) -> set[str]:
    """Reports how many seed genes actually appear anywhere in the loaded
    layers / null-model pool, BEFORE dispatching any (expensive) tasks. If
    overlap with every layer is zero, this is essentially always a naming/ID
    mismatch (not a biological finding), so the caller should stop here
    rather than run the full job to reach the same conclusion."""
    all_layer_nodes: set[str] = set()
    for G in layers.values():
        all_layer_nodes |= set(G.nodes())
    layer_genes = {_gene_of(n) for n in all_layer_nodes}          # NEW
    found_in_layers = seed_symbols & layer_genes  
    print(f"Seed/layer overlap check: {len(found_in_layers):,} / {len(seed_symbols):,} seed genes found "
          f"in the union of all {len(layers)} cell-type PPI layers ({len(all_layer_nodes):,} distinct genes)")

    if pool_degree_by_ct:
        pool_genes = {_gene_of(n) for n in next(iter(pool_degree_by_ct.values())).keys()}   # NEW
        found_in_pool = seed_symbols & pool_genes                                            # was node-level
        print(f"  {len(found_in_pool):,} / {len(seed_symbols):,} seed genes found in the null-model pool")
  
    if not found_in_layers:
        unmatched_sample = sorted(seed_symbols)[:10]
        node_sample = sorted(all_layer_nodes)[:10]
        print("\n  ⚠ ZERO overlap between seed genes and ANY cell-type layer. This is almost certainly a "
              "gene-identifier/formatting mismatch, NOT a biological result — check for:")
        print("      - Ensembl/UniProt IDs in one side vs. gene symbols in the other")
        print("      - wrong column picked up from --seed-genes (only 'gene' or the first column is used)")
        print("      - species mismatch (mouse-case symbols vs. human uppercase, etc.)")
        print("      - stray characters (quotes, BOM, non-breaking spaces) surviving the strip/upper step")
        print(f"    sample seed genes (as loaded): {unmatched_sample}")
        print(f"    sample layer node names:       {node_sample}\n")
    else:
        print()
    return found_in_layers


# ============================================================================
# ANALYSIS
# ============================================================================
def analyze_layer(eval_graph, pool_degree, layer_name, seed_symbols, num_bins=10, num_simulations=1000):
    if eval_graph is None or eval_graph.number_of_nodes() == 0:
        return None
    seed_nodes = [n for n in eval_graph.nodes() if _gene_of(n) in seed_symbols]   # was: n in seed_symbols
    if not seed_nodes:
        return None

    print(f"\n  {layer_name}")
    print(f"    nodes: {eval_graph.number_of_nodes():,} | edges: {eval_graph.number_of_edges():,} | seeds: {len(seed_nodes)}")

    sig = sig_lccOfSubgraph_adaptive(eval_graph, pool_degree, seed_nodes, numBins=num_bins,
                                      numSimulations=num_simulations, desc=f"LCC [{layer_name}]")

    print(f"    LCC: {sig['lcc']} | p={sig['pVal']:.4g} | z={sig['zScore']:.2f} | "
          f"significant: {'YES ✓' if sig['pVal'] < 0.05 else 'no'}")

    cat_info = get_category_from_cell_type(layer_name)

    return {
        "layer": layer_name, "cell_type": layer_name,
        "major_category": cat_info["major"], "minor_category": cat_info["minor"],
        "n_nodes": eval_graph.number_of_nodes(), "n_edges": eval_graph.number_of_edges(),
        "n_seeds": sig["n_seeds_in_graph"], "lcc_size": sig["lcc"], "lcc_p_value": sig["pVal"],
        "lcc_z_score": sig["zScore"], "lcc_random_mean": sig["mean"], "lcc_random_std": sig["std"],
        "significant_lcc": sig["pVal"] < 0.05, "seed_nodes": seed_nodes,
        "eval_graph": eval_graph, "simResults": sig["simResults"],
    }


# ============================================================================
# MULTIPROCESSING
# ============================================================================
_POOL_DEGREE = None


def init_worker(degree_dict):
    global _POOL_DEGREE
    _POOL_DEGREE = DegreeOnlyGraph(degree_dict)


def analyze_layer_worker(args):
    eval_graph, layer_name, seed_symbols, num_bins, num_simulations = args
    try:
        return analyze_layer(eval_graph, _POOL_DEGREE, layer_name, seed_symbols, num_bins=num_bins, num_simulations=num_simulations)
    except Exception as e:
        print(f"ERROR on {layer_name}: {e}")
        traceback.print_exc()
        return None


# ============================================================================
# REPORTING
# ============================================================================
def plot_lcc_results(results, plots_dir):
    """Bars are colored by DRG major_category (same palette as step 7) rather
    than by significance. Significance (p < 0.05) is instead shown via a
    black outline around the bar plus a '*' above it, so both pieces of
    information stay visible at once."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_sorted = sorted(results, key=lambda r: r["lcc_p_value"])
    names = [r["cell_type"] for r in results_sorted]
    lccs = [r["lcc_size"] for r in results_sorted]
    ps = [r["lcc_p_value"] for r in results_sorted]
    sigs = [r["significant_lcc"] for r in results_sorted]
    majors = [r.get("major_category", "unknown") for r in results_sorted]

    palette = _major_category_palette()
    colors = [palette.get(m, "#999999") for m in majors]
    edge_colors = ["black" if s else "none" for s in sigs]
    edge_widths = [1.6 if s else 0 for s in sigs]

    fig, ax1 = plt.subplots(figsize=(max(12, len(names) * 0.5), 6))
    ax2 = ax1.twinx()
    bars = ax1.bar(range(len(names)), lccs, color=colors, alpha=0.85,
                    edgecolor=edge_colors, linewidth=edge_widths, label="LCC size")
    ax1.set_ylabel("LCC size", fontsize=12, fontweight="bold")
    ymax = max(lccs) * 1.2 if lccs else 1
    ax1.set_ylim(0, ymax)

    # asterisk above each significant bar
    for i, (bar, sig) in enumerate(zip(bars, sigs)):
        if sig:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.02,
                      "*", ha="center", va="bottom", fontsize=14, fontweight="bold", color="black")

    # ax2.plot(range(len(names)), ps, "o-", color="#3498DB", lw=2, markersize=6, label="p-value")
    # ax2.axhline(0.05, color="#E74C3C", ls="--", lw=2, alpha=0.7, label="p=0.05 threshold")
    # ax2.set_ylabel("p-value (degree-normalized)", fontsize=12, fontweight="bold")
    # ax2.set_ylim(0, min(1.0, max(ps) * 1.5 if ps else 0.1))
    ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax1.set_xlabel("Cell type", fontsize=12, fontweight="bold")
    ax1.set_title("LCC Significance by Cell Type (colored by major category, "
                  "'*' = p < 0.05)", fontsize=14, fontweight="bold")

    present = set(majors)
    cat_patches = [mpatches.Patch(color=c, label=cat) for cat, c in palette.items() if cat in present]
    line_handles, line_labels = ax2.get_legend_handles_labels()
    ax1.legend(handles=cat_patches, loc="upper left", title="Major category", fontsize=9)
    ax2.legend(line_handles, line_labels, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(plots_dir / "lcc_significance_by_celltype.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot -> {plots_dir / 'lcc_significance_by_celltype.png'}")


def write_summary(results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in r.items() if k not in ("seed_nodes", "eval_graph", "simResults")} for r in results]
    df = pd.DataFrame(rows).sort_values("lcc_p_value")
    df.to_csv(out_dir / "lcc_summary.csv", index=False)

    with open(out_dir / "lcc_summary.txt", "w") as f:
        f.write("=" * 80 + "\nCELL-TYPE LCC SIGNIFICANCE ANALYSIS (DEGREE-NORMALIZED)\n" + "=" * 80 + "\n\n")
        f.write("Interpretation:\n  - LCC: Largest connected component size in seed subgraph\n"
                "  - p-value < 0.05: Cell type has SIGNIFICANT structural role\n"
                "  - z-score: How many std devs above random (degree-normalized)\n\n")
        sig = df[df["significant_lcc"] == True]
        f.write(f"SIGNIFICANT CELL TYPES (p < 0.05): {len(sig)} / {len(df)}\n" + "-" * 80 + "\n")
        for _, r in sig.iterrows():
            f.write(f"  [{r['cell_type']}] ({r.get('major_category', 'unknown')})\n"
                    f"    Nodes: {r['n_nodes']:,} | Edges: {r['n_edges']:,} | Seeds: {r['n_seeds']}\n")
            f.write(f"    LCC: {r['lcc_size']} (z={r['lcc_z_score']:.2f}, p={r['lcc_p_value']:.6f})\n")
            f.write(f"    Random mean +/- std: {r['lcc_random_mean']:.1f} +/- {r['lcc_random_std']:.1f}\n\n")
        f.write("\n" + "=" * 80 + f"\nNON-SIGNIFICANT (p >= 0.05): {len(df) - len(sig)} / {len(df)}\n" + "-" * 80 + "\n")
        for _, r in df[df["significant_lcc"] == False].iterrows():
            f.write(f"  [{r['cell_type']}] ({r.get('major_category', 'unknown')}) LCC={r['lcc_size']}, p={r['lcc_p_value']:.6f}\n")
        f.write("\n" + "=" * 80 + "\nSUMMARY STATISTICS\n" + "-" * 80 + "\n")
        f.write(f"Total cell types analyzed: {len(df)}\nMean LCC size: {df['lcc_size'].mean():.1f}\n"
                f"Median LCC size: {df['lcc_size'].median():.1f}\nMax LCC: {df['lcc_size'].max()}\n"
                f"Cell types with LCC > 0: {(df['lcc_size'] > 0).sum()}\n")
    print(f"Saved summary -> {out_dir / 'lcc_summary.csv'}")
    print(f"Saved report -> {out_dir / 'lcc_summary.txt'}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Degree-normalized LCC significance per cell type (heterogeneous structure)")
    ap.add_argument("--wdir", required=True, help="MultiXrank wdir (contains multiplex/<cell_type>/ppi_layer.tsv)")
    ap.add_argument("--seed-genes", required=True)
    ap.add_argument("--pool-mode", choices=["reference", "per-layer"], default="reference",
                     help="'reference' (default): shared null pool from gene_reference_graph.pkl "
                          "(PPI + bipartite reach, see module docstring). 'per-layer': each cell "
                          "type's own PPI graph is its own pool.")
    ap.add_argument("--num-bins", type=int, default=10)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--output-dir", default="layer_lcc_analysis")
    ap.add_argument("--plots-dir", default="layer_lcc_analysis/plots")
    ap.add_argument("--per-component", action="store_true")
    ap.add_argument("--max-workers", type=int, default=None,
                     help="Cap on worker processes. Defaults to SLURM_CPUS_PER_TASK (or 1).")
    ap.add_argument("--allow-zero-overlap", action="store_true",
                     help="Normally the script exits immediately if --seed-genes has zero overlap with "
                          "every cell-type layer (almost always a naming/ID mismatch, not a real result). "
                          "Pass this to run anyway despite that.")
    args = ap.parse_args()

    t0 = time.time()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CELL-TYPE LCC SIGNIFICANCE ANALYSIS")
    print("=" * 80)

    wdir = Path(args.wdir)
    layers = load_celltype_layers(wdir)
    seed_symbols = load_seed_genes(args.seed_genes)
    pool_degree_by_ct = load_pool_degree(wdir, args.pool_mode, layers)

    found_in_layers = diagnose_seed_overlap(seed_symbols, layers, pool_degree_by_ct)
    if not found_in_layers and not args.allow_zero_overlap:
        raise SystemExit(
            "ERROR: --seed-genes has zero overlap with every cell-type layer — stopping before running "
            "any tasks (see diagnostic above for likely causes). Fix the mismatch, or pass "
            "--allow-zero-overlap to run anyway."
        )

    print("Preparing per-cell-type tasks ...")
    tasks = []
    for ct, G in layers.items():
        if G.number_of_nodes() == 0:
            continue
        tasks.append((G, ct, seed_symbols, args.num_bins, args.num_simulations))
    print(f"Prepared {len(tasks)} tasks\n")

    gc.collect()
    print_memory_usage()

    if args.max_workers is not None:
        n_workers = args.max_workers
    else:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_JOB_CPUS_PER_NODE", 1)))
    n_workers = min(max(1, n_workers), len(tasks)) if tasks else 1
    print(f"Using {n_workers} workers\n")

    # If pool differs per cell type (per-layer mode), we can't share one
    # initializer arg across all workers cleanly with ProcessPoolExecutor's
    # single initargs — fall back to sequential in that case. 'reference'
    # mode (the default) uses one shared pool, so it parallelizes fine.
    if args.pool_mode == "per-layer" and n_workers > 1:
        print("⚠ --pool-mode per-layer gives each cell type its own pool, which the "
              "worker-pool initializer can't share; running sequentially instead.\n")
        n_workers = 1

    print("Analyzing LCC significance:")
    results = []

    if n_workers > 1:
        shared_degree_dict = next(iter(pool_degree_by_ct.values()))
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker, initargs=(shared_degree_dict,)) as executor:
            futures = {executor.submit(analyze_layer_worker, task): task[1] for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cell types analyzed"):
                result = future.result()
                if result:
                    results.append(result)
                gc.collect()
    else:
        for G, ct, seeds_, nb, ns in tqdm(tasks, desc="Cell types analyzed"):
            pool = DegreeOnlyGraph(pool_degree_by_ct[ct])
            result = analyze_layer(G, pool, ct, seeds_, num_bins=nb, num_simulations=ns)
            if result:
                results.append(result)
            gc.collect()

    if not results:
        print("\nNo results produced. Check seed/cell-type matching.")
        return

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    write_summary(results, args.output_dir)
    plot_lcc_results(results, args.plots_dir)

    if args.per_component:
        print("\nComputing per-component metrics (significant cell types only)...")
        comp_dir = Path(args.output_dir) / "components"
        comp_dir.mkdir(exist_ok=True)
        sig_results = [r for r in results if r["significant_lcc"]]
        nonsig_sample = [r for r in results if not r["significant_lcc"]][:3]
        for r in sig_results + nonsig_sample:
            comp = compute_per_component_metrics(r["layer"], r["eval_graph"], r["seed_nodes"])
            if comp and not comp["components_df"].empty:
                comp["components_df"].to_csv(comp_dir / f"components_{r['layer']}.csv", index=False)
        print(f"Saved per-component data -> {comp_dir}")

    print(f"\nCompleted in {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    sys.setrecursionlimit(40000)
    main()