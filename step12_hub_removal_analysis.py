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

CHANGES FROM ORIGINAL (perf, from a previous pass -- no change in what is
measured/reported unless you opt into the approximation flags below):

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

NEW IN THIS VERSION -- mechanism-level diagnostics for "why is this module
more fragile than a degree-matched average?" These are all cheap
(O(V+E)-ish) relative to the attack-curve/null machinery already here, so
they're computed by default rather than gated behind flags, except the
ablation comparison (needs you to name a cell type) and the topology-hub
percentile (has a CLI default but is overridable):

  6. Degree assortativity (real module + every null replicate). A
     disease module whose hubs preferentially connect to LOW-degree
     nodes (disassortative / more negative r than the null distribution)
     is expected to be more fragile at matched degree sequence, because
     those hubs are acting as sole bridges for peripheral nodes with no
     redundant path. -> module_topology_stats.csv, null_topology_stats.csv,
     null_topology_comparison.csv.

  7. Articulation points, on the real module and on every null replicate,
     plus their overlap with each hub definition (what fraction of a
     given hub set IS a cut vertex). -> hub_articulation_overlap.csv,
     module_topology_stats.csv / null_topology_comparison.csv.

  8. Hub-neighborhood clustering coefficient (top --topology-hub-pct by
     degree), real module vs. null distribution. Low clustering around
     hubs means no local backup path once the hub is removed.
     -> module_topology_stats.csv / null_topology_comparison.csv.

  9. Layer participation: for every node, the fraction of its edges that
     are INTER-layer (the bipartite identity edge joining the same gene
     across cell types) vs. INTRA-layer (a real PPI edge within one cell
     type). Cross-referenced against every hub definition, since a "hub"
     that is mostly inter-layer edges is a hub because it exists in many
     cell types, not because it's a real within-layer PPI hub -- that
     distinction matters for biological interpretation.
     -> layer_participation.csv, hub_layer_participation_summary.csv.

 10. Per-cell-type knockout: remove every node belonging to a single
     cell-type layer (independent of any degree/betweenness hub
     definition) and record the same fragmentation stats as the hub
     knockouts. Tells you which layer's node complement is actually
     driving fragility. -> celltype_knockout_summary.csv.

 11. Optional layer-ablation attack-curve comparison: rerun the full
     attack-curve suite on the module with one or more cell types
     stripped out first (--ablation-exclude-celltypes), so you can see
     how much of the measured fragility depends on that layer being
     present in the final module at all. This ablates from the already-
     built MODULE graph (post-bridging/LCC) -- for a from-scratch
     ablation at the reference-graph level, rerun step6 with
     --exclude-celltypes / --exclude-ppi-layer instead and point this
     script at the resulting module_graph.pkl.
     -> ablation_f_c_f90.csv, ablation_attack_curve_<mode>_<tag>.csv.

 12. Seed-driven vs. bridging-gene-driven fragility: every hub definition
     is additionally split into its seed-gene members vs. its non-seed
     (RWR-bridging-gene) members, and each subset is knocked out
     separately. Tells you whether the fragility-driving hubs are
     biologically-anchored seeds or genes that were only pulled in to
     connect disconnected seed components -- if it's mostly the latter,
     revisit --max-bridge / the bridging elbow from step6 before treating
     the fragility as a biological finding.
     -> hub_knockout_by_seed_status.csv. Requires --disease-module (for
     the is_seed column) -- skipped with a note if not provided.

 13. Molloy-Reed ratio (<k^2>/<k>) reported once for the real module, with
     an explicit note: because the null is degree-PRESERVING
     (double_edge_swap only rewires which edges connect which nodes, not
     the degree sequence), this ratio -- and the percolation threshold it
     implies -- is identical in expectation between module and null by
     construction. So if f_c/f_90 still differ from the null (see
     observed_vs_null_comparison.csv), that divergence is coming from
     topology/wiring (items 6-9), not from the degree distribution
     itself. This is a one-line sanity check, not a new statistical test.

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
    --ablation-exclude-celltypes global_ppi \
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
# NEW: cheap topology diagnostics (assortativity, articulation points,
# hub-neighborhood clustering) -- shared by the real module and every null
# replicate so they can be compared directly.
# --------------------------------------------------------------------------

def molloy_reed_ratio(G):
    """<k^2> / <k> over the degree sequence. This is the standard
    percolation-theory robustness proxy (Molloy-Reed criterion / Cohen et
    al.): higher values mean a theoretically more robust degree
    distribution. Because the degree-preserving null (double_edge_swap)
    keeps the exact degree sequence, this value is identical in
    expectation for the module and every null replicate BY CONSTRUCTION --
    it's reported once, as a sanity anchor, not as a per-null comparison.
    If f_c/f_90 still diverge from the null despite this being fixed, the
    divergence must come from wiring (assortativity, articulation points,
    hub clustering), not from the degree distribution itself.
    """
    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.size == 0 or degs.mean() == 0:
        return np.nan
    return float(np.mean(degs ** 2) / np.mean(degs))


def _articulation_points_any(H):
    """nx.articulation_points requires a connected graph; run it per
    connected component so this also works on a module/null graph that
    isn't fully connected (components of size <= 2 can't have cut
    vertices by definition, so they're skipped)."""
    points = []
    for comp in nx.connected_components(H):
        if len(comp) > 2:
            sub = H.subgraph(comp)
            points.extend(nx.articulation_points(sub))
    return points


def compute_topology_stats(H, hub_pct=0.10, include_expensive=False, expensive_max_nodes=3000):
    """
    Bundle of graph-level structural diagnostics, computed identically
    for the real module and every null replicate so they can be
    compared directly (see degree_preserving_null_curves /
    null_topology_comparison.csv).

    ALWAYS computed (cheap, O(V+E)-ish -- negligible next to the
    betweenness/attack-curve costs already dominating this script):
      - assortativity : degree assortativity coefficient. More negative
        = hubs preferentially attach to low-degree nodes (disassortative),
        a classic driver of fragility beyond what degree sequence alone
        predicts.
      - n_articulation_points / frac_articulation_points : cut vertices.
      - n_bridges / frac_bridges : the edge-level analogue of
        articulation points -- edges whose removal alone disconnects a
        component.
      - max_core_number / mean_core_number : k-core decomposition
        (nx.core_number). How deep the module's densely-interconnected
        'core' goes; a shallow core relative to the null means the
        module is more tree-like / bottleneck-prone than its degree
        sequence would suggest.
      - hub_neighbor_clustering : mean clustering coefficient of the top
        `hub_pct` nodes by degree IN THIS GRAPH. Low clustering means a
        hub's neighbors aren't otherwise connected to each other, so
        there's no backup path once the hub is removed.
      - avg_clustering / transitivity : module-wide (not just
        hub-neighborhood) redundancy measures.
      - molloy_reed_ratio : see molloy_reed_ratio() docstring.

    Only computed when include_expensive=True AND the graph has at most
    `expensive_max_nodes` nodes (these scale much worse than the above,
    so they're gated to keep a null benchmark of dozens of replicates
    tractable -- see degree_preserving_null_curves' n_null_expensive_metrics):
      - node_connectivity / edge_connectivity : global min-cut size
        (nodes / edges) -- how many removals it takes to disconnect ANY
        two nodes in the graph. A stricter, degree-independent analogue
        of "is there an articulation point" (0 for an already-
        disconnected graph, by definition).
      - algebraic_connectivity : second-smallest Laplacian eigenvalue
        (Fiedler value) of the graph's LARGEST CONNECTED COMPONENT -- a
        classic global, degree-independent robustness measure.
      - global_efficiency : nx.global_efficiency(H) (on the full graph).
      - avg_shortest_path_length : on the largest connected component
        (undefined for a disconnected graph as a whole).
    """
    n = H.number_of_nodes()
    stats = {
        "n_nodes": n,
        "n_edges": H.number_of_edges(),
        "assortativity": np.nan,
        "n_articulation_points": 0,
        "frac_articulation_points": 0.0,
        "n_bridges": 0,
        "frac_bridges": 0.0,
        "max_core_number": 0,
        "mean_core_number": np.nan,
        "hub_neighbor_clustering": np.nan,
        "avg_clustering": np.nan,
        "transitivity": np.nan,
        "molloy_reed_ratio": molloy_reed_ratio(H) if n else np.nan,
        "node_connectivity": np.nan,
        "edge_connectivity": np.nan,
        "algebraic_connectivity": np.nan,
        "global_efficiency": np.nan,
        "avg_shortest_path_length": np.nan,
        "expensive_metrics_computed": False,
    }
    if n < 3 or H.number_of_edges() == 0:
        return stats

    try:
        stats["assortativity"] = nx.degree_assortativity_coefficient(H)
    except Exception:
        pass

    artic = _articulation_points_any(H)
    stats["n_articulation_points"] = len(artic)
    stats["frac_articulation_points"] = len(artic) / n

    try:
        n_bridges = sum(1 for _ in nx.bridges(H))
        stats["n_bridges"] = n_bridges
        stats["frac_bridges"] = n_bridges / H.number_of_edges()
    except Exception:
        pass

    try:
        core = nx.core_number(H)
        if core:
            core_vals = np.array(list(core.values()), dtype=float)
            stats["max_core_number"] = int(core_vals.max())
            stats["mean_core_number"] = float(core_vals.mean())
    except Exception:
        pass

    degree = dict(H.degree())
    n_top = max(1, int(np.ceil(hub_pct * n)))
    ranked = sorted(H.nodes(), key=lambda x: degree[x], reverse=True)
    hub_nodes = ranked[:n_top]
    clustering = nx.clustering(H, hub_nodes)
    stats["hub_neighbor_clustering"] = float(np.mean(list(clustering.values()))) if clustering else np.nan
    try:
        stats["avg_clustering"] = nx.average_clustering(H)
    except Exception:
        pass
    try:
        stats["transitivity"] = nx.transitivity(H)
    except Exception:
        pass

    if include_expensive and n <= expensive_max_nodes:
        stats["expensive_metrics_computed"] = True
        try:
            stats["node_connectivity"] = nx.node_connectivity(H)
        except Exception as e:
            print(f"    [topology] node_connectivity failed: {e}")
        try:
            stats["edge_connectivity"] = nx.edge_connectivity(H)
        except Exception as e:
            print(f"    [topology] edge_connectivity failed: {e}")

        try:
            largest_cc_nodes = max(nx.connected_components(H), key=len)
            H_lcc = H.subgraph(largest_cc_nodes)
            if H_lcc.number_of_nodes() >= 3:
                try:
                    stats["algebraic_connectivity"] = nx.algebraic_connectivity(H_lcc)
                except Exception as e:
                    print(f"    [topology] algebraic_connectivity failed: {e}")
                try:
                    stats["avg_shortest_path_length"] = nx.average_shortest_path_length(H_lcc)
                except Exception as e:
                    print(f"    [topology] avg_shortest_path_length failed: {e}")
        except Exception:
            pass
        try:
            stats["global_efficiency"] = nx.global_efficiency(H)
        except Exception as e:
            print(f"    [topology] global_efficiency failed: {e}")

    return stats


# --------------------------------------------------------------------------
# NEW: layer participation -- inter-layer (cross-cell-type identity edge)
# vs. intra-layer (within-cell-type PPI edge) composition of each node's
# neighborhood.
# --------------------------------------------------------------------------

def compute_layer_participation(G):
    """For every node, the fraction of its edges that connect to a
    DIFFERENT cell-type layer (an inter-layer/'identity' edge joining the
    same gene across cell types) vs. the SAME cell-type layer (an
    intra-layer PPI edge). A node whose edges are mostly inter-layer is a
    hub because it exists in many cell types, not because it's a highly
    connected PPI hub within any one layer -- worth knowing before
    interpreting a 'hub' biologically as a within-tissue signaling
    bottleneck.
    """
    rows = []
    for n in G.nodes():
        ct_n = _celltype_of(n)
        gene_n = _bare_gene(n)
        neighbors = list(G.neighbors(n))
        deg = len(neighbors)
        if deg == 0:
            rows.append({
                "node": n, "gene": gene_n, "cell_type": ct_n, "degree": 0,
                "n_inter_layer": 0, "n_intra_layer": 0, "frac_inter_layer": np.nan,
            })
            continue
        n_inter = sum(1 for m in neighbors if _celltype_of(m) != ct_n)
        n_intra = deg - n_inter
        rows.append({
            "node": n, "gene": gene_n, "cell_type": ct_n, "degree": deg,
            "n_inter_layer": n_inter, "n_intra_layer": n_intra,
            "frac_inter_layer": n_inter / deg,
        })
    return pd.DataFrame(rows)


def hub_sets_layer_participation_summary(hub_sets, participation_df):
    """For every hub definition, the mean frac_inter_layer of its member
    nodes vs. the module-wide mean -- a quick check for whether a given
    hub set is dominated by cross-layer connector nodes rather than
    genuine within-layer PPI hubs."""
    module_mean = participation_df["frac_inter_layer"].mean(skipna=True)
    lookup = participation_df.set_index("node")["frac_inter_layer"]
    rows = []
    for name, nodes in hub_sets.items():
        vals = lookup.reindex(list(nodes)).dropna()
        rows.append({
            "hub_definition": name,
            "n_nodes": len(nodes),
            "mean_frac_inter_layer": float(vals.mean()) if len(vals) else np.nan,
            "module_wide_mean_frac_inter_layer": module_mean,
            "difference_from_module_mean": (float(vals.mean()) - module_mean) if len(vals) else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# NEW: bridge-gene dependency diagnostics -- is fragility concentrated in
# the genes step6's greedy bridging loop added, rather than in the seeds?
# --------------------------------------------------------------------------

def compute_removal_impact(G, nodes_to_test):
    """
    For each node in `nodes_to_test`, remove it ALONE and record the
    resulting LCC fraction (relative to the full module) as
    1 - lcc_fraction_after -- a degree-independent "removal impact"
    ranking of which single nodes matter most structurally. Scoped to a
    bounded node set (articulation points, hub-set members, etc.) rather
    than every node in the module: a full O(V) single-node-removal sweep
    is O(V*(V+E)), and a non-hub, non-cut-vertex node's impact is
    essentially always ~0 by construction, so it isn't worth the cost.
    """
    n_original = G.number_of_nodes()
    impact = {}
    for node in nodes_to_test:
        if node not in G:
            continue
        H = G.copy()
        H.remove_node(node)
        _, lcc_frac = _lcc_size_and_fraction(H, n_original)
        impact[node] = 1.0 - lcc_frac
    return impact


def compute_node_level_diagnostics(G, degree, betweenness, articulation_set,
                                    seed_node_set, core_numbers, impact_dict):
    """
    One row per module node: is_seed / is_bridge_added, degree,
    betweenness, core_number, is_articulation, delta_lcc (single-node
    removal impact from compute_removal_impact; NaN where not computed).

    NOTE on is_bridge_added: in this pipeline every non-seed node in the
    final module got there via step6's greedy RWR-ranked bridging loop
    (module = seeds_in_h + bridging genes, nothing else) -- so here
    is_bridge_added is exactly `not is_seed`, not a third category. The
    advisor's "seeds vs. bridge-added vs. other non-seed" three-way split
    collapses to a two-way split for this specific pipeline; this table
    (and knockout_by_origin, below) is what actually distinguishes them.
    """
    rows = []
    for n in G.nodes():
        is_seed = n in seed_node_set
        rows.append({
            "node": n,
            "gene": _bare_gene(n),
            "cell_type": _celltype_of(n),
            "is_seed": is_seed,
            "is_bridge_added": not is_seed,
            "degree": degree.get(n, np.nan),
            "betweenness": betweenness.get(n, np.nan),
            "core_number": core_numbers.get(n, np.nan),
            "is_articulation": n in articulation_set,
            "delta_lcc": impact_dict.get(n, np.nan),
        })
    return pd.DataFrame(rows)


def summarize_node_diagnostics_by_origin(node_diag_df):
    """Group node_level_diagnostics.csv by is_seed and report
    mean/median degree, betweenness, core_number, delta_lcc, plus the
    fraction of each group that's an articulation point -- the
    seed-vs-bridge-added distributional comparison the advisor asked
    for."""
    rows = []
    for is_seed, label in [(True, "seed"), (False, "bridge_added")]:
        sub = node_diag_df[node_diag_df["is_seed"] == is_seed]
        if sub.empty:
            continue
        rows.append({
            "origin": label,
            "n_nodes": len(sub),
            "mean_degree": sub["degree"].mean(),
            "median_degree": sub["degree"].median(),
            "mean_betweenness": sub["betweenness"].mean(),
            "median_betweenness": sub["betweenness"].median(),
            "mean_core_number": sub["core_number"].mean(),
            "frac_articulation_points": sub["is_articulation"].mean(),
            "mean_delta_lcc": sub["delta_lcc"].mean(skipna=True),
            "max_delta_lcc": sub["delta_lcc"].max(skipna=True),
        })
    return pd.DataFrame(rows)


def knockout_by_origin(G, seed_node_set):
    """Single-shot knockout of the ENTIRE seed set vs. the ENTIRE
    bridge-added (non-seed) set -- not just their hub-defined subsets
    (see knockout_hub_sets_by_seed_status for that). Answers: does the
    module depend more on its biologically-anchored seed genes, or on
    the genes step6's bridging step pulled in to connect them?"""
    n_original = G.number_of_nodes()
    all_nodes = set(G.nodes())
    groups = {
        "all_seeds": set(seed_node_set) & all_nodes,
        "all_bridge_added": all_nodes - set(seed_node_set),
    }
    rows = []
    for label, subset in groups.items():
        if not subset:
            continue
        H = G.copy()
        H.remove_nodes_from(subset)
        n_components = nx.number_connected_components(H) if H.number_of_nodes() > 0 else 0
        _, lcc_frac = _lcc_size_and_fraction(H, n_original)
        rows.append({
            "origin_group": label,
            "n_nodes_removed": len(subset),
            "frac_removed": len(subset) / n_original,
            "n_components_after": n_components,
            "lcc_fraction_after": lcc_frac,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# NEW: per-cell-type knockout (independent of any hub definition)
# --------------------------------------------------------------------------

def celltype_knockout(G):
    """Remove every node belonging to a single cell-type layer at a time
    (rather than a degree/betweenness-defined hub set) and record the same
    fragmentation stats as knockout_hub_sets. Tells you which cell type's
    node complement is driving fragility, independent of how 'hub' is
    defined."""
    n_original = G.number_of_nodes()
    celltypes = sorted({_celltype_of(n) for n in G.nodes()})
    rows = []
    for ct in celltypes:
        nodes_ct = [n for n in G.nodes() if _celltype_of(n) == ct]
        H = G.copy()
        H.remove_nodes_from(nodes_ct)
        n_components = nx.number_connected_components(H) if H.number_of_nodes() > 0 else 0
        _, lcc_frac = _lcc_size_and_fraction(H, n_original)
        rows.append({
            "cell_type": ct,
            "n_nodes_removed": len(nodes_ct),
            "frac_removed": len(nodes_ct) / n_original,
            "n_components_after": n_components,
            "lcc_fraction_after": lcc_frac,
        })
    return pd.DataFrame(rows).sort_values("lcc_fraction_after").reset_index(drop=True)


def celltype_subgraph_fragility(G, modes=("degree",), step=1):
    """
    For each cell type, take its INDUCED subgraph (only that cell type's
    nodes and the intra-layer PPI edges between them -- inter-layer
    identity edges never appear here, since their other endpoint is a
    different cell type and so isn't in the induced node set) and run
    attack curves on it as its OWN isolated network. This is different
    from celltype_knockout (which measures damage to the FULL module
    when a layer is stripped out) -- this measures how fragile that
    layer would be entirely on its own.
    """
    rows = []
    celltypes = sorted({_celltype_of(n) for n in G.nodes()})
    for ct in celltypes:
        nodes_ct = [n for n in G.nodes() if _celltype_of(n) == ct]
        sub = G.subgraph(nodes_ct).copy()
        if sub.number_of_nodes() < 3 or sub.number_of_edges() == 0:
            for mode in modes:
                rows.append({"cell_type": ct, "n_nodes": sub.number_of_nodes(),
                             "n_edges": sub.number_of_edges(), "mode": mode,
                             "f_c": np.nan, "f_90": np.nan})
            continue
        for mode in modes:
            df = attack_curve(sub, mode=mode, cascading=False, step=step)
            fc, f90 = critical_fraction(df)
            rows.append({"cell_type": ct, "n_nodes": sub.number_of_nodes(),
                         "n_edges": sub.number_of_edges(), "mode": mode,
                         "f_c": fc, "f_90": f90})
    return pd.DataFrame(rows)


def celltype_hub_knockout(G, hub_sets):
    """
    Extends hub_sets_celltype_breakdown (which just counts hub-set
    membership per cell type) into actual damage: for every
    (cell_type, hub_definition) pair, remove ONLY that cell type's
    members of that hub set from the FULL module and record the
    resulting LCC fraction -- 'delta_lcc by cell type' for each hub
    definition, so you can see whether a hub set's damage is concentrated
    in one neuron population or spread across layers.
    """
    n_original = G.number_of_nodes()
    celltypes = sorted({_celltype_of(n) for n in G.nodes()})
    rows = []
    for name, hub_nodes in hub_sets.items():
        for ct in celltypes:
            ct_hub_nodes = {n for n in hub_nodes if _celltype_of(n) == ct}
            if not ct_hub_nodes:
                continue
            H = G.copy()
            H.remove_nodes_from(ct_hub_nodes)
            n_components = nx.number_connected_components(H) if H.number_of_nodes() > 0 else 0
            _, lcc_frac = _lcc_size_and_fraction(H, n_original)
            rows.append({
                "hub_definition": name,
                "cell_type": ct,
                "n_hub_nodes_removed": len(ct_hub_nodes),
                "frac_removed": len(ct_hub_nodes) / n_original,
                "n_components_after": n_components,
                "lcc_fraction_after": lcc_frac,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# NEW: layer-ablation attack-curve comparison
# --------------------------------------------------------------------------

def ablate_celltypes_and_compare(G, celltypes_to_exclude, modes, step, cascading_modes,
                                  betweenness_k, seed, output_path):
    """Remove all nodes from the given cell-type layer(s) out of the
    (already built) module graph, then rerun the same attack-curve suite
    on what's left. This tells you how much of the fragility measured for
    the full module depended on that layer being present -- e.g. is
    'global_ppi' propping up connectivity, or contributing
    disproportionately to the fragile wiring?

    NOTE: this ablates from the already-built MODULE graph (post
    bridging/LCC), not from the full multiplex reference graph -- for a
    from-scratch ablation, rerun step6_build_disease_module.py with
    --exclude-celltypes / --exclude-ppi-layer and point this script at the
    resulting module_graph.pkl instead.
    """
    exclude_upper = {c.strip().upper() for c in celltypes_to_exclude}
    keep_nodes = [n for n in G.nodes() if _celltype_of(n).upper() not in exclude_upper]
    G_ablated = G.subgraph(keep_nodes).copy()
    print(f"  [ablation] Excluding cell type(s) {sorted(exclude_upper)}: "
          f"{G.number_of_nodes():,} -> {G_ablated.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} -> {G_ablated.number_of_edges():,} edges")

    if G_ablated.number_of_nodes() == 0:
        print("  [ablation] Nothing left after exclusion -- skipping.")
        return None, None

    curves = {}
    fc_f90_rows = []
    for mode in modes:
        df_static = attack_curve(G_ablated, mode=mode, cascading=False, step=step,
                                  betweenness_k=betweenness_k, seed=seed)
        curves[(mode, False)] = df_static
        fc, f90 = critical_fraction(df_static)
        fc_f90_rows.append({"mode": mode, "cascading": False, "f_c": fc, "f_90": f90})

        if mode in cascading_modes:
            df_casc = attack_curve(G_ablated, mode=mode, cascading=True, step=step,
                                    betweenness_k=betweenness_k, seed=seed)
            curves[(mode, True)] = df_casc
            fc_c, f90_c = critical_fraction(df_casc)
            fc_f90_rows.append({"mode": mode, "cascading": True, "f_c": fc_c, "f_90": f90_c})

    fc_f90_df = pd.DataFrame(fc_f90_rows)
    fc_f90_df.to_csv(os.path.join(output_path, "ablation_f_c_f90.csv"), index=False)
    for key, df in curves.items():
        mode, cascading = key
        tag = "cascading" if cascading else "static"
        df.to_csv(os.path.join(output_path, f"ablation_attack_curve_{mode}_{tag}.csv"), index=False)
    print(f"  [ablation] f_c/f_90 saved -> ablation_f_c_f90.csv")
    return fc_f90_df, curves


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


def hub_sets_articulation_overlap(hub_sets, articulation_set):
    """For every hub definition, what fraction of its member nodes are
    articulation points (cut vertices) of the real module. A hub
    definition that is mostly articulation points is the clearest
    possible mechanistic explanation for fragility: removing those nodes
    doesn't just lower degree, it structurally disconnects the graph."""
    rows = []
    for name, nodes in hub_sets.items():
        n_artic = sum(1 for n in nodes if n in articulation_set)
        rows.append({
            "hub_definition": name,
            "n_hub_nodes": len(nodes),
            "n_hub_nodes_that_are_articulation_points": n_artic,
            "frac_hub_nodes_that_are_articulation_points": (n_artic / len(nodes)) if nodes else np.nan,
        })
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
     betweenness_k, swap_factor, topology_hub_pct, compute_expensive,
     expensive_max_nodes, seed) = args
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from(edges)
    n_edges = H.number_of_edges()
    n_swaps = max(1, swap_factor * n_edges)
    try:
        nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)
    except nx.NetworkXError as e:
        print(f"  [null {null_id}] double_edge_swap warning: {e}")

    # Cheap topology stats (assortativity, articulation points, bridges,
    # k-core, clustering/transitivity...) on every null replicate, plus
    # the expensive ones (connectivity, algebraic connectivity,
    # efficiency) only for the first `n_null_expensive_metrics`
    # replicates (compute_expensive) -- computed once per null graph
    # (not once per attack mode) and carried on every row for this
    # null_id so the per-mode f_c/f_90 rows and the topology numbers can
    # be joined/filtered together downstream.
    topo = compute_topology_stats(H, hub_pct=topology_hub_pct,
                                   include_expensive=compute_expensive,
                                   expensive_max_nodes=expensive_max_nodes)

    rows = []
    for mode in modes:
        df = attack_curve(H, mode=mode, cascading=False, step=step)
        fc, f90 = critical_fraction(df)
        rows.append({"null_id": null_id, "mode": mode, "cascading": False, "f_c": fc, "f_90": f90, **topo})
        if mode in cascading_modes:
            df_c = attack_curve(H, mode=mode, cascading=True, step=step,
                                 cascading_step=cascading_step,
                                 betweenness_k=betweenness_k, seed=seed)
            fc_c, f90_c = critical_fraction(df_c)
            rows.append({"null_id": null_id, "mode": mode, "cascading": True, "f_c": fc_c, "f_90": f90_c, **topo})
    return rows


def degree_preserving_null_curves(G, modes, n_null=20, n_null_cascading=None,
                                   cascading_modes=(), step=1, cascading_step=None,
                                   betweenness_k=None, swap_factor=10, seed=0, n_jobs=1,
                                   topology_hub_pct=0.10, n_null_expensive_metrics=None,
                                   expensive_max_nodes=3000):
    """
    Generate degree-preserving rewired graphs (double_edge_swap) and run
    attack modes on each, recording f_c / f_90 for comparison against the
    real module. Each replicate also gets the topology_stats bundle
    (assortativity, articulation points, bridges, k-core, clustering,
    transitivity, molloy-reed -- and, for the first
    `n_null_expensive_metrics` replicates, node/edge connectivity,
    algebraic connectivity, global efficiency and avg shortest path
    length) so those can be compared against the real module's values the
    same way f_c/f_90 are.

    Static modes run on `n_null` replicates; cascading modes (the
    expensive ones for betweenness-based rankings) run on only the first
    `n_null_cascading` of those replicates if given (defaults to n_null).
    The expensive topology metrics (connectivity/efficiency/algebraic
    connectivity) run on only the first `n_null_expensive_metrics`
    replicates if given (defaults to min(5, n_null), since these scale
    much worse than everything else here and a handful of replicates is
    enough for a null_median/null_std estimate). Report null_std in the
    output -- fewer replicates means a wider (still valid) null CI, which
    the downstream comparison CSVs already surface.

    double_edge_swap rewires edges among the EXISTING node set, so
    "celltype::gene" node identities (and thus which layer each node
    belongs to) are preserved in the null graphs -- only which edges
    connect them changes. NOTE: this null is degree-preserving but NOT
    layer-aware -- it can (and typically will) swap an intra-layer PPI
    edge for what looks like a cross-layer connection in the same graph
    object, i.e. it does not respect the multiplex's edge-type semantics.
    For a null that does, see step12b_layer_preserving_null.py.
    """
    if n_null_cascading is None:
        n_null_cascading = n_null
    if n_null_expensive_metrics is None:
        n_null_expensive_metrics = min(5, n_null)

    edges = list(G.edges())
    nodes = list(G.nodes())
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n_null):
        this_cascading_modes = cascading_modes if i < n_null_cascading else set()
        compute_expensive = i < n_null_expensive_metrics
        tasks.append((i, edges, nodes, modes, this_cascading_modes, step, cascading_step,
                      betweenness_k, swap_factor, topology_hub_pct, compute_expensive,
                      expensive_max_nodes, int(rng.integers(0, 2**31 - 1))))

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


def knockout_hub_sets_by_seed_status(G, hub_sets, seed_node_set):
    """Same idea as knockout_hub_sets, but each hub definition is
    additionally split into its seed-gene members vs. its non-seed
    (RWR-bridging-gene) members, and each subset is knocked out
    separately. This distinguishes fragility driven by biologically
    anchored seed hubs from fragility driven by genes that only entered
    the module to connect disconnected seed components -- if it's mostly
    the latter, the fragility result is partly an artifact of the
    bridging step (--max-bridge / the elbow point from step6), not a
    biological property of the disease module."""
    n_original = G.number_of_nodes()
    rows = []
    for name, hub_nodes in hub_sets.items():
        seed_hubs = {n for n in hub_nodes if n in seed_node_set}
        nonseed_hubs = hub_nodes - seed_hubs
        for subset_name, subset in [("seed_only", seed_hubs), ("nonseed_bridging_only", nonseed_hubs)]:
            if not subset:
                continue
            H = G.copy()
            H.remove_nodes_from(subset)
            n_components = nx.number_connected_components(H) if H.number_of_nodes() > 0 else 0
            _, lcc_frac = _lcc_size_and_fraction(H, n_original)
            rows.append({
                "hub_definition": name,
                "subset": subset_name,
                "n_nodes_removed": len(subset),
                "frac_removed": len(subset) / n_original,
                "n_components_after": n_components,
                "lcc_fraction_after": lcc_frac,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# NEW: community structure -- connector vs. provincial hubs (Guimera-Amaral)
# --------------------------------------------------------------------------

def compute_community_structure(G, seed=0):
    """
    Louvain community detection + modularity, then Guimera-Amaral roles
    for every node:
      - within-module degree z-score z_i = (k_i - mean(k within i's
        community)) / std(k within i's community), where k_i counts only
        edges to i's OWN community.
      - participation coefficient P_i = 1 - sum_s (k_is / k_i)^2, where
        k_is is i's degree into community s (all of i's neighbors, not
        just within-community).
    and a simplified 4-way role label using the standard Guimera-Amaral
    cutoffs (z >= 2.5 for 'hub', P >= 0.62 for 'connector'):
      connector_hub, provincial_hub, connector_non_hub, peripheral
    Fragility from removing CONNECTOR hubs is expected to fragment the
    module (they bridge communities); PROVINCIAL hubs are more
    replaceable (high degree, but embedded in one community).

    Returns (community_summary_df, node_roles_df), or (None, None) if no
    community-detection implementation is available (networkx < 2.8
    without louvain_communities, and python-louvain not installed).
    """
    communities = None
    try:
        communities = nx.algorithms.community.louvain_communities(G, seed=seed)
    except AttributeError:
        try:
            import community as community_louvain  # python-louvain package
            partition = community_louvain.best_partition(G, random_state=seed)
            comm_ids = sorted(set(partition.values()))
            communities = [{n for n, c in partition.items() if c == cid} for cid in comm_ids]
        except ImportError:
            print("  [community] Neither networkx's louvain_communities nor the "
                  "python-louvain package is available -- skipping community structure "
                  "analysis. `pip install python-louvain` or upgrade networkx>=2.8 to enable.")
            return None, None

    try:
        modularity = nx.algorithms.community.modularity(G, communities)
    except Exception as e:
        print(f"  [community] modularity computation failed: {e}")
        modularity = np.nan

    node_to_comm = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            node_to_comm[n] = cid

    degree = dict(G.degree())
    comm_summary_rows = []
    for cid, comm in enumerate(communities):
        conductance = np.nan
        try:
            if 0 < len(comm) < G.number_of_nodes():
                conductance = nx.algorithms.cuts.conductance(G, comm)
        except Exception:
            pass
        comm_summary_rows.append({
            "community_id": cid, "n_nodes": len(comm),
            "dominant_cell_type": (pd.Series([_celltype_of(n) for n in comm]).mode().iloc[0]
                                    if comm else "n/a"),
            "conductance": conductance,
        })
    comm_summary_df = pd.DataFrame(comm_summary_rows)
    comm_summary_df["modularity"] = modularity

    # Within-community degree (edges to same-community neighbors only),
    # used for the z-score.
    within_deg = {n: 0 for n in G.nodes()}
    for u, v in G.edges():
        if node_to_comm.get(u) == node_to_comm.get(v):
            within_deg[u] += 1
            within_deg[v] += 1
    comm_within_deg_stats = {}
    for cid, comm in enumerate(communities):
        vals = np.array([within_deg[n] for n in comm], dtype=float)
        comm_within_deg_stats[cid] = (vals.mean(), vals.std())

    node_rows = []
    for n in G.nodes():
        cid = node_to_comm.get(n)
        k_i = degree[n]
        mean_k, std_k = comm_within_deg_stats.get(cid, (np.nan, np.nan))
        z = (within_deg[n] - mean_k) / std_k if std_k and std_k > 0 else 0.0

        if k_i > 0:
            neighbor_comms = pd.Series([node_to_comm.get(m) for m in G.neighbors(n)])
            comm_frac = neighbor_comms.value_counts(normalize=True)
            participation = 1.0 - float((comm_frac ** 2).sum())
        else:
            participation = 0.0

        if z >= 2.5 and participation >= 0.62:
            role = "connector_hub"
        elif z >= 2.5:
            role = "provincial_hub"
        elif participation >= 0.62:
            role = "connector_non_hub"
        else:
            role = "peripheral"

        node_rows.append({
            "node": n, "community_id": cid, "degree": k_i,
            "within_module_z": z, "participation_coefficient": participation,
            "role": role,
        })
    node_roles_df = pd.DataFrame(node_rows)
    return comm_summary_df, node_roles_df


def hub_sets_community_role_summary(hub_sets, node_roles_df):
    """For every hub definition, the composition of Guimera-Amaral roles
    among its members -- connector hubs (bridge communities) are the ones
    whose removal is expected to fragment the module; provincial hubs
    (high degree, embedded in one community) are more replaceable."""
    if node_roles_df is None or node_roles_df.empty:
        return None
    role_lookup = node_roles_df.set_index("node")["role"]
    rows = []
    for name, nodes in hub_sets.items():
        roles = role_lookup.reindex(list(nodes)).dropna()
        counts = roles.value_counts()
        row = {"hub_definition": name, "n_nodes": len(nodes)}
        for role in ("connector_hub", "provincial_hub", "connector_non_hub", "peripheral"):
            row[f"n_{role}"] = int(counts.get(role, 0))
            row[f"frac_{role}"] = float(counts.get(role, 0) / len(roles)) if len(roles) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_seed_component_connectivity(G, seed_node_set, max_components=6):
    """
    Reconstructs the 'original seed components' (connected components of
    the seed-only subgraph within the final module, matching step6's
    seed_diag notion) and computes the local node connectivity (= max
    node-disjoint paths, by Menger's theorem) between a representative
    (highest-module-degree) node of every pair of components, via the
    FULL module graph. Low connectivity between two seed components
    means the module's ability to keep them connected relies on very few
    alternate routes. Capped at the `max_components` largest components
    (by size) to bound the number of all-pairs max-flow computations.
    """
    seed_sub = G.subgraph([n for n in seed_node_set if n in G]).copy()
    comps = sorted(nx.connected_components(seed_sub), key=len, reverse=True)
    if len(comps) < 2:
        print("  [seed-component-connectivity] Seeds form a single connected "
              "component (or none) within the module -- nothing to compare, skipping.")
        return pd.DataFrame()
    if len(comps) > max_components:
        print(f"  [seed-component-connectivity] {len(comps)} seed components found; "
              f"capping to the {max_components} largest.")
    comps = comps[:max_components]

    degree = dict(G.degree())
    reps = [max(comp, key=lambda n: degree.get(n, 0)) for comp in comps]

    rows = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            s, t = reps[i], reps[j]
            try:
                k = nx.node_connectivity(G, s=s, t=t)
            except Exception as e:
                print(f"  [seed-component-connectivity] node_connectivity({s}, {t}) failed: {e}")
                k = np.nan
            rows.append({
                "component_i": i, "component_i_size": len(comps[i]), "representative_i": s,
                "component_j": j, "component_j_size": len(comps[j]), "representative_j": t,
                "node_disjoint_paths": k,
            })
    return pd.DataFrame(rows)


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
                         help="step6's disease_module.csv (node/gene/cell_type/is_seed/... "
                              "columns). Used for reference/reporting AND, if the 'node' and "
                              "'is_seed' columns are present, to split hub knockouts into "
                              "seed vs. bridging-gene subsets (see "
                              "hub_knockout_by_seed_status.csv).")
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

    # NEW arguments
    parser.add_argument("--topology-hub-pct", type=float, default=0.10,
                         help="Top fraction (by degree) used to define the 'hub neighborhood' "
                              "for the hub_neighbor_clustering topology stat, on both the real "
                              "module and every null replicate. Default 0.10 matches the "
                              "top10pct_degree hub definition.")
    parser.add_argument("--ablation-exclude-celltypes", nargs="+", default=[], metavar="CELLTYPE",
                         help="Optional: cell type(s) (the '<celltype>::' namespace prefix) to "
                              "strip out of the already-built module graph before rerunning the "
                              "attack-curve suite, e.g. --ablation-exclude-celltypes global_ppi. "
                              "Lets you see how much of the measured fragility depends on that "
                              "layer being present. Skipped entirely if not given.")
    parser.add_argument("--expensive-metrics-max-nodes", type=int, default=3000,
                         help="Node/edge connectivity, algebraic connectivity, global efficiency, "
                              "and avg shortest path length are only computed on graphs with at "
                              "most this many nodes (real module AND null replicates) -- they scale "
                              "much worse than everything else here. Set higher if your module is "
                              "large and you're willing to pay for it.")
    parser.add_argument("--n-null-expensive-metrics", type=int, default=None,
                         help="Null replicates that get the EXPENSIVE topology metrics "
                              "(connectivity/algebraic connectivity/efficiency). Default: "
                              "min(5, --n-null-networks) -- a handful is enough for a null "
                              "median/std estimate and these are the priciest thing in the "
                              "whole null loop.")
    parser.add_argument("--skip-community-detection", action="store_true",
                         help="Skip Louvain community detection / Guimera-Amaral connector-vs-"
                              "provincial hub roles (community_summary.csv, "
                              "node_community_roles.csv, hub_sets_community_role_summary.csv). "
                              "On by default; needs networkx>=2.8 or the python-louvain package.")
    parser.add_argument("--max-seed-components-for-connectivity", type=int, default=6,
                         help="Cap on the number of (largest) seed components compared pairwise "
                              "in compute_seed_component_connectivity -- an all-pairs max-flow "
                              "computation, so bounded to keep it tractable.")

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

        dm_df = None
        seed_node_set = set()
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
            if "node" in dm_df.columns and "is_seed" in dm_df.columns:
                seed_node_set = set(dm_df.loc[dm_df["is_seed"] == True, "node"])
                print(f"  {len(seed_node_set)} seed node(s) available for seed-vs-bridging split")
            else:
                print("  NOTE: disease_module CSV is missing 'node' and/or 'is_seed' columns -- "
                      "hub_knockout_by_seed_status.csv will be skipped.")

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

        # ------------------------------------------------------------------
        # NEW: mechanism-level topology diagnostics on the REAL module
        # (assortativity, articulation points, hub-neighborhood clustering,
        # Molloy-Reed ratio, layer participation, per-cell-type knockout,
        # seed-vs-bridging knockout split).
        # ------------------------------------------------------------------
        print("\n--- Topology diagnostics on the real module (advisor #1: cut-vertices, "
              "assortativity, connectivity, k-core, redundancy) ---")
        module_topo = compute_topology_stats(G, hub_pct=args.topology_hub_pct, include_expensive=True,
                                              expensive_max_nodes=args.expensive_metrics_max_nodes)
        mr = module_topo["molloy_reed_ratio"]
        print(f"  Degree assortativity              : {module_topo['assortativity']:.4f}")
        print(f"  Articulation points                : {module_topo['n_articulation_points']} "
              f"({module_topo['frac_articulation_points']*100:.2f}% of nodes)")
        print(f"  Bridges                            : {module_topo['n_bridges']} "
              f"({module_topo['frac_bridges']*100:.2f}% of edges)")
        print(f"  k-core                             : max={module_topo['max_core_number']}, "
              f"mean={module_topo['mean_core_number']:.3f}")
        print(f"  Hub-neighborhood clustering (top {args.topology_hub_pct*100:.0f}%): "
              f"{module_topo['hub_neighbor_clustering']:.4f}")
        print(f"  Avg clustering / transitivity      : {module_topo['avg_clustering']:.4f} / "
              f"{module_topo['transitivity']:.4f}")
        print(f"  Molloy-Reed ratio <k^2>/<k>        : {mr:.4f}  "
              f"(NOTE: identical in expectation to the degree-preserving null by "
              f"construction -- any f_c/f_90 divergence from the null is coming from "
              f"wiring, not from this)")
        if module_topo["expensive_metrics_computed"]:
            print(f"  Node / edge connectivity           : {module_topo['node_connectivity']} / "
                  f"{module_topo['edge_connectivity']}")
            print(f"  Algebraic connectivity (LCC)       : {module_topo['algebraic_connectivity']:.4f}")
            print(f"  Global efficiency                  : {module_topo['global_efficiency']:.4f}")
            print(f"  Avg shortest path length (LCC)     : {module_topo['avg_shortest_path_length']:.4f}")
        else:
            print(f"  Node/edge connectivity, algebraic connectivity, efficiency, avg path "
                  f"length: SKIPPED (module has {module_topo['n_nodes']} nodes > "
                  f"--expensive-metrics-max-nodes={args.expensive_metrics_max_nodes})")
        pd.DataFrame([module_topo]).to_csv(os.path.join(output_path, "module_topology_stats.csv"), index=False)

        articulation_points = set(_articulation_points_any(G))
        core_numbers = nx.core_number(G)
        artic_overlap_df = hub_sets_articulation_overlap(hub_sets, articulation_points)
        artic_overlap_df.to_csv(os.path.join(output_path, "hub_articulation_overlap.csv"), index=False)
        print("\n  Hub-set overlap with articulation points:")
        print(artic_overlap_df.to_string(index=False))

        # ------------------------------------------------------------------
        # NEW (advisor #2): bridge-gene dependency -- is fragility
        # concentrated in step6's greedily-added bridging genes rather
        # than in the seeds?
        # ------------------------------------------------------------------
        print("\n--- Bridge-gene dependency diagnostics (advisor #2) ---")
        bounded_impact_nodes = set(articulation_points)
        for nodes in hub_sets.values():
            bounded_impact_nodes |= nodes
        print(f"  Computing single-node removal impact (delta_lcc) for {len(bounded_impact_nodes)} "
              f"node(s) (all articulation points + all hub-set members)...")
        impact_dict = compute_removal_impact(G, bounded_impact_nodes)
        node_diag_df = compute_node_level_diagnostics(
            G, centrality["degree"], centrality["betweenness"], articulation_points,
            seed_node_set, core_numbers, impact_dict,
        )
        node_diag_df.to_csv(os.path.join(output_path, "node_level_diagnostics.csv"), index=False)
        if seed_node_set:
            origin_summary_df = summarize_node_diagnostics_by_origin(node_diag_df)
            origin_summary_df.to_csv(os.path.join(output_path, "node_diagnostics_by_origin_summary.csv"), index=False)
            print(origin_summary_df.to_string(index=False))

            knockout_origin_df = knockout_by_origin(G, seed_node_set)
            knockout_origin_df.to_csv(os.path.join(output_path, "knockout_by_origin_summary.csv"), index=False)
            print("\n  Full seed-set vs. full bridge-added-set knockout:")
            print(knockout_origin_df.to_string(index=False))
        else:
            print("  Seed-vs-bridge-added origin comparison SKIPPED (no usable --disease-module "
                  "'node'/'is_seed' columns) -- node_level_diagnostics.csv is still written with "
                  "is_bridge_added=True for every node.")

        print("\n--- Layer participation (inter- vs. intra-layer edges per node) ---")
        participation_df = compute_layer_participation(G)
        participation_df.to_csv(os.path.join(output_path, "layer_participation.csv"), index=False)
        hub_participation_df = hub_sets_layer_participation_summary(hub_sets, participation_df)
        hub_participation_df.to_csv(os.path.join(output_path, "hub_layer_participation_summary.csv"), index=False)
        print(hub_participation_df.to_string(index=False))

        print("\n--- Per-cell-type knockout (independent of hub definition) ---")
        celltype_knockout_df = celltype_knockout(G)
        celltype_knockout_df.to_csv(os.path.join(output_path, "celltype_knockout_summary.csv"), index=False)
        print(celltype_knockout_df.to_string(index=False))

        print("\n--- Per-cell-type induced-subgraph fragility (advisor #4: is a layer "
              "fragile on its own?) ---")
        celltype_subgraph_df = celltype_subgraph_fragility(G, modes=("degree",), step=args.step)
        celltype_subgraph_df.to_csv(os.path.join(output_path, "celltype_subgraph_attack_summary.csv"), index=False)
        print(celltype_subgraph_df.to_string(index=False))

        celltype_hub_ko_df = celltype_hub_knockout(G, hub_sets)
        celltype_hub_ko_df.to_csv(os.path.join(output_path, "celltype_hub_knockout.csv"), index=False)
        print(f"  celltype x hub-definition knockout saved -> celltype_hub_knockout.csv "
              f"({len(celltype_hub_ko_df)} rows)")

        if seed_node_set:
            print("\n--- Hub knockout split by seed vs. non-seed (bridging-gene) status ---")
            seed_split_df = knockout_hub_sets_by_seed_status(G, hub_sets, seed_node_set)
            seed_split_df.to_csv(os.path.join(output_path, "hub_knockout_by_seed_status.csv"), index=False)
            print(seed_split_df.to_string(index=False))
        else:
            print("\n--- Hub knockout split by seed vs. non-seed status: SKIPPED "
                  "(no usable --disease-module 'node'/'is_seed' columns) ---")

        # ------------------------------------------------------------------
        # NEW (advisor #5): community structure -- connector vs. provincial
        # hubs (Guimera-Amaral), modularity, conductance.
        # ------------------------------------------------------------------
        node_roles_df = None
        if not args.skip_community_detection:
            print("\n--- Community structure (advisor #5: connector vs. provincial hubs) ---")
            comm_summary_df, node_roles_df = compute_community_structure(G, seed=args.seed)
            if comm_summary_df is not None:
                comm_summary_df.to_csv(os.path.join(output_path, "community_summary.csv"), index=False)
                node_roles_df.to_csv(os.path.join(output_path, "node_community_roles.csv"), index=False)
                print(f"  Modularity: {comm_summary_df['modularity'].iloc[0]:.4f}  "
                      f"({len(comm_summary_df)} communities)")
                print("  Role counts:")
                print(node_roles_df["role"].value_counts().to_string())
                hub_role_df = hub_sets_community_role_summary(hub_sets, node_roles_df)
                if hub_role_df is not None:
                    hub_role_df.to_csv(os.path.join(output_path, "hub_sets_community_role_summary.csv"), index=False)
                    print("\n  Connector/provincial-hub composition per hub definition:")
                    print(hub_role_df.to_string(index=False))
        else:
            print("\n--- Community structure: SKIPPED (--skip-community-detection) ---")

        if seed_node_set:
            print("\n--- Seed-component connectivity (advisor #6: node-disjoint paths "
                  "between original seed components) ---")
            seed_comp_conn_df = compute_seed_component_connectivity(
                G, seed_node_set, max_components=args.max_seed_components_for_connectivity,
            )
            if not seed_comp_conn_df.empty:
                seed_comp_conn_df.to_csv(os.path.join(output_path, "seed_component_connectivity.csv"), index=False)
                print(seed_comp_conn_df.to_string(index=False))

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

            # NEW (advisor #7): high-impact hubs and articulation points as
            # their own gene sets, distinct from the residual-after-knockout
            # sets above.
            if not node_diag_df.empty:
                top_impact = node_diag_df.dropna(subset=["delta_lcc"]).sort_values("delta_lcc", ascending=False)
                if not top_impact.empty:
                    try_enrichment(to_gene_symbols(top_impact.head(50)["node"]), "top_delta_lcc_nodes", enrich_dir)
            if articulation_points:
                try_enrichment(to_gene_symbols(articulation_points), "articulation_points", enrich_dir)

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
            topology_hub_pct=args.topology_hub_pct,
            n_null_expensive_metrics=args.n_null_expensive_metrics,
            expensive_max_nodes=args.expensive_metrics_max_nodes,
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

        # NEW: same observed-vs-null percentile treatment, but for the
        # topology stats (assortativity / articulation points / hub
        # clustering) instead of f_c/f_90. null_df carries one row per
        # (null_id, mode, cascading) but the topology columns are constant
        # within a null_id (computed once per null graph), so de-duplicate
        # on null_id before comparing.
        print("\n--- Real module vs. null: topology stats (assortativity, "
              "articulation points, hub clustering) ---")
        topo_cols = ["null_id", "n_nodes", "n_edges", "assortativity", "n_articulation_points",
                     "frac_articulation_points", "n_bridges", "frac_bridges", "max_core_number",
                     "mean_core_number", "hub_neighbor_clustering", "avg_clustering", "transitivity",
                     "molloy_reed_ratio", "node_connectivity", "edge_connectivity",
                     "algebraic_connectivity", "global_efficiency", "avg_shortest_path_length",
                     "expensive_metrics_computed"]
        null_topo_df = null_df.drop_duplicates(subset="null_id")[topo_cols].reset_index(drop=True)
        null_topo_df.to_csv(os.path.join(output_path, "null_topology_stats.csv"), index=False)

        topo_comparison_rows = []
        for stat in ("assortativity", "n_articulation_points", "frac_articulation_points",
                     "n_bridges", "frac_bridges", "max_core_number", "mean_core_number",
                     "hub_neighbor_clustering", "avg_clustering", "transitivity",
                     "node_connectivity", "edge_connectivity", "algebraic_connectivity",
                     "global_efficiency", "avg_shortest_path_length"):
            null_vals = null_topo_df[stat].dropna().values
            obs = module_topo[stat]
            if len(null_vals) == 0 or (obs is None) or (isinstance(obs, float) and np.isnan(obs)):
                continue
            # For assortativity and hub clustering, LOWER than the null is
            # the "more fragile than expected" direction; for articulation
            # points, HIGHER is. Report the percentile either way and let
            # the sign of (observed - null_median) tell the story.
            percentile = (np.sum(null_vals <= obs) / len(null_vals)) * 100
            topo_comparison_rows.append({
                "statistic": stat,
                "observed": obs,
                "null_median": np.median(null_vals),
                "null_std": np.std(null_vals),
                "null_n": len(null_vals),
                "observed_percentile_in_null": percentile,
            })
        topo_comparison_df = pd.DataFrame(topo_comparison_rows)
        topo_comparison_df.to_csv(os.path.join(output_path, "null_topology_comparison.csv"), index=False)
        print(topo_comparison_df.to_string(index=False))
        if not topo_comparison_df.empty:
            assort_row = topo_comparison_df[topo_comparison_df["statistic"] == "assortativity"]
            if not assort_row.empty and assort_row.iloc[0]["observed"] < assort_row.iloc[0]["null_median"]:
                print("  NOTE: module assortativity is more negative (disassortative) than the "
                      "degree-matched null median -- hubs preferentially connect to low-degree "
                      "nodes, a classic driver of excess fragility at matched degree sequence.")
            artic_row = topo_comparison_df[topo_comparison_df["statistic"] == "frac_articulation_points"]
            if not artic_row.empty and artic_row.iloc[0]["observed"] > artic_row.iloc[0]["null_median"]:
                print("  NOTE: module has a higher fraction of articulation points (cut vertices) "
                      "than the null median -- structural bottlenecks, not just degree, are "
                      "contributing to fragility.")
            clust_row = topo_comparison_df[topo_comparison_df["statistic"] == "hub_neighbor_clustering"]
            if not clust_row.empty and clust_row.iloc[0]["observed"] < clust_row.iloc[0]["null_median"]:
                print("  NOTE: hub neighborhoods in the module are less clustered than the null "
                      "median -- hub neighbors have fewer alternate paths to each other, so "
                      "there's less local redundancy when a hub is removed.")

        plot_attack_curves(curves, null_df, output_path)

        if args.run_enrichment:
            degree_row = fc_f90_df[(fc_f90_df["mode"] == "degree") & (~fc_f90_df["cascading"])]
            if not degree_row.empty and not np.isnan(degree_row.iloc[0]["f_c"]):
                fc_degree = degree_row.iloc[0]["f_c"]
                n_removed_at_fc = int(round(fc_degree * G.number_of_nodes()))
                if n_removed_at_fc > 0:
                    ordered = _centrality_order(G, "degree")
                    lost_nodes = ordered[:n_removed_at_fc]
                    try_enrichment(to_gene_symbols(lost_nodes), "nodes_removed_by_fc_degree_attack", enrich_dir)

        # NEW: optional layer-ablation comparison
        if args.ablation_exclude_celltypes:
            print(f"\n--- Layer-ablation comparison (excluding {args.ablation_exclude_celltypes}) ---")
            ablate_celltypes_and_compare(
                G, args.ablation_exclude_celltypes, modes=modes, step=args.step,
                cascading_modes=cascading_modes & set(modes),
                betweenness_k=args.betweenness_k, seed=args.seed, output_path=output_path,
            )

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