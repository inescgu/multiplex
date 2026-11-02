#!/usr/bin/env python3
"""
Step 6 (replaces step9_build_disease_module.py): Build a disease module from
MultiXrank output.

FIX (this version): the heterogeneous pipeline (step 2.5 onward) namespaces
every node as "<celltype>::GENE" in gene_reference_graph.pkl and seeds.txt.
step4's *aggregated* gene_rwr_scores.csv deliberately strips that prefix
back to bare gene symbols (it's a max-across-cell-types summary) -- but
per_celltype_scores.csv is still bare-gene-indexed with cell types as
COLUMNS, not as part of the row identity.

The previous version of this script compared scores_df["gene"] (bare) and
per_ct_df's index (bare) directly against H.nodes() (namespaced), which
never match -- silently producing 0 candidate bridging genes and 100%
"unknown" cell-type annotation, even though everything *looked* like it
ran fine.

What changed vs. that fix
--------------------------
1. A single _norm_node() helper is used everywhere a node string is
   normalized, so the gene portion is uppercased but the cell-type prefix
   keeps its original case (cell types like "Ab-LTMR.CCKAR" are
   mixed-case) -- consistent with the fix already applied in step 4.
2. The ranked-gene pool used to bridge disconnected seed components is now
   built by melting per_celltype_scores.csv into namespaced
   "<celltype>::GENE" -> score pairs, i.e. the SAME namespace as H and
   seeds.txt, sorted descending by score.
3. Cell-type annotation no longer needs per_ct_df lookups at all -- since
   every node in the module already IS "<celltype>::GENE", the cell type
   is read directly off the node string.
4. Final output CSV reports "node" (full namespaced id), "gene" (bare
   symbol), and "cell_type" (parsed from the node) as separate columns.

Elbow diagnostics for the bridging loop
---------------------------------------------------------------
Greedily adding ranked genes until the module is fully connected has no
built-in stopping criterion other than the hard `--k` / `--max-bridge`
caps, and there's no guarantee those caps are actually where the
cost/benefit tradeoff is best. Every gene added past the point where
components stop dropping quickly is diluting the module for little
connectivity benefit.

So the bridging loop in `build_module()` records, at every single
step: (genes_added, n_components, n_nodes, n_edges). This trajectory is
- saved as a CSV (`bridging_trajectory.csv`)
- plotted as components-vs-genes-added ("elbow plot",
  `bridging_elbow.png`)
- used to auto-suggest an elbow point via the standard max-distance-from-
  chord method (draw a line from the first to the last point of the
  curve, find the point on the curve farthest from that line).

The suggested elbow is a *suggestion*, not an enforced cutoff — the
script still runs bridging out to the requested --k/--max-bridge (or
until fully connected), so you can inspect the plot/CSV after the fact
and rerun with a smaller --max-bridge if you want to actually cut there.

NEW: --exclude-celltypes / --exclude-ppi-layer
---------------------------------------------------------------
step1 (--base-ppi) can add a comprehensive, non-cell-type-specific PPI
network as its own pseudo-cell-type multiplex, namespaced
"global_ppi::GENE" just like every real cell type, and step2 wires it to
every real cell type via identity bipartite edges. If you want to build
the disease module using ONLY the real (e.g. neuron) cell-type layers --
as if that backbone had never been added -- pass either:

    --exclude-celltypes global_ppi
    --exclude-ppi-layer                 # shorthand for the line above

This removes every "<celltype>::" node matching an excluded cell type
from the reference graph BEFORE --lcc-per-layer, BEFORE seed-connectivity
diagnostics, and BEFORE the ranking pool is built (any per-cell-type
score columns for an excluded cell type are dropped before melting), so
the excluded layer has zero influence anywhere downstream -- not on
connectivity, not on bridging candidates, not on the final module.

NEW: --exclude-seed-celltypes / --exclude-ppi-seeds
---------------------------------------------------------------
This is the lighter-touch sibling of --exclude-celltypes above, for the
"full" (--connectivity all) module: instead of removing global_ppi from
the graph entirely, KEEP it (and its bipartite edges) so it can still act
as a bridging resource, but drop any "global_ppi::GENE" entries from the
SEED list only, so global_ppi never anchors the module as a seed -- it
can only ever be pulled in opportunistically while bridging disconnected
neuron seed components.

    --exclude-seed-celltypes global_ppi
    --exclude-ppi-seeds                 # shorthand for the line above

This is orthogonal to --exclude-celltypes: use --exclude-celltypes (or
--exclude-ppi-layer) if you want global_ppi gone from the graph
altogether (the "neurons only" case), and --exclude-seed-celltypes (or
--exclude-ppi-seeds) if you want to keep it available for connectivity
but stop it from seeding the module. Passing both for the same cell type
is harmless but redundant (the seed-level exclusion becomes a no-op once
the graph-level one has already removed the nodes).
"""
import argparse
import pickle
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd


def _norm_node(s: str) -> str:
    """Uppercase only the gene portion of a possibly-namespaced node
    string, preserving the cell-type prefix's original case."""
    s = s.strip()
    if "::" in s:
        ct, gene = s.rsplit("::", 1)
        return f"{ct}::{gene.strip().upper()}"
    return s.upper()


def _gene_of(node: str) -> str:
    """Bare gene symbol from a possibly-namespaced node string."""
    return node.rsplit("::", 1)[-1] if "::" in node else node


def _celltype_of(node: str) -> str:
    """Cell-type prefix from a namespaced node string, or 'unknown'."""
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


# ---------------------------------------------------------------------------
# Cell-type exclusion (e.g. drop the --base-ppi 'global_ppi' backbone layer)
# ---------------------------------------------------------------------------
def filter_out_celltypes(H, excluded_celltypes):
    """Remove every node whose '<celltype>::' prefix is in
    excluded_celltypes, entirely -- node, its edges, everything. This is
    how you get a "neurons only" view of a graph that also has a
    non-cell-type-specific backbone (e.g. 'global_ppi') mixed in: the
    backbone nodes are namespaced exactly like real cell types
    ("global_ppi::GENE"), so dropping the namespace drops the whole layer
    cleanly, including its bipartite identity edges to every other cell
    type (those edges have one endpoint in the removed node set, so
    subgraph() drops them automatically).

    Matching is case-insensitive on the cell-type prefix; nodes have
    already been through _norm_node() by the time this runs, which
    preserves the prefix's original case, so we uppercase both sides here.
    """
    if not excluded_celltypes:
        return H
    excluded_set = {c.strip().upper() for c in excluded_celltypes}
    to_remove = [n for n in H.nodes() if _celltype_of(n).upper() in excluded_set]
    if not to_remove:
        print(f"  [exclude-celltypes] None of {sorted(excluded_set)} found in graph "
              f"(checked against node prefixes) — nothing removed.")
        return H
    remove_set = set(to_remove)
    kept = [n for n in H.nodes() if n not in remove_set]
    H_filtered = H.subgraph(kept).copy()
    print(f"  [exclude-celltypes] Removed cell type(s) {sorted(excluded_set)}: "
          f"{len(to_remove):,} nodes dropped -> {H_filtered.number_of_nodes():,} nodes, "
          f"{H_filtered.number_of_edges():,} edges remain "
          f"(was {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges)")
    return H_filtered


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_inputs(scores_csv, ref_graph_pkl, seeds_txt, per_ct_csv=None, exclude_seeds=None,
                 exclude_celltypes=None, exclude_seed_celltypes=None):
    print(f"Loading RWR scores from {scores_csv} ...")
    scores = pd.read_csv(scores_csv)
    scores["gene"] = scores["gene"].astype(str).str.strip().str.upper()
    print(f"  {len(scores):,} ranked genes")

    print(f"\nLoading reference graph from {ref_graph_pkl} ...")
    with open(ref_graph_pkl, "rb") as f:
        H = pickle.load(f)
    mapping = {n: _norm_node(str(n)) for n in H.nodes()}
    H = nx.relabel_nodes(H, mapping)
    print(f"  {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

    if exclude_celltypes:
        print(f"\nExcluding cell type(s) {exclude_celltypes} from reference graph ...")
        H = filter_out_celltypes(H, exclude_celltypes)

    excluded_ct_upper = {c.strip().upper() for c in (exclude_celltypes or [])}

    per_ct_df = None
    ranked_pool = []  # list of (namespaced_node, score), sorted descending
    if per_ct_csv and Path(per_ct_csv).exists():
        per_ct_df = pd.read_csv(per_ct_csv)
        gene_col = next(
            (c for c in per_ct_df.columns if c.lower() in ("gene", "node", "index")),
            per_ct_df.columns[0],
        )
        per_ct_df = per_ct_df.rename(columns={gene_col: "gene"})
        per_ct_df["gene"] = per_ct_df["gene"].astype(str).str.strip().str.upper()
        per_ct_df = per_ct_df.set_index("gene")
        print(f"  Per-cell-type scores: {per_ct_df.shape[0]:,} genes "
              f"× {per_ct_df.shape[1]} cell types (columns ARE the cell-type names)")

        if excluded_ct_upper:
            drop_cols = [c for c in per_ct_df.columns if str(c).strip().upper() in excluded_ct_upper]
            if drop_cols:
                print(f"  [exclude-celltypes] Dropping per-cell-type score column(s) "
                      f"before building the bridging pool: {drop_cols}")
                per_ct_df = per_ct_df.drop(columns=drop_cols)

        # Melt into the SAME namespace as H: "<celltype>::GENE" -> score.
        # This is the actual candidate pool for bridging disconnected seeds.
        long_df = per_ct_df.reset_index().melt(id_vars="gene", var_name="celltype", value_name="score")
        long_df = long_df[long_df["score"] > 0]
        long_df["node"] = long_df["celltype"].astype(str) + "::" + long_df["gene"]
        long_df = long_df.sort_values("score", ascending=False)
        ranked_pool = list(long_df["node"])
        print(f"  Namespaced ranking pool: {len(ranked_pool):,} (celltype::gene) candidates")

    print(f"\nLoading seeds from {seeds_txt} ...")
    with open(seeds_txt) as f:
        raw_seeds = [_norm_node(line) for line in f if line.strip()]

    if excluded_ct_upper:
        seeds_in_excluded_ct = [s for s in raw_seeds if _celltype_of(s).upper() in excluded_ct_upper]
        if seeds_in_excluded_ct:
            print(f"  ⚠ {len(seeds_in_excluded_ct)} seed(s) belong to an excluded cell type "
                  f"and will show up as 'missing from reference' below: {seeds_in_excluded_ct}")

    # NEW: seed-only cell-type exclusion. Unlike --exclude-celltypes (which
    # removes the cell type from the graph entirely, so it can't seed OR
    # bridge), this only drops matching seeds from the seed list -- the
    # cell type's nodes stay in H and in ranked_pool, so they remain
    # available as bridging candidates. Typical use: keep global_ppi as a
    # connectivity resource for the "full" module, but never let it anchor
    # the module as a seed.
    seed_ct_exclude_upper = {c.strip().upper() for c in (exclude_seed_celltypes or [])}
    implicit_excluded_seeds = []
    if seed_ct_exclude_upper:
        implicit_excluded_seeds = [s for s in raw_seeds if _celltype_of(s).upper() in seed_ct_exclude_upper]
        if implicit_excluded_seeds:
            print(f"  [exclude-seed-celltypes] Dropping {len(implicit_excluded_seeds)} seed(s) "
                  f"namespaced under {sorted(seed_ct_exclude_upper)} from the SEED list only -- "
                  f"these cell type(s) remain in the graph/ranking pool and can still be pulled "
                  f"in as bridging genes, they just won't anchor the module as seeds: "
                  f"{implicit_excluded_seeds}")
            raw_seeds = [s for s in raw_seeds if s not in set(implicit_excluded_seeds)]
        else:
            print(f"  [exclude-seed-celltypes] No seeds found under {sorted(seed_ct_exclude_upper)} "
                  f"-- nothing to drop.")

    exclude_set = set(_norm_node(g) for g in (exclude_seeds or []))
    if exclude_set:
        excluded_found = [s for s in raw_seeds if s in exclude_set]
        seeds = [s for s in raw_seeds if s not in exclude_set]
        print(f"  {len(raw_seeds)} seeds loaded")
        print(f"  Excluding {len(excluded_found)} seed(s): {excluded_found}")
        print(f"  {len(seeds)} seeds remaining after exclusion")
    else:
        seeds = raw_seeds
        print(f"  {len(seeds)} seed genes")

    print()
    return scores, H, seeds, per_ct_df, ranked_pool, implicit_excluded_seeds


def filter_to_lcc_per_layer(H):
    """For each cell type (identified via the 'cell_types' set on PPI edges),
    keep only nodes in that cell type's LCC. A node is kept in H if it
    belongs to the LCC of at least one cell type it participates in."""
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


# ---------------------------------------------------------------------------
# Module construction
# ---------------------------------------------------------------------------
def module_subgraph(genes, H):
    nodes_in = [g for g in genes if g in H.nodes()]
    return H.subgraph(nodes_in).copy()


def build_module(H, seeds, ranked_pool, k, max_bridge, exclude_set=None):
    """
    Greedily add ranked genes to the seed set until the module subgraph is
    connected (or the k+max_bridge cap is hit).

    Returns
    -------
    module, n_bridges_added, fully_connected, seeds_in_h, seeds_missing, history

    `history` is a list of dicts, one per gene added:
        {"genes_added": int, "n_components": int, "n_nodes": int, "n_edges": int}
    This is the trajectory used for the elbow plot -- it lets you see
    exactly where adding more bridging genes stops paying off in terms of
    merging components.
    """
    exclude_set = exclude_set or set()
    seeds_in_h = [s for s in seeds if s in H.nodes()]
    seeds_miss = [s for s in seeds if s not in H.nodes()]
    seed_set = set(seeds)

    ranked = [n for n in ranked_pool
              if n not in seed_set and n not in exclude_set and n in H.nodes()]
    print(f"Ranked non-seed genes available: {len(ranked):,}")

    module = list(seeds_in_h)
    sub = module_subgraph(module, H)
    history = []

    if sub.number_of_nodes() > 0 and nx.is_connected(sub):
        print("Seeds alone are connected!")
        return module, 0, True, seeds_in_h, seeds_miss, history

    print("Seeds not fully connected — expanding with ranked genes ...")
    added = 0
    for g in ranked:
        if added >= k + max_bridge:
            break
        module.append(g)
        added += 1

        sub = module_subgraph(module, H)
        n_nodes = sub.number_of_nodes()
        ncc = nx.number_connected_components(sub) if n_nodes else 0
        history.append({
            "genes_added": added,
            "n_components": ncc,
            "n_nodes": n_nodes,
            "n_edges": sub.number_of_edges(),
        })

        if ncc == 1:
            print(f"  Connected after adding {added} gene(s)!")
            return module, added, True, seeds_in_h, seeds_miss, history

        if added % 200 == 0:
            print(f"  +{added:>5} genes -> {n_nodes} nodes, {ncc} component(s)")

    sub = module_subgraph(module, H)
    fc = nx.is_connected(sub) if sub.number_of_nodes() > 0 else False
    ncc = nx.number_connected_components(sub) if sub.number_of_nodes() > 0 else 0
    print(f"Reached limit of {added} genes. Connected={fc}, components={ncc}")
    return module, added, fc, seeds_in_h, seeds_miss, history


def extract_lcc(module, H):
    sub = module_subgraph(module, H)
    if sub.number_of_nodes() == 0:
        return [], sub
    largest = max(nx.connected_components(sub), key=len)
    lcc_sub = sub.subgraph(largest).copy()
    return sorted(largest), lcc_sub


def diagnose_seed_connectivity(seeds, H):
    sub = module_subgraph(seeds, H)
    if sub.number_of_nodes() == 0:
        return {"seed_components": 0, "largest_seed_cc_size": 0,
                "isolated_seeds": [], "seed_subgraph_edges": 0}
    comps = list(nx.connected_components(sub))
    largest = max(comps, key=len) if comps else set()
    isolated = [n for n in sub.nodes() if sub.degree(n) == 0]
    return {
        "seed_components": len(comps),
        "largest_seed_cc_size": len(largest),
        "isolated_seeds": sorted(isolated),
        "seed_subgraph_edges": sub.number_of_edges(),
    }


# ---------------------------------------------------------------------------
# Elbow detection (max-distance-from-chord method)
# ---------------------------------------------------------------------------
def compute_elbow_point(history):
    """
    Classic elbow/knee detection: draw a straight line ("chord") from the
    first point of the curve to the last, then find the point on the curve
    with the greatest perpendicular distance from that chord. That's the
    point of maximum "bend" -- i.e. where the curve departs most from a
    straight-line tradeoff, which is usually a good proxy for "diminishing
    returns kick in here".

    Needs at least 3 points (first, last, and something in between) to be
    meaningful; returns None otherwise.
    """
    if not history or len(history) < 3:
        return None

    xs = np.array([h["genes_added"] for h in history], dtype=float)
    ys = np.array([h["n_components"] for h in history], dtype=float)

    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    line_vec = np.array([x2 - x1, y2 - y1])
    line_len = np.hypot(*line_vec)
    if line_len == 0:
        return None
    line_unit = line_vec / line_len

    pts = np.stack([xs - x1, ys - y1], axis=1)
    proj_len = pts @ line_unit
    proj = np.outer(proj_len, line_unit)
    perp = pts - proj
    distances = np.hypot(perp[:, 0], perp[:, 1])

    idx = int(np.argmax(distances))
    return {
        "genes_added": int(xs[idx]),
        "n_components": int(ys[idx]),
        "distance_from_chord": float(distances[idx]),
        "index": idx,
    }


def plot_elbow(history, elbow, out_path):
    """Save a components-vs-genes-added elbow plot. Returns True on success,
    False if matplotlib isn't available (caller should degrade gracefully)."""
    if not history:
        print("  [elbow-plot] No bridging history to plot (seeds were already connected).")
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [elbow-plot] matplotlib not available — skipping plot "
              "(trajectory CSV was still saved).")
        return False

    xs = [h["genes_added"] for h in history]
    ys = [h["n_components"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, linewidth=1.2)
    plt.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], color="gray", s=20, zorder=4,
                label="chord endpoints")
    if elbow:
        plt.scatter([elbow["genes_added"]], [elbow["n_components"]],
                     color="red", s=60, zorder=5,
                     label=f"suggested elbow: +{elbow['genes_added']} genes "
                           f"({elbow['n_components']} components)")
        plt.axvline(elbow["genes_added"], color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.xlabel("Bridging genes added")
    plt.ylabel("Number of connected components")
    plt.title("Disease module bridging: components vs. genes added")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Elbow plot saved -> {out_path}")
    return True


# ---------------------------------------------------------------------------
# Cell-type annotation — trivial now: the node string already encodes it
# ---------------------------------------------------------------------------
def annotate_cell_types(final_module):
    rows = [{"node": n, "dominant_cell_type": _celltype_of(n)} for n in final_module]
    df = pd.DataFrame(rows).set_index("node")
    print(f"\nCell-type annotation:")
    print(f"  Genes annotated : {len(df):,}")
    print(f"  Unique cell types: {df['dominant_cell_type'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def write_diagnostics(path, info):
    lines = ["=" * 70,
             "DISEASE MODULE DIAGNOSTICS (heterogeneous multiplex + bipartite)",
             "=" * 70,
             f"Connectivity mode      : {info['connectivity_mode']}",
             f"Reference graph        : {info['ref_path']}",
             f"  nodes                : {info['ref_nodes']:,}",
             f"  edges                : {info['ref_edges']:,}",
             f"Excluded cell types (graph+seeds) : {info['excluded_celltypes'] or '(none)'}",
             f"Excluded cell types (seeds only, still bridge-eligible) : "
             f"{info['excluded_seed_celltypes'] or '(none)'}",
             ]
    if info["excluded_seed_celltypes"]:
        lines.append(f"  Implicitly dropped seeds: {info['implicit_excluded_seeds']}")
    lines += ["",
             "--- SEEDS ---",
             f"Seeds provided         : {info['n_seeds_provided']}",
             f"Seeds excluded         : {info['n_seeds_excluded']}",
             f"  Excluded list        : {info['excluded_seeds']}",
             f"Seeds after exclusion  : {info['n_seeds_after_exclusion']}",
             f"Seeds in reference     : {info['n_seeds_in_ref']}",
             f"Seeds missing          : {len(info['seeds_missing'])}",
             ]
    if info["seeds_missing"]:
        lines.append(f"  Missing: {info['seeds_missing']}")

    sd = info["seed_diag"]
    lines += ["",
              "--- SEED-ONLY SUBGRAPH ---",
              f"Connected components   : {sd['seed_components']}",
              f"Largest seed CC size   : {sd['largest_seed_cc_size']}",
              f"Seed subgraph edges    : {sd['seed_subgraph_edges']}",
              f"Isolated seeds         : {len(sd['isolated_seeds'])}",
              ]
    if sd["isolated_seeds"]:
        lines.append(f"  Examples: {sd['isolated_seeds'][:20]}")

    lines += ["",
              "--- MODULE CONSTRUCTION ---",
              f"Top-k requested            : {info['k']}",
              f"Max bridging genes allowed : {info['max_bridge']}",
              f"Ranking pool size          : {info['ranked_pool_size']}",
              f"Genes added                : {info['n_bridges']}",
              f"Full module size           : {info['full_module_size']}",
              f"Fully connected            : {info['fully_connected']}",
              ]
    if not info["fully_connected"]:
        lines.append(f"  Components in full module: {info['full_module_components']}")

    lines += ["",
              "--- BRIDGING ELBOW (components vs. genes added) ---",
              f"Trajectory points saved    : {info['n_history_points']}",
              f"Trajectory CSV              : {info['trajectory_csv']}",
              f"Elbow plot                  : {info['elbow_plot_path'] or '(not generated)'}",
              ]
    if info["elbow"]:
        e = info["elbow"]
        lines.append(f"Suggested elbow             : +{e['genes_added']} genes bridged "
                      f"-> {e['n_components']} components remaining "
                      f"(max distance from chord = {e['distance_from_chord']:.3f})")
        lines.append("  This is a SUGGESTION (max-distance-from-chord method), not enforced. "
                      "If you want to actually cut here, rerun with "
                      f"--max-bridge {e['genes_added']}.")
    else:
        lines.append("Suggested elbow             : n/a (fewer than 3 bridging steps, or "
                      "seeds were already connected)")

    lines += ["",
              "--- FINAL MODULE ---",
              f"Final module size       : {info['final_module_size']}",
              f"Final module edges      : {info['final_module_edges']}",
              f"Used LCC fallback       : {info['used_lcc_fallback']}",
              f"Seeds in final module   : {info['seeds_in_final']} / {info['n_seeds_in_ref']}",
              f"Seeds dropped by LCC    : {info['seeds_dropped_by_lcc']}",
              ]
    if info["seeds_dropped_by_lcc"]:
        lines.append(f"  Dropped: {info['dropped_seed_list'][:20]}")

    if info["dominant_ct_counts"]:
        lines += ["", "--- DOMINANT CELL TYPE DISTRIBUTION ---"]
        for ct, n in sorted(info["dominant_ct_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"  {ct:<40} {n:>6,} genes")

    if not info["fully_connected"]:
        lines += ["", "=" * 70, "⚠  MODULE NOT FULLY CONNECTED — SUGGESTIONS", "=" * 70]
        if sd["seed_components"] > 1:
            lines.append(f"• Seeds split into {sd['seed_components']} components. "
                          "Consider increasing --k or reviewing seed list.")
        if info["n_bridges"] >= info["max_bridge"]:
            lines.append(f"• Hit --max-bridge cap ({info['max_bridge']}). Try increasing it.")
        if info["used_lcc_fallback"]:
            lines.append("• LCC fallback used. Use --no-lcc-fallback to keep all genes.")
        if info["ranked_pool_size"] == 0:
            lines.append("• Ranking pool was EMPTY — pass --per-celltype-scores so bridging "
                          "candidates can be found (aggregated gene_rwr_scores.csv alone is "
                          "not enough to bridge a namespaced graph).")
        if info["excluded_celltypes"]:
            lines.append(f"• Cell type(s) {info['excluded_celltypes']} were excluded -- if "
                          "connectivity is poor, that backbone/layer may have been providing "
                          "bridging paths you've now removed.")

    lines.append("=" * 70)
    text = "\n".join(lines)
    print(text)
    Path(path).write_text(text)
    print(f"\nDiagnostics saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build disease module from MultiXrank scores (heterogeneous structure)")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--per-celltype-scores", default=None)
    parser.add_argument("--input-dir", required=True, help="MultiXrank wdir (output of step1-3)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--connectivity", choices=["all", "ppi"], default="all")
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--max-bridge", type=int, default=2000)
    parser.add_argument("--no-lcc-fallback", action="store_true")
    parser.add_argument("--lcc-per-layer", action="store_true",
                         help="Restrict each cell type to its largest connected component before building the module.")
    parser.add_argument("--exclude-seeds", nargs="*", default=[], metavar="GENE",
                         help="Seed gene(s) to exclude from module building. "
                              "Example: --exclude-seeds A-LTMR.TAC3::TRPV1")
    parser.add_argument("--exclude-celltypes", nargs="+", default=[], metavar="CELLTYPE",
                         help="Cell type(s) (the '<celltype>::' namespace prefix) to drop "
                              "entirely from the reference graph, ranking pool, and seed list "
                              "before any analysis, e.g. --exclude-celltypes global_ppi. "
                              "Case-insensitive.")
    parser.add_argument("--exclude-ppi-layer", action="store_true",
                         help="Shorthand for --exclude-celltypes global_ppi -- drops the "
                              "base/comprehensive PPI pseudo-cell-type added in step1 via "
                              "--base-ppi, so the module is built from only the real (e.g. "
                              "neuron) cell-type layers.")
    parser.add_argument("--exclude-seed-celltypes", nargs="+", default=[], metavar="CELLTYPE",
                         help="Cell type(s) whose seeds should be dropped from the SEED list "
                              "only -- the cell type's nodes stay in the graph and ranking "
                              "pool, so they remain usable as bridging genes. Lighter-touch "
                              "alternative to --exclude-celltypes for the 'full' module. "
                              "Example: --exclude-seed-celltypes global_ppi. Case-insensitive.")
    parser.add_argument("--exclude-ppi-seeds", action="store_true",
                         help="Shorthand for --exclude-seed-celltypes global_ppi -- global_ppi "
                              "genes remain available as bridging candidates but never anchor "
                              "the module as seeds themselves.")
    parser.add_argument("--no-elbow-plot", action="store_true",
                         help="Skip generating the components-vs-genes-added elbow plot "
                              "(the trajectory CSV is still saved).")
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_path = (inp / "gene_reference_graph.pkl" if args.connectivity == "all"
                else inp / "gene_ppi_reference_graph.pkl")
    seeds_path = inp / "seeds" / "seeds.txt"

    exclude_seeds = [_norm_node(g) for g in args.exclude_seeds]

    exclude_celltypes = list(args.exclude_celltypes)
    if args.exclude_ppi_layer and "global_ppi" not in {c.lower() for c in exclude_celltypes}:
        exclude_celltypes.append("global_ppi")

    exclude_seed_celltypes = list(args.exclude_seed_celltypes)
    if args.exclude_ppi_seeds and "global_ppi" not in {c.lower() for c in exclude_seed_celltypes}:
        exclude_seed_celltypes.append("global_ppi")

    scores, H, seeds, per_ct_df, ranked_pool, implicit_excluded_seeds = load_inputs(
        args.scores, ref_path, seeds_path,
        per_ct_csv=args.per_celltype_scores,
        exclude_seeds=exclude_seeds,
        exclude_celltypes=exclude_celltypes,
        exclude_seed_celltypes=exclude_seed_celltypes,
    )

    if not ranked_pool:
        print("⚠ No --per-celltype-scores ranking pool available — module bridging "
              "will have no candidate genes to expand with (seeds-only connectivity).\n")

    if args.lcc_per_layer:
        print("\nFiltering each cell type to its largest connected component ...")
        H = filter_to_lcc_per_layer(H)
        print()

    seed_diag = diagnose_seed_connectivity(seeds, H)
    print(f"Seed-only subgraph: {seed_diag['seed_components']} components, "
          f"largest CC = {seed_diag['largest_seed_cc_size']}\n")

    module, n_bridges, fully_connected, seeds_in_h, seeds_missing, history = build_module(
        H, seeds, ranked_pool, args.k, args.max_bridge, exclude_set=set(exclude_seeds)
    )

    # --- bridging trajectory: CSV + elbow plot -----------------------------
    trajectory_csv = out / "bridging_trajectory.csv"
    pd.DataFrame(history).to_csv(trajectory_csv, index=False)
    print(f"\nBridging trajectory ({len(history)} steps) saved -> {trajectory_csv}")

    elbow = compute_elbow_point(history)
    elbow_plot_path = out / "bridging_elbow.png"
    plot_ok = False
    if not args.no_elbow_plot:
        plot_ok = plot_elbow(history, elbow, elbow_plot_path)
    if elbow:
        print(f"Suggested elbow: +{elbow['genes_added']} genes bridged -> "
              f"{elbow['n_components']} components remaining "
              f"(vs. {n_bridges} genes actually added this run)")
    # -------------------------------------------------------------------

    if not fully_connected:
        full_sub_diag = module_subgraph(module, H)
        components = sorted(nx.connected_components(full_sub_diag), key=len, reverse=True)
        seed_set_diag = set(seeds_in_h)
        print("\nComponent size breakdown (before LCC):")
        for i, comp in enumerate(components):
            seeds_in_comp = [g for g in comp if g in seed_set_diag]
            non_seeds_in_comp = [g for g in comp if g not in seed_set_diag]
            line = f"  Component {i+1}: {len(comp):>6,} genes, {len(seeds_in_comp):>3} seeds"
            if seeds_in_comp:
                line += f"  -> seeds: {seeds_in_comp}"
            if non_seeds_in_comp and len(non_seeds_in_comp) < 10:
                line += f"  -> non-seeds: {non_seeds_in_comp}"
            print(line)
        print()

    full_sub = module_subgraph(module, H)
    full_cc = nx.number_connected_components(full_sub) if full_sub.number_of_nodes() else 0

    used_lcc = False
    if not fully_connected and not args.no_lcc_fallback:
        print("Applying LCC fallback ...")
        module_lcc, lcc_sub = extract_lcc(module, H)
        print(f"  LCC: {len(module_lcc)} genes (dropped {len(module) - len(module_lcc)})")
        final_module = module_lcc
        final_sub = lcc_sub
        used_lcc = True
    else:
        final_module = module
        final_sub = full_sub

    seeds_in_final = sum(1 for s in seeds_in_h if s in set(final_module))
    dropped_seeds = sorted(set(seeds_in_h) - set(final_module))

    ct_annotation = annotate_cell_types(final_module)
    dominant_ct_counts = ct_annotation["dominant_cell_type"].value_counts().to_dict()

    score_lookup = dict(zip(scores["gene"], scores["rwr_score"]))
    original_seeds = set(_norm_node(line) for line in open(seeds_path) if line.strip())
    rows = [{
        "node": g,
        "gene": _gene_of(g),
        "cell_type": _celltype_of(g),
        "is_seed": g in original_seeds,
        "is_excluded_seed": g in set(exclude_seeds) or g in set(implicit_excluded_seeds),
        "rwr_score": score_lookup.get(_gene_of(g), 0.0),
        "degree_in_module": final_sub.degree(g) if g in final_sub.nodes() else 0,
    } for g in final_module]

    module_df = pd.DataFrame(rows).sort_values(["is_seed", "rwr_score"], ascending=[False, False]).reset_index(drop=True)

    module_csv = out / "disease_module.csv"
    module_df.to_csv(module_csv, index=False)
    print(f"\nSaved disease module -> {module_csv}")
    print(f"  {len(module_df):,} genes, {module_df['is_seed'].sum()} seeds")
    if exclude_seeds:
        print(f"  Excluded seeds    : {exclude_seeds}")
    if exclude_celltypes:
        print(f"  Excluded cell types (graph+seeds): {exclude_celltypes}")
    if exclude_seed_celltypes:
        print(f"  Excluded cell types (seeds only, still bridge-eligible): {exclude_seed_celltypes}")
        print(f"    Implicitly dropped seeds: {implicit_excluded_seeds}")

    with open(out / "module_graph.pkl", "wb") as f:
        pickle.dump(final_sub, f, protocol=4)
    print(f"Saved module graph -> {out / 'module_graph.pkl'}")

    all_original_seeds = [_norm_node(line) for line in open(seeds_path) if line.strip()]

    info = {
        "connectivity_mode": args.connectivity,
        "ref_path": str(ref_path),
        "ref_nodes": H.number_of_nodes(),
        "ref_edges": H.number_of_edges(),
        "excluded_celltypes": exclude_celltypes,
        "excluded_seed_celltypes": exclude_seed_celltypes,
        "implicit_excluded_seeds": implicit_excluded_seeds,
        "n_seeds_provided": len(all_original_seeds),
        "n_seeds_excluded": len(exclude_seeds),
        "excluded_seeds": exclude_seeds,
        "n_seeds_after_exclusion": len(seeds),
        "n_seeds_in_ref": len(seeds_in_h),
        "seeds_missing": seeds_missing,
        "seed_diag": seed_diag,
        "k": args.k,
        "max_bridge": args.max_bridge,
        "ranked_pool_size": len(ranked_pool),
        "n_bridges": n_bridges,
        "full_module_size": len(module),
        "fully_connected": fully_connected,
        "full_module_components": full_cc,
        "n_history_points": len(history),
        "trajectory_csv": str(trajectory_csv),
        "elbow_plot_path": str(elbow_plot_path) if plot_ok else None,
        "elbow": elbow,
        "final_module_size": len(final_module),
        "final_module_edges": final_sub.number_of_edges(),
        "used_lcc_fallback": used_lcc,
        "seeds_in_final": seeds_in_final,
        "seeds_dropped_by_lcc": len(dropped_seeds),
        "dropped_seed_list": dropped_seeds,
        "dominant_ct_counts": dominant_ct_counts,
    }
    write_diagnostics(out / "module_diagnostics.txt", info)


if __name__ == "__main__":
    main()