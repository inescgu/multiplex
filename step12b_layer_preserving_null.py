#!/usr/bin/env python
"""
step12b_layer_preserving_null.py

The "essential control" for step12_hub_removal_analysis.py's fragility
finding, per the advisor's review:

    step12's degree-preserving null uses nx.double_edge_swap on the
    WHOLE module graph, which treats every edge as interchangeable. For
    a heterogeneous multiplex graph, that's wrong: it can (and typically
    will) rewire an intra-layer PPI edge into what looks like a
    cross-layer connection in the same rewired graph object, silently
    destroying the multiplex's edge-type semantics -- and quite possibly
    the very bottleneck structure that's making the real module fragile.
    That means step12's null may be OVERSTATING how robust a "typical"
    graph of this degree sequence would be, i.e. it may be MANUFACTURING
    the "more fragile than degree-matched average" result rather than
    correctly controlling for it.

This script builds a LAYER-PRESERVING null instead: edges are rewired
only among other edges of the same type, so the null exactly preserves:
  - the degree sequence WITHIN each cell-type layer (intra-layer PPI
    edges are only ever swapped for other intra-layer PPI edges of the
    SAME cell type);
  - the complete, unmodified set of inter-layer identity/bipartite edges
    (these are never touched, so the number of cell types each gene
    participates in is exactly preserved too, since that's entirely
    determined by which identity edges exist);
  - the cell-type composition of the module (node set never changes).

Optionally (--stratify-by-seed-status), it goes one step further and
also keeps seed-seed / seed-nonseed / nonseed-nonseed edges as separate
rewiring strata within each layer, so the seed-vs-bridge-added degree
distribution doesn't drift under the null either.

It then re-runs the SAME attack-curve + topology-stats suite as step12
on this null (and, by default, also re-runs step12's naive whole-graph
null here for a directly comparable baseline) and reports a 3-way
comparison: real module vs. naive null vs. layer-preserving null.

    - If the module looks fragile against the naive null but NOT against
      the layer-preserving null, the original "more fragile than
      degree-matched average" finding was a multiplex-CONSTRUCTION
      artifact (sparse cross-layer connectors are a generic feature of
      any multiplex, not something specific to this disease module) --
      not a biological property of the disease genes.
    - If it remains fragile against BOTH nulls, the finding survives the
      stricter, semantically-correct control and should be treated as
      real.

DEPENDS on step12_hub_removal_analysis.py living in the SAME DIRECTORY --
this script imports attack_curve, critical_fraction, compute_topology_stats,
degree_preserving_null_curves, and a handful of small helpers from it
rather than duplicating ~800 lines of already-reviewed code.

Usage
-----
python step12b_layer_preserving_null.py \
    --module-graph 06_disease_module/disease_module_output/module_graph.pkl \
    --disease-module 06_disease_module/disease_module_output/disease_module.csv \
    --output-dir 12_hub_removal \
    --stratify-by-seed-status \
    --n-null 20 \
    --n-null-cascading 10 \
    --cascading-modes degree,random \
    --n-jobs 8
"""

import os
import sys
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from step12_hub_removal_analysis import (
        load_pickle, ensure_dir, _celltype_of, check_namespaced,
        attack_curve, critical_fraction, compute_topology_stats,
        degree_preserving_null_curves,
    )
except ImportError:
    print("ERROR: could not import from step12_hub_removal_analysis.py -- "
          "this script must live in the same directory as it.")
    raise


# --------------------------------------------------------------------------
# Layer-preserving (edge-type-stratified) null generator
# --------------------------------------------------------------------------

def partition_edges(G, seed_node_set=None, stratify_by_seed=False):
    """
    Splits G's edges into the strata a layer-preserving null must NOT mix:

      - inter_layer_edges : edges between two DIFFERENT cell types (the
        bipartite identity edges joining the same gene across layers).
        Left completely untouched by the null.
      - intra_layer_edges[key] : edges within one cell type. key is the
        cell type name (stratify_by_seed=False), or (cell_type,
        seed_pair_type) with seed_pair_type in {'seed_seed',
        'seed_nonseed', 'nonseed_nonseed'} (stratify_by_seed=True). Each
        stratum is rewired independently, so a node's degree WITHIN
        every stratum it participates in is exactly preserved -- and
        since a node's total degree is the sum of its per-stratum
        degrees, its TOTAL degree in the null exactly equals its total
        degree in the real module.
    """
    inter_layer_edges = []
    intra_layer_edges = {}
    for u, v in G.edges():
        ct_u, ct_v = _celltype_of(u), _celltype_of(v)
        if ct_u != ct_v:
            inter_layer_edges.append((u, v))
            continue
        key = ct_u
        if stratify_by_seed and seed_node_set is not None:
            su, sv = u in seed_node_set, v in seed_node_set
            if su and sv:
                pair_type = "seed_seed"
            elif not su and not sv:
                pair_type = "nonseed_nonseed"
            else:
                pair_type = "seed_nonseed"
            key = (ct_u, pair_type)
        intra_layer_edges.setdefault(key, []).append((u, v))
    return inter_layer_edges, intra_layer_edges


def generate_layer_preserving_null(G, seed_node_set=None, stratify_by_seed=False,
                                    swap_factor=10, seed=0):
    """
    Builds one layer-preserving null replicate:
      1. Partition edges (see partition_edges).
      2. For each intra-layer stratum, build the induced subgraph on
         JUST that stratum's nodes+edges and run nx.double_edge_swap on
         it -- too-small strata (<4 nodes or <2 edges; double_edge_swap
         raises NetworkXError for these) are simply left un-rewired.
      3. Recombine into a new graph over G's FULL original node set: all
         (rewired) intra-layer strata plus every untouched inter-layer
         identity edge.
    """
    inter_layer_edges, intra_layer_edges = partition_edges(G, seed_node_set, stratify_by_seed)

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(inter_layer_edges)

    for i, (key, edges) in enumerate(sorted(intra_layer_edges.items(), key=lambda kv: str(kv[0]))):
        stratum_nodes = {n for e in edges for n in e}
        sub = nx.Graph()
        sub.add_nodes_from(stratum_nodes)
        sub.add_edges_from(edges)
        n_edges = sub.number_of_edges()
        if n_edges >= 2 and sub.number_of_nodes() >= 4:
            n_swaps = max(1, swap_factor * n_edges)
            try:
                nx.double_edge_swap(sub, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed + i + 1)
            except nx.NetworkXError:
                pass  # too sparse/small to swap meaningfully; left as-is
        H.add_edges_from(sub.edges())

    return H


def _layer_null_worker(args):
    (null_id, edges, nodes, seed_node_ids, stratify_by_seed, modes, cascading_modes,
     step, cascading_step, betweenness_k, swap_factor, topology_hub_pct,
     compute_expensive, expensive_max_nodes, seed) = args
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    seed_node_set = set(seed_node_ids) if seed_node_ids is not None else None

    H = generate_layer_preserving_null(G, seed_node_set=seed_node_set,
                                        stratify_by_seed=stratify_by_seed,
                                        swap_factor=swap_factor, seed=seed)
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
                                 cascading_step=cascading_step, betweenness_k=betweenness_k, seed=seed)
            fc_c, f90_c = critical_fraction(df_c)
            rows.append({"null_id": null_id, "mode": mode, "cascading": True, "f_c": fc_c, "f_90": f90_c, **topo})
    return rows


def layer_preserving_null_curves(G, modes, seed_node_set=None, stratify_by_seed=False,
                                  n_null=20, n_null_cascading=None, cascading_modes=(),
                                  step=1, cascading_step=None, betweenness_k=None,
                                  swap_factor=10, seed=0, n_jobs=1, topology_hub_pct=0.10,
                                  n_null_expensive_metrics=None, expensive_max_nodes=3000):
    """Same shape/semantics as step12's degree_preserving_null_curves, but
    every replicate is built by generate_layer_preserving_null instead of
    a single whole-graph double_edge_swap."""
    if n_null_cascading is None:
        n_null_cascading = n_null
    if n_null_expensive_metrics is None:
        n_null_expensive_metrics = min(5, n_null)

    edges = list(G.edges())
    nodes = list(G.nodes())
    seed_node_ids = list(seed_node_set) if seed_node_set else None
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n_null):
        this_cascading_modes = cascading_modes if i < n_null_cascading else set()
        compute_expensive = i < n_null_expensive_metrics
        tasks.append((i, edges, nodes, seed_node_ids, stratify_by_seed, modes,
                      this_cascading_modes, step, cascading_step, betweenness_k,
                      swap_factor, topology_hub_pct, compute_expensive, expensive_max_nodes,
                      int(rng.integers(0, 2**31 - 1))))

    rows = []
    if n_jobs and n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_layer_null_worker, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futures):
                rows.extend(fut.result())
                done += 1
                if done % 5 == 0 or done == n_null:
                    print(f"  Completed {done}/{n_null} layer-preserving null networks")
    else:
        for t in tasks:
            rows.extend(_layer_null_worker(t))
            i = t[0]
            if (i + 1) % 5 == 0 or i == n_null - 1:
                print(f"  Completed {i+1}/{n_null} layer-preserving null networks")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer-preserving (edge-type-stratified) degree-preserving null for a "
                     "HETEROGENEOUS MULTIPLEX disease module -- the essential control "
                     "step12's naive whole-graph double_edge_swap null does not provide.")
    parser.add_argument("--module-graph", type=str, required=True,
                         help="step6's module_graph.pkl (same file step12 uses).")
    parser.add_argument("--disease-module", type=str, default=None,
                         help="step6's disease_module.csv. Needed only for "
                              "--stratify-by-seed-status (uses its node/is_seed columns).")
    parser.add_argument("--output-dir", type=str, default="output",
                         help="Outputs are written under <output-dir>/layer_preserving_null/.")
    parser.add_argument("--stratify-by-seed-status", action="store_true",
                         help="Within each cell-type layer, ALSO keep seed-seed / "
                              "seed-nonseed / nonseed-nonseed edges as separate rewiring "
                              "strata, so the seed-vs-bridge-added degree distribution "
                              "doesn't drift under the null either. Requires "
                              "--disease-module with usable node/is_seed columns.")
    parser.add_argument("--skip-naive-null-comparison", action="store_true",
                         help="By default this script ALSO re-runs step12's naive "
                              "whole-graph double_edge_swap null here (with matching "
                              "settings) so the 3-way comparison is apples-to-apples. Set "
                              "this to skip that and only report real-module vs. "
                              "layer-preserving-null.")
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--n-null-cascading", type=int, default=None)
    parser.add_argument("--n-null-expensive-metrics", type=int, default=None)
    parser.add_argument("--expensive-metrics-max-nodes", type=int, default=3000)
    parser.add_argument("--swap-factor", type=int, default=10)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--cascading-step", type=int, default=None)
    parser.add_argument("--betweenness-k", type=int, default=None)
    parser.add_argument("--null-betweenness-k", type=int, default=None)
    parser.add_argument("--cascading-modes", type=str, default="degree,random")
    parser.add_argument("--topology-hub-pct", type=float, default=0.10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_path = ensure_dir(os.path.join(args.output_dir, "layer_preserving_null"))
    print(f"\n{'='*80}\nLayer-Preserving Null Control (advisor recommendation #3)\n{'='*80}")
    print(f"Output directory: {output_path}\n")

    try:
        print(f"Loading module graph from {args.module_graph}...")
        G = load_pickle(args.module_graph)
        print(f"Module graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        check_namespaced(G, "--module-graph")

        seed_node_set = None
        if args.disease_module:
            dm_df = pd.read_csv(args.disease_module)
            if "node" in dm_df.columns and "is_seed" in dm_df.columns:
                seed_node_set = set(dm_df.loc[dm_df["is_seed"] == True, "node"])
                print(f"  {len(seed_node_set)} seed node(s) loaded from --disease-module")
            else:
                print("  NOTE: --disease-module is missing 'node'/'is_seed' columns.")
        if args.stratify_by_seed_status and not seed_node_set:
            print("  WARNING: --stratify-by-seed-status requested but no usable seed set "
                  "loaded -- proceeding WITHOUT seed stratification (layer-only null).")
            args.stratify_by_seed_status = False

        inter_layer_edges, intra_layer_edges = partition_edges(G, seed_node_set, args.stratify_by_seed_status)
        n_intra = sum(len(v) for v in intra_layer_edges.values())
        print(f"\nEdge-type composition: {len(inter_layer_edges)} inter-layer (identity) edge(s) "
              f"[untouched by the null], {n_intra} intra-layer edge(s) across "
              f"{len(intra_layer_edges)} rewiring stratum/strata:")
        edge_comp_rows = []
        for key, edges in sorted(intra_layer_edges.items(), key=lambda kv: -len(kv[1])):
            print(f"  {key}: {len(edges)} edges")
            edge_comp_rows.append({"stratum": str(key), "n_edges": len(edges)})
        edge_comp_rows.append({"stratum": "inter_layer_identity (untouched)", "n_edges": len(inter_layer_edges)})
        pd.DataFrame(edge_comp_rows).to_csv(os.path.join(output_path, "edge_type_composition.csv"), index=False)

        modes = ["degree", "betweenness", "degree_betweenness"]
        cascading_modes = set(m.strip() for m in args.cascading_modes.split(",") if m.strip())

        # --- Real module's own attack curves + topology (self-contained --------
        # --- so this script doesn't require step12 to have run first) ----------
        print("\n--- Real module attack curves ---")
        fc_f90_rows = []
        for mode in modes:
            df_static = attack_curve(G, mode=mode, cascading=False, step=args.step,
                                      betweenness_k=args.betweenness_k, seed=args.seed)
            fc, f90 = critical_fraction(df_static)
            fc_f90_rows.append({"mode": mode, "cascading": False, "f_c": fc, "f_90": f90})
            if mode in cascading_modes:
                df_casc = attack_curve(G, mode=mode, cascading=True, step=args.step,
                                        cascading_step=args.cascading_step,
                                        betweenness_k=args.betweenness_k, seed=args.seed)
                fc_c, f90_c = critical_fraction(df_casc)
                fc_f90_rows.append({"mode": mode, "cascading": True, "f_c": fc_c, "f_90": f90_c})
        fc_f90_df = pd.DataFrame(fc_f90_rows)
        fc_f90_df.to_csv(os.path.join(output_path, "f_c_f90_observed.csv"), index=False)
        print(fc_f90_df.to_string(index=False))

        module_topo = compute_topology_stats(G, hub_pct=args.topology_hub_pct, include_expensive=True,
                                              expensive_max_nodes=args.expensive_metrics_max_nodes)
        pd.DataFrame([module_topo]).to_csv(os.path.join(output_path, "module_topology_stats.csv"), index=False)

        # --- Layer-preserving null ----------------------------------------------
        print(f"\n--- Layer-preserving null ({args.n_null} replicates"
              f"{', seed-status-stratified' if args.stratify_by_seed_status else ''}) ---")
        layer_null_df = layer_preserving_null_curves(
            G, modes=modes, seed_node_set=seed_node_set,
            stratify_by_seed=args.stratify_by_seed_status,
            n_null=args.n_null, n_null_cascading=args.n_null_cascading,
            cascading_modes=cascading_modes & set(modes),
            step=args.step, cascading_step=args.cascading_step,
            betweenness_k=args.null_betweenness_k, swap_factor=args.swap_factor,
            seed=args.seed, n_jobs=args.n_jobs, topology_hub_pct=args.topology_hub_pct,
            n_null_expensive_metrics=args.n_null_expensive_metrics,
            expensive_max_nodes=args.expensive_metrics_max_nodes,
        )
        layer_null_df.to_csv(os.path.join(output_path, "layer_preserving_null_fc_f90.csv"), index=False)

        # --- Naive (whole-graph) null, re-run here for a matching-settings -----
        # --- apples-to-apples comparison ----------------------------------------
        naive_null_df = None
        if not args.skip_naive_null_comparison:
            print(f"\n--- Naive whole-graph degree-preserving null, re-run here for "
                  f"comparison ({args.n_null} replicates) ---")
            naive_null_df = degree_preserving_null_curves(
                G, modes=modes, n_null=args.n_null, n_null_cascading=args.n_null_cascading,
                cascading_modes=cascading_modes & set(modes), step=args.step,
                cascading_step=args.cascading_step, betweenness_k=args.null_betweenness_k,
                swap_factor=args.swap_factor, seed=args.seed, n_jobs=args.n_jobs,
                topology_hub_pct=args.topology_hub_pct,
                n_null_expensive_metrics=args.n_null_expensive_metrics,
                expensive_max_nodes=args.expensive_metrics_max_nodes,
            )
            naive_null_df.to_csv(os.path.join(output_path, "naive_null_fc_f90.csv"), index=False)

        # --- 3-way f_c / f_90 comparison -----------------------------------------
        comparison_rows = []
        for _, row in fc_f90_df.iterrows():
            layer_sub = layer_null_df[(layer_null_df["mode"] == row["mode"]) &
                                       (layer_null_df["cascading"] == row["cascading"])]
            naive_sub = (naive_null_df[(naive_null_df["mode"] == row["mode"]) &
                                        (naive_null_df["cascading"] == row["cascading"])]
                         if naive_null_df is not None else pd.DataFrame())
            for stat in ("f_c", "f_90"):
                obs = row[stat]
                entry = {"mode": row["mode"], "cascading": row["cascading"], "statistic": stat, "observed": obs}
                layer_vals = layer_sub[stat].dropna().values
                if len(layer_vals) and pd.notna(obs):
                    entry["layer_preserving_null_median"] = np.median(layer_vals)
                    entry["layer_preserving_null_std"] = np.std(layer_vals)
                    entry["observed_percentile_in_layer_preserving_null"] = (
                        np.sum(layer_vals <= obs) / len(layer_vals) * 100)
                if not naive_sub.empty:
                    naive_vals = naive_sub[stat].dropna().values
                    if len(naive_vals) and pd.notna(obs):
                        entry["naive_null_median"] = np.median(naive_vals)
                        entry["naive_null_std"] = np.std(naive_vals)
                        entry["observed_percentile_in_naive_null"] = (
                            np.sum(naive_vals <= obs) / len(naive_vals) * 100)
                comparison_rows.append(entry)
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(os.path.join(output_path, "three_way_fc_f90_comparison.csv"), index=False)
        print("\n--- Real module vs. naive null vs. layer-preserving null (f_c / f_90) ---")
        print(comparison_df.to_string(index=False))

        if "observed_percentile_in_naive_null" in comparison_df.columns:
            for _, r in comparison_df.iterrows():
                if r["statistic"] != "f_c" or r["cascading"]:
                    continue
                naive_pct = r.get("observed_percentile_in_naive_null", np.nan)
                layer_pct = r.get("observed_percentile_in_layer_preserving_null", np.nan)
                if pd.notna(naive_pct) and pd.notna(layer_pct):
                    if naive_pct <= 25 and layer_pct > 25:
                        print(f"  NOTE [{r['mode']}]: module looks fragile vs. the NAIVE null "
                              f"({naive_pct:.0f}th pct) but NOT vs. the layer-preserving null "
                              f"({layer_pct:.0f}th pct) -- this looks like a multiplex-construction "
                              f"artifact, not biology.")
                    elif naive_pct <= 25 and layer_pct <= 25:
                        print(f"  NOTE [{r['mode']}]: module remains fragile vs. BOTH the naive null "
                              f"({naive_pct:.0f}th pct) and the layer-preserving null ({layer_pct:.0f}th "
                              f"pct) -- this fragility survives the stricter control and looks real.")

        # --- Topology-stats 3-way comparison -------------------------------------
        topo_stats = ["assortativity", "n_articulation_points", "frac_articulation_points",
                      "n_bridges", "frac_bridges", "max_core_number", "mean_core_number",
                      "hub_neighbor_clustering", "avg_clustering", "transitivity",
                      "node_connectivity", "edge_connectivity", "algebraic_connectivity",
                      "global_efficiency", "avg_shortest_path_length"]
        layer_topo_df = layer_null_df.drop_duplicates(subset="null_id").reset_index(drop=True)
        layer_topo_df.to_csv(os.path.join(output_path, "layer_preserving_null_topology_stats.csv"), index=False)
        naive_topo_df = naive_null_df.drop_duplicates(subset="null_id").reset_index(drop=True) if naive_null_df is not None else None

        topo_comparison_rows = []
        for stat in topo_stats:
            obs = module_topo.get(stat, np.nan)
            if obs is None or (isinstance(obs, float) and np.isnan(obs)):
                continue
            row = {"statistic": stat, "observed": obs}
            if stat in layer_topo_df.columns:
                layer_vals = layer_topo_df[stat].dropna().values
                if len(layer_vals):
                    row["layer_preserving_null_median"] = np.median(layer_vals)
                    row["layer_preserving_null_std"] = np.std(layer_vals)
                    row["observed_percentile_in_layer_preserving_null"] = np.sum(layer_vals <= obs) / len(layer_vals) * 100
            if naive_topo_df is not None and stat in naive_topo_df.columns:
                naive_vals = naive_topo_df[stat].dropna().values
                if len(naive_vals):
                    row["naive_null_median"] = np.median(naive_vals)
                    row["naive_null_std"] = np.std(naive_vals)
                    row["observed_percentile_in_naive_null"] = np.sum(naive_vals <= obs) / len(naive_vals) * 100
            topo_comparison_rows.append(row)
        topo_comparison_df = pd.DataFrame(topo_comparison_rows)
        topo_comparison_df.to_csv(os.path.join(output_path, "three_way_topology_comparison.csv"), index=False)
        print("\n--- Real module vs. naive null vs. layer-preserving null (topology stats) ---")
        print(topo_comparison_df.to_string(index=False))

        print(f"\n{'='*80}")
        print("Layer-preserving null analysis complete.")
        print(f"All outputs saved to: {output_path}")
        print(f"{'='*80}")
        return 0

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Error in layer-preserving null analysis: {e}")
        traceback.print_exc()
        print(f"{'='*80}")
        return 1


if __name__ == "__main__":
    sys.exit(main())