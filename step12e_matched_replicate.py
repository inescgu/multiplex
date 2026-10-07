#!/usr/bin/env python
"""
step12e_matched_replicate.py

ONE replicate per invocation (Slurm array). Every variant is measured with the
SAME attack settings (same step, same sampled betweenness k, same early stop),
so differences between variants are not method artifacts.

Variants (--variant):
  observed    the real module, no rewiring. Replicates differ only in the
              betweenness-sampling seed -> gives the sampling noise of the
              observed value under k-sampled betweenness.
  naive       whole-graph degree-preserving double_edge_swap (what step12 did).
  layer       layer-preserving null: swaps only among intra-layer edges of the
              SAME cell type; inter-layer identity edges untouched.
  layer_seed  as 'layer', but also stratified seed-seed / seed-nonseed /
              nonseed-nonseed (needs --disease-module).

Records the first fraction removed at which LCC fraction <= t for several
thresholds t (0.9, 0.75, 0.5, 0.25, 0.10, 0.05), not just f_c / f_90.
Threshold 0.05 ~ the size of ONE cell-type layer in your module (3.3-5.7% of
nodes), so f_c alone may mostly measure "layers got separated".

Output: <output-dir>/hub_removal/matched/<variant>/rep_<id>.csv
Skips replicates whose final file exists (resubmit-safe). Needs
step12_hub_removal_analysis.py and step12b_layer_preserving_null.py in the
same directory.
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step12_hub_removal_analysis as s12
from step12b_layer_preserving_null import generate_layer_preserving_null

THRESHOLDS = [0.9, 0.75, 0.5, 0.25, 0.10, 0.05]
VARIANTS = ["observed", "naive", "layer", "layer_seed"]
TOPO_KEYS = ["assortativity", "n_articulation_points", "frac_articulation_points",
             "n_bridges", "max_core_number", "mean_core_number",
             "hub_neighbor_clustering", "avg_clustering", "transitivity"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def first_crossings(G, mode, cascading, step, cascading_step, k, seed, thresholds=THRESHOLDS):
    """First frac_removed at which LCC fraction <= t, for each t. Stops at
    the smallest threshold (nothing later can change a first crossing)."""
    H = G.copy()
    n0 = H.number_of_nodes()
    stop = min(thresholds)
    out = {t: np.nan for t in thresholds}
    removed = 0

    def record(frac_lcc, frac_removed):
        for t in thresholds:
            if np.isnan(out[t]) and frac_lcc <= t:
                out[t] = frac_removed
        return frac_lcc <= stop

    if not cascading:
        order = s12._centrality_order(H, mode, betweenness_k=k, seed=seed)
        for i in range(0, len(order), step):
            batch = order[i:i + step]
            H.remove_nodes_from(batch)
            removed += len(batch)
            _, frac = s12._lcc_size_and_fraction(H, n0)
            if record(frac, removed / n0):
                break
    else:
        while H.number_of_nodes() > 0:
            order = s12._centrality_order(H, mode, betweenness_k=k, seed=seed)
            batch = order[:cascading_step]
            H.remove_nodes_from(batch)
            removed += len(batch)
            _, frac = s12._lcc_size_and_fraction(H, n0)
            if record(frac, removed / n0):
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-graph", required=True)
    ap.add_argument("--disease-module", default=None, help="needed for --variant layer_seed")
    ap.add_argument("--output-dir", default="12_hub_removal")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--rep-id", type=int, default=None, help="default: $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--cascading-step", type=int, default=10)
    ap.add_argument("--betweenness-k", type=int, default=200)
    ap.add_argument("--n-cascading", type=int, default=15,
                    help="only reps with id < this run cascading modes (observed: all reps run them)")
    ap.add_argument("--swap-factor", type=int, default=10)
    ap.add_argument("--topology-hub-pct", type=float, default=0.10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rep = args.rep_id
    if rep is None:
        if "SLURM_ARRAY_TASK_ID" not in os.environ:
            sys.exit("Need --rep-id or $SLURM_ARRAY_TASK_ID")
        rep = int(os.environ["SLURM_ARRAY_TASK_ID"])

    out_dir = s12.ensure_dir(os.path.join(args.output_dir, "hub_removal", "matched", args.variant))
    final = os.path.join(out_dir, f"rep_{rep}.csv")
    partial = final + ".partial"
    if os.path.exists(final) and not args.force:
        log(f"{final} exists -- skipping (use --force).")
        return 0

    seed = int(np.random.SeedSequence([args.seed, VARIANTS.index(args.variant), rep])
               .generate_state(1)[0] % (2**31 - 1))
    do_cascading = True if args.variant == "observed" else rep < args.n_cascading
    modes = ["degree", "betweenness", "degree_betweenness"]
    log(f"{args.variant} rep {rep}: seed={seed} cascading={do_cascading} step={args.step} "
        f"casc_step={args.cascading_step} k={args.betweenness_k}")

    G = s12.load_pickle(args.module_graph)
    seed_node_set = None
    if args.variant == "layer_seed":
        if not args.disease_module:
            sys.exit("--variant layer_seed needs --disease-module")
        dm = pd.read_csv(args.disease_module)
        if not {"node", "is_seed"} <= set(dm.columns):
            sys.exit("--disease-module needs 'node' and 'is_seed' columns")
        seed_node_set = set(dm.loc[dm["is_seed"] == True, "node"])
        log(f"{len(seed_node_set)} seed nodes loaded")

    if args.variant == "observed":
        H = G
    elif args.variant == "naive":
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges())
        n_swaps = max(1, args.swap_factor * H.number_of_edges())
        try:
            nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)
        except nx.NetworkXError as e:
            log(f"double_edge_swap warning: {e}")
    else:
        H = generate_layer_preserving_null(G, seed_node_set=seed_node_set,
                                           stratify_by_seed=(args.variant == "layer_seed"),
                                           swap_factor=args.swap_factor, seed=seed)

    # sanity: degree sequence must be preserved by every null
    if args.variant != "observed":
        dg, dh = dict(G.degree()), dict(H.degree())
        if any(dg[n] != dh[n] for n in dg):
            sys.exit("FATAL: null did not preserve the degree sequence")
    log(f"graph ready: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    topo = s12.compute_topology_stats(H, hub_pct=args.topology_hub_pct, include_expensive=False)
    topo = {k: topo[k] for k in TOPO_KEYS}

    rows = []
    for mode in modes:
        for casc in ([False, True] if do_cascading else [False]):
            t0 = time.time()
            cr = first_crossings(H, mode, casc, args.step, args.cascading_step,
                                 args.betweenness_k, seed)
            row = {"variant": args.variant, "rep_id": rep, "mode": mode, "cascading": casc,
                   **{f"f_lcc_le_{t}": cr[t] for t in THRESHOLDS}, **topo}
            rows.append(row)
            pd.DataFrame(rows).to_csv(partial, index=False)
            log(f"  {mode} cascading={casc}: f(0.05)={cr[0.05]:.4f} f(0.5)={cr[0.5]:.4f} "
                f"({time.time()-t0:.0f}s)")
    os.replace(partial, final)
    log(f"done -> {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())