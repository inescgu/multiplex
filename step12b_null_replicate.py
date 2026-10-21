#!/usr/bin/env python
"""
step12b_null_replicate.py  (Script B)

ONE degree-preserving null replicate per invocation, designed for a Slurm
array (--array=0-49%10). Writes  <output-dir>/hub_removal/null/null_<id>.csv
with the same columns as the original null_fc_f90.csv rows (null_id, mode,
cascading, f_c, f_90 + the topology stats bundle).

Differences from the original _null_worker (results for f_c / f_90 are
unchanged by the early stop; the rest are speed fixes):

  1. STATIC betweenness / degree_betweenness now use --betweenness-k
     (source-sampled). The original passed no k there, so every replicate did
     two EXACT betweenness runs on ~10k nodes -- the main reason the job never
     finished.
  2. EARLY STOP: attack curves stop as soon as the LCC fraction drops to
     <= 0.05 (f_c). f_c and f_90 are first-crossing statistics, so nothing
     after that point can change them. For cascading betweenness (f_c ~ 0.08)
     this skips >90% of the recomputations.
  3. Recording/recompute granularity is --step / --cascading-step (default 10).
     f_c/f_90 resolution becomes 0.1% of module size. The observed module's
     values were computed at step 1; the difference is <= 0.001 and negligible
     against null_std.
  4. Rows are flushed to a .partial file after every (mode, static/cascading)
     unit, and the final file is renamed into place only when complete. An
     existing complete null_<id>.csv is skipped, so re-submitting the array
     only redoes missing replicates.

Usage (normally via Slurm array):
    python step12b_null_replicate.py \
        --module-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
        --output-dir 12_hub_removal \
        --null-id $SLURM_ARRAY_TASK_ID
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import networkx as nx

import step12_hub_removal_analysis as s12


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def attack_curve_early_stop(G, mode, cascading, step, cascading_step,
                            betweenness_k, seed, stop_lcc=0.05):
    """Same measurement as s12.attack_curve for the non-random modes, but
    stops once lcc_fraction <= stop_lcc. Returns a DataFrame usable with
    s12.critical_fraction (default thresholds 0.05 / 0.9)."""
    H = G.copy()
    n0 = H.number_of_nodes()
    records = [(0, 0.0, n0, 1.0)]
    removed = 0
    if not cascading:
        order = s12._centrality_order(H, mode, betweenness_k=betweenness_k, seed=seed)
        for i in range(0, len(order), step):
            batch = order[i:i + step]
            H.remove_nodes_from(batch)
            removed += len(batch)
            size, frac = s12._lcc_size_and_fraction(H, n0)
            records.append((removed, removed / n0, size, frac))
            if frac <= stop_lcc:
                break
    else:
        while H.number_of_nodes() > 0:
            order = s12._centrality_order(H, mode, betweenness_k=betweenness_k, seed=seed)
            batch = order[:cascading_step]
            H.remove_nodes_from(batch)
            removed += len(batch)
            size, frac = s12._lcc_size_and_fraction(H, n0)
            records.append((removed, removed / n0, size, frac))
            if frac <= stop_lcc:
                break
    return pd.DataFrame(records, columns=["n_removed", "frac_removed", "lcc_size", "lcc_fraction"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-graph", required=True)
    ap.add_argument("--output-dir", default="12_hub_removal",
                    help="Same --output-dir you gave the main script; files go under "
                         "<output-dir>/hub_removal/null/")
    ap.add_argument("--null-id", type=int, default=None,
                    help="Replicate index. Defaults to $SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--seed", type=int, default=0, help="Base seed (combined with null-id).")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--cascading-step", type=int, default=10)
    ap.add_argument("--betweenness-k", type=int, default=200,
                    help="Sampled sources for approximate betweenness (static AND cascading).")
    ap.add_argument("--cascading-modes", default="degree,betweenness,degree_betweenness")
    ap.add_argument("--n-null-cascading", type=int, default=15,
                    help="Only replicates with null-id < this run the cascading modes.")
    ap.add_argument("--swap-factor", type=int, default=10)
    ap.add_argument("--topology-hub-pct", type=float, default=0.10)
    ap.add_argument("--expensive-metrics-max-nodes", type=int, default=3000)
    ap.add_argument("--force", action="store_true", help="Recompute even if the output exists.")
    args = ap.parse_args()

    null_id = args.null_id
    if null_id is None:
        if "SLURM_ARRAY_TASK_ID" not in os.environ:
            sys.exit("Need --null-id or $SLURM_ARRAY_TASK_ID")
        null_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    out_dir = s12.ensure_dir(os.path.join(args.output_dir, "hub_removal", "null"))
    final = os.path.join(out_dir, f"null_{null_id}.csv")
    partial = final + ".partial"
    if os.path.exists(final) and not args.force:
        log(f"{final} already exists -- skipping (use --force to redo).")
        return 0

    seed = int(np.random.SeedSequence([args.seed, null_id]).generate_state(1)[0] % (2**31 - 1))
    modes = ["degree", "betweenness", "degree_betweenness"]
    cascading_modes = {m.strip() for m in args.cascading_modes.split(",") if m.strip()} & set(modes)
    do_cascading = null_id < args.n_null_cascading

    log(f"null {null_id}: seed={seed}, cascading={'yes' if do_cascading else 'no'}, "
        f"step={args.step}, cascading_step={args.cascading_step}, k={args.betweenness_k}")

    G = s12.load_pickle(args.module_graph)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    n_swaps = max(1, args.swap_factor * H.number_of_edges())
    try:
        nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)
    except nx.NetworkXError as e:
        log(f"double_edge_swap warning: {e}")
    log(f"rewired: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    topo = s12.compute_topology_stats(H, hub_pct=args.topology_hub_pct,
                                      include_expensive=False,
                                      expensive_max_nodes=args.expensive_metrics_max_nodes)
    log("topology stats done")

    rows = []

    def flush():
        pd.DataFrame(rows).to_csv(partial, index=False)

    for mode in modes:
        t0 = time.time()
        df = attack_curve_early_stop(H, mode, False, args.step, args.cascading_step,
                                     args.betweenness_k, seed)
        fc, f90 = s12.critical_fraction(df)
        rows.append({"null_id": null_id, "mode": mode, "cascading": False,
                     "f_c": fc, "f_90": f90, **topo})
        flush()
        log(f"  static {mode}: f_c={fc:.4f} f_90={f90:.4f} ({time.time()-t0:.0f}s)")

        if do_cascading and mode in cascading_modes:
            t0 = time.time()
            df = attack_curve_early_stop(H, mode, True, args.step, args.cascading_step,
                                         args.betweenness_k, seed)
            fc, f90 = s12.critical_fraction(df)
            rows.append({"null_id": null_id, "mode": mode, "cascading": True,
                         "f_c": fc, "f_90": f90, **topo})
            flush()
            log(f"  cascading {mode}: f_c={fc:.4f} f_90={f90:.4f} ({time.time()-t0:.0f}s)")

    os.replace(partial, final)
    log(f"done -> {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())