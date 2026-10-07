#!/usr/bin/env python
"""
step12c_aggregate_null.py  (Script C)

Collects the per-replicate CSVs from Script B and produces the outputs the
original script would have written at the end of the null stage:

  null_fc_f90.csv                  all replicates, all rows
  null_topology_stats.csv          one row per replicate
  observed_vs_null_comparison.csv  f_c / f_90, observed vs null
  null_topology_comparison.csv     assortativity, articulation points, ... vs null
  attack_curves.png                observed curves + null-median f_c line

Inputs already written by the main job (in <output-dir>/hub_removal/):
  f_c_f90_observed.csv, module_topology_stats.csv,
  attack_curve_<mode>_<static|cascading>.csv, attack_curve_random.csv

Usage:
    python step12c_aggregate_null.py --output-dir 12_hub_removal [--expected 50]
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

import step12_hub_removal_analysis as s12

TOPO_COLS = ["null_id", "n_nodes", "n_edges", "assortativity", "n_articulation_points",
             "frac_articulation_points", "n_bridges", "frac_bridges", "max_core_number",
             "mean_core_number", "hub_neighbor_clustering", "avg_clustering", "transitivity",
             "molloy_reed_ratio", "node_connectivity", "edge_connectivity",
             "algebraic_connectivity", "global_efficiency", "avg_shortest_path_length",
             "expensive_metrics_computed"]

TOPO_STATS = ["assortativity", "n_articulation_points", "frac_articulation_points",
              "n_bridges", "frac_bridges", "max_core_number", "mean_core_number",
              "hub_neighbor_clustering", "avg_clustering", "transitivity",
              "node_connectivity", "edge_connectivity", "algebraic_connectivity",
              "global_efficiency", "avg_shortest_path_length"]


def summarize(null_vals, obs):
    null_vals = np.asarray(null_vals, dtype=float)
    sd = np.std(null_vals)
    return {
        "observed": obs,
        "null_median": float(np.median(null_vals)),
        "null_mean": float(np.mean(null_vals)),
        "null_std": float(sd),
        "null_n": int(len(null_vals)),
        "observed_percentile_in_null": float(np.sum(null_vals <= obs) / len(null_vals) * 100),
        "z_score": float((obs - np.mean(null_vals)) / sd) if sd > 0 else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="12_hub_removal")
    ap.add_argument("--expected", type=int, default=None,
                    help="Expected number of replicates; warns about missing ids.")
    args = ap.parse_args()

    out = os.path.join(args.output_dir, "hub_removal")
    files = sorted(glob.glob(os.path.join(out, "null", "null_*.csv")))
    if not files:
        sys.exit(f"No null_*.csv files found in {out}/null/")
    null_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ids = sorted(null_df["null_id"].unique())
    print(f"Loaded {len(files)} replicate file(s): {len(ids)} unique null ids, {len(null_df)} rows")
    if args.expected is not None:
        missing = sorted(set(range(args.expected)) - set(ids))
        if missing:
            print(f"WARNING: missing replicate ids: {missing}")
    n_casc = null_df[null_df["cascading"]].groupby("mode")["null_id"].nunique()
    print("Replicates per cascading mode:\n" + n_casc.to_string())

    null_df.to_csv(os.path.join(out, "null_fc_f90.csv"), index=False)

    # ---- f_c / f_90 vs null ----
    obs_df = pd.read_csv(os.path.join(out, "f_c_f90_observed.csv"))
    rows = []
    for _, r in obs_df.iterrows():
        sub = null_df[(null_df["mode"] == r["mode"]) & (null_df["cascading"] == bool(r["cascading"]))]
        if sub.empty:
            continue
        for stat in ("f_c", "f_90"):
            vals = sub[stat].dropna().values
            if len(vals) == 0 or pd.isna(r[stat]):
                continue
            rows.append({"mode": r["mode"], "cascading": bool(r["cascading"]),
                         "statistic": stat, **summarize(vals, r[stat])})
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(out, "observed_vs_null_comparison.csv"), index=False)
    print("\n--- observed vs null (f_c / f_90) ---")
    print(comp.to_string(index=False))

    # ---- topology vs null ----
    topo_cols = [c for c in TOPO_COLS if c in null_df.columns]
    null_topo = null_df.drop_duplicates(subset="null_id")[topo_cols].reset_index(drop=True)
    null_topo.to_csv(os.path.join(out, "null_topology_stats.csv"), index=False)

    module_topo = pd.read_csv(os.path.join(out, "module_topology_stats.csv")).iloc[0]
    trows = []
    for stat in TOPO_STATS:
        if stat not in null_topo.columns:
            continue
        vals = null_topo[stat].dropna().values
        obs = module_topo.get(stat, np.nan)
        if len(vals) == 0 or pd.isna(obs):
            continue
        trows.append({"statistic": stat, **summarize(vals, float(obs))})
    tcomp = pd.DataFrame(trows)
    tcomp.to_csv(os.path.join(out, "null_topology_comparison.csv"), index=False)
    print("\n--- observed vs null (topology) ---")
    print(tcomp.to_string(index=False))

    def get(stat, col):
        r = tcomp[tcomp["statistic"] == stat]
        return None if r.empty else r.iloc[0][col]

    if get("assortativity", "observed") is not None and \
            get("assortativity", "observed") < get("assortativity", "null_median"):
        print("NOTE: module is more disassortative than the null median.")
    if get("frac_articulation_points", "observed") is not None and \
            get("frac_articulation_points", "observed") > get("frac_articulation_points", "null_median"):
        print("NOTE: module has more articulation points than the null median.")
    if get("hub_neighbor_clustering", "observed") is not None and \
            get("hub_neighbor_clustering", "observed") < get("hub_neighbor_clustering", "null_median"):
        print("NOTE: hub neighborhoods are less clustered than the null median.")

    # ---- plot ----
    curves = {}
    for mode in ("degree", "betweenness", "degree_betweenness"):
        for casc, tag in ((False, "static"), (True, "cascading")):
            p = os.path.join(out, f"attack_curve_{mode}_{tag}.csv")
            if os.path.exists(p):
                curves[(mode, casc)] = pd.read_csv(p)
    p = os.path.join(out, "attack_curve_random.csv")
    if os.path.exists(p):
        curves["random_mean"] = pd.read_csv(p)
    if any(isinstance(k, tuple) for k in curves):
        s12.plot_attack_curves(curves, null_df, out)
    else:
        print("No observed attack_curve_*.csv files found; skipping plot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())