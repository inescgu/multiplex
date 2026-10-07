#!/usr/bin/env python
"""
step12f_aggregate_matched.py

Aggregates step12e_matched_replicate.py outputs and compares the OBSERVED
module (measured with the same sampled-betweenness settings as the nulls)
against each null variant, at every LCC threshold.

Usage:  python step12f_aggregate_matched.py --output-dir 12_hub_removal

Writes <output-dir>/hub_removal/matched/:
  matched_fc_comparison.csv      observed vs each null, per mode/cascading/threshold
  matched_topology_comparison.csv  topology stats, observed vs each null
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd

THRESHOLDS = [0.9, 0.75, 0.5, 0.25, 0.10, 0.05]
NULLS = ["naive", "layer", "layer_seed"]
TOPO = ["assortativity", "n_articulation_points", "frac_articulation_points", "n_bridges",
        "max_core_number", "mean_core_number", "hub_neighbor_clustering",
        "avg_clustering", "transitivity"]


def load(base, variant):
    files = sorted(glob.glob(os.path.join(base, variant, "rep_*.csv")))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="12_hub_removal")
    a = ap.parse_args()
    base = os.path.join(a.output_dir, "hub_removal", "matched")

    obs = load(base, "observed")
    if obs is None:
        raise SystemExit("No observed reps found (run --variant observed first).")
    nulls = {v: load(base, v) for v in NULLS}
    nulls = {v: d for v, d in nulls.items() if d is not None}
    print(f"observed reps: {obs.rep_id.nunique()} | " +
          " | ".join(f"{v}: {d.rep_id.nunique()} reps" for v, d in nulls.items()))

    rows = []
    for (mode, casc), og in obs.groupby(["mode", "cascading"]):
        for t in THRESHOLDS:
            col = f"f_lcc_le_{t}"
            ov = og[col].dropna().values
            if len(ov) == 0:
                continue
            row = {"mode": mode, "cascading": casc, "lcc_threshold": t,
                   "observed_median": np.median(ov), "observed_min": ov.min(),
                   "observed_max": ov.max(), "observed_n": len(ov)}
            for v, d in nulls.items():
                nv = d[(d["mode"] == mode) & (d["cascading"] == casc)][col].dropna().values
                if len(nv) == 0:
                    continue
                row[f"{v}_median"] = np.median(nv)
                row[f"{v}_std"] = np.std(nv)
                row[f"{v}_n"] = len(nv)
                # fraction of null replicates at or below the observed median
                row[f"{v}_obs_percentile"] = float(np.mean(nv <= np.median(ov)) * 100)
                # gap in units of null SD (descriptive; nulls are tight)
                row[f"{v}_gap_in_null_sd"] = ((np.median(ov) - np.median(nv)) / np.std(nv)
                                              if np.std(nv) > 0 else np.nan)
                # observed sampling noise vs gap: gap must exceed observed spread
                row[f"{v}_gap_exceeds_obs_range"] = bool(
                    np.median(ov) < nv.min() - (ov.max() - ov.min())
                    or np.median(ov) > nv.max() + (ov.max() - ov.min()))
            rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(base, "matched_fc_comparison.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    show = ["observed_median"] + [f"{v}_median" for v in nulls]
    for (mode, casc), g in comp.groupby(["mode", "cascading"]):
        print(f"\n=== {mode} ({'cascading' if casc else 'static'}): fraction removed when LCC <= t ===")
        print(g.set_index("lcc_threshold")[show].round(4).to_string())

    # topology
    trows = []
    otopo = obs.drop_duplicates("rep_id").iloc[0]
    for s in TOPO:
        row = {"statistic": s, "observed": otopo[s]}
        for v, d in nulls.items():
            vals = d.drop_duplicates("rep_id")[s].dropna().values
            if len(vals):
                row[f"{v}_median"] = np.median(vals)
                row[f"{v}_std"] = np.std(vals)
        trows.append(row)
    tcomp = pd.DataFrame(trows)
    tcomp.to_csv(os.path.join(base, "matched_topology_comparison.csv"), index=False)
    print("\n=== topology: observed vs nulls ===")
    print(tcomp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()