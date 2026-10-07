#!/usr/bin/env python3
"""
step11i_aggregate.py

Pools the per-task outputs of step11i_cluster_level_null_model.py and produces
the final tables / plots.

    python step11i_aggregate.py --root results/11i_null/nperms200_XXXX \
        --output-dir results/11i_null/nperms200_XXXX/aggregate [--expected-perms 200]

Reads:   <root>/true/true_summary.json
         <root>/null/task_*/null_perms.csv
         <root>/loo/task_*/loo.csv           (optional)
Writes:  null_hierarchy_summaries.csv, step11i_null_vs_true_results.csv,
         step11i_null_distributions.png, null_run_diagnostics.csv,
         step11i_leave_one_out.csv/.png (if LOO was run)

Comparison subsets (true = the file module):
  all                 every successful null
  size_matched_20pct  nulls within +-20% of the true module size
  size_adjusted       linear regression of each metric on module size across the
                      nulls; tests the true module's residual (uses all nulls)
p_two_sided_BH = Benjamini-Hochberg across the 6 metrics within each subset.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMMARY_KEYS = ["n_clusters", "max_persistence", "frac_at_max_persistence",
                "n_distinct_persistence_values", "top_cluster_frac_of_module",
                "mean_persistence"]
HIGHER_IS_BETTER = {"max_persistence", "n_distinct_persistence_values", "mean_persistence"}


def compare(true_summary, null_df, ref, subset):
    rows = []
    for metric in SUMMARY_KEYS:
        if metric not in null_df.columns:
            continue
        vals = null_df[metric].dropna().to_numpy(dtype=float)
        n = len(vals)
        if n == 0:
            continue
        tv = float(true_summary[metric])
        mu = vals.mean()
        sd = vals.std(ddof=1) if n > 1 else np.nan
        z = (tv - mu) / sd if np.isfinite(sd) and sd > 0 else np.nan
        k_two = int(np.sum(np.abs(vals - mu) >= abs(tv - mu)))
        p_two = (1 + k_two) / (1 + n)
        if metric in HIGHER_IS_BETTER:
            p_one = (1 + int(np.sum(vals >= tv))) / (1 + n)
        else:
            p_one = (1 + int(np.sum(vals <= tv))) / (1 + n)
        rows.append({"true_reference": ref, "null_subset": subset, "n_null": n,
                     "metric": metric, "true_value": tv, "null_mean": mu, "null_std": sd,
                     "z_score": z, "p_two_sided": p_two, "p_one_sided_prespecified": p_one})
    return rows


def compare_size_adjusted(true_summary, true_n, null_df, ref):
    """Regress each metric on module size across nulls; test the true module's residual."""
    rows = []
    d = null_df.dropna(subset=["n_nodes"])
    if len(d) < 10 or np.ptp(d["n_nodes"].to_numpy(float)) == 0:
        return rows
    x = d["n_nodes"].to_numpy(float)
    for metric in SUMMARY_KEYS:
        if metric not in d.columns:
            continue
        y = d[metric].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 10:
            continue
        b, a = np.polyfit(x[ok], y[ok], 1)
        resid = y[ok] - (a + b * x[ok])
        sd = resid.std(ddof=2)
        expected = a + b * true_n
        tr = float(true_summary[metric]) - expected
        z = tr / sd if sd > 0 else np.nan
        k = int(np.sum(np.abs(resid) >= abs(tr)))
        rows.append({"true_reference": ref, "null_subset": "size_adjusted", "n_null": int(ok.sum()),
                     "metric": metric, "true_value": float(true_summary[metric]),
                     "null_mean": expected, "null_std": sd, "z_score": z,
                     "p_two_sided": (1 + k) / (1 + ok.sum()), "p_one_sided_prespecified": np.nan})
    return rows


def add_bh(res):
    res["p_two_sided_BH"] = np.nan
    for _, idx in res.groupby(["true_reference", "null_subset"]).groups.items():
        idx = list(idx)
        p = res.loc[idx, "p_two_sided"].to_numpy(float)
        m = len(p)
        order = np.argsort(p)
        adj_sorted = p[order] * m / np.arange(1, m + 1)
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(adj_sorted, 1.0)
        res.loc[idx, "p_two_sided_BH"] = adj
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--expected-perms", type=int, default=None)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(root / "true" / "true_summary.json") as f:
        true = json.load(f)
    refs = [r for r in ("file", "rebuilt") if r in true]

    files = sorted(root.glob("null/task_*/null_perms.csv"))
    if not files:
        raise SystemExit("No null_perms.csv files found.")
    allp = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    allp = allp.drop_duplicates(subset="perm_idx", keep="last").sort_values("perm_idx")
    ok = allp[allp["status"] == "ok"].copy()

    print("=" * 70)
    print("RUN DIAGNOSTICS")
    print("=" * 70)
    print(f"task files: {len(files)}; perms recorded: {len(allp)}"
          + (f" / expected {args.expected_perms}" if args.expected_perms else ""))
    print("status counts:\n" + allp["status"].value_counts().to_string())
    print(f"fraction of perms NOT ok (skipped/failed): {(allp['status'] != 'ok').mean():.2f}"
          "   (report this if > 0.05)")
    if len(ok) == 0:
        raise SystemExit("No successful perms to analyse.")
    ok.to_csv(out / "null_hierarchy_summaries.csv", index=False)

    tn = true["file"]["n_nodes"]
    print(f"\nmodule size: true(file)={tn}"
          + (f", true(rebuilt)={true['rebuilt']['n_nodes']}" if "rebuilt" in true else ""))
    print("null final module size:\n" + ok["module_size_final"].describe().round(1).to_string())
    print(f"frac null modules connected before cap: {ok['connected_raw'].astype(bool).mean():.2f}; "
          f"hit cap: {ok['capped'].astype(bool).mean():.2f}")
    print("gamma status counts:\n" + ok["gamma_status"].value_counts().to_string())
    print(f"gamma_lo range [{ok['gamma_lo'].min():.3g}, {ok['gamma_lo'].max():.3g}], "
          f"gamma_hi range [{ok['gamma_hi'].min():.3g}, {ok['gamma_hi'].max():.3g}]")

    mc = true.get("mx_check")
    if mc:
        print(f"\nMultiXrank builder check: MultiXrank on the true seeds gives a module of "
              f"{mc['module_size']} nodes, Jaccard with the file module = {mc['jaccard_with_file_module']:.3f}"
              + ("   <-- LOW, nulls may not be comparable" if mc["jaccard_with_file_module"] < 0.8 else ""))

    # bridging genes: true module = file module size - anchoring seeds (when no LCC fallback dropped seeds)
    nb = ok["n_bridges"].dropna()
    if len(nb):
        tb = true.get("rebuilt", {}).get("n_bridges")
        if tb is None:
            tb = true["n_true_module"] - true["n_true_seeds"]
        print("\nbridging genes needed to connect null seeds:\n" + nb.describe().round(1).to_string())
        k_le = int((nb <= tb).sum())
        print(f"  true seeds needed {tb}; nulls needing <= that many: {k_le}/{len(nb)}; "
              f"empirical one-sided p (true seeds unusually well connected) = {(1 + k_le) / (1 + len(nb)):.4f}")
    if "rebuilt" in true:
        print(f"true(rebuilt) vs true(file) module Jaccard: "
              f"{true['rebuilt'].get('jaccard_with_file_module', float('nan')):.3f}")

    t_mean, t_max = ok["t_total_s"].mean(), ok["t_total_s"].max()
    peak = allp["peak_rss_gb"].max()
    print(f"\nper-perm time: mean {t_mean:.0f}s, max {t_max:.0f}s; peak RSS over all tasks: {peak:.2f} GiB")
    print("BUDGET SUGGESTION (from this run):")
    print(f"  --mem  ~ {int(np.ceil(peak * 1.5))}G   (peak x 1.5)")
    for p in (5, 10, 25, 50):
        h = t_mean * p * 1.3 / 3600
        print(f"  --time for {p:>2} perms/task ~ {h:.1f} h (+ ~10 min graph load)")
    allp.to_csv(out / "null_run_diagnostics.csv", index=False)

    rows = []
    for ref in refs:
        ts, tnn = true[ref]["summary"], true[ref]["n_nodes"]
        rows += compare(ts, ok, ref, "all")
        sm = ok[(ok["n_nodes"] >= 0.8 * tnn) & (ok["n_nodes"] <= 1.2 * tnn)]
        rows += compare(ts, sm, ref, "size_matched_20pct")
        rows += compare_size_adjusted(ts, tnn, ok, ref)
    res = add_bh(pd.DataFrame(rows))
    res.to_csv(out / "step11i_null_vs_true_results.csv", index=False)
    print("\n" + "=" * 70)
    print("TRUE vs NULL: hierarchy-shape statistics")
    print("=" * 70)
    print(res.round(4).to_string(index=False))
    print("\nHow to read: p_two_sided is an empirical p = (1+k)/(1+n); with n nulls the smallest "
          "possible value is 1/(1+n). Use p_two_sided_BH for the 6-metric family. 'size_adjusted' "
          "removes the effect of module size; if it disagrees with 'all', size is driving the result. "
          "If the true module is not separated from the null, the broad/specific nesting is not "
          "evidence of disease-specific structure.")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, metric in zip(axes.ravel(), SUMMARY_KEYS):
        ax.hist(ok[metric], bins=20, color="steelblue", alpha=0.7, edgecolor="white")
        if "file" in true:
            ax.axvline(true["file"]["summary"][metric], color="crimson", lw=2, label="true (file)")
        if "rebuilt" in true:
            ax.axvline(true["rebuilt"]["summary"][metric], color="darkorange", lw=2, ls="--",
                       label="true (rebuilt)")
        ax.set_title(metric)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out / "step11i_null_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    loo_files = sorted(root.glob("loo/task_*/loo.csv"))
    if loo_files:
        loo = pd.concat([pd.read_csv(p) for p in loo_files], ignore_index=True)
        loo = loo.drop_duplicates(subset="dropped_seed", keep="last")
        print("\n" + "=" * 70)
        print("LEAVE-ONE-SEED-OUT")
        print("=" * 70)
        print("status counts:\n" + loo["status"].value_counts().to_string())
        l = loo[loo["status"] == "ok"].sort_values("best_match_jaccard_vs_rebuilt")
        l.to_csv(out / "step11i_leave_one_out.csv", index=False)
        if len(l):
            j = l["best_match_jaccard_vs_rebuilt"]
            print(f"best-match Jaccard (does the full-seed top cluster still exist, in any rank?): "
                  f"mean={j.mean():.3f}, median={j.median():.3f}, min={j.min():.3f} "
                  f"(seed {l.iloc[0]['dropped_seed']})")
            print(f"strict top-cluster Jaccard (same cluster ranked #1): "
                  f"mean={l['top_cluster_jaccard_vs_rebuilt'].mean():.3f}")
            print(f"module Jaccard vs reference: mean={l['module_jaccard_vs_rebuilt'].mean():.3f}, "
                  f"min={l['module_jaccard_vs_rebuilt'].min():.3f}")
            print("5 most influential seeds:\n" +
                  l[["dropped_seed", "module_jaccard_vs_rebuilt",
                     "best_match_jaccard_vs_rebuilt"]].head(5).round(3).to_string(index=False))
            fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(l))))
            ax.barh(l["dropped_seed"], l["best_match_jaccard_vs_rebuilt"], color="#4C72B0")
            ax.set_xlabel("Best-match Jaccard of the full-seed top cluster")
            ax.set_title("Leave-one-seed-out sensitivity of the broadest cluster")
            plt.tight_layout()
            fig.savefig(out / "step11i_leave_one_out.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"\nAll outputs in {out}")


if __name__ == "__main__":
    main()