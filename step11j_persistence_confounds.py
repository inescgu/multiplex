#!/usr/bin/env python3
"""
step11j_persistence_confounds.py

FIX 1 -- Deconfound persistence from trivial network properties before you
claim persistence tracks "biological generality."

Modularity-family community detection (RB/Leiden, which is what your whole
pipeline runs) has a well-known resolution limit: bigger, denser clusters
mechanically resist being split as gamma increases, independent of any
biology. If persistence correlates strongly with size/degree/cross-layer
touch alone, your headline claim ("high persistence = broad pain biology")
is at serious risk of being "big clusters are big."

INPUT: the *_cluster_summary.csv written by step11e_characterize_clusters.py
(already has persistence, size, n_cell_types, cross_edge_fraction,
dominant_cell_type_frac, conserved_gene_frac -- everything needed, no
re-computation required).

OUTPUT:
  <prefix>_confound_report.txt   -- Spearman correlations + partial
                                     correlations (controlling for size)
  <prefix>_confound_scatter.png  -- persistence vs. each confound

USAGE
-----
    python step11j_persistence_confounds.py \
        --cluster-summary 11_hidef_results_faithful/characterized/heterogeneous_hidef_cluster_summary.csv \
        --output-prefix 11_hidef_results_faithful/characterized/heterogeneous_hidef
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFOUND_COLUMNS = [
    "size",
    "n_cell_types",
    "n_edges_intra",
    "n_edges_bipartite",
    "cross_edge_fraction",
    "dominant_cell_type_frac",
    "conserved_gene_frac",
]


def spearman_with_ci(x, y, n_boot=2000, seed=0):
    """Spearman rho + a bootstrap 95% CI, since with a modest number of
    clusters a single point estimate invites over-interpretation."""
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boots.append(r)
    lo, hi = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (np.nan, np.nan)
    return rho, p, lo, hi, n


def partial_spearman(x, y, z, n_boot=2000, seed=0):
    """Spearman correlation between x and y, controlling for z, via rank
    residuals (a standard nonparametric partial-correlation approximation).
    Use this to ask: does persistence still track n_cell_types (say) once
    you've removed what's explained by size?
    """
    mask = (~pd.isna(x)) & (~pd.isna(y)) & (~pd.isna(z))
    x, y, z = np.asarray(x)[mask], np.asarray(y)[mask], np.asarray(z)[mask]
    if len(x) < 6:
        return np.nan, np.nan, len(x)
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)

    def resid(a, b):
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)

    rx_resid = resid(rx, rz)
    ry_resid = resid(ry, rz)
    rho, p = stats.pearsonr(rx_resid, ry_resid)
    return rho, p, len(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster-summary", required=True,
                     help="*_cluster_summary.csv from step11e_characterize_clusters.py")
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.cluster_summary)
    persistence = df["persistence"]

    lines = ["=" * 72, "FIX 1: PERSISTENCE CONFOUND REPORT", "=" * 72,
              f"Input: {args.cluster_summary}", f"N clusters: {len(df)}", ""]

    print("Spearman(persistence, confound) -- raw:")
    raw_results = {}
    for col in CONFOUND_COLUMNS:
        if col not in df.columns:
            continue
        rho, p, lo, hi, n = spearman_with_ci(persistence, df[col])
        raw_results[col] = (rho, p, lo, hi, n)
        line = f"  persistence vs {col:<28} rho={rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.2e}  n={n}"
        print(" ", line)
        lines.append(line)

    lines.append("")
    lines.append("-" * 72)
    lines.append("Partial correlations controlling for SIZE (the classic resolution-limit "
                  "confound: bigger clusters resist splitting mechanically, not biologically)")
    lines.append("-" * 72)
    print("\nPartial Spearman(persistence, confound | size):")
    for col in CONFOUND_COLUMNS:
        if col == "size" or col not in df.columns:
            continue
        rho_p, p_p, n = partial_spearman(persistence, df[col], df["size"])
        line = f"  persistence vs {col:<28} | size   rho={rho_p:+.3f}  p={p_p:.2e}  n={n}"
        print(" ", line)
        lines.append(line)

    # ---- interpretation flags ----
    lines.append("")
    lines.append("-" * 72)
    lines.append("AUTOMATED FLAGS (read these before writing anything up)")
    lines.append("-" * 72)
    size_rho = raw_results.get("size", (np.nan,))[0]
    if not np.isnan(size_rho) and abs(size_rho) > 0.5:
        msg = (f"  \u26a0 persistence vs size raw rho = {size_rho:+.3f} (|rho|>0.5). "
               f"This is a strong signal that persistence is substantially explained by "
               f"cluster size alone (resolution-limit artifact), independent of biology. "
               f"Any 'persistence = generality' claim MUST be made net of this, e.g. by "
               f"reporting the size-controlled partial correlations above, or by binning "
               f"clusters into similar-size groups and checking the pattern holds within bins.")
        print(msg)
        lines.append(msg)
    else:
        msg = f"  persistence vs size raw rho = {size_rho:+.3f} -- not alarmingly high, but still report it."
        print(msg)
        lines.append(msg)

    # ---- plots ----
    present_cols = [c for c in CONFOUND_COLUMNS if c in df.columns]
    ncols = 3
    nrows = int(np.ceil(len(present_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4))
    axes = np.atleast_1d(axes).flatten()
    for i, col in enumerate(present_cols):
        ax = axes[i]
        ax.scatter(df[col], persistence, s=14, alpha=0.5, edgecolors="none")
        if col in ("size",):
            ax.set_xscale("log")
        rho = raw_results.get(col, (np.nan,))[0]
        ax.set_xlabel(col)
        ax.set_ylabel("persistence")
        ax.set_title(f"rho={rho:+.2f}" if not np.isnan(rho) else "n/a")
        ax.grid(alpha=0.3)
    for j in range(len(present_cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Persistence vs. structural confounds", y=1.02, fontsize=13)
    plt.tight_layout()
    plot_path = f"{args.output_prefix}_confound_scatter.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved scatter grid -> {plot_path}")

    report_path = f"{args.output_prefix}_confound_report.txt"
    Path(report_path).write_text("\n".join(lines))
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()