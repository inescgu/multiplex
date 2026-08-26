#!/usr/bin/env python
"""
Hub-Removal Robustness: Observed vs. Degree-Preserving Null
--------------------------------------------------------------
Builds a grouped bar chart comparing the real disease module's
attack-tolerance (f_c: fraction of nodes removed before the giant
component collapses to <=5% of its original size) against the median
of degree-preserving rewired null networks, for each attack mode.

Reads directly from observed_vs_null_comparison.csv (output of
hub_removal_analysis.py). The CSV path is required -- this script does
not fall back to any hard-coded values.

Usage:
    python plot_hub_removal_robustness.py <path/to/observed_vs_null_comparison.csv> [output_dir]
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load data (from CSV if provided/found, else fall back to known run values)
# ---------------------------------------------------------------------------

def load_data(csv_path):
    if not csv_path:
        print("ERROR: no CSV path given.")
        print("Usage: python plot_hub_removal_robustness.py <path/to/observed_vs_null_comparison.csv> [output_dir]")
        sys.exit(1)
    if not os.path.exists(csv_path):
        print(f"ERROR: file not found: {csv_path}")
        print("Run hub_removal_analysis.py first to generate observed_vs_null_comparison.csv.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required_cols = {"mode", "cascading", "statistic", "observed", "null_median", "null_std"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: {csv_path} is missing expected column(s): {sorted(missing)}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_robustness(df, statistic="f_c", cascading=False, output_path="hub_removal_robustness.png"):
    """
    Grouped bar chart: observed vs null-median, one group per attack mode,
    for a single statistic (f_c or f_90) and attack style (static/cascading).
    """
    sub = df[(df["statistic"] == statistic) & (df["cascading"] == cascading)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for statistic={statistic}, cascading={cascading}")

    # Nice display order / labels
    mode_order = ["degree", "betweenness", "degree_betweenness"]
    mode_labels = {"degree": "Degree", "betweenness": "Betweenness", "degree_betweenness": "Degree +\nBetweenness"}
    sub["mode"] = pd.Categorical(sub["mode"], categories=mode_order, ordered=True)
    sub = sub.sort_values("mode")

    modes = sub["mode"].tolist()
    observed = sub["observed"].values
    null_median = sub["null_median"].values
    null_std = sub["null_std"].values

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars_obs = ax.bar(
        x - width / 2, observed, width,
        label="Real disease module", color="#c0392b", edgecolor="black", zorder=3,
    )
    bars_null = ax.bar(
        x + width / 2, null_median, width,
        yerr=null_std, capsize=5,
        label="Degree-preserving null\n(median \u00b1 SD)", color="#95a5a6", edgecolor="black", zorder=3,
    )

    # Value labels on top of bars
    for bar, val in zip(bars_obs, observed):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c0392b")
    for bar, val in zip(bars_null, null_median):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=10, color="#555555")

    stat_label = {
        "f_c": "Fraction of nodes removed before network\ncollapses to \u22645% of original size (f\u1D04)",
        "f_90": "Fraction of nodes removed before network\ndrops to \u226490% of original size (f\u2089\u2080)",
    }[statistic]
    style_label = "Cascading attack (re-ranked as nodes are removed)" if cascading else "Static attack (fixed ranking)"

    ax.set_ylabel(stat_label, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([mode_labels[m] for m in modes], fontsize=11)
    ax.set_title(
        f"Disease Module Does Not Resist Hub Removal Better Than Expected\n{style_label}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, max(observed.max(), null_median.max()) * 1.25)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate the takeaway
    ax.text(
        0.5, -0.18,
        "Higher bars = more removal needed to fragment the network \u2192 the real module is not more \n"
        "robust than a random network with the same degree sequence, across most attack strategies tested.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.5, style="italic", color="#333333",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_percentiles(df, output_path="hub_removal_percentile_summary.png"):
    """
    Compact summary panel: for every mode/cascading/statistic combo, show
    observed_percentile_in_null (or compute it if not present) as a single
    bar chart. Communicates "how extreme is the real module vs the null
    distribution" in one glance.
    """
    sub = df.copy()
    if "observed_percentile_in_null" not in sub.columns:
        raise ValueError("observed_percentile_in_null column not found; "
                          "pass the full observed_vs_null_comparison.csv for this plot.")

    sub["label"] = (
        sub["mode"].str.replace("_", "+").str.title()
        + " (" + np.where(sub["cascading"], "cascading", "static") + ", "
        + sub["statistic"] + ")"
    )
    sub = sub.sort_values("observed_percentile_in_null")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c0392b" if p >= 95 else "#e67e22" if p >= 80 else "#95a5a6" for p in sub["observed_percentile_in_null"]]
    ax.barh(sub["label"], sub["observed_percentile_in_null"], color=colors, edgecolor="black")
    ax.axvline(50, color="gray", linestyle=":", alpha=0.6, label="Null median (50th pct)")
    ax.set_xlabel("Percentile of observed value within null distribution", fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_title("How Extreme Is the Real Module vs. Degree-Preserving Nulls?",
                  fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_hub_removal_robustness.py <path/to/observed_vs_null_comparison.csv> [output_dir]")
        sys.exit(1)

    csv_arg = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    df = load_data(csv_arg)
    os.makedirs(out_dir, exist_ok=True)

    # Main slide chart: static f_c, all three attack modes
    plot_robustness(df, statistic="f_c", cascading=False,
                     output_path=os.path.join(out_dir, "hub_removal_robustness_static_fc.png"))

    # Cascading version, for a second slide / appendix if useful
    plot_robustness(df, statistic="f_c", cascading=True,
                     output_path=os.path.join(out_dir, "hub_removal_robustness_cascading_fc.png"))

    # Percentile summary panel (only works with full CSV incl. percentile column)
    if "observed_percentile_in_null" in df.columns:
        plot_all_percentiles(df, output_path=os.path.join(out_dir, "hub_removal_percentile_summary.png"))
    else:
        print("Skipping percentile summary plot (no observed_percentile_in_null column in fallback data).")