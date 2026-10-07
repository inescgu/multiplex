#!/usr/bin/env python3
"""
Step 11h: Do clusters look different depending on how persistent they are?
Bins consensus clusters into persistence tiers and compares size, density,
and composition metrics across tiers -- built entirely from step 11e's
cluster_summary.csv, no graph reload needed.

IMPORTANT -- persistence tiers are NOT a hidef_finder concept
----------------------------------------------------------------
hidef_finder itself only produces a raw persistence count (0 to
n_partitions, the number of resolution steps a cluster's structure
survived across). It has no built-in notion of "top 10% most persistent"
or similar percentile/percentage tiers -- that's an analysis choice you
make downstream, and this script exists to make that choice explicit and
comparable rather than silently picking one.

Two different denominators give two different answers to "what counts as
90%":
  --persistence-basis ceiling   (default) -- % of n_partitions, the actual
      resolution sweep size (e.g. 40). This is the more principled choice:
      it reflects what persistence literally measures ("survived this
      fraction of the sweep you ran"), and is stable across reruns even if
      the observed max happens to come in under the ceiling.
  --persistence-basis observed  -- % of the max persistence value actually
      seen in this run (e.g. 38). More relative/circular: by construction
      the most persistent cluster in any run is always "100%" under this
      basis, and the meaning shifts if you rerun with different sweep
      parameters.
This script prints the raw persistence cutoff each tier boundary
corresponds to under BOTH bases before doing anything else, specifically
so a mismatch between what you intended and what got computed is visible
immediately rather than silently baked into the plots.

DENSITY, DEFINED
-----------------
"Denser" is given a concrete definition here: density = (total edges
within the cluster's induced subgraph) / (size choose 2), i.e. the
fraction of all possible gene pairs in the cluster that are actually
connected. This is computed from columns step 11e already wrote
(n_edges_intra + n_edges_bipartite + n_edges_identity, and size) -- no
need to reload the supra-graph.

USAGE
-----
    python step11h_persistence_tier_analysis.py \
        --cluster-summary 11_hidef_results_faithful_cpm/characterized/heterogeneous_hidef_faithful_cpm_cluster_summary.csv \
        --n-partitions 40 \
        --persistence-basis ceiling \
        --tier-edges 50 75 90 100 \
        --output-dir 11_hidef_results_faithful_cpm/persistence_tiers
"""
import argparse
from pathlib import Path
from provenance import write_manifest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import kruskal
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

NAVY = "#1B263B"
SLATE = "#415A77"
AMBER = "#E0A458"
PALETTE = ["#415A77", "#778DA9", "#E0A458", "#B03A2E", "#5A6472"]


# ============================================================================
# LOAD + DERIVE
# ============================================================================

def load_and_prepare(cluster_summary_path: Path):
    df = pd.read_csv(cluster_summary_path)

    n_edges_total = (
        df.get("n_edges_intra", 0) + df.get("n_edges_bipartite", 0) + df.get("n_edges_identity", 0)
    )
    possible_pairs = df["size"] * (df["size"] - 1) / 2
    df["density"] = np.where(possible_pairs > 0, n_edges_total / possible_pairs, np.nan)
    df["n_edges_total"] = n_edges_total

    return df


def assign_tiers(df, n_partitions, basis, tier_edges):
    """tier_edges is a sorted list of percentage boundaries, e.g. [50, 75, 90, 100].
    Produces tiers: [0, e0), [e0, e1), ..., [e_{n-1}, 100] using persistence
    thresholds derived from the chosen basis."""
    basis_value = n_partitions if basis == "ceiling" else df["persistence"].max()

    edges_pct = [0] + list(tier_edges)
    thresholds = [p / 100 * basis_value for p in edges_pct]

    print(f"Persistence tier boundaries (basis={basis!r}, basis_value={basis_value:g}):")
    for lo_pct, hi_pct, lo_val, hi_val in zip(edges_pct[:-1], edges_pct[1:], thresholds[:-1], thresholds[1:]):
        print(f"  {lo_pct:>3}%-{hi_pct:>3}%  ->  persistence in [{lo_val:.1f}, {hi_val:.1f}]")

    # also show what the SAME percentage cutoffs would mean under the other basis,
    # so a silent mismatch between intended and actual basis is visible up front
    # (skipped if n_partitions wasn't given and the other basis needs it)
    if basis == "ceiling":
        other_basis, other_value = "observed-max", df["persistence"].max()
    elif n_partitions is not None:
        other_basis, other_value = "ceiling", n_partitions
    else:
        other_basis, other_value = None, None

    if other_value is not None:
        other_thresholds = [p / 100 * other_value for p in edges_pct]
        print(f"\n  (for comparison, under basis={other_basis!r} (value={other_value:g}), "
              f"the same percentages would be: "
              + ", ".join(f"{p}%={v:.1f}" for p, v in zip(edges_pct, other_thresholds)))

    labels = []
    for lo, hi in zip(edges_pct[:-1], edges_pct[1:]):
        labels.append(f"{lo}-{hi}%" if hi < 100 else f">{lo}%")

    df = df.copy()
    df["persistence_pct"] = df["persistence"] / basis_value * 100
    df["tier"] = pd.cut(df["persistence_pct"], bins=edges_pct, labels=labels,
                         include_lowest=True, right=True)
    return df, labels


def assign_tiers_percentile(df, percentiles):
    """Replicates the legacy single-layer pipeline's get_persistence_tiers()
    EXACTLY: thresholds are np.percentile() of the observed persistence
    distribution itself (e.g. "the value below which 90% of clusters fall"),
    NOT a percentage of any fixed maximum. This is a genuinely different
    definition from assign_tiers() above -- see the module docstring. Use
    this mode (--persistence-basis percentile) for direct continuity with
    prior single-layer results; --tier-percentiles controls the cutoffs
    (default [90, 75, 50], matching the legacy default)."""
    percentiles_sorted = sorted(set(percentiles), reverse=True)
    values = df["persistence"].values
        # collapse duplicate thresholds (heavy ties at max persistence) so no tier is empty
    thresholds = [float(np.percentile(values, p)) for p in percentiles_sorted]
    _pairs, _seen = [], set()
    for p, t in zip(percentiles_sorted, thresholds):
        if t in _seen:
            print(f"  ⚠ {p}th percentile threshold ({t:.2f}) equals a higher percentile's; "
                  f"dropped to avoid an empty tier")
            continue
        _seen.add(t)
        _pairs.append((p, t))
    percentiles_sorted = [p for p, _ in _pairs]
    thresholds = [t for _, t in _pairs]

    print("Persistence tier thresholds (TRUE statistical percentiles of the "
          "observed persistence distribution -- matches the legacy single-layer logic):")
    for p, t in zip(percentiles_sorted, thresholds):
        n_at_or_above = int((values >= t).sum())
        print(f"  {p}th percentile -> persistence >= {t:.2f}  "
              f"({n_at_or_above:,} clusters, {n_at_or_above/len(values):.1%})")

    def tier_for(value):
        for idx, t in enumerate(thresholds):
            if value >= t:
                lo = percentiles_sorted[idx]
                tag = f"{lo}th+" if idx == 0 else f"{lo}-{percentiles_sorted[idx-1]}th"
                return f"Tier {idx + 1} ({tag})"
        return f"Tier {len(thresholds) + 1} (<{percentiles_sorted[-1]}th)"

    df = df.copy()
    df["tier"] = df["persistence"].apply(tier_for)

    tier_labels = []
    for idx, p in enumerate(percentiles_sorted):
        tag = f"{p}th+" if idx == 0 else f"{p}-{percentiles_sorted[idx - 1]}th"
        tier_labels.append(f"Tier {idx + 1} ({tag})")
    tier_labels.append(f"Tier {len(percentiles_sorted) + 1} (<{percentiles_sorted[-1]}th)")

    # Built above highest-first (Tier 1 = 90th+, matching the legacy naming), but for
    # plotting/display we want ascending order (lowest -> highest, left -> right) --
    # same convention assign_tiers() above already uses. Reverse the LIST ORDER only;
    # the label text itself (which tier number means what) is unchanged.
    tier_labels = list(reversed(tier_labels))

    return df, tier_labels


# ============================================================================
# SUMMARY TABLE
# ============================================================================

METRICS = [
    ("size", "Cluster size (# genes)"),
    ("density", "Density (edges / possible pairs)"),
    ("n_cell_types", "# cell types spanned"),
    ("conserved_gene_frac", "Conserved-gene fraction"),
    ("cross_edge_fraction", "Cross-cell-type edge fraction"),
    ("dominant_cell_type_frac", "Dominant cell-type fraction"),
]


def print_tier_summary(df, tier_labels, run_stats=True):
    print("\n" + "=" * 70)
    print("Per-tier summary (median [IQR], n)")
    print("=" * 70)
    for col, label in METRICS:
        if col not in df.columns:
            continue
        print(f"\n{label}:")
        for tier in tier_labels:
            sub = df.loc[df["tier"] == tier, col].dropna()
            if len(sub) == 0:
                print(f"  {tier:>10}: (no clusters)")
                continue
            q1, med, q3 = sub.quantile([0.25, 0.5, 0.75])
            print(f"  {tier:>10}: median={med:.3g}  IQR=[{q1:.3g}, {q3:.3g}]  n={len(sub):,}")

        if run_stats and HAVE_SCIPY:
            groups = [df.loc[df["tier"] == t, col].dropna() for t in tier_labels]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) >= 2:
                stat, p = kruskal(*groups)
                sig = "significant (p<0.05)" if p < 0.05 else "not significant"
                print(f"  Kruskal-Wallis across tiers: H={stat:.2f}, p={p:.4g}  -- {sig}")


def save_tier_summary_csv(df, tier_labels, out_path: Path):
    rows = []
    for col, label in METRICS:
        if col not in df.columns:
            continue
        for tier in tier_labels:
            sub = df.loc[df["tier"] == tier, col].dropna()
            if len(sub) == 0:
                continue
            rows.append({
                "metric": col, "tier": tier, "n": len(sub),
                "median": sub.median(), "q1": sub.quantile(0.25), "q3": sub.quantile(0.75),
                "mean": sub.mean(), "std": sub.std(),
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


# ============================================================================
# PLOTS
# ============================================================================

def plot_tier_comparison(df, tier_labels, out_path: Path):
    present_metrics = [(c, l) for c, l in METRICS if c in df.columns]
    n = len(present_metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.6, nrows * 4.2))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (col, label) in zip(axes_flat, present_metrics):
        data = [df.loc[df["tier"] == t, col].dropna().values for t in tier_labels]
        try:
            bp = ax.boxplot(data, tick_labels=tier_labels, patch_artist=True, showfliers=False,
                             medianprops=dict(color=NAVY, linewidth=1.6))
        except TypeError:
            # older matplotlib (<3.9) doesn't have tick_labels yet
            bp = ax.boxplot(data, labels=tier_labels, patch_artist=True, showfliers=False,
                             medianprops=dict(color=NAVY, linewidth=1.6))
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        # overlay a light jitter of the actual points for a sense of n / spread
        for i, vals in enumerate(data, start=1):
            if len(vals) == 0:
                continue
            jitter = np.random.normal(i, 0.06, size=len(vals))
            ax.scatter(jitter, vals, s=6, color=NAVY, alpha=0.25, zorder=3)

        if col == "size":
            ax.set_yscale("log")
        ax.set_title(label, fontsize=11.5)
        ax.set_xticklabels(tier_labels, rotation=20, ha="right", fontsize=9.5)
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes_flat[len(present_metrics):]:
        ax.axis("off")

    fig.suptitle("Cluster metrics across persistence tiers", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_tier_sizes_bar(df, tier_labels, out_path: Path):
    counts = [int((df["tier"] == t).sum()) for t in tier_labels]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.patch.set_facecolor("white")
    bars = ax.bar(tier_labels, counts, color=PALETTE[:len(tier_labels)])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c:,}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("# clusters")
    ax.set_title("Clusters per persistence tier", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster-summary", required=True,
                     help="cluster_summary.csv from step 11e")
    ap.add_argument("--n-partitions", type=int, default=None,
                     help="The resolution-sweep ceiling (n_partitions from step 11b/11c). Required "
                          "for --persistence-basis ceiling; optional otherwise (printed for context "
                          "either way if given).")
    ap.add_argument("--persistence-basis", choices=["ceiling", "observed-max", "percentile"],
                     default="percentile",
                     help="ceiling: %% of n_partitions, the actual sweep size. "
                          "observed-max: %% of the max persistence value actually seen in this run. "
                          "percentile (default): TRUE statistical percentiles of the observed "
                          "persistence distribution -- matches the legacy single-layer pipeline's "
                          "get_persistence_tiers() exactly. See the module docstring for the "
                          "distinction between all three.")
    ap.add_argument("--tier-edges", type=float, nargs="+", default=[50, 75, 90, 100],
                     help="Used for --persistence-basis ceiling/observed-max only. Percentage "
                          "boundaries, ascending, ending at 100 (default: 50 75 90 100 "
                          "-> tiers 0-50%%, 50-75%%, 75-90%%, >90%%)")
    ap.add_argument("--tier-percentiles", type=float, nargs="+", default=[90, 75, 50],
                     help="Used for --persistence-basis percentile only. Percentile cutoffs "
                          "(default: 90 75 50, matching the legacy pipeline's default)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-stats", action="store_true", help="Skip Kruskal-Wallis significance tests")
    ap.add_argument("--exclude-lowest-tier", action="store_true",
                     help="Drop the lowest persistence tier from the summary table, plots, "
                          "significance tests, and saved CSV -- useful when that tier is mostly "
                          "noise/singletons and you only want to compare the more persistent tiers "
                          "against each other")
    args = ap.parse_args()
    write_manifest(args.output_dir, args, inputs=[args.cluster_summary])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.persistence_basis == "ceiling" and args.n_partitions is None:
        ap.error("--n-partitions is required when --persistence-basis=ceiling")

    df = load_and_prepare(Path(args.cluster_summary))
    print(f"Loaded {len(df):,} clusters from {args.cluster_summary}")
    ceiling_note = f" (ceiling given: {args.n_partitions})" if args.n_partitions else ""
    print(f"Observed persistence range: {df['persistence'].min():.0f}-{df['persistence'].max():.0f}"
          f"{ceiling_note}\n")

    df, tier_labels = (
        assign_tiers_percentile(df, args.tier_percentiles)
        if args.persistence_basis == "percentile"
        else assign_tiers(df, args.n_partitions, args.persistence_basis, sorted(args.tier_edges))
    )

    if args.exclude_lowest_tier:
        lowest = tier_labels[0]  # tier_labels is always ascending low->high now
        n_dropped = int((df["tier"] == lowest).sum())
        df = df[df["tier"] != lowest].copy()
        tier_labels = tier_labels[1:]
        print(f"\n--exclude-lowest-tier: dropped {n_dropped:,} clusters in tier {lowest!r}; "
              f"{len(df):,} clusters remain across {len(tier_labels)} tier(s)")

    if not HAVE_SCIPY and not args.no_stats:
        print("\n(scipy not available -- skipping Kruskal-Wallis significance tests; "
              "pip install scipy to enable)")

    print_tier_summary(df, tier_labels, run_stats=not args.no_stats and HAVE_SCIPY)
    save_tier_summary_csv(df, tier_labels, out_dir / "persistence_tier_summary.csv")

    plot_tier_comparison(df, tier_labels, out_dir / "persistence_tier_comparison.png")
    plot_tier_sizes_bar(df, tier_labels, out_dir / "persistence_tier_counts.png")

    df.to_csv(out_dir / "clusters_with_tiers.csv", index=False)
    print(f"Saved -> {out_dir / 'clusters_with_tiers.csv'} (full per-cluster data with tier labels)")


if __name__ == "__main__":
    main()