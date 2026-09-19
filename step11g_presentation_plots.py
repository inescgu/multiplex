#!/usr/bin/env python3
"""
Step 11g: Regenerate the two presentation figures used in the methods deck
-- results_summary.png (module size / persistence range) and
resolution_selection.png (resolution-parameter scan) -- from REAL pipeline
output files, instead of the hardcoded numbers used to mock them up.

results_summary.png
--------------------
Reads a step 11c Phase 1 consensus pickle (cluG_collapsed_w_len: consensus
cluster membership arrays + their real hidef_finder persistence values,
plus n_partitions = the persistence ceiling) and draws an honest min/mean/
max range plot for module size and persistence. Deliberately does NOT draw
a fabricated distribution shape -- only min/mean/max are computed, so
that's all that's plotted. (If you want the real distribution shape, that's
what step 11e's summary_plots.png histogram is for -- this figure is the
presentation-friendly companion, not a replacement.)

resolution_selection.png
-------------------------
Reads one or two CSVs written by step11b_resolution_profile.py (a wide
scan and, optionally, a fine scan covering a narrower range in more
detail) and plots n_communities vs. gamma on log-log axes, with the
production resolution range shaded and any fully-degenerate points
(n_communities == total node count, i.e. every node its own singleton)
annotated automatically if --total-nodes is given.

USAGE
-----
    # Both figures in one run:
    python step11g_presentation_plots.py \
        --consensus 11_hidef_results_faithful_cpm/consensus.pkl \
        --wide-scan-csv 11_hidef_hetero/resolution_profile.csv \
        --fine-scan-csv 11_hidef_hetero/resolution_profile_cpm_finescan.csv \
        --partition-type CPM \
        --production-min-resolution 1e-5 \
        --production-max-resolution 0.9 \
        --total-nodes 140139 \
        --output-dir presentation_figures

    # Just one figure -- omit the inputs you don't have:
    python step11g_presentation_plots.py \
        --consensus 11_hidef_results_faithful_cpm/consensus.pkl \
        --output-dir presentation_figures
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#1B263B"
SLATE = "#415A77"
AMBER = "#E0A458"
AMBER_DARK = "#8a5a1f"
RED = "#B03A2E"


# ============================================================================
# results_summary.png -- from the real consensus pickle
# ============================================================================

def load_consensus_stats(consensus_path: Path):
    with open(consensus_path, "rb") as f:
        data = pickle.load(f)
    cluG_collapsed_w_len = data["cluG_collapsed_w_len"]
    n_partitions = data["n_partitions"]

    sizes = [int(np.sum(c[0])) for c in cluG_collapsed_w_len]
    persistences = [c[1] for c in cluG_collapsed_w_len]

    if not sizes:
        raise SystemExit(f"ERROR: {consensus_path} contains zero consensus clusters -- "
                          "nothing to plot.")

    return {
        "n_clusters": len(sizes),
        "size_min": min(sizes), "size_mean": float(np.mean(sizes)), "size_max": max(sizes),
        "pers_min": min(persistences), "pers_mean": float(np.mean(persistences)),
        "pers_max": max(persistences), "pers_ceiling": n_partitions,
        "n_distinct_persistence": len(set(persistences)),
    }


def plot_results_summary(stats: dict, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.6))
    fig.patch.set_facecolor("white")

    def range_panel(ax, vmin, vmean, vmax, title, ceiling=None, ceiling_label=None,
                     log=False, int_fmt=True):
        if log:
            ax.set_xscale("log")
        ax.hlines(0, vmin, vmax, color=SLATE, linewidth=3, alpha=0.6)
        if ceiling is not None:
            ax.axvline(ceiling, color=RED, linestyle=":", linewidth=1.3, alpha=0.8)
            ax.text(ceiling, 0.75, ceiling_label or f"max possible\n({ceiling})",
                    ha="center", va="bottom", fontsize=8.5, color=RED)
        ax.plot([vmin], [0], marker="|", markersize=18, color=NAVY, mew=2.5)
        ax.plot([vmax], [0], marker="|", markersize=18, color=NAVY, mew=2.5)
        ax.plot([vmean], [0], marker="o", markersize=11, color=AMBER, zorder=5,
                markeredgecolor=NAVY, markeredgewidth=1.2)
        fmt = (lambda v: f"{v:,.0f}") if int_fmt else (lambda v: f"{v:.1f}")
        ax.text(vmin, 0.35, f"min = {fmt(vmin)}", ha="center", fontsize=10, color=NAVY)
        max_ha = "right" if (ceiling is not None and vmax / ceiling > 0.85) else "center"
        ax.text(vmax, 0.35, f"max = {fmt(vmax)}", ha=max_ha, fontsize=10, color=NAVY)
        ax.text(vmean, -0.55, f"mean = {vmean:.1f}", ha="center", fontsize=10.5,
                color=AMBER_DARK, fontweight="bold")
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.set_title(title, fontsize=12, loc="left")
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    range_panel(
        axes[0], stats["size_min"], stats["size_mean"], stats["size_max"],
        title=f"Module size (# genes, log scale) \u2014 n = {stats['n_clusters']:,} consensus modules",
        log=True,
    )
    axes[0].set_xlim(max(1, stats["size_min"] * 0.5), stats["size_max"] * 2)

    range_panel(
        axes[1], stats["pers_min"], stats["pers_mean"], stats["pers_max"],
        title=(f"Persistence (# resolution steps survived) \u2014 "
               f"{stats['n_distinct_persistence']} of {stats['pers_ceiling']} "
               f"possible values observed"),
        ceiling=stats["pers_ceiling"],
    )
    axes[1].set_xlim(0, stats["pers_ceiling"] * 1.1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")
    print(f"  size:        min={stats['size_min']}  mean={stats['size_mean']:.1f}  "
          f"max={stats['size_max']:,}  (n={stats['n_clusters']:,})")
    print(f"  persistence: min={stats['pers_min']}  mean={stats['pers_mean']:.1f}  "
          f"max={stats['pers_max']}  ({stats['n_distinct_persistence']}/{stats['pers_ceiling']} "
          f"distinct values)")


# ============================================================================
# resolution_selection.png -- from step11b_resolution_profile.py's CSV output
# ============================================================================

def load_scan_csv(path: Path, partition_type: str):
    if not path.exists():
        raise SystemExit(f"ERROR: {path} does not exist. If you don't have this scan, "
                          f"omit the corresponding --wide-scan-csv/--fine-scan-csv flag "
                          f"entirely rather than pointing it at a nonexistent path.")
    df = pd.read_csv(path)
    if "partition_type" in df.columns:
        df = df[df["partition_type"] == partition_type]
    if df.empty:
        raise SystemExit(f"ERROR: no rows for partition_type={partition_type!r} in {path}")
    return df.sort_values("gamma")


def plot_resolution_selection(wide_df, fine_df, out_path: Path,
                               production_min=None, production_max=None,
                               total_nodes=None):
    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    fig.patch.set_facecolor("white")

    have_both = wide_df is not None and fine_df is not None

    if wide_df is not None:
        if have_both:
            # Fine scan will be overlaid on top -- keep the wide scan light
            # so the fine-grained detail reads as the "foreground" series.
            ax.plot(wide_df["gamma"], wide_df["n_communities"], marker="o", markersize=4,
                    linewidth=1.4, color=NAVY, alpha=0.35, label="Exploratory scan")
        else:
            # Only scan we have -- draw it as the primary series, full contrast,
            # not the faded background style (that style looks washed-out/broken
            # on its own since it was designed to sit under a bolder overlay).
            ax.plot(wide_df["gamma"], wide_df["n_communities"], marker="o", markersize=5,
                    linewidth=2.2, color=NAVY, label="Resolution scan")
    if fine_df is not None:
        ax.plot(fine_df["gamma"], fine_df["n_communities"], marker="o", markersize=5,
                linewidth=2.2, color=NAVY, label="Fine-grained scan")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Resolution parameter (log scale)", fontsize=12.5)
    ax.set_ylabel("# communities detected (log scale)", fontsize=12.5)

    if production_min is not None and production_max is not None:
        ax.axvspan(production_min, production_max, color=NAVY, alpha=0.06, zorder=0)

    if total_nodes is not None:
        all_df = pd.concat([d for d in [wide_df, fine_df] if d is not None])
        collapsed = all_df[all_df["n_communities"] >= total_nodes]
        if not collapsed.empty:
            first_collapse = collapsed.sort_values("gamma").iloc[0]
            ax.annotate(
                "degenerate regime\n(complete fragmentation)",
                xy=(first_collapse["gamma"], first_collapse["n_communities"]),
                xytext=(first_collapse["gamma"] * 2.2, first_collapse["n_communities"] * 0.45),
                fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
            )

    ax.legend(loc="lower right", fontsize=10.5, frameon=True)
    ax.grid(True, which="both", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if production_min is not None and production_max is not None:
        y_top = ax.get_ylim()[1]
        ax.text(production_min * 2, y_top * 0.82, "range used for the production analysis",
                fontsize=10.5, color=NAVY, ha="left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consensus", default=None,
                     help="Pickle from step 11c Phase 1 -- if given, writes results_summary.png")
    ap.add_argument("--wide-scan-csv", default=None,
                     help="CSV from step11b_resolution_profile.py (wide scan)")
    ap.add_argument("--fine-scan-csv", default=None,
                     help="CSV from step11b_resolution_profile.py (fine scan) -- optional, "
                          "adds a denser overlay in a narrower range")
    ap.add_argument("--partition-type", default="CPM",
                     help="Which partition_type to plot from the scan CSVs (default CPM). "
                          "Only matters if a CSV contains multiple partition types.")
    ap.add_argument("--production-min-resolution", type=float, default=None,
                     help="Shades this as the range actually used for the production sweep")
    ap.add_argument("--production-max-resolution", type=float, default=None)
    ap.add_argument("--total-nodes", type=int, default=None,
                     help="If given, auto-annotates the first scan point where n_communities "
                          "reaches this value as full singleton collapse")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.consensus:
        stats = load_consensus_stats(Path(args.consensus))
        plot_results_summary(stats, out_dir / "results_summary.png")
    else:
        print("Skipping results_summary.png (no --consensus given)")

    if args.wide_scan_csv or args.fine_scan_csv:
        wide_df = load_scan_csv(Path(args.wide_scan_csv), args.partition_type) if args.wide_scan_csv else None
        fine_df = load_scan_csv(Path(args.fine_scan_csv), args.partition_type) if args.fine_scan_csv else None
        plot_resolution_selection(
            wide_df, fine_df, out_dir / "resolution_selection.png",
            production_min=args.production_min_resolution,
            production_max=args.production_max_resolution,
            total_nodes=args.total_nodes,
        )
    else:
        print("Skipping resolution_selection.png (no --wide-scan-csv/--fine-scan-csv given)")


if __name__ == "__main__":
    main()