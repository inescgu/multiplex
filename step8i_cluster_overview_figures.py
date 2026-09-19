"""
step8i_cluster_overview_figures.py
------------------------------------------------------------------------------------
Generates the "basic cluster results" figure for the new overview slide: per-cluster
size, percent seed genes, and structural role composition (hub / bridge / peripheral),
for both with_ppi and no_ppi conditions side by side.

Reads directly from files step8 already produces:
  <dir>/spectral_clusters.csv   -- needs columns: cluster, gene, role
  <seed-genes-csv>              -- needs a 'gene' column

Nothing here recomputes any analysis -- it's a straight visualization of existing
per-node output.

Usage:
  python step8i_cluster_overview_figures.py \
      --with-ppi-dir 08_spectral_analysis/module_with_ppi \
      --no-ppi-dir 08_spectral_analysis/module_no_ppi \
      --seed-genes-csv 00_inputs/seed_genes.csv \
      --out-dir 08_spectral_analysis/slide_figures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_WITH_PPI = "#7f8c8d"
COLOR_NO_PPI = "#27ae60"
COLOR_SEED_LINE = "#c0392b"
ROLE_COLORS = {
    "multiplex_hub": "#8e44ad",
    "intra_hub": "#2980b9",
    "bridge": "#f39c12",
    "peripheral": "#bdc3c7",
}
ROLE_ORDER = ["multiplex_hub", "intra_hub", "bridge", "peripheral"]

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_cluster_stats(clusters_csv: Path, seed_set: set):
    df = pd.read_csv(clusters_csv)
    df["is_seed"] = df["gene"].astype(str).str.strip().str.upper().isin(seed_set)

    total_n = len(df)
    overall_seed_frac = df["is_seed"].mean()

    rows = []
    for cid, grp in df.groupby("cluster"):
        row = {
            "cluster": cid,
            "size": len(grp),
            "pct_of_module": 100 * len(grp) / total_n,
            "pct_seed": 100 * grp["is_seed"].mean(),
        }
        if "role" in df.columns:
            role_counts = grp["role"].value_counts(normalize=True)
            for role in ROLE_ORDER:
                row[f"role_{role}"] = 100 * role_counts.get(role, 0.0)
        rows.append(row)

    stats_df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    return stats_df, overall_seed_frac


def make_cluster_overview_figure(with_ppi_dir: Path, no_ppi_dir: Path, seed_genes_csv: Path,
                                  out_dir: Path):
    seed_df = pd.read_csv(seed_genes_csv)
    gene_col = "gene" if "gene" in seed_df.columns else seed_df.columns[0]
    seed_set = set(seed_df[gene_col].astype(str).str.strip().str.upper())

    conditions = []
    for label, d, color in [("with_ppi", with_ppi_dir, COLOR_WITH_PPI),
                             ("no_ppi", no_ppi_dir, COLOR_NO_PPI)]:
        clusters_csv = d / "spectral_clusters.csv"
        if not clusters_csv.exists():
            print(f"  [skip] {clusters_csv} not found, skipping {label}")
            continue
        stats_df, overall_seed_frac = load_cluster_stats(clusters_csv, seed_set)
        conditions.append((label, color, stats_df, overall_seed_frac))

    if not conditions:
        print("  [skip] no spectral_clusters.csv files found, skipping cluster overview figure")
        return

    has_roles = all("role_multiplex_hub" in c[2].columns for c in conditions)
    n_panels = 3 if has_roles else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: cluster sizes ---
    ax = axes[0]
    offset = 0
    xticks, xlabels = [], []
    bar_width = 0.7
    group_gap = 1.5
    x_cursor = 0
    for label, color, stats_df, _ in conditions:
        xs = x_cursor + np.arange(len(stats_df))
        bars = ax.bar(xs, stats_df["size"], color=color, alpha=0.85, width=bar_width)
        for x, size, pct in zip(xs, stats_df["size"], stats_df["pct_of_module"]):
            ax.text(x, size, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)
        xticks.extend(xs)
        xlabels.extend([f"C{c}" for c in stats_df["cluster"]])
        x_cursor += len(stats_df) + group_gap
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("cluster size (# genes)")
    ax.set_title("Cluster sizes\n(% = share of module)")
    # condition labels beneath groups
    x_cursor = 0
    for label, color, stats_df, _ in conditions:
        mid = x_cursor + (len(stats_df) - 1) / 2
        ax.text(mid, -0.12, label, transform=ax.get_xaxis_transform(),
               ha="center", va="top", fontsize=10, color=color, fontweight="bold")
        x_cursor += len(stats_df) + group_gap

    # --- Panel 2: % seed genes per cluster ---
    ax = axes[1]
    x_cursor = 0
    xticks, xlabels = [], []
    for label, color, stats_df, overall_seed_frac in conditions:
        xs = x_cursor + np.arange(len(stats_df))
        ax.bar(xs, stats_df["pct_seed"], color=color, alpha=0.85, width=bar_width)
        ax.hlines(100 * overall_seed_frac, xs.min() - 0.5, xs.max() + 0.5,
                 color=COLOR_SEED_LINE, linestyle="--", linewidth=1.3)
        xticks.extend(xs)
        xlabels.extend([f"C{c}" for c in stats_df["cluster"]])
        x_cursor += len(stats_df) + group_gap
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("% seed genes in cluster")
    ax.set_title("Seed gene share per cluster\n(dashed = module-wide average)")
    x_cursor = 0
    for label, color, stats_df, _ in conditions:
        mid = x_cursor + (len(stats_df) - 1) / 2
        ax.text(mid, -0.12, label, transform=ax.get_xaxis_transform(),
               ha="center", va="top", fontsize=10, color=color, fontweight="bold")
        x_cursor += len(stats_df) + group_gap

    # --- Panel 3: role composition (stacked bar) ---
    if has_roles:
        ax = axes[2]
        x_cursor = 0
        xticks, xlabels = [], []
        for label, color, stats_df, _ in conditions:
            xs = x_cursor + np.arange(len(stats_df))
            bottom = np.zeros(len(stats_df))
            for role in ROLE_ORDER:
                vals = stats_df[f"role_{role}"].values
                ax.bar(xs, vals, bottom=bottom, color=ROLE_COLORS[role], width=bar_width,
                      label=role if x_cursor == 0 else None)
                bottom += vals
            xticks.extend(xs)
            xlabels.extend([f"C{c}" for c in stats_df["cluster"]])
            x_cursor += len(stats_df) + group_gap
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel("% of cluster")
        ax.set_title("Structural role composition per cluster")
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        x_cursor = 0
        for label, color, stats_df, _ in conditions:
            mid = x_cursor + (len(stats_df) - 1) / 2
            ax.text(mid, -0.12, label, transform=ax.get_xaxis_transform(),
                   ha="center", va="top", fontsize=10, color=color, fontweight="bold")
            x_cursor += len(stats_df) + group_gap

    fig.tight_layout()
    out_path = out_dir / "slide_cluster_overview.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")

    # Also dump the underlying numbers to CSV so the exact figures are copy-pasteable for the script/notes
    all_stats = []
    for label, _, stats_df, overall_seed_frac in conditions:
        sd = stats_df.copy()
        sd.insert(0, "condition", label)
        sd["module_seed_frac_pct"] = 100 * overall_seed_frac
        all_stats.append(sd)
    combined = pd.concat(all_stats, ignore_index=True)
    csv_path = out_dir / "cluster_overview_stats.csv"
    combined.to_csv(csv_path, index=False)
    print(f"  saved -> {csv_path}  (exact numbers behind the figure, for your speaker notes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-ppi-dir", required=True)
    ap.add_argument("--no-ppi-dir", required=True)
    ap.add_argument("--seed-genes-csv", required=True)
    ap.add_argument("--out-dir", default="08_spectral_analysis/slide_figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    make_cluster_overview_figure(Path(args.with_ppi_dir), Path(args.no_ppi_dir),
                                  Path(args.seed_genes_csv), out_dir)


if __name__ == "__main__":
    main()