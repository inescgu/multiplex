#!/usr/bin/env python3
"""
multiplex_layer_topology_analysis.py
======================================
Computes per-layer topological metrics for a MultiXrank multiplex:
  - Node count, Edge count
  - Density
  - Average clustering coefficient
  - Largest Connected Component (LCC) size (absolute + fraction)
  - Top-k eigenvalue centrality nodes
  - Degree distribution summary (mean, median, max)

Also aggregates metrics by 'category' (COL_AI) from layers_categories.csv.

Usage
-----
python multiplex_layer_topology_analysis.py \
    --wdir /path/to/multixrank_wdir \
    --csv  /path/to/layers_categories.csv \
    [--output-dir topology_output] \
    [--exclude-ai "immune tissue"] \
    [--top-n-eigen 10] \
    [--min-edges 1]
"""

import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

DIVIDER    = "=" * 72
SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Column name constants  (must match layers_categories.csv)
# ---------------------------------------------------------------------------
COL_CELL_TYPE = "Cell Type"
COL_BROAD     = "Broad Category - ai"
COL_AI        = "category"

# Layer stems that are NOT cell-type layers
SPECIAL_STEMS = {"ppi_layer", "ligand_receptor_layer", "pathway_layer"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_block(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def load_categories(
    csv_path: Path,
    exclude_ai: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0].strip() in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]
    required = {COL_CELL_TYPE, COL_BROAD, COL_AI}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"layers_categories.csv missing columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    before = len(df)
    if exclude_ai:
        mask = df[COL_AI].str.lower().isin([e.lower() for e in exclude_ai])
        df   = df[~mask]
        print(f"Excluded {before - len(df)} rows by '{COL_AI}' ({exclude_ai})")
    print(f"Retained {len(df)} cell types")
    return df.reset_index(drop=True)


def stem_from_cell_type(cell_type: str) -> str:
    return cell_type.replace(" ", "_") + "_layer"


def cell_type_from_stem(stem: str) -> str:
    """Reverse of stem_from_cell_type."""
    return stem.replace("_layer", "").replace("_", " ")


def load_graph_from_tsv(path: Path) -> nx.Graph:
    """Load a two-column TSV edge list into an undirected NetworkX graph."""
    G = nx.Graph()
    if not path.exists():
        return G
    try:
        df = pd.read_csv(
            path, sep="\t", header=None,
            names=["g1", "g2"], dtype=str,
            on_bad_lines="skip",
        )
    except Exception:
        return G
    for _, row in df.iterrows():
        a = str(row["g1"]).strip().upper()
        b = str(row["g2"]).strip().upper()
        if a and b and a != "NAN" and b != "NAN" and a != b:
            G.add_edge(a, b)
    return G


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------
def compute_layer_metrics(
    G: nx.Graph,
    layer_name: str,
    top_n_eigen: int = 10,
) -> dict:
    """
    Compute topological metrics for a single layer graph.

    Returns a flat dict of scalar metrics plus a sub-dict for
    per-node eigen-centrality (top_n only).
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {
            "layer":             layer_name,
            "n_nodes":           0,
            "n_edges":           0,
            "density":           0.0,
            "avg_clustering":    0.0,
            "lcc_size":          0,
            "lcc_fraction":      0.0,
            "degree_mean":       0.0,
            "degree_median":     0.0,
            "degree_max":        0,
            "top_eigen_nodes":   [],
            "top_eigen_scores":  [],
            "is_connected":      False,
            "n_components":      0,
        }

    # Basic
    density = nx.density(G)

    # Clustering (can be slow for large graphs; use approximation if needed)
    avg_clust = nx.average_clustering(G)

    # LCC
    components  = list(nx.connected_components(G))
    lcc_size    = max(len(c) for c in components)
    lcc_frac    = lcc_size / n_nodes
    n_comp      = len(components)
    is_conn     = (n_comp == 1)

    # Degree stats
    degrees     = [d for _, d in G.degree()]
    deg_mean    = float(np.mean(degrees))
    deg_median  = float(np.median(degrees))
    deg_max     = int(np.max(degrees))

    # Eigenvalue centrality — only on LCC to guarantee convergence
    lcc_nodes   = max(components, key=len)
    G_lcc       = G.subgraph(lcc_nodes).copy()
    try:
        eigen_cent  = nx.eigenvector_centrality_numpy(G_lcc, weight=None)
    except Exception:
        # Fallback: degree centrality
        eigen_cent  = nx.degree_centrality(G_lcc)

    top_pairs   = sorted(eigen_cent.items(), key=lambda x: x[1], reverse=True)[:top_n_eigen]
    top_nodes   = [p[0] for p in top_pairs]
    top_scores  = [round(p[1], 6) for p in top_pairs]

    return {
        "layer":            layer_name,
        "n_nodes":          n_nodes,
        "n_edges":          n_edges,
        "density":          round(density,       6),
        "avg_clustering":   round(avg_clust,     6),
        "lcc_size":         lcc_size,
        "lcc_fraction":     round(lcc_frac,      6),
        "degree_mean":      round(deg_mean,      4),
        "degree_median":    round(deg_median,    4),
        "degree_max":       deg_max,
        "top_eigen_nodes":  top_nodes,
        "top_eigen_scores": top_scores,
        "is_connected":     is_conn,
        "n_components":     n_comp,
    }


# ---------------------------------------------------------------------------
# Build full metrics table
# ---------------------------------------------------------------------------
def analyse_all_layers(
    multiplex_dir: Path,
    categories_df: pd.DataFrame,
    extra_stems: list[str],
    top_n_eigen: int,
    min_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterate over every layer TSV, compute metrics, and return:
      - metrics_df  : one row per layer (flat scalar metrics)
      - eigen_df    : one row per (layer, gene) for top-eigen nodes
    """
    _print_block("Computing per-layer topology metrics")

    # Build stem → category / broad mappings from the CSV
    ct_to_ai    = dict(zip(categories_df[COL_CELL_TYPE], categories_df[COL_AI]))
    ct_to_broad = dict(zip(categories_df[COL_CELL_TYPE], categories_df[COL_BROAD]))

    def _meta(stem: str) -> tuple[str, str, str]:
        """Return (cell_type_label, category, broad) for a stem."""
        if stem in SPECIAL_STEMS:
            return stem, "special", "special"
        ct = cell_type_from_stem(stem)
        return ct, ct_to_ai.get(ct, "unknown"), ct_to_broad.get(ct, "unknown")

    # Collect all TSV paths to analyse
    allowed_stems = {stem_from_cell_type(ct) for ct in categories_df[COL_CELL_TYPE]}
    stems_to_run  = (allowed_stems | set(extra_stems))

    all_tsv = sorted(multiplex_dir.glob("*.tsv"))
    print(f"  TSV files in multiplex dir : {len(all_tsv)}")

    metrics_rows = []
    eigen_rows   = []
    skipped      = 0

    for tsv_path in all_tsv:
        stem = tsv_path.stem
        if stem not in stems_to_run:
            continue

        G = load_graph_from_tsv(tsv_path)
        if G.number_of_edges() < min_edges:
            skipped += 1
            continue

        ct_label, category, broad = _meta(stem)
        m = compute_layer_metrics(G, stem, top_n_eigen=top_n_eigen)

        # Augment with categorical metadata
        m["cell_type"] = ct_label
        m["category"]  = category
        m["broad"]     = broad

        metrics_rows.append({k: v for k, v in m.items()
                              if k not in ("top_eigen_nodes", "top_eigen_scores")})

        # Eigen rows
        for gene, score in zip(m["top_eigen_nodes"], m["top_eigen_scores"]):
            eigen_rows.append({
                "layer":     stem,
                "cell_type": ct_label,
                "category":  category,
                "broad":     broad,
                "gene":      gene,
                "eigen_score": score,
                "eigen_rank":  m["top_eigen_nodes"].index(gene) + 1,
            })

        print(
            f"  {stem:<55} "
            f"N={m['n_nodes']:>6,}  E={m['n_edges']:>7,}  "
            f"dens={m['density']:.5f}  "
            f"clust={m['avg_clustering']:.4f}  "
            f"LCC={m['lcc_fraction']:.3f}"
        )

    print(f"\n  Layers analysed : {len(metrics_rows)}")
    print(f"  Layers skipped  : {skipped}  (< {min_edges} edges)")

    metrics_df = pd.DataFrame(metrics_rows)
    eigen_df   = pd.DataFrame(eigen_rows)
    return metrics_df, eigen_df


# ---------------------------------------------------------------------------
# Category-level aggregation
# ---------------------------------------------------------------------------
def aggregate_by_category(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise scalar metrics at the 'category' level.
    Excludes 'special' layers from the cell-type grouping.
    """
    _print_block("Aggregating metrics by category")

    cell_type_df = metrics_df[metrics_df["category"] != "special"].copy()

    agg = (
        cell_type_df
        .groupby("category")
        .agg(
            n_layers        =("layer",          "count"),
            total_nodes     =("n_nodes",         "sum"),
            total_edges     =("n_edges",         "sum"),
            mean_nodes      =("n_nodes",         "mean"),
            mean_edges      =("n_edges",         "mean"),
            mean_density    =("density",         "mean"),
            mean_clustering =("avg_clustering",  "mean"),
            mean_lcc_frac   =("lcc_fraction",    "mean"),
            mean_deg_mean   =("degree_mean",     "mean"),
            mean_deg_max    =("degree_max",      "mean"),
            pct_connected   =("is_connected",
                               lambda x: 100 * x.sum() / len(x)),
        )
        .reset_index()
        .sort_values("total_edges", ascending=False)
    )

    for col in ("mean_nodes", "mean_edges", "mean_density",
                "mean_clustering", "mean_lcc_frac",
                "mean_deg_mean", "mean_deg_max"):
        agg[col] = agg[col].round(4)

    print(agg.to_string(index=False))
    return agg


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
METRIC_LABELS = {
    "n_nodes":        "Node count",
    "n_edges":        "Edge count",
    "density":        "Density",
    "avg_clustering": "Avg clustering coeff.",
    "lcc_fraction":   "LCC fraction",
    "degree_mean":    "Mean degree",
    "degree_max":     "Max degree",
}


def _category_palette(categories: list[str]) -> dict[str, str]:
    palette = plt.cm.tab20.colors
    return {cat: palette[i % 20] for i, cat in enumerate(sorted(set(categories)))}


# ---------------------------------------------------------------------------
# Plot 1 — per-layer bar plots for each metric
# ---------------------------------------------------------------------------
def plot_per_layer_metrics(
    metrics_df: pd.DataFrame,
    out: Path,
    max_layers: int = 80,
) -> None:
    _print_block("Plotting per-layer metric bar charts")

    cell_df = metrics_df[metrics_df["category"] != "special"].copy()
    # Sort by n_edges descending; take top max_layers
    cell_df = cell_df.sort_values("n_edges", ascending=False).head(max_layers)

    palette  = _category_palette(cell_df["category"].tolist())
    colors   = [palette[c] for c in cell_df["category"]]
    y_labels = cell_df["cell_type"].tolist()

    metrics_to_plot = list(METRIC_LABELS.keys())
    n_m = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_m, figsize=(n_m * 3.8, max(6, len(cell_df) * 0.22)))

    for ax, metric in zip(axes, metrics_to_plot):
        vals = cell_df[metric].tolist()
        ax.barh(y_labels, vals, color=colors, edgecolor="white", linewidth=0.3)
        ax.invert_yaxis()
        ax.set_xlabel(METRIC_LABELS[metric], fontsize=7)
        ax.tick_params(axis="y", labelsize=5)
        ax.tick_params(axis="x", labelsize=6)
        ax.set_title(METRIC_LABELS[metric], fontsize=7, fontweight="bold")

    # Legend for categories
    patches = [
        mpatches.Patch(color=c, label=cat)
        for cat, c in sorted(palette.items())
    ]
    axes[-1].legend(
        handles=patches, loc="lower right",
        fontsize=5, ncol=1, framealpha=0.8,
        title="category", title_fontsize=5,
    )

    fig.suptitle(
        f"Per-layer topology metrics  (top {max_layers} by edge count)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"bar_per_layer_metrics.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — heatmap of metrics × layers
# ---------------------------------------------------------------------------
def plot_metrics_heatmap(
    metrics_df: pd.DataFrame,
    out: Path,
    max_layers: int = 60,
) -> None:
    _print_block("Plotting metrics heatmap (layers × metrics)")

    cell_df = (
        metrics_df[metrics_df["category"] != "special"]
        .sort_values("n_edges", ascending=False)
        .head(max_layers)
        .copy()
    )

    scalar_cols = list(METRIC_LABELS.keys())
    mat = cell_df.set_index("cell_type")[scalar_cols].copy().astype(float)

    # Z-score normalise each column for visual comparability
    mat_norm = (mat - mat.mean()) / (mat.std() + 1e-12)

    fig_h = max(8, len(mat_norm) * 0.25)
    fig_w = max(10, len(scalar_cols) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        mat_norm,
        ax=ax,
        cmap="RdYlBu_r",
        center=0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Z-score", "shrink": 0.6},
        xticklabels=[METRIC_LABELS[c] for c in scalar_cols],
        yticklabels=True,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5)
    ax.set_title(
        f"Layer topology metrics (Z-scored, top {max_layers})",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"heatmap_layer_metrics.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3 — category-level box/violin plots
# ---------------------------------------------------------------------------
def plot_category_distributions(
    metrics_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Plotting category-level metric distributions")

    cell_df  = metrics_df[metrics_df["category"] != "special"].copy()
    palette  = _category_palette(cell_df["category"].tolist())
    order    = (
        cell_df.groupby("category")["n_edges"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    metrics_to_plot = [
        ("n_nodes",        "Node count"),
        ("n_edges",        "Edge count"),
        ("density",        "Density"),
        ("avg_clustering", "Avg clustering"),
        ("lcc_fraction",   "LCC fraction"),
        ("degree_mean",    "Mean degree"),
    ]

    n_m  = len(metrics_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics_to_plot):
        sns.violinplot(
            data=cell_df,
            x="category",
            y=metric,
            order=order,
            palette=palette,
            ax=ax,
            inner="box",
            cut=0,
            linewidth=0.8,
        )
        sns.stripplot(
            data=cell_df,
            x="category",
            y=metric,
            order=order,
            color="black",
            size=2.5,
            alpha=0.5,
            ax=ax,
            jitter=True,
        )
        ax.set_xlabel("Category", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=40, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for j in range(n_m, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Topology metric distributions per category",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"violin_category_metrics.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4 — category summary bar chart (mean metrics)
# ---------------------------------------------------------------------------
def plot_category_summary_bars(
    cat_agg_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Plotting category summary bar charts")

    palette = _category_palette(cat_agg_df["category"].tolist())
    colors  = [palette[c] for c in cat_agg_df["category"]]

    metrics_to_plot = [
        ("mean_nodes",      "Mean node count"),
        ("mean_edges",      "Mean edge count"),
        ("mean_density",    "Mean density"),
        ("mean_clustering", "Mean clustering"),
        ("mean_lcc_frac",   "Mean LCC fraction"),
        ("pct_connected",   "% fully connected layers"),
    ]

    n_m  = len(metrics_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, metrics_to_plot):
        sorted_df = cat_agg_df.sort_values(col, ascending=False)
        ax.bar(
            sorted_df["category"], sorted_df[col],
            color=[palette[c] for c in sorted_df["category"]],
            edgecolor="white", linewidth=0.5,
        )
        ax.set_xlabel("Category", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        # Annotate bars with value
        for bar, val in zip(ax.patches, sorted_df[col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bar.get_height() * 0.02,
                f"{val:.3g}",
                ha="center", va="bottom", fontsize=5.5,
            )

    for j in range(n_m, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Category-level mean topology metrics",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"bar_category_summary_metrics.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5 — scatter: density vs clustering, coloured by category
# ---------------------------------------------------------------------------
def plot_scatter_density_clustering(
    metrics_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Plotting scatter: density vs clustering (per layer)")

    cell_df = metrics_df[metrics_df["category"] != "special"].copy()
    palette = _category_palette(cell_df["category"].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    scatter_pairs = [
        ("density",        "avg_clustering", "Density",     "Avg clustering"),
        ("n_nodes",        "n_edges",        "Node count",  "Edge count"),
        ("lcc_fraction",   "avg_clustering", "LCC fraction","Avg clustering"),
    ]

    for ax, (xc, yc, xl, yl) in zip(axes, scatter_pairs):
        for cat, grp in cell_df.groupby("category"):
            ax.scatter(
                grp[xc], grp[yc],
                label=cat,
                color=palette[cat],
                s=35, alpha=0.75, edgecolors="white", linewidths=0.4,
            )
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(f"{xl} vs {yl}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5, fontsize=7,
        bbox_to_anchor=(0.5, -0.04), framealpha=0.9,
        title="category", title_fontsize=7,
    )
    fig.suptitle(
        "Layer topology scatter plots (coloured by category)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"scatter_layer_topology.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6 — top eigen-centrality genes per category (bubble chart)
# ---------------------------------------------------------------------------
def plot_top_eigen_per_category(
    eigen_df: pd.DataFrame,
    out: Path,
    top_n_genes: int = 15,
) -> None:
    _print_block("Plotting top eigen-centrality genes per category")

    if eigen_df.empty:
        print("  ⚠  No eigen data, skipping")
        return

    cell_eigen = eigen_df[eigen_df["category"] != "special"].copy()

    # For each category, average eigen score across layers per gene
    cat_gene_agg = (
        cell_eigen
        .groupby(["category", "gene"])
        .agg(
            mean_eigen=("eigen_score", "mean"),
            n_layers=("layer",        "count"),
        )
        .reset_index()
    )

    categories = sorted(cat_gene_agg["category"].unique())
    palette    = _category_palette(categories)

    n_cols = min(4, len(categories))
    n_rows = int(np.ceil(len(categories) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, n_rows * 4.5),
    )
    axes_flat = np.array(axes).flatten()

    for i, cat in enumerate(categories):
        ax  = axes_flat[i]
        sub = (
            cat_gene_agg[cat_gene_agg["category"] == cat]
            .sort_values("mean_eigen", ascending=False)
            .head(top_n_genes)
        )
        ax.barh(
            sub["gene"], sub["mean_eigen"],
            color=palette[cat], edgecolor="white", linewidth=0.4,
        )
        ax.invert_yaxis()
        ax.set_title(cat, fontsize=8, fontweight="bold", color=palette[cat])
        ax.set_xlabel("Mean eigen-centrality", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=6)

    for j in range(len(categories), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Top {top_n_genes} eigen-central genes per category\n"
        "(averaged across layers in that category)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"bar_top_eigen_per_category.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7 — LCC size heatmap: category × broad
# ---------------------------------------------------------------------------
def plot_lcc_heatmap_category_broad(
    metrics_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Plotting LCC heatmap (category × broad)")

    cell_df = metrics_df[metrics_df["category"] != "special"].copy()

    pivot = (
        cell_df
        .groupby(["category", "broad"])
        .agg(mean_lcc=("lcc_fraction", "mean"))
        .reset_index()
        .pivot(index="category", columns="broad", values="mean_lcc")
        .fillna(0)
    )

    if pivot.empty:
        print("  ⚠  Not enough data for LCC heatmap, skipping")
        return

    fig_h = max(6, len(pivot)         * 0.55)
    fig_w = max(8, len(pivot.columns) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot, ax=ax,
        cmap="YlOrRd",
        annot=True, fmt=".2f",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Mean LCC fraction", "shrink": 0.6},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_title(
        "Mean LCC fraction — category × broad category",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"heatmap_lcc_category_broad.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 8 — degree distribution (log-log) per category
# ---------------------------------------------------------------------------
def plot_degree_distributions(
    multiplex_dir: Path,
    metrics_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    out: Path,
    max_layers_per_cat: int = 5,
) -> None:
    """
    For each category, overlay degree distributions (CCDF) of up to
    max_layers_per_cat representative layers on the same axes.
    """
    _print_block("Plotting degree distributions per category")

    ct_to_ai = dict(zip(categories_df[COL_CELL_TYPE], categories_df[COL_AI]))

    cell_df    = metrics_df[metrics_df["category"] != "special"].copy()
    categories = sorted(cell_df["category"].unique())
    palette    = _category_palette(categories)

    n_cols = min(4, len(categories))
    n_rows = int(np.ceil(len(categories) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, n_rows * 4),
    )
    axes_flat = np.array(axes).flatten()

    for i, cat in enumerate(categories):
        ax   = axes_flat[i]
        sub  = (
            cell_df[cell_df["category"] == cat]
            .sort_values("n_edges", ascending=False)
            .head(max_layers_per_cat)
        )

        for _, row in sub.iterrows():
            tsv_path = multiplex_dir / f"{row['layer']}.tsv"
            G = load_graph_from_tsv(tsv_path)
            if G.number_of_nodes() == 0:
                continue
            degrees = sorted([d for _, d in G.degree()], reverse=True)
            # CCDF
            ccdf = np.arange(1, len(degrees) + 1) / len(degrees)
            ax.loglog(
                degrees, ccdf,
                alpha=0.65, linewidth=1.2,
                label=row["cell_type"][:22],
                color=palette[cat],
            )

        ax.set_xlabel("Degree (log)", fontsize=7)
        ax.set_ylabel("P(K ≥ k) (log)", fontsize=7)
        ax.set_title(cat, fontsize=8, fontweight="bold", color=palette[cat])
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=4.5, loc="lower left")

    for j in range(len(categories), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Degree distribution CCDF per category\n"
        f"(up to {max_layers_per_cat} layers per category)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"degree_distribution_per_category.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------
def save_tables(
    metrics_df: pd.DataFrame,
    cat_agg_df: pd.DataFrame,
    eigen_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Saving output tables")

    # Drop list columns for clean CSV output
    flat_metrics = metrics_df.copy()
    out_path = out / "table_layer_metrics.csv"
    flat_metrics.to_csv(out_path, index=False)
    print(f"  → {out_path.name}  ({len(flat_metrics):,} rows)")

    out_path2 = out / "table_category_summary.csv"
    cat_agg_df.to_csv(out_path2, index=False)
    print(f"  → {out_path2.name}  ({len(cat_agg_df):,} rows)")

    if not eigen_df.empty:
        out_path3 = out / "table_top_eigen_genes.csv"
        eigen_df.to_csv(out_path3, index=False)
        print(f"  → {out_path3.name}  ({len(eigen_df):,} rows)")

    # Pivot: category × metric (mean ± std)
    cell_df = metrics_df[metrics_df["category"] != "special"]
    scalar_cols = list(METRIC_LABELS.keys())
    stats_rows = []
    for cat, grp in cell_df.groupby("category"):
        row = {"category": cat}
        for col in scalar_cols:
            row[f"{col}_mean"] = round(grp[col].mean(), 5)
            row[f"{col}_std"]  = round(grp[col].std(),  5)
            row[f"{col}_min"]  = round(grp[col].min(),  5)
            row[f"{col}_max"]  = round(grp[col].max(),  5)
        stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows).sort_values("n_edges_mean", ascending=False)
    out_path4 = out / "table_category_metric_stats.csv"
    stats_df.to_csv(out_path4, index=False)
    print(f"  → {out_path4.name}  ({len(stats_df):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and visualise multiplex layer topology metrics"
    )
    parser.add_argument(
        "--wdir", required=True,
        help="MultiXrank working directory (contains multiplex/gene_multiplex/)",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to layers_categories.csv",
    )
    parser.add_argument(
        "--output-dir", default="topology_output",
        help="Output directory (default: topology_output)",
    )
    parser.add_argument(
        "--exclude-ai", nargs="+", default=[],
        help="Exclude rows by 'category' value (case-insensitive)",
    )
    parser.add_argument(
        "--extra-stems", nargs="+",
        default=list(SPECIAL_STEMS),
        help="Additional layer stems to analyse (e.g. ppi_layer)",
    )
    parser.add_argument(
        "--top-n-eigen", type=int, default=10,
        help="Number of top eigen-centrality genes to record per layer",
    )
    parser.add_argument(
        "--min-edges", type=int, default=1,
        help="Minimum edge count to include a layer in analysis",
    )
    parser.add_argument(
        "--max-layers-plot", type=int, default=70,
        help="Maximum layers shown in bar/heatmap plots",
    )
    args = parser.parse_args()

    wdir          = Path(args.wdir).resolve()
    out           = Path(args.output_dir)
    multiplex_dir = wdir / "multiplex" / "gene_multiplex"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = (
        Path(args.csv) if args.csv
        else SCRIPT_DIR / "layers_categories.csv"
    )

    # ── Load categories ───────────────────────────────────────────────────
    _print_block("Loading input files")
    categories_df = load_categories(
        csv_path, exclude_ai=args.exclude_ai or None
    )

    # ── Compute metrics ───────────────────────────────────────────────────
    metrics_df, eigen_df = analyse_all_layers(
        multiplex_dir      = multiplex_dir,
        categories_df      = categories_df,
        extra_stems        = args.extra_stems,
        top_n_eigen        = args.top_n_eigen,
        min_edges          = args.min_edges,
    )

    if metrics_df.empty:
        print("\n⚠  No layers found/analysed — check --wdir path.")
        return

    # ── Category aggregation ──────────────────────────────────────────────
    cat_agg_df = aggregate_by_category(metrics_df)

    # ── Save tables ───────────────────────────────────────────────────────
    save_tables(metrics_df, cat_agg_df, eigen_df, out)

    # ── Plots ─────────────────────────────────────────────────────────────
    _print_block("Generating plots")

    plot_per_layer_metrics(
        metrics_df, out, max_layers=args.max_layers_plot
    )
    plot_metrics_heatmap(
        metrics_df, out, max_layers=args.max_layers_plot
    )
    plot_category_distributions(metrics_df, out)
    plot_category_summary_bars(cat_agg_df, out)
    plot_scatter_density_clustering(metrics_df, out)
    plot_top_eigen_per_category(eigen_df, out)
    plot_lcc_heatmap_category_broad(metrics_df, out)
    plot_degree_distributions(
        multiplex_dir, metrics_df, categories_df, out
    )

    # ── Done ──────────────────────────────────────────────────────────────
    _print_block("DONE")
    print(f"  Working dir      : {wdir}")
    print(f"  Output dir       : {out.resolve()}")
    print(f"  Layers analysed  : {len(metrics_df):,}")
    print(f"  Categories found : {metrics_df['category'].nunique() - 1}")
    print()
    print("  Output files:")
    for fp in sorted(out.glob("*")):
        if fp.is_file():
            size_kb = fp.stat().st_size / 1024
            print(f"    {fp.name:<55} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()