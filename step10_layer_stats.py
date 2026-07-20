#!/usr/bin/env python3
"""
multiplex_layer_topology_analysis.py
======================================
Computes per-layer topological metrics for a MultiXrank multiplex with
hierarchical DRG neuron cell-type grouping:
  - A fibers (proprioceptors, A-beta LTMR, A-LTMR, peptidergic)
  - C fibers (C-LTMR, peptidergic, MRGPRX nociceptors, thermosensitive)
  - Non-classified (ATF3 injured)
  - Non-neuronal (immune, glial, etc.)
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
# DRG NEURON HIERARCHY DEFINITION
# ---------------------------------------------------------------------------
DRG_HIERARCHY = {
    "A-fiber": {
        "color": "#1f77b4",  # blue
        "subcategories": {
            "Proprioceptors": {
                "marker": "o",
                "cell_types": [
                    "A-Propr.EPHA3",
                    "A-Propr.HAPLN4",
                    "A-Propr.PCDH8",
                ]
            },
            "A-beta LTMR": {
                "marker": "s",
                "cell_types": [
                    "Ab-LTMR.CCKAR",
                    "Ab-LTMR.ETV1",
                    "Ab-LTMR.LGI2",
                    "Ab-LTMR.NSG2",
                ]
            },
            "A-LTMR": {
                "marker": "^",
                "cell_types": [
                    "A-LTMR.TAC3",
                ]
            },
            "A-peptidergic": {
                "marker": "D",
                "cell_types": [
                    "A-PEP.CHRNA7_SLC18A3",
                    "A-PEP.KIT",
                    "A-PEP.NTRK3_S100A16",
                    "A-PEP.SCGN_ADRA2C",
                ]
            },
        }
    },
    "C-fiber": {
        "color": "#ff7f0e",  # orange
        "subcategories": {
            "C-LTMR": {
                "marker": "o",
                "cell_types": [
                    "C-LTMR.CDH9",
                ]
            },
            "C-peptidergic": {
                "marker": "D",
                "cell_types": [
                    "C-PEP.ADORA2B",
                    "C-PEP.TAC1_CACNG5",
                    "C-PEP.TAC1_CHRNA3",
                ]
            },
            "MRGPRX nociceptors": {
                "marker": "s",
                "cell_types": [
                    "C-NP.MRGPRX1_GFRA2",
                    "C-NP.MRGPRX1_MRGPRX4",
                ]
            },
            "C-thermosensitive": {
                "marker": "*",
                "cell_types": [
                    "C-Thermo.RXFP1",
                    "C-Thermo.TRPM8",
                ]
            },
            "C-SST nociceptor": {
                "marker": "v",
                "cell_types": [
                    "C-NP.SST",
                ]
            },
        }
    },
    "Non-classified": {
        "color": "#d62728",  # red
        "subcategories": {
            "ATF3 (injured)": {
                "marker": "x",
                "cell_types": [
                    "ATF3",
                ]
            },
        }
    },
    "Non-neuronal": {
        "color": "#2ca02c",  # green
        "subcategories": {
            "Immune": {
                "marker": "o",
                "cell_types": [
                    "B cell",
                    "CD4-positive, alpha-beta T cell",
                    "CD8-positive, alpha-beta T cell",
                    "activated CD4-positive, alpha-beta T cell",
                    "activated CD8-positive, alpha-beta T cell",
                    "basophil",
                    "classical monocyte",
                    "gamma-delta T cell",
                    "granulocyte",
                    "innate lymphoid cell",
                    "intermediate monocyte",
                    "leukocyte",
                    "macrophage",
                    "mast cell",
                    "mature NK T cell",
                    "myeloid cell",
                    "myeloid dendritic cell",
                    "myeloid leukocyte",
                    "naive thymus-derived CD4-positive, alpha-beta T cell",
                    "naive thymus-derived CD8-positive, alpha-beta T cell",
                    "natural killer cell",
                    "neutrophil",
                    "non-classical monocyte",
                    "plasma cell",
                    "plasmacytoid dendritic cell",
                    "regulatory T cell",
                ]
            },
            "Glial/Nervous": {
                "marker": "s",
                "cell_types": [
                    "Schwann cell",
                    "enteroglial cell",
                    "microglial cell",
                    "neuron",
                ]
            },
            "Immune/Hematopoietic": {
                "marker": "^",
                "cell_types": [
                    "Langerhans cell",
                    "colon macrophage",
                    "common myeloid progenitor",
                    "erythroid progenitor cell",
                    "hematopoietic cell",
                    "hematopoietic precursor cell",
                    "hematopoietic stem cell",
                    "tissue-resident macrophage",
                ]
            },
        }
    }
}

# Build reverse lookup: cell_type → (major_category, subcategory, info)
CELL_TYPE_TO_CATEGORY = {}
for major_cat, major_data in DRG_HIERARCHY.items():
    for subcat, subcat_data in major_data["subcategories"].items():
        for cell_type in subcat_data["cell_types"]:
            CELL_TYPE_TO_CATEGORY[cell_type] = {
                "major": major_cat,
                "minor": subcat,
                "color": major_data["color"],
                "marker": subcat_data["marker"],
            }

# ---------------------------------------------------------------------------
# Column name constants (must match layers_categories.csv)
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

def get_category_from_cell_type(cell_type: str) -> dict:
    """
    Return category info for a cell type.
    If not in DRG_HIERARCHY, return a default 'unknown' category.
    """
    if cell_type in CELL_TYPE_TO_CATEGORY:
        return CELL_TYPE_TO_CATEGORY[cell_type]
    else:
        return {
            "major": "unknown",
            "minor": "unknown",
            "color": "#808080",
            "marker": ".",
        }

def load_categories(
    csv_path: Path,
    exclude_ai: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0].strip() in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]
    
    required = {COL_CELL_TYPE, COL_BROAD}
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
        if COL_AI in df.columns:
            mask = df[COL_AI].str.lower().isin([e.lower() for e in exclude_ai])
            df   = df[~mask]
            print(f"Excluded {before - len(df)} rows by '{COL_AI}' ({exclude_ai})")
    
    print(f"Retained {len(df)} cell types")
    return df.reset_index(drop=True)

def stem_from_cell_type(cell_type: str) -> str:
    return cell_type.replace(" ", "_") + "_ppi_layer"

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
    """Compute topological metrics for a single layer graph."""
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

    density = nx.density(G)
    avg_clust = nx.average_clustering(G)

    components  = list(nx.connected_components(G))
    lcc_size    = max(len(c) for c in components)
    lcc_frac    = lcc_size / n_nodes
    n_comp      = len(components)
    is_conn     = (n_comp == 1)

    degrees     = [d for _, d in G.degree()]
    deg_mean    = float(np.mean(degrees))
    deg_median  = float(np.median(degrees))
    deg_max     = int(np.max(degrees))

    lcc_nodes   = max(components, key=len)
    G_lcc       = G.subgraph(lcc_nodes).copy()
    try:
        eigen_cent  = nx.eigenvector_centrality_numpy(G_lcc, weight=None)
    except Exception:
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
      - metrics_df  : one row per layer with category metadata
      - eigen_df    : one row per (layer, gene) for top-eigen nodes
    """
    _print_block("Computing per-layer topology metrics")

    # Map exact stem name to original metadata row
    stem_to_meta = {}
    for _, row in categories_df.iterrows():
        ct = row[COL_CELL_TYPE]
        stem = stem_from_cell_type(ct)
        cat_info = get_category_from_cell_type(ct)
        stem_to_meta[stem] = {
            "cell_type": ct,
            "major_category": cat_info["major"],
            "minor_category": cat_info["minor"],
            "color": cat_info["color"],
            "marker": cat_info["marker"],
            "broad": row[COL_BROAD] if COL_BROAD in row else "unknown",
        }

    def _meta(stem: str) -> dict:
        """Return category metadata for a stem."""
        if stem in SPECIAL_STEMS:
            return {
                "cell_type": stem,
                "major_category": "special",
                "minor_category": "special",
                "color": "#666666",
                "marker": ".",
                "broad": "special",
            }
        if stem in stem_to_meta:
            return stem_to_meta[stem]
        # Fallback
        return {
            "cell_type": stem,
            "major_category": "unknown",
            "minor_category": "unknown",
            "color": "#808080",
            "marker": ".",
            "broad": "unknown",
        }

    allowed_stems = set(stem_to_meta.keys())
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

        meta = _meta(stem)
        m = compute_layer_metrics(G, stem, top_n_eigen=top_n_eigen)

        # Augment with categorical metadata
        m.update({
            "cell_type": meta["cell_type"],
            "major_category": meta["major_category"],
            "minor_category": meta["minor_category"],
            "color": meta["color"],
            "marker": meta["marker"],
            "broad": meta["broad"],
        })

        metrics_rows.append({k: v for k, v in m.items()
                              if k not in ("top_eigen_nodes", "top_eigen_scores")})

        # Eigen rows
        for gene, score in zip(m["top_eigen_nodes"], m["top_eigen_scores"]):
            eigen_rows.append({
                "layer":     stem,
                "cell_type": meta["cell_type"],
                "major_category": meta["major_category"],
                "minor_category": meta["minor_category"],
                "gene":      gene,
                "eigen_score": score,
                "eigen_rank":  m["top_eigen_nodes"].index(gene) + 1,
            })

        print(
            f"  {stem:<55} "
            f"N={m['n_nodes']:>6,}  E={m['n_edges']:>7,}  "
            f"dens={m['density']:.5f}  "
            f"clust={m['avg_clustering']:.4f}  "
            f"LCC={m['lcc_fraction']:.3f}  "
            f"[{meta['major_category']}:{meta['minor_category']}]"
        )

    print(f"\n  Layers analysed : {len(metrics_rows)}")
    print(f"  Layers skipped  : {skipped}  (< {min_edges} edges)")

    metrics_df = pd.DataFrame(metrics_rows)
    eigen_df   = pd.DataFrame(eigen_rows)
    return metrics_df, eigen_df

# ---------------------------------------------------------------------------
# Category-level aggregation
# ---------------------------------------------------------------------------
def aggregate_by_major_category(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by major DRG category."""
    _print_block("Aggregating metrics by major category")
    
    cell_df = metrics_df[metrics_df["major_category"] != "special"].copy()
    
    agg = (
        cell_df
        .groupby("major_category")
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

def aggregate_by_minor_category(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by minor (subcategory) DRG category."""
    _print_block("Aggregating metrics by minor category")
    
    cell_df = metrics_df[metrics_df["major_category"] != "special"].copy()
    
    agg = (
        cell_df
        .groupby(["major_category", "minor_category"])
        .agg(
            n_layers        =("layer",          "count"),
            total_nodes     =("n_nodes",         "sum"),
            total_edges     =("n_edges",         "sum"),
            mean_density    =("density",         "mean"),
            mean_clustering =("avg_clustering",  "mean"),
            mean_lcc_frac   =("lcc_fraction",    "mean"),
        )
        .reset_index()
        .sort_values(["major_category", "total_edges"], ascending=[True, False])
    )
    
    for col in ("mean_density", "mean_clustering", "mean_lcc_frac"):
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
    "lcc_size":       "LCC size (nodes)",
    "degree_mean":    "Mean degree",
    "degree_max":     "Max degree",
}

def _major_category_palette() -> dict[str, str]:
    """Return color mapping for major DRG categories."""
    return {
        "A-fiber": "#1f77b4",
        "C-fiber": "#ff7f0e",
        "Non-classified": "#d62728",
        "Non-neuronal": "#2ca02c",
        "special": "#999999",
        "unknown": "#999999",
    }

# ---------------------------------------------------------------------------
# Plot 1 — per-layer bar plots colored by major category
# ---------------------------------------------------------------------------
def plot_per_layer_metrics(
    metrics_df: pd.DataFrame,
    out: Path,
    max_layers: int = 80,
) -> None:
    _print_block("Plotting per-layer metric bar charts (colored by major category)")
    
    cell_df = metrics_df[metrics_df["major_category"] != "special"].copy()
    
    # Check if we have data to plot
    if cell_df.empty:
        print("  ⚠  No cell-type layers available for per-layer metrics plot (skipping)")
        return
    
    cell_df = cell_df.sort_values("n_edges", ascending=False).head(max_layers)
    
    palette  = _major_category_palette()
    colors   = [palette.get(c, "#999999") for c in cell_df["major_category"]]
    # y_labels = [f"{ct}\n({mc})" for ct, mc in 
    #              zip(cell_df["cell_type"], cell_df["minor_category"])]
    y_labels = cell_df["cell_type"].tolist()
    metrics_to_plot = list(METRIC_LABELS.keys())
    
    # Check if we have metrics to plot
    if not metrics_to_plot:
        print("  ⚠  No metrics available for per-layer plot (skipping)")
        return
    
    n_m = len(metrics_to_plot)
    
    fig, axes = plt.subplots(
        1, n_m,
        figsize=(n_m * 3.8, max(6, len(cell_df) * 0.22)),
        sharey=True,
    )
    
    for idx, (ax, metric) in enumerate(zip(axes, metrics_to_plot)):
        vals = cell_df[metric].tolist()
        ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.3)
        ax.invert_yaxis()
        ax.set_xlabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=10)
        
        if idx == 0:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=10)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    
    # after
    present = set(cell_df["major_category"].unique())
    patches = [
        mpatches.Patch(color=c, label=cat)
        for cat, c in sorted(palette.items()) if cat in present]

    axes[-1].legend(
        handles=patches, loc="upper right",
        fontsize=10, ncol=1, framealpha=0.8,
        title="Major Category", title_fontsize=10,
    )
    
    fig.suptitle(
        f"Per-layer topology metrics  (top {max_layers} by edge count)\nColored by DRG neuron type",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"01_bar_per_layer_metrics.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2 — heatmap of metrics × layers (colored by category)
# ---------------------------------------------------------------------------
def plot_metrics_heatmap(
    metrics_df: pd.DataFrame,
    out: Path,
    max_layers: int = 60,
) -> None:
    _print_block("Plotting metrics heatmap (layers × metrics)")
    
    # Filter and get top layers
    cell_df = (
        metrics_df[metrics_df["major_category"] != "special"]
        .sort_values("n_edges", ascending=False)
        .head(max_layers)
        .copy()
    )
    
    # Check if we have data to plot
    if cell_df.empty:
        print("  ⚠  No cell-type layers available for heatmap (skipping)")
        return
    
    scalar_cols = list(METRIC_LABELS.keys())
    if not scalar_cols:
        print("  ⚠  No metrics available for heatmap (skipping)")
        return
    
    # Check if all required columns exist
    missing_cols = [col for col in scalar_cols if col not in cell_df.columns]
    if missing_cols:
        print(f"  ⚠  Missing metric columns: {missing_cols} (skipping)")
        return
    
    # Build matrix
    mat = cell_df.set_index("cell_type")[scalar_cols].copy().astype(float)
    
    # Check if matrix is empty or all NaN
    if mat.empty or mat.isna().all().all():
        print("  ⚠  Metric matrix is empty or all NaN (skipping)")
        return
    
    # Normalize (handle case where std is 0)
    mat_std = mat.std()
    mat_std[mat_std == 0] = 1  # Avoid division by zero
    mat_norm = (mat - mat.mean()) / (mat_std + 1e-12)
    
    # Check if normalized matrix has any valid (non-NaN) data
    if mat_norm.isna().all().all():
        print("  ⚠  All metrics are NaN after normalization (skipping)")
        return
    
    # Create row colors based on major category
    palette = _major_category_palette()
    row_colors = [palette.get(cat, "#999999") 
                  for cat in cell_df["major_category"].values]
    
    fig_h = max(8, len(mat_norm) * 0.25)
    fig_w = max(10, len(scalar_cols) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    try:
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
    except ValueError as e:
        print(f"  ⚠  Heatmap rendering failed: {e} (skipping)")
        plt.close(fig)
        return
    
    # Add row-side color bar
    for i, color in enumerate(row_colors):
        ax.add_patch(plt.Rectangle((-0.15, i), 0.1, 1, facecolor=color, edgecolor="none"))
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(
        f"Layer topology metrics (Z-scored, top {max_layers})\nRow colors = DRG neuron type",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"02_heatmap_layer_metrics.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3 — category-level violin plots (major × minor)
# ---------------------------------------------------------------------------
def plot_category_distributions(
    metrics_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Plotting category-level metric distributions")
    
    cell_df  = metrics_df[metrics_df["major_category"] != "special"].copy()
    
    # Check if we have data
    if cell_df.empty:
        print("  ⚠  No cell-type layers available for category distributions (skipping)")
        return
    
    palette  = _major_category_palette()
    
    # Order by major category, then by mean edges within minor category
    order = (
        cell_df.groupby("major_category")["n_edges"]
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
    
    n_m = len(metrics_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    for ax, (metric, label) in zip(axes, metrics_to_plot):
        sns.violinplot(
            data=cell_df,
            x="major_category",
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
            x="major_category",
            y=metric,
            order=order,
            color="black",
            size=2.5,
            alpha=0.5,
            ax=ax,
            jitter=True,
        )
        ax.set_xlabel("Major Category", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
    
    for j in range(n_m, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(
        "Topology metric distributions per major DRG neuron category",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"03_violin_category_metrics.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 4 — major category summary bars
# ---------------------------------------------------------------------------
def plot_major_category_summary(
    cat_agg_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Plotting major category summary bars")
    
    # Check if we have data
    if cat_agg_df.empty:
        print("  ⚠  No category summary data available (skipping)")
        return
    
    palette = _major_category_palette()
    metrics_to_plot = [
        ("mean_nodes",      "Mean node count"),
        ("mean_edges",      "Mean edge count"),
        ("mean_density",    "Mean density"),
        ("mean_clustering", "Mean clustering"),
        ("mean_lcc_frac",   "Mean LCC fraction"),
        ("pct_connected",   "% fully connected"),
    ]
    
    n_m = len(metrics_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for ax, (col, label) in zip(axes, metrics_to_plot):
        sorted_df = cat_agg_df.sort_values(col, ascending=False)
        colors = [palette.get(c, "#999999") for c in sorted_df["major_category"]]
        
        bars = ax.bar(
            sorted_df["major_category"], sorted_df[col],
            color=colors, edgecolor="white", linewidth=0.5,
        )
        ax.set_xlabel("Major Category", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        
        for bar, val in zip(bars, sorted_df[col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bar.get_height() * 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )
    
    for j in range(n_m, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(
        "Major category-level mean topology metrics",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"04_bar_major_category_summary.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 5 — scatter plots colored by major category
# ---------------------------------------------------------------------------
def plot_scatter_density_clustering(
    metrics_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Plotting scatter plots (colored by major category)")
    
    cell_df = metrics_df[metrics_df["major_category"] != "special"].copy()
    
    # Check if we have data
    if cell_df.empty:
        print("  ⚠  No cell-type layers available for scatter plots (skipping)")
        return
    
    palette = _major_category_palette()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    scatter_pairs = [
        ("density",        "avg_clustering", "Density",     "Avg clustering"),
        ("n_nodes",        "n_edges",        "Node count",  "Edge count"),
        ("lcc_fraction",   "avg_clustering", "LCC fraction","Avg clustering"),
    ]
    
    for ax, (xc, yc, xl, yl) in zip(axes, scatter_pairs):
        for major_cat in sorted(cell_df["major_category"].unique()):
            grp = cell_df[cell_df["major_category"] == major_cat]
            ax.scatter(
                grp[xc], grp[yc],
                label=major_cat,
                color=palette.get(major_cat, "#999999"),
                s=50, alpha=0.75, edgecolors="white", linewidths=0.5,
            )
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(f"{xl} vs {yl}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=4, fontsize=11,
        bbox_to_anchor=(0.5, -0.08), framealpha=0.95,
        title="Major DRG Category", title_fontsize=11,
    )
    
    fig.suptitle(
        "Layer topology scatter plots (coloured by DRG neuron type)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"05_scatter_layer_topology.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 6 — top eigen genes per major category
# ---------------------------------------------------------------------------
def plot_top_eigen_per_category(
    eigen_df: pd.DataFrame,
    out: Path,
    top_n_genes: int = 15,
) -> None:
    _print_block("Plotting top eigen-centrality genes per major category")
    
    if eigen_df.empty:
        print("  ⚠  No eigen data, skipping")
        return
    
    cell_eigen = eigen_df[eigen_df["major_category"] != "special"].copy()
    
    # Check if we have cell-type eigen data
    if cell_eigen.empty:
        print("  ⚠  No cell-type eigen data (all marked as 'special'), skipping")
        return
    
    cat_gene_agg = (
        cell_eigen
        .groupby(["major_category", "gene"])
        .agg(
            mean_eigen=("eigen_score", "mean"),
            n_layers=("layer",        "count"),
        )
        .reset_index()
    )
    
    # Check if aggregation produced any data
    if cat_gene_agg.empty:
        print("  ⚠  No genes found in eigen data, skipping")
        return
    
    categories = sorted(cat_gene_agg["major_category"].unique())
    
    # Check if we have categories
    if not categories or len(categories) == 0:
        print("  ⚠  No categories found in eigen data, skipping")
        return
    
    palette    = _major_category_palette()
    
    n_cols = min(2, len(categories))
    n_rows = int(np.ceil(len(categories) / n_cols))
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 6, n_rows * 5),
    )
    axes_flat = np.array(axes).flatten()
    
    for i, cat in enumerate(categories):
        ax = axes_flat[i]
        sub = (
            cat_gene_agg[cat_gene_agg["major_category"] == cat]
            .sort_values("mean_eigen", ascending=False)
            .head(top_n_genes)
        )
        ax.barh(
            sub["gene"], sub["mean_eigen"],
            color=palette.get(cat, "#999999"), edgecolor="white", linewidth=0.5,
        )
        ax.invert_yaxis()
        ax.set_title(f"{cat}\n({len(sub)} top genes)", 
                     fontsize=11, fontweight="bold", color=palette.get(cat))
        ax.set_xlabel("Mean eigen-centrality", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
    
    for j in range(len(categories), len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    fig.suptitle(
        f"Top {top_n_genes} eigen-central genes per major DRG category\n"
        "(averaged across layers in each category)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"06_bar_top_eigen_per_category.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 7 — degree distribution CCDF per major category
# ---------------------------------------------------------------------------
def plot_degree_distributions(
    multiplex_dir: Path,
    metrics_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    out: Path,
    max_layers_per_cat: int = 15,
) -> None:
    _print_block("Plotting degree distributions per major category")

    stem_to_ct = {}
    for _, row in categories_df.iterrows():
        stem_to_ct[stem_from_cell_type(row[COL_CELL_TYPE])] = row[COL_CELL_TYPE]

    cell_df    = metrics_df[metrics_df["major_category"] != "special"].copy()
    
    # Check if we have data
    if cell_df.empty:
        print("  ⚠  No cell-type layers available for degree distributions (skipping)")
        return
    
    categories = sorted(cell_df["major_category"].unique())
    
    # Check if categories list is empty
    if not categories or len(categories) == 0:
        print("  ⚠  No categories found for degree distributions (skipping)")
        return

    n_cols = min(2, len(categories))
    n_rows = int(np.ceil(len(categories) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 6, n_rows * 5),
    )
    axes_flat = np.array(axes).flatten()

    for i, cat in enumerate(categories):
        ax  = axes_flat[i]
        sub = (
            cell_df[cell_df["major_category"] == cat]
            .sort_values("n_edges", ascending=False)
            .head(max_layers_per_cat)
        )

        # ── one distinct color per line within this panel ──────────────────
        line_colors = plt.cm.tab20(np.linspace(0, 1, max(len(sub), 1)))

        for line_idx, (_, row) in enumerate(sub.iterrows()):
            tsv_path = multiplex_dir / f"{row['layer']}.tsv"
            G = load_graph_from_tsv(tsv_path)
            if G.number_of_nodes() == 0:
                continue

            degrees = sorted([d for _, d in G.degree()], reverse=True)
            ccdf    = np.arange(1, len(degrees) + 1) / len(degrees)
            ct_display = stem_to_ct.get(row["layer"], row["layer"])

            ax.loglog(
                degrees, ccdf,
                alpha=0.85, linewidth=1.8,
                label=ct_display[:30],
                color=line_colors[line_idx],   # ← per-line color
            )

        ax.set_xlabel("Degree (log)", fontsize=10)
        ax.set_ylabel("P(K ≥ k) (log)", fontsize=10)
        ax.set_title(cat, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8, loc="lower left")

    for j in range(len(categories), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Degree distribution CCDF per major DRG category\n"
        f"(up to {max_layers_per_cat} layers per category)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"07_degree_distribution_ccdf.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 8 — LCC fraction heatmap (major × minor)
# ---------------------------------------------------------------------------
def plot_lcc_heatmap_hierarchical(
    metrics_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Plotting LCC heatmap (major × minor category)")
    
    cell_df = metrics_df[metrics_df["major_category"] != "special"].copy()
    
    # Check if we have data
    if cell_df.empty:
        print("  ⚠  No cell-type layers available for LCC heatmap (skipping)")
        return
    
    pivot = (
        cell_df
        .groupby(["major_category", "minor_category"])
        .agg(mean_lcc=("lcc_fraction", "mean"))
        .reset_index()
        .pivot(index="major_category", columns="minor_category", values="mean_lcc")
        .fillna(0)
    )
    
    if pivot.empty:
        print("  ⚠  Not enough data for LCC heatmap, skipping")
        return
    
    fig_h = max(6, len(pivot) * 0.7)
    fig_w = max(10, len(pivot.columns) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    sns.heatmap(
        pivot, ax=ax,
        cmap="YlOrRd",
        annot=True, fmt=".3f",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Mean LCC fraction", "shrink": 0.7},
        vmin=0, vmax=1,
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_title(
        "Mean LCC fraction — DRG hierarchy (major × minor category)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    
    for ext in ("pdf", "png"):
        fp = out / f"08_heatmap_lcc_hierarchical.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 9 — sankey-style flow: major → minor categories
# ---------------------------------------------------------------------------
def plot_drg_hierarchy_summary(
    metrics_df: pd.DataFrame,
    out: Path,
) -> None:
    """
    Create a visual summary of DRG hierarchy with layer counts.
    """
    _print_block("Plotting DRG hierarchy summary")
    
    cell_df = metrics_df[metrics_df["major_category"] != "special"].copy()
    
    # Check if we have data
    if cell_df.empty:
        print("  ⚠  No cell-type layers available for hierarchy summary (skipping)")
        return
    
    # Count layers per major/minor category
    hier = (
        cell_df
        .groupby(["major_category", "minor_category"])
        .agg(n_layers=("layer", "count"))
        .reset_index()
    )
    
    if hier.empty:
        print("  ⚠  No hierarchy data available (skipping)")
        return
    
    palette = _major_category_palette()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    major_cats = sorted(hier["major_category"].unique())
    y_offset = 0
    bar_height = 0.6
    
    for major_idx, major_cat in enumerate(major_cats):
        sub = hier[hier["major_category"] == major_cat].sort_values("n_layers", ascending=False)
        
        for minor_idx, (_, row) in enumerate(sub.iterrows()):
            ax.barh(
                y_offset,
                row["n_layers"],
                height=bar_height,
                label=f"{major_cat}: {row['minor_category']}",
                color=palette[major_cat],
                alpha=0.6 + 0.4 * (minor_idx / max(1, len(sub) - 1)),
                edgecolor="white",
                linewidth=1,
            )
            
            # Add text label
            ax.text(
                row["n_layers"] / 2,
                y_offset,
                f"{row['n_layers']}",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="white",
            )
            
            y_offset += 1
        
        y_offset += 0.5  # gap between major categories
    
    ax.set_xlabel("Number of layers", fontsize=11, fontweight="bold")
    ax.set_ylabel("DRG Neuron Subtype", fontsize=11, fontweight="bold")
    ax.set_title(
        "DRG Neuron Hierarchy: Layer count per category",
        fontsize=12, fontweight="bold",
    )
    
    # Create custom legend for major categories
    handles = [mpatches.Patch(color=palette[cat], label=cat) 
               for cat in major_cats]
    ax.legend(handles=handles, loc="lower right", fontsize=10, title="Major Category")
    
    ax.set_yticks([])
    plt.tight_layout()
    
    for ext in ("pdf", "png"):
        fp = out / f"09_drg_hierarchy_summary.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------
def save_tables(
    metrics_df: pd.DataFrame,
    major_agg_df: pd.DataFrame,
    minor_agg_df: pd.DataFrame,
    eigen_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Saving output tables")
    
    flat_metrics = metrics_df.copy()
    out_path = out / "01_table_layer_metrics.csv"
    flat_metrics.to_csv(out_path, index=False)
    print(f"  → {out_path.name}  ({len(flat_metrics):,} rows)")

    out_path2 = out / "02_table_major_category_summary.csv"
    major_agg_df.to_csv(out_path2, index=False)
    print(f"  → {out_path2.name}  ({len(major_agg_df):,} rows)")
    
    out_path2b = out / "03_table_minor_category_summary.csv"
    minor_agg_df.to_csv(out_path2b, index=False)
    print(f"  → {out_path2b.name}  ({len(minor_agg_df):,} rows)")

    if not eigen_df.empty:
        out_path3 = out / "04_table_top_eigen_genes.csv"
        eigen_df.to_csv(out_path3, index=False)
        print(f"  → {out_path3.name}  ({len(eigen_df):,} rows)")

    cell_df = metrics_df[metrics_df["major_category"] != "special"]
    scalar_cols = list(METRIC_LABELS.keys())
    stats_rows = []
    
    if not cell_df.empty and scalar_cols:
        for major_cat, major_grp in cell_df.groupby("major_category"):
            for minor_cat, minor_grp in major_grp.groupby("minor_category"):
                row = {
                    "major_category": major_cat,
                    "minor_category": minor_cat,
                    "n_layers": len(minor_grp),
                }
                for col in scalar_cols:
                    row[f"{col}_mean"] = round(minor_grp[col].mean(), 5)
                    row[f"{col}_std"]  = round(minor_grp[col].std(),  5)
                    row[f"{col}_min"]  = round(minor_grp[col].min(),  5)
                    row[f"{col}_max"]  = round(minor_grp[col].max(),  5)
                stats_rows.append(row)
        
        stats_df = pd.DataFrame(stats_rows)
        # Only sort if we have data
        if not stats_df.empty and "major_category" in stats_df.columns:
            stats_df = stats_df.sort_values(
                ["major_category", "n_layers"], ascending=[True, False]
            )
    else:
        # Create empty dataframe - no cell-type layers to analyze
        stats_df = pd.DataFrame()
        if cell_df.empty:
            print(f"  ⚠  Warning: No cell-type layers found (all layers are marked as 'special')")
        if not scalar_cols:
            print(f"  ⚠  Warning: No scalar metrics available in METRIC_LABELS")
    
    # Only save and report if we have data
    if not stats_df.empty:
        out_path4 = out / "05_table_hierarchical_metric_stats.csv"
        stats_df.to_csv(out_path4, index=False)
        print(f"  → {out_path4.name}  ({len(stats_df):,} rows)")
    else:
        print(f"  ⚠  No hierarchical statistics table generated (insufficient data)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and visualise multiplex layer topology metrics with DRG hierarchy"
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
        "--max-layers-plot", type=int, default=45,
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
    
    _print_block("Loading input files")
    categories_df = load_categories(
        csv_path, exclude_ai=args.exclude_ai or None
    )
    
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
    
    major_agg_df = aggregate_by_major_category(metrics_df)
    minor_agg_df = aggregate_by_minor_category(metrics_df)
    
    save_tables(metrics_df, major_agg_df, minor_agg_df, eigen_df, out)
    
    _print_block("Generating plots")
    plot_per_layer_metrics(metrics_df, out, max_layers=args.max_layers_plot)
    plot_metrics_heatmap(metrics_df, out, max_layers=args.max_layers_plot)
    plot_category_distributions(metrics_df, out)
    plot_major_category_summary(major_agg_df, out)
    plot_scatter_density_clustering(metrics_df, out)
    plot_top_eigen_per_category(eigen_df, out)
    plot_lcc_heatmap_hierarchical(metrics_df, out)
    plot_degree_distributions(multiplex_dir, metrics_df, categories_df, out)
    plot_drg_hierarchy_summary(metrics_df, out)
    
    _print_block("DONE")
    print(f"  Working dir           : {wdir}")
    print(f"  Output dir            : {out.resolve()}")
    print(f"  Layers analysed       : {len(metrics_df):,}")
    print(f"  Major categories      : {metrics_df['major_category'].nunique() - 1}")
    print(f"  Minor categories      : {metrics_df['minor_category'].nunique() - 1}")
    print()
    print("  Output files:")
    for fp in sorted(out.glob("*")):
        if fp.is_file():
            size_kb = fp.stat().st_size / 1024
            print(f"    {fp.name:<55} {size_kb:>8.1f} KB")

if __name__ == "__main__":
    main()