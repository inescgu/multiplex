#!/usr/bin/env python3
"""
Step 7 (replaces step10_layer_stats.py / multiplex_layer_topology_analysis.py):
per-cell-type (intra-layer) topology metrics with DRG neuron hierarchy grouping.

What changed vs. the old version
-----------------------------------
The old version scanned a flat multiplex/gene_multiplex/*.tsv directory and
matched filenames against a CSV-derived "<name>_ppi_layer" stem, with a
SPECIAL_STEMS set to skip the shared ppi_layer/pathway_layer/
ligand_receptor_layer files that didn't belong to any single cell type.

In the new heterogeneous structure there's no flat shared directory and no
special files to skip: every cell type has exactly one file, at
multiplex/<cell_type>/ppi_layer.tsv, and the folder name IS the cell type
name directly — no stem suffix, no CSV-driven name translation needed to
find it. Discovery is just `multiplex/*/ppi_layer.tsv`.

Everything else — per-layer density/clustering/LCC/degree/eigen-centrality,
the DRG hierarchy grouping and every original plot — is unchanged, since
it's all intra-cell-type structure and doesn't care how cell types relate
to each other (that's step 7c's job).

NEW: cross-layer comparison, folded in from a separate draft script
---------------------------------------------------------------------
A separate exploratory script (step7_layer_comparison.py) computed
pairwise layer-vs-layer PPI similarity, a union-of-all-layers graph with
per-gene/per-edge layer counts, cross-layer "hub gene" flagging, and
per-layer network pictures. All four are intra-layer PPI questions —
"how similar is this layer's own PPI structure to that one's", not
anything about bipartite/LR/Reactome connectivity — so they belong here
rather than in step 7c (which is specifically about inter-cell-type
bipartite edges and already covers gene-overlap-across-cell-types on its
own terms). Folding them in here also means zero extra I/O:
analyse_all_layers() was already parsing every multiplex/<ct>/ppi_layer.tsv
into an nx.Graph and then discarding it after computing that layer's own
metrics; it now hangs onto those graphs (`layer_graphs`) so the new
section below can reuse them directly.

  - Pairwise similarity   : node Jaccard/overlap coefficient, edge Jaccard/
                            overlap coefficient, and a Spearman correlation
                            of degree for genes shared between two layers
                            (do the same genes play a similarly connected
                            role in both layers, or is the overlap
                            structurally coincidental?).
  - Union graph           : every layer's PPI edges merged into one
                            nx.Graph, each node/edge tagged with which
                            layers it came from. Saved as union_ppi_graph.pkl.
  - Cross-layer hub genes : genes present in >= --hub-threshold-frac of
                            layers, ranked by how connected they are on
                            average where they do appear.
  - Network pictures      : one spring-layout PNG per cell type, capped to
                            the --max-plot-nodes highest-degree genes so
                            it stays legible. Off by default
                            (--network-plots to enable) — step 7c's own
                            docstring already flags this exact plot type
                            as "nice to look at, lowest analytical value
                            per line of code", so it's opt-in here too.
"""
import argparse
import pickle
import warnings
from itertools import combinations
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

DIVIDER = "=" * 72
SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# DRG NEURON HIERARCHY DEFINITION (same constant used by step 7c — keep in
# sync if you edit cell-type names; consider centralizing in a shared module
# if you maintain both)
# ---------------------------------------------------------------------------
DRG_HIERARCHY = {
    "A-fiber": {"color": "#1f77b4", "subcategories": {
        "Proprioceptors": {"marker": "o", "cell_types": ["A-Propr.EPHA3", "A-Propr.HAPLN4", "A-Propr.PCDH8"]},
        "A-beta LTMR": {"marker": "s", "cell_types": ["Ab-LTMR.CCKAR", "Ab-LTMR.ETV1", "Ab-LTMR.LGI2", "Ab-LTMR.NSG2"]},
        "A-LTMR": {"marker": "^", "cell_types": ["A-LTMR.TAC3"]},
        "A-peptidergic": {"marker": "D", "cell_types": ["A-PEP.CHRNA7_SLC18A3", "A-PEP.KIT", "A-PEP.NTRK3_S100A16", "A-PEP.SCGN_ADRA2C"]},
    }},
    "C-fiber": {"color": "#ff7f0e", "subcategories": {
        "C-LTMR": {"marker": "o", "cell_types": ["C-LTMR.CDH9"]},
        "C-peptidergic": {"marker": "D", "cell_types": ["C-PEP.ADORA2B", "C-PEP.TAC1_CACNG5", "C-PEP.TAC1_CHRNA3"]},
        "MRGPRX nociceptors": {"marker": "s", "cell_types": ["C-NP.MRGPRX1_GFRA2", "C-NP.MRGPRX1_MRGPRX4"]},
        "C-thermosensitive": {"marker": "*", "cell_types": ["C-Thermo.RXFP1", "C-Thermo.TRPM8"]},
        "C-SST nociceptor": {"marker": "v", "cell_types": ["C-NP.SST"]},
    }},
    "ATF3": {"color": "#d62728", "subcategories": {
        "ATF3 (injured)": {"marker": "x", "cell_types": ["ATF3"]},
    }},
    "Non-neuronal": {"color": "#2ca02c", "subcategories": {
        "Immune": {"marker": "o", "cell_types": [
            "B cell", "CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell",
            "activated CD4-positive, alpha-beta T cell", "activated CD8-positive, alpha-beta T cell",
            "basophil", "classical monocyte", "gamma-delta T cell", "granulocyte",
            "innate lymphoid cell", "intermediate monocyte", "leukocyte", "macrophage",
            "mast cell", "mature NK T cell", "myeloid cell", "myeloid dendritic cell",
            "myeloid leukocyte", "naive thymus-derived CD4-positive, alpha-beta T cell",
            "naive thymus-derived CD8-positive, alpha-beta T cell", "natural killer cell",
            "neutrophil", "non-classical monocyte", "plasma cell",
            "plasmacytoid dendritic cell", "regulatory T cell",
        ]},
        "Glial/Nervous": {"marker": "s", "cell_types": ["Schwann cell", "enteroglial cell", "microglial cell", "neuron"]},
        "Immune/Hematopoietic": {"marker": "^", "cell_types": [
            "Langerhans cell", "colon macrophage", "common myeloid progenitor",
            "erythroid progenitor cell", "hematopoietic cell", "hematopoietic precursor cell",
            "hematopoietic stem cell", "tissue-resident macrophage",
        ]},
    }},
}

CELL_TYPE_TO_CATEGORY = {}
for major_cat, major_data in DRG_HIERARCHY.items():
    for subcat, subcat_data in major_data["subcategories"].items():
        for cell_type in subcat_data["cell_types"]:
            CELL_TYPE_TO_CATEGORY[cell_type] = {
                "major": major_cat, "minor": subcat,
                "color": major_data["color"], "marker": subcat_data["marker"],
            }

COL_CELL_TYPE = "Cell Type"
COL_BROAD = "Broad Category - ai"
COL_AI = "category"


def _print_block(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def get_category_from_cell_type(cell_type: str) -> dict:
    if cell_type in CELL_TYPE_TO_CATEGORY:
        return CELL_TYPE_TO_CATEGORY[cell_type]
    return {"major": "unknown", "minor": "unknown", "color": "#808080", "marker": "."}


def load_categories(csv_path: Path, exclude_ai=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0].strip() in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]
    if COL_CELL_TYPE not in df.columns:
        raise ValueError(f"layers_categories.csv missing '{COL_CELL_TYPE}' column. Found: {list(df.columns)}")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    before = len(df)
    if exclude_ai and COL_AI in df.columns:
        mask = df[COL_AI].str.lower().isin([e.lower() for e in exclude_ai])
        df = df[~mask]
        print(f"Excluded {before - len(df)} rows by '{COL_AI}' ({exclude_ai})")
    print(f"Retained {len(df)} cell types")
    return df.reset_index(drop=True)


def load_graph_from_tsv(path: Path) -> nx.Graph:
    G = nx.Graph()
    if not path.exists():
        return G
    try:
        df = pd.read_csv(path, sep="\t", header=None, names=["g1", "g2"], dtype=str, on_bad_lines="skip")
    except Exception:
        return G
    for a, b in zip(df["g1"], df["g2"]):
        a, b = str(a).strip().upper(), str(b).strip().upper()
        if a and b and a != "NAN" and b != "NAN" and a != b:
            G.add_edge(a, b)
    return G


# ---------------------------------------------------------------------------
# Core metric computation (unchanged)
# ---------------------------------------------------------------------------
def compute_layer_metrics(G: nx.Graph, layer_name: str, top_n_eigen: int = 10) -> dict:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        return {"layer": layer_name, "n_nodes": 0, "n_edges": 0, "density": 0.0,
                "avg_clustering": 0.0, "lcc_size": 0, "lcc_fraction": 0.0,
                "degree_mean": 0.0, "degree_median": 0.0, "degree_max": 0,
                "top_eigen_nodes": [], "top_eigen_scores": [], "is_connected": False, "n_components": 0}

    density = nx.density(G)
    avg_clust = nx.average_clustering(G)
    components = list(nx.connected_components(G))
    lcc_size = max(len(c) for c in components)
    lcc_frac = lcc_size / n_nodes
    n_comp = len(components)
    is_conn = (n_comp == 1)
    degrees = [d for _, d in G.degree()]
    deg_mean, deg_median, deg_max = float(np.mean(degrees)), float(np.median(degrees)), int(np.max(degrees))

    lcc_nodes = max(components, key=len)
    G_lcc = G.subgraph(lcc_nodes).copy()
    try:
        eigen_cent = nx.eigenvector_centrality_numpy(G_lcc, weight=None)
    except Exception:
        eigen_cent = nx.degree_centrality(G_lcc)

    top_pairs = sorted(eigen_cent.items(), key=lambda x: x[1], reverse=True)[:top_n_eigen]
    return {
        "layer": layer_name, "n_nodes": n_nodes, "n_edges": n_edges,
        "density": round(density, 6), "avg_clustering": round(avg_clust, 6),
        "lcc_size": lcc_size, "lcc_fraction": round(lcc_frac, 6),
        "degree_mean": round(deg_mean, 4), "degree_median": round(deg_median, 4), "degree_max": deg_max,
        "top_eigen_nodes": [p[0] for p in top_pairs], "top_eigen_scores": [round(p[1], 6) for p in top_pairs],
        "is_connected": is_conn, "n_components": n_comp,
    }


# ---------------------------------------------------------------------------
# Build full metrics table — discovery changed, everything else the same.
# NEW: also returns layer_graphs (cell_type -> nx.Graph) so the cross-layer
# section below doesn't need to re-parse every tsv a second time.
# ---------------------------------------------------------------------------
def analyse_all_layers(wdir: Path, categories_df: pd.DataFrame, top_n_eigen: int, min_edges: int):
    _print_block("Computing per-cell-type topology metrics")

    multiplex_root = wdir / "multiplex"
    ct_dirs = sorted(p for p in multiplex_root.iterdir() if p.is_dir())
    print(f"  Cell-type multiplex folders found: {len(ct_dirs)}")

    def _meta(cell_type: str) -> dict:
        cat_info = get_category_from_cell_type(cell_type)
        broad = "unknown"
        if categories_df is not None and COL_BROAD in categories_df.columns:
            match = categories_df.loc[categories_df[COL_CELL_TYPE] == cell_type, COL_BROAD]
            if len(match):
                broad = match.iloc[0]
        return {"cell_type": cell_type, "major_category": cat_info["major"], "minor_category": cat_info["minor"],
                "color": cat_info["color"], "marker": cat_info["marker"], "broad": broad}

    metrics_rows, eigen_rows, skipped = [], [], 0
    layer_graphs: dict[str, nx.Graph] = {}  # NEW

    for ct_dir in ct_dirs:
        cell_type = ct_dir.name
        tsv_path = ct_dir / "ppi_layer.tsv"
        G = load_graph_from_tsv(tsv_path)
        if G.number_of_edges() < min_edges:
            skipped += 1
            continue

        layer_graphs[cell_type] = G  # NEW: keep it around for cross-layer analysis below

        meta = _meta(cell_type)
        m = compute_layer_metrics(G, cell_type, top_n_eigen=top_n_eigen)
        m.update({k: meta[k] for k in ("cell_type", "major_category", "minor_category", "color", "marker", "broad")})
        metrics_rows.append({k: v for k, v in m.items() if k not in ("top_eigen_nodes", "top_eigen_scores")})

        for gene, score in zip(m["top_eigen_nodes"], m["top_eigen_scores"]):
            eigen_rows.append({
                "layer": cell_type, "cell_type": cell_type,
                "major_category": meta["major_category"], "minor_category": meta["minor_category"],
                "gene": gene, "eigen_score": score, "eigen_rank": m["top_eigen_nodes"].index(gene) + 1,
            })

        print(f"  {cell_type:<45} N={m['n_nodes']:>6,}  E={m['n_edges']:>7,}  "
              f"dens={m['density']:.5f}  clust={m['avg_clustering']:.4f}  LCC={m['lcc_fraction']:.3f}  "
              f"[{meta['major_category']}:{meta['minor_category']}]")

    print(f"\n  Cell types analysed : {len(metrics_rows)}")
    print(f"  Cell types skipped  : {skipped}  (< {min_edges} edges)")
    return pd.DataFrame(metrics_rows), pd.DataFrame(eigen_rows), layer_graphs


# ---------------------------------------------------------------------------
# Category-level aggregation (unchanged)
# ---------------------------------------------------------------------------
def aggregate_by_major_category(metrics_df: pd.DataFrame) -> pd.DataFrame:
    _print_block("Aggregating metrics by major category")
    agg = (metrics_df.groupby("major_category").agg(
        n_layers=("layer", "count"), total_nodes=("n_nodes", "sum"), total_edges=("n_edges", "sum"),
        mean_nodes=("n_nodes", "mean"), mean_edges=("n_edges", "mean"), mean_density=("density", "mean"),
        mean_clustering=("avg_clustering", "mean"), mean_lcc_frac=("lcc_fraction", "mean"),
        mean_deg_mean=("degree_mean", "mean"), mean_deg_max=("degree_max", "mean"),
        pct_connected=("is_connected", lambda x: 100 * x.sum() / len(x)),
    ).reset_index().sort_values("total_edges", ascending=False))
    for col in ("mean_nodes", "mean_edges", "mean_density", "mean_clustering", "mean_lcc_frac", "mean_deg_mean", "mean_deg_max"):
        agg[col] = agg[col].round(4)
    print(agg.to_string(index=False))
    return agg


def aggregate_by_minor_category(metrics_df: pd.DataFrame) -> pd.DataFrame:
    _print_block("Aggregating metrics by minor category")
    agg = (metrics_df.groupby(["major_category", "minor_category"]).agg(
        n_layers=("layer", "count"), total_nodes=("n_nodes", "sum"), total_edges=("n_edges", "sum"),
        mean_density=("density", "mean"), mean_clustering=("avg_clustering", "mean"), mean_lcc_frac=("lcc_fraction", "mean"),
    ).reset_index().sort_values(["major_category", "total_edges"], ascending=[True, False]))
    for col in ("mean_density", "mean_clustering", "mean_lcc_frac"):
        agg[col] = agg[col].round(4)
    print(agg.to_string(index=False))
    return agg


# ---------------------------------------------------------------------------
# Plotting (unchanged from old version)
# ---------------------------------------------------------------------------
METRIC_LABELS = {
    "n_nodes": "Node count", "n_edges": "Edge count", "density": "Density",
    "avg_clustering": "Avg clustering coeff.", "lcc_fraction": "LCC fraction",
    "lcc_size": "LCC size (nodes)", "degree_mean": "Mean degree", "degree_max": "Max degree",
}


def _major_category_palette() -> dict:
    return {"A-fiber": "#1f77b4", "C-fiber": "#ff7f0e", "Non-classified": "#d62728",
            "Non-neuronal": "#2ca02c", "unknown": "#999999"}


def plot_per_layer_metrics(metrics_df, out, max_layers=80):
    _print_block("Plotting per-cell-type metric bar charts")
    df = metrics_df.sort_values("n_edges", ascending=False).head(max_layers)
    if df.empty:
        print("  ⚠  No data, skipping"); return
    palette = _major_category_palette()
    colors = [palette.get(c, "#999999") for c in df["major_category"]]
    y_labels = df["cell_type"].tolist()
    metrics_to_plot = list(METRIC_LABELS.keys())
    n_m = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_m, figsize=(n_m * 3.8, max(6, len(df) * 0.22)), sharey=True)
    for idx, (ax, metric) in enumerate(zip(axes, metrics_to_plot)):
        vals = df[metric].tolist()
        ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.3)
        ax.invert_yaxis()
        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=10)
        if idx == 0:
            ax.set_yticks(range(len(y_labels))); ax.set_yticklabels(y_labels, fontsize=10)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    present = set(df["major_category"].unique())
    patches = [mpatches.Patch(color=c, label=cat) for cat, c in sorted(palette.items()) if cat in present]
    axes[-1].legend(handles=patches, loc="upper right", fontsize=10, title="Major Category")
    fig.suptitle(f"Per-cell-type topology metrics (top {max_layers} by edge count)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"01_bar_per_layer_metrics.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {fp}")
    plt.close(fig)


def plot_metrics_heatmap(metrics_df, out, max_layers=60):
    _print_block("Plotting metrics heatmap (cell types x metrics)")
    df = metrics_df.sort_values("n_edges", ascending=False).head(max_layers).copy()
    if df.empty:
        print("  ⚠  No data, skipping"); return
    scalar_cols = list(METRIC_LABELS.keys())
    mat = df.set_index("cell_type")[scalar_cols].astype(float)
    std = mat.std(); std[std == 0] = 1
    mat_norm = (mat - mat.mean()) / (std + 1e-12)
    palette = _major_category_palette()
    row_colors = [palette.get(c, "#999999") for c in df["major_category"].values]
    fig_h, fig_w = max(8, len(mat_norm) * 0.25), max(10, len(scalar_cols) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    try:
        sns.heatmap(mat_norm, ax=ax, cmap="RdYlBu_r", center=0, linewidths=0.2, linecolor="white",
                    cbar_kws={"label": "Z-score", "shrink": 0.6},
                    xticklabels=[METRIC_LABELS[c] for c in scalar_cols], yticklabels=True)
    except ValueError as e:
        print(f"  ⚠  Heatmap failed: {e}"); plt.close(fig); return
    for i, color in enumerate(row_colors):
        ax.add_patch(plt.Rectangle((-0.15, i), 0.1, 1, facecolor=color, edgecolor="none"))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(f"Cell-type topology metrics (Z-scored, top {max_layers})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"02_heatmap_layer_metrics.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {fp}")
    plt.close(fig)


def plot_category_distributions(metrics_df, out):
    _print_block("Plotting category-level metric distributions")
    if metrics_df.empty:
        print("  ⚠  No data, skipping"); return
    order = metrics_df.groupby("major_category")["n_edges"].mean().sort_values(ascending=False).index.tolist()
    # seaborn needs every category in the palette (e.g. "ATF3", the injured-neuron type)
    palette = {c: _major_category_palette().get(c, "#999999") for c in order}
    metrics_to_plot = [("n_nodes", "Node count"), ("n_edges", "Edge count"), ("density", "Density"),
                       ("avg_clustering", "Avg clustering"), ("lcc_fraction", "LCC fraction"), ("degree_mean", "Mean degree")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11)); axes = axes.flatten()
    for ax, (metric, label) in zip(axes, metrics_to_plot):
        sns.violinplot(data=metrics_df, x="major_category", y=metric, order=order, palette=palette, ax=ax, inner="box", cut=0, linewidth=0.8)
        sns.stripplot(data=metrics_df, x="major_category", y=metric, order=order, color="black", size=2.5, alpha=0.5, ax=ax, jitter=True)
        ax.set_title(label, fontsize=10, fontweight="bold"); ax.tick_params(axis="x", rotation=25, labelsize=9)
    fig.suptitle("Topology metric distributions per major DRG category", fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"03_violin_category_metrics.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {fp}")
    plt.close(fig)


def plot_major_category_summary(cat_agg_df, out):
    _print_block("Plotting major category summary bars")
    if cat_agg_df.empty:
        print("  ⚠  No data, skipping"); return
    palette = _major_category_palette()
    metrics_to_plot = [("mean_nodes", "Mean node count"), ("mean_edges", "Mean edge count"),
                       ("mean_density", "Mean density"), ("mean_clustering", "Mean clustering"),
                       ("mean_lcc_frac", "Mean LCC fraction"), ("pct_connected", "% fully connected")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.flatten()
    for ax, (col, label) in zip(axes, metrics_to_plot):
        sdf = cat_agg_df.sort_values(col, ascending=False)
        colors = [palette.get(c, "#999999") for c in sdf["major_category"]]
        bars = ax.bar(sdf["major_category"], sdf[col], color=colors, edgecolor="white")
        ax.set_title(label, fontsize=10, fontweight="bold"); ax.tick_params(axis="x", rotation=25, labelsize=9)
        for bar, val in zip(bars, sdf[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_height() * 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Major category-level mean topology metrics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"04_bar_major_category_summary.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {fp}")
    plt.close(fig)


def plot_scatter_density_clustering(metrics_df, out):
    _print_block("Plotting scatter plots")
    if metrics_df.empty:
        print("  ⚠  No data, skipping"); return
    palette = _major_category_palette()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pairs = [("density", "avg_clustering", "Density", "Avg clustering"),
             ("n_nodes", "n_edges", "Node count", "Edge count"),
             ("lcc_fraction", "avg_clustering", "LCC fraction", "Avg clustering")]
    for ax, (xc, yc, xl, yl) in zip(axes, pairs):
        for cat in sorted(metrics_df["major_category"].unique()):
            grp = metrics_df[metrics_df["major_category"] == cat]
            ax.scatter(grp[xc], grp[yc], label=cat, color=palette.get(cat, "#999999"), s=50, alpha=0.75, edgecolors="white")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{xl} vs {yl}", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.08), title="Major DRG Category")
    fig.suptitle("Cell-type topology scatter plots", fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"05_scatter_layer_topology.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {fp}")
    plt.close(fig)


def plot_top_eigen_per_category(eigen_df, out, top_n_genes=15):
    _print_block("Plotting top eigen-centrality genes per major category")
    if eigen_df.empty:
        print("  ⚠  No eigen data, skipping"); return
    cat_gene_agg = eigen_df.groupby(["major_category", "gene"]).agg(mean_eigen=("eigen_score", "mean"), n_layers=("layer", "count")).reset_index()
    categories = sorted(cat_gene_agg["major_category"].unique())
    if not categories:
        print("  ⚠  No categories, skipping"); return
    palette = _major_category_palette()
    n_cols = min(2, len(categories)); n_rows = int(np.ceil(len(categories) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes_flat = np.array(axes).flatten()
    for i, cat in enumerate(categories):
        ax = axes_flat[i]
        sub = cat_gene_agg[cat_gene_agg["major_category"] == cat].sort_values("mean_eigen", ascending=False).head(top_n_genes)
        ax.barh(sub["gene"], sub["mean_eigen"], color=palette.get(cat, "#999999"), edgecolor="white")
        ax.invert_yaxis(); ax.set_title(f"{cat}\n({len(sub)} top genes)", fontweight="bold", color=palette.get(cat))
    for j in range(len(categories), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"Top {top_n_genes} eigen-central genes per major DRG category", fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"06_bar_top_eigen_per_category.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {fp}")
    plt.close(fig)


def plot_lcc_heatmap_hierarchical(metrics_df, out):
    _print_block("Plotting LCC heatmap (major x minor category)")
    if metrics_df.empty:
        print("  ⚠  No data, skipping"); return
    pivot = (metrics_df.groupby(["major_category", "minor_category"]).agg(mean_lcc=("lcc_fraction", "mean"))
             .reset_index().pivot(index="major_category", columns="minor_category", values="mean_lcc").fillna(0))
    if pivot.empty:
        print("  ⚠  Not enough data, skipping"); return
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2), max(6, len(pivot) * 0.7)))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f", linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Mean LCC fraction"}, vmin=0, vmax=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_title("Mean LCC fraction — DRG hierarchy (major x minor category)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"07_heatmap_lcc_hierarchical.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {fp}")
    plt.close(fig)


def save_tables(metrics_df, major_agg_df, minor_agg_df, eigen_df, out):
    _print_block("Saving output tables")
    metrics_df.to_csv(out / "01_table_layer_metrics.csv", index=False)
    print(f"  -> 01_table_layer_metrics.csv ({len(metrics_df):,} rows)")
    major_agg_df.to_csv(out / "02_table_major_category_summary.csv", index=False)
    print(f"  -> 02_table_major_category_summary.csv ({len(major_agg_df):,} rows)")
    minor_agg_df.to_csv(out / "03_table_minor_category_summary.csv", index=False)
    print(f"  -> 03_table_minor_category_summary.csv ({len(minor_agg_df):,} rows)")
    if not eigen_df.empty:
        eigen_df.to_csv(out / "04_table_top_eigen_genes.csv", index=False)
        print(f"  -> 04_table_top_eigen_genes.csv ({len(eigen_df):,} rows)")


# ============================================================================
# NEW: cross-layer comparison -- pairwise similarity, union graph, hub genes,
# per-layer network pictures. See module docstring for why this lives here.
# ============================================================================
def compute_pairwise_similarity(layer_graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    rows = []
    cts = sorted(layer_graphs.keys())
    for ct_a, ct_b in combinations(cts, 2):
        Ga, Gb = layer_graphs[ct_a], layer_graphs[ct_b]
        na, nb = set(Ga.nodes()), set(Gb.nodes())
        ea = {frozenset(e) for e in Ga.edges()}
        eb = {frozenset(e) for e in Gb.edges()}

        node_union = na | nb
        node_jaccard = len(na & nb) / len(node_union) if node_union else 0.0
        node_overlap = len(na & nb) / min(len(na), len(nb)) if min(len(na), len(nb)) else 0.0

        edge_union = ea | eb
        edge_jaccard = len(ea & eb) / len(edge_union) if edge_union else 0.0
        edge_overlap = len(ea & eb) / min(len(ea), len(eb)) if min(len(ea), len(eb)) else 0.0

        shared = na & nb
        spearman = np.nan
        if len(shared) >= 3:
            da = np.array([Ga.degree(g) for g in shared], dtype=float)
            db = np.array([Gb.degree(g) for g in shared], dtype=float)
            if da.std() > 0 and db.std() > 0:
                ra, rb = pd.Series(da).rank(), pd.Series(db).rank()
                spearman = float(np.corrcoef(ra, rb)[0, 1])

        for metric, value in [
            ("node_jaccard", node_jaccard), ("node_overlap_coef", node_overlap),
            ("edge_jaccard", edge_jaccard), ("edge_overlap_coef", edge_overlap),
            ("shared_degree_spearman", spearman),
        ]:
            rows.append({"cell_type_a": ct_a, "cell_type_b": ct_b, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def plot_similarity_heatmap(sim_df: pd.DataFrame, metric: str, cts: list[str], out: Path):
    if sim_df.empty:
        print(f"  ⚠  No similarity data for '{metric}', skipping"); return
    sub = sim_df[sim_df["metric"] == metric]
    n = len(cts)
    idx = {ct: i for i, ct in enumerate(cts)}
    M = np.full((n, n), np.nan)
    np.fill_diagonal(M, 1.0 if ("jaccard" in metric or "overlap" in metric) else np.nan)
    for row in sub.itertuples(index=False):
        i, j = idx[row.cell_type_a], idx[row.cell_type_b]
        M[i, j] = row.value
        M[j, i] = row.value
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n), max(6, 0.35 * n)))
    finite = M[np.isfinite(M)]
    im = ax.imshow(M, cmap="viridis", vmin=finite.min() if finite.size else 0, vmax=finite.max() if finite.size else 1)
    ax.set_xticks(range(n)); ax.set_xticklabels(cts, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(cts, fontsize=6)
    ax.set_title(f"Layer similarity: {metric}", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"08_heatmap_similarity_{metric}.{ext}"
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {fp}")
    plt.close(fig)


def compute_union_layercounts(layer_graphs: dict[str, nx.Graph]):
    """Returns (node_df, edge_df, union_graph). union_graph is every layer's
    PPI edges merged into one nx.Graph, with node/edge attributes n_layers
    (int) and layers (sorted list of cell types)."""
    node_layers: dict[str, set] = defaultdict(set)
    edge_layers: dict[frozenset, set] = defaultdict(set)
    for ct, G in layer_graphs.items():
        for g in G.nodes():
            node_layers[g].add(ct)
        for u, v in G.edges():
            edge_layers[frozenset((u, v))].add(ct)

    node_rows = [{"gene": g, "n_layers": len(cts), "layers": ";".join(sorted(cts))}
                 for g, cts in node_layers.items()]
    node_df = pd.DataFrame(node_rows).sort_values("n_layers", ascending=False)

    edge_rows = []
    union_graph = nx.Graph()
    for e, cts in edge_layers.items():
        g1, g2 = tuple(e)
        edge_rows.append({"g1": g1, "g2": g2, "n_layers": len(cts), "layers": ";".join(sorted(cts))})
        union_graph.add_edge(g1, g2, n_layers=len(cts), layers=sorted(cts))
    for g, cts in node_layers.items():
        if g in union_graph:
            union_graph.nodes[g]["n_layers"] = len(cts)
            union_graph.nodes[g]["layers"] = sorted(cts)
        else:
            union_graph.add_node(g, n_layers=len(cts), layers=sorted(cts))
    edge_df = pd.DataFrame(edge_rows).sort_values("n_layers", ascending=False)

    return node_df, edge_df, union_graph


def hub_genes_across_layers(node_df: pd.DataFrame, layer_graphs: dict[str, nx.Graph], threshold_frac: float) -> pd.DataFrame:
    n_layers_total = len(layer_graphs)
    cutoff = max(2, int(np.ceil(threshold_frac * n_layers_total)))
    hubs = node_df[node_df["n_layers"] >= cutoff].copy()
    avg_degrees = []
    for row in hubs.itertuples(index=False):
        cts = row.layers.split(";")
        degs = [layer_graphs[ct].degree(row.gene) for ct in cts if row.gene in layer_graphs[ct]]
        avg_degrees.append(float(np.mean(degs)) if degs else 0.0)
    hubs["avg_degree_across_present_layers"] = avg_degrees
    return hubs.sort_values(["n_layers", "avg_degree_across_present_layers"], ascending=False)


def plot_layer_network(cell_type: str, G: nx.Graph, max_nodes: int, out: Path):
    if G.number_of_nodes() == 0:
        return
    H = G
    if H.number_of_nodes() > max_nodes:
        top_nodes = [n for n, _ in sorted(H.degree(), key=lambda x: -x[1])[:max_nodes]]
        H = H.subgraph(top_nodes).copy()
    if H.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(H, seed=1, k=1.5 / max(np.sqrt(H.number_of_nodes()), 1))
    degrees = dict(H.degree())
    sizes = [30 + 15 * degrees[n] for n in H.nodes()]
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, width=0.5)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=sizes, node_color="#3b6fa0", alpha=0.85)
    top_labels = {n: n for n, _ in sorted(degrees.items(), key=lambda x: -x[1])[:15]}
    nx.draw_networkx_labels(H, pos, labels=top_labels, ax=ax, font_size=7)
    ax.set_title(f"{cell_type} (top {H.number_of_nodes()} nodes by degree)", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fp = out / f"network_{cell_type}.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Per-cell-type topology metrics with DRG hierarchy grouping (heterogeneous structure)")
    ap.add_argument("--wdir", required=True, help="MultiXrank wdir (contains multiplex/<cell_type>/ppi_layer.tsv)")
    ap.add_argument("--csv", default=None, help="Optional CSV with Cell Type / Broad Category - ai / category columns")
    ap.add_argument("--output-dir", default="topology_output")
    ap.add_argument("--exclude-ai", nargs="+", default=[])
    ap.add_argument("--top-n-eigen", type=int, default=10)
    ap.add_argument("--min-edges", type=int, default=1)
    ap.add_argument("--max-layers-plot", type=int, default=45)
    ap.add_argument("--no-pairwise-similarity", action="store_true",
                     help="Skip the cross-layer similarity / union-graph / hub-gene section. On by default "
                          "since it's cheap -- the layer graphs are already loaded for the metrics above.")
    ap.add_argument("--hub-threshold-frac", type=float, default=0.5,
                     help="A gene counts as a cross-layer hub if present in >= this fraction of layers (default 0.5)")
    ap.add_argument("--network-plots", action="store_true",
                     help="Also draw a spring-layout network picture per cell type. Off by default -- step 7c's "
                          "own docstring flags this plot type as low analytical value per line of code; opt in "
                          "with this flag if you want them anyway.")
    ap.add_argument("--max-plot-nodes", type=int, default=150,
                     help="Cap per-layer network pictures to this many highest-degree nodes, for legibility")
    args = ap.parse_args()

    wdir = Path(args.wdir).resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    categories_df = None
    if args.csv:
        _print_block("Loading input files")
        categories_df = load_categories(Path(args.csv), exclude_ai=args.exclude_ai or None)

    metrics_df, eigen_df, layer_graphs = analyse_all_layers(wdir, categories_df, args.top_n_eigen, args.min_edges)

    if metrics_df.empty:
        print("\n⚠  No cell types found/analysed — check --wdir path.")
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

    if not args.no_pairwise_similarity:
        _print_block("Cross-layer comparison: pairwise similarity, union graph, hub genes")
        sim_df = compute_pairwise_similarity(layer_graphs)
        sim_path = out / "05_table_pairwise_similarity.csv"
        sim_df.to_csv(sim_path, index=False)
        print(f"  -> {sim_path.name} ({len(sim_df):,} rows)")

        cts_sorted = sorted(layer_graphs.keys())
        for metric in ["node_jaccard", "node_overlap_coef", "edge_jaccard", "edge_overlap_coef", "shared_degree_spearman"]:
            plot_similarity_heatmap(sim_df, metric, cts_sorted, out)

        node_df, edge_df, union_graph = compute_union_layercounts(layer_graphs)
        node_df.to_csv(out / "06_table_union_node_layercounts.csv", index=False)
        edge_df.to_csv(out / "07_table_union_edge_layercounts.csv", index=False)
        print(f"  -> 06_table_union_node_layercounts.csv ({len(node_df):,} rows)")
        print(f"  -> 07_table_union_edge_layercounts.csv ({len(edge_df):,} rows)")

        union_pkl_path = out / "union_ppi_graph.pkl"
        with open(union_pkl_path, "wb") as f:
            pickle.dump(union_graph, f, protocol=4)
        print(f"  -> {union_pkl_path.name} ({union_graph.number_of_nodes():,} nodes, "
              f"{union_graph.number_of_edges():,} edges -- every layer's PPI merged into one graph, "
              f"each node/edge tagged with n_layers/layers)")

        hub_df = hub_genes_across_layers(node_df, layer_graphs, args.hub_threshold_frac)
        hub_df.to_csv(out / "08_table_hub_genes_across_layers.csv", index=False)
        print(f"  -> 08_table_hub_genes_across_layers.csv "
              f"({len(hub_df):,} genes present in >= {args.hub_threshold_frac:.0%} of layers)")

    if args.network_plots:
        _print_block("Plotting per-layer network pictures")
        for ct, G in layer_graphs.items():
            plot_layer_network(ct, G, args.max_plot_nodes, out)

    _print_block("DONE")
    print(f"  wdir             : {wdir}")
    print(f"  Output dir       : {out.resolve()}")
    print(f"  Cell types       : {len(metrics_df):,}")
    print(f"  Major categories : {metrics_df['major_category'].nunique()}")


if __name__ == "__main__":
    main()