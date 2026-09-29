#!/usr/bin/env python3
"""
Step 11f: Visualize the persistent heterogeneous HiDeF clusters characterized
in step 11e, on the "<cell_type>::GENE" namespaced supra-graph from step 11a.

Unlike the single-layer viz scripts this is modeled on, there's no existing
UMAP layout to reuse: the supra-graph is heterogeneous (disjoint-but-coupled
per-cell-type PPI blocks), so this script computes and caches its OWN layout
of the full supra-graph using igraph's DrL algorithm (Martin et al. 2008),
which -- unlike force-directed layouts (Fruchterman-Reingold, spring_layout)
or UMAP built from a single flattened PPI -- is specifically designed to
scale to the large, dense graphs a multi-cell-type supra-graph produces.

Three complementary views are produced, because no single plot tells the
whole story for a heterogeneous graph:
  1. OVERVIEW  -- the full supra-graph, background nodes colored by cell
     type, persistent clusters overlaid in distinct colors (same idea as
     the single-layer "all communities" plot)
  2. COMPOSITION HEATMAP -- cluster x cell-type member-gene counts, so you
     can immediately see which clusters are single-cell-type PPI modules
     vs. genuine cross-cell-type modules, without eyeballing a busy network
  3. PER-CLUSTER ZOOM PANELS -- induced subgraph of each top cluster, laid
     out fresh (cheap at cluster scale), nodes colored by cell type, edges
     styled by etype (intra=solid, bipartite=dashed, identity=dotted) so
     the cross-cell-type coupling structure of that specific cluster is
     visible directly, plus a combined summary sheet

Reads step 11e's output directly (cluster_summary.csv, cluster_details.json,
celltype_participation.csv) rather than re-deriving anything, so run 11e
first.

USAGE
-----
    python step11f_visualize_clusters.py \
        --supra-graph 11_hidef_hetero/supra_graph.pkl \
        --char-prefix 11_hidef_results_faithful/characterized/heterogeneous_hidef \
        --output-dir plots/hetero_hidef \
        --top-n 40 \
        --min-persistence-pct 75
"""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from adjustText import adjust_text
    HAVE_ADJUST_TEXT = True
except ImportError:
    HAVE_ADJUST_TEXT = False

EDGE_STYLE = {"intra": "solid", "bipartite": "dashed", "identity": "dotted"}
EDGE_ALPHA = {"intra": 0.45, "bipartite": 0.55, "identity": 0.35}
BG_NODE_ALPHA = 0.35
BG_EDGE_COLOR = "#cccccc"
SEED_MARKER_EC = "gold"


# ============================================================================
# LOADING
# ============================================================================

def load_supra_graph(path: Path) -> ig.Graph:
    with open(path, "rb") as f:
        g = pickle.load(f)
    print(f"Loaded supra-graph: {g.vcount():,} nodes, {g.ecount():,} edges\n")
    return g


def load_characterization(char_prefix: str):
    summary = pd.read_csv(f"{char_prefix}_cluster_summary.csv", dtype={"cluster_id": str})
    with open(f"{char_prefix}_cluster_details.json") as f:
        details = json.load(f)
    participation = pd.read_csv(f"{char_prefix}_celltype_participation.csv", dtype={"cluster_id": str})
    print(f"Loaded characterization for {len(summary):,} clusters\n")
    return summary, details, participation


def load_seed_genes(path):
    if not path:
        return set()
    return set(pd.read_csv(path)["gene"].tolist())


# ============================================================================
# COLORS
# ============================================================================

def make_categorical_colors(keys, colormap="tab20"):
    cmap = plt.get_cmap(colormap)
    n = cmap.N if hasattr(cmap, "N") else 256
    return {k: cmap(i % n) for i, k in enumerate(sorted(keys))}


# ============================================================================
# FULL SUPRA-GRAPH LAYOUT (cached)
# ============================================================================

def get_or_compute_layout(g: ig.Graph, cache_path: Path, algo="drl"):
    names = g.vs["name"]

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            pos = pickle.load(f)
        missing = [n for n in names if n not in pos]
        if not missing:
            print(f"Loaded cached layout ({algo}) for {len(pos):,} nodes from {cache_path}\n")
            return pos
        print(f"Cached layout at {cache_path} is stale: {len(missing):,} of "
              f"{len(names):,} supra-graph nodes are missing (e.g. {missing[:3]}). "
              f"Recomputing ...")

    print(f"Computing {algo} layout for {g.vcount():,} nodes, {g.ecount():,} edges "
          f"(this can take a while) ...")
    if algo == "drl":
        coords = g.layout_drl()
    elif algo == "fr":
        coords = g.layout_fruchterman_reingold()
    elif algo == "kk":
        coords = g.layout_kamada_kawai()
    else:
        raise ValueError(f"Unknown layout algo: {algo}")

    pos = {name: (xy[0], xy[1]) for name, xy in zip(names, coords)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(pos, f)
    print(f"  Computed and cached layout -> {cache_path}\n")
    return pos


# ============================================================================
# PLOT 1 -- OVERVIEW: background by cell type, clusters overlaid
# ============================================================================

def plot_overview(g, pos, summary, details, celltype_colors,
                   min_persistence=None, percentile=None,
                   colormap="tab20", draw_background_edges=False,
                   out_path=Path("overview.png")):
    sel = summary.copy()
    if percentile is not None:
        thresh = np.percentile(sel["persistence"], percentile)
        sel = sel[sel["persistence"] >= thresh]
        pct_label = f"{percentile:.0f}th pct (persist >= {thresh:.1f})"
    elif min_persistence is not None:
        sel = sel[sel["persistence"] >= min_persistence]
        pct_label = f"persist >= {min_persistence}"
    else:
        pct_label = "all clusters"

    cluster_ids = sel["cluster_id"].tolist()
    cluster_colors = make_categorical_colors(cluster_ids, colormap=colormap)
    print(f"\nOverview plot: {len(cluster_ids)} clusters ({pct_label})")

    member_genes = set()
    for cid in cluster_ids:
        for ct, genes in details[cid]["genes_by_celltype"].items():
            member_genes.update(f"{ct}::{gene}" for gene in genes)
    background = set(g.vs["name"]) - member_genes

    fig, ax = plt.subplots(figsize=(20, 20))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if draw_background_edges:
        print("  Drawing background edges (this is the slow part) ...")
        xs, ys = [], []
        for e in g.es:
            u, v = e.tuple
            un, vn = g.vs[u]["name"], g.vs[v]["name"]
            if un in background and vn in background:
                xs += [pos[un][0], pos[vn][0], None]
                ys += [pos[un][1], pos[vn][1], None]
        chunk = 15000
        for s in range(0, len(xs), chunk):
            ax.plot(xs[s:s + chunk], ys[s:s + chunk], color=BG_EDGE_COLOR,
                     alpha=0.03, linewidth=0.15, zorder=0, rasterized=True)

    print("  Drawing background nodes by cell type ...")
    for ct in sorted(set(g.vs["cell_type"])):
        bxs = [pos[n][0] for n in background if n.split("::", 1)[0] == ct]
        bys = [pos[n][1] for n in background if n.split("::", 1)[0] == ct]
        if bxs:
            ax.scatter(bxs, bys, c=[celltype_colors[ct]], s=6, alpha=BG_NODE_ALPHA,
                       linewidths=0, zorder=1, rasterized=True)

    print("  Drawing cluster nodes/edges ...")
    for cid in sorted(cluster_ids, key=lambda c: sel.set_index("cluster_id").loc[c, "persistence"]):
        col = cluster_colors[cid]
        genes = member_genes_for(details[cid])
        xs = [pos[n][0] for n in genes if n in pos]
        ys = [pos[n][1] for n in genes if n in pos]
        ax.scatter(xs, ys, c=[col] * len(xs), s=18, alpha=0.85,
                   linewidths=0.3, edgecolors="white", zorder=3, rasterized=True)

    pers = sel.set_index("cluster_id")["persistence"]
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cluster_colors[cid],
               markersize=9, label=f"Cluster {cid}  persist={pers[cid]:.1f}")
        for cid in sorted(cluster_ids, key=lambda c: pers[c], reverse=True)[:20]
    ]
    if len(cluster_ids) > 20:
        handles.append(Line2D([0], [0], color="none", label=f"... and {len(cluster_ids) - 20} more"))
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=celltype_colors[ct],
               markersize=8, label=f"Background: {ct}")
        for ct in sorted(celltype_colors)
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92,
              edgecolor="#dddddd", fontsize=8, borderpad=0.8, handlelength=1.2)

    ax.set_title(f"Heterogeneous HiDeF clusters -- {pct_label}", fontsize=20, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def member_genes_for(detail):
    return {f"{ct}::{g}" for ct, genes in detail["genes_by_celltype"].items() for g in genes}


# ============================================================================
# PLOT 2 -- CLUSTER x CELL-TYPE COMPOSITION HEATMAP
# ============================================================================

def plot_composition_heatmap(summary, participation, top_n, out_path):
    top = summary.sort_values("persistence", ascending=False).head(top_n)
    ct_cols = [c for c in participation.columns if c != "cluster_id"]
    mat = participation.set_index("cluster_id").reindex(top["cluster_id"]).fillna(0)
    mat = mat[sorted(ct_cols)]

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(ct_cols) + 4), max(8, 0.25 * len(top) + 3)))
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=9)

    row_labels = [f"{cid}  (p={p:.1f})" for cid, p in
                  zip(top["cluster_id"], top["persistence"])]
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(row_labels, fontsize=7)

    ax.set_xlabel("Cell type")
    ax.set_title(f"Top {len(top)} clusters by persistence -- member-gene count per cell type",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="# member genes", shrink=0.7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved composition heatmap -> {out_path}")


# ============================================================================
# PLOT 3 -- PER-CLUSTER ZOOM PANELS
# ============================================================================

def draw_cluster_zoom(ax, g, node_index, detail, celltype_colors, seed_genes,
                       label=True, adjust_kw=None):
    members = member_genes_for(detail)
    idxs = [node_index[m] for m in members if m in node_index]
    sub = g.induced_subgraph(idxs)
    if sub.vcount() == 0:
        ax.axis("off")
        return

    layout = sub.layout_fruchterman_reingold()
    sub_pos = {v["name"]: (xy[0], xy[1]) for v, xy in zip(sub.vs, layout)}

    for e in sub.es:
        u, v = e.tuple
        un, vn = sub.vs[u]["name"], sub.vs[v]["name"]
        etype = e["etype"]
        ax.plot([sub_pos[un][0], sub_pos[vn][0]], [sub_pos[un][1], sub_pos[vn][1]],
                color="#999999", alpha=EDGE_ALPHA.get(etype, 0.4), linewidth=0.8,
                linestyle=EDGE_STYLE.get(etype, "solid"), zorder=1)

    texts = []
    for v in sub.vs:
        ct = v["cell_type"]
        gene = v["gene"]
        x, y = sub_pos[v["name"]]
        is_seed = gene in seed_genes
        ax.scatter([x], [y], c=[celltype_colors[ct]],
                   s=140 if is_seed else 55,
                   marker="*" if is_seed else "o",
                   linewidths=1.2 if is_seed else 0.4,
                   edgecolors=SEED_MARKER_EC if is_seed else "white",
                   alpha=0.95, zorder=3)
        if label:
            texts.append(ax.annotate(gene, xy=(x, y), fontsize=7, zorder=4,
                                      bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                                edgecolor=celltype_colors[ct], alpha=0.85, linewidth=0.7)))

    if label and texts and HAVE_ADJUST_TEXT:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5, alpha=0.6))

    ax.axis("off")


def plot_cluster_zooms(g, summary, details, celltype_colors, seed_genes, top_n, out_dir):
    node_index = {n: i for i, n in enumerate(g.vs["name"])}
    top = summary.sort_values("persistence", ascending=False).head(top_n)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_legend = [Line2D([0], [0], color="#999999", linestyle=style, label=f"{etype} edge")
                   for etype, style in EDGE_STYLE.items()]

    print(f"\nRendering {len(top)} per-cluster zoom panels ...")
    for _, row in top.iterrows():
        cid = row["cluster_id"]
        detail = details[cid]
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor("white")
        draw_cluster_zoom(ax, g, node_index, detail, celltype_colors, seed_genes)

        ct_counts = detail["cell_type_counts"]
        ct_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=celltype_colors[ct],
                             markersize=9, label=f"{ct} (n={n})") for ct, n in ct_counts.items()]
        seed_legend = [Line2D([0], [0], marker="*", color="w", markerfacecolor="#888888",
                               markeredgecolor=SEED_MARKER_EC, markersize=12, label="Seed gene")] \
            if seed_genes else []
        ax.legend(handles=ct_legend + edge_legend + seed_legend, loc="lower left",
                  fontsize=7, frameon=True, framealpha=0.9, edgecolor="#dddddd")

        n_conserved = len(detail["conserved_genes"])
        ax.set_title(f"Cluster {cid}  |  persistence={detail['persistence']:.1f}  |  "
                      f"size={detail['size']}  |  {len(ct_counts)} cell types  |  "
                      f"{n_conserved} conserved genes",
                      fontsize=11, fontweight="bold")
        plt.tight_layout()
        save_path = out_dir / f"cluster_{cid}_zoom.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Saved zoom panels -> {out_dir}/")

    # summary sheet
    ncols = 4
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten() if len(top) > 1 else [axes]
    for ax, (_, row) in zip(axes_flat, top.iterrows()):
        cid = row["cluster_id"]
        detail = details[cid]
        draw_cluster_zoom(ax, g, node_index, detail, celltype_colors, seed_genes,
                           label=(detail["size"] <= 25))
        ax.set_title(f"{cid}  p={detail['persistence']:.1f}  n={detail['size']}", fontsize=8)
    for ax in axes_flat[len(top):]:
        ax.axis("off")
    fig.suptitle(f"Top {len(top)} persistent clusters -- summary sheet", fontsize=14, fontweight="bold")
    plt.tight_layout()
    summary_path = out_dir.parent / "cluster_zoom_summary_sheet.png"
    fig.savefig(summary_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved summary sheet -> {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--supra-graph", required=True, help="Pickle from step 11a")
    ap.add_argument("--char-prefix", required=True,
                     help="Prefix used for step 11e's output "
                          "(<prefix>_cluster_summary.csv etc.)")
    ap.add_argument("--seed-genes", default=None, help="Optional CSV with a 'gene' column")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--layout-cache", default=None,
                     help="Where to cache the full supra-graph layout "
                          "(default: <output-dir>/supra_graph_layout.pkl)")
    ap.add_argument("--layout-algo", choices=["drl", "fr", "kk"], default="drl",
                     help="drl (default) scales to large graphs; fr/kk for smaller graphs")
    ap.add_argument("--top-n", type=int, default=40,
                     help="How many top-persistence clusters to zoom into / label in overview legend")
    ap.add_argument("--min-persistence-pct", type=float, default=None,
                     help="Percentile of persistence to overlay in the overview plot "
                          "(e.g. 75 = top 25%%). Default: --top-n clusters by persistence")
    ap.add_argument("--draw-background-edges", action="store_true",
                     help="Draw faint background edges in the overview (slow on large graphs)")
    ap.add_argument("--skip-overview", action="store_true")
    ap.add_argument("--skip-heatmap", action="store_true")
    ap.add_argument("--skip-zooms", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = load_supra_graph(Path(args.supra_graph))
    summary, details, participation = load_characterization(args.char_prefix)
    seed_genes = load_seed_genes(args.seed_genes)
    celltype_colors = make_categorical_colors(set(g.vs["cell_type"]), colormap="Set2")

    if not args.skip_overview:
        layout_cache = Path(args.layout_cache) if args.layout_cache else out_dir / "supra_graph_layout.pkl"
        pos = get_or_compute_layout(g, layout_cache, algo=args.layout_algo)
        if args.min_persistence_pct is not None:
            plot_overview(g, pos, summary, details, celltype_colors,
                          percentile=args.min_persistence_pct,
                          draw_background_edges=args.draw_background_edges,
                          out_path=out_dir / "overview.png")
        else:
            top_thresh = summary.sort_values("persistence", ascending=False)["persistence"].iloc[
                min(args.top_n, len(summary)) - 1]
            plot_overview(g, pos, summary, details, celltype_colors,
                          min_persistence=top_thresh,
                          draw_background_edges=args.draw_background_edges,
                          out_path=out_dir / "overview.png")

    if not args.skip_heatmap:
        plot_composition_heatmap(summary, participation, args.top_n,
                                  out_dir / "composition_heatmap.png")

    if not args.skip_zooms:
        plot_cluster_zooms(g, summary, details, celltype_colors, seed_genes,
                           args.top_n, out_dir / "cluster_zooms")

    print("\n" + "=" * 60)
    print(f"Done. Outputs in {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()