#!/usr/bin/env python3
"""
05_per_layer_subgraph_view.py

For each (community, layer) pair, restrict the multiplex graph to nodes
belonging to that layer within that community, and compute subgraph metrics.

Consumes:
  - integrated_multiplex.pkl
  - node_registry.csv
  - persistent_communities.pkl

Produces (in --outdir):
  - community_layer_subgraph_metrics.csv   (long format: one row per community x layer)
  - community_layer_subgraph_wide.csv      (wide: community x layer for a chosen metric)
  - plots/
      - subgraph_density_heatmap.png
      - subgraph_connectedness_heatmap.png
      - subgraph_size_vs_density.png
      - per_community_layer_profiles.png   (top-N communities, multi-metric bars)

Metrics per (community, layer) restriction:
  - n_nodes, n_edges
  - density
  - avg_degree
  - avg_clustering
  - n_connected_components
  - largest_cc_size
  - largest_cc_fraction
  - is_connected
  - n_isolates
  - intra_layer_edges  (same layer endpoints — should equal n_edges here)
  - boundary_edges_to_other_layers (edges from this layer's community nodes
                                    to nodes of the SAME community in OTHER layers)

Usage:
  python 05_per_layer_subgraph_view.py \
      --graph integrated_multiplex.pkl \
      --registry node_registry.csv \
      --communities persistent_communities.pkl \
      --outdir multiplex_community_out \
      --top-n 10 \
      --wide-metric density
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("per_layer_subgraph")


# --------------------------------------------------------------------------- #
# I/O (same conventions as 04_)
# --------------------------------------------------------------------------- #
def load_graph(path: Path) -> nx.Graph:
    log.info(f"Loading graph: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    log.info(f"  |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}")
    return G


def load_registry(path: Path) -> pd.DataFrame:
    log.info(f"Loading registry: {path}")
    df = pd.read_csv(path)
    required = {"node_id", "source", "cell_type", "gene"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Registry missing columns: {missing}")
    df["layer"] = df["source"].astype(str) + "::" + df["cell_type"].astype(str)
    return df.set_index("node_id")


def load_communities(path: Path) -> dict[Any, set[int]]:
    log.info(f"Loading communities: {path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    comms: dict[Any, set[int]] = {}
    if isinstance(raw, dict):
        for cid, val in raw.items():
            if isinstance(val, dict) and "nodes" in val:
                comms[cid] = set(val["nodes"])
            else:
                comms[cid] = set(val)
    elif isinstance(raw, list):
        for i, val in enumerate(raw):
            comms[i] = set(val)
    else:
        raise ValueError(f"Unsupported communities format: {type(raw)}")
    log.info(f"  {len(comms)} communities")
    return comms


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def per_layer_subgraph_metrics(
    G: nx.Graph,
    comms: dict[Any, set[int]],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    log.info("Computing per (community, layer) subgraph metrics...")
    all_layers = sorted(registry["layer"].unique())
    rows = []

    for cid, nodes in comms.items():
        # nodes in this community, grouped by layer
        valid = [n for n in nodes if n in registry.index]
        if not valid:
            continue
        node_layer = registry.loc[valid, "layer"]
        # Set for fast boundary computation
        community_nodes_set = set(valid)

        for layer in all_layers:
            layer_nodes = node_layer[node_layer == layer].index.tolist()
            if not layer_nodes:
                # Skip empty (community has no node in this layer) to keep file small
                continue

            sub = G.subgraph(layer_nodes)
            n = sub.number_of_nodes()
            m = sub.number_of_edges()
            density = nx.density(sub) if n > 1 else 0.0
            avg_deg = (2 * m / n) if n else 0.0

            # avg clustering on subgraph (skip for very small subgraphs to save time)
            try:
                avg_clust = nx.average_clustering(sub) if n >= 3 else 0.0
            except Exception:
                avg_clust = np.nan

            ccs = list(nx.connected_components(sub))
            n_cc = len(ccs)
            largest = max((len(c) for c in ccs), default=0)
            largest_frac = (largest / n) if n else 0.0
            connected = (n_cc == 1) if n > 0 else False
            n_isolates = sum(1 for v in sub.nodes() if sub.degree(v) == 0)

            # Boundary edges: this layer's community nodes -> SAME community nodes in OTHER layers
            boundary = 0
            other_layer_community_nodes = community_nodes_set.difference(layer_nodes)
            if other_layer_community_nodes:
                for u in layer_nodes:
                    for v in G.neighbors(u):
                        if v in other_layer_community_nodes:
                            boundary += 1
                # Each cross-layer edge counted once (u in this layer, v not in this layer)

            rows.append({
                "community": cid,
                "layer": layer,
                "n_nodes": int(n),
                "n_edges": int(m),
                "density": float(density),
                "avg_degree": float(avg_deg),
                "avg_clustering": float(avg_clust),
                "n_connected_components": int(n_cc),
                "largest_cc_size": int(largest),
                "largest_cc_fraction": float(largest_frac),
                "is_connected": bool(connected),
                "n_isolates": int(n_isolates),
                "intra_layer_edges": int(m),  # by construction
                "boundary_edges_to_other_layers": int(boundary),
            })

    df = pd.DataFrame(rows)
    log.info(f"  produced {len(df)} (community, layer) rows")
    return df


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_heatmap(long_df: pd.DataFrame, metric: str, outpath: Path, top_n: int = 30):
    if long_df.empty:
        return
    sizes = long_df.groupby("community")["n_nodes"].sum().sort_values(ascending=False)
    top_comms = sizes.head(top_n).index
    sub = long_df[long_df["community"].isin(top_comms)]
    pivot = sub.pivot(index="community", columns="layer", values=metric).fillna(0)
    pivot = pivot.loc[[c for c in top_comms if c in pivot.index]]

    plt.figure(figsize=(max(8, 0.5 * pivot.shape[1] + 4), max(6, 0.3 * len(pivot))))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={"label": metric})
    plt.title(f"Per-layer subgraph {metric} (top {len(pivot)} communities by total size)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_size_vs_density(long_df: pd.DataFrame, outpath: Path):
    if long_df.empty:
        return
    plt.figure(figsize=(7, 5))
    layers = sorted(long_df["layer"].unique())
    palette = sns.color_palette("tab20", n_colors=len(layers))
    for ly, color in zip(layers, palette):
        sub = long_df[long_df["layer"] == ly]
        plt.scatter(sub["n_nodes"], sub["density"], s=12, alpha=0.6,
                    color=color, label=ly)
    plt.xscale("log")
    plt.xlabel("Subgraph size (n_nodes, log)")
    plt.ylabel("Density")
    plt.title("Per (community, layer) subgraph size vs density")
    if len(layers) <= 20:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_per_community_profiles(long_df: pd.DataFrame, outpath: Path, top_n: int = 10):
    if long_df.empty:
        return
    sizes = long_df.groupby("community")["n_nodes"].sum().sort_values(ascending=False)
    top_comms = sizes.head(top_n).index.tolist()
    sub = long_df[long_df["community"].isin(top_comms)]

    fig, axes = plt.subplots(len(top_comms), 1, figsize=(10, 2.2 * len(top_comms)),
                             sharex=False)
    if len(top_comms) == 1:
        axes = [axes]
    for ax, cid in zip(axes, top_comms):
        s = sub[sub["community"] == cid].set_index("layer")
        if s.empty:
            continue
        s = s.sort_values("n_nodes", ascending=False)
        x = np.arange(len(s))
        width = 0.25
        ax.bar(x - width, s["n_nodes"] / s["n_nodes"].max() if s["n_nodes"].max() else s["n_nodes"],
               width, label="n_nodes (norm)", color="steelblue")
        ax.bar(x, s["density"], width, label="density", color="darkorange")
        ax.bar(x + width, s["largest_cc_fraction"], width,
               label="largest_cc_frac", color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(s.index, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"Community {cid}", fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0].legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--communities", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("multiplex_community_out"))
    ap.add_argument("--top-n", type=int, default=10,
                    help="Top-N communities (by total size) for profile plots")
    ap.add_argument("--wide-metric", default="density",
                    help="Metric to pivot into the wide CSV")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.outdir / "plots"
    plot_dir.mkdir(exist_ok=True)

    G = load_graph(args.graph)
    registry = load_registry(args.registry)
    comms = load_communities(args.communities)

    long_df = per_layer_subgraph_metrics(G, comms, registry)
    long_df.to_csv(args.outdir / "community_layer_subgraph_metrics.csv", index=False)

    if not long_df.empty and args.wide_metric in long_df.columns:
        wide = long_df.pivot(index="community", columns="layer",
                             values=args.wide_metric).fillna(0)
        wide.to_csv(args.outdir / f"community_layer_subgraph_wide_{args.wide_metric}.csv")

    # Plots
    plot_heatmap(long_df, "density", plot_dir / "subgraph_density_heatmap.png")
    plot_heatmap(long_df, "largest_cc_fraction",
                 plot_dir / "subgraph_connectedness_heatmap.png")
    plot_size_vs_density(long_df, plot_dir / "subgraph_size_vs_density.png")
    plot_per_community_profiles(long_df,
                                plot_dir / "per_community_layer_profiles.png",
                                top_n=args.top_n)

    log.info(f"Done. Outputs in {args.outdir}")


if __name__ == "__main__":
    main()