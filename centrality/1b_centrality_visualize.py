#!/usr/bin/env python3
"""
01b_centrality_visualize.py  (multiplex-aware)

Loads centrality_results.pkl from 01a_ and produces:
  - CSVs: node_centrality.csv, cluster_centrality.csv, layer_degree.csv
  - Cytoscape export: cytoscape_nodes.csv
  - Plots:
      centrality_distributions.png
      cluster_metrics.png
      top_genes_betweenness.png
      degree_vs_betweenness.png
      cluster_significance.png
      layer_participation_distribution.png
      per_layer_degree_heatmap.png

Usage:
  python 01b_centrality_visualize.py \
      --results centrality_results.pkl \
      --outdir centrality_out
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("centrality_visualize")


def load_results(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_csvs(results: dict, outdir: Path):
    results["node_df"].to_csv(outdir / "node_centrality.csv")
    results["cluster_df"].to_csv(outdir / "cluster_centrality.csv")
    if "layer_degree_df" in results:
        results["layer_degree_df"].to_csv(outdir / "layer_degree.csv")
    log.info("CSVs saved.")


def save_cytoscape(node_df: pd.DataFrame, outdir: Path):
    cols = ["gene", "source", "cell_type", "layer", "degree",
            "betweenness", "eigenvector", "clustering",
            "participation_coef", "perm_z", "perm_p", "pop_z", "pop_p"]
    export = node_df[[c for c in cols if c in node_df.columns]].copy()
    export.to_csv(outdir / "cytoscape_nodes.csv")
    log.info("Cytoscape export saved.")


def plot_centrality_distributions(node_df: pd.DataFrame, outdir: Path):
    metrics = ["degree", "betweenness", "eigenvector", "clustering",
               "participation_coef"]
    metrics = [m for m in metrics if m in node_df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        ax.hist(node_df[m].dropna(), bins=50, color="steelblue", edgecolor="k")
        ax.set_xlabel(m)
        ax.set_ylabel("# nodes")
        ax.set_title(m.replace("_", " ").title())
    plt.suptitle("Node centrality distributions (multiplex)", y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / "centrality_distributions.png", dpi=200)
    plt.close()


def plot_cluster_metrics(cluster_df: pd.DataFrame, outdir: Path):
    metrics = ["mean_betweenness", "mean_eigenvector", "mean_clustering",
               "mean_degree", "mean_participation"]
    metrics = [m for m in metrics if m in cluster_df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        vals = cluster_df[m].sort_values(ascending=False)
        ax.barh(vals.index[::-1], vals.values[::-1], color="darkorange")
        ax.set_xlabel(m)
        ax.set_title(m.replace("_", " ").title())
        ax.tick_params(axis="y", labelsize=7)
    plt.suptitle("Per-layer cluster centrality", y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / "cluster_metrics.png", dpi=200)
    plt.close()


def plot_top_genes_betweenness(node_df: pd.DataFrame, outdir: Path, top_n: int = 30):
    top = node_df.nlargest(top_n, "betweenness")[["gene", "layer", "betweenness"]]
    fig, ax = plt.subplots(figsize=(8, 0.35 * top_n + 2))
    colors = plt.cm.tab20(np.linspace(0, 1, top["layer"].nunique()))
    layer_color = {l: c for l, c in zip(sorted(top["layer"].unique()), colors)}
    bars = ax.barh(
        [f"{r.gene} [{r.layer}]" for _, r in top.iterrows()][::-1],
        top["betweenness"].values[::-1],
        color=[layer_color[l] for l in top["layer"].values[::-1]],
    )
    ax.set_xlabel("Betweenness centrality")
    ax.set_title(f"Top {top_n} nodes by betweenness (colored by layer)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in layer_color.values()]
    ax.legend(handles, layer_color.keys(), loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "top_genes_betweenness.png", dpi=200)
    plt.close()


def plot_degree_vs_betweenness(node_df: pd.DataFrame, outdir: Path):
    layers = sorted(node_df["layer"].unique())
    palette = sns.color_palette("tab20", n_colors=len(layers))
    plt.figure(figsize=(8, 6))
    for ly, color in zip(layers, palette):
        sub = node_df[node_df["layer"] == ly]
        plt.scatter(sub["degree"], sub["betweenness"],
                    s=10, alpha=0.5, color=color, label=ly)
    plt.xlabel("Degree")
    plt.ylabel("Betweenness centrality")
    plt.title("Degree vs betweenness (colored by layer)")
    if len(layers) <= 20:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "degree_vs_betweenness.png", dpi=200)
    plt.close()


def plot_cluster_significance(node_df: pd.DataFrame, outdir: Path):
    if "perm_z" not in node_df.columns or "pop_z" not in node_df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    layers = sorted(node_df["layer"].unique())
    palette = sns.color_palette("tab20", n_colors=len(layers))
    for ly, color in zip(layers, palette):
        sub = node_df[node_df["layer"] == ly]
        axes[0].scatter(sub["perm_z"], sub["pop_z"],
                        s=10, alpha=0.5, color=color, label=ly)
    axes[0].axhline(0, ls="--", color="k", lw=0.8)
    axes[0].axvline(0, ls="--", color="k", lw=0.8)
    axes[0].set_xlabel("Permutation z-score")
    axes[0].set_ylabel("Population z-score")
    axes[0].set_title("Perm vs population z-scores")
    if len(layers) <= 20:
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)

    axes[1].scatter(node_df["perm_z"], node_df["pop_z"],
                    s=8, alpha=0.3, color="steelblue")
    axes[1].set_xlabel("Permutation z-score")
    axes[1].set_ylabel("Population z-score")
    axes[1].set_title("Correlation (all nodes)")
    plt.tight_layout()
    plt.savefig(outdir / "cluster_significance.png", dpi=200)
    plt.close()


def plot_participation_distribution(node_df: pd.DataFrame, outdir: Path):
    if "participation_coef" not in node_df.columns:
        return
    plt.figure(figsize=(7, 4))
    layers = sorted(node_df["layer"].unique())
    palette = sns.color_palette("tab20", n_colors=len(layers))
    for ly, color in zip(layers, palette):
        sub = node_df[node_df["layer"] == ly]["participation_coef"].dropna()
        plt.hist(sub, bins=30, alpha=0.5, color=color, label=ly, density=True)
    plt.xlabel("Participation coefficient")
    plt.ylabel("Density")
    plt.title("Layer participation coefficient by layer")
    if len(layers) <= 20:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "layer_participation_distribution.png", dpi=200)
    plt.close()


def plot_per_layer_degree_heatmap(layer_deg_df: pd.DataFrame, outdir: Path,
                                   top_n: int = 50):
    if layer_deg_df is None or layer_deg_df.empty:
        return
    # Top nodes by total degree
    top = layer_deg_df.sum(axis=1).nlargest(top_n).index
    sub = layer_deg_df.loc[top]
    plt.figure(figsize=(max(8, 0.5 * sub.shape[1] + 4), max(6, 0.2 * top_n)))
    sns.heatmap(sub, cmap="YlOrRd", cbar_kws={"label": "degree within layer"})
    plt.title(f"Per-layer degree (top {top_n} nodes by total degree)")
    plt.tight_layout()
    plt.savefig(outdir / "per_layer_degree_heatmap.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("centrality_results.pkl"))
    ap.add_argument("--outdir", type=Path, default=Path("centrality_out"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results)
    node_df = results["node_df"]
    cluster_df = results["cluster_df"]
    layer_deg_df = results.get("layer_degree_df")

    save_csvs(results, args.outdir)
    save_cytoscape(node_df, args.outdir)

    sns.set_style("whitegrid")
    plot_centrality_distributions(node_df, args.outdir)
    plot_cluster_metrics(cluster_df, args.outdir)
    plot_top_genes_betweenness(node_df, args.outdir)
    plot_degree_vs_betweenness(node_df, args.outdir)
    plot_cluster_significance(node_df, args.outdir)
    plot_participation_distribution(node_df, args.outdir)
    plot_per_layer_degree_heatmap(layer_deg_df, args.outdir)

    log.info(f"Done. Outputs in {args.outdir}")


if __name__ == "__main__":
    main()