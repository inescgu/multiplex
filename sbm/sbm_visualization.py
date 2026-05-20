#!/usr/bin/env python3
"""
sbm_visualization.py — Visualisation utilities for multiplex SBM results
==========================================================================

Generates:
  1. Community assignment heatmap by layer / cell type
  2. Block probability matrices (one per layer)
  3. Community size distribution (bar chart)
  4. Model comparison plot (BIC / AIC / ICL vs K)
  5. ELBO convergence trace
  6. Network layout coloured by community (subgraph for large networks)

All plots are saved to disk as PNG files.

Author: Abacus AI Agent
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from matplotlib.colors import Normalize

from sbm_core import MultiplexData, SBMParams

logger = logging.getLogger(__name__)

# Global style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
})


# ---------------------------------------------------------------------------
# 1. Community assignment heatmap
# ---------------------------------------------------------------------------

def plot_community_heatmap(
    data: MultiplexData,
    result: SBMParams,
    output_dir: str,
    filename: str = "community_heatmap.png",
) -> str:
    """
    Heatmap: rows = communities, columns = (source, cell_type) layers.
    Cell values = number of nodes from that layer assigned to that community.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = data.node_attrs.copy()
    df["community"] = result.community_assignments
    df["layer"] = df["source"] + " | " + df["cell_type"]

    # Pivot: community × layer
    pivot = df.groupby(["community", "layer"]).size().unstack(fill_value=0)

    # Sort communities by size
    comm_sizes = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[comm_sizes.index]

    # Limit to top 30 communities for readability
    if len(pivot) > 30:
        pivot = pivot.head(30)

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.8),
                                     max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=len(pivot.columns) <= 15 and len(pivot) <= 20,
                fmt="d" if len(pivot.columns) <= 15 and len(pivot) <= 20 else "",
                cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title(f"Community Assignment by Layer (K={result.K})")
    ax.set_ylabel("Community")
    ax.set_xlabel("Layer (source | cell_type)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    path = out / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Community heatmap saved → %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# 2. Block probability matrices
# ---------------------------------------------------------------------------

def plot_block_matrices(
    result: SBMParams,
    layer_names: list,
    output_dir: str,
    filename_prefix: str = "block_matrix",
) -> list:
    """
    Plot block probability matrix B^l for each layer.
    Shows inter/intra-community edge probabilities.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    K = result.K
    # If K is large, show only top communities by size
    show_K = min(K, 25)

    # Sort communities by size for display
    comm_counts = np.bincount(result.community_assignments, minlength=K)
    top_comms = np.argsort(-comm_counts)[:show_K]

    for l_idx, (B, name) in enumerate(zip(result.B, layer_names)):
        B_sub = B[np.ix_(top_comms, top_comms)]

        fig, ax = plt.subplots(figsize=(max(6, show_K * 0.5),
                                         max(5, show_K * 0.45)))
        im = ax.imshow(B_sub, cmap="viridis", aspect="equal",
                       vmin=0, vmax=min(B_sub.max() * 1.2, 1.0))
        plt.colorbar(im, ax=ax, label="Edge probability")

        if show_K <= 20:
            ax.set_xticks(range(show_K))
            ax.set_xticklabels(top_comms, fontsize=7)
            ax.set_yticks(range(show_K))
            ax.set_yticklabels(top_comms, fontsize=7)

        ax.set_title(f"Block Probability Matrix — {name}\n"
                     f"(Top {show_K} communities by size, K={K})")
        ax.set_xlabel("Community")
        ax.set_ylabel("Community")
        plt.tight_layout()

        path = out / f"{filename_prefix}_{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
        logger.info("Block matrix [%s] saved → %s", name, path)

    return paths


# ---------------------------------------------------------------------------
# 3. Community size distribution
# ---------------------------------------------------------------------------

def plot_community_sizes(
    result: SBMParams,
    output_dir: str,
    filename: str = "community_sizes.png",
) -> str:
    """Bar chart of community sizes, sorted descending."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    K = result.K
    comm_counts = np.bincount(result.community_assignments, minlength=K)
    order = np.argsort(-comm_counts)

    fig, ax = plt.subplots(figsize=(max(8, K * 0.3), 5))
    x = range(len(order))
    bars = ax.bar(x, comm_counts[order], color=plt.cm.tab20(np.linspace(0, 1, K))[order])
    ax.set_xlabel("Community (sorted by size)")
    ax.set_ylabel("Number of nodes")
    ax.set_title(f"Community Size Distribution (K={K})")

    if K <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=7, rotation=45)
    else:
        ax.set_xticks([])

    # Add text annotations for top 10
    for i in range(min(10, len(order))):
        ax.text(i, comm_counts[order[i]] + max(comm_counts) * 0.01,
                str(comm_counts[order[i]]), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Community sizes saved → %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# 4. Model comparison plot
# ---------------------------------------------------------------------------

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_dir: str,
    filename: str = "model_comparison.png",
) -> str:
    """Plot BIC, AIC, ICL vs K with the optimal K marked."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    criteria = ["BIC", "AIC", "ICL"]

    for ax, crit in zip(axes, criteria):
        if crit in comparison_df.columns:
            vals = comparison_df[crit].values
            Ks = comparison_df["K"].values
            ax.plot(Ks, vals, "o-", markersize=4, linewidth=1.5)

            # Mark best
            best_idx = np.argmin(vals)
            ax.axvline(Ks[best_idx], color="red", linestyle="--", alpha=0.6,
                       label=f"Best K={Ks[best_idx]}")
            ax.scatter([Ks[best_idx]], [vals[best_idx]], color="red", s=80,
                       zorder=5, marker="*")

            ax.set_xlabel("K (number of communities)")
            ax.set_ylabel(crit)
            ax.set_title(f"{crit} vs K")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison plot saved → %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# 5. ELBO convergence trace
# ---------------------------------------------------------------------------

def plot_elbo_convergence(
    result: SBMParams,
    output_dir: str,
    filename: str = "elbo_convergence.png",
) -> str:
    """Plot ELBO over EM iterations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.elbo_history, "b-", linewidth=1.5)
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("ELBO")
    ax.set_title(f"ELBO Convergence (K={result.K}, "
                 f"{'converged' if result.converged else 'not converged'})")
    ax.grid(True, alpha=0.3)

    if len(result.elbo_history) > 1:
        ax.axhline(result.elbo_history[-1], color="red", linestyle="--",
                   alpha=0.5, label=f"Final: {result.elbo_history[-1]:.4e}")
        ax.legend()

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("ELBO convergence saved → %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# 6. Network layout coloured by community
# ---------------------------------------------------------------------------

def plot_network_communities(
    data: MultiplexData,
    result: SBMParams,
    graph_path: str,
    output_dir: str,
    max_nodes: int = 2000,
    filename: str = "network_communities.png",
) -> str:
    """
    Spring layout of the network coloured by community.
    For large networks, samples a subgraph of max_nodes.
    """
    import pickle as pkl
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading graph for network layout plot ...")
    with open(graph_path, "rb") as f:
        G = pkl.load(f)

    N = data.n_nodes
    all_nodes = list(G.nodes())

    if N > max_nodes:
        logger.info("  Subsampling %d / %d nodes for layout", max_nodes, N)
        # Sample nodes proportional to community sizes
        rng = np.random.default_rng(42)
        K = result.K
        comm_counts = np.bincount(result.community_assignments, minlength=K)
        per_comm = np.maximum((comm_counts / N * max_nodes).astype(int), 1)
        total_budget = max_nodes

        sampled_indices = []
        for k in range(K):
            members = np.where(result.community_assignments == k)[0]
            n_sample = min(per_comm[k], len(members), total_budget - len(sampled_indices))
            if n_sample > 0:
                chosen = rng.choice(members, size=n_sample, replace=False)
                sampled_indices.extend(chosen.tolist())

        sampled_node_ids = data.node_ids[sampled_indices]
        subG = G.subgraph(sampled_node_ids).copy()
        comm_map = {data.node_ids[i]: result.community_assignments[i]
                    for i in sampled_indices}
    else:
        subG = G
        comm_map = {data.node_ids[i]: result.community_assignments[i]
                    for i in range(N)}

    del G
    import gc
    gc.collect()

    # Layout
    logger.info("  Computing spring layout for %d nodes ...", subG.number_of_nodes())
    pos = nx.spring_layout(subG, k=1.5 / np.sqrt(max(subG.number_of_nodes(), 1)),
                           iterations=50, seed=42)

    # Colours
    K = result.K
    cmap = plt.cm.get_cmap("tab20", K) if K <= 20 else plt.cm.get_cmap("nipy_spectral", K)
    node_list = list(subG.nodes())
    colours = [cmap(comm_map.get(n, 0) % K) for n in node_list]

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(subG, pos, alpha=0.05, width=0.3, ax=ax)
    nx.draw_networkx_nodes(subG, pos, node_color=colours, node_size=8,
                           alpha=0.7, ax=ax)
    ax.set_title(f"Network Layout by Community (K={K}, "
                 f"{subG.number_of_nodes()} nodes shown)")
    ax.axis("off")

    # Legend for top communities
    top_k = min(K, 15)
    comm_counts = np.bincount(result.community_assignments, minlength=K)
    top_comms = np.argsort(-comm_counts)[:top_k]
    legend_handles = []
    for k in top_comms:
        patch = plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cmap(k % K), markersize=8,
                          label=f"C{k} (n={comm_counts[k]})")
        legend_handles.append(patch)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              ncol=max(1, top_k // 8), title="Communities")

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Network layout saved → %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# Master function: generate all visualisations
# ---------------------------------------------------------------------------

def generate_all_visualisations(
    data: MultiplexData,
    result: SBMParams,
    comparison_df: Optional[pd.DataFrame],
    graph_path: str,
    output_dir: str,
    max_network_nodes: int = 2000,
) -> dict:
    """
    Generate all visualisations and return dict of output paths.
    """
    paths = {}

    logger.info("Generating visualisations ...")

    # 1. Community heatmap
    try:
        paths["community_heatmap"] = plot_community_heatmap(
            data, result, output_dir)
    except Exception as e:
        logger.warning("Failed to generate community heatmap: %s", e)

    # 2. Block matrices
    try:
        paths["block_matrices"] = plot_block_matrices(
            result, data.layer_names, output_dir)
    except Exception as e:
        logger.warning("Failed to generate block matrices: %s", e)

    # 3. Community sizes
    try:
        paths["community_sizes"] = plot_community_sizes(result, output_dir)
    except Exception as e:
        logger.warning("Failed to generate community sizes: %s", e)

    # 4. Model comparison
    if comparison_df is not None and len(comparison_df) > 1:
        try:
            paths["model_comparison"] = plot_model_comparison(
                comparison_df, output_dir)
        except Exception as e:
            logger.warning("Failed to generate model comparison plot: %s", e)

    # 5. ELBO convergence
    try:
        paths["elbo_convergence"] = plot_elbo_convergence(result, output_dir)
    except Exception as e:
        logger.warning("Failed to generate ELBO convergence: %s", e)

    # 6. Network layout
    try:
        paths["network_layout"] = plot_network_communities(
            data, result, graph_path, output_dir,
            max_nodes=max_network_nodes)
    except Exception as e:
        logger.warning("Failed to generate network layout: %s", e)

    logger.info("Visualisation generation complete. %d plots created.", len(paths))
    return paths
