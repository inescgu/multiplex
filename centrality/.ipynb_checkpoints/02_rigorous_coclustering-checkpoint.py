#!/usr/bin/env python3
"""
02_rigorous_coclustering.py  (multiplex-aware)

HiDeF persistence-based co-clustering with multiplex-aware null models.

Key multiplex additions vs original:
  - Co-clustering matrix computed per-layer AND globally
  - Layer-stratified permutation null: rewiring preserves layer membership
  - Cross-layer co-clustering: how often do copies of the same gene
    (across layers) land in the same community?
  - Significance scores tagged with layer metadata

Consumes:
  - integrated_multiplex.pkl
  - node_registry.csv
  - persistent_communities.pkl

Produces (in --outdir):
  - coclustering_matrix.npz
  - gene_coclustering_scores.csv
  - cross_layer_coclustering.csv
  - layer_coclustering_summary.csv
  - plots/

Usage:
  python 02_rigorous_coclustering.py \
      --graph integrated_multiplex.pkl \
      --registry node_registry.csv \
      --communities persistent_communities.pkl \
      --outdir coclustering_out \
      --n-perm 200 \
      --null-mode rewire \
      --workers 4 \
      --seed 42
"""

from __future__ import annotations

import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.sparse import csr_matrix, save_npz

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("coclustering")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_graph(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    log.info(f"Graph: |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}")
    return G


def load_registry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["layer"] = df["source"].astype(str) + "::" + df["cell_type"].astype(str)
    return df.set_index("node_id")


def load_communities(path: Path) -> tuple[dict[Any, set[int]], dict[Any, float]]:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    comms: dict[Any, set[int]] = {}
    persistence: dict[Any, float] = {}
    if isinstance(raw, dict):
        for cid, val in raw.items():
            if isinstance(val, dict) and "nodes" in val:
                comms[cid] = set(val["nodes"])
                if "persistence" in val:
                    persistence[cid] = float(val["persistence"])
            else:
                comms[cid] = set(val)
    elif isinstance(raw, list):
        for i, val in enumerate(raw):
            comms[i] = set(val)
    log.info(f"Communities: {len(comms)}")
    return comms, persistence


# --------------------------------------------------------------------------- #
# Co-clustering matrix
# --------------------------------------------------------------------------- #
def build_coclustering_matrix(
    comms: dict[Any, set[int]],
    all_nodes: list[int],
    persistence: dict[Any, float] | None = None,
) -> np.ndarray:
    """
    C[i,j] = sum of persistence weights of communities containing both i and j.
    If no persistence, weight = 1.
    """
    n = len(all_nodes)
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    C = np.zeros((n, n), dtype=np.float32)
    for cid, nodes in comms.items():
        w = persistence.get(cid, 1.0) if persistence else 1.0
        idx = [node_idx[nd] for nd in nodes if nd in node_idx]
        for a in range(len(idx)):
            for b in range(a, len(idx)):
                C[idx[a], idx[b]] += w
                if a != b:
                    C[idx[b], idx[a]] += w
    return C


# --------------------------------------------------------------------------- #
# Layer-stratified co-clustering
# --------------------------------------------------------------------------- #
def layer_coclustering_summary(
    comms: dict[Any, set[int]],
    registry: pd.DataFrame,
    persistence: dict[Any, float],
) -> pd.DataFrame:
    """
    For each layer, compute the mean co-clustering weight among node pairs
    that are BOTH in that layer.
    """
    log.info("Computing layer-stratified co-clustering...")
    rows = []
    for layer, sub in registry.groupby("layer"):
        layer_nodes = [n for n in sub.index if any(n in c for c in comms.values())]
        if len(layer_nodes) < 2:
            continue
        node_idx = {n: i for i, n in enumerate(layer_nodes)}
        C = build_coclustering_matrix(
            {cid: nodes & set(layer_nodes) for cid, nodes in comms.items()},
            layer_nodes, persistence
        )
        upper = C[np.triu_indices(len(layer_nodes), k=1)]
        rows.append({
            "layer": layer,
            "n_nodes": len(layer_nodes),
            "mean_coclustering": float(upper.mean()),
            "median_coclustering": float(np.median(upper)),
            "frac_nonzero": float((upper > 0).mean()),
        })
    return pd.DataFrame(rows).set_index("layer")


# --------------------------------------------------------------------------- #
# Cross-layer co-clustering
# --------------------------------------------------------------------------- #
def cross_layer_coclustering(
    comms: dict[Any, set[int]],
    registry: pd.DataFrame,
    persistence: dict[Any, float],
) -> pd.DataFrame:
    """
    For each gene with copies in >=2 layers, compute the mean co-clustering
    weight between all pairs of its copies across different layers.
    """
    log.info("Computing cross-layer co-clustering...")
    node_to_comm_weight: dict[int, dict[Any, float]] = defaultdict(dict)
    for cid, nodes in comms.items():
        w = persistence.get(cid, 1.0)
        for n in nodes:
            node_to_comm_weight[n][cid] = w

    rows = []
    for gene, sub in registry.reset_index().groupby("gene"):
        if sub["layer"].nunique() < 2:
            continue
        node_ids = sub["node_id"].tolist()
        pairs = [(node_ids[i], node_ids[j])
                 for i in range(len(node_ids))
                 for j in range(i + 1, len(node_ids))
                 if sub.iloc[i]["layer"] != sub.iloc[j]["layer"]]
        if not pairs:
            continue
        scores = []
        for u, v in pairs:
            shared = set(node_to_comm_weight.get(u, {}).keys()) & \
                     set(node_to_comm_weight.get(v, {}).keys())
            score = sum(node_to_comm_weight[u][c] for c in shared)
            scores.append(score)
        rows.append({
            "gene": gene,
            "n_copies": len(node_ids),
            "n_layers": sub["layer"].nunique(),
            "n_cross_pairs": len(pairs),
            "mean_cross_coclustering": float(np.mean(scores)),
            "max_cross_coclustering": float(np.max(scores)),
            "frac_nonzero_pairs": float(np.mean([s > 0 for s in scores])),
        })
    return pd.DataFrame(rows).sort_values("mean_cross_coclustering", ascending=False)


# --------------------------------------------------------------------------- #
# Null models
# --------------------------------------------------------------------------- #
def _rewire_worker(args):
    edges, n_nodes, node_layers, seed = args
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges)
    # Layer-stratified rewiring: only swap edges within same layer pair
    layer_edges: dict[tuple, list] = defaultdict(list)
    for u, v in G.edges():
        lu, lv = node_layers.get(u, "?"), node_layers.get(v, "?")
        key = tuple(sorted([lu, lv]))
        layer_edges[key].append((u, v))
    G_rewired = G.copy()
    for key, elist in layer_edges.items():
        if len(elist) < 2:
            continue
        sub = nx.Graph()
        sub.add_edges_from(elist)
        try:
            sub = nx.double_edge_swap(
                sub, nswap=max(10, len(elist) * 2),
                max_tries=len(elist) * 20,
                seed=int(rng.integers(1e9))
            )
        except Exception:
            pass
        G_rewired.remove_edges_from(elist)
        G_rewired.add_edges_from(sub.edges())
    return list(G_rewired.edges())


def permutation_null(
    G: nx.Graph,
    comms: dict[Any, set[int]],
    registry: pd.DataFrame,
    all_nodes: list[int],
    persistence: dict[Any, float],
    n_perm: int,
    workers: int,
    seed: int,
    mode: str = "rewire",
) -> np.ndarray:
    """
    Returns null_C: (n_perm, n_nodes, n_nodes) — too large for big graphs.
    Instead returns null_mean and null_std matrices.
    """
    log.info(f"Running {n_perm} null permutations (mode={mode})...")
    node_layers = {n: registry.loc[n, "layer"]
                   if n in registry.index else "unknown"
                   for n in G.nodes()}
    ss = np.random.SeedSequence(seed)
    child_seeds = [int(s.entropy) for s in ss.spawn(n_perm)]
    edges = list(G.edges())
    n_nodes_g = G.number_of_nodes()
    tasks = [(edges, n_nodes_g, node_layers, s) for s in child_seeds]

    n = len(all_nodes)
    null_sum = np.zeros((n, n), dtype=np.float64)
    null_sq = np.zeros((n, n), dtype=np.float64)

    def _process(rewired_edges):
        G_null = nx.Graph()
        G_null.add_nodes_from(G.nodes())
        G_null.add_edges_from(rewired_edges)
        # Re-detect communities on null graph using label propagation (fast)
        null_comms_raw = list(nx.community.label_propagation_communities(G_null))
        null_comms = {i: set(c) for i, c in enumerate(null_comms_raw)}
        return build_coclustering_matrix(null_comms, all_nodes, None)

    if workers <= 1:
        for i, t in enumerate(tasks):
            rewired = _rewire_worker(t)
            C_null = _process(rewired)
            null_sum += C_null
            null_sq += C_null ** 2
            if (i + 1) % 20 == 0:
                log.info(f"  {i+1}/{n_perm} null iterations done")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_rewire_worker, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                C_null = _process(fut.result())
                null_sum += C_null
                null_sq += C_null ** 2
                if i % 20 == 0:
                    log.info(f"  {i}/{n_perm} null iterations done")

    null_mean = null_sum / n_perm
    null_std = np.sqrt(np.maximum(null_sq / n_perm - null_mean ** 2, 0))
    return null_mean, null_std


# --------------------------------------------------------------------------- #
# Gene significance scores
# --------------------------------------------------------------------------- #
def gene_significance_scores(
    C_obs: np.ndarray,
    null_mean: np.ndarray,
    null_std: np.ndarray,
    all_nodes: list[int],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-gene score = mean z-score of co-clustering with all other nodes.
    """
    log.info("Computing gene significance scores...")
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(null_std > 0, (C_obs - null_mean) / null_std, 0.0)

    rows = []
    for i, n in enumerate(all_nodes):
        meta = registry.loc[n] if n in registry.index else {}
        z_row = Z[i]
        rows.append({
            "node_id": n,
            "gene": meta.get("gene", "?") if isinstance(meta, pd.Series) else "?",
            "layer": meta.get("layer", "?") if isinstance(meta, pd.Series) else "?",
            "mean_z": float(np.mean(z_row)),
            "max_z": float(np.max(z_row)),
            "n_sig_partners": int(np.sum(z_row > 1.96)),
        })
    return pd.DataFrame(rows).sort_values("mean_z", ascending=False)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(
    gene_df: pd.DataFrame,
    layer_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    outdir: Path,
):
    sns.set_style("whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Gene z-score distribution by layer
    if "layer" in gene_df.columns and "mean_z" in gene_df.columns:
        plt.figure(figsize=(8, 4))
        layers = sorted(gene_df["layer"].unique())
        palette = sns.color_palette("tab20", n_colors=len(layers))
        for ly, color in zip(layers, palette):
            sub = gene_df[gene_df["layer"] == ly]["mean_z"].dropna()
            plt.hist(sub, bins=30, alpha=0.5, color=color, label=ly, density=True)
        plt.xlabel("Mean co-clustering z-score")
        plt.ylabel("Density")
        plt.title("Co-clustering significance by layer")
        if len(layers) <= 20:
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        plt.savefig(outdir / "coclustering_zscore_by_layer.png", dpi=200)
        plt.close()

    # 2. Layer co-clustering summary
    if not layer_df.empty and "mean_coclustering" in layer_df.columns:
        plt.figure(figsize=(8, 4))
        layer_df["mean_coclustering"].sort_values().plot(
            kind="barh", color="steelblue", edgecolor="k"
        )
        plt.xlabel("Mean co-clustering weight")
        plt.title("Per-layer mean co-clustering")
        plt.tight_layout()
        plt.savefig(outdir / "layer_coclustering_summary.png", dpi=200)
        plt.close()

    # 3. Cross-layer co-clustering top genes
    if not cross_df.empty:
        top = cross_df.head(30)
        plt.figure(figsize=(8, 6))
        plt.barh(top["gene"][::-1], top["mean_cross_coclustering"][::-1],
                 color="darkorange")
        plt.xlabel("Mean cross-layer co-clustering weight")
        plt.title("Top 30 genes by cross-layer co-clustering")
        plt.tight_layout()
        plt.savefig(outdir / "cross_layer_coclustering_top_genes.png", dpi=200)
        plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--communities", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("coclustering_out"))
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--null-mode", choices=["rewire", "sample"], default="rewire")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    G = load_graph(args.graph)
    registry = load_registry(args.registry)
    comms, persistence = load_communities(args.communities)

    all_nodes = sorted(G.nodes())

    # Observed co-clustering matrix
    log.info("Building observed co-clustering matrix...")
    C_obs = build_coclustering_matrix(comms, all_nodes, persistence)
    save_npz(str(args.outdir / "coclustering_matrix.npz"),
             csr_matrix(C_obs.astype(np.float32)))

    # Layer-stratified summary
    layer_df = layer_coclustering_summary(comms, registry, persistence)
    layer_df.to_csv(args.outdir / "layer_coclustering_summary.csv")

    # Cross-layer co-clustering
    cross_df = cross_layer_coclustering(comms, registry, persistence)
    cross_df.to_csv(args.outdir / "cross_layer_coclustering.csv", index=False)

    # Null model
    null_mean, null_std = permutation_null(
        G, comms, registry, all_nodes, persistence,
        n_perm=args.n_perm, workers=args.workers,
        seed=args.seed, mode=args.null_mode,
    )

    # Gene significance
    gene_df = gene_significance_scores(C_obs, null_mean, null_std,
                                        all_nodes, registry)
    gene_df.to_csv(args.outdir / "gene_coclustering_scores.csv", index=False)

    # Plots
    make_plots(gene_df, layer_df, cross_df, args.outdir / "plots")
    log.info(f"Done. Outputs in {args.outdir}")


if __name__ == "__main__":
    main()