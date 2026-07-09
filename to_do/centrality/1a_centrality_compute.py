#!/usr/bin/env python3
"""
01a_centrality_compute.py  (multiplex-aware)

Centrality analysis on the integrated multiplex interactome.

Key multiplex additions vs original:
  - All node lookups go through node_registry (source, cell_type, gene)
  - Centrality computed on the full multiplex graph (integer node IDs)
  - Per-layer degree centrality: degree within each (source, cell_type) layer
  - Layer participation coefficient per node
  - Dual significance: permutation (degree-preserving) + population (full PPI)
  - Results tagged with layer metadata throughout

Consumes:
  - integrated_multiplex.pkl
  - node_registry.csv
  - full_ppi.pkl              (population baseline — same format as builder)

Produces:
  - centrality_results.pkl    (intermediate, consumed by 01b_)

Usage:
  python 01a_centrality_compute.py \
      --graph integrated_multiplex.pkl \
      --registry node_registry.csv \
      --ppi full_ppi.pkl \
      --outfile centrality_results.pkl \
      --n-perm 1000 \
      --workers 4 \
      --seed 42
"""

from __future__ import annotations

import argparse
import logging
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("centrality_compute")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_graph(path: Path) -> nx.Graph:
    log.info(f"Loading multiplex graph: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    log.info(f"  |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}")
    return G


def load_registry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"node_id", "source", "cell_type", "gene"}
    if missing := required - set(df.columns):
        raise ValueError(f"Registry missing: {missing}")
    df["layer"] = df["source"].astype(str) + "::" + df["cell_type"].astype(str)
    return df.set_index("node_id")


def load_ppi(path: Path) -> nx.Graph:
    log.info(f"Loading full PPI: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip().upper() for n in G.nodes()})
    log.info(f"  |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}")
    return G


# --------------------------------------------------------------------------- #
# Global metrics
# --------------------------------------------------------------------------- #
def compute_global_metrics(G: nx.Graph) -> dict:
    log.info("Computing global metrics...")
    lcc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc)
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_connected_components": nx.number_connected_components(G),
        "lcc_size": len(lcc),
        "lcc_fraction": len(lcc) / G.number_of_nodes(),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_shortest_path_lcc": nx.average_shortest_path_length(G_lcc)
        if len(lcc) < 5000 else np.nan,
    }


# --------------------------------------------------------------------------- #
# Per-layer degree
# --------------------------------------------------------------------------- #
def per_layer_degree(G: nx.Graph, registry: pd.DataFrame) -> pd.DataFrame:
    """
    For each node, compute its degree within each layer's induced subgraph.
    Returns a DataFrame: node_id x layer -> degree (0 if node not in layer).
    """
    log.info("Computing per-layer degree...")
    all_layers = sorted(registry["layer"].unique())
    rows = {}
    for layer in all_layers:
        layer_nodes = registry[registry["layer"] == layer].index.tolist()
        sub = G.subgraph([n for n in layer_nodes if n in G])
        for n, d in sub.degree():
            rows.setdefault(n, {})[layer] = d

    df = pd.DataFrame(rows).T.fillna(0).astype(int)
    df.index.name = "node_id"
    return df


# --------------------------------------------------------------------------- #
# Layer participation coefficient
# --------------------------------------------------------------------------- #
def layer_participation(G: nx.Graph, registry: pd.DataFrame) -> pd.Series:
    """
    Participation coefficient: 1 - sum_l (k_l / k_total)^2
    where k_l = degree of node in layer l.
    1 = perfectly distributed; 0 = all edges in one layer.
    """
    log.info("Computing layer participation coefficients...")
    all_layers = sorted(registry["layer"].unique())
    layer_deg: dict[int, dict[str, int]] = defaultdict(dict)
    for layer in all_layers:
        layer_nodes = registry[registry["layer"] == layer].index.tolist()
        sub = G.subgraph([n for n in layer_nodes if n in G])
        for n, d in sub.degree():
            layer_deg[n][layer] = d

    results = {}
    for n in G.nodes():
        k_total = G.degree(n)
        if k_total == 0:
            results[n] = 0.0
            continue
        k_layers = layer_deg.get(n, {})
        results[n] = 1.0 - sum((kl / k_total) ** 2
                                for kl in k_layers.values())
    return pd.Series(results, name="participation_coef")


# --------------------------------------------------------------------------- #
# Gene-level centrality
# --------------------------------------------------------------------------- #
def compute_node_centrality(G: nx.Graph, registry: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing node centrality (betweenness, eigenvector, clustering, degree)...")
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes)

    log.info("  betweenness (LCC)...")
    betweenness = nx.betweenness_centrality(G_lcc, normalized=True)

    log.info("  eigenvector (LCC)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            eigenvector = nx.eigenvector_centrality(G_lcc, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            eigenvector = {n: np.nan for n in G_lcc.nodes()}

    log.info("  clustering...")
    clustering = nx.clustering(G)

    log.info("  degree...")
    degree = dict(G.degree())

    # Participation coefficient
    participation = layer_participation(G, registry)

    rows = []
    for n in G.nodes():
        meta = registry.loc[n] if n in registry.index else {}
        rows.append({
            "node_id": n,
            "gene": meta.get("gene", "unknown") if isinstance(meta, pd.Series) else "unknown",
            "source": meta.get("source", "unknown") if isinstance(meta, pd.Series) else "unknown",
            "cell_type": meta.get("cell_type", "unknown") if isinstance(meta, pd.Series) else "unknown",
            "layer": meta.get("layer", "unknown") if isinstance(meta, pd.Series) else "unknown",
            "degree": degree.get(n, 0),
            "betweenness": betweenness.get(n, 0.0),
            "eigenvector": eigenvector.get(n, np.nan),
            "clustering": clustering.get(n, 0.0),
            "participation_coef": float(participation.get(n, 0.0)),
            "in_lcc": n in lcc_nodes,
        })
    return pd.DataFrame(rows).set_index("node_id")


# --------------------------------------------------------------------------- #
# Permutation test (degree-preserving)
# --------------------------------------------------------------------------- #
def _perm_worker(args):
    edges, n_nodes, seed = args
    rng = np.random.default_rng(seed)
    G_perm = nx.Graph()
    G_perm.add_nodes_from(range(n_nodes))
    G_perm.add_edges_from(edges)
    G_perm = nx.double_edge_swap(G_perm,
                                  nswap=max(100, G_perm.number_of_edges() * 2),
                                  max_tries=G_perm.number_of_edges() * 20,
                                  seed=int(rng.integers(1e9)))
    lcc = max(nx.connected_components(G_perm), key=len)
    sub = G_perm.subgraph(lcc)
    bc = nx.betweenness_centrality(sub, normalized=True)
    return {n: bc.get(n, 0.0) for n in G_perm.nodes()}


def permutation_significance(
    G: nx.Graph,
    node_df: pd.DataFrame,
    n_perm: int,
    workers: int,
    seed: int,
) -> pd.DataFrame:
    log.info(f"Running {n_perm} permutation tests (degree-preserving)...")
    ss = np.random.SeedSequence(seed)
    child_seeds = [int(s.entropy) for s in ss.spawn(n_perm)]
    edges = list(G.edges())
    n_nodes = G.number_of_nodes()
    tasks = [(edges, n_nodes, s) for s in child_seeds]

    null_bc: dict[int, list[float]] = defaultdict(list)
    if workers <= 1:
        for t in tasks:
            for n, v in _perm_worker(t).items():
                null_bc[n].append(v)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_perm_worker, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                for n, v in fut.result().items():
                    null_bc[n].append(v)
                if i % 50 == 0:
                    log.info(f"  {i}/{n_perm} permutations done")

    rows = []
    for n in G.nodes():
        obs = node_df.loc[n, "betweenness"] if n in node_df.index else 0.0
        null = np.array(null_bc.get(n, [0.0]))
        mu, sd = null.mean(), null.std()
        z = (obs - mu) / sd if sd > 0 else np.nan
        p = (np.sum(null >= obs) + 1) / (len(null) + 1)
        rows.append({"node_id": n, "perm_z": z, "perm_p": p,
                     "perm_null_mean": mu, "perm_null_std": sd})
    return pd.DataFrame(rows).set_index("node_id")


# --------------------------------------------------------------------------- #
# Population test (vs full PPI)
# --------------------------------------------------------------------------- #
def population_significance(
    node_df: pd.DataFrame,
    ppi: nx.Graph,
) -> pd.DataFrame:
    log.info("Running population significance test (vs full PPI)...")
    ppi_bc = nx.betweenness_centrality(ppi, normalized=True)
    pop_vals = np.array(list(ppi_bc.values()))
    pop_mu, pop_sd = pop_vals.mean(), pop_vals.std()

    rows = []
    for n, row in node_df.iterrows():
        obs = row["betweenness"]
        z = (obs - pop_mu) / pop_sd if pop_sd > 0 else np.nan
        p = float(np.mean(pop_vals >= obs))
        rows.append({"node_id": n, "pop_z": z, "pop_p": p,
                     "pop_mu": pop_mu, "pop_sd": pop_sd})
    return pd.DataFrame(rows).set_index("node_id")


# --------------------------------------------------------------------------- #
# Cluster-level metrics
# --------------------------------------------------------------------------- #
def cluster_level_metrics(
    G: nx.Graph,
    node_df: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate node-level centrality to the community level.
    Here 'cluster' = layer (source::cell_type). For HiDeF communities,
    pass community assignments via node_df['community'] if available.
    """
    log.info("Computing cluster-level metrics...")
    rows = []
    for layer, sub_reg in registry.groupby("layer"):
        layer_nodes = [n for n in sub_reg.index if n in G]
        if not layer_nodes:
            continue
        sub = node_df.loc[[n for n in layer_nodes if n in node_df.index]]
        if sub.empty:
            continue
        rows.append({
            "layer": layer,
            "n_nodes": len(layer_nodes),
            "mean_betweenness": sub["betweenness"].mean(),
            "mean_eigenvector": sub["eigenvector"].mean(),
            "mean_clustering": sub["clustering"].mean(),
            "mean_degree": sub["degree"].mean(),
            "mean_participation": sub["participation_coef"].mean(),
            "max_betweenness": sub["betweenness"].max(),
            "max_degree": sub["degree"].max(),
        })
    return pd.DataFrame(rows).set_index("layer")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--ppi", type=Path, required=True)
    ap.add_argument("--outfile", type=Path, default=Path("centrality_results.pkl"))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    G = load_graph(args.graph)
    registry = load_registry(args.registry)
    ppi = load_ppi(args.ppi)

    global_metrics = compute_global_metrics(G)
    node_df = compute_node_centrality(G, registry)
    layer_deg_df = per_layer_degree(G, registry)

    perm_df = permutation_significance(G, node_df, args.n_perm, args.workers, args.seed)
    pop_df = population_significance(node_df, ppi)
    cluster_df = cluster_level_metrics(G, node_df, registry)

    node_df = node_df.join(perm_df, how="left").join(pop_df, how="left")

    results = {
        "global_metrics": global_metrics,
        "node_df": node_df,
        "layer_degree_df": layer_deg_df,
        "cluster_df": cluster_df,
        "registry": registry,
    }
    with open(args.outfile, "wb") as f:
        pickle.dump(results, f)
    log.info(f"Saved → {args.outfile}")


if __name__ == "__main__":
    main()