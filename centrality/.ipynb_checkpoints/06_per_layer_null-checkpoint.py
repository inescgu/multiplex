#!/usr/bin/env python3
"""
06_per_layer_null.py

Per (community, layer) null comparison: is the observed subgraph density
(and other metrics) higher than expected given the layer's own background?

Consumes:
  - integrated_multiplex.pkl
  - node_registry.csv
  - persistent_communities.pkl
  - community_layer_subgraph_metrics.csv   (output of 05_)

Two complementary nulls per (community, layer) with k = n_nodes in that cell:

  1) random_subgraph: sample k nodes uniformly at random from the FULL layer
     (induced subgraph on the layer). Compares the community to "any random
     bag of k nodes living in this layer".

  2) degree_matched: sample k nodes from the layer with probability
     proportional to their degree within the layer subgraph. Controls for the
     fact that high-degree nodes inflate density just by being picked.

For each (community, layer) we report observed value, null mean/std, z-score,
and empirical p-value (two-sided) for: density, avg_clustering,
largest_cc_fraction.

Produces (in --outdir):
  - per_layer_null_results.csv
  - plots/
      - per_layer_density_zscore_hist.png
      - per_layer_density_volcano.png        (z vs -log10(p), colored by layer)
      - top_enriched_community_layers.png    (bar chart of top z-scores)

Usage:
  python 06_per_layer_null.py \
      --graph integrated_multiplex.pkl \
      --registry node_registry.csv \
      --communities persistent_communities.pkl \
      --observed multiplex_community_out/community_layer_subgraph_metrics.csv \
      --outdir multiplex_community_out \
      --n-null 200 \
      --min-k 3 \
      --seed 42 \
      --workers 1
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
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("per_layer_null")


# --------------------------------------------------------------------------- #
# I/O
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
# Per-layer background
# --------------------------------------------------------------------------- #
def build_layer_backgrounds(
    G: nx.Graph, registry: pd.DataFrame
) -> dict[str, dict]:
    """
    For each layer, precompute:
      - node array
      - induced subgraph (used as the universe to sample from)
      - degree array (within that layer subgraph) for degree-weighted sampling
    """
    log.info("Building per-layer backgrounds...")
    out = {}
    for layer, sub in registry.groupby("layer"):
        nodes = [n for n in sub.index if n in G]
        if not nodes:
            continue
        H = G.subgraph(nodes).copy()
        deg = np.array([H.degree(n) for n in nodes], dtype=float)
        deg_p = deg / deg.sum() if deg.sum() > 0 else None
        out[layer] = {
            "nodes": np.array(nodes),
            "graph": H,
            "deg_p": deg_p,
        }
        log.info(f"  layer {layer}: |V|={H.number_of_nodes():,}  "
                 f"|E|={H.number_of_edges():,}")
    return out


def _metrics_of(sub: nx.Graph) -> tuple[float, float, float]:
    n = sub.number_of_nodes()
    if n == 0:
        return 0.0, 0.0, 0.0
    dens = nx.density(sub) if n > 1 else 0.0
    try:
        clust = nx.average_clustering(sub) if n >= 3 else 0.0
    except Exception:
        clust = 0.0
    ccs = list(nx.connected_components(sub))
    largest = max((len(c) for c in ccs), default=0)
    largest_frac = largest / n if n else 0.0
    return dens, clust, largest_frac


def _empirical_p(obs: float, null: np.ndarray) -> float:
    """Two-sided empirical p-value with +1 smoothing."""
    if len(null) == 0:
        return np.nan
    centered_obs = abs(obs - null.mean())
    centered_null = np.abs(null - null.mean())
    return (np.sum(centered_null >= centered_obs) + 1) / (len(null) + 1)


def null_for_cell(
    H: nx.Graph,
    nodes: np.ndarray,
    deg_p: np.ndarray | None,
    k: int,
    n_null: int,
    rng: np.random.Generator,
    mode: str,
) -> dict[str, np.ndarray]:
    """Sample n_null subgraphs of size k from layer subgraph H."""
    dens_n = np.empty(n_null)
    clust_n = np.empty(n_null)
    lcc_n = np.empty(n_null)
    N = len(nodes)
    if k > N:
        k = N
    for i in range(n_null):
        if mode == "degree_matched" and deg_p is not None:
            picked = rng.choice(nodes, size=k, replace=False, p=deg_p)
        else:
            picked = rng.choice(nodes, size=k, replace=False)
        s = H.subgraph(picked)
        d, c, l = _metrics_of(s)
        dens_n[i] = d
        clust_n[i] = c
        lcc_n[i] = l
    return {"density": dens_n, "avg_clustering": clust_n,
            "largest_cc_fraction": lcc_n}


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _process_cell(args):
    (cid, layer, k, obs_dens, obs_clust, obs_lcc,
     nodes, edges, deg_p, n_null, seed) = args
    rng = np.random.default_rng(seed)
    H = nx.Graph()
    H.add_nodes_from(nodes.tolist())
    H.add_edges_from(edges)

    results = {}
    for mode in ("random_subgraph", "degree_matched"):
        if mode == "degree_matched" and deg_p is None:
            continue
        nulls = null_for_cell(H, nodes, deg_p, k, n_null, rng, mode)
        for metric, obs in (("density", obs_dens),
                            ("avg_clustering", obs_clust),
                            ("largest_cc_fraction", obs_lcc)):
            arr = nulls[metric]
            mu, sd = float(arr.mean()), float(arr.std())
            z = (obs - mu) / sd if sd > 0 else np.nan
            p = _empirical_p(obs, arr)
            results[(mode, metric)] = (mu, sd, z, p)

    return cid, layer, results


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run(
    G: nx.Graph,
    registry: pd.DataFrame,
    comms: dict[Any, set[int]],
    observed: pd.DataFrame,
    n_null: int,
    min_k: int,
    seed: int,
    workers: int,
) -> pd.DataFrame:
    backgrounds = build_layer_backgrounds(G, registry)

    # Build task list
    tasks = []
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(observed))
    for i, (_, row) in enumerate(observed.iterrows()):
        k = int(row["n_nodes"])
        if k < min_k:
            continue
        layer = row["layer"]
        if layer not in backgrounds:
            continue
        bg = backgrounds[layer]
        edges = list(bg["graph"].edges())
        tasks.append((
            row["community"], layer, k,
            float(row["density"]),
            float(row["avg_clustering"]),
            float(row["largest_cc_fraction"]),
            bg["nodes"], edges, bg["deg_p"],
            n_null, child_seeds[i].entropy,
        ))

    log.info(f"Running null tests on {len(tasks)} (community, layer) cells "
             f"with n_null={n_null}, workers={workers}")

    rows = []
    if workers <= 1:
        for t in tasks:
            cid, layer, res = _process_cell(t)
            rows.append((cid, layer, res))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_cell, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 50 == 0:
                    log.info(f"  {i}/{len(futs)} cells done")

    # Flatten
    out_rows = []
    for cid, layer, res in rows:
        rec = {"community": cid, "layer": layer}
        for (mode, metric), (mu, sd, z, p) in res.items():
            rec[f"{mode}__{metric}__null_mean"] = mu
            rec[f"{mode}__{metric}__null_std"] = sd
            rec[f"{mode}__{metric}__zscore"] = z
            rec[f"{mode}__{metric}__pvalue"] = p
        out_rows.append(rec)
    df = pd.DataFrame(out_rows)
    # Attach observed back for convenience
    df = df.merge(
        observed[["community", "layer", "n_nodes", "density",
                  "avg_clustering", "largest_cc_fraction"]],
        on=["community", "layer"], how="left",
    )
    return df


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(df: pd.DataFrame, outdir: Path):
    if df.empty:
        return
    sns.set_style("whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)

    zcol = "random_subgraph__density__zscore"
    pcol = "random_subgraph__density__pvalue"
    if zcol in df.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(df[zcol].dropna(), bins=40, color="steelblue", edgecolor="k")
        plt.axvline(0, ls="--", color="k", lw=0.8)
        plt.xlabel("Density z-score (vs random subgraph null)")
        plt.ylabel("# (community, layer) cells")
        plt.title("Per-layer density enrichment")
        plt.tight_layout()
        plt.savefig(outdir / "per_layer_density_zscore_hist.png", dpi=200)
        plt.close()

    if zcol in df.columns and pcol in df.columns:
        plt.figure(figsize=(7, 5))
        layers = sorted(df["layer"].unique())
        palette = sns.color_palette("tab20", n_colors=len(layers))
        for ly, color in zip(layers, palette):
            sub = df[df["layer"] == ly]
            plt.scatter(sub[zcol], -np.log10(sub[pcol].clip(lower=1e-6)),
                        s=14, alpha=0.6, color=color, label=ly)
        plt.axhline(-np.log10(0.05), ls="--", color="k", lw=0.8)
        plt.xlabel("Density z-score")
        plt.ylabel("-log10(empirical p)")
        plt.title("Volcano: per (community, layer) density enrichment")
        if len(layers) <= 20:
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        plt.savefig(outdir / "per_layer_density_volcano.png", dpi=200)
        plt.close()

        top = df.dropna(subset=[zcol]).nlargest(25, zcol)
        if not top.empty:
            plt.figure(figsize=(9, 6))
            labels = [f"C{c} | {l}" for c, l in zip(top["community"], top["layer"])]
            plt.barh(labels[::-1], top[zcol].values[::-1], color="darkorange")
            plt.xlabel("Density z-score")
            plt.title("Top 25 most density-enriched (community, layer) cells")
            plt.tight_layout()
            plt.savefig(outdir / "top_enriched_community_layers.png", dpi=200)
            plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--communities", type=Path, required=True)
    ap.add_argument("--observed", type=Path, required=True,
                    help="community_layer_subgraph_metrics.csv from 05_")
    ap.add_argument("--outdir", type=Path, default=Path("multiplex_community_out"))
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--min-k", type=int, default=3,
                    help="Skip cells with fewer nodes than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.outdir / "plots"
    plot_dir.mkdir(exist_ok=True)

    G = load_graph(args.graph)
    registry = load_registry(args.registry)
    comms = load_communities(args.communities)
    observed = pd.read_csv(args.observed)

    df = run(G, registry, comms, observed,
             n_null=args.n_null, min_k=args.min_k,
             seed=args.seed, workers=args.workers)
    df.to_csv(args.outdir / "per_layer_null_results.csv", index=False)
    make_plots(df, plot_dir)
    log.info(f"Done. Outputs in {args.outdir}")


if __name__ == "__main__":
    main()