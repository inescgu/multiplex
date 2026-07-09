#!/usr/bin/env python3
"""
04_multiplex_community_analysis.py

Layer-aware community analysis for the integrated multiplex interactome.

Consumes:
  - integrated_multiplex.pkl       (networkx graph from integrated_multiplex_builder.py)
  - node_registry.csv              (node_id -> source, cell_type, gene)
  - persistent_communities.pkl     (HiDeF output: list/dict of community -> set(node_ids)
                                    plus optional 'persistence' field)

Produces (in --outdir):
  - community_layer_composition.csv     (per community x layer fractions)
  - community_layer_summary.csv         (entropy, participation, dominant layer, ...)
  - community_edge_composition.csv      (per community: ppi/pathway/LR fractions,
                                         intra- vs inter-layer edge counts)
  - gene_multiplicity.csv               (per gene: #layers, layer list)
  - gene_cross_layer_coherence.csv      (per gene: #communities its copies fall into,
                                         coherence score, dominant community)
  - global_coherence_summary.csv        (aggregate coherence stats + null comparison)
  - plots/*.png                         (overview visualizations)

Usage:
  python 04_multiplex_community_analysis.py \
      --graph integrated_multiplex.pkl \
      --registry node_registry.csv \
      --communities persistent_communities.pkl \
      --outdir multiplex_community_out \
      --n-null 100 \
      --seed 42
"""

from __future__ import annotations

import argparse
import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("multiplex_community")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_graph(path: Path) -> nx.Graph:
    log.info(f"Loading multiplex graph from {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    log.info(f"  |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}")
    return G


def load_registry(path: Path) -> pd.DataFrame:
    log.info(f"Loading node registry from {path}")
    df = pd.read_csv(path)
    # Expect columns: node_id, source, cell_type, gene
    required = {"node_id", "source", "cell_type", "gene"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Registry missing columns: {missing}")
    df["layer"] = df["source"].astype(str) + "::" + df["cell_type"].astype(str)
    return df.set_index("node_id")


def load_communities(path: Path) -> dict[Any, set[int]]:
    """
    Accepts:
      - dict {community_id: iterable of node_ids}
      - dict {community_id: {'nodes': iterable, 'persistence': float}}
      - list of iterables (assigned auto IDs)
    """
    log.info(f"Loading communities from {path}")
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
    else:
        raise ValueError(f"Unsupported communities format: {type(raw)}")

    log.info(f"  loaded {len(comms)} communities")
    return comms, persistence


# --------------------------------------------------------------------------- #
# Per-community metrics
# --------------------------------------------------------------------------- #
def layer_composition(
    comms: dict[Any, set[int]],
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - wide composition (community x layer) of fractions
      - summary with entropy, normalized entropy, participation coefficient,
        dominant layer, dominant fraction, n_layers_present, size
    """
    log.info("Computing layer composition...")
    all_layers = sorted(registry["layer"].unique())
    L = len(all_layers)
    log.info(f"  {L} distinct layers")

    rows = []
    summary = []
    for cid, nodes in comms.items():
        valid = [n for n in nodes if n in registry.index]
        if not valid:
            continue
        layers = registry.loc[valid, "layer"].value_counts()
        size = layers.sum()
        frac = layers / size

        row = {"community": cid, "size": int(size)}
        for ly in all_layers:
            row[ly] = float(frac.get(ly, 0.0))
        rows.append(row)

        # Shannon entropy on layer distribution
        p = frac.values
        H = float(entropy(p, base=2)) if len(p) > 1 else 0.0
        H_norm = H / np.log2(L) if L > 1 else 0.0
        # Multiplex participation coefficient: 1 - sum(p_i^2)
        # (1 = perfectly mixed across L layers when p_i=1/L; 0 = concentrated)
        participation = float(1.0 - np.sum(p ** 2))

        summary.append({
            "community": cid,
            "size": int(size),
            "n_layers_present": int((frac > 0).sum()),
            "dominant_layer": frac.idxmax(),
            "dominant_fraction": float(frac.max()),
            "entropy_bits": H,
            "entropy_normalized": H_norm,
            "participation_coef": participation,
        })

    comp_df = pd.DataFrame(rows).set_index("community")
    summary_df = pd.DataFrame(summary).set_index("community")
    return comp_df, summary_df


def edge_composition(
    G: nx.Graph,
    comms: dict[Any, set[int]],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each community, compute:
      - n_edges_internal
      - fraction of edges by edge_type (ppi / pathway / ligand_receptor)
      - intra-layer vs inter-layer edge counts (using each endpoint's layer)
    Assumes edges have attribute 'edge_type' (set by your builder) and that
    endpoint layers can be derived from the registry.
    """
    log.info("Computing edge composition...")
    rows = []
    for cid, nodes in comms.items():
        node_set = set(nodes)
        sub = G.subgraph(node_set)
        n_edges = sub.number_of_edges()
        type_counts: Counter = Counter()
        intra = 0
        inter = 0

        for u, v, data in sub.edges(data=True):
            etype = data.get("edge_type", "unknown")
            type_counts[etype] += 1

            lu = registry.loc[u, "layer"] if u in registry.index else None
            lv = registry.loc[v, "layer"] if v in registry.index else None
            if lu is not None and lv is not None:
                if lu == lv:
                    intra += 1
                else:
                    inter += 1

        row = {
            "community": cid,
            "n_edges_internal": int(n_edges),
            "n_intra_layer_edges": int(intra),
            "n_inter_layer_edges": int(inter),
            "frac_intra_layer": (intra / n_edges) if n_edges else 0.0,
            "frac_inter_layer": (inter / n_edges) if n_edges else 0.0,
        }
        for et in ("ppi", "pathway", "ligand_receptor", "unknown"):
            row[f"n_{et}"] = int(type_counts.get(et, 0))
            row[f"frac_{et}"] = (type_counts.get(et, 0) / n_edges) if n_edges else 0.0
        rows.append(row)

    return pd.DataFrame(rows).set_index("community")


# --------------------------------------------------------------------------- #
# Gene-level metrics
# --------------------------------------------------------------------------- #
def gene_multiplicity(registry: pd.DataFrame) -> pd.DataFrame:
    """
    Per gene: which layers it appears in, how many.
    """
    log.info("Computing gene multiplicity...")
    g = (
        registry.reset_index()
        .groupby("gene")
        .agg(
            n_layers=("layer", "nunique"),
            layers=("layer", lambda s: ";".join(sorted(set(s)))),
            n_nodes=("node_id", "count"),
            node_ids=("node_id", lambda s: ";".join(map(str, sorted(s)))),
        )
        .sort_values("n_layers", ascending=False)
    )
    return g


def cross_layer_coherence(
    comms: dict[Any, set[int]],
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    For each gene with >=2 layer copies, check how its copies distribute
    across communities.

    Coherence score = (size of largest community group) / (n copies assigned)
      1.0  -> all copies in the same community
      1/k  -> copies maximally scattered across k communities
    """
    log.info("Computing cross-layer coherence...")
    # Map node_id -> community
    node_to_comm: dict[int, Any] = {}
    for cid, nodes in comms.items():
        for n in nodes:
            node_to_comm[n] = cid

    rows = []
    for gene, sub in registry.reset_index().groupby("gene"):
        node_ids = sub["node_id"].tolist()
        if len(node_ids) < 2:
            continue
        comm_assignments = [node_to_comm.get(n) for n in node_ids]
        assigned = [c for c in comm_assignments if c is not None]
        if not assigned:
            continue
        counts = Counter(assigned)
        top_comm, top_n = counts.most_common(1)[0]
        coherence = top_n / len(assigned)
        rows.append({
            "gene": gene,
            "n_copies_total": len(node_ids),
            "n_copies_assigned": len(assigned),
            "n_distinct_communities": len(counts),
            "dominant_community": top_comm,
            "dominant_fraction": coherence,
            "coherence_score": coherence,
        })

    df = pd.DataFrame(rows).sort_values("coherence_score", ascending=False)

    # Global summary
    summary = {
        "n_multilayer_genes": float(len(df)),
        "mean_coherence": float(df["coherence_score"].mean()) if len(df) else np.nan,
        "median_coherence": float(df["coherence_score"].median()) if len(df) else np.nan,
        "frac_genes_fully_coherent": float((df["coherence_score"] == 1.0).mean())
        if len(df) else np.nan,
    }
    return df, summary


def null_coherence(
    registry: pd.DataFrame,
    comms: dict[Any, set[int]],
    n_null: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """
    Null model: shuffle community labels across nodes (size-preserving) and
    recompute mean coherence. Reports null mean/std and z-score of observed.
    """
    log.info(f"Running null coherence with {n_null} shuffles...")
    rng = np.random.default_rng(seed)

    all_assigned_nodes = [n for nodes in comms.values() for n in nodes]
    community_labels = []
    for cid, nodes in comms.items():
        community_labels.extend([cid] * len(nodes))
    community_labels = np.array(community_labels, dtype=object)

    # Gene -> indices into all_assigned_nodes (only for nodes that ARE assigned)
    node_index = {n: i for i, n in enumerate(all_assigned_nodes)}
    gene_to_idx: dict[str, list[int]] = defaultdict(list)
    for n in all_assigned_nodes:
        if n in registry.index:
            gene_to_idx[registry.loc[n, "gene"]].append(node_index[n])
    multilayer_idx = {g: idx for g, idx in gene_to_idx.items() if len(idx) >= 2}

    null_means = []
    for k in range(n_null):
        shuffled = community_labels.copy()
        rng.shuffle(shuffled)
        scores = []
        for idx in multilayer_idx.values():
            assigned = shuffled[idx]
            counts = Counter(assigned)
            _, top_n = counts.most_common(1)[0]
            scores.append(top_n / len(assigned))
        if scores:
            null_means.append(np.mean(scores))

    null_means = np.asarray(null_means)
    return {
        "null_mean_coherence_mean": float(null_means.mean()) if len(null_means) else np.nan,
        "null_mean_coherence_std": float(null_means.std()) if len(null_means) else np.nan,
        "n_null_iterations": int(len(null_means)),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(
    comp_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    multiplicity_df: pd.DataFrame,
    coherence_df: pd.DataFrame,
    outdir: Path,
) -> None:
    log.info("Generating plots...")
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Layer composition heatmap (top 30 communities by size)
    top = summary_df.sort_values("size", ascending=False).head(30).index
    sub = comp_df.loc[top].drop(columns=["size"], errors="ignore")
    if not sub.empty:
        plt.figure(figsize=(max(8, 0.4 * sub.shape[1] + 4), max(6, 0.3 * len(top))))
        sns.heatmap(sub, cmap="viridis", cbar_kws={"label": "fraction"})
        plt.title("Layer composition (top 30 communities by size)")
        plt.tight_layout()
        plt.savefig(outdir / "layer_composition_heatmap.png", dpi=200)
        plt.close()

    # 2. Entropy / participation distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(summary_df["entropy_normalized"].dropna(), bins=30,
                 color="steelblue", edgecolor="k")
    axes[0].set_xlabel("Normalized layer entropy")
    axes[0].set_ylabel("# communities")
    axes[0].set_title("Layer entropy")
    axes[1].hist(summary_df["participation_coef"].dropna(), bins=30,
                 color="darkorange", edgecolor="k")
    axes[1].set_xlabel("Participation coefficient")
    axes[1].set_title("Layer participation")
    plt.tight_layout()
    plt.savefig(outdir / "layer_entropy_participation.png", dpi=200)
    plt.close()

    # 3. Edge type composition stacked bar (top 30 by size)
    et_cols = [c for c in edge_df.columns if c.startswith("frac_")
               and c not in ("frac_intra_layer", "frac_inter_layer")]
    if et_cols:
        sub = edge_df.loc[edge_df.index.intersection(top), et_cols]
        if not sub.empty:
            sub.plot(kind="bar", stacked=True, figsize=(14, 5), colormap="tab10")
            plt.ylabel("Edge fraction")
            plt.title("Edge-type composition (top 30 communities)")
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            plt.tight_layout()
            plt.savefig(outdir / "edge_type_composition.png", dpi=200)
            plt.close()

    # 4. Intra vs inter-layer fraction scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(edge_df["frac_intra_layer"], edge_df["frac_inter_layer"],
                alpha=0.5, s=20)
    plt.xlabel("Fraction intra-layer edges")
    plt.ylabel("Fraction inter-layer edges")
    plt.title("Intra vs inter-layer edge fractions per community")
    plt.plot([0, 1], [1, 0], "k--", lw=0.8)
    plt.tight_layout()
    plt.savefig(outdir / "intra_vs_inter_layer.png", dpi=200)
    plt.close()

    # 5. Gene multiplicity distribution
    plt.figure(figsize=(7, 4))
    multiplicity_df["n_layers"].value_counts().sort_index().plot(
        kind="bar", color="seagreen", edgecolor="k"
    )
    plt.xlabel("# layers gene appears in")
    plt.ylabel("# genes")
    plt.title("Gene multiplicity")
    plt.tight_layout()
    plt.savefig(outdir / "gene_multiplicity.png", dpi=200)
    plt.close()

    # 6. Cross-layer coherence distribution
    if not coherence_df.empty:
        plt.figure(figsize=(7, 4))
        plt.hist(coherence_df["coherence_score"], bins=30,
                 color="crimson", edgecolor="k")
        plt.xlabel("Coherence score (largest fraction in one community)")
        plt.ylabel("# multi-layer genes")
        plt.title("Cross-layer coherence")
        plt.tight_layout()
        plt.savefig(outdir / "cross_layer_coherence.png", dpi=200)
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
    ap.add_argument("--n-null", type=int, default=100,
                    help="Null shuffles for coherence (0 to skip)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "plots").mkdir(exist_ok=True)

    G = load_graph(args.graph)
    registry = load_registry(args.registry)
    comms, persistence = load_communities(args.communities)

    # --- 1 & 2. Layer composition + entropy/participation
    comp_df, summary_df = layer_composition(comms, registry)
    if persistence:
        summary_df["persistence"] = pd.Series(persistence)

    comp_df.to_csv(args.outdir / "community_layer_composition.csv")
    summary_df.to_csv(args.outdir / "community_layer_summary.csv")

    # --- 3 & 4. Edge composition + intra/inter-layer
    edge_df = edge_composition(G, comms, registry)
    edge_df.to_csv(args.outdir / "community_edge_composition.csv")

    # --- 5. Gene multiplicity
    mult_df = gene_multiplicity(registry)
    mult_df.to_csv(args.outdir / "gene_multiplicity.csv")

    # --- 6. Cross-layer coherence (+ null)
    coh_df, coh_summary = cross_layer_coherence(comms, registry)
    coh_df.to_csv(args.outdir / "gene_cross_layer_coherence.csv", index=False)

    global_summary = dict(coh_summary)
    if args.n_null > 0 and coh_summary.get("n_multilayer_genes", 0):
        null_stats = null_coherence(registry, comms,
                                    n_null=args.n_null, seed=args.seed)
        global_summary.update(null_stats)
        if null_stats["null_mean_coherence_std"] > 0:
            z = (coh_summary["mean_coherence"] -
                 null_stats["null_mean_coherence_mean"]) / null_stats["null_mean_coherence_std"]
            global_summary["coherence_zscore_vs_null"] = float(z)

    pd.DataFrame([global_summary]).to_csv(
        args.outdir / "global_coherence_summary.csv", index=False
    )

    # --- Plots
    make_plots(comp_df, summary_df, edge_df, mult_df, coh_df,
               args.outdir / "plots")

    log.info(f"Done. Outputs in {args.outdir}")


if __name__ == "__main__":
    main()