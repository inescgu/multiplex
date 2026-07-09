#!/usr/bin/env python3
"""
03_community_analysis.py  (multiplex-aware)

Multi-resolution community analysis with full layer-awareness.

Key multiplex additions vs original:
  - Communities operate on integer node IDs (from NodeRegistry)
  - All structural metrics computed on the multiplex graph
  - Per-community layer composition, entropy, participation coefficient
  - Edge-type breakdown (ppi / pathway / ligand_receptor) per community
  - Intra- vs inter-layer edge counts per community
  - Gene multiplicity: how many layers each gene appears in
  - Gene-community membership table enriched with layer metadata

Consumes:
  - integrated_multiplex.pkl
  - node_registry.csv
  - persistent_communities.pkl

Produces (in --outdir):
  - community_analysis.csv
  - gene_community_membership.csv
  - gene_multiplicity.csv
  - plots/

Usage:
  python 03_community_analysis.py \
      --graph integrated_multiplex.pkl \
      --registry node_registry.csv \
      --communities persistent_communities.pkl \
      --outdir community_out
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
from scipy.stats import entropy as scipy_entropy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("community_analysis")


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
    required = {"node_id", "source", "cell_type", "gene"}
    if missing := required - set(df.columns):
        raise ValueError(f"Registry missing: {missing}")
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
# Community structural metrics
# --------------------------------------------------------------------------- #
def analyze_communities(
    G: nx.Graph,
    comms: dict[Any, set[int]],
    registry: pd.DataFrame,
    persistence: dict[Any, float],
) -> pd.DataFrame:
    log.info(f"Analyzing {len(comms)} communities...")
    all_layers = sorted(registry["layer"].unique())
    L = len(all_layers)
    rows = []

    for cid, nodes in comms.items():
        valid = [n for n in nodes if n in G]
        if not valid:
            continue
        sub = G.subgraph(valid)
        n = sub.number_of_nodes()
        m = sub.number_of_edges()
        degrees = [d for _, d in sub.degree()]

        # Basic structural
        density = nx.density(sub) if n > 1 else 0.0
        avg_deg = np.mean(degrees) if degrees else 0.0
        avg_clust = nx.average_clustering(sub) if n >= 3 else 0.0
        ccs = list(nx.connected_components(sub))
        n_cc = len(ccs)
        largest_cc = max((len(c) for c in ccs), default=0)
        max_edges = n * (n - 1) / 2
        internal_edge_ratio = m / max_edges if max_edges > 0 else 0.0
        external_edges = len(list(nx.edge_boundary(G, set(valid))))

        # Layer composition
        node_layers = registry.loc[
            [nd for nd in valid if nd in registry.index], "layer"
        ].value_counts()
        layer_frac = node_layers / node_layers.sum()
        p = layer_frac.values
        H = float(scipy_entropy(p, base=2)) if len(p) > 1 else 0.0
        H_norm = H / np.log2(L) if L > 1 else 0.0
        participation = float(1.0 - np.sum(p ** 2))
        dominant_layer = layer_frac.idxmax() if len(layer_frac) else "unknown"
        n_layers_present = int((layer_frac > 0).sum())

        # Edge-type breakdown
        type_counts: Counter = Counter()
        intra_layer = 0
        inter_layer = 0
        for u, v, data in sub.edges(data=True):
            etype = data.get("edge_type", "unknown")
            type_counts[etype] += 1
            lu = registry.loc[u, "layer"] if u in registry.index else None
            lv = registry.loc[v, "layer"] if v in registry.index else None
            if lu is not None and lv is not None:
                if lu == lv:
                    intra_layer += 1
                else:
                    inter_layer += 1

        row = {
            "community_id": cid,
            "persistence": persistence.get(cid, np.nan),
            "n_nodes": n,
            "n_edges": m,
            "density": density,
            "avg_degree": avg_deg,
            "avg_clustering": avg_clust,
            "n_components": n_cc,
            "largest_component_size": largest_cc,
            "largest_component_frac": largest_cc / n if n else 0.0,
            "internal_edge_ratio": internal_edge_ratio,
            "external_edges": external_edges,
            "external_edge_ratio": external_edges / n if n else 0.0,
            # Layer metrics
            "n_layers_present": n_layers_present,
            "dominant_layer": dominant_layer,
            "dominant_layer_frac": float(layer_frac.max()) if len(layer_frac) else 0.0,
            "layer_entropy_bits": H,
            "layer_entropy_normalized": H_norm,
            "layer_participation_coef": participation,
            # Edge type
            "n_ppi_edges": type_counts.get("ppi", 0),
            "n_pathway_edges": type_counts.get("pathway", 0),
            "n_lr_edges": type_counts.get("ligand_receptor", 0),
            "n_unknown_edges": type_counts.get("unknown", 0),
            "frac_ppi": type_counts.get("ppi", 0) / m if m else 0.0,
            "frac_pathway": type_counts.get("pathway", 0) / m if m else 0.0,
            "frac_lr": type_counts.get("ligand_receptor", 0) / m if m else 0.0,
            "n_intra_layer_edges": intra_layer,
            "n_inter_layer_edges": inter_layer,
            "frac_intra_layer": intra_layer / m if m else 0.0,
            "frac_inter_layer": inter_layer / m if m else 0.0,
        }
        # Per-layer node fractions
        for ly in all_layers:
            row[f"frac_layer_{ly}"] = float(layer_frac.get(ly, 0.0))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("community_id")
    log.info(f"  {len(df)} communities analyzed")
    return df


# --------------------------------------------------------------------------- #
# Gene-community membership
# --------------------------------------------------------------------------- #
def gene_community_membership(
    comms: dict[Any, set[int]],
    registry: pd.DataFrame,
    persistence: dict[Any, float],
) -> pd.DataFrame:
    log.info("Building gene-community membership table...")
    node_to_comms: dict[int, list] = defaultdict(list)
    for cid, nodes in comms.items():
        p = persistence.get(cid, np.nan)
        for n in nodes:
            node_to_comms[n].append({"community_id": cid, "persistence": p})

    rows = []
    for node_id, comm_list in node_to_comms.items():
        meta = registry.loc[node_id] if node_id in registry.index else {}
        comm_list_sorted = sorted(comm_list,
                                   key=lambda x: x["persistence"]
                                   if not np.isnan(x["persistence"]) else -1,
                                   reverse=True)
        rows.append({
            "node_id": node_id,
            "gene": meta.get("gene", "?") if isinstance(meta, pd.Series) else "?",
            "source": meta.get("source", "?") if isinstance(meta, pd.Series) else "?",
            "cell_type": meta.get("cell_type", "?") if isinstance(meta, pd.Series) else "?",
            "layer": meta.get("layer", "?") if isinstance(meta, pd.Series) else "?",
            "n_communities": len(comm_list_sorted),
            "primary_community": comm_list_sorted[0]["community_id"],
            "primary_persistence": comm_list_sorted[0]["persistence"],
            "community_list": ",".join(str(c["community_id"]) for c in comm_list_sorted),
            "persistence_list": ",".join(
                f"{c['persistence']:.3f}" if not np.isnan(c["persistence"]) else "nan"
                for c in comm_list_sorted
            ),
        })
    df = pd.DataFrame(rows).sort_values("primary_persistence", ascending=False)
    return df


# --------------------------------------------------------------------------- #
# Gene multiplicity
# --------------------------------------------------------------------------- #
def gene_multiplicity(registry: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing gene multiplicity...")
    return (
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


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(comm_df: pd.DataFrame, gene_df: pd.DataFrame,
               mult_df: pd.DataFrame, outdir: Path):
    sns.set_style("whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Community size distribution
    plt.figure(figsize=(7, 4))
    plt.hist(comm_df["n_nodes"], bins=40, color="steelblue", edgecolor="k")
    plt.xlabel("Community size (# nodes)")
    plt.ylabel("# communities")
    plt.title("Community size distribution")
    plt.tight_layout()
    plt.savefig(outdir / "community_size_distribution.png", dpi=200)
    plt.close()

    # 2. Layer entropy distribution
    plt.figure(figsize=(7, 4))
    plt.hist(comm_df["layer_entropy_normalized"].dropna(), bins=30,
             color="darkorange", edgecolor="k")
    plt.xlabel("Normalized layer entropy")
    plt.ylabel("# communities")
    plt.title("Community layer entropy")
    plt.tight_layout()
    plt.savefig(outdir / "community_layer_entropy.png", dpi=200)
    plt.close()

    # 3. Edge type composition (top 30 by size)
    et_cols = ["frac_ppi", "frac_pathway", "frac_lr"]
    et_cols = [c for c in et_cols if c in comm_df.columns]
    if et_cols:
        top = comm_df.nlargest(30, "n_nodes")[et_cols]
        top.plot(kind="bar", stacked=True, figsize=(14, 5), colormap="tab10")
        plt.ylabel("Edge fraction")
        plt.title("Edge-type composition (top 30 communities by size)")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / "edge_type_composition.png", dpi=200)
        plt.close()

    # 4. Intra vs inter-layer edges
    if "frac_intra_layer" in comm_df.columns:
        plt.figure(figsize=(6, 6))
        plt.scatter(comm_df["frac_intra_layer"], comm_df["frac_inter_layer"],
                    s=comm_df["n_nodes"], alpha=0.6, c=comm_df["n_nodes"],
                    cmap="viridis", edgecolors="k", linewidth=0.5)
        plt.xlabel("Fraction intra-layer edges")
        plt.ylabel("Fraction inter-layer edges")
        plt.title("Intra vs inter-layer edge composition")
        cbar = plt.colorbar(label="Community size")
        plt.tight_layout()
        plt.savefig(outdir / "intra_inter_layer_edges.png", dpi=200)
        plt.close()

    # 5. Gene multiplicity distribution
    if mult_df is not None and len(mult_df) > 0:
        plt.figure(figsize=(7, 4))
        plt.hist(mult_df["n_layers"], bins=range(1, mult_df["n_layers"].max() + 2),
                 color="mediumseagreen", edgecolor="k", align="left")
        plt.xlabel("Number of layers")
        plt.ylabel("# genes")
        plt.title("Gene multiplicity distribution")
        plt.tight_layout()
        plt.savefig(outdir / "gene_multiplicity_distribution.png", dpi=200)
        plt.close()

    # 6. Participation coefficient distribution
    if "layer_participation_coef" in comm_df.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(comm_df["layer_participation_coef"].dropna(), bins=30,
                 color="crimson", edgecolor="k")
        plt.xlabel("Layer participation coefficient")
        plt.ylabel("# communities")
        plt.title("Community layer participation coefficient")
        plt.tight_layout()
        plt.savefig(outdir / "participation_coefficient.png", dpi=200)
        plt.close()

    log.info(f"Plots saved to {outdir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Multi-resolution community analysis with layer awareness"
    )
    parser.add_argument("--graph", type=Path, required=True,
                        help="Path to integrated_multiplex.pkl")
    parser.add_argument("--registry", type=Path, required=True,
                        help="Path to node_registry.csv")
    parser.add_argument("--communities", type=Path, required=True,
                        help="Path to persistent_communities.pkl")
    parser.add_argument("--outdir", type=Path, default=Path("community_out"),
                        help="Output directory")
    args = parser.parse_args()

    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.outdir / "plots"

    # Load data
    log.info("Loading data...")
    G = load_graph(args.graph)
    registry = load_registry(args.registry)
    comms, persistence = load_communities(args.communities)

    # Analyze communities
    comm_df = analyze_communities(G, comms, registry, persistence)
    comm_df.to_csv(args.outdir / "community_analysis.csv")
    log.info(f"Community analysis saved to {args.outdir / 'community_analysis.csv'}")

    # Gene-community membership
    gene_df = gene_community_membership(comms, registry, persistence)
    gene_df.to_csv(args.outdir / "gene_community_membership.csv", index=False)
    log.info(f"Gene-community membership saved to {args.outdir / 'gene_community_membership.csv'}")

    # Gene multiplicity
    mult_df = gene_multiplicity(registry)
    mult_df.to_csv(args.outdir / "gene_multiplicity.csv")
    log.info(f"Gene multiplicity saved to {args.outdir / 'gene_multiplicity.csv'}")

    # Generate plots
    make_plots(comm_df, gene_df, mult_df, plots_dir)

    log.info("Community analysis complete!")


if __name__ == "__main__":
    main()