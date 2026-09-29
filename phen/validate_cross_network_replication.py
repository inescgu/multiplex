#!/usr/bin/env python3
"""
validate_cross_network_replication.py

FIX 5 -- You said the single-layer and multiplex networks "show similar
patterns with HiDeF" -- this turns that eyeball comparison into a number a
reviewer can't wave away.

Two complementary tests, both gene-level (multiplex nodes de-namespaced):

  A) CLUSTER-LEVEL CONCORDANCE. For each pair (monoplex cluster, multiplex
     cluster), compute gene-set Jaccard overlap. For each monoplex cluster,
     take its BEST-matching multiplex cluster (max Jaccard) and ask: is that
     match's persistence rank consistent with the monoplex cluster's own
     persistence rank? (Spearman correlation of matched persistence ranks.)
     This tests whether "broad" in one network structure corresponds to
     "broad" in the other, on genes both networks agree overlap.

  B) SHARED-GENE PERSISTENCE CONCORDANCE. Restrict to genes present in BOTH
     modules. For each such gene, look up (a) the persistence of its
     monoplex cluster and (b) the persistence of its multiplex cluster.
     Correlate the two persistence values gene-by-gene. This is a stronger,
     more granular test than (A) because it doesn't require whole clusters
     to match -- it just asks whether a gene sitting in a "broad" region of
     one network's hierarchy also sits in a "broad" region of the other's.

USAGE
-----
    python validate_cross_network_replication.py \
        --mono-communities-pkl hidef_results/hidef_communities.pkl \
        --multi-nodes-file heterogeneous_hidef_faithful_cpm.nodes \
        --multi-supra-graph supra_graph.pkl \
        --multi-persistence-threshold 8 \
        --output-prefix cross_network/replication
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hidef_multiplex_adapter import load_multiplex_clusters


def load_mono_clusters(pkl_path):
    """Monoplex hidef_communities.pkl: {comm_id: {'members': [...], 'persistence': ...}}"""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    clusters, persistence = {}, {}
    for cid, c in raw.items():
        name = f"Mono_Cluster_{cid}"
        clusters[name] = sorted(set(str(g) for g in c["members"]))
        persistence[name] = float(c.get("persistence", 0.0))
    return clusters, persistence


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0


def best_match_table(clusters_a, persistence_a, clusters_b, persistence_b, min_overlap=3):
    rows = []
    for a_name, a_genes in clusters_a.items():
        best_j, best_b = 0.0, None
        for b_name, b_genes in clusters_b.items():
            if len(set(a_genes) & set(b_genes)) < min_overlap:
                continue
            j = jaccard(a_genes, b_genes)
            if j > best_j:
                best_j, best_b = j, b_name
        if best_b is not None:
            rows.append({
                "cluster_a": a_name, "persistence_a": persistence_a[a_name], "size_a": len(a_genes),
                "best_match_b": best_b, "persistence_b": persistence_b[best_b],
                "size_b": len(clusters_b[best_b]), "jaccard": best_j,
            })
    return pd.DataFrame(rows)


def shared_gene_concordance(clusters_a, persistence_a, clusters_b, persistence_b):
    gene_to_persist_a, gene_to_persist_b = {}, {}
    for name, genes in clusters_a.items():
        for g in genes:
            # if a gene is in multiple clusters, keep the max persistence it participates in
            gene_to_persist_a[g] = max(gene_to_persist_a.get(g, -np.inf), persistence_a[name])
    for name, genes in clusters_b.items():
        for g in genes:
            gene_to_persist_b[g] = max(gene_to_persist_b.get(g, -np.inf), persistence_b[name])

    shared = sorted(set(gene_to_persist_a) & set(gene_to_persist_b))
    df = pd.DataFrame({
        "gene": shared,
        "persistence_mono": [gene_to_persist_a[g] for g in shared],
        "persistence_multi": [gene_to_persist_b[g] for g in shared],
    })
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mono-communities-pkl", required=True)
    ap.add_argument("--multi-nodes-file", required=True)
    ap.add_argument("--multi-supra-graph", default=None)
    ap.add_argument("--multi-persistence-threshold", type=int, default=5)
    ap.add_argument("--min-overlap", type=int, default=3,
                     help="Minimum shared genes to consider two clusters a candidate match")
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    print("Loading monoplex clusters ...")
    mono_clusters, mono_persist = load_mono_clusters(args.mono_communities_pkl)
    print(f"  {len(mono_clusters)} monoplex clusters")

    print("Loading multiplex clusters ...")
    node_names = None
    if args.multi_supra_graph:
        with open(args.multi_supra_graph, "rb") as f:
            g = pickle.load(f)
        node_names = g.vs["name"]
    multi_clusters, _, extra = load_multiplex_clusters(
        args.multi_nodes_file, node_names=node_names,
        persistence_threshold=args.multi_persistence_threshold)
    multi_persist = extra["persistence"]
    print(f"  {len(multi_clusters)} multiplex clusters")

    # ---- A) cluster-level best-match concordance ----
    print("\n[A] Best-match cluster-level concordance ...")
    match_df = best_match_table(mono_clusters, mono_persist, multi_clusters, multi_persist,
                                 min_overlap=args.min_overlap)
    match_df.to_csv(f"{args.output_prefix}_cluster_matches.csv", index=False)
    if len(match_df) >= 5:
        rho_a, p_a = stats.spearmanr(match_df["persistence_a"], match_df["persistence_b"])
        print(f"  Matched {len(match_df)} monoplex clusters to a multiplex best-match "
              f"(min_overlap>={args.min_overlap}).")
        print(f"  Spearman(persistence_mono, persistence_multi) across matches: "
              f"rho={rho_a:+.3f}  p={p_a:.2e}")
    else:
        rho_a, p_a = np.nan, np.nan
        print("  Too few matched cluster pairs for a meaningful correlation.")

    # ---- B) shared-gene persistence concordance ----
    print("\n[B] Shared-gene persistence concordance ...")
    gene_df = shared_gene_concordance(mono_clusters, mono_persist, multi_clusters, multi_persist)
    gene_df.to_csv(f"{args.output_prefix}_shared_gene_persistence.csv", index=False)
    if len(gene_df) >= 10:
        rho_b, p_b = stats.spearmanr(gene_df["persistence_mono"], gene_df["persistence_multi"])
        print(f"  {len(gene_df)} genes present in both networks' modules.")
        print(f"  Spearman(persistence_mono, persistence_multi) gene-by-gene: "
              f"rho={rho_b:+.3f}  p={p_b:.2e}")
    else:
        rho_b, p_b = np.nan, np.nan
        print("  Too few shared genes for a meaningful correlation.")

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    if len(match_df):
        axes[0].scatter(match_df["persistence_a"], match_df["persistence_b"], s=18, alpha=0.6)
        axes[0].set_xlabel("monoplex cluster persistence")
        axes[0].set_ylabel("best-match multiplex cluster persistence")
        axes[0].set_title(f"Cluster best-match (rho={rho_a:+.2f})")
        axes[0].grid(alpha=0.3)
    if len(gene_df):
        axes[1].scatter(gene_df["persistence_mono"], gene_df["persistence_multi"], s=10, alpha=0.4)
        axes[1].set_xlabel("gene's monoplex-cluster persistence")
        axes[1].set_ylabel("gene's multiplex-cluster persistence")
        axes[1].set_title(f"Shared-gene concordance (rho={rho_b:+.2f})")
        axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{args.output_prefix}_concordance_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame([
        {"test": "A_cluster_best_match", "rho": rho_a, "p_value": p_a, "n": len(match_df)},
        {"test": "B_shared_gene", "rho": rho_b, "p_value": p_b, "n": len(gene_df)},
    ])
    summary.to_csv(f"{args.output_prefix}_summary.csv", index=False)
    print(f"\nSaved summary -> {args.output_prefix}_summary.csv")
    print("If both rho are positive and significant, that's real, quantified evidence "
          "the persistence-generality structure replicates across two very differently "
          "built networks -- much stronger than 'both pipelines produced HiDeF hierarchies "
          "that looked similar.'")


if __name__ == "__main__":
    main()