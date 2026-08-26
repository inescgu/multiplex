"""
step8e_forced_k_and_compare.py
------------------------------------------------------------------------------------
Steps 1 + 2 of the multiplex-vs-single-layer comparison:

  1. Report the cluster size distribution of an existing spectral_clusters.csv
     (e.g. no_ppi's silhouette-picked k=3, to check whether that spike was a
     real balanced 3-way split or one dominant cluster + tiny leftovers).

  2. Force-cluster the SAME cached embedding.npy at a chosen k (default 2,
     to match the single-layer script's validated k=2 hub/periphery split),
     using the same row-normalize-then-kmeans recipe step8 now uses. No SVD
     recompute -- embedding.npy is reused as-is.

     If --single-layer-csv is given (gene_roles.csv from
     spectral_clustering_gene_roles.py), also computes the Adjusted Rand
     Index between the forced-k multiplex clustering and the single-layer
     clustering, restricted to genes present in both.

NOTE on gene-level comparison: the multiplex module has one node per
(cell_type, gene) pair -- e.g. multiple "<celltype>::TP53" nodes for a gene
that appears in several cell types -- while the single-layer network has one
node per gene, full stop. --aggregate controls how multiplex nodes collapse
to one label per gene before comparison:
  majority           : most common cluster label across that gene's cell-type
                        instances (default)
  first               : just take the first instance found (arbitrary but fast)
  restrict-celltype   : only use nodes from one cell type (e.g. global_ppi),
                        which is the closest structural analog to a flat
                        single-layer PPI network -- only meaningful for the
                        with_ppi run, since global_ppi nodes don't exist in
                        no_ppi's graph.

Usage:
  python step8e_forced_k_and_compare.py \
      --clusters-csv 08_spectral_analysis/module_with_ppi/spectral_clusters.csv \
      --embedding 08_spectral_analysis/module_with_ppi/embedding.npy \
      --force-k 2 \
      --single-layer-csv output/<ppi>/spectral_clustering/gene_roles.csv \
      --aggregate restrict-celltype --restrict-celltype global_ppi \
      --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
      --label with_ppi

  python step8e_forced_k_and_compare.py \
      --clusters-csv 08_spectral_analysis/module_no_ppi/spectral_clusters.csv \
      --embedding 08_spectral_analysis/module_no_ppi/embedding.npy \
      --force-k 2 \
      --single-layer-csv output/<ppi>/spectral_clustering/gene_roles.csv \
      --aggregate majority \
      --out-dir 08_spectral_analysis/module_no_ppi/diagnostics \
      --label no_ppi
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, silhouette_score


def report_cluster_sizes(df: pd.DataFrame, cluster_col: str, label: str):
    sizes = df[cluster_col].value_counts().sort_index()
    print(f"\n=== Cluster sizes: {label} ({cluster_col}) ===")
    for cid, cnt in sizes.items():
        print(f"  cluster {cid}: {cnt:,} genes ({100*cnt/len(df):.1f}%)")
    if len(sizes) > 1 and sizes.min() > 0:
        print(f"  largest/smallest ratio: {sizes.max()/sizes.min():.2f}  "
              f"(near 1 = balanced split; large = one dominant cluster + small leftovers)")
    return sizes


def force_cluster(embedding: np.ndarray, k: int, seed: int = 42,
                   row_normalize: bool = True, n_init: int = 20):
    emb_k = embedding[:, :k]
    if row_normalize:
        emb_k = normalize(emb_k)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(emb_k)
    sil = silhouette_score(emb_k, km.labels_) if k > 1 else float("nan")
    return km.labels_, sil


def aggregate_multiplex_to_gene(df: pd.DataFrame, cluster_col: str,
                                method: str = "majority", restrict_celltype: str = None):
    work = df.copy()
    if method == "restrict-celltype":
        if not restrict_celltype:
            raise SystemExit("--aggregate restrict-celltype requires --restrict-celltype.")
        work = work[work["cell_type"].str.upper() == restrict_celltype.upper()]
        if work.empty:
            raise SystemExit(f"No nodes with cell_type == {restrict_celltype!r} found "
                              f"in this run's spectral_clusters.csv.")
        return work.groupby("gene")[cluster_col].first()
    elif method == "majority":
        return work.groupby("gene")[cluster_col].agg(lambda s: s.value_counts().idxmax())
    elif method == "first":
        return work.groupby("gene")[cluster_col].first()
    else:
        raise ValueError(f"unknown aggregation method: {method}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters-csv", required=True,
                     help="Existing spectral_clusters.csv from step8 -- row order must "
                          "match --embedding (true for anything step8 produced).")
    ap.add_argument("--embedding", required=True, help="Cached embedding.npy from the same run.")
    ap.add_argument("--force-k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-row-normalize", action="store_true",
                     help="Match step8's --no-row-normalize if you used it upstream.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--single-layer-csv", default=None,
                     help="gene_roles.csv from spectral_clustering_gene_roles.py, for ARI comparison.")
    ap.add_argument("--single-layer-gene-col", default="Gene")
    ap.add_argument("--single-layer-cluster-col", default="Cluster")
    ap.add_argument("--aggregate", choices=["majority", "first", "restrict-celltype"],
                     default="majority")
    ap.add_argument("--restrict-celltype", default="global_ppi")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_normalize = not args.no_row_normalize

    df = pd.read_csv(args.clusters_csv)
    U = np.load(args.embedding)
    if len(df) != U.shape[0]:
        raise SystemExit(f"{args.clusters_csv} has {len(df)} rows but {args.embedding} has "
                          f"{U.shape[0]} rows -- must come from the same step8 run.")

    # --- Step 1: existing (silhouette-picked) cluster sizes ---
    report_cluster_sizes(df, "cluster", f"{args.label} (existing k_opt)")

    # --- Step 2a: force-cluster at the requested k on the cached embedding ---
    print(f"\nForcing k={args.force_k} on cached embedding (row_normalize={row_normalize}) ...")
    labels, sil = force_cluster(U, args.force_k, seed=args.seed, row_normalize=row_normalize)
    force_col = f"cluster_k{args.force_k}"
    df[force_col] = labels
    print(f"  silhouette at k={args.force_k}: {sil:.4f}")
    report_cluster_sizes(df, force_col, f"{args.label} (forced k={args.force_k})")

    out_csv = out_dir / f"spectral_clusters_k{args.force_k}_{args.label}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  saved -> {out_csv}")

    # --- Step 2b: ARI against the single-layer result ---
    if args.single_layer_csv:
        sl = pd.read_csv(args.single_layer_csv)
        sl = sl[[args.single_layer_gene_col, args.single_layer_cluster_col]].rename(
            columns={args.single_layer_gene_col: "gene", args.single_layer_cluster_col: "sl_cluster"})
        sl = sl.drop_duplicates(subset="gene")

        restrict = args.restrict_celltype if args.aggregate == "restrict-celltype" else None
        mux_gene_labels = aggregate_multiplex_to_gene(
            df, force_col, method=args.aggregate, restrict_celltype=restrict
        ).rename("mux_cluster").reset_index()

        merged = mux_gene_labels.merge(sl, on="gene", how="inner")
        print(f"\n=== ARI vs. single-layer labels: {args.label} ===")
        print(f"  aggregation method: {args.aggregate}"
              + (f" (restrict_celltype={restrict})" if restrict else ""))
        print(f"  genes compared: {len(merged):,} "
              f"(of {mux_gene_labels['gene'].nunique():,} multiplex genes, "
              f"{sl['gene'].nunique():,} single-layer genes)")
        if len(merged) < 10:
            print("  WARNING: very few overlapping genes -- ARI below is not meaningful.")

        ari = adjusted_rand_score(merged["mux_cluster"], merged["sl_cluster"])
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"    1.0 = identical partitions, 0.0 = no relationship, can go slightly negative")

        ct = pd.crosstab(merged["mux_cluster"], merged["sl_cluster"])
        print(f"\n  contingency table (rows=multiplex forced-k cluster, cols=single-layer cluster):")
        print(ct.to_string())

        summary = {
            "label": args.label,
            "force_k": args.force_k,
            "aggregate_method": args.aggregate,
            "restrict_celltype": restrict,
            "n_genes_compared": int(len(merged)),
            "adjusted_rand_index": float(ari),
        }
        summary_path = out_dir / f"ari_vs_single_layer_{args.label}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        ct.to_csv(out_dir / f"contingency_vs_single_layer_{args.label}.csv")
        print(f"  saved summary -> {summary_path}")


if __name__ == "__main__":
    main()