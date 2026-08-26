"""
step8c_embedding_diagnostic.py
------------------------------------------------------------------------------------
Diagnostic for step8_multiplex_spectral_clustering.py.

Follow-up to step8b (connected components) and the cell-type ARI check, both of
which came back negative -- i.e. the implausibly high, flat silhouette curve is
NOT explained by graph fragmentation or by clusters just tracking cell type.

This script checks the remaining likely explanation: degeneracy in the embedding
itself, where many low-degree "leaf" nodes (touching only 1-2 hub genes) end up
with near-identical rows in embedding.npy. K-means on a point cloud full of
near-duplicate points produces near-1.0 silhouette almost regardless of k, purely
as a geometric artifact -- unrelated to real community structure.

Reads only files step8 already wrote -- no graph reload, no recomputation:
  - spectral_clusters.csv  (for deg_ppi / deg_inter / ... raw degree columns)
  - embedding.npy
  - eigvals_<mode>.npy

NOTE on spectral_clusters.csv: its "total_deg" column is a NORMALIZED
composite bridge-score (each layer's degree divided by that layer's max
degree, then summed) -- NOT a node's literal degree. This script ignores
that column and instead sums the raw "deg_<layer>" columns (e.g. deg_ppi +
deg_inter) to get actual node degree.

Three checks:
  1. Degree distribution -- how many nodes are low-degree "leaves"?
  2. Embedding geometry -- how many rows are exact/near-duplicates after
     rounding, what does the nearest-neighbor distance distribution look
     like, and does degree correlate with how close a node sits to its
     nearest neighbor in the embedding (i.e. do low-degree nodes collapse
     toward each other)?
  3. Eigenvalue spectrum -- how much of the embedding's total "spread"
     (participation ratio / effective rank) is concentrated in just the
     first 1-2 components vs. spread across all K?

Usage (matching both step8 output dirs):
  python step8c_embedding_diagnostic.py \
      --clusters-csv 08_spectral_analysis/module_with_ppi/spectral_clusters.csv \
      --embedding 08_spectral_analysis/module_with_ppi/embedding.npy \
      --eigvals 08_spectral_analysis/module_with_ppi/eigvals_sos.npy \
      --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
      --label with_ppi

  python step8c_embedding_diagnostic.py \
      --clusters-csv 08_spectral_analysis/module_no_ppi/spectral_clusters.csv \
      --embedding 08_spectral_analysis/module_no_ppi/embedding.npy \
      --eigvals 08_spectral_analysis/module_no_ppi/eigvals_sos.npy \
      --out-dir 08_spectral_analysis/module_no_ppi/diagnostics \
      --label no_ppi
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# 1. Degree distribution
# ---------------------------------------------------------------------------
def degree_report(df: pd.DataFrame, out_dir: Path, label: str):
    deg_cols = [c for c in df.columns if c.startswith("deg_")]
    if not deg_cols:
        print("  WARNING: no deg_* columns (e.g. deg_ppi, deg_inter) found in clusters csv, "
              "skipping degree report.")
        return None

    raw_total_deg = df[deg_cols].sum(axis=1)
    n = len(df)

    print(f"\n=== Degree distribution: {label} ===")
    print(f"  nodes: {n:,}  (raw degree = sum of {deg_cols})")
    for thresh in (0, 1, 2, 5):
        n_le = int((raw_total_deg <= thresh).sum())
        print(f"  degree <= {thresh}: {n_le:,} nodes ({100*n_le/n:.1f}%)")
    print(f"  median degree: {raw_total_deg.median():.1f}, "
          f"mean degree: {raw_total_deg.mean():.2f}, max: {raw_total_deg.max():.0f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    n_bins = min(60, int(raw_total_deg.max()) + 1) if raw_total_deg.max() > 0 else 1
    ax.hist(raw_total_deg, bins=n_bins)
    ax.set_yscale("log")
    ax.set_xlabel("raw total degree (" + " + ".join(deg_cols) + ")")
    ax.set_ylabel("count of nodes (log scale)")
    ax.set_title(f"Degree distribution — {label}")
    fig.tight_layout()
    plot_path = out_dir / f"degree_distribution_{label}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  saved degree histogram -> {plot_path}")

    return raw_total_deg


# ---------------------------------------------------------------------------
# 2. Embedding geometry: duplicate rows + nearest-neighbor distances
# ---------------------------------------------------------------------------
def embedding_report(embedding_path: str, out_dir: Path, label: str,
                     round_decimals: int, degree_series=None):
    U = np.load(embedding_path)
    n = U.shape[0]
    print(f"\n=== Embedding geometry: {label} ===")
    print(f"  embedding shape: {U.shape}")

    # Exact/near-duplicate rows after rounding to `round_decimals` places.
    U_round = np.round(U, round_decimals)
    _, inverse, counts = np.unique(U_round, axis=0, return_inverse=True, return_counts=True)
    n_unique = len(counts)
    n_in_shared_position = int((counts[inverse] > 1).sum())
    print(f"  unique rows @ {round_decimals} decimals: {n_unique:,} / {n:,} "
          f"({100*n_unique/n:.1f}% unique)")
    print(f"  rows sharing their rounded position with >=1 other row: "
          f"{n_in_shared_position:,} ({100*n_in_shared_position/n:.1f}%)")

    # Nearest-neighbor distance for every node (k=2 -> nearest OTHER point).
    nn = NearestNeighbors(n_neighbors=2).fit(U)
    dists, _ = nn.kneighbors(U)
    nn_dist = dists[:, 1]
    print(f"  nearest-neighbor distance: min={nn_dist.min():.2e}, "
          f"1st pct={np.percentile(nn_dist, 1):.2e}, median={np.median(nn_dist):.2e}, "
          f"mean={nn_dist.mean():.2e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nn_dist, bins=60)
    ax.set_yscale("log")
    ax.set_xlabel("distance to nearest other node in embedding")
    ax.set_ylabel("count of nodes (log scale)")
    ax.set_title(f"Nearest-neighbor distance distribution — {label}")
    fig.tight_layout()
    plot_path = out_dir / f"nn_distance_distribution_{label}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  saved nearest-neighbor distance histogram -> {plot_path}")

    corr = pval = None
    if degree_series is not None and len(degree_series) == n:
        corr, pval = spearmanr(degree_series, nn_dist)
        print(f"  Spearman correlation(degree, nearest-neighbor distance): "
              f"rho={corr:.4f} (p={pval:.2e})")
        print(f"    positive rho = low-degree nodes tend to sit closer to their nearest "
              f"neighbor (collapsing toward each other) — consistent with the "
              f"'low-degree nodes become near-duplicate points' theory")
    else:
        print("  (no matching degree series supplied, skipping degree/nn-distance correlation)")

    return {
        "n_nodes": int(n),
        "embedding_dim": int(U.shape[1]),
        "round_decimals": round_decimals,
        "n_unique_rows_rounded": int(n_unique),
        "frac_unique_rows_rounded": n_unique / n,
        "n_rows_sharing_position": n_in_shared_position,
        "frac_rows_sharing_position": n_in_shared_position / n,
        "nn_dist_min": float(nn_dist.min()),
        "nn_dist_p1": float(np.percentile(nn_dist, 1)),
        "nn_dist_median": float(np.median(nn_dist)),
        "nn_dist_mean": float(nn_dist.mean()),
        "spearman_degree_vs_nn_dist": None if corr is None else float(corr),
        "spearman_degree_vs_nn_dist_pval": None if pval is None else float(pval),
    }


# ---------------------------------------------------------------------------
# 3. Eigenvalue spectrum concentration
# ---------------------------------------------------------------------------
def eigval_report(eigvals_path: str, out_dir: Path, label: str):
    sigma = np.load(eigvals_path)
    sigma = np.sort(sigma)[::-1]
    k = len(sigma)
    total = sigma.sum()
    frac1 = sigma[0] / total if total else float("nan")
    frac2 = sigma[:2].sum() / total if total else float("nan")
    sq_sum = np.sum(sigma ** 2)
    pr = (total ** 2) / sq_sum if sq_sum else float("nan")

    print(f"\n=== Eigenvalue spectrum: {label} ===")
    print(f"  top-{k} eigenvalues: {np.round(sigma, 3).tolist()}")
    print(f"  fraction of total captured by top-1: {frac1:.3f}")
    print(f"  fraction of total captured by top-2: {frac2:.3f}")
    print(f"  participation ratio (effective rank): {pr:.2f} out of {k} dimensions")
    print(f"    PR much smaller than K = most of the embedding's spread lives in a few "
          f"dimensions, with the rest doing little work -- consistent with most nodes "
          f"collapsing together along those remaining directions")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, k + 1), sigma)
    ax.set_xlabel("component index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(f"Top-{k} spectrum — {label}")
    fig.tight_layout()
    plot_path = out_dir / f"eigval_spectrum_{label}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  saved eigenvalue spectrum plot -> {plot_path}")

    return {
        "k": int(k),
        "top_eigenvalues": sigma.tolist(),
        "frac_top1": float(frac1),
        "frac_top2": float(frac2),
        "participation_ratio": float(pr),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters-csv", required=True,
                     help="spectral_clusters.csv from the step8 run being diagnosed.")
    ap.add_argument("--embedding", required=True,
                     help="embedding.npy from the same step8 run.")
    ap.add_argument("--eigvals", required=True,
                     help="eigvals_<mode>.npy from the same step8 run.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--round-decimals", type=int, default=6,
                     help="decimal places embedding rows are rounded to when counting "
                          "exact/near-duplicate positions (default: 6).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clusters_csv)
    raw_total_deg = degree_report(df, out_dir, args.label)

    emb_summary = embedding_report(args.embedding, out_dir, args.label,
                                    args.round_decimals, degree_series=raw_total_deg)
    eig_summary = eigval_report(args.eigvals, out_dir, args.label)

    summary = {"embedding": emb_summary, "eigenvalues": eig_summary}
    summary_path = out_dir / f"embedding_diagnostic_summary_{args.label}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved combined summary -> {summary_path}")


if __name__ == "__main__":
    main()