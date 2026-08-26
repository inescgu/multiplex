"""
step8b_component_diagnostic.py
------------------------------------------------------------------------------------
Diagnostic for step8_multiplex_spectral_clustering.py.

Checks whether the spectral clusters in spectral_clusters.csv are mostly just
rediscovering disconnected components of the reference/module graph, rather
than real community structure. This is the classic failure mode behind a
silhouette curve that's implausibly high (>0.85-0.99) and nearly flat across
every k tested: spectral embeddings localize on connected components, so
k-means can carve out near-perfect clusters almost regardless of k if the
graph has many small disconnected (or near-isolated) pieces.

For a given graph + its spectral_clusters.csv, this script:
  1. Finds all connected components and reports their size distribution
     (table + log-scale histogram).
  2. Maps every node in spectral_clusters.csv to its connected component.
  3. Computes the Adjusted Rand Index (ARI) between "connected component
     membership" and "spectral cluster assignment". ARI near 1.0 means the
     clustering is essentially a relabeling of connected components. ARI
     near 0.0 means cluster membership is unrelated to raw connectivity
     (i.e. more likely to reflect genuine substructure within a connected
     graph).
  4. Computes a per-cluster "component purity" (fraction of each spectral
     cluster's members that sit in that cluster's single most common
     connected component) as a second, more interpretable view of the
     same question.

IMPORTANT: --ref-graph and --exclude-celltypes (or --exclude-ppi-layer) here
must match whatever was passed to step8 for the run you're diagnosing, or
node IDs won't line up with the clusters CSV.

Usage (matching the two runs in the job script):
  python step8b_component_diagnostic.py \
      --ref-graph 06_disease_module/disease_module_output/module_graph.pkl \
      --clusters-csv 08_spectral_analysis/module_with_ppi/spectral_clusters.csv \
      --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
      --label with_ppi

  python step8b_component_diagnostic.py \
      --ref-graph 06_disease_module/disease_module_output/module_graph.pkl \
      --exclude-ppi-layer \
      --clusters-csv 08_spectral_analysis/module_no_ppi/spectral_clusters.csv \
      --out-dir 08_spectral_analysis/module_no_ppi/diagnostics \
      --label no_ppi
"""

import argparse, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# Node-name parsing / cell-type exclusion, kept identical to step8 so the
# node set here matches whatever step8 actually clustered.
# ---------------------------------------------------------------------------
def _celltype_of(node: str) -> str:
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


def filter_out_celltypes(G: nx.Graph, excluded_celltypes):
    if not excluded_celltypes:
        return G
    excluded_set = {c.strip().upper() for c in excluded_celltypes}
    to_remove = [n for n in G.nodes() if _celltype_of(n).upper() in excluded_set]
    if not to_remove:
        print(f"  [exclude-celltypes] None of {sorted(excluded_set)} found — nothing removed.")
        return G
    remove_set = set(to_remove)
    kept = [n for n in G.nodes() if n not in remove_set]
    G_filtered = G.subgraph(kept).copy()
    print(f"  [exclude-celltypes] Removed {sorted(excluded_set)}: {len(to_remove):,} nodes dropped "
          f"-> {G_filtered.number_of_nodes():,} nodes remain")
    return G_filtered


def load_graph(ref_graph_pkl, exclude_celltypes):
    print(f"Loading {ref_graph_pkl} ...")
    with open(ref_graph_pkl, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip() for n in G.nodes()})
    if exclude_celltypes:
        G = filter_out_celltypes(G, exclude_celltypes)
    return G


# ---------------------------------------------------------------------------
# Connected component report
# ---------------------------------------------------------------------------
def component_report(G: nx.Graph, out_dir: Path, label: str):
    comps = list(nx.connected_components(G))
    sizes = sorted((len(c) for c in comps), reverse=True)
    n_nodes = G.number_of_nodes()

    node_to_comp = {}
    for i, c in enumerate(comps):
        for n in c:
            node_to_comp[n] = i

    print(f"\n=== Connected components: {label} ===")
    print(f"  nodes: {n_nodes:,}")
    print(f"  connected components: {len(comps):,}")
    print(f"  largest component: {sizes[0]:,} nodes ({100*sizes[0]/n_nodes:.1f}% of graph)")
    print(f"  singleton (isolated, size-1) components: {sum(1 for s in sizes if s == 1):,}")
    print(f"  components with <5 nodes: {sum(1 for s in sizes if s < 5):,}")
    print(f"  components with >=5 nodes: {sum(1 for s in sizes if s >= 5):,}")

    size_df = pd.DataFrame({"component_id": range(len(comps)), "size": sizes})
    size_csv = out_dir / f"component_sizes_{label}.csv"
    size_df.to_csv(size_csv, index=False)
    print(f"  saved component size table -> {size_csv}")

    fig, ax = plt.subplots(figsize=(6, 4))
    n_bins = max(1, min(50, len(set(sizes))))
    ax.hist(sizes, bins=n_bins)
    ax.set_yscale("log")
    ax.set_xlabel("component size (# nodes)")
    ax.set_ylabel("count of components (log scale)")
    ax.set_title(f"Connected component size distribution — {label}")
    fig.tight_layout()
    plot_path = out_dir / f"component_sizes_{label}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  saved component size histogram -> {plot_path}")

    return node_to_comp


# ---------------------------------------------------------------------------
# Compare spectral clusters to connected components
# ---------------------------------------------------------------------------
def compare_to_clusters(node_to_comp: dict, clusters_csv: str, out_dir: Path, label: str):
    df = pd.read_csv(clusters_csv)
    if "node_id" not in df.columns or "cluster" not in df.columns:
        print(f"  WARNING: {clusters_csv} is missing node_id/cluster columns — skipping comparison.")
        return

    df["component_id"] = df["node_id"].map(node_to_comp)
    missing = int(df["component_id"].isna().sum())
    if missing:
        print(f"  WARNING: {missing:,} nodes in {clusters_csv} were not found in the graph loaded "
              f"here — double check --ref-graph / --exclude-celltypes match the step8 run that "
              f"produced this CSV.")
    df = df.dropna(subset=["component_id"]).copy()
    df["component_id"] = df["component_id"].astype(int)

    ari = adjusted_rand_score(df["component_id"], df["cluster"])
    print(f"\n=== Cluster vs. component comparison: {label} ===")
    print(f"  nodes compared: {len(df):,}")
    print(f"  connected components represented: {df['component_id'].nunique():,}")
    print(f"  spectral clusters represented: {df['cluster'].nunique():,}")
    print(f"  Adjusted Rand Index (component vs. spectral cluster): {ari:.4f}")
    print(f"    1.0 = clusters are just relabeled connected components (likely degenerate)")
    print(f"    0.0 = clusters are unrelated to raw connectivity (more likely real structure)")

    purities = []
    for cl, grp in df.groupby("cluster"):
        top = grp["component_id"].value_counts().iloc[0]
        purities.append(top / len(grp))
    mean_purity = float(np.mean(purities))
    print(f"  mean cluster component-purity: {mean_purity:.4f}")
    print(f"    1.0 = every spectral cluster sits entirely inside one connected component")

    comp_cluster_counts = df.groupby("component_id")["cluster"].nunique()
    n_components_split = int((comp_cluster_counts > 1).sum())

    summary = {
        "n_nodes_compared": int(len(df)),
        "n_components": int(df["component_id"].nunique()),
        "n_spectral_clusters": int(df["cluster"].nunique()),
        "adjusted_rand_index": ari,
        "mean_cluster_component_purity": mean_purity,
        "components_split_across_multiple_clusters": n_components_split,
    }
    summary_path = out_dir / f"component_vs_cluster_summary_{label}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved comparison summary -> {summary_path}")

    if ari > 0.9:
        print("\n  >>> ARI is very high: the spectral clustering is likely just rediscovering "
              "disconnected components, not real community structure. Treat k_opt with caution.")
    elif ari < 0.3:
        print("\n  >>> ARI is low: clusters look substantially independent of raw connectivity, "
              "which is more consistent with genuine substructure within connected regions.")
    else:
        print("\n  >>> ARI is in between — components probably explain part but not all of the "
              "cluster structure. Worth a manual look at spectral_clusters.csv 'role' column.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-graph", required=True,
                     help="Same --ref-graph used for the step8 run being diagnosed.")
    ap.add_argument("--exclude-celltypes", nargs="+", default=[], metavar="CELLTYPE",
                     help="Same --exclude-celltypes used for that step8 run, if any.")
    ap.add_argument("--exclude-ppi-layer", action="store_true",
                     help="Shorthand for --exclude-celltypes global_ppi (must match the step8 run).")
    ap.add_argument("--clusters-csv", required=True,
                     help="spectral_clusters.csv produced by that step8 run.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="run",
                     help="Short tag used in output filenames, e.g. 'with_ppi' or 'no_ppi'.")
    args = ap.parse_args()

    exclude_celltypes = list(args.exclude_celltypes)
    if args.exclude_ppi_layer and "global_ppi" not in {c.lower() for c in exclude_celltypes}:
        exclude_celltypes.append("global_ppi")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = load_graph(args.ref_graph, exclude_celltypes)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges after any exclusion")

    node_to_comp = component_report(G, out_dir, args.label)
    compare_to_clusters(node_to_comp, args.clusters_csv, out_dir, args.label)


if __name__ == "__main__":
    main()