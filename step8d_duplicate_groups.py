"""
step8d_duplicate_groups.py
------------------------------------------------------------------------------------
Follow-up to step8c_embedding_diagnostic.py.

step8c established THAT a large fraction of nodes collapse into near-duplicate
embedding positions (31.0% with_ppi, 51.5% no_ppi) and that this correlates with
low degree. This script pulls out WHO is in each collapsed group, and -- if you
point it at the same --ref-graph step8 used -- reports each group's shared
neighbor(s): the hub gene(s) common to every member of a duplicate group. That's
the direct mechanistic story: nodes that touch only the same single hub (or same
small hub set) get mathematically near-identical rows in a squared/normalized
adjacency embedding, regardless of what gene they actually are.

Usage:
  python step8d_duplicate_groups.py \
      --clusters-csv 08_spectral_analysis/module_with_ppi/spectral_clusters.csv \
      --embedding 08_spectral_analysis/module_with_ppi/embedding.npy \
      --ref-graph 06_disease_module/disease_module_output/module_graph.pkl \
      --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
      --label with_ppi

  python step8d_duplicate_groups.py \
      --clusters-csv 08_spectral_analysis/module_no_ppi/spectral_clusters.csv \
      --embedding 08_spectral_analysis/module_no_ppi/embedding.npy \
      --ref-graph 06_disease_module/disease_module_output/module_graph.pkl \
      --exclude-ppi-layer \
      --out-dir 08_spectral_analysis/module_no_ppi/diagnostics \
      --label no_ppi
"""

import argparse, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


def _celltype_of(node: str) -> str:
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


def filter_out_celltypes(G: nx.Graph, excluded_celltypes):
    if not excluded_celltypes:
        return G
    excluded_set = {c.strip().upper() for c in excluded_celltypes}
    to_remove = [n for n in G.nodes() if _celltype_of(n).upper() in excluded_set]
    if not to_remove:
        return G
    remove_set = set(to_remove)
    kept = [n for n in G.nodes() if n not in remove_set]
    return G.subgraph(kept).copy()


def load_graph(ref_graph_pkl, exclude_celltypes):
    print(f"  loading {ref_graph_pkl} for neighbor lookup ...")
    with open(ref_graph_pkl, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip() for n in G.nodes()})
    if exclude_celltypes:
        G = filter_out_celltypes(G, exclude_celltypes)
    return G


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters-csv", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--ref-graph", default=None,
                     help="Optional. Same --ref-graph used by step8 for this run. If given, "
                          "each duplicate group's shared neighbor(s) are reported.")
    ap.add_argument("--exclude-celltypes", nargs="+", default=[])
    ap.add_argument("--exclude-ppi-layer", action="store_true",
                     help="Must match the step8 run this embedding came from.")
    ap.add_argument("--round-decimals", type=int, default=6)
    ap.add_argument("--min-group-size", type=int, default=2)
    ap.add_argument("--top-n-print", type=int, default=20,
                     help="How many of the largest groups to print to stdout.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clusters_csv).reset_index(drop=True)
    U = np.load(args.embedding)
    if len(df) != U.shape[0]:
        raise SystemExit(f"clusters-csv has {len(df)} rows but embedding has {U.shape[0]} rows "
                          f"-- these must come from the same step8 run.")

    U_round = np.round(U, args.round_decimals)
    _, inverse = np.unique(U_round, axis=0, return_inverse=True)
    df["group_id"] = inverse

    group_sizes = df.groupby("group_id").size()
    df["group_size"] = df["group_id"].map(group_sizes)

    dup_df = df[df["group_size"] >= args.min_group_size].copy()
    n_groups = dup_df["group_id"].nunique()
    big_sizes = group_sizes[group_sizes >= args.min_group_size]

    print(f"=== Duplicate embedding groups: {args.label} ===")
    print(f"  nodes in a group of size >= {args.min_group_size}: {len(dup_df):,} / {len(df):,}")
    print(f"  number of such groups: {n_groups:,}")
    if len(big_sizes):
        print(f"  group size distribution: min={big_sizes.min()}, "
              f"median={big_sizes.median():.0f}, max={big_sizes.max()}")

    G = None
    if args.ref_graph:
        exclude_celltypes = list(args.exclude_celltypes)
        if args.exclude_ppi_layer and "global_ppi" not in {c.lower() for c in exclude_celltypes}:
            exclude_celltypes.append("global_ppi")
        G = load_graph(args.ref_graph, exclude_celltypes)

    def shared_neighbors(node_ids):
        if G is None:
            return None
        sets = [set(G.neighbors(nid)) if nid in G else set() for nid in node_ids]
        if not sets:
            return set()
        shared = sets[0]
        for s in sets[1:]:
            shared &= s
        return shared

    deg_col = "deg_ppi" if "deg_ppi" in df.columns else None

    rows = []
    for gid, grp in dup_df.groupby("group_id"):
        node_ids = grp["node_id"].tolist()
        shared = shared_neighbors(node_ids)
        rows.append({
            "group_id": int(gid),
            "group_size": int(len(grp)),
            "genes": ";".join(grp["gene"].astype(str).tolist()),
            "cell_types": ";".join(sorted(set(grp["cell_type"].astype(str)))),
            "clusters_in_group": ";".join(sorted(set(grp["cluster"].astype(str)))),
            "degrees": ";".join(grp[deg_col].astype(str).tolist()) if deg_col else "",
            "n_shared_neighbors": None if shared is None else len(shared),
            "shared_neighbors": None if shared is None else ";".join(sorted(shared)[:10]),
        })

    groups_df = pd.DataFrame(rows).sort_values("group_size", ascending=False)
    groups_csv = out_dir / f"duplicate_groups_{args.label}.csv"
    groups_df.to_csv(groups_csv, index=False)
    print(f"  saved full duplicate group table -> {groups_csv}")

    if G is not None and len(groups_df):
        clean = int((groups_df["n_shared_neighbors"] >= 1).sum())
        print(f"  groups where ALL members share >=1 common neighbor: {clean:,} / {len(groups_df):,} "
              f"({100*clean/len(groups_df):.1f}%) -- these are the clean 'attach to the same hub(s)' cases")
        print(f"  groups with NO common neighbor across all members: {len(groups_df) - clean:,} "
              f"-- these collapsed for some other structural reason (e.g. shared 2-hop neighborhoods, "
              f"isolated pairs, or the squared-adjacency embedding picking up longer-range equivalence)")

    print(f"\n  top {min(args.top_n_print, len(groups_df))} largest groups:")
    for _, row in groups_df.head(args.top_n_print).iterrows():
        shared_str = row["shared_neighbors"] if row["shared_neighbors"] else "(none / not computed)"
        print(f"    group {row['group_id']}: {row['group_size']} genes "
              f"[{row['genes'][:80]}{'...' if len(row['genes']) > 80 else ''}], "
              f"cell types={row['cell_types']}, shared neighbor(s)={shared_str}")


if __name__ == "__main__":
    main()