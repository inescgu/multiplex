#!/usr/bin/env python3
"""
Step 11e: Characterize the persistent consensus communities produced by the
heterogeneous (multiplex) HiDeF pipeline (steps 11a-11c) -- the equivalent
of "look at what's actually in each cluster" for the single-layer HiDeF
communities, but adapted to the "<cell_type>::GENE" namespaced supra-graph.

Because every node carries an explicit cell-type label, a cluster here can
mean several different things, and this script's job is to tell them apart:
  - a single-cell-type PPI module (members overwhelmingly from one cell
    type, linked mostly by 'intra' edges)
  - a genuine cross-cell-type communication module (members spread across
    several cell types, linked substantially by 'bipartite' LR/Reactome/
    scored-LR/global_ppi-identity edges from step 2, or 'identity'
    coupling edges from step 11a's --add-identity-coupling)
  - a "conserved" module: the SAME gene recurring across multiple cell
    types within one cluster -- i.e. cell types that independently
    recruited the same gene into the same functional neighborhood

INPUT
-----
Reads directly from the .nodes file hidef_finder's own output_all() wrote
in step 11c Phase 2 (community / size / members / persistence -- the real,
unmodified hidef_finder persistence values). This deliberately does NOT
depend on the hand-rolled extract_communities() path that step 11d's
changelog flagged as buggy (trivial single-node communities) -- if you
only have that broken pickle, re-run step 11c Phase 2 first, or point
--communities-pkl at the RECOVERED pickle instead (see --help).

members in the .nodes file may be written as either "cell_type::gene"
names directly, or as integer indices into the node_names list saved
alongside the supra-graph (step 11a) / ensemble (step 11b) -- this script
detects which and resolves indices against the supra-graph's own vertex
order, since that's the canonical index used throughout the pipeline.

OUTPUT
------
  <prefix>_cluster_summary.csv          one row per cluster: size,
                                         persistence, cell-type diversity,
                                         gene/conserved-gene counts, edge
                                         composition, (optional) seed-gene
                                         overlap
  <prefix>_cluster_details.json         per-cluster full breakdown:
                                         cell-type -> member genes,
                                         conserved genes -> which cell
                                         types share them, bipartite
                                         cell-type-pair edge counts,
                                         seed genes present
  <prefix>_celltype_participation.csv   cluster x cell_type matrix of
                                         member-gene counts (feeds
                                         step 11f's heatmap directly)
  <prefix>_summary_plots.png            one multi-panel figure: persistence
                                         and size distributions, cell-type
                                         diversity, gene conservation,
                                         cross-cell-type edge coupling, and
                                         which cell types dominate most
                                         often (population-level view --
                                         per-cluster network zooms live in
                                         step 11f)

USAGE
-----
    python step11e_characterize_clusters.py \
        --nodes-file 11_hidef_results_faithful/heterogeneous_hidef_faithful.nodes \
        --supra-graph 11_hidef_hetero/supra_graph.pkl \
        --seed-genes ../seed_genes.csv \
        --output-prefix 11_hidef_results_faithful/characterized/heterogeneous_hidef
"""
import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# LOADING
# ============================================================================

def load_supra_graph(path: Path) -> ig.Graph:
    with open(path, "rb") as f:
        g = pickle.load(f)
    print(f"Loaded supra-graph: {g.vcount():,} nodes, {g.ecount():,} edges "
          f"across {len(set(g.vs['cell_type']))} cell type(s)\n")
    return g


def parse_nodes_file(nodes_path: Path, node_names: list):
    """Parse hidef_finder's own .nodes output: community, size, members,
    persistence (tab-separated, no header). 'members' may be whitespace- or
    comma-separated, and may contain either "cell_type::gene" names or
    integer vertex indices -- both are handled. Rows with an empty/NaN
    members field (e.g. a degenerate 0-member cluster) are skipped rather
    than crashing downstream."""
    df = pd.read_csv(nodes_path, sep="\t", header=None,
                      names=["cluster_id", "size", "members", "persistence"],
                      dtype={"members": str})
    clusters = []
    n_skipped_empty = 0
    n_fallback_to_literal = 0
    for _, row in df.iterrows():
        raw_field = row["members"]
        if pd.isna(raw_field) or not str(raw_field).strip():
            n_skipped_empty += 1
            continue

        raw = [tok for tok in re.split(r"[,\s]+", str(raw_field).strip()) if tok]
        if not raw:
            n_skipped_empty += 1
            continue

        # Sniff format from up to the first 5 tokens rather than just the
        # first one -- more robust if the very first token happens to be
        # an edge case (e.g. a negative index, or a name that starts with
        # a digit-like cell type label).
        looks_like_indices = all(tok.lstrip("-").isdigit() for tok in raw[:5])
        if looks_like_indices:
            try:
                members = [node_names[int(tok)] for tok in raw]
            except (IndexError, ValueError):
                # Index out of range for node_names -- fall back to treating
                # tokens as literal names instead of crashing the whole run.
                members = raw
                n_fallback_to_literal += 1
        else:
            members = raw

        clusters.append({
            "cluster_id": str(row["cluster_id"]),
            "size": int(row["size"]),
            "persistence": float(row["persistence"]),
            "members": members,
        })

    print(f"Parsed {len(clusters):,} clusters from {nodes_path}")
    if n_skipped_empty:
        print(f"  ⚠ skipped {n_skipped_empty} row(s) with empty/NaN members field")
    if n_fallback_to_literal:
        print(f"  ⚠ {n_fallback_to_literal} row(s) looked like indices but resolution "
              f"failed against node_names -- kept as literal tokens instead")
    print()
    return clusters


def load_clusters_from_communities_pkl(path: Path):
    """Fallback path: a {comm_id: {'members': [...], 'persistence': ...}}
    pickle (e.g. the RECOVERED heterogeneous_hidef_communities_FIXED.pkl
    from step 11d's changelog), rather than the .nodes file."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    clusters = []
    for cid, c in raw.items():
        members = c["members"]
        clusters.append({
            "cluster_id": str(cid),
            "size": c.get("size", len(members)),
            "persistence": float(c.get("persistence", 0.0)),
            "members": list(members),
        })
    print(f"Loaded {len(clusters):,} clusters from communities pickle {path}\n")
    return clusters


def load_seed_genes(path):
    if not path:
        return None
    seeds = set(pd.read_csv(path)["gene"].tolist())
    print(f"Loaded {len(seeds):,} seed genes\n")
    return seeds


# ============================================================================
# PER-CLUSTER CHARACTERIZATION
# ============================================================================

def characterize_cluster(cluster, node_index, g, seed_genes=None):
    raw_members = cluster["members"]
    # Some rows can carry tokens that never resolved into "cell_type::gene"
    # form (empty tokens, stray indices out of range, malformed entries in
    # the source .nodes file). Drop those here rather than crashing --
    # they're counted in the summary row so it stays visible if this is
    # happening a lot (which would point at a real upstream parsing issue).
    members = [m for m in raw_members if isinstance(m, str) and "::" in m]
    n_malformed = len(raw_members) - len(members)

    idxs = [node_index[m] for m in members if m in node_index]
    n_missing = len(members) - len(idxs)

    cell_types = [m.rsplit("::", 1)[0] for m in members]
    genes = [m.rsplit("::", 1)[1] for m in members]

    # ---- cell-type composition ----
    ct_counts = Counter(cell_types)
    n_ct = len(ct_counts)
    dominant_ct, dominant_n = ct_counts.most_common(1)[0]
    dominant_frac = dominant_n / len(members)
    probs = np.array(list(ct_counts.values())) / len(members)
    shannon_h = float(-(probs * np.log(probs)).sum())
    evenness = shannon_h / math.log(n_ct) if n_ct > 1 else 0.0

    # ---- gene composition / conservation across cell types ----
    gene_to_cts = defaultdict(set)
    for ct, gene in zip(cell_types, genes):
        gene_to_cts[gene].add(ct)
    n_distinct_genes = len(gene_to_cts)
    conserved = {gene: sorted(cts) for gene, cts in gene_to_cts.items() if len(cts) >= 2}
    n_conserved = len(conserved)
    conserved_frac = n_conserved / n_distinct_genes if n_distinct_genes else 0.0

    # ---- induced-subgraph edge composition ----
    sub = g.induced_subgraph(idxs) if idxs else None
    etype_counts = Counter(sub.es["etype"]) if sub is not None and sub.ecount() else Counter()
    n_intra = etype_counts.get("intra", 0)
    n_bip = etype_counts.get("bipartite", 0)
    n_id = etype_counts.get("identity", 0)
    n_edges_total = n_intra + n_bip + n_id
    cross_frac = (n_bip + n_id) / n_edges_total if n_edges_total else None

    bip_pairs = Counter()
    if sub is not None:
        for e in sub.es:
            if e["etype"] != "bipartite":
                continue
            u, v = e.tuple
            pair = tuple(sorted((sub.vs[u]["cell_type"], sub.vs[v]["cell_type"])))
            bip_pairs[pair] += 1

    # ---- seed gene overlap (optional) ----
    seed_present = sorted(set(genes) & seed_genes) if seed_genes else []

    summary_row = {
        "cluster_id": cluster["cluster_id"],
        "size": cluster["size"],
        "n_members_resolved": len(idxs),
        "n_members_malformed": n_malformed,
        "n_members_missing_from_graph": n_missing,
        "persistence": cluster["persistence"],
        "n_cell_types": n_ct,
        "dominant_cell_type": dominant_ct,
        "dominant_cell_type_frac": round(dominant_frac, 4),
        "celltype_shannon_entropy": round(shannon_h, 4),
        "celltype_evenness": round(evenness, 4),
        "n_distinct_genes": n_distinct_genes,
        "n_conserved_genes": n_conserved,
        "conserved_gene_frac": round(conserved_frac, 4),
        "n_edges_intra": n_intra,
        "n_edges_bipartite": n_bip,
        "n_edges_identity": n_id,
        "cross_edge_fraction": round(cross_frac, 4) if cross_frac is not None else None,
    }
    if seed_genes is not None:
        summary_row["n_seed_genes_present"] = len(seed_present)
        summary_row["seed_gene_frac_of_cluster"] = (
            round(len(seed_present) / n_distinct_genes, 4) if n_distinct_genes else 0.0
        )

    detail = {
        "cluster_id": cluster["cluster_id"],
        "size": cluster["size"],
        "persistence": cluster["persistence"],
        "cell_type_counts": dict(ct_counts),
        "genes_by_celltype": {
            ct: sorted({g for c, g in zip(cell_types, genes) if c == ct})
            for ct in ct_counts
        },
        "conserved_genes": conserved,
        "bipartite_celltype_pairs": {f"{a}|{b}": n for (a, b), n in bip_pairs.items()},
        "seed_genes_present": seed_present,
    }

    return summary_row, detail, ct_counts


# ============================================================================
# SUMMARY PLOTS (population-level -- per-cluster network zooms live in 11f)
# ============================================================================

def plot_cluster_summary_figures(summary_df: pd.DataFrame, out_path: Path):
    """One multi-panel figure summarizing the whole cluster population:
    persistence/size distributions, cell-type diversity, gene conservation,
    cross-cell-type edge coupling, and which cell types tend to dominate.
    This is about characterizing the population as a whole -- for zoomed
    views of individual clusters on the supra-graph, see step 11f."""
    has_seed = "seed_gene_frac_of_cluster" in summary_df.columns
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.3))
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    ax = axes_flat[0]
    pmin, pmax = int(summary_df["persistence"].min()), int(summary_df["persistence"].max())
    # Persistence is integer-valued (a count of surviving resolutions), so bin
    # on integers rather than arbitrary continuous bins -- and log the y-axis,
    # since a real persistence distribution is typically dominated by a spike
    # at max persistence (highly robust "core" modules survive the whole
    # sweep) with a long tail below it; a linear y-axis makes that tail
    # invisible next to the spike.
    bins = np.arange(pmin, pmax + 2) - 0.5
    ax.hist(summary_df["persistence"], bins=bins, color="#4C72B0", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Persistence")
    ax.set_ylabel("# clusters (log scale)")
    ax.set_title("Persistence distribution")
    frac_at_max = float((summary_df["persistence"] == pmax).mean())
    n_at_max = int((summary_df["persistence"] == pmax).sum())
    ax.text(0.02, 0.96, f"{n_at_max:,} clusters ({frac_at_max:.2%}) at persistence={pmax}",
            transform=ax.transAxes, fontsize=8, va="top", ha="left", color="#444444")

    ax = axes_flat[1]
    sizes = summary_df["size"].clip(lower=1)
    bins = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 40)
    ax.hist(sizes, bins=bins, color="#DD8452", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Cluster size (log scale)")
    ax.set_ylabel("# clusters")
    ax.set_title("Cluster size distribution")

    ax = axes_flat[2]
    ax.scatter(summary_df["size"], summary_df["persistence"], s=14, alpha=0.5,
               color="#4C72B0", edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Cluster size (log scale)")
    ax.set_ylabel("Persistence")
    ax.set_title("Persistence vs. size")

    ax = axes_flat[3]
    ct_counts = summary_df["n_cell_types"].value_counts().sort_index()
    ax.bar(ct_counts.index.astype(str), ct_counts.values, color="#55A868")
    ax.set_xlabel("# cell types in cluster")
    ax.set_ylabel("# clusters")
    ax.set_title("Cell-type diversity per cluster")

    ax = axes_flat[4]
    sc = ax.scatter(summary_df["persistence"], summary_df["conserved_gene_frac"],
                     c=summary_df["n_cell_types"], cmap="viridis", s=16, alpha=0.7,
                     edgecolors="none")
    ax.set_xlabel("Persistence")
    ax.set_ylabel("Conserved-gene fraction")
    ax.set_title("Persistence vs. gene conservation")
    fig.colorbar(sc, ax=ax, label="# cell types", shrink=0.8)

    ax = axes_flat[5]
    valid = summary_df.dropna(subset=["cross_edge_fraction"])
    ax.scatter(valid["n_cell_types"], valid["cross_edge_fraction"], s=16, alpha=0.6,
               color="#C44E52", edgecolors="none")
    ax.set_xlabel("# cell types in cluster")
    ax.set_ylabel("Cross-edge fraction\n(bipartite + identity / total)")
    ax.set_title("Cell-type spread vs. cross-cell-type coupling")

    ax = axes_flat[6]
    top_dom = summary_df["dominant_cell_type"].value_counts().head(15)
    ax.barh(top_dom.index[::-1], top_dom.values[::-1], color="#8172B2")
    ax.set_xlabel("# clusters")
    ax.set_title("Most common dominant cell type (top 15)")

    ax = axes_flat[7]
    ax.hist(summary_df["conserved_gene_frac"], bins=30, color="#64B5CD", edgecolor="white")
    ax.set_xlabel("Conserved-gene fraction")
    ax.set_ylabel("# clusters")
    ax.set_title("Gene-conservation distribution")

    ax = axes_flat[8]
    if has_seed:
        ax.hist(summary_df["seed_gene_frac_of_cluster"].dropna(), bins=30,
                color="#937860", edgecolor="white")
        ax.set_xlabel("Seed-gene fraction of cluster")
        ax.set_ylabel("# clusters")
        ax.set_title("Seed-gene enrichment")
    else:
        ax.axis("off")

    fig.suptitle(f"Heterogeneous HiDeF cluster characterization overview "
                 f"(n={len(summary_df):,} clusters)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved summary plots -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--supra-graph", required=True, help="Pickle from step 11a")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--nodes-file", help="hidef_finder .nodes output from step 11c Phase 2")
    src.add_argument("--communities-pkl",
                      help="Fallback: a {comm_id: {'members', 'persistence'}} pickle "
                           "(e.g. the RECOVERED heterogeneous_hidef_communities_FIXED.pkl)")
    ap.add_argument("--seed-genes", default=None, help="Optional CSV with a 'gene' column")
    ap.add_argument("--min-size", type=int, default=2,
                     help="Drop clusters smaller than this before characterizing (default 2)")
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--skip-plots", action="store_true",
                     help="Skip writing <prefix>_summary_plots.png")
    args = ap.parse_args()

    g = load_supra_graph(Path(args.supra_graph))
    node_names = g.vs["name"]
    node_index = {n: i for i, n in enumerate(node_names)}

    if args.nodes_file:
        clusters = parse_nodes_file(Path(args.nodes_file), node_names)
    else:
        clusters = load_clusters_from_communities_pkl(Path(args.communities_pkl))

    n_before = len(clusters)
    clusters = [c for c in clusters if c["size"] >= args.min_size]
    print(f"Size filter (--min-size {args.min_size}): {n_before:,} -> {len(clusters):,} clusters\n")

    seed_genes = load_seed_genes(args.seed_genes)

    summary_rows, details, participation_rows = [], {}, []
    print(f"Characterizing {len(clusters):,} clusters ...")
    for i, cluster in enumerate(clusters, 1):
        summary_row, detail, ct_counts = characterize_cluster(cluster, node_index, g, seed_genes)
        summary_rows.append(summary_row)
        details[cluster["cluster_id"]] = detail
        participation_rows.append({"cluster_id": cluster["cluster_id"], **ct_counts})
        if i % 200 == 0 or i == len(clusters):
            print(f"  [{i}/{len(clusters)}] done")

    summary_df = pd.DataFrame(summary_rows).sort_values("persistence", ascending=False)
    total_malformed = int(summary_df["n_members_malformed"].sum())
    if total_malformed:
        n_affected = int((summary_df["n_members_malformed"] > 0).sum())
        print(f"\n⚠ {total_malformed:,} member token(s) across {n_affected:,} cluster(s) could not "
              f"be resolved to a 'cell_type::gene' name and were dropped -- see "
              f"n_members_malformed in the summary CSV. If this is a large fraction of any "
              f"cluster's size, double check the .nodes file's members-column format/separator.\n")
    participation_df = pd.DataFrame(participation_rows).fillna(0)
    for col in participation_df.columns:
        if col != "cluster_id":
            participation_df[col] = participation_df[col].astype(int)

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_path = Path(f"{out_prefix}_cluster_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    details_path = Path(f"{out_prefix}_cluster_details.json")
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2)

    participation_path = Path(f"{out_prefix}_celltype_participation.csv")
    participation_df.to_csv(participation_path, index=False)

    plots_path = Path(f"{out_prefix}_summary_plots.png")
    if not args.skip_plots:
        plot_cluster_summary_figures(summary_df, plots_path)

    print("\n" + "=" * 60)
    print("Cluster characterization complete")
    print("=" * 60)
    print(f"  {summary_path}")
    print(f"  {details_path}")
    print(f"  {participation_path}")
    if not args.skip_plots:
        print(f"  {plots_path}")
    print()
    if len(summary_df):
        pmax = int(summary_df["persistence"].max())
        frac_at_max = float((summary_df["persistence"] == pmax).mean())
        n_at_max = int((summary_df["persistence"] == pmax).sum())
        print(f"  persistence             {n_at_max:,} clusters ({frac_at_max:.2%}) sit at max "
              f"persistence ({pmax})")
        print(f"  n_cell_types            median={summary_df['n_cell_types'].median():.0f}  "
              f"max={summary_df['n_cell_types'].max():.0f}")
        print(f"  dominant_cell_type_frac median={summary_df['dominant_cell_type_frac'].median():.2f}")
        print(f"  conserved_gene_frac     median={summary_df['conserved_gene_frac'].median():.2f}")
        print(f"  cross_edge_fraction     median={summary_df['cross_edge_fraction'].median():.2f}")
        n_single_ct = int((summary_df["n_cell_types"] == 1).sum())
        print(f"\n  {n_single_ct:,}/{len(summary_df):,} clusters are confined to a single cell type")


if __name__ == "__main__":
    main()