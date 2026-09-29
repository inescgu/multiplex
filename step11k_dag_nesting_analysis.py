#!/usr/bin/env python3
"""
step11k_dag_nesting_analysis.py

FIX 4 -- The most direct structural test of your actual hypothesis, using
data you already have and currently aren't using: hidef_finder's own
weave() output (step11c_phase2_weave.py's <prefix>.edges file) is a literal
parent -> child DAG over clusters. Your hypothesis makes a specific,
falsifiable structural prediction: disease-specific, LOW-persistence
clusters should be DESCENDANTS of the broad, HIGH-persistence ones -- not
just correlated with them in a scatter plot (which is all step11e's
existing plots show).

This script:
  1. Parses the .nodes file (cluster_id, size, members, persistence) and
     the .edges file (parent, child[, relation]) into a networkx DiGraph.
  2. Checks MONOTONICITY: for what fraction of parent->child edges is
     persistence(parent) >= persistence(child)? If your hypothesis is
     right, this should be high (clusters should get MORE specific, i.e.
     less persistent, as you go down the hierarchy, not the reverse or
     random).
  3. Computes DEPTH (distance from root) and N_DESCENDANTS per cluster and
     correlates both against persistence -- expect strong negative
     depth-persistence and positive descendants-persistence correlations
     under your hypothesis.
  4. OPTIONAL: if you pass --enrichment-csv (any of your analyzer output
     CSVs with a cluster_id + adjusted_p_value + a label column, e.g.
     top_cell_type / top_phenotype), tags each cluster as "broad" (e.g.
     significant across many distinct labels / many analyses) or
     "specific" (significant for exactly one), then tests whether
     "specific" clusters are disproportionately DESCENDANTS of "broad"
     ones (via a label-permutation test on the DAG, since a parametric
     test isn't appropriate for a DAG's dependency structure).

USAGE
-----
    python step11k_dag_nesting_analysis.py \
        --nodes-file heterogeneous_hidef_faithful_cpm.nodes \
        --edges-file heterogeneous_hidef_faithful_cpm.edges \
        --output-prefix characterized/heterogeneous_hidef_dag \
        --enrichment-csv output/combined26_multiplex/.../combined_gene_results_multiplex.tsv \
        --enrichment-label-col top_cell_type --enrichment-pval-col adjusted_p_value
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_nodes(nodes_file):
    df = pd.read_csv(nodes_file, sep="\t", header=None,
                      names=["cluster_id", "size", "members", "persistence"],
                      dtype={"cluster_id": str})
    df["cluster_id"] = df["cluster_id"].astype(str)
    return df.set_index("cluster_id")


def load_edges(edges_file):
    """hidef_finder's .edges format is tab-separated parent/child pairs,
    sometimes with a third 'relation' column ('default'/'gain'/'loss' in
    some hidef versions). We read defensively: take the first two columns
    as parent/child always; keep a third column if present but don't
    filter on it unless --relation-filter is passed, since the exact label
    set varies by hidef version and silently dropping edges on a guessed
    label would corrupt the DAG."""
    raw = pd.read_csv(edges_file, sep="\t", header=None, dtype=str)
    raw = raw.rename(columns={0: "parent", 1: "child"})
    if raw.shape[1] >= 3:
        raw = raw.rename(columns={2: "relation"})
    return raw


def build_dag(nodes_df, edges_df, relation_filter=None):
    if relation_filter and "relation" in edges_df.columns:
        before = len(edges_df)
        edges_df = edges_df[edges_df["relation"].isin(relation_filter)]
        print(f"  --relation-filter kept {len(edges_df)}/{before} edges")

    G = nx.DiGraph()
    for cid, row in nodes_df.iterrows():
        G.add_node(cid, size=int(row["size"]), persistence=float(row["persistence"]))
    for _, row in edges_df.iterrows():
        p, c = str(row["parent"]), str(row["child"])
        if p in G.nodes and c in G.nodes:
            G.add_edge(p, c)
    return G


def compute_monotonicity(G):
    ok, total = 0, 0
    violations = []
    for p, c in G.edges():
        total += 1
        pp, pc = G.nodes[p]["persistence"], G.nodes[c]["persistence"]
        if pp >= pc:
            ok += 1
        else:
            violations.append((p, c, pp, pc))
    frac = ok / total if total else np.nan
    return frac, violations


def compute_depth_and_descendants(G):
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    depth = {}
    for r in roots:
        for n, d in nx.single_source_shortest_path_length(G, r).items():
            depth[n] = min(d, depth.get(n, d))
    for n in G.nodes:
        depth.setdefault(n, np.nan)
    n_descendants = {n: len(nx.descendants(G, n)) for n in G.nodes}
    return depth, n_descendants, roots


def label_nesting_permutation_test(G, labels, n_perm=2000, seed=0):
    """labels: dict cluster_id -> 'broad' | 'specific' | None (unlabeled
    clusters ignored). Test statistic: mean depth-difference between
    'specific' descendants and their nearest labeled 'broad' ancestor,
    vs. the same statistic computed on randomly permuted labels (holding
    the DAG topology fixed). A more negative/positive observed value than
    the permutation distribution supports genuine nesting rather than
    label placement being independent of hierarchy position."""
    labeled_nodes = [n for n, lab in labels.items() if lab in ("broad", "specific") and n in G.nodes]
    if len(labeled_nodes) < 6:
        return None

    def statistic(lab_map):
        broad_nodes = {n for n in labeled_nodes if lab_map[n] == "broad"}
        specific_nodes = {n for n in labeled_nodes if lab_map[n] == "specific"}
        if not broad_nodes or not specific_nodes:
            return np.nan
        hits = 0
        for s in specific_nodes:
            ancestors = nx.ancestors(G, s)
            if ancestors & broad_nodes:
                hits += 1
        return hits / len(specific_nodes)

    observed = statistic(labels)
    rng = np.random.default_rng(seed)
    lab_values = [labels[n] for n in labeled_nodes]
    null_stats = []
    for _ in range(n_perm):
        shuffled = rng.permutation(lab_values)
        perm_map = dict(zip(labeled_nodes, shuffled))
        full_map = {**labels, **perm_map}
        null_stats.append(statistic(full_map))
    null_stats = np.array([x for x in null_stats if not np.isnan(x)])
    p_value = float(np.mean(null_stats >= observed)) if len(null_stats) else np.nan
    return {
        "observed_frac_specific_with_broad_ancestor": observed,
        "null_mean": float(null_stats.mean()) if len(null_stats) else np.nan,
        "null_std": float(null_stats.std()) if len(null_stats) else np.nan,
        "p_value": p_value,
        "n_broad_labeled": sum(1 for v in labels.values() if v == "broad"),
        "n_specific_labeled": sum(1 for v in labels.values() if v == "specific"),
    }


def derive_broad_specific_labels(enrichment_df, cluster_id_col, label_col, pval_col,
                                  alpha=0.05, broad_min_labels=3):
    """A cluster is 'broad' if it's significantly enriched for >= broad_min_labels
    distinct categories (cell types / phenotypes / diseases) across your
    enrichment analyses, 'specific' if significant for exactly 1, else unlabeled.
    Expects a long-format table: one row per (cluster, category) test."""
    sig = enrichment_df[enrichment_df[pval_col] < alpha]
    n_sig_labels = sig.groupby(cluster_id_col)[label_col].nunique()
    labels = {}
    for cid, n in n_sig_labels.items():
        if n >= broad_min_labels:
            labels[str(cid)] = "broad"
        elif n == 1:
            labels[str(cid)] = "specific"
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes-file", required=True)
    ap.add_argument("--edges-file", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--relation-filter", nargs="+", default=None,
                     help="If your .edges file has a 3rd 'relation' column and you know "
                          "which value(s) denote real containment edges, filter to those.")
    ap.add_argument("--enrichment-csv", default=None,
                     help="Long-format enrichment table: cluster_id, label, p-value columns")
    ap.add_argument("--enrichment-cluster-col", default="cluster_id")
    ap.add_argument("--enrichment-label-col", default="top_cell_type")
    ap.add_argument("--enrichment-pval-col", default="adjusted_p_value")
    ap.add_argument("--broad-min-labels", type=int, default=3)
    args = ap.parse_args()

    print("Loading .nodes and .edges ...")
    nodes_df = load_nodes(args.nodes_file)
    edges_df = load_edges(args.edges_file)
    G = build_dag(nodes_df, edges_df, relation_filter=args.relation_filter)
    print(f"  DAG: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, "
          f"{sum(1 for n in G if G.in_degree(n) == 0)} root(s)")

    lines = ["=" * 70, "FIX 4: DAG NESTING ANALYSIS", "=" * 70, ""]

    # ---- 1. monotonicity ----
    frac_ok, violations = compute_monotonicity(G)
    line = f"Monotonicity: {frac_ok:.1%} of parent->child edges satisfy persistence(parent) >= persistence(child)"
    print(line)
    lines.append(line)
    if violations:
        lines.append(f"  {len(violations)} violating edges (parent, child, p_parent, p_child), first 20:")
        for v in violations[:20]:
            lines.append(f"    {v}")
    if frac_ok < 0.8:
        warn = ("  \u26a0 Monotonicity below 80% -- your persistence hierarchy is NOT cleanly "
                "nested (children are often MORE persistent than parents). This directly "
                "undercuts a 'broad clusters contain specific sub-clusters' narrative; "
                "investigate whether this stems from --neighbors being too small/large, "
                "or from weave()'s own cutoff parameter.")
        print(warn)
        lines.append(warn)

    # ---- 2. depth / descendants vs persistence ----
    depth, n_desc, roots = compute_depth_and_descendants(G)
    persistence = nx.get_node_attributes(G, "persistence")
    common = [n for n in G.nodes if not np.isnan(depth.get(n, np.nan))]
    depth_arr = np.array([depth[n] for n in common])
    persist_arr = np.array([persistence[n] for n in common])
    desc_arr = np.array([n_desc[n] for n in common])

    rho_depth, p_depth = stats.spearmanr(depth_arr, persist_arr)
    rho_desc, p_desc = stats.spearmanr(desc_arr, persist_arr)
    lines.append("")
    lines.append(f"Spearman(depth_from_root, persistence): rho={rho_depth:+.3f}  p={p_depth:.2e}  "
                 f"(expect NEGATIVE under the hypothesis -- deeper = less persistent = more specific)")
    lines.append(f"Spearman(n_descendants, persistence):   rho={rho_desc:+.3f}  p={p_desc:.2e}  "
                 f"(expect POSITIVE -- broader/more persistent clusters should have more descendants)")
    print(lines[-2])
    print(lines[-1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(depth_arr, persist_arr, s=12, alpha=0.5, edgecolors="none")
    axes[0].set_xlabel("depth from root")
    axes[0].set_ylabel("persistence")
    axes[0].set_title(f"rho={rho_depth:+.2f}")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(desc_arr, persist_arr, s=12, alpha=0.5, edgecolors="none")
    axes[1].set_xscale("symlog")
    axes[1].set_xlabel("# descendants")
    axes[1].set_ylabel("persistence")
    axes[1].set_title(f"rho={rho_desc:+.2f}")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{args.output_prefix}_depth_descendants_vs_persistence.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. optional label-nesting permutation test ----
    if args.enrichment_csv:
        print("\nLoading enrichment table for broad/specific labeling ...")
        sep = "\t" if args.enrichment_csv.endswith((".tsv", ".txt")) else ","
        enr = pd.read_csv(args.enrichment_csv, sep=sep)
        labels = derive_broad_specific_labels(
            enr, args.enrichment_cluster_col, args.enrichment_label_col,
            args.enrichment_pval_col, broad_min_labels=args.broad_min_labels)
        print(f"  Labeled {sum(v=='broad' for v in labels.values())} clusters 'broad', "
              f"{sum(v=='specific' for v in labels.values())} 'specific'")
        result = label_nesting_permutation_test(G, labels)
        lines.append("")
        lines.append("-" * 70)
        lines.append("Label-nesting permutation test (are 'specific' clusters descendants "
                     "of 'broad' clusters more often than chance, given this DAG's topology?)")
        lines.append("-" * 70)
        if result is None:
            lines.append("  Not enough labeled clusters (need >=6) to run this test.")
        else:
            for k, v in result.items():
                lines.append(f"  {k}: {v}")
            print(result)

    report_path = f"{args.output_prefix}_report.txt"
    Path(report_path).write_text("\n".join(lines))
    print(f"\nSaved report -> {report_path}")
    print(f"Saved plots -> {args.output_prefix}_depth_descendants_vs_persistence.png")


if __name__ == "__main__":
    main()