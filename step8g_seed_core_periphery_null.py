"""
step8g_seed_core_periphery_null.py
------------------------------------------------------------------------------------
Ports the ACTUAL null-model methodology found in structure_function_analysis.py
(written for the persistence/"hdief" communities) onto the multiplex spectral
k=2 (or k=3) partition from step8e, to directly test the single-layer
pipeline's headline finding -- seed genes enriched in the periphery, not the
hub core -- on the multiplex clustering.

CORRECTION vs. step8f_null_models.py: the real "permutation" null shuffles
SEED LABELS on a FIXED, already-computed structure -- not the graph itself.
Two different, complementary null hypotheses:

  step8f (rewiring)    : is the multiplex PARTITION ITSELF non-random,
                          independent of any seed-gene labels? Expensive --
                          a fresh embedding per replicate.
  step8g (this script) : is the seed/non-seed LABEL meaningfully associated
                          with an already-established core/periphery split?
                          Cheap -- structure computed once, only labels
                          reshuffle. No embedding recomputation at all.

CORE/PERIPHERY DEFINITION: the original script derives core/periphery
per-cluster via a composite k-core + degree-centrality + clustering-
coefficient threshold, applied within each of many small persistence
communities. Here we instead have exactly k (2 or 3) large clusters from
forced spectral clustering, so core vs. periphery is defined the way your
single-layer narrative described its own result: whichever cluster has the
HIGHEST average clustering coefficient is the "core"/hub cluster; the rest
are pooled as "periphery". Flagging this as a simplifying, explicit choice
-- not a hidden one.

Two null models, ported as faithfully as possible to the real code:
  PERMUTATION : --n-permutation reps reshuffling which genes are "seed"
    (seed count preserved), recomputing core/periphery enrichment, bridge
    scores, homogeneity, density under each shuffle. Cheap.
  POPULATION  : --n-population reps of plain uniform-random same-size
    subgraph sampling from --full-ref-graph (matches the real code -- NOT
    degree-binned).
  GLOBAL check: direct Mann-Whitney U on k-core number, seed vs. non-seed,
    across the whole module or --full-ref-graph -- no sampling, ported
    directly from test_global_core_periphery.

Usage:
  python step8g_seed_core_periphery_null.py \
      --ref-graph 06_disease_module/disease_module_output/module_graph.pkl \
      --exclude-ppi-layer \  # only if this run's clusters-csv came from module_no_ppi
      --clusters-csv 08_spectral_analysis/module_with_ppi/diagnostics/spectral_clusters_k2_with_ppi.csv \
      --cluster-col cluster_k2 \
      --seed-genes-csv <path to seed genes csv with a 'gene' column> \
      --full-ref-graph <path to gene_reference_graph.pkl, optional> \
      --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
      --label with_ppi
"""

import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import mannwhitneyu

from step8_multiplex_spectral_clustering import load_reference_graph, filter_out_celltypes


def compute_node_metrics(G: nx.Graph):
    print("  computing k-core number and clustering coefficient ...")
    k_core = nx.core_number(G)
    clustering = nx.clustering(G)
    return k_core, clustering


def assign_core_periphery(df: pd.DataFrame, cluster_col: str, clustering: dict):
    cluster_avg_clust = {}
    for cid, grp in df.groupby(cluster_col):
        vals = [clustering.get(n, 0.0) for n in grp["node_id"]]
        cluster_avg_clust[cid] = float(np.mean(vals)) if vals else 0.0
    core_cid = max(cluster_avg_clust, key=cluster_avg_clust.get)
    print(f"  per-cluster avg clustering coefficient: {cluster_avg_clust}")
    print(f"  -> cluster {core_cid} designated CORE/hub (highest avg clustering coefficient); "
          f"remaining cluster(s) pooled as PERIPHERY")
    df = df.copy()
    df["region"] = np.where(df[cluster_col] == core_cid, "core", "periphery")
    return df, core_cid, cluster_avg_clust


def compute_bridge_scores(G: nx.Graph, nodes, cluster_of: dict):
    scores = {}
    for n in nodes:
        if n not in G:
            scores[n] = 0.0
            continue
        neighbors = list(G.neighbors(n))
        if not neighbors:
            scores[n] = 0.0
            continue
        my_cluster = cluster_of.get(n)
        diff = sum(1 for nb in neighbors if cluster_of.get(nb) != my_cluster)
        scores[n] = diff / len(neighbors)
    return scores


def compute_homogeneity(G: nx.Graph, nodes, label_of: dict):
    scores = {}
    for n in nodes:
        if n not in G:
            scores[n] = 0.0
            continue
        neighbors = list(G.neighbors(n))
        if not neighbors:
            scores[n] = 0.0
            continue
        my_label = label_of.get(n, "non-seed")
        same = sum(1 for nb in neighbors if label_of.get(nb, "non-seed") == my_label)
        scores[n] = same / len(neighbors)
    return scores


def observed_statistics(G: nx.Graph, df: pd.DataFrame, cluster_col: str, seed_set: set):
    df = df.copy()
    df["is_seed"] = df["gene"].astype(str).str.strip().str.upper().isin(seed_set)
    cluster_of = dict(zip(df["node_id"], df[cluster_col]))

    core_nodes = df.loc[df["region"] == "core", "node_id"].tolist()
    periph_nodes = df.loc[df["region"] == "periphery", "node_id"].tolist()
    seed_nodes = df.loc[df["is_seed"], "node_id"].tolist()
    nonseed_nodes = df.loc[~df["is_seed"], "node_id"].tolist()

    n_total = len(df)
    seed_fraction = len(seed_nodes) / n_total if n_total else 0.0

    n_core_seeds = df.loc[(df["region"] == "core") & df["is_seed"]].shape[0]
    n_periph_seeds = df.loc[(df["region"] == "periphery") & df["is_seed"]].shape[0]

    core_seed_enrichment = ((n_core_seeds / len(core_nodes)) / seed_fraction
                             if core_nodes and seed_fraction > 0 else float("nan"))
    periph_seed_enrichment = ((n_periph_seeds / len(periph_nodes)) / seed_fraction
                               if periph_nodes and seed_fraction > 0 else float("nan"))

    label_of = {row.node_id: ("seed" if row.is_seed else "non-seed") for row in df.itertuples()}
    bridge_scores = compute_bridge_scores(G, df["node_id"].tolist(), cluster_of)
    homog_scores = compute_homogeneity(G, df["node_id"].tolist(), label_of)

    seed_bridge = [bridge_scores[n] for n in seed_nodes if n in bridge_scores]
    nonseed_bridge = [bridge_scores[n] for n in nonseed_nodes if n in bridge_scores]
    seed_homog = [homog_scores[n] for n in seed_nodes if n in homog_scores]
    nonseed_homog = [homog_scores[n] for n in nonseed_nodes if n in homog_scores]

    seed_sub = G.subgraph(seed_nodes)
    nonseed_sub = G.subgraph(nonseed_nodes)
    seed_density = nx.density(seed_sub) if len(seed_nodes) > 1 else 0.0
    nonseed_density = nx.density(nonseed_sub) if len(nonseed_nodes) > 1 else 0.0
    seed_avg_degree = float(np.mean([G.degree(n) for n in seed_nodes])) if seed_nodes else 0.0
    nonseed_avg_degree = float(np.mean([G.degree(n) for n in nonseed_nodes])) if nonseed_nodes else 0.0

    observed = {
        "n_total": n_total, "n_seeds": len(seed_nodes), "n_nonseeds": len(nonseed_nodes),
        "seed_fraction": seed_fraction,
        "n_core": len(core_nodes), "n_periphery": len(periph_nodes),
        "core_seed_enrichment": core_seed_enrichment,
        "periphery_seed_enrichment": periph_seed_enrichment,
        "seed_bridge_mean": float(np.mean(seed_bridge)) if seed_bridge else 0.0,
        "non_seed_bridge_mean": float(np.mean(nonseed_bridge)) if nonseed_bridge else 0.0,
        "seed_homogeneity_mean": float(np.mean(seed_homog)) if seed_homog else 0.0,
        "non_seed_homogeneity_mean": float(np.mean(nonseed_homog)) if nonseed_homog else 0.0,
        "seed_density": seed_density, "non_seed_density": nonseed_density,
        "seed_avg_degree": seed_avg_degree, "non_seed_avg_degree": nonseed_avg_degree,
    }
    meta = {"core_nodes": core_nodes, "periph_nodes": periph_nodes,
            "cluster_of": cluster_of, "bridge_scores": bridge_scores}
    return observed, meta


def permutation_null(G: nx.Graph, df: pd.DataFrame, meta: dict, n_seeds: int,
                     n_reps: int, seed_base: int = 0):
    all_nodes = df["node_id"].tolist()
    core_nodes = meta["core_nodes"]
    periph_nodes = meta["periph_nodes"]
    bridge_scores = meta["bridge_scores"]  # structure-based, fixed across reps

    n_total = len(all_nodes)
    seed_fraction = n_seeds / n_total if n_total else 0.0

    keys = ["core_seed_enrichment", "periphery_seed_enrichment",
            "seed_bridge_mean", "non_seed_bridge_mean",
            "seed_homogeneity_mean", "non_seed_homogeneity_mean",
            "seed_density", "non_seed_density",
            "seed_avg_degree", "non_seed_avg_degree"]
    null = {k: [] for k in keys}

    for rep in range(n_reps):
        random.seed(seed_base + rep)
        perm_seeds = set(random.sample(all_nodes, n_seeds))
        perm_nonseeds = set(all_nodes) - perm_seeds

        n_core_seeds = sum(1 for n in core_nodes if n in perm_seeds)
        n_periph_seeds = sum(1 for n in periph_nodes if n in perm_seeds)
        null["core_seed_enrichment"].append(
            (n_core_seeds / len(core_nodes)) / seed_fraction if core_nodes and seed_fraction > 0 else 0.0)
        null["periphery_seed_enrichment"].append(
            (n_periph_seeds / len(periph_nodes)) / seed_fraction if periph_nodes and seed_fraction > 0 else 0.0)

        seed_b = [bridge_scores[n] for n in perm_seeds if n in bridge_scores]
        nonseed_b = [bridge_scores[n] for n in perm_nonseeds if n in bridge_scores]
        null["seed_bridge_mean"].append(float(np.mean(seed_b)) if seed_b else 0.0)
        null["non_seed_bridge_mean"].append(float(np.mean(nonseed_b)) if nonseed_b else 0.0)

        label_of = {n: ("seed" if n in perm_seeds else "non-seed") for n in all_nodes}
        homog = compute_homogeneity(G, all_nodes, label_of)
        seed_h = [homog[n] for n in perm_seeds if n in homog]
        nonseed_h = [homog[n] for n in perm_nonseeds if n in homog]
        null["seed_homogeneity_mean"].append(float(np.mean(seed_h)) if seed_h else 0.0)
        null["non_seed_homogeneity_mean"].append(float(np.mean(nonseed_h)) if nonseed_h else 0.0)

        seed_sub = G.subgraph(list(perm_seeds))
        nonseed_sub = G.subgraph(list(perm_nonseeds))
        null["seed_density"].append(nx.density(seed_sub) if len(perm_seeds) > 1 else 0.0)
        null["non_seed_density"].append(nx.density(nonseed_sub) if len(perm_nonseeds) > 1 else 0.0)
        null["seed_avg_degree"].append(
            float(np.mean([G.degree(n) for n in perm_seeds])) if perm_seeds else 0.0)
        null["non_seed_avg_degree"].append(
            float(np.mean([G.degree(n) for n in perm_nonseeds])) if perm_nonseeds else 0.0)

        if (rep + 1) % 200 == 0:
            print(f"    ... {rep+1}/{n_reps} permutation reps done", flush=True)

    return null


def summarize_null(observed: dict, null: dict, out_dir: Path, label: str, kind: str):
    rows = []
    print()
    for metric, null_vals in null.items():
        obs = observed.get(metric)
        if obs is None or (isinstance(obs, float) and np.isnan(obs)):
            continue
        arr = np.array(null_vals, dtype=float)
        if len(arr) == 0:
            continue
        null_mean, null_std = arr.mean(), arr.std()
        z = (obs - null_mean) / null_std if null_std > 0 else 0.0
        p = float(np.mean(np.abs(arr - null_mean) >= abs(obs - null_mean)))
        rows.append({"metric": metric, "observed": obs, "null_mean": null_mean,
                     "null_std": null_std, "z": z, "p": p})
        flag = "  <-- p<0.05" if p < 0.05 else ""
        print(f"  {metric:<28} observed={obs:9.4f}  null_mean={null_mean:9.4f}  "
              f"z={z:8.3f}  p={p:.4f}{flag}")
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_dir / f"{kind}_null_summary_{label}.csv", index=False)
    return res_df


def population_null(full_G: nx.Graph, size: int, n_reps: int, seed_base: int = 0):
    all_nodes = list(full_G.nodes())
    densities, avg_degrees = [], []
    for rep in range(n_reps):
        random.seed(seed_base + rep)
        if size >= len(all_nodes) or size < 2:
            continue
        sampled = random.sample(all_nodes, size)
        sub = full_G.subgraph(sampled)
        if sub.number_of_edges() == 0:
            continue
        densities.append(nx.density(sub))
        avg_degrees.append(float(np.mean([d for _, d in sub.degree()])))
    return {"density": densities, "avg_degree": avg_degrees}


def global_core_periphery_test(k_core: dict, seed_set: set, all_nodes, gene_of: dict, label: str):
    seed_core = np.array([k_core[n] for n in all_nodes if gene_of.get(n) in seed_set])
    nonseed_core = np.array([k_core[n] for n in all_nodes if gene_of.get(n) not in seed_set])
    if len(seed_core) == 0 or len(nonseed_core) == 0:
        print("  WARNING: seeds or non-seeds missing from network -- skipping global test.")
        return None
    stat, p = mannwhitneyu(seed_core, nonseed_core, alternative="less")
    print(f"\n=== Global core-periphery test: {label} ===")
    print(f"  seeds n={len(seed_core)}, median k-core={np.median(seed_core):.2f}")
    print(f"  non-seeds n={len(nonseed_core)}, median k-core={np.median(nonseed_core):.2f}")
    print(f"  Mann-Whitney U={stat:.1f}, p (seeds<non-seeds)={p:.4g}")
    if p < 0.05:
        print("  -> seeds ARE globally peripheral (p<0.05); module-level periphery "
              "enrichment (if found) may partly reflect this global bias, not be "
              "module-specific.")
    else:
        print("  -> seeds are NOT globally peripheral; module-level periphery "
              "enrichment (if found) looks module-specific.")
    return {"mw_stat": float(stat), "mw_p": float(p),
            "seed_median_kcore": float(np.median(seed_core)),
            "nonseed_median_kcore": float(np.median(nonseed_core))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-graph", required=True)
    ap.add_argument("--exclude-celltypes", nargs="+", default=[])
    ap.add_argument("--exclude-ppi-layer", action="store_true")
    ap.add_argument("--clusters-csv", required=True,
                     help="Output of step8e with a forced-k cluster column, e.g. "
                          "spectral_clusters_k2_with_ppi.csv")
    ap.add_argument("--cluster-col", default="cluster_k2")
    ap.add_argument("--seed-genes-csv", required=True, help="CSV with a 'gene' column.")
    ap.add_argument("--full-ref-graph", default=None,
                     help="Full heterogeneous reference graph, for the population null "
                          "and the global test against the whole network rather than just "
                          "the module. If omitted, the global test runs on the module only "
                          "and the population null is skipped.")
    ap.add_argument("--n-permutation", type=int, default=1000)
    ap.add_argument("--n-population", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_celltypes = list(args.exclude_celltypes)
    if args.exclude_ppi_layer and "global_ppi" not in {c.lower() for c in exclude_celltypes}:
        exclude_celltypes.append("global_ppi")

    print(f"Loading module graph: {args.ref_graph}")
    G, nodelist, idx_of = load_reference_graph(args.ref_graph)
    if exclude_celltypes:
        G = filter_out_celltypes(G, exclude_celltypes)

    df = pd.read_csv(args.clusters_csv)
    if args.cluster_col not in df.columns:
        raise SystemExit(f"{args.cluster_col!r} not found in {args.clusters_csv}; "
                          f"available columns: {list(df.columns)}")

    seed_df = pd.read_csv(args.seed_genes_csv)
    gene_col = "gene" if "gene" in seed_df.columns else seed_df.columns[0]
    seed_set = set(seed_df[gene_col].astype(str).str.strip().str.upper())
    print(f"  {len(seed_set):,} seed genes loaded from {args.seed_genes_csv}")

    k_core, clustering = compute_node_metrics(G)
    df, core_cid, cluster_avg_clust = assign_core_periphery(df, args.cluster_col, clustering)

    observed, meta = observed_statistics(G, df, args.cluster_col, seed_set)
    print(f"\n=== Observed statistics: {args.label} ===")
    for k, v in observed.items():
        print(f"  {k}: {v}")

    print(f"\n=== Permutation null (seed-label shuffling, {args.n_permutation} reps): {args.label} ===")
    perm_null = permutation_null(G, df, meta, observed["n_seeds"], args.n_permutation, seed_base=args.seed)
    summarize_null(observed, perm_null, out_dir, args.label, "permutation")

    global_result = None
    if args.full_ref_graph:
        print(f"\nLoading full reference graph for population null / global test: {args.full_ref_graph}")
        full_G, full_nodelist, _ = load_reference_graph(args.full_ref_graph)

        for size, tag, dens_key, deg_key in [
            (observed["n_core"], "core", "seed_density", "seed_avg_degree"),
            (observed["n_periphery"], "periphery", "non_seed_density", "non_seed_avg_degree"),
        ]:
            print(f"\n=== Population null ({tag}, size={size}, {args.n_population} reps): {args.label} ===")
            pop_dist = population_null(full_G, size, args.n_population, seed_base=args.seed + 9000)
            for metric, obs_val in [("density", observed[dens_key]), ("avg_degree", observed[deg_key])]:
                vals = np.array(pop_dist[metric])
                if len(vals) == 0:
                    print(f"  {metric}: no valid population samples.")
                    continue
                pct = float((vals < obs_val).mean() * 100)
                p = 2 * min(pct, 100 - pct) / 100.0
                print(f"  {metric}: observed={obs_val:.4f}, pop_mean={vals.mean():.4f} "
                      f"(n={len(vals)}), percentile={pct:.1f}, two-sided p={p:.4f}")

        full_gene_of = {n: (n.rsplit("::", 1)[-1] if "::" in n else n) for n in full_G.nodes()}
        full_kcore = nx.core_number(full_G)
        global_result = global_core_periphery_test(
            full_kcore, seed_set, list(full_G.nodes()), full_gene_of,
            f"{args.label} (full reference graph)")
    else:
        module_gene_of = {n: (n.rsplit("::", 1)[-1] if "::" in n else n) for n in G.nodes()}
        global_result = global_core_periphery_test(
            k_core, seed_set, list(G.nodes()), module_gene_of, f"{args.label} (module only)")

    summary = {
        "label": args.label, "cluster_col": args.cluster_col, "core_cluster_id": str(core_cid),
        "cluster_avg_clustering_coeff": {str(k): v for k, v in cluster_avg_clust.items()},
        "observed": observed,
        "n_permutation": args.n_permutation,
        "n_population": args.n_population if args.full_ref_graph else 0,
        "global_core_periphery_test": global_result,
    }
    summary_path = out_dir / f"seed_periphery_summary_{args.label}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()