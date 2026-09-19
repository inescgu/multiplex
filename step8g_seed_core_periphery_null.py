"""
step8g_seed_core_periphery_null.py
------------------------------------------------------------------------------------
Ports the ACTUAL null-model methodology found in structure_function_analysis.py
(written for the persistence/"hdief" communities) onto the multiplex spectral
k=2/k=3 partition from step8e, to directly test the single-layer pipeline's
headline finding -- seed genes enriched in the periphery, not the hub core --
on the multiplex clustering.

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

CORE/PERIPHERY DEFINITION -- UPDATED (this version):

  Two options, via --core-definition:

  "composite" (DEFAULT, as of this update): a direct port of
    structure_function_analysis.py's compute_core_periphery_structure(),
    the actual per-gene definition used in the single-layer analysis.
    For each spectral cluster's own induced subgraph, every gene is scored
    on three metrics -- k-core number, degree centrality, and local
    clustering coefficient -- each thresholded at that CLUSTER's own
    33rd/67th percentile. A gene is "core" if it's in the top third on
    >=2 of the 3 metrics, "periphery" if it's in the bottom third on >=2
    of the 3, and "intermediate" (dropped from the core/periphery
    comparison, exactly as the single-layer script does) otherwise.
    Core/periphery genes are then POOLED across all spectral clusters
    into one core set and one periphery set for the enrichment stats
    below -- i.e. this treats each spectral cluster the way the
    single-layer script treated each persistence community: a local
    neighborhood within which to ask "who's central, who's peripheral,"
    not a single binary hub/non-hub label slapped on an entire cluster.
    This is a much closer methodological match to the single-layer
    result than the old cluster-level shortcut, though it's still an
    adaptation -- clusters here are large spectral partitions, not many
    small persistence communities, so "within-cluster" percentiles are
    computed over a much bigger, more heterogeneous pool of genes than
    in the original. Flagging that as a real, if smaller, difference
    that remains.

  "cluster-level" (legacy, pre-this-update behavior, kept for
    comparison/reproducibility only): whichever WHOLE spectral cluster
    has the highest average clustering coefficient is entirely labeled
    "core"; every other whole cluster is pooled as "periphery." No
    per-gene distinction within a cluster. This was flagged in the
    original version of this script as an explicit simplification, not
    a hidden one -- it's kept here only so old results can still be
    regenerated if needed.

Two null models, ported as faithfully as possible to the real code:
  PERMUTATION : --n-permutation reps reshuffling which genes are "seed"
    (seed count preserved), recomputing core/periphery enrichment, bridge
    scores, homogeneity, density under each shuffle. Cheap -- core/
    periphery structure is computed ONCE up front and held fixed; only
    the seed labels reshuffle each rep.
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
      --clusters-csv 08_spectral_analysis/module_with_ppi/spectral_clusters.csv \
      --cluster-col cluster \
      --core-definition composite \
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


# ---------------------------------------------------------------------------
# NEW: composite per-gene core/periphery definition, ported directly from
# structure_function_analysis.py's StructureFunctionAnalyzer.compute_core_
# periphery_structure(). Operates on ONE cluster's induced subgraph at a
# time -- call this once per spectral cluster, same role as "once per
# persistence community" in the single-layer script.
# ---------------------------------------------------------------------------
def compute_core_periphery_structure(subgraph: nx.Graph, cluster_genes: list):
    """
    Returns None if the cluster is too small (<3 genes) to define tertiles on,
    matching the single-layer script's own guard. Otherwise returns a dict
    with core_genes / periphery_genes / intermediate_genes (sets) plus the
    raw per-gene metric dicts, in case you want to inspect them later.
    """
    if len(cluster_genes) < 3:
        return None

    try:
        k_cores = nx.core_number(subgraph)
    except Exception:
        return None

    degree_cent = nx.degree_centrality(subgraph)
    clustering = nx.clustering(subgraph)

    k_core_values = np.array([k_cores[g] for g in cluster_genes])
    degree_values = np.array([degree_cent[g] for g in cluster_genes])
    clustering_values = np.array([clustering[g] for g in cluster_genes])

    k_core_hi, k_core_lo = np.percentile(k_core_values, [67, 33])
    degree_hi, degree_lo = np.percentile(degree_values, [67, 33])
    clust_hi, clust_lo = np.percentile(clustering_values, [67, 33])

    core_genes, periphery_genes, intermediate_genes = set(), set(), set()
    for gene in cluster_genes:
        kc, dc, cl = k_cores[gene], degree_cent[gene], clustering[gene]

        core_score = sum([kc >= k_core_hi, dc >= degree_hi, cl >= clust_hi])
        periphery_score = sum([kc <= k_core_lo, dc <= degree_lo, cl <= clust_lo])

        if core_score >= 2:
            core_genes.add(gene)
        elif periphery_score >= 2:
            periphery_genes.add(gene)
        else:
            intermediate_genes.add(gene)

    return {
        "core_genes": core_genes,
        "periphery_genes": periphery_genes,
        "intermediate_genes": intermediate_genes,
        "k_cores": k_cores,
        "degree_centrality": degree_cent,
        "clustering": clustering,
    }


def assign_core_periphery_composite(df: pd.DataFrame, cluster_col: str, G: nx.Graph,
                                     min_cluster_size: int = 3):
    """
    NEW default path. Runs compute_core_periphery_structure() independently
    within each spectral cluster's own induced subgraph, then pools core/
    periphery genes across all clusters into df["region"]. Genes in a
    too-small cluster, or scored "intermediate," end up labeled
    "intermediate" and are excluded from the core-vs-periphery comparison
    downstream (same convention as the single-layer script).
    """
    df = df.copy()
    df["region"] = "intermediate"

    per_cluster_summary = {}
    for cid, grp in df.groupby(cluster_col):
        genes = grp["node_id"].tolist()
        if len(genes) < min_cluster_size:
            print(f"  cluster {cid}: only {len(genes)} genes (<{min_cluster_size}), "
                  f"skipping composite scoring -- all left as 'intermediate'.")
            per_cluster_summary[str(cid)] = None
            continue

        sub = G.subgraph(genes).copy()
        cp = compute_core_periphery_structure(sub, genes)
        if cp is None:
            print(f"  cluster {cid}: core/periphery structure computation failed, "
                  f"skipping -- all left as 'intermediate'.")
            per_cluster_summary[str(cid)] = None
            continue

        df.loc[df["node_id"].isin(cp["core_genes"]), "region"] = "core"
        df.loc[df["node_id"].isin(cp["periphery_genes"]), "region"] = "periphery"
        # genes not in core_genes or periphery_genes keep "intermediate"

        per_cluster_summary[str(cid)] = {
            "n_genes": len(genes),
            "n_core": len(cp["core_genes"]),
            "n_periphery": len(cp["periphery_genes"]),
            "n_intermediate": len(cp["intermediate_genes"]),
        }
        print(f"  cluster {cid}: {len(genes)} genes -> "
              f"core={len(cp['core_genes'])}, periphery={len(cp['periphery_genes'])}, "
              f"intermediate={len(cp['intermediate_genes'])}")

    n_core = int((df["region"] == "core").sum())
    n_periph = int((df["region"] == "periphery").sum())
    n_inter = int((df["region"] == "intermediate").sum())
    print(f"\n  POOLED across all clusters: core={n_core}, periphery={n_periph}, "
          f"intermediate={n_inter} (of {len(df)} total)")
    if n_inter / len(df) > 0.5:
        print("  NOTE: more than half the genes landed in 'intermediate' -- this is "
              "expected behavior (mixed-signal genes are deliberately excluded, same "
              "as the single-layer script), but it does mean the core/periphery "
              "comparison below is running on a minority of the module. Worth "
              "reporting the intermediate fraction alongside any enrichment numbers.")

    return df, per_cluster_summary


# ---------------------------------------------------------------------------
# LEGACY: old whole-cluster shortcut, kept only for reproducing/comparing
# against pre-this-update results. Not used unless --core-definition
# cluster-level is passed explicitly.
# ---------------------------------------------------------------------------
def assign_core_periphery_cluster_level(df: pd.DataFrame, cluster_col: str, clustering: dict):
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
    """
    Unchanged from the original script -- this reads whatever df["region"]
    says, so it works identically regardless of which core/periphery
    assignment function populated it. Note that with the new composite
    definition, "core" + "periphery" no longer necessarily sum to n_total
    (some genes are "intermediate" and excluded), so n_core + n_periphery
    is reported alongside n_total explicitly.
    """
    df = df.copy()
    df["is_seed"] = df["gene"].astype(str).str.strip().str.upper().isin(seed_set)
    cluster_of = dict(zip(df["node_id"], df[cluster_col]))

    core_nodes = df.loc[df["region"] == "core", "node_id"].tolist()
    periph_nodes = df.loc[df["region"] == "periphery", "node_id"].tolist()
    n_intermediate = int((df["region"] == "intermediate").sum())
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
        "n_intermediate": n_intermediate,
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
    """
    Unchanged in logic from the original script. Core/periphery structure
    (meta["core_nodes"], meta["periph_nodes"], meta["bridge_scores"]) is
    computed ONCE before this is called and held fixed across all reps --
    only which genes count as "seed" is reshuffled each rep. This is what
    keeps this null cheap regardless of which core/periphery definition
    was used upstream.
    """
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
                     help="Output of step8 (or step8e's forced-k version) with a cluster column.")
    ap.add_argument("--cluster-col", default="cluster_k2")
    ap.add_argument("--core-definition", choices=["composite", "cluster-level"], default="composite",
                     help="'composite' (default): per-gene k-core+degree-centrality+clustering "
                          "composite, ported from structure_function_analysis.py, scored within "
                          "each spectral cluster and pooled -- the closer match to the single-"
                          "layer script's own definition. 'cluster-level': legacy pre-update "
                          "shortcut (whole highest-avg-clustering-coefficient cluster = core), "
                          "kept only for reproducing old results.")
    ap.add_argument("--min-cluster-size", type=int, default=3,
                     help="Clusters smaller than this are skipped for composite scoring (all "
                          "their genes end up 'intermediate'). Only used with --core-definition "
                          "composite. Matches the single-layer script's own n<3 guard.")
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

    print(f"\nCore/periphery definition: {args.core_definition}")
    core_cid = None
    cluster_avg_clust = None
    per_cluster_summary = None
    if args.core_definition == "composite":
        df, per_cluster_summary = assign_core_periphery_composite(
            df, args.cluster_col, G, min_cluster_size=args.min_cluster_size)
    else:
        print("  computing k-core number and clustering coefficient (whole module) ...")
        _, clustering = nx.core_number(G), nx.clustering(G)
        df, core_cid, cluster_avg_clust = assign_core_periphery_cluster_level(
            df, args.cluster_col, clustering)

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
            k_core=nx.core_number(G), seed_set=seed_set, all_nodes=list(G.nodes()),
            gene_of=module_gene_of, label=f"{args.label} (module only)")

    summary = {
        "label": args.label, "cluster_col": args.cluster_col,
        "core_definition": args.core_definition,
        "core_cluster_id": None if core_cid is None else str(core_cid),
        "cluster_avg_clustering_coeff": (None if cluster_avg_clust is None
                                          else {str(k): v for k, v in cluster_avg_clust.items()}),
        "per_cluster_composite_summary": per_cluster_summary,
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