"""
step8f_null_models.py
------------------------------------------------------------------------------------
Step 3 of the multiplex-vs-single-layer comparison: null-model validation of
the forced-k multiplex split, echoing the two null models described for the
single-layer analysis:

  PERMUTATION null : degree-preserving rewiring. Each of the module's layer
    adjacency matrices (ppi, inter, ...) is independently rewired with
    networkx's double_edge_swap (which preserves each node's degree within
    that layer), the full embedding + forced-k clustering pipeline is rerun
    from scratch on the rewired network, and the resulting cluster-size
    ratio / spectral gap / silhouette are recorded. Repeated N times to build
    a null distribution; the real module's statistics are then compared
    against it.

  POPULATION null : degree-matched subgraph sampling. Nodes are drawn from
    the FULL heterogeneous reference graph (not just the module), binned by
    degree so the sample's degree distribution matches the module's, forming
    a same-size random "module" -- then the same embedding + clustering
    pipeline runs on that. Repeated N times.

IMPORTANT CAVEATS -- read before trusting the output:

  1. THIS IS A RECONSTRUCTION, NOT A PORT. It mirrors the *description* of
     the single-layer null models (degree-preserving rewiring + population
     sampling, full pipeline rerun per replicate) but was written from
     scratch for the multiplex/SoS setting -- it is not guaranteed to match
     the single-layer script's exact mechanics (e.g. its swap count, its
     exact degree-binning scheme). Treat this as an independently-designed
     analog, not a validated identical procedure.

  2. LAYER-WISE REWIRING, NOT WHOLE-GRAPH REWIRING. Each layer's adjacency
     is rewired independently, preserving each node's per-layer degree
     separately (the natural multiplex generalization). One side effect:
     edges originally tagged "both" (present in ppi AND inter) are not
     guaranteed to land on the same node pair in both layers after
     rewiring. This is a deliberate simplification, not an oversight --
     flagging it so it's a documented modeling choice if this ends up in a
     methods section.

  3. POPULATION DEGREE MATCHING USES BINS, same known limitation your
     single-layer work already flagged for itself -- coarser bins mean
     coarser matching. --n-degree-bins controls this.

  4. COST. Every replicate reruns a real SVD (sos mode) or a supra-Laplacian
     eigendecomposition (supra mode) -- this is NOT free, and is the reason
     N defaults to a small pilot value rather than 1000/200. A single real
     run for this module took ~90-270s depending on mode/config; budget
     accordingly (n_permutation=50 could take over an hour on 1 core).
     Recommended workflow: run with --pilot first (5 reps each) to see
     per-replicate wall time on your node, THEN size --n-permutation /
     --n-population and --n-jobs for a real batch, ideally as its own
     SLURM job with a generous --time.

Requires step8_multiplex_spectral_clustering.py importable from the same
directory (imports its embedding/graph-loading functions directly, so the
null model uses the IDENTICAL embedding code as the real run, not a
reimplementation).

Usage (pilot timing run first):
  python step8f_null_models.py \
      --ref-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
      --mode sos --sample-p 0.4 --K 3 --force-k 2 --degree-normalize \
      --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
      --label with_ppi --pilot

Then a real batch, e.g.:
  python step8f_null_models.py \
      --ref-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
      --mode sos --sample-p 0.4 --K 3 --force-k 2 --degree-normalize \
      --full-ref-graph <path to gene_reference_graph.pkl, for population null> \
      --n-permutation 100 --n-population 100 --n-jobs 4 \
      --out-dir 08_spectral_analysis/module_with_ppi/diagnostics \
      --label with_ppi

For module_no_ppi, add --exclude-ppi-layer to match.
"""

import argparse, io, json, sys, time, contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# Reuse the real pipeline's own code -- not a reimplementation.
from step8_multiplex_spectral_clustering import (
    load_reference_graph, filter_out_celltypes, split_layers,
    sos_embedding, supra_laplacian, supra_embedding,
)


# ---------------------------------------------------------------------------
# Shared: forced-k clustering + statistics, identical logic to step8e
# ---------------------------------------------------------------------------
def force_cluster_stats(U: np.ndarray, sigma: np.ndarray, k: int,
                         seed: int, row_normalize: bool):
    emb_k = U[:, :k]
    if row_normalize:
        emb_k = normalize(emb_k)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb_k)
    sil = silhouette_score(emb_k, km.labels_) if k > 1 else float("nan")
    sizes = np.bincount(km.labels_)
    ratio = float(sizes.max() / sizes.min()) if sizes.min() > 0 else float("inf")
    top_gap = float(sigma[0] - sigma[1]) if len(sigma) > 1 else float("nan")
    return {"silhouette": sil, "size_ratio": ratio, "top_gap": top_gap,
            "cluster_sizes": sizes.tolist()}


def run_embedding_and_cluster(layers, mode, K_emb, sample_p, omega, debias,
                               degree_normalize, reg_tau, force_k, seed, row_normalize):
    if mode == "sos":
        sigma, U, _ = sos_embedding(layers, K=K_emb, p=sample_p, seed=seed, debias=debias,
                                     degree_normalize=degree_normalize, reg_tau=reg_tau)
    else:
        L_sup, layer_names, n = supra_laplacian(layers, omega)
        sigma, U = supra_embedding(L_sup, K=K_emb, seed=seed)
        k = len(layer_names)
        U = U.reshape(k, n, -1).mean(axis=0)
    return force_cluster_stats(U, sigma, force_k, seed, row_normalize)


# ---------------------------------------------------------------------------
# PERMUTATION null: layer-wise degree-preserving rewiring
# ---------------------------------------------------------------------------
def rewire_layer_matrix(A: csr_matrix, seed: int, swap_factor: int = 10):
    n = A.shape[0]
    G = nx.from_scipy_sparse_array(A)
    n_edges = G.number_of_edges()
    if n_edges < 4:
        return A  # too sparse to meaningfully rewire
    nswap = max(n_edges * swap_factor, 1)
    try:
        nx.double_edge_swap(G, nswap=nswap, max_tries=nswap * 20, seed=seed)
    except nx.NetworkXError:
        pass  # partial rewiring only -- acceptable, matches the sparser layers' reality
    return nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr", dtype=np.float64)


def _one_permutation_rep(rep_idx, seed, layers, mode, K_emb, sample_p, omega,
                          debias, degree_normalize, reg_tau, force_k, row_normalize):
    t0 = time.time()
    rewired = {name: rewire_layer_matrix(A, seed=seed + i)
               for i, (name, A) in enumerate(layers.items())}
    # Suppress the very verbose per-call prints from sos_embedding/supra_embedding
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        stats = run_embedding_and_cluster(rewired, mode, K_emb, sample_p, omega, debias,
                                           degree_normalize, reg_tau, force_k, seed, row_normalize)
    stats["rep"] = rep_idx
    stats["seed"] = seed
    stats["elapsed_sec"] = time.time() - t0
    print(f"  [permutation {rep_idx+1}] silhouette={stats['silhouette']:.4f} "
          f"size_ratio={stats['size_ratio']:.2f} top_gap={stats['top_gap']:.4f} "
          f"({stats['elapsed_sec']:.1f}s)", flush=True)
    return stats


# ---------------------------------------------------------------------------
# POPULATION null: degree-matched subgraph sampling from the FULL reference graph
# ---------------------------------------------------------------------------
def sample_population_subgraph(full_G: nx.Graph, target_degrees: np.ndarray,
                               n_bins: int, seed: int):
    rng = np.random.default_rng(seed)
    full_nodes = np.array(list(full_G.nodes()))
    full_degrees = np.array([d for _, d in full_G.degree()])

    bin_edges = np.quantile(target_degrees, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1
    bin_edges[-1] += 1
    target_bin_counts = np.histogram(target_degrees, bins=bin_edges)[0]

    sampled = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (full_degrees >= lo) & (full_degrees < hi)
        pool = full_nodes[mask]
        need = int(target_bin_counts[i])
        if need == 0:
            continue
        if len(pool) == 0:
            chosen = np.array([], dtype=full_nodes.dtype)
        elif len(pool) < need:
            chosen = rng.choice(pool, size=need, replace=True)
        else:
            chosen = rng.choice(pool, size=need, replace=False)
        sampled.extend(chosen.tolist())
        sampled.extend(chosen.tolist())
    return full_G.subgraph(sampled).copy()


def _one_population_rep(rep_idx, seed, full_G, target_degrees, n_bins, refine_inter,
                         mode, K_emb, sample_p, omega, debias, degree_normalize,
                         reg_tau, force_k, row_normalize):
    t0 = time.time()
    sub_G = sample_population_subgraph(full_G, target_degrees, n_bins, seed)
    sub_nodelist = sorted(sub_G.nodes())
    idx_of = {n: i for i, n in enumerate(sub_nodelist)}
    n = len(sub_nodelist)
    if n < 20:
        return {"rep": rep_idx, "seed": seed, "silhouette": float("nan"),
                "size_ratio": float("nan"), "top_gap": float("nan"),
                "cluster_sizes": [], "elapsed_sec": time.time() - t0,
                "warning": f"sampled subgraph too small (n={n})"}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        layers = split_layers(sub_G, idx_of, n, refine_inter)
        stats = run_embedding_and_cluster(layers, mode, min(K_emb, n - 1), sample_p, omega,
                                           debias, degree_normalize, reg_tau,
                                           min(force_k, n - 1), seed, row_normalize)
    stats["rep"] = rep_idx
    stats["seed"] = seed
    stats["elapsed_sec"] = time.time() - t0
    print(f"  [population {rep_idx+1}] n={n} silhouette={stats['silhouette']:.4f} "
          f"size_ratio={stats['size_ratio']:.2f} top_gap={stats['top_gap']:.4f} "
          f"({stats['elapsed_sec']:.1f}s)", flush=True)
    return stats


# ---------------------------------------------------------------------------
# Comparison against the observed (real module) statistics
# ---------------------------------------------------------------------------
def summarize_against_null(observed: dict, null_results: list, stat_name: str, out_dir: Path, label: str, null_kind: str):
    vals = np.array([r[stat_name] for r in null_results if not np.isnan(r.get(stat_name, np.nan))])
    obs = observed[stat_name]
    if len(vals) == 0:
        print(f"  {stat_name}: no valid null replicates to compare against.")
        return None
    # two-sided empirical p-value
    p_ge = (np.sum(vals >= obs) + 1) / (len(vals) + 1)
    p_le = (np.sum(vals <= obs) + 1) / (len(vals) + 1)
    p_two_sided = 2 * min(p_ge, p_le)
    p_two_sided = min(p_two_sided, 1.0)
    print(f"  {stat_name}: observed={obs:.4f}, null mean={vals.mean():.4f} "
          f"(std={vals.std():.4f}, n={len(vals)}), empirical two-sided p={p_two_sided:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=min(30, max(5, len(vals) // 2)), alpha=0.75, label="null")
    ax.axvline(obs, color="red", linestyle="--", label="observed (real module)")
    ax.set_xlabel(stat_name)
    ax.set_ylabel("count")
    ax.set_title(f"{null_kind} null: {stat_name} — {label}")
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / f"null_{null_kind}_{stat_name}_{label}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {"stat": stat_name, "observed": float(obs), "null_mean": float(vals.mean()),
            "null_std": float(vals.std()), "n_valid_reps": int(len(vals)),
            "p_two_sided": float(p_two_sided)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-graph", required=True)
    ap.add_argument("--exclude-celltypes", nargs="+", default=[])
    ap.add_argument("--exclude-ppi-layer", action="store_true")
    ap.add_argument("--full-ref-graph", default=None,
                     help="Full heterogeneous reference graph (e.g. gene_reference_graph.pkl), "
                          "required only for the population null. If omitted, population null is skipped.")
    ap.add_argument("--mode", choices=["sos", "supra"], default="sos")
    ap.add_argument("--K", type=int, default=3,
                     help="Embedding dimension for null reps. Kept small by default (must be "
                          ">= --force-k) since it directly drives per-replicate SVD cost -- raise "
                          "only if you specifically need a wider null embedding.")
    ap.add_argument("--force-k", type=int, default=2)
    ap.add_argument("--sample-p", type=float, default=0.5)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--no-debias", action="store_true")
    ap.add_argument("--degree-normalize", action="store_true")
    ap.add_argument("--reg-tau", type=float, default=None)
    ap.add_argument("--no-row-normalize", action="store_true")
    ap.add_argument("--n-permutation", type=int, default=20)
    ap.add_argument("--n-population", type=int, default=20)
    ap.add_argument("--n-degree-bins", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=1,
                     help="Parallel replicates. Each worker holds its own copy of the "
                          "layer matrices / SVD workspace -- keep modest unless you've "
                          "confirmed memory headroom with --pilot first.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot", action="store_true",
                     help="Overrides n-permutation/n-population to 5 each, for a quick "
                          "per-replicate timing check before committing to a real batch.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    if args.pilot:
        args.n_permutation = min(args.n_permutation, 5)
        args.n_population = min(args.n_population, 5)
        print("PILOT MODE: n_permutation and n_population capped at 5 each for timing purposes.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_normalize = not args.no_row_normalize

    exclude_celltypes = list(args.exclude_celltypes)
    if args.exclude_ppi_layer and "global_ppi" not in {c.lower() for c in exclude_celltypes}:
        exclude_celltypes.append("global_ppi")

    print(f"Loading module graph: {args.ref_graph}")
    G, nodelist, idx_of = load_reference_graph(args.ref_graph)
    if exclude_celltypes:
        G = filter_out_celltypes(G, exclude_celltypes)
        nodelist = sorted(G.nodes())
        idx_of = {n: i for i, n in enumerate(nodelist)}
    n = len(nodelist)
    print(f"  module: {n:,} nodes")

    layers = split_layers(G, idx_of, n, refine_inter=False)
    target_degrees = np.array([d for _, d in G.degree()])

    K_emb = max(args.K, args.force_k)

    # ---- observed (real) statistics ----
    print("\nComputing OBSERVED statistics on the real module (this is the reference point)...")
    t0 = time.time()
    observed = run_embedding_and_cluster(layers, args.mode, K_emb, args.sample_p, args.omega,
                                          not args.no_debias, args.degree_normalize, args.reg_tau,
                                          args.force_k, args.seed, row_normalize)
    print(f"  observed: silhouette={observed['silhouette']:.4f}, "
          f"size_ratio={observed['size_ratio']:.2f}, top_gap={observed['top_gap']:.4f} "
          f"({time.time()-t0:.1f}s)")

    # ---- PERMUTATION null ----
    print(f"\n=== Permutation null (degree-preserving rewiring): {args.n_permutation} reps ===")
    t0 = time.time()
    perm_results = Parallel(n_jobs=args.n_jobs)(
        delayed(_one_permutation_rep)(
            i, args.seed + 1000 + i, layers, args.mode, K_emb, args.sample_p, args.omega,
            not args.no_debias, args.degree_normalize, args.reg_tau, args.force_k, row_normalize
        ) for i in range(args.n_permutation)
    )
    print(f"  permutation null took {time.time()-t0:.1f}s total "
          f"({(time.time()-t0)/max(args.n_permutation,1):.1f}s/rep average)")

    perm_df = pd.DataFrame(perm_results)
    perm_df.to_csv(out_dir / f"null_permutation_results_{args.label}.csv", index=False)

    perm_summary = {}
    print("\nPermutation null vs. observed:")
    for stat in ("silhouette", "size_ratio", "top_gap"):
        res = summarize_against_null(observed, perm_results, stat, out_dir, args.label, "permutation")
        if res:
            perm_summary[stat] = res

    # ---- POPULATION null ----
    pop_summary = {}
    if args.full_ref_graph:
        print(f"\n=== Population null (degree-matched subgraphs of {args.full_ref_graph}): "
              f"{args.n_population} reps ===")
        full_G, _, _ = load_reference_graph(args.full_ref_graph)
        t0 = time.time()
        pop_results = Parallel(n_jobs=args.n_jobs)(
            delayed(_one_population_rep)(
                i, args.seed + 5000 + i, full_G, target_degrees, args.n_degree_bins, False,
                args.mode, K_emb, args.sample_p, args.omega, not args.no_debias,
                args.degree_normalize, args.reg_tau, args.force_k, row_normalize
            ) for i in range(args.n_population)
        )
        print(f"  population null took {time.time()-t0:.1f}s total "
              f"({(time.time()-t0)/max(args.n_population,1):.1f}s/rep average)")

        pop_df = pd.DataFrame(pop_results)
        pop_df.to_csv(out_dir / f"null_population_results_{args.label}.csv", index=False)

        print("\nPopulation null vs. observed:")
        for stat in ("silhouette", "size_ratio", "top_gap"):
            res = summarize_against_null(observed, pop_results, stat, out_dir, args.label, "population")
            if res:
                pop_summary[stat] = res
    else:
        print("\n--full-ref-graph not given -- skipping population null.")

    # ---- save everything ----
    summary = {
        "label": args.label, "mode": args.mode, "force_k": args.force_k,
        "n_permutation": args.n_permutation, "n_population": args.n_population if args.full_ref_graph else 0,
        "observed": observed,
        "permutation_vs_observed": perm_summary,
        "population_vs_observed": pop_summary,
    }
    summary_path = out_dir / f"null_model_summary_{args.label}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full summary -> {summary_path}")


if __name__ == "__main__":
    main()