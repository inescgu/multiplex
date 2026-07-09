#!/usr/bin/env python3
"""
multiplex_core_periphery.py  (v2)
=================================
MPNSM core-periphery detection (Bergermann, Stoll, Tudisco 2024) for the
integrated TS+HRDG multiplex, with two extensions:

  (A) Inter-layer-aware virtual layers
      ------------------------------------------------
      Pathway and ligand-receptor edges, originally inter-layer in the
      integrated multiplex, are aggregated to the GENE level and added as
      additional node-aligned "virtual" layers:
          - one virtual layer per pathway database (reactome, kegg, ...)
          - one virtual layer for ligand-receptor edges
      Edge weights = number of (source, cell_type) layer-pairs that share
      that gene-gene relation (capped at 1 by default for unweighted runs).

  (B) Seed-aware Mann-Whitney hook
      ------------------------------------------------
      After computing per-gene MPNSM coreness x, run a one-sided
      Mann-Whitney U test comparing seed-gene coreness vs non-seed
      coreness. Mirrors test_global_core_periphery in
      core_periph_testing.py but uses the multiplex coreness instead of
      k-core number.
"""

import os
import gc
import json
import argparse
import pickle
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu


# ============================================================================
# 1. ALIGNED-MULTIPLEX TENSOR BUILDER  (with virtual layers)
# ============================================================================
def build_aligned_tensor(G,
                         intra_edge_types=("ppi",),
                         virtual_pathway=True,
                         virtual_lr=True,
                         weighted_virtual=False,
                         verbose=True):
    """
    Convert the integer-keyed multiplex graph G into a node-aligned multiplex
    over UNIQUE GENES.

    Layers:
      - Intra-layer: one (source, cell_type) layer per edge_type in
        `intra_edge_types`. PPI by default.
      - Virtual: pathway edges aggregated to gene-level, split by pathway DB
        (reactome, kegg, ...). One virtual layer per DB.
      - Virtual: ligand-receptor edges aggregated to gene-level. One layer.

    Virtual layer weights = number of (source, cell_type) pairs in which the
    gene-gene relation occurs. If `weighted_virtual=False`, binarised.

    Returns
    -------
    A_layers   : list[csr_matrix]  (n, n) symmetric
    layer_keys : list[tuple]       ("intra"|"virtual", source, cell_type|db)
    genes      : list[str]
    """
    if verbose:
        print(f"\n[align] intra_edge_types={intra_edge_types}  "
              f"virtual_pathway={virtual_pathway}  virtual_lr={virtual_lr}  "
              f"weighted_virtual={weighted_virtual}")

    genes = sorted({d["gene"] for _, d in G.nodes(data=True)})
    gene_index = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    # ----- Intra-layer edges -----
    intra_edges = defaultdict(set)       # (src, ct) -> {(i,j) i<j}
    # ----- Virtual layer edges -----
    virt_path_edges = defaultdict(lambda: defaultdict(int))  # db -> (i,j) -> count
    virt_lr_edges   = defaultdict(int)                       # (i,j) -> count

    for u, v, d in G.edges(data=True):
        et = d.get("edge_type")
        du, dv = G.nodes[u], G.nodes[v]
        gi, gj = gene_index.get(du["gene"]), gene_index.get(dv["gene"])
        if gi is None or gj is None or gi == gj:
            continue
        a, b = (gi, gj) if gi < gj else (gj, gi)

        if et in intra_edge_types:
            # Only intra-layer geometric edges count for the intra layer
            if du["source"] == dv["source"] and du["cell_type"] == dv["cell_type"]:
                intra_edges[(du["source"], du["cell_type"])].add((a, b))

        elif et == "pathway" and virtual_pathway:
            db = d.get("pathway_db", "unknown")
            virt_path_edges[db][(a, b)] += 1

        elif et == "ligand_receptor" and virtual_lr:
            virt_lr_edges[(a, b)] += 1

    # ----- Materialise sparse matrices -----
    def _edges_to_csr(edge_iter, weight=False):
        edges = list(edge_iter)
        if not edges:
            return csr_matrix((n, n), dtype=np.float64)
        if weight:
            E = np.array([(i, j) for (i, j), _ in edges], dtype=np.int64)
            w = np.array([w_ for _, w_ in edges], dtype=np.float64)
        else:
            E = np.array([ij for ij in edges], dtype=np.int64)
            w = np.ones(len(edges), dtype=np.float64)
        rows = np.concatenate([E[:, 0], E[:, 1]])
        cols = np.concatenate([E[:, 1], E[:, 0]])
        data = np.concatenate([w, w])
        return csr_matrix((data, (rows, cols)), shape=(n, n))

    A_layers, layer_keys = [], []

    # Intra layers
    for key in sorted(intra_edges.keys()):
        A_layers.append(_edges_to_csr(intra_edges[key], weight=False))
        layer_keys.append(("intra", key[0], key[1]))

    # Virtual pathway layers (one per DB)
    if virtual_pathway:
        for db in sorted(virt_path_edges.keys()):
            edges = virt_path_edges[db].items()
            A_layers.append(_edges_to_csr(edges, weight=weighted_virtual))
            layer_keys.append(("virtual", "pathway", db))

    # Virtual LR layer
    if virtual_lr and virt_lr_edges:
        A_layers.append(_edges_to_csr(virt_lr_edges.items(), weight=weighted_virtual))
        layer_keys.append(("virtual", "ligand_receptor", "all"))

    if verbose:
        for k, (kind, a, b) in enumerate(layer_keys):
            print(f"  layer {k:>3} [{kind:<7}] {a:<20} {b:<25} "
                  f"n_edges={A_layers[k].nnz//2:,}")
        print(f"[align] genes={n:,}  layers={len(layer_keys):,}\n")

    return A_layers, layer_keys, genes


# ============================================================================
# 2. MPNSM CORE ITERATION  (unchanged from v1)
# ============================================================================
def _grad_fx(A_layers, x, c, alpha):
    n = x.shape[0]
    Fx = np.zeros(n)
    xa = x ** alpha
    for k, Ak in enumerate(A_layers):
        if Ak.nnz == 0:
            continue
        coo = Ak.tocoo()
        i_idx, j_idx, v = coo.row, coo.col, coo.data
        kernel = (xa[i_idx] + xa[j_idx]) ** (1.0 / alpha - 1.0)
        contrib = c[k] * v * (x[i_idx] ** (alpha - 1.0)) * kernel
        np.add.at(Fx, i_idx, contrib)
    return 2.0 * Fx


def _grad_fc(A_layers, x, alpha):
    L = len(A_layers)
    Fc = np.zeros(L)
    xa = x ** alpha
    for k, Ak in enumerate(A_layers):
        if Ak.nnz == 0:
            continue
        coo = Ak.tocoo()
        i_idx, j_idx, v = coo.row, coo.col, coo.data
        Fc[k] = np.sum(v * (xa[i_idx] + xa[j_idx]) ** (1.0 / alpha))
    return Fc


def _pnorm(v, p):
    return np.sum(v ** p) ** (1.0 / p)


def mpnsm(A_layers, alpha=10.0, p=1.4, q=1.4,
          tol=1e-6, max_iter=200, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    n = A_layers[0].shape[0]
    L = len(A_layers)
    pp = p / (p - 1.0)
    qq = q / (q - 1.0)

    x = rng.uniform(0.5, 1.0, size=n)
    c = rng.uniform(0.5, 1.0, size=L)
    x /= _pnorm(x, pp)
    c /= _pnorm(c, qq)

    history = {"f": [], "dx": [], "dc": []}
    if verbose:
        print(f"\n[MPNSM] α={alpha} p={p} q={q} tol={tol} max_iter={max_iter} "
              f"n={n} L={L}")

    t0 = time.time()
    for it in range(1, max_iter + 1):
        gx = _grad_fx(A_layers, x, c, alpha)
        gx = np.maximum(gx, 1e-300); gx /= _pnorm(gx, pp)
        x_new = gx ** (1.0 / (p - 1.0))

        gc_ = _grad_fc(A_layers, x, alpha)
        gc_ = np.maximum(gc_, 1e-300); gc_ /= _pnorm(gc_, qq)
        c_new = gc_ ** (1.0 / (q - 1.0))

        dx = np.linalg.norm(x_new - x); dc = np.linalg.norm(c_new - c)
        f_val = float(np.sum(c_new * _grad_fc(A_layers, x_new, alpha)) / alpha)
        history["f"].append(f_val); history["dx"].append(dx); history["dc"].append(dc)
        x, c = x_new, c_new
        if verbose and (it <= 5 or it % 10 == 0):
            print(f"  iter {it:>4}  f={f_val:.6e}  |dx|={dx:.2e}  |dc|={dc:.2e}")
        if dx < tol and dc < tol:
            if verbose:
                print(f"[MPNSM] converged at iter {it} ({time.time()-t0:.1f}s)")
            break
    else:
        if verbose:
            print(f"[MPNSM] hit max_iter ({max_iter})")

    x /= _pnorm(x, p)
    c /= _pnorm(c, q)
    return x, c, history


# ============================================================================
# 3. QUBO SWEEP  (unchanged from v1)
# ============================================================================
def qubo_sweep(A_layers, x, c, verbose=True):
    n = A_layers[0].shape[0]
    L = len(A_layers)
    perm = np.argsort(-x)
    norm_c1 = np.sum(np.abs(c))

    N1 = np.array([Ak.nnz for Ak in A_layers], dtype=np.float64)
    N2 = np.array([n * n - nnz for nnz in N1], dtype=np.float64)
    degs = [np.asarray(Ak.sum(axis=1)).ravel() for Ak in A_layers]

    A_perm = [Ak[perm][:, perm].tocsr() for Ak in A_layers]
    deg_perm = [d[perm] for d in degs]
    indptr  = [Ak.indptr  for Ak in A_perm]
    indices = [Ak.indices for Ak in A_perm]

    sum_deg = np.zeros(L); xAx = np.zeros(L)
    Q_curve = np.zeros(n)

    if verbose:
        print(f"\n[QUBO] sweeping {n:,} core sizes over {L} layers ...")
    for s in range(1, n + 1):
        i = s - 1
        for l in range(L):
            sum_deg[l] += deg_perm[l][i]
            neigh = indices[l][indptr[l][i]:indptr[l][i + 1]]
            if neigh.size:
                xAx[l] += 2 * np.count_nonzero(neigh < i)
        q = 0.0
        for l in range(L):
            if N1[l] == 0 or N2[l] == 0:
                continue
            a = 1.0 / N1[l] + 1.0 / N2[l]; b = 1.0 / N2[l]
            q_l = (2 * a * sum_deg[l] - 2 * (n - 1) * b * s
                   - a * xAx[l] + b * (s * s - s))
            q += (c[l] / norm_c1) * q_l
        Q_curve[s - 1] = q

    s_star = int(np.argmax(Q_curve) + 1)
    Q_star = float(Q_curve[s_star - 1])
    if verbose:
        print(f"[QUBO] s* = {s_star}/{n}    Q* = {Q_star:.4e}")
    return s_star, Q_star, Q_curve, perm


# ============================================================================
# 4. SEED-AWARE MANN-WHITNEY HOOK
# ============================================================================
def seed_aware_mannwhitney(gene_df, seed_genes, alternative="less", verbose=True):
    """
    One-sided Mann-Whitney U test on MPNSM coreness:
        H1: seed coreness  <  non-seed coreness     (alternative="less")
        → seeds are peripheral in the multiplex.

    Mirrors test_global_core_periphery in core_periph_testing.py but uses
    multiplex coreness instead of nx.core_number.

    Parameters
    ----------
    gene_df : DataFrame with columns ["gene", "node_coreness_x"]
    seed_genes : iterable of gene symbols
    alternative : "less" | "greater" | "two-sided"

    Returns
    -------
    dict with seed/nonseed sample stats, Mann-Whitney U, p-value, and a
    rank-biserial effect size r = 1 - 2U/(n1*n2) ∈ [-1, 1]
    (negative => seeds rank lower than non-seeds).
    """
    seeds = {str(g).strip().upper() for g in seed_genes}
    df = gene_df.copy()
    df["gene_u"] = df["gene"].astype(str).str.strip().str.upper()
    df["is_seed"] = df["gene_u"].isin(seeds)

    seed_x    = df.loc[df["is_seed"],  "node_coreness_x"].to_numpy()
    nonseed_x = df.loc[~df["is_seed"], "node_coreness_x"].to_numpy()

    if len(seed_x) == 0 or len(nonseed_x) == 0:
        print("⚠ Cannot run seed-aware Mann-Whitney: empty seed or non-seed set.")
        return None

    U, p = mannwhitneyu(seed_x, nonseed_x, alternative=alternative)
    rb = 1.0 - 2.0 * U / (len(seed_x) * len(nonseed_x))  # rank-biserial effect size

    res = {
        "n_seeds_in_network"     : int(len(seed_x)),
        "n_nonseeds_in_network"  : int(len(nonseed_x)),
        "seed_median_coreness"   : float(np.median(seed_x)),
        "nonseed_median_coreness": float(np.median(nonseed_x)),
        "seed_mean_coreness"     : float(np.mean(seed_x)),
        "nonseed_mean_coreness"  : float(np.mean(nonseed_x)),
        "mw_U"                   : float(U),
        "mw_p"                   : float(p),
        "alternative"            : alternative,
        "rank_biserial_r"        : float(rb),
    }

    if verbose:
        print(f"\n{'='*78}")
        print("SEED-AWARE MANN-WHITNEY (multiplex MPNSM coreness)")
        print(f"{'='*78}")
        print(f"  Seeds in network    : {res['n_seeds_in_network']}")
        print(f"  Non-seeds in network: {res['n_nonseeds_in_network']}")
        print(f"  Median coreness — seeds    : {res['seed_median_coreness']:.4e}")
        print(f"  Median coreness — non-seeds: {res['nonseed_median_coreness']:.4e}")
        print(f"  Mann-Whitney U : {res['mw_U']:.1f}")
        print(f"  p-value ({alternative}): {res['mw_p']:.4g}")
        print(f"  Rank-biserial r: {res['rank_biserial_r']:+.3f}  "
              f"(negative => seeds more peripheral)")
        if res["mw_p"] < 0.05:
            tag = "peripheral" if alternative == "less" else "central"
            print(f"\n  ⚠️  Seeds ARE significantly {tag} in the multiplex "
                  f"(p<0.05).")
        else:
            print(f"\n  ✅ No significant seed/non-seed coreness shift.")

    return res


# ============================================================================
# 5. PIPELINE
# ============================================================================
def run(multiplex_pkl, output_dir,
        intra_edge_types=("ppi",),
        virtual_pathway=True, virtual_lr=True, weighted_virtual=False,
        alpha=10.0, p=1.4, q=1.4, tol=1e-6, max_iter=200, seed=0,
        seed_genes_path=None, mw_alternative="less"):

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"Loading multiplex from {multiplex_pkl} ...")
    with open(multiplex_pkl, "rb") as f:
        G = pickle.load(f)
    print(f"  nodes={G.number_of_nodes():,}  edges={G.number_of_edges():,}")

    A_layers, layer_keys, genes = build_aligned_tensor(
        G,
        intra_edge_types=intra_edge_types,
        virtual_pathway=virtual_pathway,
        virtual_lr=virtual_lr,
        weighted_virtual=weighted_virtual,
    )
    del G; gc.collect()

    x, c, hist = mpnsm(A_layers, alpha=alpha, p=p, q=q,
                       tol=tol, max_iter=max_iter, seed=seed)
    s_star, Q_star, Q_curve, perm = qubo_sweep(A_layers, x, c)

    # ---- Per-gene results ----
    gene_df = pd.DataFrame({
        "gene"           : genes,
        "node_coreness_x": x,
    })
    gene_df["rank"]    = np.argsort(np.argsort(-x)) + 1
    gene_df["is_core"] = gene_df["rank"] <= s_star
    gene_df = gene_df.sort_values("node_coreness_x", ascending=False)
    gene_df.to_csv(out / "mpnsm_gene_coreness.csv", index=False)

    # ---- Per-layer results ----
    layer_df = pd.DataFrame({
        "layer_index"     : np.arange(len(layer_keys)),
        "kind"            : [k[0] for k in layer_keys],
        "source_or_topic" : [k[1] for k in layer_keys],
        "cell_type_or_db" : [k[2] for k in layer_keys],
        "n_edges"         : [Ak.nnz // 2 for Ak in A_layers],
        "layer_coreness_c": c,
    }).sort_values("layer_coreness_c", ascending=False)
    layer_df.to_csv(out / "mpnsm_layer_coreness.csv", index=False)

    pd.DataFrame({"core_size": np.arange(1, len(Q_curve) + 1),
                  "Q"        : Q_curve}).to_csv(
        out / "mpnsm_qubo_curve.csv", index=False)

    summary = {
        "n_genes"         : len(genes),
        "n_layers"        : len(layer_keys),
        "n_intra_layers"  : sum(1 for k in layer_keys if k[0] == "intra"),
        "n_virtual_layers": sum(1 for k in layer_keys if k[0] == "virtual"),
        "alpha"           : alpha, "p": p, "q": q,
        "tol"             : tol, "max_iter": max_iter,
        "iterations_run"  : len(hist["f"]),
        "s_star"          : s_star,
        "Q_star"          : Q_star,
        "core_fraction"   : s_star / len(genes),
        "intra_edge_types": list(intra_edge_types),
        "virtual_pathway" : virtual_pathway,
        "virtual_lr"      : virtual_lr,
        "weighted_virtual": weighted_virtual,
    }

    # ---- Seed-aware Mann-Whitney ----
    if seed_genes_path:
        print(f"\nLoading seed genes from {seed_genes_path} ...")
        seed_df = pd.read_csv(seed_genes_path)
        col = "gene" if "gene" in seed_df.columns else seed_df.columns[0]
        seed_genes = seed_df[col].astype(str).tolist()
        print(f"  {len(seed_genes)} seeds loaded (column='{col}')")

        mw = seed_aware_mannwhitney(gene_df, seed_genes,
                                    alternative=mw_alternative)
        if mw is not None:
            summary["seed_mannwhitney"] = mw
            # Annotate gene_df with is_seed and re-save
            seeds_set = {str(g).strip().upper() for g in seed_genes}
            gene_df["is_seed"] = gene_df["gene"].astype(str).str.strip()\
                                        .str.upper().isin(seeds_set)
            gene_df.to_csv(out / "mpnsm_gene_coreness.csv", index=False)

    with open(out / "mpnsm_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n==========================================================")
    print("MPNSM COMPLETE")
    print("==========================================================")
    for k, v in summary.items():
        if k == "seed_mannwhitney":
            continue
        print(f"  {k:<18}: {v}")
    print(f"\n  Top-10 core genes:")
    print(gene_df.head(10).to_string(index=False))
    print(f"\n  Top-10 core layers:")
    print(layer_df.head(10).to_string(index=False))
    print(f"\nResults saved → {out}")

    return gene_df, layer_df, Q_curve, summary


# ============================================================================
# 6. CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="MPNSM core-periphery (with virtual layers + seed test)")
    ap.add_argument("--multiplex-pkl", required=True)
    ap.add_argument("--output-dir",    default="mpnsm_output")

    ap.add_argument("--intra-edge-types", nargs="+", default=["ppi"],
                    help="Edge types kept as intra (source,cell_type) layers.")
    ap.add_argument("--no-virtual-pathway", action="store_true",
                    help="Disable virtual pathway layers (one per DB).")
    ap.add_argument("--no-virtual-lr", action="store_true",
                    help="Disable virtual ligand-receptor layer.")
    ap.add_argument("--weighted-virtual", action="store_true",
                    help="Weight virtual edges by #(layers) they occur in.")

    ap.add_argument("--alpha",  type=float, default=10.0)
    ap.add_argument("--p",      type=float, default=1.4)
    ap.add_argument("--q",      type=float, default=1.4)
    ap.add_argument("--tol",    type=float, default=1e-6)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--seed",   type=int, default=0)

    ap.add_argument("--seed-genes", default=None,
                    help="CSV with a 'gene' column for seed-aware MW test.")
    ap.add_argument("--mw-alternative", default="less",
                    choices=["less", "greater", "two-sided"])
    args = ap.parse_args()

    run(args.multiplex_pkl, args.output_dir,
        intra_edge_types=tuple(args.intra_edge_types),
        virtual_pathway=not args.no_virtual_pathway,
        virtual_lr=not args.no_virtual_lr,
        weighted_virtual=args.weighted_virtual,
        alpha=args.alpha, p=args.p, q=args.q,
        tol=args.tol, max_iter=args.max_iter, seed=args.seed,
        seed_genes_path=args.seed_genes,
        mw_alternative=args.mw_alternative)


if __name__ == "__main__":
    main()