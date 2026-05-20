"""
multiplex_marchenko_pastur.py
-----------------------------
Marchenko–Pastur analysis adapted to multiplex networks.

  1. Per-layer MP fit on each sparsified adjacency Ã_ℓ.
  2. Aggregated MP on the bulk of S = Σ_ℓ Ã_ℓ²  (Lei & Lin debiased).
     Eigenvalues > λ⁺ are interpreted as informative directions → K_eff.
  3. Supra-Laplacian (DeFord–Pauls diagonal model) spectrum + Fiedler vector.
  4. Multiplex bridge detection: node copies whose Fiedler sign is negative
     in MULTIPLE layers are flagged as cross-layer bottlenecks.
  5. Network perturbation: drop bridge genes and recompute supra-spectrum.
"""

import os, json, pickle, argparse, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix, diags, identity, bmat
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Re-use helpers from the spectral script (kept inline for standalone use).
# ---------------------------------------------------------------------------
def load_multiplex(graph_pkl, registry_csv):
    with open(graph_pkl, "rb") as f:
        G = pickle.load(f)
    reg = pd.read_csv(registry_csv).set_index("node_id")
    nodelist = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodelist)}
    return G, reg, nodelist, idx


def split_layers(G, idx_of, n):
    rows = defaultdict(list); cols = defaultdict(list); data = defaultdict(list)
    for u, v, attr in G.edges(data=True):
        et = attr.get("edge_type", "ppi")
        w  = float(attr.get("weight", 1.0))
        i, j = idx_of[u], idx_of[v]
        rows[et] += [i, j]; cols[et] += [j, i]; data[et] += [w, w]
    layers = {}
    for et in rows:
        A = csr_matrix((data[et], (rows[et], cols[et])), shape=(n, n))
        A.sum_duplicates(); layers[et] = A
    return layers


def sparsify(A, p, rng):
    if p >= 1.0:  return A
    keep = rng.random(A.nnz) < p
    A_s  = A.copy(); A_s.data = A_s.data * keep.astype(A_s.dtype) / p
    A_s.eliminate_zeros(); return A_s


# ---------------------------------------------------------------------------
# 1. MARCHENKO–PASTUR
# ---------------------------------------------------------------------------
def mp_density(x, q, sigma2):
    """Marchenko–Pastur PDF.  q = p/n  (here treat as 1 for square symmetric)."""
    lam_plus  = sigma2 * (1 + np.sqrt(q)) ** 2
    lam_minus = sigma2 * (1 - np.sqrt(q)) ** 2
    val = np.zeros_like(x, dtype=float)
    mask = (x > lam_minus) & (x < lam_plus)
    val[mask] = np.sqrt((lam_plus - x[mask]) * (x[mask] - lam_minus)) \
                / (2 * np.pi * sigma2 * q * x[mask])
    return val, lam_plus, lam_minus


def fit_mp(eigs, label, out_path, n_top=2000):
    """Fit MP to the bulk; report λ⁺, σ², #outliers above λ⁺."""
    eigs = np.asarray(eigs, dtype=float)
    eigs = eigs[np.isfinite(eigs)]
    eigs = np.sort(eigs)

    # Use bulk (middle 80%) to estimate σ²
    lo, hi  = np.quantile(eigs, 0.10), np.quantile(eigs, 0.90)
    bulk    = eigs[(eigs >= lo) & (eigs <= hi)]
    sigma2  = float(np.var(bulk)) if bulk.size else float(np.var(eigs))

    # Aspect ratio: for square symmetric we set q ≈ 1; for SoS use q = 1
    q = 1.0
    lam_plus  = sigma2 * (1 + np.sqrt(q)) ** 2
    lam_minus = sigma2 * (1 - np.sqrt(q)) ** 2

    outliers  = eigs[eigs > lam_plus]
    K_eff     = int(outliers.size)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(eigs, bins=120, density=True, alpha=0.5, label="empirical")
    xs = np.linspace(max(lam_minus, 1e-6), lam_plus, 400)
    pdf, _, _ = mp_density(xs, q, sigma2)
    ax.plot(xs, pdf, "r-", lw=2, label=f"MP (σ²={sigma2:.3g})")
    ax.axvline(lam_plus,  color="k", ls="--", label=f"λ⁺={lam_plus:.3g}")
    ax.set_title(f"MP fit — {label}  |  outliers={K_eff}")
    ax.set_xlabel("eigenvalue"); ax.set_ylabel("density"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / f"mp_{label}.png", dpi=150); plt.close(fig)

    return dict(label=label, sigma2=sigma2, lam_plus=lam_plus,
                lam_minus=lam_minus, n_outliers=K_eff,
                outliers_top10=outliers[-10:].tolist())


# ---------------------------------------------------------------------------
# 2. SOS SPECTRUM (matches spectral script)
# ---------------------------------------------------------------------------
def sos_matrix(layers, p, rng, debias=True):
    n = next(iter(layers.values())).shape[0]
    S = csr_matrix((n, n), dtype=np.float64); bias = np.zeros(n)
    for A in layers.values():
        A_s = sparsify(A.astype(np.float64), p, rng)
        S  += A_s @ A_s
        if debias:
            bias += np.asarray(A_s.multiply(A_s).sum(axis=1)).ravel()
    if debias: S = S - diags(bias)
    return S


# ---------------------------------------------------------------------------
# 3. SUPRA-LAPLACIAN
# ---------------------------------------------------------------------------
def supra_laplacian(layers, omega):
    names = list(layers); k = len(names); n = next(iter(layers.values())).shape[0]
    blocks = []
    for name in names:
        A = layers[name]; d = np.asarray(A.sum(axis=1)).ravel()
        blocks.append(diags(d) - A)
    L_intra = bmat([[blocks[i] if i == j else None for j in range(k)]
                    for i in range(k)], format="csr")
    I_n = identity(n, format="csr")
    A_inter = bmat([[(omega * I_n) if i != j else None for j in range(k)]
                    for i in range(k)], format="csr")
    L_inter = diags(np.asarray(A_inter.sum(axis=1)).ravel()) - A_inter
    return (L_intra + L_inter).tocsr(), names, n


def smallest_k_eigs(M, k):
    """Smallest k eigenvalues (sym, sparse). sigma=0 shift for Lanczos."""
    vals, vecs = eigsh(M, k=k, sigma=0, which="LM")
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


# ---------------------------------------------------------------------------
# 4. MULTIPLEX FIEDLER + BRIDGE DETECTION
# ---------------------------------------------------------------------------
def supra_fiedler(L_sup, layer_names, n):
    vals, vecs = smallest_k_eigs(L_sup, k=3)
    fiedler = vecs[:, 1]                                  # skip trivial
    # Reshape to (k, n) — node copies per layer
    Fkn = fiedler.reshape(len(layer_names), n)
    return vals, fiedler, Fkn


def multiplex_bridges(Fkn, layer_names, nodelist, reg, out_csv, min_neg_layers=2):
    """
    A node is a multiplex bridge if its copy is on the NEGATIVE side of the
    Fiedler vector in at least `min_neg_layers` layers (DeFord–Pauls Exp. 3).
    """
    neg = (Fkn < 0).astype(int)                # k x n
    n_neg_layers = neg.sum(axis=0)             # how many layers vote negative
    mean_fiedler = Fkn.mean(axis=0)
    std_fiedler  = Fkn.std(axis=0)

    rec = reg.loc[nodelist].reset_index(drop=True)
    df  = pd.DataFrame({
        "node_id": nodelist,
        "source":  rec["source"].values,
        "cell_type": rec["cell_type"].values,
        "gene":    rec["gene"].values,
        "fiedler_mean": mean_fiedler,
        "fiedler_std":  std_fiedler,
        "n_neg_layers": n_neg_layers,
    })
    for li, name in enumerate(layer_names):
        df[f"fiedler_{name}"] = Fkn[li]
    df["is_multiplex_bridge"] = df["n_neg_layers"] >= min_neg_layers
    df.sort_values("fiedler_std", ascending=False).to_csv(out_csv, index=False)
    print(f"  multiplex bridges (>= {min_neg_layers} neg layers): "
          f"{df['is_multiplex_bridge'].sum()} / {len(df)}")
    return df


# ---------------------------------------------------------------------------
# 5. PERTURBATION
# ---------------------------------------------------------------------------
def perturb_drop_bridges(layers, bridge_node_idxs):
    """Zero out rows/cols of bridge nodes in every layer."""
    keep = np.ones(next(iter(layers.values())).shape[0], dtype=bool)
    keep[list(bridge_node_idxs)] = False
    new_layers = {name: A[keep][:, keep] for name, A in layers.items()}
    return new_layers, keep


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph",     required=True)
    ap.add_argument("--registry",  required=True)
    ap.add_argument("--out-dir",   required=True)
    ap.add_argument("--sample-p",  type=float, default=0.5)
    ap.add_argument("--omega",     type=float, default=1.0)
    ap.add_argument("--n-eigs",    type=int,   default=400,
                    help="top eigenvalues to compute for MP/SoS bulk")
    ap.add_argument("--min-neg-layers", type=int, default=2)
    ap.add_argument("--seed",      type=int,   default=42)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0  = time.time()

    G, reg, nodelist, idx_of = load_multiplex(args.graph, args.registry)
    n = len(nodelist)
    layers = split_layers(G, idx_of, n)
    print(f"{n:,} nodes, layers: { {k: v.nnz for k, v in layers.items()} }")

    # ---- 1. Per-layer MP fits --------------------------------------------
    print("\n[1] Per-layer Marchenko–Pastur fits")
    per_layer = {}
    for name, A in layers.items():
        A_s = sparsify(A.astype(np.float64), args.sample_p, rng)
        k_  = min(args.n_eigs, n - 2)
        vals, _ = eigsh(A_s, k=k_, which="LM")
        per_layer[name] = fit_mp(vals, f"layer_{name}", out)

    # ---- 2. SoS bulk MP --------------------------------------------------
    print("\n[2] Sum-of-Squared adjacency MP fit")
    S = sos_matrix(layers, args.sample_p, rng, debias=True)
    k_ = min(args.n_eigs, n - 2)
    sos_vals, _ = eigsh(S, k=k_, which="LM")
    mp_sos = fit_mp(sos_vals, "SoS_aggregated", out)

    # ---- 3. Supra-Laplacian spectrum + Fiedler ---------------------------
    print(f"\n[3] Supra-Laplacian spectrum (ω={args.omega})")
    L_sup, layer_names, n_ = supra_laplacian(layers, args.omega)
    sup_vals, fiedler, Fkn = supra_fiedler(L_sup, layer_names, n_)
    print(f"  smallest eigenvalues: {sup_vals}")

    bridges_df = multiplex_bridges(Fkn, layer_names, nodelist, reg,
                                   out / "multiplex_bridges.csv",
                                   min_neg_layers=args.min_neg_layers)

    # spectral gap plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(sos_vals, bins=80, alpha=0.7); ax[0].axvline(mp_sos["lam_plus"], color="r", ls="--")
    ax[0].set_title("SoS spectrum vs. MP λ⁺"); ax[0].set_xlabel("eigenvalue")
    ax[1].plot(sup_vals, "o-"); ax[1].set_title("Supra-Laplacian smallest eigvals")
    ax[1].set_xlabel("idx"); ax[1].set_ylabel("λ")
    fig.tight_layout(); fig.savefig(out / "spectra_overview.png", dpi=150); plt.close(fig)

    # ---- 4. Perturbation: drop multiplex bridges --------------------------
    print("\n[4] Perturbation: dropping multiplex bridges")
    bridge_nodes = bridges_df.loc[bridges_df["is_multiplex_bridge"], "node_id"].tolist()
    bridge_idxs  = [idx_of[n_] for n_ in bridge_nodes]
    if bridge_idxs:
        new_layers, _ = perturb_drop_bridges(layers, bridge_idxs)
        L_new, _, _   = supra_laplacian(new_layers, args.omega)
        new_vals, _   = smallest_k_eigs(L_new, k=min(20, L_new.shape[0] - 2))
        np.save(out / "supra_eigvals_perturbed.npy", new_vals)
        print(f"  removed {len(bridge_idxs)} nodes; new λ_2 = {new_vals[1]:.4g}")
    else:
        new_vals = None

    # ---- 5. Save summary --------------------------------------------------
    np.save(out / "sos_eigvals.npy",   sos_vals)
    np.save(out / "supra_eigvals.npy", sup_vals)
    np.save(out / "fiedler.npy",       fiedler)

    summary = dict(
        n_nodes=n, layers=list(layers.keys()),
        sample_p=args.sample_p, omega=args.omega,
        per_layer_mp=per_layer,
        sos_mp=mp_sos,
        supra_smallest_eigvals=sup_vals.tolist(),
        n_multiplex_bridges=int(bridges_df["is_multiplex_bridge"].sum()),
        perturbed_smallest_eigvals=(new_vals.tolist() if new_vals is not None else None),
        runtime_sec=time.time() - t0,
    )
    with open(out / "marchenko_pastur_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nDone in {summary['runtime_sec']:.1f}s — outputs in {out}")


if __name__ == "__main__":
    main()