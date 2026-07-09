"""
multiplex_spectral_clustering.py
--------------------------------
Randomized spectral clustering for the integrated multiplex disease network.

Two modes:
  --mode sos     : Su, Guo, Chang & Yang (2025) — randomized sum-of-squared
                   sparsified adjacencies with Lei & Lin (2023) bias adjustment.
                   Squaring handles assortative/disassortative layer mixing.
  --mode supra   : DeFord & Pauls (2017) diagonal supra-Laplacian
                   with inter-layer coupling weight `--omega`.

Layers are inferred from edge attribute `edge_type` ∈ {ppi, pathway, lr}.
"""

import os, gc, json, pickle, argparse, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix, diags, identity, vstack, hstack, bmat
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. LOAD MULTIPLEX
# ---------------------------------------------------------------------------
def load_multiplex(graph_pkl: str, registry_csv: str):
    print(f"Loading supra-graph from {graph_pkl} ...")
    with open(graph_pkl, "rb") as f:
        G = pickle.load(f)
    reg = pd.read_csv(registry_csv).set_index("node_id")
    nodelist = sorted(G.nodes())
    idx_of = {n: i for i, n in enumerate(nodelist)}
    print(f"  {len(nodelist):,} supra-nodes, {G.number_of_edges():,} edges")
    return G, reg, nodelist, idx_of


def split_layers(G: nx.Graph, idx_of: dict, n: int):
    """
    Build per-edge-type sparse adjacency matrices on the SAME node index.
    Returns dict {layer_name: csr_matrix (n x n, symmetric, weighted)}.
    """
    rows = defaultdict(list); cols = defaultdict(list); data = defaultdict(list)
    for u, v, attr in G.edges(data=True):
        et = attr.get("edge_type", "ppi")
        w  = float(attr.get("weight", 1.0))
        i, j = idx_of[u], idx_of[v]
        rows[et] += [i, j]; cols[et] += [j, i]; data[et] += [w, w]

    layers = {}
    for et in rows:
        A = csr_matrix((data[et], (rows[et], cols[et])), shape=(n, n))
        A.sum_duplicates()
        layers[et] = A
        print(f"  layer '{et}': nnz = {A.nnz:,}")
    return layers


# ---------------------------------------------------------------------------
# 2. RANDOMIZED SUM-OF-SQUARED ADJACENCY EMBEDDING (Su et al. 2025)
# ---------------------------------------------------------------------------
def sparsify(A: csr_matrix, p: float, rng) -> csr_matrix:
    """Bernoulli(p) edge sampling with 1/p rescaling — preserves expectation."""
    if p >= 1.0:
        return A
    keep = rng.random(A.nnz) < p
    A_s = A.copy()
    A_s.data = A_s.data * keep.astype(A_s.dtype) / p
    A_s.eliminate_zeros()
    return A_s


def sos_embedding(layers: dict, K: int, p: float = 0.5,
                  oversample: int = 10, seed: int = 42, debias: bool = True):
    """
    Compute top-K eigenvectors of S = Σ_ℓ Ã_ℓ²  (optionally bias-adjusted)
    via randomized SVD. Ã_ℓ are sparsified layer adjacencies.
    """
    rng = np.random.default_rng(seed)
    n   = next(iter(layers.values())).shape[0]

    S = csr_matrix((n, n), dtype=np.float64)
    bias = np.zeros(n)
    for name, A in layers.items():
        A_s = sparsify(A.astype(np.float64), p, rng)
        S = S + (A_s @ A_s)
        if debias:
            # Diagonal of A_s A_sᵀ = row-sum of A_s squared element-wise
            bias += np.asarray(A_s.multiply(A_s).sum(axis=1)).ravel()
        print(f"  squared+accumulated '{name}'")
    if debias:
        S = S - diags(bias)

    # Symmetric → randomized_svd yields eigen-pairs (S is symmetric PSD up to bias term)
    print(f"  randomized SVD: n={n}, K={K}, oversample={oversample}")
    U, sigma, _ = randomized_svd(S, n_components=K,
                                 n_oversamples=oversample,
                                 random_state=seed)
    return sigma, U, S


# ---------------------------------------------------------------------------
# 3. DEFORD–PAULS DIAGONAL SUPRA-LAPLACIAN (alternative mode)
# ---------------------------------------------------------------------------
def supra_laplacian(layers: dict, omega: float):
    """
    Build the diagonal supra-Laplacian for k layers, n nodes each:
        L_sup = blockdiag(L_1..L_k) + ω (D_inter − A_inter)
    where A_inter connects copies of the same node across all layer pairs.
    Returns L_sup (sparse, kn × kn) and the layer order.
    """
    layer_names = list(layers.keys())
    k = len(layer_names); n = next(iter(layers.values())).shape[0]

    blocks = []
    for name in layer_names:
        A = layers[name]
        d = np.asarray(A.sum(axis=1)).ravel()
        L = diags(d) - A
        blocks.append(L)
    L_intra = bmat([[blocks[i] if i == j else None for j in range(k)]
                    for i in range(k)], format="csr")

    # Inter-layer: each node copy connected to its (k-1) siblings with weight ω
    I_n = identity(n, format="csr")
    Akk = (np.ones((k, k)) - np.eye(k))  # k x k all-ones off-diag
    A_inter = csr_matrix(np.kron(Akk, np.eye(1))) if False else None
    # Build directly:
    A_inter = bmat([[(omega * I_n) if i != j else None for j in range(k)]
                    for i in range(k)], format="csr")
    d_inter = np.asarray(A_inter.sum(axis=1)).ravel()
    L_inter = diags(d_inter) - A_inter

    L_sup = L_intra + L_inter
    print(f"  supra-Laplacian: {L_sup.shape[0]} rows, nnz={L_sup.nnz:,}, ω={omega}")
    return L_sup, layer_names, n


def supra_embedding(L_sup, K: int, seed: int = 42):
    """Smallest-K eigenvectors of L_sup via shift-invert randomized SVD trick."""
    # randomized_svd gives LARGEST singular values; for smallest eigvals of L_sup
    # we operate on (λ_max·I − L_sup). λ_max ≈ ||L_sup||_inf cheap bound.
    lam_max = float(abs(L_sup).sum(axis=1).max())
    M = (lam_max * identity(L_sup.shape[0])) - L_sup
    U, sigma, _ = randomized_svd(M, n_components=K + 1,
                                 n_oversamples=10, random_state=seed)
    # Drop the trivial constant eigenvector (largest sigma in M = smallest in L)
    return (lam_max - sigma)[1:K + 1], U[:, 1:K + 1]


# ---------------------------------------------------------------------------
# 4. CHOOSE K + CLUSTER + CHARACTERIZE
# ---------------------------------------------------------------------------
def find_optimal_k(embedding: np.ndarray, k_min=2, k_max=20, seed=42):
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(embedding)
        s  = silhouette_score(embedding, km.labels_) if k > 1 else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": s})
        print(f"  k={k:2d}  silhouette={s:.4f}  inertia={km.inertia_:.2f}")
    df = pd.DataFrame(rows)
    k_opt = int(df.loc[df["silhouette"].idxmax(), "k"])
    return k_opt, df


def run_clustering(embedding: np.ndarray, k: int, seed: int = 42):
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(embedding)
    return km.labels_


def characterize_nodes(labels, embedding, layers, nodelist, reg, out_csv):
    """Per-node table: cluster, layer (source), degree per layer, role."""
    n = len(nodelist)
    rec = reg.loc[nodelist].reset_index(drop=True)

    # degree per edge type
    deg = {name: np.asarray(A.sum(axis=1)).ravel() for name, A in layers.items()}

    # bridge score = participation across layers (std of normalised degrees)
    deg_mat = np.vstack([d / (d.max() + 1e-12) for d in deg.values()]).T  # n x k
    bridge  = deg_mat.std(axis=1)
    total   = deg_mat.sum(axis=1)

    hub_thr    = np.quantile(total,  0.90)
    bridge_thr = np.quantile(bridge, 0.90)

    role = np.where((total >= hub_thr) & (bridge >= bridge_thr), "multiplex_hub",
            np.where(total >= hub_thr,                            "intra_hub",
            np.where(bridge >= bridge_thr,                        "bridge",
                                                                  "peripheral")))
    df = pd.DataFrame({
        "node_id":  nodelist,
        "source":   rec["source"].values,
        "cell_type":rec["cell_type"].values,
        "gene":     rec["gene"].values,
        "cluster":  labels,
        "total_deg":total,
        "bridge_score": bridge,
        "role":     role,
    })
    for name, d in deg.items():
        df[f"deg_{name}"] = d
    df.to_csv(out_csv, index=False)
    print(f"Saved gene/cluster table → {out_csv}")
    return df


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph",      required=True, help="supra-graph .pkl")
    ap.add_argument("--registry",   required=True, help="node registry .csv")
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--mode",       choices=["sos", "supra"], default="sos")
    ap.add_argument("--K",          type=int, default=None,
                    help="embedding dim; if None, picked by silhouette")
    ap.add_argument("--k-min",      type=int, default=2)
    ap.add_argument("--k-max",      type=int, default=20)
    ap.add_argument("--sample-p",   type=float, default=0.5,
                    help="Bernoulli edge sampling prob (Su et al. 2025)")
    ap.add_argument("--omega",      type=float, default=1.0,
                    help="inter-layer coupling for supra mode")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--no-debias",  action="store_true")
    ap.add_argument("--force-recompute", action="store_true",
                help="recompute embedding even if cached")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    G, reg, nodelist, idx_of = load_multiplex(args.graph, args.registry)
    n = len(nodelist)
    layers = split_layers(G, idx_of, n)

    # ---- embedding --------------------------------------------------------
    K_emb = args.K if args.K else max(args.k_max, 20)
    
    # Check if embedding already exists
    embedding_path = out / "embedding.npy"
    if embedding_path.exists() and not args.force_recompute:
        print(f"Loading cached embedding from {embedding_path}")
        U = np.load(embedding_path)
    else:
        if args.mode == "sos":
            print(f"\n[mode=sos] randomized SoS embedding (K={K_emb}, p={args.sample_p})")
            sigma, U, S = sos_embedding(layers, K=K_emb, p=args.sample_p,
                                        seed=args.seed, debias=not args.no_debias)
            np.save(out / "eigvals_sos.npy", sigma)
        else:
            print(f"\n[mode=supra] DeFord–Pauls supra-Laplacian (ω={args.omega})")
            L_sup, layer_names, _= supra_laplacian(layers, args.omega)
            sigma, U = supra_embedding(L_sup, K=K_emb, seed=args.seed)
            k = len(layer_names)
            U = U.reshape(k, n, -1).mean(axis=0)
            np.save(out / "eigvals_supra.npy", sigma)
        np.save(out / "embedding.npy", U)
        
    print(f"\nEmbedding shape: {U.shape}")

    # ---- choose k ---------------------------------------------------------
    print("\nSilhouette sweep:")
    k_opt, k_df = find_optimal_k(U, args.k_min, args.k_max, seed=args.seed)
    k_df.to_csv(out / "k_selection.csv", index=False)
    print(f"  → optimal k = {k_opt}")

    # ---- cluster ----------------------------------------------------------
    labels = run_clustering(U, k_opt, seed=args.seed)
    gene_df = characterize_nodes(labels, U, layers, nodelist, reg,
                                 out / "spectral_clusters.csv")

    # ---- summary plot -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(args.k_min, args.k_max + 1), k_df["silhouette"], "o-")
    ax[0].axvline(k_opt, color="r", ls="--"); ax[0].set_xlabel("k"); ax[0].set_ylabel("silhouette")
    ax[0].set_title("Cluster selection")
    ax[1].bar(range(len(sigma)), sigma); ax[1].set_xlabel("idx"); ax[1].set_ylabel("eigenvalue")
    ax[1].set_title(f"Top-{len(sigma)} spectrum ({args.mode})")
    fig.tight_layout(); fig.savefig(out / "spectral_summary.png", dpi=150)

    # ---- meta -------------------------------------------------------------
    meta = dict(mode=args.mode, n_nodes=n, n_layers=len(layers),
                K_emb=int(K_emb), k_opt=int(k_opt),
                sample_p=args.sample_p, omega=args.omega,
                seed=args.seed, runtime_sec=time.time() - t0,
                layers={name: int(A.nnz) for name, A in layers.items()})
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone in {meta['runtime_sec']:.1f}s — outputs in {out}")


if __name__ == "__main__":
    main()