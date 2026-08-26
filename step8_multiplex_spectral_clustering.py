"""
step8_multiplex_spectral_clustering.py
------------------------------------------------------------------------------------
Randomized spectral clustering for the integrated multiplex disease network.

Two modes:
  --mode sos     : Su, Guo, Chang & Yang (2025) — randomized sum-of-squared
                   sparsified adjacencies with Lei & Lin (2023) bias adjustment.
                   Squaring handles assortative/disassortative layer mixing.
  --mode supra   : DeFord & Pauls (2017) diagonal supra-Laplacian
                   with inter-layer coupling weight `--omega`.

UPDATE (this version): can now run directly on a disease-module subgraph
instead of the full heterogeneous reference graph, and can drop the
global_ppi backbone before clustering the same way step6 can drop it
before module-building.
------------------------------------------------------------------------
Two independent things changed:

1. `--ref-graph` now accepts EITHER gene_reference_graph.pkl (the full
   heterogeneous graph) OR module_graph.pkl (step6's disease-module
   subgraph, e.g. 06_disease_module/disease_module_output/module_graph.pkl).
   No format conversion needed: module_graph.pkl is produced by
   `H.subgraph(final_module).copy()` in step6, so it's already namespaced
   "<celltype>::GENE" and already carries the same per-edge `edge_type`
   attribute ("ppi" / "inter" / "both"). Pointing --ref-graph at it
   restricts every downstream step (layer splitting, embedding, k
   selection, clustering, spectral_clusters.csv) to just the module's
   ~few-thousand genes instead of all 78,890.

2. NEW `--exclude-celltypes` / `--exclude-ppi-layer`, ported verbatim from
   step6's `filter_out_celltypes()`. Why this is still needed even when
   pointing at module_graph.pkl: step6's --exclude-ppi-seeds (the flag
   actually used to build this module) is the LIGHT-touch exclusion --
   global_ppi was kept in the graph as a bridging resource and only
   dropped from the SEED list, so global_ppi genes can still have been
   pulled into the final module as non-seed bridging nodes. If those
   nodes are still dense/high-degree, they can still dominate the SoS
   embedding the same way the full-graph global_ppi backbone did
   (mirrors the finding that motivated this change: the k=5 clustering
   on the full graph was substantially explained by global_ppi, per
   Marchenko-Pastur / silhouette diagnostics). Passing
   `--exclude-celltypes global_ppi` (or the shorthand `--exclude-ppi-layer`)
   here removes any remaining "global_ppi::GENE" nodes from the graph
   BEFORE layer splitting and embedding -- not after -- so they have zero
   influence on the clustering, the same guarantee step6 makes for module
   building.

   Check first whether this is actually needed: open disease_module.csv
   from step6 and look at the cell_type value_counts -- if global_ppi is a
   small minority of the module, this flag won't change much; if it's a
   large chunk, it's likely worth excluding here too.

------------------------------------------------------------------------
UPDATE (this version): row-normalized clustering + eigengap cross-check,
ported from the single-layer normalized-Laplacian pipeline
(spectral_clustering_gene_roles.py), to keep the two pipelines' clustering
methodology aligned rather than diverging into two different recipes.

Motivation: diagnostics on this module (step8b/c/d) found silhouette
staying implausibly high (>0.85) and nearly flat across every k from 2 to
50, on a graph that turned out to be connected (not fragmented) and whose
clusters don't track cell type either. Root cause, isolated via
step8c/step8d: a large fraction of low-degree "leaf" nodes were landing on
near-identical RAW embedding coordinates -- Spearman(degree, nearest-
neighbor distance) ~0.57-0.59 -- which unnormalized Euclidean k-means was
free to exploit as spuriously clean "clusters" regardless of k. The
single-layer script never had this failure mode because it always
row-normalizes (`sklearn.preprocessing.normalize`) before k-means, which
removes exactly this magnitude-driven collapse, and because it always
clusters in exactly k dimensions for a candidate k, not a fixed larger
embedding for every k in the sweep.

Two behavioral changes, both matching the single-layer script exactly:

  (a) ROW NORMALIZATION: embedding rows are L2-normalized immediately
      before k-means and before silhouette scoring, in both the k-sweep
      and the final clustering. On by default; disable with
      --no-row-normalize to recover the old (pre-this-update) behavior
      for comparison.

  (b) PER-k EMBEDDING DIMENSIONALITY: when testing candidate cluster
      count k, only the first k columns of the embedding are used (then
      row-normalized), not the full --K/K_emb-dimensional embedding for
      every k. This matches classic spectral clustering (use k
      eigenvectors to look for k clusters) and the single-layer script's
      `normalize(eigenvectors[:, :k])`. NOTE: if --k-max exceeds --K,
      candidate k values above K_emb are capped at K_emb columns (the
      pre-update behavior) since there's no more embedding to slice --
      the run prints a warning if this applies.

  (c) EIGENGAP CROSS-CHECK: alongside the silhouette sweep, an eigengap
      heuristic estimate is now computed and printed/saved, exactly as an
      independent cross-check the way the single-layer script reports
      both eigengap_k and best_sil_k. Not a hard override of the
      silhouette-picked k -- disagreement between the two is a flag to
      look closer (which is exactly what surfaced this whole issue).
      For --mode supra, sigma is ascending Laplacian-style eigenvalues
      (trivial already dropped), so the classic "biggest gap" eigengap
      applies directly. For --mode sos, sigma is descending singular
      values of a squared-adjacency sum (not a Laplacian), so this is a
      scree-plot analog: the biggest drop marks how many leading
      components carry disproportionate signal, not a literal spectral
      graph theory eigengap. Treat it as a heuristic hint either way.

NONE of this changes how the embedding itself (the SVD) is computed, so
existing cached embedding.npy / eigvals_<mode>.npy / embedding_cache_meta.json
from prior runs remain valid and will still be reused -- only the
downstream k-selection and clustering logic changed.

Everything else (node-name parsing, layer splitting, SoS/supra embedding,
output files) is unchanged from the previous version.
"""

import os, gc, json, pickle, argparse, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import os
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, lil_matrix, diags, identity, vstack, hstack, bmat
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Node-name parsing (nodes are "<celltype>::GENE", written by
# step2.5_namespace_and_seeds.py -- see that script for the exact
# namespacing convention this must stay in sync with).
# ---------------------------------------------------------------------------
def _celltype_of(node: str) -> str:
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


def _gene_of(node: str) -> str:
    return node.rsplit("::", 1)[-1] if "::" in node else node


def _embedding_cache_meta(args, n, K_emb):
    return dict(mode=args.mode, n_nodes=n, K_emb=int(K_emb),
                sample_p=args.sample_p, omega=args.omega,
                seed=args.seed, debias=not args.no_debias,
                ref_graph=str(args.ref_graph),
                excluded_celltypes=sorted(c.strip().upper() for c in (args.exclude_celltypes or [])),
                degree_normalize=bool(args.degree_normalize),
                reg_tau=args.reg_tau)


# ---------------------------------------------------------------------------
# 1. LOAD REFERENCE GRAPH (works on either gene_reference_graph.pkl or
#    step6's module_graph.pkl -- both are namespaced nx.Graphs with a
#    per-edge edge_type attribute)
# ---------------------------------------------------------------------------
def load_reference_graph(ref_graph_pkl: str):
    print(f"Loading reference graph from {ref_graph_pkl} ...")
    with open(ref_graph_pkl, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip() for n in G.nodes()})
    nodelist = sorted(G.nodes())
    idx_of = {n: i for i, n in enumerate(nodelist)}
    n_ppi = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") in ("ppi", "both"))
    n_inter = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") in ("inter", "both"))
    print(f"  {len(nodelist):,} nodes, {G.number_of_edges():,} edges "
          f"({n_ppi:,} ppi-tagged, {n_inter:,} inter-tagged)")
    return G, nodelist, idx_of


# ---------------------------------------------------------------------------
# NEW: cell-type exclusion, ported from step6's filter_out_celltypes().
# Removes every node whose "<celltype>::" prefix matches, entirely --
# node, its edges, everything. Case-insensitive on the prefix.
# ---------------------------------------------------------------------------
def filter_out_celltypes(G: nx.Graph, excluded_celltypes):
    if not excluded_celltypes:
        return G
    excluded_set = {c.strip().upper() for c in excluded_celltypes}
    to_remove = [n for n in G.nodes() if _celltype_of(n).upper() in excluded_set]
    if not to_remove:
        print(f"  [exclude-celltypes] None of {sorted(excluded_set)} found in graph "
              f"(checked against node prefixes) — nothing removed.")
        return G
    remove_set = set(to_remove)
    kept = [n for n in G.nodes() if n not in remove_set]
    G_filtered = G.subgraph(kept).copy()
    print(f"  [exclude-celltypes] Removed cell type(s) {sorted(excluded_set)}: "
          f"{len(to_remove):,} nodes dropped -> {G_filtered.number_of_nodes():,} nodes, "
          f"{G_filtered.number_of_edges():,} edges remain "
          f"(was {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")
    return G_filtered


# ---------------------------------------------------------------------------
# NEW: optional finer re-derivation of the "inter" layer into
# ligand_receptor / reactome / other, mirroring step7c's edge labeling, for
# users who want more than the default 2-layer (ppi / inter) split.
# ---------------------------------------------------------------------------
def load_lr_set(lr_csv: str) -> set:
    df = pd.read_csv(lr_csv)
    if "ligand_gname" not in df.columns:
        df = pd.read_csv(lr_csv, sep="\t")
    df["ligand_gname"] = df["ligand_gname"].str.strip().str.upper()
    df["receptor_gname"] = df["receptor_gname"].str.strip().str.upper()
    return set(zip(df["ligand_gname"], df["receptor_gname"]))


def load_scored_lr_pairs(provenance_csv: str) -> set:
    df = pd.read_csv(provenance_csv)
    df["ligand"] = df["ligand"].astype(str).str.strip().str.upper()
    df["receptor"] = df["receptor"].astype(str).str.strip().str.upper()
    return set(zip(df["ligand"], df["receptor"]))


def load_pathway_pairs(reactome_csv: str, uniprot_map_csv: str | None) -> set:
    u2s = {}
    if uniprot_map_csv:
        u2s_df = pd.read_csv(uniprot_map_csv)
        u2s = dict(zip(u2s_df["uniprotid"], u2s_df["symbol"]))
    df_temp = pd.read_csv(reactome_csv, nrows=5)
    use_symbol_cols = {"symbol1", "symbol2"}.issubset(df_temp.columns)
    id_cols = ["symbol1", "symbol2"] if use_symbol_cols else ["uniprot1", "uniprot2"]
    df = pd.read_csv(reactome_csv, low_memory=False, usecols=id_cols + ["common_pathway"]).dropna(subset=["common_pathway"])
    c1 = df[id_cols[0]].astype(str).str.strip().str.upper()
    c2 = df[id_cols[1]].astype(str).str.strip().str.upper()
    if not use_symbol_cols and u2s:
        lookup = {str(k).upper(): v for k, v in u2s.items()}
        c1 = c1.map(lookup).fillna(c1).astype(str).str.upper()
        c2 = c2.map(lookup).fillna(c2).astype(str).str.upper()
    return set(zip(c1, c2)) | set(zip(c2, c1))


def refine_inter_edge_type(u, v, lr_set, scored_lr_set, pw_set) -> str:
    ga, gb = _gene_of(u), _gene_of(v)
    is_lr = (lr_set and ((ga, gb) in lr_set or (gb, ga) in lr_set)) or \
            (scored_lr_set and ((ga, gb) in scored_lr_set or (gb, ga) in scored_lr_set))
    is_pw = pw_set is not None and (ga, gb) in pw_set
    if is_lr:
        return "ligand_receptor"
    if is_pw:
        return "reactome"
    return "other"


# ---------------------------------------------------------------------------
# 2. SPLIT INTO PER-LAYER SPARSE ADJACENCY MATRICES
# ---------------------------------------------------------------------------
def split_layers(G: nx.Graph, idx_of: dict, n: int, refine_inter: bool,
                  lr_set=None, scored_lr_set=None, pw_set=None):
    """
    Build per-edge-type sparse adjacency matrices on the SAME node index.
    Default (refine_inter=False): 2 layers, "ppi" and "inter", straight off
    the reference/module graph's own `edge_type` attribute. With
    refine_inter=True (i.e. at least one of --lr-csv/--reactome-csv/
    --scored-lr-provenance-csv given), the "inter" layer is instead split
    into "ligand_receptor" / "reactome" / "other" by re-checking each
    inter-tagged edge's gene pair, same idea as step7c's edge labeling.
    Returns dict {layer_name: csr_matrix (n x n, symmetric, weighted)}.
    """
    rows = defaultdict(list); cols = defaultdict(list); data = defaultdict(list)
    for u, v, attr in G.edges(data=True):
        et = attr.get("edge_type", "ppi")
        w = 1.0  # reference/module graph edges are unweighted presence/absence
        i, j = idx_of[u], idx_of[v]

        target_layers = []
        if et in ("ppi", "both"):
            target_layers.append("ppi")
        if et in ("inter", "both"):
            if refine_inter:
                target_layers.append(refine_inter_edge_type(u, v, lr_set, scored_lr_set, pw_set))
            else:
                target_layers.append("inter")

        for layer in target_layers:
            rows[layer] += [i, j]; cols[layer] += [j, i]; data[layer] += [w, w]

    layers = {}
    for et in rows:
        A = csr_matrix((data[et], (rows[et], cols[et])), shape=(n, n))
        A.sum_duplicates()
        layers[et] = A
        print(f"  layer '{et}': nnz = {A.nnz:,}")
    return layers


# ---------------------------------------------------------------------------
# 3. RANDOMIZED SUM-OF-SQUARED ADJACENCY EMBEDDING (Su et al. 2025)
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


def normalize_adjacency(A: csr_matrix, tau=None):
    """
    Symmetric degree normalization D^(-1/2) A D^(-1/2), regularized by
    adding tau to every degree (Amini et al. 2013 / Qin & Rohe 2013 style
    "regularized graph Laplacian" trick) so near-isolated nodes don't get
    blown up by dividing by a near-zero degree. If tau is None, defaults
    to this layer's mean degree, a common rule-of-thumb choice.

    Without this, raw adjacency (or its square, as in SoS) is dominated
    by degree heterogeneity: entries scale with the product of endpoint
    degrees, so on a heavy-tailed PPI-style degree distribution a handful
    of hubs can swamp the leading eigenvectors and the resulting
    "clusters" end up splitting nodes by hubness (high-degree vs.
    low-degree) rather than by community/module membership.
    """
    d = np.asarray(A.sum(axis=1)).ravel()
    if tau is None:
        tau = float(d.mean()) if d.size else 0.0
    d_reg = d + tau
    d_reg[d_reg == 0] = 1.0  # guard fully-isolated nodes if tau==0
    d_inv_sqrt = 1.0 / np.sqrt(d_reg)
    D_inv_sqrt = diags(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def sos_embedding(layers: dict, K: int, p: float = 0.5,
                  oversample: int = 10, seed: int = 42, debias: bool = True,
                  degree_normalize: bool = False, reg_tau=None):
    """
    Compute top-K eigenvectors of S = Σ_ℓ Ã_ℓ²  (optionally bias-adjusted)
    via randomized SVD. Ã_ℓ are sparsified layer adjacencies.

    If degree_normalize=True, each layer is symmetrically degree-normalized
    (see normalize_adjacency()) BEFORE sparsification/squaring, which is
    the standard remedy for hub-vs-periphery splits caused by degree
    heterogeneity (see normalize_adjacency() docstring).
    """
    rng = np.random.default_rng(seed)
    n   = next(iter(layers.values())).shape[0]

    S = csr_matrix((n, n), dtype=np.float64)
    bias = np.zeros(n)
    for name, A in layers.items():
        A_f = A.astype(np.float64)
        if degree_normalize:
            A_f = normalize_adjacency(A_f, tau=reg_tau)
            print(f"  degree-normalized '{name}' (tau={'mean-degree' if reg_tau is None else reg_tau})")
        A_s = sparsify(A_f, p, rng)
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
# 4. DEFORD–PAULS DIAGONAL SUPRA-LAPLACIAN (alternative mode)
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
# NEW: eigengap heuristic cross-check, ported from the single-layer
# normalized-Laplacian pipeline (spectral_clustering_gene_roles.py's
# find_optimal_k). Independent second opinion alongside silhouette -- see
# module docstring for why mode changes which direction "the gap" is read.
# ---------------------------------------------------------------------------
def estimate_eigengap_k(sigma, mode: str, k_max: int):
    sigma = np.asarray(sigma)
    usable = sigma[:k_max] if k_max else sigma
    if len(usable) < 3:
        return None
    gaps = np.diff(usable)
    if mode == "supra":
        # ascending Laplacian-style eigenvalues (trivial already dropped):
        # classic eigengap = biggest jump. +2 matches the single-layer
        # script's convention (+1 for the dropped trivial eigenvalue,
        # +1 to convert a 0-indexed gap position into a cluster count).
        k_est = int(np.argmax(gaps) + 2)
    else:
        # descending scree-style spectrum: biggest DROP marks how many
        # leading components carry disproportionate signal.
        k_est = int(np.argmax(-gaps) + 1)
    return k_est


# ---------------------------------------------------------------------------
# 5. CHOOSE K + CLUSTER + CHARACTERIZE
# ---------------------------------------------------------------------------
def _fit_one_k(k, embedding, seed, silhouette_sample_size, row_normalize):
    """
    NOTE (this update): now slices embedding[:, :k] -- exactly k columns for
    candidate cluster count k -- before optionally row-normalizing, matching
    the single-layer script's `normalize(eigenvectors[:, :k])`. Previously
    this always used the full fixed-width embedding regardless of k. If k
    exceeds the embedding's actual width (K_emb), the slice is simply
    capped at all available columns (numpy does this silently) -- this only
    happens when --k-max > --K, which main() warns about up front.
    """
    n = embedding.shape[0]
    emb_k = embedding[:, :k]
    if row_normalize:
        emb_k = normalize(emb_k)
    samp = min(silhouette_sample_size, n) if silhouette_sample_size else None
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb_k)
    s = silhouette_score(emb_k, km.labels_, sample_size=samp, random_state=seed) if k > 1 else np.nan
    print(f"  k={k:2d}  silhouette={s:.4f}  inertia={km.inertia_:.2f}")
    return {"k": k, "inertia": km.inertia_, "silhouette": s, "labels": km.labels_}


def find_optimal_k(embedding, k_min=2, k_max=20, seed=42, n_jobs=1,
                    silhouette_sample_size=5000, row_normalize=True):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_k)(k, embedding, seed, silhouette_sample_size, row_normalize)
        for k in range(k_min, k_max + 1)
    )
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "labels"} for r in results])
    k_opt = int(df.loc[df["silhouette"].idxmax(), "k"])

    # Rigor check: exact (non-sampled) silhouette for the chosen k only
    labels_opt = next(r["labels"] for r in results if r["k"] == k_opt)
    emb_opt = embedding[:, :k_opt]
    if row_normalize:
        emb_opt = normalize(emb_opt)
    exact_s = silhouette_score(emb_opt, labels_opt)  # no sample_size -> full O(n^2)
    print(f"  → optimal k = {k_opt}  (sampled silhouette={df.loc[df['k']==k_opt,'silhouette'].values[0]:.4f}, "
          f"exact silhouette={exact_s:.4f})")
    df.loc[df["k"] == k_opt, "silhouette_exact"] = exact_s

    return k_opt, df

def run_clustering(embedding: np.ndarray, k: int, seed: int = 42, row_normalize: bool = True):
    emb_k = embedding[:, :k]
    if row_normalize:
        emb_k = normalize(emb_k)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(emb_k)
    return km.labels_


def characterize_nodes(labels, embedding, layers, nodelist, out_csv):
    """Per-node table: cluster, cell_type/gene (parsed off the node name),
    degree per layer, role."""
    n = len(nodelist)
    cell_types = [_celltype_of(nd) for nd in nodelist]
    genes = [_gene_of(nd) for nd in nodelist]

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
        "cell_type":cell_types,
        "gene":     genes,
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
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-graph", required=True,
                     help="Either gene_reference_graph.pkl / gene_ppi_reference_graph.pkl "
                          "(full heterogeneous graph) OR step6's module_graph.pkl "
                          "(e.g. 06_disease_module/disease_module_output/module_graph.pkl) "
                          "to restrict clustering to the disease module only. Both formats "
                          "are namespaced '<celltype>::GENE' nodes with 'edge_type' in "
                          "{ppi, inter, both}.")
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
    ap.add_argument("--n-jobs", type=int,
                     default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                     help="parallel workers for the k-sweep; defaults to $SLURM_CPUS_PER_TASK if set")
    ap.add_argument("--silhouette-sample-size", type=int, default=5000,
                     help="subsample size for silhouette during k-sweep (speed); "
                          "the winning k also gets one exact full-n silhouette for the record")
    ap.add_argument("--no-debias",  action="store_true")
    ap.add_argument("--degree-normalize", action="store_true",
                     help="Symmetrically degree-normalize each layer (D^-1/2 A D^-1/2, "
                          "regularized by --reg-tau) before sparsification/squaring in "
                          "sos mode. Use this if clustering is splitting nodes by hub-vs-"
                          "periphery degree instead of by community/module -- diagnosable "
                          "via a big skew in deg_ppi/deg_inter between clusters in "
                          "spectral_clusters.csv, especially one giant low-degree cluster "
                          "plus a few tiny high-degree ones. No effect in supra mode (the "
                          "Laplacian there is already degree-aware via D_inter/D_intra).")
    ap.add_argument("--reg-tau", type=float, default=None,
                     help="Regularization constant added to every node's degree before "
                          "the D^-1/2 normalization in --degree-normalize (Amini et al. "
                          "2013 / Qin & Rohe 2013 'regularized graph Laplacian' style). "
                          "Default: each layer's own mean degree.")
    ap.add_argument("--no-row-normalize", action="store_true",
                     help="Disable L2 row-normalization before k-means/silhouette (on by "
                          "default as of this update, matching the single-layer normalized-"
                          "Laplacian pipeline's `normalize(eigenvectors[:, :k])`). Only "
                          "affects clustering, not the cached embedding -- pass this to "
                          "reproduce pre-update clustering behavior on an existing cached "
                          "embedding.npy without recomputing anything.")
    ap.add_argument("--force-recompute", action="store_true",
                help="recompute embedding even if cached")
    ap.add_argument("--exclude-celltypes", nargs="+", default=[], metavar="CELLTYPE",
                     help="Cell type(s) (the '<celltype>::' namespace prefix) to drop "
                          "entirely from the graph BEFORE layer splitting/embedding, e.g. "
                          "--exclude-celltypes global_ppi. Case-insensitive. Use this if "
                          "you're clustering module_graph.pkl and it still contains "
                          "global_ppi nodes pulled in as non-seed bridging genes (step6's "
                          "--exclude-ppi-seeds keeps global_ppi available for bridging, it "
                          "only drops it from the SEED list -- so it can still end up in "
                          "the final module).")
    ap.add_argument("--exclude-ppi-layer", action="store_true",
                     help="Shorthand for --exclude-celltypes global_ppi.")
    ap.add_argument("--lr-csv", default=None,
                     help="Optional. If given (with/without --reactome-csv/"
                          "--scored-lr-provenance-csv), the 'inter' layer is split "
                          "into ligand_receptor/reactome/other instead of staying "
                          "one combined layer -- see module docstring.")
    ap.add_argument("--reactome-csv", default=None)
    ap.add_argument("--uniprot-map", default=None)
    ap.add_argument("--scored-lr-provenance-csv", default=None,
                     help="scored_lr_edge_provenance.csv from step2_build_bipartite_edges.py "
                          "(recognizes LR edges that came only from LIANA/NeuronChat, not the "
                          "flat --lr-csv resource)")
    args = ap.parse_args()

    row_normalize = not args.no_row_normalize

    exclude_celltypes = list(args.exclude_celltypes)
    if args.exclude_ppi_layer and "global_ppi" not in {c.lower() for c in exclude_celltypes}:
        exclude_celltypes.append("global_ppi")
    args.exclude_celltypes = exclude_celltypes  # so _embedding_cache_meta sees the resolved list

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    G, nodelist, idx_of = load_reference_graph(args.ref_graph)

    if exclude_celltypes:
        print(f"\nExcluding cell type(s) {exclude_celltypes} before layer splitting/embedding ...")
        G = filter_out_celltypes(G, exclude_celltypes)
        nodelist = sorted(G.nodes())
        idx_of = {n: i for i, n in enumerate(nodelist)}

    n = len(nodelist)
    if n == 0:
        raise SystemExit("No nodes left after loading/exclusion — check --ref-graph and --exclude-celltypes.")

    refine_inter = bool(args.lr_csv or args.reactome_csv or args.scored_lr_provenance_csv)
    lr_set = load_lr_set(args.lr_csv) if args.lr_csv else None
    scored_lr_set = load_scored_lr_pairs(args.scored_lr_provenance_csv) if args.scored_lr_provenance_csv else None
    pw_set = load_pathway_pairs(args.reactome_csv, args.uniprot_map) if args.reactome_csv else None
    if refine_inter:
        print("Refining 'inter' layer into ligand_receptor/reactome/other ...")

    layers = split_layers(G, idx_of, n, refine_inter, lr_set, scored_lr_set, pw_set)

    # ---- embedding --------------------------------------------------------
    K_emb = args.K if args.K else max(args.k_max, 20)
    if K_emb >= n:
        print(f"  WARNING: K_emb={K_emb} >= n_nodes={n}; clamping K_emb to {max(n - 1, 1)}. "
              f"This is expected if you're clustering a small disease module -- lower "
              f"--K/--k-max explicitly to avoid the warning.")
        K_emb = max(n - 1, 1)
    if args.k_max > K_emb:
        print(f"  NOTE: --k-max ({args.k_max}) exceeds the embedding width K_emb ({K_emb}). "
              f"For candidate k > {K_emb}, per-k clustering is capped at all {K_emb} available "
              f"embedding columns (same as pre-row-normalize-update behavior for those k) -- "
              f"raise --K if you want a wider embedding to slice from.")

    embedding_path = out / "embedding.npy"
    eigvals_path = out / f"eigvals_{args.mode}.npy"
    cache_meta_path = out / "embedding_cache_meta.json"
    expected_meta = _embedding_cache_meta(args, n, K_emb)

    use_cache = (embedding_path.exists() and eigvals_path.exists()
                 and cache_meta_path.exists() and not args.force_recompute)
    if use_cache:
        with open(cache_meta_path) as f:
            cached_meta = json.load(f)
        if cached_meta != expected_meta:
            print(f"  WARNING: cached embedding params don't match this run.\n"
                  f"    cached:   {cached_meta}\n"
                  f"    expected: {expected_meta}\n"
                  f"  Recomputing.")
            use_cache = False

    if use_cache:
        print(f"Loading cached embedding from {embedding_path}")
        U = np.load(embedding_path)
        sigma = np.load(eigvals_path)
        if U.shape[0] != n:  # belt-and-suspenders for pre-existing caches without meta
            print(f"  WARNING: cached embedding has {U.shape[0]:,} rows but graph has "
                  f"{n:,} nodes. Recomputing.")
            use_cache = False

    if not use_cache:
        if args.mode == "sos":
            print(f"\n[mode=sos] randomized SoS embedding (K={K_emb}, p={args.sample_p})")
            sigma, U, S = sos_embedding(layers, K=K_emb, p=args.sample_p,
                                        seed=args.seed, debias=not args.no_debias,
                                        degree_normalize=args.degree_normalize,
                                        reg_tau=args.reg_tau)
        else:
            print(f"\n[mode=supra] DeFord–Pauls supra-Laplacian (ω={args.omega})")
            L_sup, layer_names, _ = supra_laplacian(layers, args.omega)
            sigma, U = supra_embedding(L_sup, K=K_emb, seed=args.seed)
            k = len(layer_names)
            U = U.reshape(k, n, -1).mean(axis=0)
        np.save(embedding_path, U)
        np.save(eigvals_path, sigma)
        with open(cache_meta_path, "w") as f:
            json.dump(expected_meta, f, indent=2)
    print(f"\nEmbedding shape: {U.shape}")
    print(f"Row-normalization before clustering: {'ON' if row_normalize else 'OFF (--no-row-normalize)'}")

    # ---- choose k ---------------------------------------------------------
    k_max_eff = min(args.k_max, n - 1) if n > 1 else args.k_max
    if k_max_eff != args.k_max:
        print(f"  NOTE: clamping --k-max from {args.k_max} to {k_max_eff} (n_nodes={n:,}).")

    eigengap_k = estimate_eigengap_k(sigma, args.mode, k_max_eff)
    if eigengap_k is not None:
        print(f"\nEigengap heuristic (cross-check only, mirrors the single-layer pipeline): "
              f"suggests k = {eigengap_k}")

    print("\nSilhouette sweep:")
    k_opt, k_df = find_optimal_k(U, args.k_min, k_max_eff, seed=args.seed,
                                  n_jobs=args.n_jobs,
                                  silhouette_sample_size=args.silhouette_sample_size,
                                  row_normalize=row_normalize)
    k_df.to_csv(out / "k_selection.csv", index=False)
    print(f"  → optimal k (silhouette) = {k_opt}")
    if eigengap_k is not None and eigengap_k != k_opt:
        print(f"  NOTE: eigengap heuristic (k={eigengap_k}) and silhouette-picked k ({k_opt}) "
              f"disagree -- worth a closer look before trusting k_opt biologically (this is "
              f"exactly the kind of disagreement that surfaced the leaf-collapse issue on "
              f"this module before this update).")

    # ---- cluster ----------------------------------------------------------
    labels = run_clustering(U, k_opt, seed=args.seed, row_normalize=row_normalize)
    gene_df = characterize_nodes(labels, U, layers, nodelist, out / "spectral_clusters.csv")

    # ---- summary plot -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(args.k_min, k_max_eff + 1), k_df["silhouette"], "o-")
    ax[0].axvline(k_opt, color="r", ls="--", label=f"silhouette k={k_opt}")
    if eigengap_k is not None and args.k_min <= eigengap_k <= k_max_eff:
        ax[0].axvline(eigengap_k, color="orange", ls="--", label=f"eigengap k={eigengap_k}")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("silhouette")
    ax[0].set_title("Cluster selection")
    ax[0].legend(fontsize=8)
    ax[1].bar(range(len(sigma)), sigma); ax[1].set_xlabel("idx"); ax[1].set_ylabel("eigenvalue")
    ax[1].set_title(f"Top-{len(sigma)} spectrum ({args.mode})")
    fig.tight_layout(); fig.savefig(out / "spectral_summary.png", dpi=150)

    # ---- meta -------------------------------------------------------------
    meta = dict(mode=args.mode, ref_graph=str(args.ref_graph),
                excluded_celltypes=exclude_celltypes,
                n_nodes=n, n_layers=len(layers),
                K_emb=int(K_emb), k_opt=int(k_opt),
                eigengap_k=None if eigengap_k is None else int(eigengap_k),
                row_normalize=row_normalize,
                sample_p=args.sample_p, omega=args.omega,
                seed=args.seed, runtime_sec=time.time() - t0,
                layers={name: int(A.nnz) for name, A in layers.items()})
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone in {meta['runtime_sec']:.1f}s — outputs in {out}")


if __name__ == "__main__":
    main()