#!/usr/bin/env python3
"""
step9_multiplex_marchenko_pastur_v3.py
------------------------------------------------------------------------------
Same as v2 (real disease module, cell-type layers, real inter-layer coupling,
Gavish-Donoho MP), with two changes based on inspecting v2's real output on
your data:

  1. GLOBAL_PPI EDGE PROJECTION (replaces plain drop-and-lose).
     v2 either dropped global_ppi:: nodes entirely (losing every edge that
     was ONLY supported by the background interactome, including for genes
     that aren't otherwise isolated -- e.g. two genes with a real,
     literature-known PPI edge that just wasn't captured cell-type-
     specifically) or kept global_ppi as its own giant non-cell-type-
     specific layer (which then dominates hub/bridge stats for the wrong
     reason -- being globally well-connected, not being a genuine
     cross-cell-type bridge).

     v3 instead PROJECTS each global_ppi::GENE_A <-> global_ppi::GENE_B
     background-PPI edge onto every real cell type where BOTH genes are
     already present as celltype::GENE_A / celltype::GENE_B nodes (i.e.
     you already have cell-type-specific evidence both proteins are
     relevant there) -- added as an intra-layer edge tagged
     edge_type="ppi", source="background_ppi_projected", with a
     configurable weight < 1 (default 0.5) to mark it as
     literature/background-supported rather than directly observed.
     global_ppi:: nodes themselves are then dropped as before. A gene that
     ends up isolated AFTER this projection genuinely has no cell-type-
     context edge to lean on (measured OR background-supported via a
     co-present partner), so dropping it at that point is now fully
     justified rather than a construction artifact.

  2. BRIDGE FLAGGING: MIN-PRESENCE + FRACTIONAL THRESHOLD (replaces the
     absolute ">= 2 negative layers" rule).
     Real v2 output showed the flag rate under ">=2 negative layers" was
     ~100% for every gene present in >=3 cell types -- i.e. the rule was
     not discriminating anything above n=2 (consistent with the coin-flip
     argument: P(>=2 negative out of n fair flips) -> 1 fast as n grows).
     v3 replaces it with:
       - a MINIMUM PRESENCE FILTER (--min-celltypes-present, default 3):
         genes present in fewer cell types than this are marked
         "insufficient_data" and excluded from bridge calling outright --
         you cannot support a claim of "consistently a bottleneck" from 1-2
         data points, whatever scoring rule you use on them.
       - a FRACTIONAL threshold (--frac-neg-threshold, default 0.6) on the
         genes that pass the presence filter: fraction of PRESENT layers
         with negative Fiedler sign, not a raw count. This removes the
         n-dependence the absolute rule had (a gene present in 15 layers
         no longer clears the bar with just 2 negative).
     The old absolute rule is kept available via --bridge-method absolute
     for comparison; default is --bridge-method fractional.

v4 ADDS: restrict-to-largest-connected-component before any spectral
analysis (--no-lcc-restrict to disable). Real output on an actual disease
module showed the eigenvalue-based "n_trivial" check silently undercounted
disconnection: nx.connected_components found 51 true components (a 4,280-
node giant component plus 50 fragments of 2-5 nodes), while the k=6
Lanczos call only ever reported up to 2 near-zero eigenvalues -- it simply
never requested enough eigenvalues to see the rest. A Fiedler vector is
only meaningful as the smoothest bipartition of a CONNECTED graph; computed
on a fragmented one, the eigenvector actually picked is an ambiguous
mixture, not a clean answer. v4 checks and restricts using
nx.connected_components() directly (exact, not approximate) both before
the main spectral run and again after the perturbation step (which can
itself fragment the module by removing bridge genes).

v5 FIXES A REAL BUG FOUND ON ACTUAL DATA: after the v4 LCC fix corrected
which eigenvector was being used (see v4/v5 run logs), the real global
Fiedler sign split turned out to be 4,045 negative vs. 235 positive
(94.5% / 5.5%) -- wildly imbalanced. Earlier versions hardcoded
"negative = bridge", which happens to coincide with "minority = bridge"
only when the split is roughly balanced. At 94.5/5.5 that assumption
breaks completely: "negative" was just the bulk of the module, and the
structurally distinctive group was the small POSITIVE minority. A Fiedler
eigenvector's sign is mathematically arbitrary in the first place (eigsh
could equally validly return the whole vector negated) -- what's
structurally meaningful was never "negative" per se, it's "minority side
of the partition." v5 auto-detects which sign is the minority fresh on
every run (determine_minority_sign()) and flags bridges by minority-side
consistency instead of a hardcoded direction, so this can't silently break
again if the balance shifts (or flips) on a future dataset. CLI flags
renamed accordingly: --min-neg-layers -> --min-minority-layers,
--frac-neg-threshold -> --frac-minority-threshold. Output columns renamed:
n_neg_layers -> n_minority_layers, frac_neg_layers -> frac_minority_layers.

v6 ADDS: --omega-sweep, a built-in sweep over multiple omega (inter-layer
coupling strength) values in one job. Previously this required manually
re-running the script once per omega and diffing the output CSVs by hand.
Now: `--omega-sweep 0.1 1.0 5.0` runs the full omega-dependent portion of
the pipeline (supra-Laplacian assembly, minority-sign detection, bridge
flagging, perturbation) once per value -- reusing the omega-INDEPENDENT
work (graph loading/LCC restriction, cell-type layers, per-layer MP fits,
coupling-block construction) computed only once -- then automatically
writes a comparison of the top-N candidate genes across all omega values
(pairwise Jaccard overlap + the "robust core" of genes that rank near the
top regardless of omega) to omega_sweep_comparison.csv and
omega_sweep_summary.txt. This matters because omega is a sensitivity knob
you chose, not something measured from data -- if your top candidates
change a lot across a reasonable omega range, that's telling you the
result is coupling-strength-dependent, which you want to know before
reporting it as a single fixed answer.

v7 FIXES THE SWEEP RANKING: v6's omega-sweep comparison ranked candidate
genes by raw frac_minority_layers, which is biased toward genes present in
only a few cell types (a 3-layer gene needs just 2/3 minority to score
0.667, trivially beating a 20-layer gene that's minority in 5/20=0.25 --
even though 5/20 against a small per-layer base rate is a far more
extreme, permutation-significant result). This meant the sweep's "robust
core" could easily miss your actual best hit while reporting a set of
lucky low-presence genes as "stable." v7 runs the permutation test
in-process for every omega automatically (previously a separate manual
step via permutation_test_bridges.py, now folded directly into
run_omega_analysis -- no CSV round-trip needed), adds perm_pvalue /
perm_qvalue_bh columns to each omega's multiplex_bridges.csv, and ranks
the sweep comparison by p-value instead of raw fraction. It also reports
a new ROBUST SIGNIFICANT CORE: genes with perm_qvalue_bh <= --fdr in
EVERY omega tested -- the direct answer to "does my hit stay significant
across a reasonable coupling-strength range," not just "does it stay in
an arbitrary top-N." New flags: --n-perm (default 2000), --perm-seed,
--fdr (default 0.05), --skip-permutation-test (to opt back out for speed,
falling back to raw-fraction ranking with a clear warning).

Everything else (Gavish-Donoho MP fitting, real coupling blocks, ragged
no-padding layers, shift-invert-then-fallback bulk fetch, supra-Laplacian
Fiedler, global_ppi projection, LCC restriction, minority-sign auto-
detected bridge flagging, --omega-sweep structure) is unchanged from v6.
"""

import json
import pickle
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy import integrate
from scipy.sparse import csr_matrix, coo_matrix, diags, bmat
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# Node-name parsing
# ==============================================================================
def _celltype_of(node: str) -> str:
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


def _gene_of(node: str) -> str:
    return node.rsplit("::", 1)[-1] if "::" in node else node


# ==============================================================================
# Gavish-Donoho (2014) MP machinery (unchanged from v2)
# ==============================================================================
def marchenko_pastur_pdf(x, beta, sigma=1.0):
    if beta > 1:
        beta = 1.0 / beta
    beta_minus = sigma**2 * (1 - np.sqrt(beta))**2
    beta_plus = sigma**2 * (1 + np.sqrt(beta))**2
    if x <= beta_minus or x >= beta_plus:
        return 0.0
    numerator = np.sqrt((beta_plus - x) * (x - beta_minus))
    denominator = 2 * np.pi * beta * sigma**2 * x
    return numerator / denominator


def marchenko_pastur_median(beta, sigma=1.0):
    if beta > 1:
        beta = 1.0 / beta
    beta_minus = sigma**2 * (1 - np.sqrt(beta))**2
    beta_plus = sigma**2 * (1 + np.sqrt(beta))**2

    def mp_cdf_minus_half(x):
        if x <= beta_minus:
            return -0.5
        if x >= beta_plus:
            return 0.5
        result, _ = integrate.quad(lambda t: marchenko_pastur_pdf(t, beta, sigma),
                                    beta_minus, x)
        return result - 0.5

    from scipy.optimize import brentq
    return brentq(mp_cdf_minus_half, beta_minus, beta_plus, xtol=1e-10)


def lambda_star(beta):
    if beta > 1:
        beta = 1.0 / beta
    return np.sqrt(2 * (beta + 1) + (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1)))


def omega_exact(beta, sigma=1.0):
    if beta > 1:
        beta = 1.0 / beta
    mu_beta = marchenko_pastur_median(beta, sigma)
    return lambda_star(beta) / np.sqrt(mu_beta)


def marchenko_pastur_bounds(N, M, sigma=1.0):
    beta = min(N, M) / max(N, M)
    beta_minus = sigma**2 * (1 - np.sqrt(beta))**2
    beta_plus = sigma**2 * (1 + np.sqrt(beta))**2
    return beta_minus, beta_plus


def fit_mp_gavish_donoho(bulk_eigs, outlier_candidates, label, out_path, N, M=None):
    bulk_eigs = np.asarray(bulk_eigs, dtype=float)
    bulk_eigs = bulk_eigs[np.isfinite(bulk_eigs)]
    bulk_pos = bulk_eigs[bulk_eigs > 0]

    if bulk_pos.size < 5:
        print(f"  [{label}] too few positive bulk eigenvalues ({bulk_pos.size}) -- skipping MP fit")
        return None

    M = M or N
    beta = min(N, M) / max(N, M)
    sigma_est = float(np.std(bulk_pos))
    lam_minus, lam_plus = marchenko_pastur_bounds(N, M, sigma_est)

    y_med = float(np.median(bulk_pos))
    omega_beta = omega_exact(beta, sigma=1.0)
    tau_star = omega_beta * y_med

    outlier_candidates = np.asarray(outlier_candidates, dtype=float)
    outliers = outlier_candidates[outlier_candidates > lam_plus]
    outliers_thresh = outlier_candidates[outlier_candidates > tau_star]

    print(f"  [{label}] beta={beta:.3f}  sigma_hat={sigma_est:.4f}  "
          f"MP bounds=[{lam_minus:.4f}, {lam_plus:.4f}]  "
          f"optimal thresh (Gavish-Donoho)={tau_star:.4f}")
    print(f"  [{label}] outliers above MP lambda+: {outliers.size}  |  "
          f"above optimal threshold: {outliers_thresh.size}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(bulk_pos, bins=100, density=True, alpha=0.5, label="empirical bulk")
    xs = np.linspace(max(lam_minus, 1e-6), lam_plus, 400)
    pdf = np.array([marchenko_pastur_pdf(x, beta, sigma_est) for x in xs])
    ax.plot(xs, pdf, "r-", lw=2, label=f"MP fit (sigma={sigma_est:.3g})")
    ax.axvline(lam_plus, color="k", ls="--", label=f"lambda+={lam_plus:.3g}")
    ax.axvline(tau_star, color="darkorange", ls=":", label=f"optimal thresh={tau_star:.3g}")
    for o in outlier_candidates[outlier_candidates > lam_plus][:15]:
        ax.axvline(o, color="crimson", alpha=0.3, lw=1)
    ax.set_title(f"MP fit -- {label}")
    ax.set_xlabel("eigenvalue"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / f"mp_{label}.png", dpi=150)
    plt.close(fig)

    return dict(label=label, N=N, M=M, beta=beta, sigma_hat=sigma_est,
                lambda_minus=lam_minus, lambda_plus=lam_plus,
                optimal_threshold=tau_star,
                n_outliers_lambda_plus=int(outliers.size),
                n_outliers_optimal_thresh=int(outliers_thresh.size),
                outliers_top10=sorted(outlier_candidates.tolist(), reverse=True)[:10])


def bulk_and_outlier_eigs(A, k_bulk, k_outlier):
    n = A.shape[0]
    k_bulk = max(3, min(k_bulk, n - 2))
    k_outlier = max(1, min(k_outlier, n - 2))

    try:
        bulk_vals = eigsh(A, k=k_bulk, sigma=0, which="LM", maxiter=5000)[0]
    except Exception as e:
        print(f"    shift-invert bulk fetch failed ({type(e).__name__}: {e}); "
              f"falling back to dense eigvalsh (small matrix only) or which='SM'")
        if n <= 3000:
            bulk_vals = np.linalg.eigvalsh(A.toarray())
        else:
            bulk_vals = eigsh(A, k=k_bulk, which="SM", maxiter=5000)[0]

    try:
        outlier_vals = eigsh(A, k=k_outlier, which="LA", maxiter=5000)[0]
    except Exception as e:
        print(f"    outlier fetch failed ({type(e).__name__}: {e}); using bulk max as fallback")
        outlier_vals = bulk_vals[bulk_vals > 0][-k_outlier:] if (bulk_vals > 0).any() else bulk_vals[-1:]

    return bulk_vals, outlier_vals


# ==============================================================================
# 1. LOAD MODULE + PROJECT global_ppi EDGES (NEW in v3)
# ==============================================================================
def project_global_ppi_edges(G, weight=0.5, global_ppi_label="global_ppi"):
    """
    For every real background-PPI edge global_ppi::A <-> global_ppi::B,
    add an intra-layer edge celltype::A <-> celltype::B in every real cell
    type where BOTH A and B are already present as celltype::GENE nodes.
    Skips pairs that already have a direct (measured) edge in that cell
    type -- measured evidence is never overwritten by projected evidence.

    Returns (G_with_projected_edges, n_edges_added, n_genes_rescued) where
    n_genes_rescued counts nodes that had ONLY a global_ppi identity edge
    (degree 1, all via global_ppi) and gained a real intra-layer edge from
    this projection -- i.e. genes that would have been dropped as isolated
    under the old plain-drop approach but are now legitimately retained.
    """
    gp_lower = global_ppi_label.lower()

    # gene -> set of REAL (non-global_ppi) cell types where it's present
    gene_celltypes = defaultdict(set)
    for n in G.nodes():
        ct = _celltype_of(n)
        if ct.lower() != gp_lower:
            gene_celltypes[_gene_of(n)].add(ct)

    # nodes that would be at risk of isolation once global_ppi is dropped
    # (degree 1, and that single edge goes to their own global_ppi copy)
    at_risk = set()
    for n in G.nodes():
        if _celltype_of(n).lower() == gp_lower:
            continue
        nbrs = list(G.neighbors(n))
        if len(nbrs) == 1 and _celltype_of(nbrs[0]).lower() == gp_lower:
            at_risk.add(n)

    new_edges = []
    n_pairs_seen = 0
    for u, v, attr in G.edges(data=True):
        if _celltype_of(u).lower() != gp_lower or _celltype_of(v).lower() != gp_lower:
            continue
        et = attr.get("edge_type", "ppi")
        if et not in ("ppi", "both"):
            continue
        n_pairs_seen += 1
        ga, gb = _gene_of(u), _gene_of(v)
        common_cts = gene_celltypes.get(ga, set()) & gene_celltypes.get(gb, set())
        for ct in common_cts:
            na, nb = f"{ct}::{ga}", f"{ct}::{gb}"
            if G.has_edge(na, nb):
                continue  # measured edge already exists -- don't overwrite
            new_edges.append((na, nb, {"edge_type": "ppi",
                                        "source": "background_ppi_projected",
                                        "weight": weight}))

    G.add_edges_from(new_edges)

    n_rescued = sum(1 for n in at_risk if G.degree(n) > 1)
    print(f"  [global_ppi projection] scanned {n_pairs_seen:,} background PPI edges -> "
          f"added {len(new_edges):,} projected intra-layer edges "
          f"({weight=}, source='background_ppi_projected')")
    print(f"  [global_ppi projection] {len(at_risk):,} nodes were at risk of isolation "
          f"(degree-1, only edge was to their own global_ppi copy); "
          f"{n_rescued:,} gained a real projected edge and will now be retained")

    return G, len(new_edges), n_rescued


def load_module_graph(module_graph_pkl, drop_global_ppi=True, project=True, project_weight=0.5):
    print(f"Loading disease module graph from {module_graph_pkl} ...")
    with open(module_graph_pkl, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip() for n in G.nodes()})
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    cts = pd.Series([_celltype_of(n) for n in G.nodes()]).value_counts()
    print("  Cell-type composition of module:")
    for ct, n in cts.items():
        print(f"    {ct:<30} {n:>6,} genes")

    has_global_ppi = "global_ppi" in {c.lower() for c in cts.index}

    if drop_global_ppi and has_global_ppi:
        if project:
            print("\n  Projecting global_ppi background-PPI edges onto co-present "
                  "cell-type layers before dropping global_ppi:: nodes ...")
            G, n_added, n_rescued = project_global_ppi_edges(G, weight=project_weight)
        else:
            n_added, n_rescued = 0, 0
            print("  --no-project passed: dropping global_ppi:: nodes WITHOUT "
                  "projecting their edges (old v2 behavior -- background-PPI-only "
                  "evidence will be lost).")

        gp_nodes = [n for n in G.nodes() if _celltype_of(n).lower() == "global_ppi"]
        print(f"  Dropping {len(gp_nodes):,} global_ppi:: nodes")
        G = G.subgraph([n for n in G.nodes() if n not in set(gp_nodes)]).copy()
        print(f"  After drop: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    elif has_global_ppi:
        print("  Keeping global_ppi:: nodes as their own pseudo-cell-type layer "
              "(note: this layer is not cell-type-specific and will likely "
              "dominate hub/bridge statistics -- interpret separately).")

    return G


def check_isolated_nodes(G):
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    if isolated:
        print(f"  {len(isolated)} nodes have zero degree in the module graph "
              f"after projection/drop -- these genuinely have no cell-type-context "
              f"edge (measured or background-PPI-projected) to lean on. Removing "
              f"them (they'd contribute spurious trivial eigenvalues otherwise).")
        G = G.subgraph([n for n in G.nodes() if n not in set(isolated)]).copy()
    else:
        print("  No zero-degree nodes remain -- good.")
    return G


def restrict_to_largest_component(G, min_report_size=1):
    """
    Restrict to the largest connected component of the FULL module graph
    (across all real cell-type layers + real inter-layer coupling edges --
    i.e. exactly the structure the supra-Laplacian will be built from).

    This uses nx.connected_components() -- an exact, authoritative answer --
    rather than counting near-zero eigenvalues from a small-k iterative
    solver (eigsh with k_eigs=6 or similar), which CANNOT reliably detect
    disconnection when the true number of components exceeds k: it will
    silently undercount, and the "Fiedler vector" picked afterward will not
    be the true Fiedler vector of your main component -- it'll be some
    ambiguous mixture influenced by however many small fragments happened
    to be missed. Always check the real component count this way first on
    graphs that might be fragmented; use the eigenvalue-based n_trivial
    check downstream only as a secondary sanity check on the (now
    guaranteed connected) LCC.
    """
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    n_components = len(components)
    lcc = components[0]
    n_dropped = G.number_of_nodes() - len(lcc)

    print(f"  True connected components (via nx.connected_components, not "
          f"eigenvalue counting): {n_components}")
    if n_components > 1:
        other_sizes = [len(c) for c in components[1:]]
        print(f"  Largest component: {len(lcc):,} / {G.number_of_nodes():,} nodes "
              f"({100*len(lcc)/G.number_of_nodes():.1f}%)")
        print(f"  Dropping {n_components - 1} smaller component(s), "
              f"{n_dropped:,} nodes total (sizes: "
              f"{sorted(other_sizes, reverse=True)[:20]}"
              f"{'...' if len(other_sizes) > 20 else ''})")
        if n_dropped / G.number_of_nodes() > 0.10:
            print(f"  WARNING: dropped fraction ({100*n_dropped/G.number_of_nodes():.1f}%) "
                  f"is large -- inspect the dropped genes before proceeding; this may "
                  f"indicate a real structural issue (e.g. an entire weakly-coupled "
                  f"cell type) rather than incidental small fragments.")
    else:
        print("  Graph is already fully connected -- nothing dropped.")

    G_lcc = G.subgraph(lcc).copy()
    return G_lcc, components


# ==============================================================================
# 2. CELL-TYPE LAYERS (unchanged from v2 -- projected edges just flow through
#    as normal edge_type="ppi" intra-layer edges, weight=0.5 by default)
# ==============================================================================
def build_celltype_layers(G):
    celltypes = sorted(set(_celltype_of(n) for n in G.nodes()))
    layer_nodes = {ct: sorted(n for n in G.nodes() if _celltype_of(n) == ct)
                   for ct in celltypes}
    layer_idx = {ct: {n: i for i, n in enumerate(nodes)}
                 for ct, nodes in layer_nodes.items()}

    rows = defaultdict(list); cols = defaultdict(list); data = defaultdict(list)
    n_cross_ppi_skipped = 0

    for u, v, attr in G.edges(data=True):
        et = attr.get("edge_type", "ppi")
        if et not in ("ppi", "both"):
            continue
        cu, cv = _celltype_of(u), _celltype_of(v)
        if cu != cv:
            n_cross_ppi_skipped += 1
            continue
        i, j = layer_idx[cu][u], layer_idx[cu][v]
        w = float(attr.get("weight", 1.0))
        rows[cu] += [i, j]; cols[cu] += [j, i]; data[cu] += [w, w]

    if n_cross_ppi_skipped:
        print(f"  NOTE: {n_cross_ppi_skipped} edge(s) tagged edge_type='ppi' but "
              f"crossed cell types -- skipped from intra-layer blocks.")

    layer_adj = {}
    for ct, nodes in layer_nodes.items():
        n_ct = len(nodes)
        if ct in rows:
            A = csr_matrix((data[ct], (rows[ct], cols[ct])), shape=(n_ct, n_ct))
            A.sum_duplicates()
        else:
            A = csr_matrix((n_ct, n_ct))
        layer_adj[ct] = A

    print("  Cell-type layers built:")
    for ct in celltypes:
        n_ct = len(layer_nodes[ct])
        e_ct = layer_adj[ct].nnz // 2
        isolated_in_layer = int((np.asarray(layer_adj[ct].sum(axis=1)).ravel() == 0).sum())
        print(f"    {ct:<30} {n_ct:>6,} nodes, {e_ct:>6,} intra-layer edges, "
              f"{isolated_in_layer:>5,} isolated within this layer "
              f"(may still be connected via inter-layer coupling)")

    return layer_adj, layer_nodes, layer_idx


# ==============================================================================
# 3. REAL INTER-LAYER COUPLING (unchanged from v2)
# ==============================================================================
def build_coupling_blocks(G, layer_idx, celltypes):
    pairs = defaultdict(lambda: ([], [], []))
    n_inter_edges = 0

    for u, v, attr in G.edges(data=True):
        et = attr.get("edge_type", "ppi")
        if et not in ("inter", "both"):
            continue
        cu, cv = _celltype_of(u), _celltype_of(v)
        if cu == cv:
            continue
        w = float(attr.get("weight", 1.0))
        a, b = (cu, cv) if cu < cv else (cv, cu)
        na, nb = (u, v) if cu < cv else (v, u)
        if na not in layer_idx[a] or nb not in layer_idx[b]:
            continue
        rows, cols, data = pairs[(a, b)]
        rows.append(layer_idx[a][na]); cols.append(layer_idx[b][nb]); data.append(w)
        n_inter_edges += 1

    coupling = {}
    for (a, b), (rows, cols, data) in pairs.items():
        n_a, n_b = len(layer_idx[a]), len(layer_idx[b])
        B = coo_matrix((data, (rows, cols)), shape=(n_a, n_b)).tocsr()
        B.sum_duplicates()
        coupling[(a, b)] = B

    print(f"  Real inter-layer coupling blocks: {len(coupling)} cell-type pairs, "
          f"{n_inter_edges:,} signaling edges total")
    for (a, b), B in sorted(coupling.items(), key=lambda kv: -kv[1].nnz)[:20]:
        print(f"    {a} <-> {b}: {B.nnz:,} edges")

    return coupling


# ==============================================================================
# 4. SUPRA-LAPLACIAN (unchanged from v2)
# ==============================================================================
def build_supra_laplacian(layer_adj, coupling, celltypes, omega=1.0):
    k = len(celltypes)
    sizes = [layer_adj[ct].shape[0] for ct in celltypes]
    offset = np.cumsum([0] + sizes)
    N = offset[-1]

    blocks = [[None] * k for _ in range(k)]
    for i, ct in enumerate(celltypes):
        blocks[i][i] = layer_adj[ct]

    for (a, b), B in coupling.items():
        i, j = celltypes.index(a), celltypes.index(b)
        blocks[i][j] = (omega * B).tocsr()
        blocks[j][i] = (omega * B.T).tocsr()

    A_total = bmat(blocks, format="csr")
    degrees = np.asarray(A_total.sum(axis=1)).ravel()
    L = diags(degrees) - A_total
    return L.tocsr(), celltypes, sizes, offset, N


def smallest_k_eigs(M, k, max_attempts=3):
    n = M.shape[0]
    k = min(k, n - 2)
    for attempt in range(max_attempts):
        try:
            vals, vecs = eigsh(-M, k=k, which="LA", maxiter=5000, tol=1e-6,
                                ncv=min(n - 1, 200))
            vals = -vals
            order = np.argsort(vals)
            return vals[order], vecs[:, order]
        except Exception as e:
            print(f"    eig attempt {attempt+1} failed ({type(e).__name__}: {e})")
            k = max(2, k // 2)
    vals, vecs = eigsh(-M, k=2, which="LA", maxiter=3000, tol=1e-4)
    vals = -vals
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def supra_fiedler(L_sup, k_eigs=5):
    print(f"  Computing {k_eigs} smallest eigenvalues of {L_sup.shape[0]}x{L_sup.shape[0]} "
          f"supra-Laplacian...")
    vals, vecs = smallest_k_eigs(L_sup, k=k_eigs)
    n_trivial = int(np.sum(np.abs(vals) < 1e-8))
    if n_trivial > 1:
        print(f"  WARNING: {n_trivial} near-zero eigenvalues found (expected exactly 1 "
              f"if the supra-graph is connected). Check for cell types with no "
              f"coupling to the rest of the module.")
    fiedler_idx = n_trivial if n_trivial >= 1 else 1
    fiedler_idx = min(fiedler_idx, vecs.shape[1] - 1)
    fiedler_vec = vecs[:, fiedler_idx]
    return vals, fiedler_vec, n_trivial


def split_fiedler_by_layer(fiedler_vec, celltypes, layer_nodes, offset):
    out = {}
    for i, ct in enumerate(celltypes):
        out[ct] = fiedler_vec[offset[i]:offset[i + 1]]
    return out


def determine_minority_sign(fiedler_vec):
    """
    A Fiedler eigenvector's sign is mathematically arbitrary -- eigsh could
    just as validly have returned the whole vector negated. What's
    structurally meaningful is never "negative" specifically, it's which
    side of the partition is the MINORITY. On a real run, this mattered a
    lot: the global split was 4,045 negative vs. 235 positive (94.5% /
    5.5%) -- "negative" was just the bulk of the module, and the
    structurally distinctive group was the small positive minority.
    Hardcoding "negative = bridge" (as earlier versions did) silently
    breaks whenever the true split is this imbalanced, and there's no way
    to know in advance which sign will end up being the minority on a
    given run/dataset -- so this is detected fresh every time, not assumed.
    """
    n_neg = int(np.sum(fiedler_vec < 0))
    n_pos = int(np.sum(fiedler_vec > 0))
    n_zero = int(np.sum(fiedler_vec == 0))
    total = len(fiedler_vec)
    print(f"  Global Fiedler sign split: {n_neg:,} negative ({100*n_neg/total:.1f}%), "
          f"{n_pos:,} positive ({100*n_pos/total:.1f}%)"
          + (f", {n_zero:,} exactly zero" if n_zero else ""))

    if n_neg == 0 or n_pos == 0:
        print("  WARNING: one sign is entirely empty -- minority-sign bridge detection "
              "is not meaningful here.")
        return -1

    minority_sign = -1 if n_neg < n_pos else 1
    minority_label = "negative" if minority_sign == -1 else "positive"
    minority_frac = min(n_neg, n_pos) / total
    print(f"  Minority sign: {minority_label} ({100*minority_frac:.1f}% of node-copies). "
          f"Multiplex bridges = genes consistently on the {minority_label} (minority) "
          f"side across their present cell types.")
    if minority_frac > 0.35:
        print(f"  NOTE: split is close to balanced ({100*minority_frac:.1f}% minority) -- "
              f"minority-side flagging is less distinctive here than in a heavily skewed "
              f"case; consider whether either-side interpretation makes more sense for "
              f"this run.")
    return minority_sign


# ==============================================================================
# 4.5 IN-PROCESS PERMUTATION TEST (NEW in v7)
# ==============================================================================
# Folded directly into the sweep (rather than requiring the separate
# permutation_test_bridges.py script to be run three times by hand
# afterward) so that compare_omega_sweep() can rank candidates by actual
# significance instead of raw frac_minority_layers. Raw fraction rewards
# small-n genes that get lucky (a gene present in only 3 cell types needs
# just 2/3 minority to hit 0.667, easily beating a gene present in 20 cell
# types that's minority in 5/20 -- even though 5/20 against a ~5.5% base
# rate is a far more extreme, permutation-significant result). This is the
# same column-permutation method as permutation_test_bridges.py, just
# operating directly on the in-memory fiedler_by_layer/layer_nodes rather
# than round-tripping through a saved CSV, and running automatically as
# part of every omega in a sweep.
def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def _permute_columns(m, rng):
    out = m.copy()
    for j in range(m.shape[1]):
        col = out[:, j]
        present_idx = np.where(~np.isnan(col))[0]
        if present_idx.size < 2:
            continue
        vals = col[present_idx].copy()
        rng.shuffle(vals)
        out[present_idx, j] = vals
    return out


def _frac_minority_per_row(m):
    with np.errstate(invalid="ignore"):
        n_present = np.sum(~np.isnan(m), axis=1)
        n_minority = np.nansum(m, axis=1)
    frac = np.divide(n_minority, n_present, out=np.full(n_present.shape, np.nan),
                      where=n_present > 0)
    return frac, n_present


def run_permutation_test(fiedler_by_layer, layer_nodes, celltypes, minority_sign,
                          min_celltypes_present, n_perm, seed):
    """
    Column-permutation significance test: shuffles minority/majority labels
    within each cell type independently (preserving each gene's presence
    pattern and each layer's marginal minority rate exactly, destroying
    cross-layer correlation), giving each gene an empirical p-value for
    how extreme its real minority-fraction is versus a matched-base-rate
    null. Returns a DataFrame with columns: gene, perm_pvalue,
    perm_qvalue_bh (Benjamini-Hochberg across testable genes), n_perm.
    """
    rows_by_gene = defaultdict(dict)
    for ct in celltypes:
        vec = fiedler_by_layer[ct]
        for node, val in zip(layer_nodes[ct], vec):
            rows_by_gene[_gene_of(node)][ct] = val

    genes = sorted(rows_by_gene.keys())
    n_genes, n_ct = len(genes), len(celltypes)
    values = np.full((n_genes, n_ct), np.nan)
    ct_index = {ct: i for i, ct in enumerate(celltypes)}
    for gi, gene in enumerate(genes):
        for ct, val in rows_by_gene[gene].items():
            values[gi, ct_index[ct]] = val

    minority = np.where(np.isnan(values), np.nan, ((values * minority_sign) > 0).astype(float))
    real_frac, n_present = _frac_minority_per_row(minority)
    testable = n_present >= min_celltypes_present

    rng = np.random.default_rng(seed)
    ge_count = np.zeros(n_genes, dtype=np.int64)
    for _ in range(n_perm):
        m_perm = _permute_columns(minority, rng)
        frac_perm, _ = _frac_minority_per_row(m_perm)
        with np.errstate(invalid="ignore"):
            ge = frac_perm >= real_frac
        ge_count += np.where(testable & ~np.isnan(frac_perm), ge, False).astype(np.int64)

    pvals = np.full(n_genes, np.nan)
    pvals[testable] = (1 + ge_count[testable]) / (1 + n_perm)
    qvals = np.full(n_genes, np.nan)
    if testable.sum() > 0:
        qvals[testable] = benjamini_hochberg(pvals[testable])

    return pd.DataFrame({"gene": genes, "perm_pvalue": pvals,
                          "perm_qvalue_bh": qvals, "perm_n_perm": n_perm})


# ==============================================================================
# 5. BRIDGE FLAGGING: MIN-PRESENCE + FRACTIONAL (NEW in v3)
# ==============================================================================
def multiplex_bridges(fiedler_by_layer, layer_nodes, celltypes, out_csv,
                       minority_sign, method="fractional", min_minority_layers=2,
                       min_celltypes_present=3, frac_minority_threshold=0.6):
    """
    Flags genes by MINORITY-sign consistency, not hardcoded negativity --
    see determine_minority_sign() for why. `minority_sign` is +1 or -1;
    a node-layer value "counts toward" bridge status if
    (value * minority_sign) > 0, i.e. it's on the minority side, whichever
    literal sign that happens to be on this run.

    method="absolute"    : n_minority_layers >= min_minority_layers.
                            Kept for comparison; NOT recommended as your
                            primary result (real data showed near-100% flag
                            rate for n>=3 under a fixed-count rule).
    method="fractional"  (default, recommended): a gene must (a) be present
                            in at least `min_celltypes_present` cell types,
                            AND (b) have >= frac_minority_threshold of ITS
                            PRESENT layers on the minority side. Genes
                            failing (a) are marked insufficient_data=True
                            and never flagged, regardless of fraction.
    """
    rows_by_gene = defaultdict(dict)
    for ct in celltypes:
        vec = fiedler_by_layer[ct]
        for node, val in zip(layer_nodes[ct], vec):
            rows_by_gene[_gene_of(node)][ct] = val

    records = []
    for gene, ct_vals in rows_by_gene.items():
        n_present = len(ct_vals)
        n_minority = sum(1 for v in ct_vals.values() if (v * minority_sign) > 0)
        frac_minority = n_minority / n_present if n_present else np.nan
        insufficient = n_present < min_celltypes_present

        if method == "absolute":
            is_bridge = n_minority >= min_minority_layers
        elif method == "fractional":
            is_bridge = (not insufficient) and (frac_minority >= frac_minority_threshold)
        else:
            raise ValueError(f"unknown method {method!r}")

        rec = {"gene": gene, "n_celltypes_present": n_present,
               "n_minority_layers": n_minority, "frac_minority_layers": frac_minority,
               "insufficient_data": insufficient,
               "fiedler_mean": float(np.mean(list(ct_vals.values()))),
               "fiedler_std": float(np.std(list(ct_vals.values())))}
        for ct in celltypes:
            rec[f"fiedler_{ct}"] = ct_vals.get(ct, np.nan)
        rec["is_multiplex_bridge"] = is_bridge
        records.append(rec)

    df = pd.DataFrame(records).sort_values("fiedler_std", ascending=False)
    df.to_csv(out_csv, index=False)

    n_total = len(df)
    n_insufficient = int(df["insufficient_data"].sum())
    n_bridges = int(df["is_multiplex_bridge"].sum())
    minority_label = "negative" if minority_sign == -1 else "positive"
    print(f"  Bridge flagging method: {method} (minority sign = {minority_label})")
    if method == "fractional":
        print(f"    min_celltypes_present={min_celltypes_present}, "
              f"frac_minority_threshold={frac_minority_threshold}")
        print(f"    {n_insufficient}/{n_total} genes excluded as insufficient_data "
              f"(present in <{min_celltypes_present} cell types)")
    print(f"    Multiplex bridges flagged: {n_bridges} / {n_total}")
    return df


# ==============================================================================
# 6. PERTURBATION (unchanged from v2)
# ==============================================================================
def drop_genes_from_module(G, bridge_genes):
    to_remove = [n for n in G.nodes() if _gene_of(n) in set(bridge_genes)]
    return G.subgraph([n for n in G.nodes() if n not in set(to_remove)]).copy(), to_remove


# ==============================================================================
# 7. PER-OMEGA ANALYSIS (NEW in v6 -- everything from here down depends on
#    omega, so it's factored out to run once per value in an --omega-sweep,
#    reusing the omega-independent work (graph loading, cell-type layers,
#    per-layer MP fits, coupling blocks) computed once in main().
# ==============================================================================
def run_omega_analysis(omega, G, layer_adj, layer_nodes, layer_idx, coupling,
                        celltypes, out, args):
    """Runs steps [5]-[7] (supra-Laplacian, Fiedler, minority-sign detection,
    bridge flagging, perturbation, summary) for one omega value, writing
    into `out` (a Path -- caller decides whether that's the top-level
    out-dir or an omega_<value>/ subdirectory). Returns the summary dict
    and the bridges DataFrame, both used by the sweep comparison in main()."""
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"[5] Supra-Laplacian (omega={omega})")
    print("=" * 70)
    L_sup, celltypes_ordered, sizes, offset, N = build_supra_laplacian(
        layer_adj, coupling, celltypes, omega=omega)
    print(f"  Supra-Laplacian shape: {N}x{N} across {len(celltypes_ordered)} cell types")

    sup_vals, fiedler_vec, n_trivial = supra_fiedler(L_sup, k_eigs=args.k_eigs)
    print(f"  Smallest eigenvalues: {np.round(sup_vals, 6)}")

    minority_sign = determine_minority_sign(fiedler_vec)

    fiedler_by_layer = split_fiedler_by_layer(fiedler_vec, celltypes_ordered, layer_nodes, offset)
    bridges_df = multiplex_bridges(
        fiedler_by_layer, layer_nodes, celltypes_ordered,
        out / "multiplex_bridges.csv",
        minority_sign=minority_sign,
        method=args.bridge_method,
        min_minority_layers=args.min_minority_layers,
        min_celltypes_present=args.min_celltypes_present,
        frac_minority_threshold=args.frac_minority_threshold,
    )

    if not args.skip_permutation_test:
        print(f"  Running in-process permutation test ({args.n_perm} permutations) "
              f"so candidates can be ranked by significance, not raw fraction ...")
        perm_df = run_permutation_test(
            fiedler_by_layer, layer_nodes, celltypes_ordered, minority_sign,
            args.min_celltypes_present, args.n_perm, args.perm_seed)
        bridges_df = bridges_df.merge(perm_df, on="gene", how="left")
        bridges_df.to_csv(out / "multiplex_bridges.csv", index=False)
        n_sig = int((bridges_df["perm_qvalue_bh"] <= args.fdr).sum())
        print(f"  Permutation test done: {n_sig} genes significant at FDR <= {args.fdr} "
              f"(out of {int((~bridges_df['insufficient_data']).sum())} testable)")
    else:
        print("  --skip-permutation-test passed: bridges will be ranked by raw "
              "frac_minority_layers only (NOT recommended for the sweep comparison -- "
              "see run_permutation_test() docstring for why raw fraction is biased "
              "toward low-presence genes).")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sup_vals, "o-")
    ax.axvline(n_trivial - 0.5, color="gray", ls="--", alpha=0.5,
               label=f"{n_trivial} trivial (zero) eigenvalue(s)")
    ax.set_title(f"Supra-Laplacian smallest eigenvalues (omega={omega})")
    ax.set_xlabel("index"); ax.set_ylabel("eigenvalue"); ax.legend()
    fig.tight_layout()
    fig.savefig(out / "supra_spectrum.png", dpi=150)
    plt.close(fig)

    print("\n" + "=" * 70)
    print(f"[6] Perturbation: dropping multiplex bridge genes (omega={omega})")
    print("=" * 70)
    bridge_genes = bridges_df.loc[bridges_df["is_multiplex_bridge"], "gene"].tolist()
    perturbed_summary = None
    if bridge_genes:
        G_pert, removed_nodes = drop_genes_from_module(G, bridge_genes)
        print(f"  Removing {len(bridge_genes)} bridge genes -> {len(removed_nodes)} node-copies "
              f"-> {G_pert.number_of_nodes():,} nodes remain")

        n_true_components_pert = len(list(nx.connected_components(G_pert)))
        print(f"  True connected components after removal: {n_true_components_pert}")
        if n_true_components_pert > 1:
            print(f"  Removing bridge genes fragmented the module -- restricting "
                  f"perturbed graph to its largest component too, for a fair "
                  f"eigenvalue comparison against the (also LCC-restricted) original.")
            G_pert, _ = restrict_to_largest_component(G_pert)

        layer_adj_p, layer_nodes_p, layer_idx_p = build_celltype_layers(G_pert)
        celltypes_p = sorted(layer_adj_p.keys())
        coupling_p = build_coupling_blocks(G_pert, layer_idx_p, celltypes_p)
        L_sup_p, _, sizes_p, offset_p, N_p = build_supra_laplacian(
            layer_adj_p, coupling_p, celltypes_p, omega=omega)
        sup_vals_p, _, n_trivial_p = supra_fiedler(L_sup_p, k_eigs=args.k_eigs)
        print(f"  Perturbed smallest eigenvalues: {np.round(sup_vals_p, 6)}")
        perturbed_summary = dict(
            n_nodes_before=G.number_of_nodes(), n_nodes_after=G_pert.number_of_nodes(),
            n_bridge_genes_removed=len(bridge_genes),
            eigenvalues_before=sup_vals.tolist(),
            eigenvalues_after=sup_vals_p.tolist(),
            n_trivial_before=n_trivial, n_trivial_after=n_trivial_p,
        )
    else:
        print("  No multiplex bridges found -- skipping.")

    print("\n" + "=" * 70)
    print(f"[7] Saving summary (omega={omega})")
    print("=" * 70)
    np.save(out / "supra_eigvals.npy", sup_vals)
    np.save(out / "fiedler_vector.npy", fiedler_vec)

    summary = dict(
        module_graph=str(args.module_graph),
        n_nodes_total=G.number_of_nodes(),
        celltypes=celltypes_ordered,
        celltype_sizes=dict(zip(celltypes_ordered, sizes)),
        omega=omega,
        bridge_method=args.bridge_method,
        minority_sign=int(minority_sign),
        minority_label=("negative" if minority_sign == -1 else "positive"),
        min_celltypes_present=args.min_celltypes_present,
        frac_minority_threshold=args.frac_minority_threshold,
        n_coupling_pairs=len(coupling),
        supra_smallest_eigvals=sup_vals.tolist(),
        n_trivial_eigenvalues=n_trivial,
        n_multiplex_bridges=int(bridges_df["is_multiplex_bridge"].sum()),
        n_permutation_significant=(int((bridges_df["perm_qvalue_bh"] <= args.fdr).sum())
                                    if "perm_qvalue_bh" in bridges_df.columns else None),
        fdr_threshold=args.fdr,
        n_perm=(args.n_perm if not args.skip_permutation_test else None),
        n_insufficient_data=int(bridges_df["insufficient_data"].sum()),
        perturbation=perturbed_summary,
        runtime_sec=time.time() - t0,
    )
    with open(out / "marchenko_pastur_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nDone (omega={omega}) in {summary['runtime_sec']:.1f}s -- outputs in {out}")
    return summary, bridges_df


# ==============================================================================
# 8. OMEGA-SWEEP COMPARISON (NEW in v6)
# ==============================================================================
def compare_omega_sweep(results_by_omega, out, top_n=30, fdr_threshold=0.05):
    """
    Builds a side-by-side comparison of the top-N candidate genes across
    every omega value in the sweep.

    RANKING (v7): if the in-process permutation test ran, ranks by
    perm_pvalue ascending (ties broken by perm_qvalue_bh, then
    frac_minority_layers) -- NOT raw frac_minority_layers. Raw fraction is
    biased toward genes present in only a few cell types (a 3-layer gene
    needs just 2/3 minority to score 0.667, trivially beating a 20-layer
    gene that's minority in 5/20=0.25 even though 5/20 against a small
    base rate is far more extreme/significant). Falls back to raw fraction
    only if --skip-permutation-test was passed for this run.

    Writes:
      omega_sweep_comparison.csv : one row per gene in ANY omega's top-N,
        with its rank and p-value per omega.
      omega_sweep_summary.txt : pairwise Jaccard overlap of top-N sets,
        the top-N "robust core", AND (new in v7, the more meaningful
        check) the ROBUST SIGNIFICANT CORE -- genes with
        perm_qvalue_bh <= fdr_threshold in EVERY omega tested. This is
        the direct answer to "does my one real hit stay significant
        across a reasonable coupling-strength range," not just "does it
        stay in an arbitrary top-N."
    """
    omegas = sorted(results_by_omega.keys())
    has_pvalues = all("perm_pvalue" in results_by_omega[om]["bridges_df"].columns
                       for om in omegas)

    top_sets, top_ranks, sig_sets = {}, {}, {}
    for om in omegas:
        df = results_by_omega[om]["bridges_df"]
        testable = df[~df["insufficient_data"]].copy()
        if has_pvalues:
            testable = testable.sort_values(
                ["perm_pvalue", "perm_qvalue_bh", "frac_minority_layers"],
                ascending=[True, True, False], na_position="last")
            sig_sets[om] = set(testable.loc[testable["perm_qvalue_bh"] <= fdr_threshold, "gene"])
        else:
            testable = testable.sort_values(["frac_minority_layers", "fiedler_std"],
                                             ascending=[False, False])
            sig_sets[om] = set()
        top = testable.head(top_n)
        top_sets[om] = set(top["gene"])
        top_ranks[om] = {gene: rank + 1 for rank, gene in enumerate(top["gene"])}

    all_genes = sorted(set().union(*top_sets.values())) if top_sets else []
    rows = []
    for gene in all_genes:
        row = {"gene": gene}
        for om in omegas:
            row[f"rank_omega_{om}"] = top_ranks[om].get(gene, None)
            if has_pvalues:
                df = results_by_omega[om]["bridges_df"]
                match = df.loc[df["gene"] == gene, "perm_pvalue"]
                row[f"pvalue_omega_{om}"] = float(match.iloc[0]) if len(match) else None
        row["n_omegas_in_top_n"] = sum(1 for om in omegas if gene in top_sets[om])
        rows.append(row)
    comparison_df = pd.DataFrame(rows).sort_values("n_omegas_in_top_n", ascending=False)
    comparison_csv = out / "omega_sweep_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    ranking_note = ("ranked by permutation p-value" if has_pvalues
                     else "ranked by raw frac_minority_layers -- "
                          "permutation test was skipped, see note below")
    lines = ["=" * 70, f"OMEGA SWEEP COMPARISON (top {top_n} genes, {ranking_note})",
              "=" * 70, f"Omega values tested: {omegas}", ""]

    lines.append("Pairwise Jaccard overlap of top-N sets:")
    for i, om_a in enumerate(omegas):
        for om_b in omegas[i + 1:]:
            a, b = top_sets[om_a], top_sets[om_b]
            jac = len(a & b) / len(a | b) if (a | b) else float("nan")
            lines.append(f"  omega={om_a} vs omega={om_b}: "
                          f"{len(a & b)}/{len(a | b)} shared -> Jaccard = {jac:.3f}")

    robust_core = set.intersection(*top_sets.values()) if top_sets else set()
    lines += ["", f"Top-{top_n} robust core (in every omega's top-{top_n}): "
                  f"{len(robust_core)} gene(s)"]
    if robust_core:
        lines.append(f"  {sorted(robust_core)}")

    if has_pvalues:
        robust_sig_core = set.intersection(*sig_sets.values()) if sig_sets else set()
        lines += ["", f"ROBUST SIGNIFICANT CORE (perm_qvalue_bh <= {fdr_threshold} in "
                       f"EVERY omega tested): {len(robust_sig_core)} gene(s)",
                  "  This is the direct answer to whether a hit's significance holds "
                  "across your omega range, not just whether it stays in an arbitrary "
                  "top-N."]
        if robust_sig_core:
            lines.append(f"  {sorted(robust_sig_core)}")
        else:
            lines.append("  (none -- no gene was significant at this FDR in every "
                          "omega tested. Check each omega's individual significant "
                          "set below; a gene significant in most-but-not-all omegas "
                          "may still be worth reporting with that caveat.)")
        for om in omegas:
            lines.append(f"    omega={om}: {len(sig_sets[om])} significant gene(s)"
                          + (f" -- {sorted(sig_sets[om])}" if sig_sets[om] else ""))
    else:
        lines += ["", "NOTE: permutation test was skipped (--skip-permutation-test), "
                       "so no significance-based robust core is available -- only the "
                       "raw-fraction top-N core above, which is biased toward "
                       "low-presence genes."]

    for om in omegas:
        n_bridges = results_by_omega[om]["summary"]["n_multiplex_bridges"]
        lines.append(f"\nomega={om}: {n_bridges} genes flagged as multiplex bridges "
                      f"(fractional threshold, full criteria) in this run.")

    text = "\n".join(lines)
    print("\n" + text)
    (out / "omega_sweep_summary.txt").write_text(text)
    print(f"\nSaved sweep comparison -> {comparison_csv}")
    print(f"Saved sweep summary -> {out / 'omega_sweep_summary.txt'}")
    return comparison_df


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module-graph", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--omega", type=float, default=1.0,
                     help="Single omega value. Ignored if --omega-sweep is given.")
    ap.add_argument("--omega-sweep", type=float, nargs="+", default=None,
                     help="Run the analysis once per omega value given here (e.g. "
                          "--omega-sweep 0.1 1.0 5.0), each into its own "
                          "out-dir/omega_<value>/ subfolder, then write an automated "
                          "comparison of the top candidate genes across all values to "
                          "out-dir/omega_sweep_comparison.csv and "
                          "omega_sweep_summary.txt. Graph loading, cell-type layers, "
                          "per-layer MP fits, and coupling blocks are computed ONCE and "
                          "reused (they don't depend on omega) -- only the supra-"
                          "Laplacian assembly and everything downstream of it reruns "
                          "per value.")
    ap.add_argument("--sweep-top-n", type=int, default=30,
                     help="Used only with --omega-sweep: how many top-ranked genes "
                          "(by frac_minority_layers) to compare across omega values.")
    ap.add_argument("--k-eigs", type=int, default=6)
    ap.add_argument("--k-bulk", type=int, default=200)
    ap.add_argument("--k-outlier", type=int, default=20)
    ap.add_argument("--keep-global-ppi", action="store_true",
                     help="Keep global_ppi:: as its own layer instead of "
                          "projecting+dropping (old, not-recommended behavior).")
    ap.add_argument("--no-lcc-restrict", action="store_true",
                     help="Skip restricting to the largest connected component "
                          "before spectral analysis. NOT recommended: if the module "
                          "graph is fragmented, the Fiedler vector computed on it "
                          "will not be a clean bipartition of a single connected "
                          "structure. Use only if you specifically want to analyze "
                          "the full, possibly-disconnected graph.")
    ap.add_argument("--no-project", action="store_true",
                     help="Drop global_ppi:: WITHOUT projecting its edges first "
                          "(old v2 behavior -- loses background-PPI-only evidence). "
                          "Ignored if --keep-global-ppi is set.")
    ap.add_argument("--project-weight", type=float, default=0.5,
                     help="Edge weight assigned to projected background-PPI edges "
                          "(default 0.5, vs. 1.0 for directly measured edges).")
    ap.add_argument("--bridge-method", choices=["fractional", "absolute"],
                     default="fractional")
    ap.add_argument("--min-minority-layers", type=int, default=2,
                     help="Used only if --bridge-method absolute.")
    ap.add_argument("--min-celltypes-present", type=int, default=3,
                     help="Used only if --bridge-method fractional. Genes present "
                          "in fewer cell types than this are excluded from bridge "
                          "calling (insufficient_data).")
    ap.add_argument("--frac-minority-threshold", type=float, default=0.6,
                     help="Used only if --bridge-method fractional. Fraction of a "
                          "gene's PRESENT layers that must be on the MINORITY "
                          "Fiedler-sign side (auto-detected each run -- see "
                          "determine_minority_sign()).")
    ap.add_argument("--n-perm", type=int, default=2000,
                     help="Number of permutations for the in-process significance "
                          "test (runs automatically after bridge flagging unless "
                          "--skip-permutation-test is passed). Increase (e.g. 10000) "
                          "to resolve p-values for genes that hit the resolution "
                          "floor (1/(1+n_perm)) -- i.e. beat every permutation.")
    ap.add_argument("--perm-seed", type=int, default=0)
    ap.add_argument("--fdr", type=float, default=0.05,
                     help="FDR (Benjamini-Hochberg q-value) threshold used to call "
                          "a gene 'significant' in per-run summaries and in the "
                          "omega-sweep robust-significant-core comparison.")
    ap.add_argument("--skip-permutation-test", action="store_true",
                     help="Skip the in-process permutation test (faster, but the "
                          "omega-sweep comparison will fall back to ranking by raw "
                          "frac_minority_layers, which is biased toward low-presence "
                          "genes -- see run_permutation_test() docstring).")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("[1] Loading disease module")
    print("=" * 70)
    G = load_module_graph(args.module_graph,
                           drop_global_ppi=not args.keep_global_ppi,
                           project=not args.no_project,
                           project_weight=args.project_weight)
    G = check_isolated_nodes(G)

    if not args.no_lcc_restrict:
        print("\n  Restricting to largest connected component before spectral analysis "
              "(pass --no-lcc-restrict to skip -- not recommended, see docstring) ...")
        G, all_components = restrict_to_largest_component(G)
        dropped_nodes_by_component = [sorted(c) for c in all_components[1:]]
        with open(out / "dropped_small_components.json", "w") as f:
            json.dump({"n_components_dropped": len(dropped_nodes_by_component),
                       "components": dropped_nodes_by_component}, f, indent=2)
        print(f"  Saved dropped-component node lists -> {out / 'dropped_small_components.json'}")
    else:
        n_true_components = len(list(nx.connected_components(G)))
        if n_true_components > 1:
            print(f"  WARNING: --no-lcc-restrict passed, but the graph has "
                  f"{n_true_components} true connected components. The Fiedler "
                  f"vector computed below will NOT be a clean bipartition of a "
                  f"single connected structure -- interpret with caution.")

    print("\n" + "=" * 70)
    print("[2] Building cell-type layers")
    print("=" * 70)
    layer_adj, layer_nodes, layer_idx = build_celltype_layers(G)
    celltypes = sorted(layer_adj.keys())

    print("\n" + "=" * 70)
    print("[3] Per-cell-type Marchenko-Pastur fits (omega-independent, computed once)")
    print("=" * 70)
    per_layer_mp = {}
    for ct in celltypes:
        A = layer_adj[ct]
        if A.shape[0] < 10 or A.nnz == 0:
            print(f"  [{ct}] too small/sparse -- skipping MP fit")
            continue
        bulk_vals, outlier_vals = bulk_and_outlier_eigs(A, args.k_bulk, args.k_outlier)
        per_layer_mp[ct] = fit_mp_gavish_donoho(bulk_vals, outlier_vals, f"layer_{ct}",
                                                  out, N=A.shape[0])

    print("\n" + "=" * 70)
    print("[4] Building real inter-layer coupling blocks (signaling edges; "
          "omega-independent, computed once)")
    print("=" * 70)
    coupling = build_coupling_blocks(G, layer_idx, celltypes)

    omega_values = args.omega_sweep if args.omega_sweep else [args.omega]
    is_sweep = len(omega_values) > 1

    results_by_omega = {}
    for om in omega_values:
        sub_out = (out / f"omega_{om}") if is_sweep else out
        summary, bridges_df = run_omega_analysis(
            om, G, layer_adj, layer_nodes, layer_idx, coupling, celltypes, sub_out, args)
        summary["per_layer_mp"] = {ct: v for ct, v in per_layer_mp.items() if v is not None}
        with open(sub_out / "marchenko_pastur_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        results_by_omega[om] = dict(summary=summary, bridges_df=bridges_df)

    if is_sweep:
        print("\n" + "=" * 70)
        print("[8] Omega sweep comparison")
        print("=" * 70)
        compare_omega_sweep(results_by_omega, out, top_n=args.sweep_top_n, fdr_threshold=args.fdr)
        print("\nKey outputs:")
        print(f"  {out / 'omega_<value>/multiplex_bridges.csv'}   -- per-omega results")
        print(f"  {out / 'omega_sweep_comparison.csv'}   -- top-N gene ranks across omegas")
        print(f"  {out / 'omega_sweep_summary.txt'}   -- Jaccard overlaps + robust core")
    else:
        print("\nKey outputs:")
        print(f"  {out / 'multiplex_bridges.csv'}   -- per-gene, layer-resolved bridge scores")
        print(f"  {out / 'marchenko_pastur_summary.json'}")
        print(f"  {out / 'mp_layer_<celltype>.png'}")
        print(f"  {out / 'supra_spectrum.png'}")


if __name__ == "__main__":
    main()