#!/usr/bin/env python
"""
step10b_functional_permutation_test_multiplex.py
(formerly functional_permutation_test_v2.py / counterfactual_functional.py)

Functional permutation test for disease module validation, adapted for the
current heterogeneous multiplex pipeline. Companion to
step10_permutation_test_multiplex.py (topological metrics) -- this one
scores modules by drug-target overlap, GO enrichment, and/or phenotype gene
overlap instead of network topology, with the same three-null design:

  Null 1: degree-matched random SEEDS -> RWR -> full module
  Null 2 (--run-rewired-null): TRUE seeds fixed, degree-preserving REWIRED
          network, RWR re-run on the scrambled graph
  Null 3 (--run-fixed-seed-null): TRUE seeds fixed, RWR-selected neighbors
          replaced with randomly/degree-matched sampled genes -- no RWR at
          all for the neighbor genes

Each null is also split (A) into seed-gene vs. RWR-neighbor-gene scores, to
tell you *where* any functional signal lives.

TWO FIXES vs. the version this replaces
------------------------------------------------------------------------------
1. SEED-SET MISMATCH WITH THE ACTUAL DISEASE MODULE. The previous version
   read seed_genes.csv directly as the true seed set -- but that's the
   *candidate* seed list, not necessarily what step6_build_disease_module.py
   actually anchored the module on. If step6 was run with
   --exclude-seed-celltypes / --exclude-ppi-seeds (e.g. to keep global_ppi
   genes available as bridging candidates without letting them anchor the
   module), some candidate seeds never anchored the true module at all, and
   testing this script's nulls against the full candidate list would compare
   the true module to a different, larger seed set than it was actually
   built from. Fixed by reading directly from disease_module.csv's
   `is_seed` AND NOT `is_excluded_seed` columns -- the same anchor set step6
   itself used -- instead of a separate seed file. (This is the identical
   fix applied to step10_permutation_test_multiplex.py; if you're running
   both, they'll now agree on what "the true seeds" means.)

2. ADAPTED FOR THE HETEROGENEOUS MULTIPLEX. The old version assumed one
   node per gene (Ensembl-ID-indexed dense transition matrix W, precomputed
   once via matrix inversion and cached to disk). The current pipeline's
   reference graph namespaces nodes as "<celltype>::GENE"
   (step2.5_namespace_and_seeds.py), so the same biological gene can be
   several distinct nodes -- one per cell type it's expressed in. This
   script now:
     - Loads gene_reference_graph.pkl (or gene_ppi_reference_graph.pkl)
       directly via --reference-graph, instead of graph_lcc.pkl /
       RW_matrix_restart_*.npy / scaling_matrix.npy / d_ensembl_idx.pkl --
       none of which the current pipeline produces.
     - Runs RWR as a sparse power iteration throughout (including for Null
       1, which the old version did via a precomputed dense W -- dense is
       not viable at real multiplex node counts; see
       step10_permutation_test_multiplex.py's docstring for the same
       reasoning applied there).
     - Keeps all network-level operations (RWR, connectivity padding,
       degree-preserving rewiring, degree-matched sampling) in the
       namespaced "<celltype>::GENE" node space, since that's where the
       actual graph topology lives.
     - Converts to GENE-LEVEL (bare symbol, deduplicated across cell types)
       only at the point of functional scoring -- drug-target/GO/phenotype
       annotations are inherently gene-level, not cell-type-node-level, so
       e.g. TRPV1 present in three cell-type layers of a module counts once,
       not three times, in a "fraction of module that's a drug target"
       score. A module gene is classified as "seed" (for the split test) if
       ANY of its namespaced occurrences in the module was a true anchoring
       seed, "neighbor" otherwise. See `to_gene_level()` /
       `compute_functional_metrics_split()` below.

PERFORMANCE PATCH (this version)
------------------------------------------------------------------------------
A. build_sparse_adjacency() now uses nx.to_scipy_sparse_array() instead of a
   manual Python for-loop over G.edges(). The old loop was pure Python and
   was being re-run from scratch on EVERY Null-2 permutation (once per
   rewired graph) -- at real multiplex edge counts that Python loop, not the
   RWR itself, was almost certainly the dominant cost of Null 2. The
   vectorized networkx builtin is implemented in C-adjacent numpy/scipy code
   and produces an identical CSR matrix.

B. Null 1, Null 2, and Null 3 permutation loops are now parallelized across
   worker processes with concurrent.futures.ProcessPoolExecutor, since every
   permutation within a null is fully independent of every other
   permutation in that null. A pool initializer ships G / A / nodelist /
   idx_of / degree_bins / scorers / RWR hyperparameters into each worker
   ONCE (at worker startup), not once per task, so per-task IPC overhead is
   just a handful of small ints/lists, not the whole graph. Use
   --n-workers to set the pool size (defaults to the number of CPUs SLURM
   gave the job, via $SLURM_CPUS_PER_TASK, falling back to os.cpu_count()).

   NOTE on scorers in worker processes: DrugTargetScorer / GOEnrichmentScorer
   / PainPhenotypeScorer lazy-load their reference data (parquet / CSV /
   network calls) on first use and cache it on `self`. Because each worker
   is a separate OS process with its own memory, each worker will do its
   own lazy-load the first time it scores a module (so the known_drug
   parquet gets read from disk --n-workers times total, not once). This is
   a one-time per-worker cost and is fine for --ot-data / --phenotype-csv.
   If you use GO enrichment (gprofiler, a network call) with many workers,
   be aware you may hit gprofiler rate limits faster; --no-go sidesteps
   this entirely, which is already the default recommendation for large
   permutation counts.

Usage
-----
python step10b_functional_permutation_test_multiplex.py \
    --reference-graph 04_multixrank_prep/multixrank_hetero_input/gene_reference_graph.pkl \
    --disease-module-dir 06_disease_module \
    --n-perms 1000 \
    --n-workers 16 \
    --ot-data ot_data \
    --disease-efos EFO_0010639,EFO_0005762,EFO_0010099 \
    --phenotype-csv phenotypes.csv \
    --run-rewired-null --n-perms-rewired 100 \
    --run-fixed-seed-null --n-perms-fixed-seed 1000 --fixed-seed-null-mode degree_matched
"""

import os, sys, argparse, pickle, time, warnings, re
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# ── Default pain EFO list (extracted may 13, 2026 by searching 'pain' here: https://www.ebi.ac.uk/ols4/ontologies/efo) ──────
DEFAULT_PAIN_EFOS = [
    'EFO_0010639','EFO_0801084','EFO_0005762','EFO_0009625','EFO_0007889',
    'EFO_0010099','EFO_0010100','EFO_0022195','EFO_0022861','EFO_0022862',
    'EFO_0022863','EFO_0022864','EFO_0022865','EFO_0022837','EFO_0008330',
    'EFO_0008316','EFO_0022960','EFO_0008338','EFO_0008340','EFO_0008339',
    'EFO_0010640','EFO_0007809','EFO_0010072','EFO_1001412','EFO_0011014',
    'EFO_0005404','EFO_0022850','EFO_0010263','EFO_1001833','EFO_0004149',
    'EFO_0920015','EFO_0009515','EFO_1001370','EFO_0009615','EFO_1001289',
    'EFO_1001776','EFO_1001258','EFO_1001785','EFO_1001404','EFO_1001366',
    'EFO_1002027','EFO_1001845','EFO_1001756','EFO_1001909','EFO_1000869',
    'EFO_1000254','EFO_1000542','EFO_0007299','EFO_1000446','EFO_0008544',
    'EFO_0020919','EFO_1001367','EFO_1001453','EFO_0008998','EFO_0003901',
    'EFO_0009748','EFO_0022849','EFO_1000621','EFO_0008622','EFO_0022851',
    'EFO_0021536','EFO_1000381','EFO_0002678','EFO_1001888','EFO_0003913',
    'EFO_0007520','EFO_1000243','EFO_0003266','EFO_0004278','EFO_0007638',
    'EFO_0700072','EFO_0010582','EFO_1001301','EFO_1000861',
]


# ──────────────────────────────────────────────────────────────────────────────
# Node-name parsing (nodes are "<celltype>::GENE")
# ──────────────────────────────────────────────────────────────────────────────

def _celltype_of(node: str) -> str:
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


def _gene_of(node: str) -> str:
    return node.rsplit("::", 1)[-1] if "::" in node else node


def to_gene_level(module_namespaced):
    """Unique bare gene symbols represented in a namespaced module --
    functional annotations (drug targets, GO terms, phenotype genes) are
    gene-level, not cell-type-node-level, so duplicate occurrences of the
    same gene across cell-type layers collapse to one entry before scoring.
    See module docstring point 2."""
    return sorted({_gene_of(n) for n in module_namespaced})


def filter_to_lcc_per_layer(H):
    """Identical to step6_build_disease_module.py's / step10's function of
    the same name. If step6 was run with --lcc-per-layer, it built the true
    module from THIS filtered graph -- so the null-sampling universe here
    has to be filtered the same way, or degree bins and random
    seeds/neighbors would be drawn from a different (larger, unfiltered)
    graph than the one the true module actually came from."""
    layers = set()
    for _, _, d in H.edges(data=True):
        layers |= d.get("cell_types", set())

    if not layers:
        print("  [lcc-per-layer] No 'cell_types' edge attribute found — skipping.")
        return H

    kept_nodes = set()
    for layer in sorted(layers):
        layer_edges = [(u, v) for u, v, d in H.edges(data=True)
                        if layer in d.get("cell_types", set())]
        layer_nodes = set(n for e in layer_edges for n in e)
        Lsub = H.subgraph(layer_nodes)
        if Lsub.number_of_nodes() == 0:
            print(f"  [lcc-per-layer] Cell type '{layer}': empty, skipping.")
            continue
        lcc = max(nx.connected_components(Lsub), key=len)
        print(f"  [lcc-per-layer] Cell type '{layer}': "
              f"{Lsub.number_of_nodes():,} -> {len(lcc):,} nodes "
              f"(dropped {Lsub.number_of_nodes() - len(lcc):,})")
        kept_nodes |= lcc

    H_filtered = H.subgraph(kept_nodes).copy()
    print(f"\n  [lcc-per-layer] Total: {H.number_of_nodes():,} -> "
          f"{H_filtered.number_of_nodes():,} nodes, "
          f"{H.number_of_edges():,} -> {H_filtered.number_of_edges():,} edges")
    return H_filtered


# ──────────────────────────────────────────────────────────────────────────────
# Degree-matched sampling (network-level; namespaced nodes throughout --
# these are already generic and need no multiplex-specific changes)
# ──────────────────────────────────────────────────────────────────────────────

def build_degree_bins(G, bin_size=5):
    bins = defaultdict(list)
    for node, deg in G.degree():
        bins[(deg // bin_size) * bin_size].append(node)
    return bins


def sample_degree_matched_seeds(true_seeds, G, degree_bins, rng, bin_size=5):
    sampled = set()
    for seed in true_seeds:
        deg = G.degree(seed)
        bin_key = (deg // bin_size) * bin_size
        candidates = [n for n in degree_bins[bin_key]
                      if n not in sampled and n not in true_seeds]
        if not candidates:
            for offset in [bin_size, -bin_size, 2*bin_size, -2*bin_size]:
                candidates = [n for n in degree_bins.get(bin_key + offset, [])
                              if n not in sampled and n not in true_seeds]
                if candidates:
                    break
        if candidates:
            sampled.add(rng.choice(candidates))
    return list(sampled)


def sample_degree_matched_genes(target_nodes, G, degree_bins, rng,
                                exclude=None, bin_size=5):
    """
    General-purpose degree-matched sampler: for each node in target_nodes,
    draw one *different* node from the same degree bin in G, excluding
    anything in `exclude`. Used by the fixed-seed / random-neighbor null to
    build a degree-matched replacement neighbor set. Operates in the same
    namespaced node space as everything else -- a replacement candidate for
    a "NeuronA::GENE1" neighbor is drawn from other nodes sharing its
    degree bin, which may or may not be in the same cell type; that's
    intentional (this null is about degree, not cell-type identity).
    """
    exclude = set(exclude) if exclude else set()
    sampled = set()
    for node in target_nodes:
        if node not in G:
            continue
        deg = G.degree(node)
        bin_key = (deg // bin_size) * bin_size
        candidates = [n for n in degree_bins.get(bin_key, [])
                      if n not in sampled and n not in exclude]
        if not candidates:
            for offset in [bin_size, -bin_size, 2*bin_size, -2*bin_size,
                           3*bin_size, -3*bin_size]:
                candidates = [n for n in degree_bins.get(bin_key + offset, [])
                              if n not in sampled and n not in exclude]
                if candidates:
                    break
        if candidates:
            sampled.add(rng.choice(candidates))
    return list(sampled)


# ──────────────────────────────────────────────────────────────────────────────
# Sparse power-iteration RWR (used for ALL three nulls now -- see module
# docstring point 2 for why the old dense-matrix path for Null 1 is gone)
# ──────────────────────────────────────────────────────────────────────────────

def build_sparse_adjacency(G, nodelist, idx_of):
    """
    PERFORMANCE PATCH: previously a manual Python for-loop over G.edges()
    appending to python lists before building a COO matrix. That loop runs
    once per Null-2 permutation (the network is rewired every permutation,
    so there's nothing to cache) and was almost certainly the dominant cost
    of Null 2 at real multiplex edge counts. nx.to_scipy_sparse_array is
    the vectorized networkx builtin for this exact conversion and produces
    an identical (symmetric, 0/1-weighted, undirected) CSR adjacency.
    `idx_of` is accepted for backwards-compatible call signatures but is no
    longer needed here since nodelist already fixes the row/col order.
    """
    return nx.to_scipy_sparse_array(G, nodelist=nodelist, format='csr',
                                    dtype=np.float64)


def rwr_power_iteration(A, seed_idx, restart=0.7, scaling=0,
                        max_iter=500, tol=1e-10):
    """
    Iteratively solves p = (1-restart) * P^T p + restart * p0 where P is
    the (optionally degree-scaled) row-normalized transition matrix built
    from adjacency A. Converges to the same fixed point as a closed-form
    matrix inverse without needing to invert an n x n matrix -- essential
    at real multiplex node counts (tens of thousands of genes x dozens of
    cell types).
    """
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).flatten()
    deg_safe = np.where(deg == 0, 1.0, deg)

    if scaling == 1:
        dinvsqrt = 1.0 / np.sqrt(deg_safe)
        Dinvsqrt_sp = sparse.diags(dinvsqrt)
        P = Dinvsqrt_sp @ A @ Dinvsqrt_sp
    else:
        dinv = 1.0 / deg_safe
        P = sparse.diags(dinv) @ A

    p0 = np.zeros(n)
    for idx in seed_idx:
        p0[idx] = np.sqrt(deg_safe[idx]) if scaling == 1 else 1.0
    total = p0.sum()
    if total == 0:
        return p0
    p0 = p0 / total

    p = p0.copy()
    Pt = P.transpose().tocsr()
    for _ in range(max_iter):
        p_new = (1 - restart) * (Pt @ p) + restart * p0
        if np.abs(p_new - p).sum() < tol:
            return p_new
        p = p_new
    return p


def build_module_from_score_vector(G, pinf, seeds_in_graph, nodelist, k=200):
    """Rank nodes by RWR score, take top-k non-seed nodes, then pad with
    further ranked nodes if needed until the module is connected within G."""
    seeds_set = set(seeds_in_graph)
    ranked_idx = np.argsort(pinf)[::-1]

    ranked_nodes = []
    for idx in ranked_idx:
        node = nodelist[idx]
        if node in seeds_set:
            continue
        ranked_nodes.append(node)

    module = list(seeds_set) + ranked_nodes[:k]
    module_set = set(module)
    extra_idx = k
    while extra_idx < len(ranked_nodes):
        sg = G.subgraph(module)
        if nx.is_connected(sg):
            break
        batch = ranked_nodes[extra_idx: extra_idx + 10]
        if not batch:
            break
        for g in batch:
            if g not in module_set:
                module.append(g)
                module_set.add(g)
        extra_idx += 10

    return module


def run_rwr_with_adjacency(G, A, nodelist, idx_of, seed_nodes, restart=0.7,
                           scaling=0, k=200, max_iter=500, tol=1e-10):
    """RWR against a PRECOMPUTED adjacency (Null 1: same graph every
    permutation, so building A once outside the loop avoids redundant work)."""
    seeds_in_graph = [g for g in seed_nodes if g in idx_of]
    if not seeds_in_graph:
        return None
    seed_idx = [idx_of[g] for g in seeds_in_graph]
    pinf = rwr_power_iteration(A, seed_idx, restart=restart, scaling=scaling,
                               max_iter=max_iter, tol=tol)
    return build_module_from_score_vector(G, pinf, seeds_in_graph, nodelist, k=k)


def run_rwr_rebuild_adjacency(G, seed_nodes, idx_of, nodelist, restart=0.7,
                              scaling=0, k=200, max_iter=500, tol=1e-10):
    """RWR that rebuilds the adjacency from G first (Null 2: the graph is
    freshly rewired every permutation, so there's nothing to precompute)."""
    seeds_in_graph = [g for g in seed_nodes if g in idx_of]
    if not seeds_in_graph:
        return None
    A = build_sparse_adjacency(G, nodelist, idx_of)
    return run_rwr_with_adjacency(G, A, nodelist, idx_of, seeds_in_graph,
                                  restart=restart, scaling=scaling, k=k,
                                  max_iter=max_iter, tol=tol)


def rewire_graph_degree_preserving(G, rng_seed, nswap_factor=10, max_tries_factor=100):
    """
    Returns a copy of G with edges randomly swapped via double_edge_swap,
    preserving the exact degree sequence but scrambling which nodes are
    connected to which. Node identities (namespaced "<celltype>::GENE"
    strings) are untouched -- only edges move.
    """
    Gc = G.copy()
    n_edges = Gc.number_of_edges()
    nswap = max(1, nswap_factor * n_edges // 10)
    max_tries = max_tries_factor * nswap
    try:
        nx.double_edge_swap(Gc, nswap=nswap, max_tries=max_tries, seed=rng_seed)
    except nx.NetworkXError:
        pass
    return Gc


# ──────────────────────────────────────────────────────────────────────────────
# Functional scorers (unchanged -- operate on GENE-LEVEL lists, see
# to_gene_level() above / compute_functional_metrics[_split]() below)
# ──────────────────────────────────────────────────────────────────────────────

class DrugTargetScorer:
    def __init__(self, ot_data_dir, disease_efos=None):
        self.ot_data_dir   = ot_data_dir
        self.disease_efos  = (
            [e.replace(':', '_') for e in disease_efos]
            if disease_efos else []
        )
        self._df           = None
        self._gene_col     = None
        self._efo_col      = None
        self._all_targets  = None
        self._dis_targets  = None

    def _load(self):
        if self._df is not None:
            return True
        path = os.path.join(self.ot_data_dir, "known_drug")
        if not os.path.exists(path):
            print(f"  [DrugTargetScorer] known_drug not found at {path} — skipping")
            return False
        try:
            import pyarrow.parquet as pq
            self._df = pq.read_table(path).to_pandas()
        except Exception as e:
            print(f"  [DrugTargetScorer] load error: {e}")
            return False

        self._gene_col = next(
            (c for c in ['targetSymbol','target_symbol','approved_symbol',
                         'approvedSymbol','target_gene_symbol','gene_symbol']
             if c in self._df.columns), None)
        if self._gene_col is None:
            print(f"  [DrugTargetScorer] cannot find gene column. "
                  f"Cols: {self._df.columns.tolist()}")
            return False

        self._efo_col = next(
            (c for c in ['diseaseId','disease_id','indication_id',
                         'efo_id','diseaseEFO','disease_efo']
             if c in self._df.columns), None)

        self._all_targets = set(
            self._df[self._gene_col].dropna().str.upper())
        print(f"  [DrugTargetScorer] {len(self._all_targets):,} drug-target "
              f"gene symbols (any indication)")

        if self.disease_efos and self._efo_col:
            pattern = '|'.join(
                re.escape(e) + r'|' + re.escape(e.replace('_', ':'))
                for e in self.disease_efos)
            mask = self._df[self._efo_col].astype(str).str.contains(
                pattern, case=False, na=False)
            self._dis_targets = set(
                self._df.loc[mask, self._gene_col].dropna().str.upper())
            print(f"  [DrugTargetScorer] {len(self._dis_targets):,} drug-target "
                  f"gene symbols across {len(self.disease_efos)} disease EFOs")
        elif self.disease_efos and self._efo_col is None:
            print("  [DrugTargetScorer] WARNING: no EFO column found in parquet; "
                  "disease-specific scoring will fall back to any-indication")
        return True

    def score_any(self, module):
        if not self._load() or not self._all_targets:
            return np.nan
        hits = sum(1 for g in module if str(g).upper() in self._all_targets)
        return hits / len(module) if module else 0.0

    def score_disease(self, module):
        if not self._load():
            return np.nan
        targets = self._dis_targets if self._dis_targets is not None else self._all_targets
        if not targets:
            return np.nan
        hits = sum(1 for g in module if str(g).upper() in targets)
        return hits / len(module) if module else 0.0

    def get_hit_genes(self, module, disease_specific=True):
        if not self._load():
            return []
        targets = (self._dis_targets if (disease_specific and self._dis_targets is not None)
                   else self._all_targets)
        return [g for g in module if str(g).upper() in (targets or set())]


class _AnyDrugScorer:
    def __init__(self, base): self._b = base
    def score(self, m): return self._b.score_any(m)

class _DiseaseDrugScorer:
    def __init__(self, base): self._b = base
    def score(self, m): return self._b.score_disease(m)


class GOEnrichmentScorer:
    MIN_GENES = 5

    def score(self, module):
        if len(module) < self.MIN_GENES:
            return 0.0
        try:
            from gprofiler import GProfiler
        except ImportError:
            return np.nan
        try:
            gp  = GProfiler(return_dataframe=True)
            res = gp.profile(organism='hsapiens', query=list(module),
                             sources=['GO:BP'], no_evidences=True)
            if res.empty:
                return 0.0
            return -np.log10(res['p_value'].min() + 1e-300)
        except Exception:
            return 0.0


class PainPhenotypeScorer:
    PAIN_KEYWORDS = ['pain','nociception','analgesia','hyperalgesia',
                     'allodynia','neuropathy','neuralgia']

    def __init__(self, phenotype_csv=None):
        self._pain_genes   = None
        self.phenotype_csv = phenotype_csv

    def _load(self):
        if self._pain_genes is not None:
            return True
        if not self.phenotype_csv or not os.path.exists(self.phenotype_csv):
            return False
        try:
            df = pd.read_csv(self.phenotype_csv, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            gene_col = next((c for c in ['Loci','Gene','GeneSymbol']
                             if c in df.columns), None)
            phen_col = next((c for c in ['Phenotype','Trait','Condition']
                             if c in df.columns), None)
            if not gene_col or not phen_col:
                print("  [PainPhenotypeScorer] cannot find gene/phenotype columns")
                return False
            df = df.dropna(subset=[gene_col, phen_col])
            df[phen_col] = df[phen_col].astype(str).str.lower()
            mask = df[phen_col].apply(
                lambda p: any(kw in p for kw in self.PAIN_KEYWORDS))
            self._pain_genes = set(df.loc[mask, gene_col].astype(str).str.strip())
            print(f"  [PainPhenotypeScorer] {len(self._pain_genes):,} pain-phenotype genes")
            return True
        except Exception as e:
            print(f"  [PainPhenotypeScorer] error: {e}")
            return False

    def score(self, module):
        if not self._load() or not self._pain_genes:
            return np.nan
        hits = sum(1 for g in module if str(g) in self._pain_genes)
        return hits / len(module) if module else 0.0


def compute_functional_metrics(module_namespaced, scorers):
    """module_namespaced: list of "<celltype>::GENE" node IDs. Converted to
    a deduplicated bare-gene list before scoring -- see to_gene_level()."""
    gene_level = to_gene_level(module_namespaced)
    return {name: scorer.score(gene_level) for name, scorer in scorers.items()}


def compute_functional_metrics_split(module_namespaced, seed_nodes_namespaced, scorers):
    """
    (A) Seed / neighbor decomposition, at GENE level. A bare gene counts as
    "seed" if ANY of its namespaced occurrences in the module was a true
    anchoring seed node, "neighbor" otherwise -- see module docstring
    point 2 for why classification has to happen after de-namespacing.
    """
    seed_genes = {_gene_of(n) for n in seed_nodes_namespaced}
    gene_level_module = to_gene_level(module_namespaced)
    seed_part  = [g for g in gene_level_module if g in seed_genes]
    neigh_part = [g for g in gene_level_module if g not in seed_genes]
    out = {}
    for name, scorer in scorers.items():
        out[f"{name}__seeds"]     = scorer.score(seed_part) if seed_part else np.nan
        out[f"{name}__neighbors"] = scorer.score(neigh_part) if neigh_part else np.nan
    return out


# ──────────────────────────────────────────────────────────────────────────────
# p-value / z-score summarisation (shared across all four null distributions)
# ──────────────────────────────────────────────────────────────────────────────

def summarise_null(true_metrics, null_metrics_list):
    pvalues, zscores = {}, {}
    for metric in true_metrics:
        null_vals = np.array([m[metric] for m in null_metrics_list
                              if not np.isnan(m.get(metric, np.nan))])
        true_val = true_metrics[metric]
        if len(null_vals) == 0 or np.isnan(true_val):
            pvalues[metric] = np.nan
            zscores[metric] = np.nan
            continue
        mu, sigma = null_vals.mean(), null_vals.std()
        zscores[metric] = (true_val - mu) / sigma if sigma > 0 else np.nan
        pvalues[metric] = float(np.mean(null_vals >= true_val))  # right-tail
    return pvalues, zscores


def results_table(true_metrics, null_metrics_list, pvalues, zscores):
    return pd.DataFrame({
        'metric':     list(true_metrics.keys()),
        'true_value': [true_metrics[m] for m in true_metrics],
        'null_mean':  [np.nanmean([x[m] for x in null_metrics_list])
                       for m in true_metrics],
        'null_std':   [np.nanstd( [x[m] for x in null_metrics_list])
                       for m in true_metrics],
        'z_score':    [zscores[m] for m in true_metrics],
        'p_value':    [pvalues[m] for m in true_metrics],
    })


def print_results_table(title, true_metrics, null_metrics_list, pvalues, zscores):
    print(f"\n{'='*90}")
    print(title)
    print(f"{'Metric':<50} {'True':>8} {'Null Mean':>10} {'Null SD':>9} "
          f"{'Z-score':>9} {'p-value':>9}")
    print(f"{'-'*90}")
    for metric in true_metrics:
        null_vals = [m[metric] for m in null_metrics_list
                     if not np.isnan(m.get(metric, np.nan))]
        tv = true_metrics[metric]
        if np.isnan(tv) or not null_vals:
            print(f"{metric:<50}  {'N/A':>8}")
            continue
        print(f"{metric:<50} {tv:>8.4f} "
              f"{np.mean(null_vals):>10.4f} {np.std(null_vals):>9.4f} "
              f"{zscores[metric]:>9.2f} {pvalues[metric]:>9.4f}")
    print(f"{'='*90}")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_null_distributions(true_metrics, null_metrics_list, pvalues, zscores,
                            out_path, filename, suptitle):
    valid = {k: v for k, v in true_metrics.items() if not np.isnan(v)}
    if not valid:
        print(f"  No valid metrics to plot for {filename}.")
        return
    n = len(valid)
    fig = plt.figure(figsize=(5 * n, 5))
    gs  = gridspec.GridSpec(1, n)
    for i, metric in enumerate(valid):
        ax        = fig.add_subplot(gs[i])
        null_vals = [m[metric] for m in null_metrics_list
                     if not np.isnan(m.get(metric, np.nan))]
        true_val  = valid[metric]
        ax.hist(null_vals, bins=40, color='steelblue', alpha=0.7,
                edgecolor='white', label='Null')
        ax.axvline(true_val, color='crimson', linewidth=2.5,
                   label=f'True\n(p={pvalues[metric]:.3f}, '
                         f'z={zscores[metric]:.2f})')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count' if i == 0 else '')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(out_path, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# PARALLEL WORKER INFRASTRUCTURE
#
# Every permutation within a given null is statistically independent of
# every other permutation in that null (a fresh random draw / a fresh
# rewiring, scored on its own) -- so we can hand permutations out to a pool
# of worker processes instead of running them one at a time.
#
# Design: a ProcessPoolExecutor initializer sets module-level globals in
# each worker ONCE, when the worker starts (G, A, nodelist, idx_of,
# degree_bins, scorers, RWR hyperparameters). Per-task arguments are then
# tiny (an int seed, a small list of seed-node names), so pickling cost per
# submitted task is negligible even though the graph itself is large.
# ──────────────────────────────────────────────────────────────────────────────

_W = {}  # per-worker-process global state, populated by _worker_init


def _worker_init(G, A, nodelist, idx_of, degree_bins, scorers,
                 restart, scaling, k, bin_size, rwr_max_iter, rwr_tol,
                 rewire_nswap_factor):
    _W['G'] = G
    _W['A'] = A
    _W['nodelist'] = nodelist
    _W['idx_of'] = idx_of
    _W['degree_bins'] = degree_bins
    _W['scorers'] = scorers
    _W['restart'] = restart
    _W['scaling'] = scaling
    _W['k'] = k
    _W['bin_size'] = bin_size
    _W['rwr_max_iter'] = rwr_max_iter
    _W['rwr_tol'] = rwr_tol
    _W['rewire_nswap_factor'] = rewire_nswap_factor


def _null1_task(seed_int, true_seeds_in_graph):
    """One Null-1 permutation: degree-matched random seeds -> RWR (reusing
    the precomputed, unchanging adjacency A) -> combined + split metrics."""
    rng = np.random.default_rng(seed_int)
    G, A, nodelist, idx_of = _W['G'], _W['A'], _W['nodelist'], _W['idx_of']
    degree_bins, scorers = _W['degree_bins'], _W['scorers']

    rand_seeds = sample_degree_matched_seeds(
        true_seeds_in_graph, G, degree_bins, rng, bin_size=_W['bin_size'])
    if len(rand_seeds) < max(1, len(true_seeds_in_graph) // 2):
        return None
    rand_module = run_rwr_with_adjacency(
        G, A, nodelist, idx_of, rand_seeds, restart=_W['restart'],
        scaling=_W['scaling'], k=_W['k'])
    if rand_module is None:
        return None
    combined = compute_functional_metrics(rand_module, scorers)
    split = compute_functional_metrics_split(rand_module, rand_seeds, scorers)
    return combined, split


def _null2_task(seed_int, true_seeds_in_graph):
    """One Null-2 permutation: fresh degree-preserving rewiring of the FULL
    graph, then RWR from the TRUE seeds on the scrambled graph. Each worker
    holds its own copy of the original (unrewired) G in _W['G'] -- rewiring
    happens on a local .copy() inside rewire_graph_degree_preserving, so
    concurrent tasks in different worker processes never interfere."""
    G = _W['G']
    idx_of, nodelist = _W['idx_of'], _W['nodelist']
    scorers = _W['scorers']

    G_rewired = rewire_graph_degree_preserving(
        G, seed_int, nswap_factor=_W['rewire_nswap_factor'])
    rand_module = run_rwr_rebuild_adjacency(
        G_rewired, true_seeds_in_graph, idx_of, nodelist,
        restart=_W['restart'], scaling=_W['scaling'], k=_W['k'],
        max_iter=_W['rwr_max_iter'], tol=_W['rwr_tol'])
    if rand_module is None:
        return None
    combined = compute_functional_metrics(rand_module, scorers)
    split = compute_functional_metrics_split(rand_module, true_seeds_in_graph, scorers)
    return combined, split


def _null3_task(seed_int, true_neighbor_nodes, seeds_list, mode):
    """One Null-3 permutation: TRUE seeds fixed, neighbors replaced by
    randomly/degree-matched sampled nodes -- no RWR at all."""
    rng = np.random.default_rng(seed_int)
    G = _W['G']
    degree_bins, scorers = _W['degree_bins'], _W['scorers']
    nodelist = _W['nodelist']
    seeds_set = set(seeds_list)
    n_neighbors_needed = len(true_neighbor_nodes)

    if mode == 'uniform':
        candidates = [n for n in nodelist if n not in seeds_set]
        if len(candidates) < n_neighbors_needed:
            return None
        sampled_neighbors = list(rng.choice(
            candidates, size=n_neighbors_needed, replace=False))
    else:  # degree_matched
        sampled_neighbors = sample_degree_matched_genes(
            true_neighbor_nodes, G, degree_bins, rng,
            exclude=seeds_set, bin_size=_W['bin_size'])
        if len(sampled_neighbors) < max(1, n_neighbors_needed // 2):
            return None

    rand_module = list(seeds_set) + sampled_neighbors
    combined = compute_functional_metrics(rand_module, scorers)
    split = compute_functional_metrics_split(rand_module, seeds_list, scorers)
    return combined, split


def _run_parallel_null(executor, task_fn, task_args_list, n_expected,
                       progress_every, label):
    """Submit every permutation's task_args to the pool, gather results as
    they complete, and print periodic progress. Returns (combined_list,
    split_list, n_failed)."""
    combined_list, split_list = [], []
    n_failed = 0
    n_done = 0
    t0 = time.time()

    futures = [executor.submit(task_fn, *args) for args in task_args_list]
    for fut in as_completed(futures):
        result = fut.result()
        n_done += 1
        if result is None:
            n_failed += 1
        else:
            combined, split = result
            combined_list.append(combined)
            split_list.append(split)
        if n_done % progress_every == 0 or n_done == n_expected:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_expected - n_done) / rate if rate > 0 else float('nan')
            print(f"  [{label}] {n_done}/{n_expected} | elapsed {elapsed:.0f}s "
                  f"| ETA {eta:.0f}s | {rate:.1f} perms/s")

    return combined_list, split_list, n_failed


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Functional permutation test (heterogeneous multiplex pipeline) with '
                    'seed/neighbor decomposition and rewired-network / fixed-seed nulls')
    parser.add_argument('--reference-graph', type=str, required=True,
        help='gene_reference_graph.pkl (or gene_ppi_reference_graph.pkl) from '
             'step2.5_namespace_and_seeds.py\'s output')
    parser.add_argument('--disease-module-dir', type=str, required=True,
        help='Directory containing disease_module.csv from '
             'step6_build_disease_module.py (used for true module/seed identity)')
    parser.add_argument('--restart',       type=float, default=0.7)
    parser.add_argument('--scaling',       type=int,   default=0, choices=[0,1])
    parser.add_argument('--k',             type=int,   default=200)
    parser.add_argument('--n-perms',       type=int,   default=1000,
        help='Permutations for the ORIGINAL degree-matched-random-seed null')
    parser.add_argument('--bin-size',      type=int,   default=5)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--output-dir',    type=str,   default=None,
        help='Output directory (default: <disease-module-dir>/functional_permutation_test)')
    parser.add_argument('--lcc-per-layer', action='store_true',
        help='Apply the same per-cell-type LCC filter to --reference-graph that '
             'step6_build_disease_module.py applied before building the true module. '
             'REQUIRED here if step6 was run with --lcc-per-layer.')
    parser.add_argument('--n-workers', type=int, default=None,
        help='Worker processes for parallelizing permutations within each null. '
             'Default: $SLURM_CPUS_PER_TASK if set, else os.cpu_count().')

    parser.add_argument('--ot-data',       type=str,   default=None)
    parser.add_argument('--disease-efos',  type=str,   default=None)
    parser.add_argument('--phenotype-csv', type=str,   default=None)
    parser.add_argument('--no-go',         action='store_true')

    # (B) rewired-network null
    parser.add_argument('--run-rewired-null', action='store_true',
        help='(B) Run degree-preserving rewired-network null (fixed true seeds, '
             'scrambled edges, RWR re-run per permutation via power iteration). '
             'Slower per-permutation than the other nulls; use fewer perms.')
    parser.add_argument('--n-perms-rewired', type=int, default=100)
    parser.add_argument('--rewire-nswap-factor', type=int, default=10,
        help='double_edge_swap nswap ~= factor * n_edges / 10 per permutation')
    parser.add_argument('--rwr-max-iter', type=int, default=500)
    parser.add_argument('--rwr-tol', type=float, default=1e-10)

    # (C) fixed-seed / random-neighbor null
    parser.add_argument('--run-fixed-seed-null', action='store_true',
        help='(C) Run fixed-true-seed, randomly-replaced-neighbor null '
             '(no RWR at all for the neighbor genes).')
    parser.add_argument('--n-perms-fixed-seed', type=int, default=1000)
    parser.add_argument('--fixed-seed-null-mode', type=str, default='degree_matched',
        choices=['degree_matched', 'uniform'])

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    n_workers = args.n_workers
    if n_workers is None:
        n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    n_workers = max(1, n_workers)

    dm_dir = Path(args.disease_module_dir)
    output_path = args.output_dir or str(dm_dir / 'functional_permutation_test')
    os.makedirs(output_path, exist_ok=True)

    if args.disease_efos:
        disease_efos = [e.strip() for e in args.disease_efos.split(',') if e.strip()]
    else:
        disease_efos = DEFAULT_PAIN_EFOS

    print(f"\n{'='*70}")
    print("Functional Permutation Test (heterogeneous multiplex) -- seed/neighbor "
          "decomposition + rewired-network + fixed-seed nulls")
    print(f"  Reference graph: {args.reference_graph}")
    print(f"  Disease module dir: {args.disease_module_dir}")
    print(f"  N permutations (seed null): {args.n_perms} | k: {args.k}")
    print(f"  Worker processes: {n_workers}")
    if args.ot_data:
        print(f"  Disease EFOs: {len(disease_efos)} IDs")
    print(f"{'='*70}\n")

    # ── Load full reference graph ───────────────────────────────────────────
    print("Loading full reference graph...")
    with open(args.reference_graph, 'rb') as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip() for n in G.nodes()})
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    if args.lcc_per_layer:
        print("\nApplying --lcc-per-layer filter (matching step6) ...")
        G = filter_to_lcc_per_layer(G)

    nodelist = sorted(G.nodes())
    idx_of = {n: i for i, n in enumerate(nodelist)}

    print("\nBuilding sparse adjacency (reused across Null 1 permutations)...")
    A = build_sparse_adjacency(G, nodelist, idx_of)
    print(f"  Adjacency: {A.shape[0]:,} x {A.shape[1]:,}, nnz={A.nnz:,}")

    # ── Load true disease module + true anchoring seeds (namespaced) ───────
    print("\nLoading true disease module...")
    module_csv_path = dm_dir / "disease_module_output/disease_module.csv"
    if not module_csv_path.exists():
        print(f"ERROR: {module_csv_path} not found")
        return 1
    module_df = pd.read_csv(module_csv_path)
    if "node" not in module_df.columns:
        print(f"ERROR: {module_csv_path} has no 'node' column -- this script requires "
              f"the namespaced disease_module.csv written by the current "
              f"step6_build_disease_module.py")
        return 1
    module_df["node"] = module_df["node"].astype(str).str.strip()

    true_module = module_df["node"].tolist()

    # Same fix as step10_permutation_test_multiplex.py: is_seed alone means
    # "was listed in the original seeds.txt", not "actually anchored the
    # module" -- is_excluded_seed marks genes step6 dropped from the anchor
    # set (e.g. via --exclude-seed-celltypes global_ppi) while keeping them
    # available as bridging candidates. Using is_seed alone here would test
    # against a different (larger) seed set than the true module was built
    # from. See module docstring point 1.
    if "is_excluded_seed" in module_df.columns:
        anchor_mask = module_df["is_seed"] & (~module_df["is_excluded_seed"].astype(bool))
        n_dropped = int((module_df["is_seed"] & module_df["is_excluded_seed"].astype(bool)).sum())
        if n_dropped:
            print(f"  {n_dropped} seed(s) marked is_excluded_seed=True -- excluded from "
                  f"the anchor set here too, matching step6's own behavior.")
    else:
        anchor_mask = module_df["is_seed"]
    true_seeds_in_graph = [g for g in module_df.loc[anchor_mask, "node"].tolist() if g in idx_of]

    print(f"True seeds in graph: {len(true_seeds_in_graph)}")
    print(f"True module size (namespaced nodes): {len(true_module)}")
    true_module_genes = to_gene_level(true_module)
    print(f"True module size (unique genes): {len(true_module_genes)}")
    n_true_neighbors = len(set(true_module) - set(true_seeds_in_graph))
    print(f"True RWR-added neighbor nodes: {n_true_neighbors}")

    # ── Build scorers ──────────────────────────────────────────────────────
    print("\nInitialising functional scorers...")
    scorers = {}
    if args.ot_data:
        dt = DrugTargetScorer(args.ot_data, disease_efos=disease_efos)
        scorers['drug_target_fraction_any']     = _AnyDrugScorer(dt)
        scorers['drug_target_fraction_disease'] = _DiseaseDrugScorer(dt)
        print(f"  ✓ DrugTargetScorer — any indication + {len(disease_efos)} disease EFOs")
    if not args.no_go:
        scorers['go_bp_best_neg_log10p'] = GOEnrichmentScorer()
        print("  ✓ GOEnrichmentScorer (gProfiler GO:BP)")
    else:
        print("  – GOEnrichmentScorer skipped (--no-go)")
    if args.phenotype_csv:
        scorers['pain_phenotype_fraction'] = PainPhenotypeScorer(args.phenotype_csv)
        print(f"  ✓ PainPhenotypeScorer ({args.phenotype_csv})")
    if not scorers:
        print("\nERROR: No functional scorers configured.")
        return 1

    # ── True metrics: combined + split ─────────────────────────────────────
    print("\nScoring true disease module...")
    true_metrics = compute_functional_metrics(true_module, scorers)
    true_metrics_split = compute_functional_metrics_split(
        true_module, true_seeds_in_graph, scorers)
    print("True module functional metrics (combined module, gene-level):")
    for k_m, v in true_metrics.items():
        print(f"  {k_m:45s}: {v:.6f}" if not np.isnan(v) else f"  {k_m:45s}: N/A")
    print("\nTrue module functional metrics (seed vs. neighbor decomposition, gene-level):")
    for k_m, v in true_metrics_split.items():
        print(f"  {k_m:45s}: {v:.6f}" if not np.isnan(v) else f"  {k_m:45s}: N/A")

    if args.ot_data:
        dt_scorer = scorers['drug_target_fraction_disease']._b
        hit_genes = dt_scorer.get_hit_genes(true_module_genes, disease_specific=True)
        print(f"\n  Disease-EFO drug-target genes in true module "
              f"({len(hit_genes)}): {hit_genes[:20]}"
              + (' ...' if len(hit_genes) > 20 else ''))
        pd.DataFrame({'gene': hit_genes}).to_csv(
            os.path.join(output_path, "true_module_drug_target_genes.csv"), index=False)

    degree_bins = build_degree_bins(G, bin_size=args.bin_size)

    # ── Fire up the worker pool once, reuse it across all three nulls ──────
    # The initializer pickles G / A / nodelist / idx_of / degree_bins /
    # scorers into each worker ONE TIME at pool startup, not per task.
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(G, A, nodelist, idx_of, degree_bins, scorers,
                  args.restart, args.scaling, args.k, args.bin_size,
                  args.rwr_max_iter, args.rwr_tol, args.rewire_nswap_factor),
    )

    try:
        # ════════════════════════════════════════════════════════════════
        # NULL 1 (original): degree-matched random SEEDS → RWR → full module
        # ════════════════════════════════════════════════════════════════
        print(f"\n[Null 1/3] Degree-matched random seeds — {args.n_perms} "
              f"permutations across {n_workers} workers...")
        seed_ints = rng.integers(0, 2**31 - 1, size=args.n_perms)
        task_args = [(int(s), true_seeds_in_graph) for s in seed_ints]
        null1_combined, null1_split, failed = _run_parallel_null(
            executor, _null1_task, task_args, args.n_perms,
            progress_every=max(1, args.n_perms // 10), label="Null 1")
        print(f"  Completed {len(null1_combined)} valid permutations ({failed} failed/skipped)")

        pvals1, zs1 = summarise_null(true_metrics, null1_combined)
        pvals1_split, zs1_split = summarise_null(true_metrics_split, null1_split)
        print_results_table(
            "NULL 1: Degree-matched random SEEDS (combined module vs. combined module)",
            true_metrics, null1_combined, pvals1, zs1)
        print_results_table(
            "NULL 1: Degree-matched random SEEDS (seed/neighbor decomposition)",
            true_metrics_split, null1_split, pvals1_split, zs1_split)

        results_table(true_metrics, null1_combined, pvals1, zs1).to_csv(
            os.path.join(output_path, "null1_degree_matched_seeds_combined.csv"), index=False)
        results_table(true_metrics_split, null1_split, pvals1_split, zs1_split).to_csv(
            os.path.join(output_path, "null1_degree_matched_seeds_split.csv"), index=False)
        plot_null_distributions(true_metrics, null1_combined, pvals1, zs1, output_path,
            "null1_combined_distributions.png",
            "Null 1: Degree-matched random seeds (combined module)")
        plot_null_distributions(true_metrics_split, null1_split, pvals1_split, zs1_split,
            output_path, "null1_split_distributions.png",
            "Null 1: Degree-matched random seeds (seed vs. neighbor decomposition)")

        # ════════════════════════════════════════════════════════════════
        # NULL 2 (B): TRUE seeds fixed, degree-preserving REWIRED network,
        # RWR re-run from true seeds on the scrambled graph (power iteration)
        # ════════════════════════════════════════════════════════════════
        if args.run_rewired_null:
            print(f"\n[Null 2/3] Degree-preserving rewired network, true seeds fixed — "
                  f"{args.n_perms_rewired} permutations across {n_workers} workers...")
            seed_ints2 = rng.integers(0, 2**31 - 1, size=args.n_perms_rewired)
            task_args2 = [(int(s), true_seeds_in_graph) for s in seed_ints2]
            null2_combined, null2_split, failed2 = _run_parallel_null(
                executor, _null2_task, task_args2, args.n_perms_rewired,
                progress_every=max(1, args.n_perms_rewired // 10), label="Null 2")
            print(f"  Completed {len(null2_combined)} valid permutations ({failed2} failed/skipped)")

            pvals2, zs2 = summarise_null(true_metrics, null2_combined)
            pvals2_split, zs2_split = summarise_null(true_metrics_split, null2_split)
            print_results_table(
                "NULL 2: Degree-preserving rewired network, TRUE seeds fixed (combined)",
                true_metrics, null2_combined, pvals2, zs2)
            print_results_table(
                "NULL 2: Degree-preserving rewired network, TRUE seeds fixed (seed/neighbor)",
                true_metrics_split, null2_split, pvals2_split, zs2_split)

            results_table(true_metrics, null2_combined, pvals2, zs2).to_csv(
                os.path.join(output_path, "null2_rewired_network_combined.csv"), index=False)
            results_table(true_metrics_split, null2_split, pvals2_split, zs2_split).to_csv(
                os.path.join(output_path, "null2_rewired_network_split.csv"), index=False)
            plot_null_distributions(true_metrics, null2_combined, pvals2, zs2, output_path,
                "null2_combined_distributions.png",
                "Null 2: Rewired network, true seeds fixed (combined module)")
            plot_null_distributions(true_metrics_split, null2_split, pvals2_split, zs2_split,
                output_path, "null2_split_distributions.png",
                "Null 2: Rewired network, true seeds fixed (seed vs. neighbor decomposition)")
        else:
            print("\n[Null 2/3] Skipped (pass --run-rewired-null to enable)")

        # ════════════════════════════════════════════════════════════════
        # NULL 3 (C): TRUE seeds fixed, neighbors replaced with randomly
        # sampled nodes (uniform or degree-matched) — no RWR/topology at all
        # ════════════════════════════════════════════════════════════════
        if args.run_fixed_seed_null:
            print(f"\n[Null 3/3] Fixed true seeds, {args.fixed_seed_null_mode} random "
                  f"neighbors — {args.n_perms_fixed_seed} permutations across "
                  f"{n_workers} workers...")
            true_neighbor_nodes = [g for g in true_module if g not in set(true_seeds_in_graph)]

            seed_ints3 = rng.integers(0, 2**31 - 1, size=args.n_perms_fixed_seed)
            task_args3 = [(int(s), true_neighbor_nodes, true_seeds_in_graph,
                          args.fixed_seed_null_mode) for s in seed_ints3]
            null3_combined, null3_split, failed3 = _run_parallel_null(
                executor, _null3_task, task_args3, args.n_perms_fixed_seed,
                progress_every=max(1, args.n_perms_fixed_seed // 10), label="Null 3")
            print(f"  Completed {len(null3_combined)} valid permutations ({failed3} failed/skipped)")

            pvals3, zs3 = summarise_null(true_metrics, null3_combined)
            pvals3_split, zs3_split = summarise_null(true_metrics_split, null3_split)
            print_results_table(
                f"NULL 3: TRUE seeds fixed, {args.fixed_seed_null_mode} random neighbors "
                f"(combined module)",
                true_metrics, null3_combined, pvals3, zs3)
            print_results_table(
                f"NULL 3: TRUE seeds fixed, {args.fixed_seed_null_mode} random neighbors "
                f"(seed/neighbor decomposition)",
                true_metrics_split, null3_split, pvals3_split, zs3_split)

            results_table(true_metrics, null3_combined, pvals3, zs3).to_csv(
                os.path.join(output_path, "null3_fixed_seed_random_neighbors_combined.csv"),
                index=False)
            results_table(true_metrics_split, null3_split, pvals3_split, zs3_split).to_csv(
                os.path.join(output_path, "null3_fixed_seed_random_neighbors_split.csv"),
                index=False)
            plot_null_distributions(true_metrics, null3_combined, pvals3, zs3, output_path,
                "null3_combined_distributions.png",
                f"Null 3: True seeds fixed, {args.fixed_seed_null_mode} random neighbors "
                f"(combined module)")
            plot_null_distributions(true_metrics_split, null3_split, pvals3_split, zs3_split,
                output_path, "null3_split_distributions.png",
                f"Null 3: True seeds fixed, {args.fixed_seed_null_mode} random neighbors "
                f"(seed vs. neighbor decomposition)")
        else:
            print("\n[Null 3/3] Skipped (pass --run-fixed-seed-null to enable)")
    finally:
        executor.shutdown(wait=True)

    # Save the EFO list used
    pd.DataFrame({'efo_id': disease_efos}).to_csv(
        os.path.join(output_path, "disease_efos_used.csv"), index=False)

    print(f"\nAll outputs saved to: {output_path}")
    print("\nHow to read the three nulls together:")
    print("  Null 1 (__neighbors rows) significant AND Null 2 significant AND")
    print("  Null 3 significant  →  strong evidence the RWR-selected neighbors carry")
    print("  real, network-topology-dependent functional signal, not just seed bias.")
    print("  Null 1 significant but Null 2/3 NOT significant  →  signal is mostly")
    print("  attributable to seed-gene identity, not to the network expansion step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())