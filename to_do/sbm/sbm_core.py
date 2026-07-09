#!/usr/bin/env python3
"""
sbm_core.py — Multiplex Stochastic Block Model with Variational Inference
============================================================================

Implements a multiplex SBM where each edge type (layer) has its own block
probability matrix, but all layers share a single community assignment vector.

The model:
    - z_i ~ Categorical(π)              (community assignment for node i)
    - A^l_{ij} | z_i=k, z_j=k' ~ Bernoulli(B^l_{kk'})   for each layer l

Variational inference (mean-field EM):
    - E-step: update variational posteriors q(z_i) for each node
    - M-step: update π and B^l given current q(z)

Optionally incorporates nodal metadata (e.g. cell_type) as a prior on π
following the approach in "Stochastic blockmodels with a growing number of
classes" (ICML 2012).

Memory-efficient: uses sparse adjacency matrices (scipy.sparse) per layer.
Parallelisation: E-step updates can be distributed across cores.

Author: Abacus AI Agent
"""

import gc
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import digamma, logsumexp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MultiplexData:
    """Pre-processed multiplex network in sparse format."""
    n_nodes: int                           # N
    n_layers: int                          # L (one per edge type)
    layer_names: list                      # names of each layer
    adj_matrices: list                     # list of N×N sparse CSR matrices
    node_ids: np.ndarray                   # original integer node IDs
    node_attrs: pd.DataFrame               # node_id, gene, source, cell_type
    metadata_labels: Optional[np.ndarray] = None  # integer-encoded metadata per node
    metadata_names: Optional[list] = None  # string labels for metadata classes


@dataclass
class SBMParams:
    """Parameters of the fitted multiplex SBM."""
    K: int                                 # number of communities
    pi: np.ndarray                         # (K,) mixing proportions
    B: list                                # list of L (K×K) block probability matrices
    tau: np.ndarray                        # (N, K) variational posteriors q(z_i=k)
    community_assignments: np.ndarray      # (N,) hard assignments argmax tau
    log_likelihood: float = 0.0
    n_params: int = 0
    converged: bool = False
    n_iterations: int = 0
    elbo_history: list = field(default_factory=list)
    metadata_weight: float = 0.0           # weight of metadata prior (η)
    metadata_B: Optional[np.ndarray] = None  # (K, M) metadata emission matrix


# ---------------------------------------------------------------------------
# Loading / pre-processing
# ---------------------------------------------------------------------------

def load_multiplex_data(
    graph_path: str,
    registry_path: str,
    edge_types: tuple = ("ppi", "pathway", "ligand_receptor"),
    use_metadata: bool = False,
    metadata_col: str = "cell_type",
) -> MultiplexData:
    """
    Load integrated multiplex graph and convert to sparse adjacency matrices.

    Parameters
    ----------
    graph_path : str
        Path to integrated_multiplex.pkl (NetworkX Graph).
    registry_path : str
        Path to node_registry.csv.
    edge_types : tuple of str
        Edge types to include as separate layers. Each becomes one adjacency
        matrix. All edge types are treated equally in the model.
    use_metadata : bool
        Whether to extract node metadata for prior incorporation.
    metadata_col : str
        Column in node_registry.csv to use as metadata (default: cell_type).

    Returns
    -------
    MultiplexData
    """
    logger.info("Loading multiplex graph from %s ...", graph_path)
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    logger.info("  Loaded graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    logger.info("Loading node registry from %s ...", registry_path)
    registry = pd.read_csv(registry_path)

    # Build node ID mapping: original int IDs → contiguous 0..N-1
    all_nodes = sorted(G.nodes())
    n_nodes = len(all_nodes)
    node_to_idx = {nid: i for i, nid in enumerate(all_nodes)}
    node_ids = np.array(all_nodes, dtype=np.int64)

    # Build node attributes DataFrame aligned with contiguous indices
    node_attrs_rows = []
    for nid in all_nodes:
        d = G.nodes[nid]
        node_attrs_rows.append({
            "node_id": nid,
            "gene": d.get("gene", ""),
            "source": d.get("source", ""),
            "cell_type": d.get("cell_type", ""),
        })
    node_attrs = pd.DataFrame(node_attrs_rows)

    # Build sparse adjacency matrices per edge type
    adj_matrices = []
    layer_names = []
    for etype in edge_types:
        rows, cols = [], []
        for u, v, edata in G.edges(data=True):
            if edata.get("edge_type") == etype:
                i, j = node_to_idx[u], node_to_idx[v]
                rows.extend([i, j])  # symmetric
                cols.extend([j, i])
        data = np.ones(len(rows), dtype=np.float32)
        A = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        # Ensure binary (no duplicates)
        A.data[:] = 1.0
        A.eliminate_zeros()
        adj_matrices.append(A)
        layer_names.append(etype)
        nnz = A.nnz // 2  # undirected
        logger.info("  Layer '%s': %d edges (sparse matrix: %.1f MB)",
                     etype, nnz, A.data.nbytes / 1e6)

    # Free original graph
    del G
    gc.collect()

    # Metadata
    metadata_labels = None
    metadata_names = None
    if use_metadata and metadata_col in node_attrs.columns:
        unique_vals = sorted(node_attrs[metadata_col].dropna().unique())
        val_to_int = {v: i for i, v in enumerate(unique_vals)}
        metadata_labels = node_attrs[metadata_col].map(val_to_int).fillna(-1).astype(int).values
        metadata_names = unique_vals
        logger.info("  Metadata '%s': %d classes, %d nodes with labels",
                     metadata_col, len(unique_vals),
                     np.sum(metadata_labels >= 0))

    return MultiplexData(
        n_nodes=n_nodes,
        n_layers=len(adj_matrices),
        layer_names=layer_names,
        adj_matrices=adj_matrices,
        node_ids=node_ids,
        node_attrs=node_attrs,
        metadata_labels=metadata_labels,
        metadata_names=metadata_names,
    )


# ---------------------------------------------------------------------------
# Core variational EM
# ---------------------------------------------------------------------------

def _init_tau(N: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Initialise variational posteriors q(z_i) with random Dirichlet draws."""
    tau = rng.dirichlet(np.ones(K), size=N).astype(np.float64)
    # Small floor to avoid log(0)
    tau = np.clip(tau, 1e-10, None)
    tau /= tau.sum(axis=1, keepdims=True)
    return tau


def _m_step_pi(tau: np.ndarray) -> np.ndarray:
    """Update mixing proportions: π_k = (1/N) Σ_i τ_{ik}."""
    pi = tau.mean(axis=0)
    pi = np.clip(pi, 1e-10, None)
    pi /= pi.sum()
    return pi


def _m_step_B(tau: np.ndarray, A: sparse.csr_matrix, K: int,
              eps: float = 1e-10) -> np.ndarray:
    """
    Update block probability matrix for one layer.

    B_{kk'} = Σ_{i<j} τ_ik τ_jk' A_ij  /  Σ_{i<j} τ_ik τ_jk'

    Efficiently computed as:
        numerator   = tau^T @ A @ tau
        denominator = tau^T @ (ones - I) @ tau  (approximate: tau_sum_k * tau_sum_k' - diag)
    """
    # tau^T A tau gives (K, K) matrix of expected edge counts
    # Using sparse multiplication for efficiency
    tA = A.dot(tau)  # (N, K)
    numer = tau.T @ tA  # (K, K)

    # Expected number of possible edges between communities
    tau_sum = tau.sum(axis=0)  # (K,)
    denom = np.outer(tau_sum, tau_sum)  # (K, K)
    # Subtract self-pairs on diagonal
    diag_corr = (tau ** 2).sum(axis=0)  # (K,)
    np.fill_diagonal(denom, denom.diagonal() - diag_corr)

    # For off-diagonal, each pair counted twice in symmetric A
    # A is already symmetric, so numer counts each edge twice for off-diag
    # Correct: B = numer / denom (both count pairs symmetrically)
    B = numer / np.maximum(denom, eps)
    B = np.clip(B, eps, 1.0 - eps)

    return B


def _e_step_chunk(
    tau: np.ndarray,
    adj_matrices: list,
    B_list: list,
    log_pi: np.ndarray,
    node_range: tuple,
    metadata_labels: Optional[np.ndarray],
    log_meta_B: Optional[np.ndarray],
    meta_weight: float,
) -> np.ndarray:
    """
    E-step for a chunk of nodes [start, end).
    Updates log q(z_i=k) ∝ log π_k + Σ_l Σ_j [A^l_ij log B^l_kk' τ_jk'
                                                 + (1-A^l_ij) log(1-B^l_kk') τ_jk']
    For efficiency, we precompute sums over j for each community k'.
    """
    start, end = node_range
    N, K = tau.shape
    L = len(adj_matrices)

    log_tau_chunk = np.zeros((end - start, K), dtype=np.float64)
    log_tau_chunk[:] = log_pi[np.newaxis, :]

    for l_idx in range(L):
        A = adj_matrices[l_idx]
        B = B_list[l_idx]
        logB = np.log(np.clip(B, 1e-300, None))
        log1mB = np.log(np.clip(1.0 - B, 1e-300, None))

        # For each node i in chunk, compute contribution from layer l:
        # Σ_j τ_jk' [A_ij log B_kk' + (1-A_ij) log(1-B_kk')]
        # = Σ_j A_ij τ_jk' [log B_kk' - log(1-B_kk')] + Σ_j τ_jk' log(1-B_kk')
        # = (A[i,:] @ tau) @ (logB - log1mB)^T  + tau_sum @ log1mB^T

        # Precompute tau_sum per community
        tau_sum = tau.sum(axis=0)  # (K,)

        # Sparse slice for the chunk
        A_chunk = A[start:end, :]  # (chunk_size, N) sparse

        # A_chunk @ tau → (chunk_size, K): weighted neighbor sum
        At = A_chunk.dot(tau)  # (chunk_size, K)

        # log-odds ratio
        log_odds = logB - log1mB  # (K, K)

        # For each k, contribution = Σ_{k'} [At_{ik'} * log_odds_{kk'}
        #                                    + tau_sum_{k'} * log1mB_{kk'}]
        # = At @ log_odds.T + tau_sum @ log1mB.T (broadcast over chunk)
        # But we need to subtract self-connection terms (i's own tau)
        # For large N, the self-correction is negligible; we include it for
        # correctness by subtracting tau[i,:] contribution.

        base_term = tau_sum[np.newaxis, :] @ log1mB.T  # (1, K) → broadcast
        neighbor_term = At @ log_odds.T  # (chunk_size, K)

        # Self-correction: subtract node i's own contribution
        tau_self = tau[start:end, :]  # (chunk_size, K)
        self_corr = tau_self @ log1mB.T  # (chunk_size, K)

        log_tau_chunk += neighbor_term + base_term - self_corr

    # Metadata prior contribution
    if metadata_labels is not None and log_meta_B is not None and meta_weight > 0:
        meta_chunk = metadata_labels[start:end]
        valid = meta_chunk >= 0
        if np.any(valid):
            log_tau_chunk[valid] += meta_weight * log_meta_B[:, meta_chunk[valid]].T

    # Normalise in log space
    log_tau_chunk -= logsumexp(log_tau_chunk, axis=1, keepdims=True)
    tau_new = np.exp(log_tau_chunk)
    tau_new = np.clip(tau_new, 1e-10, None)
    tau_new /= tau_new.sum(axis=1, keepdims=True)

    return tau_new


def _compute_elbo(
    tau: np.ndarray,
    adj_matrices: list,
    B_list: list,
    pi: np.ndarray,
) -> float:
    """
    Compute the evidence lower bound (ELBO) for monitoring convergence.

    ELBO = E_q[log p(A,z|π,B)] - E_q[log q(z)]
    """
    N, K = tau.shape
    eps = 1e-300

    # E_q[log p(z|π)] = Σ_i Σ_k τ_ik log π_k
    elbo = np.sum(tau * np.log(np.clip(pi, eps, None))[np.newaxis, :])

    # E_q[log p(A|z,B)] for each layer
    for l_idx, A in enumerate(adj_matrices):
        B = B_list[l_idx]
        logB = np.log(np.clip(B, eps, None))
        log1mB = np.log(np.clip(1.0 - B, eps, None))

        tau_sum = tau.sum(axis=0)

        # Expected edge term: Σ_{i<j} Σ_{k,k'} τ_ik τ_jk' A_ij log B_kk'
        At = A.dot(tau)
        edge_term = 0.5 * np.sum(tau.T @ At * logB)

        # Expected non-edge term: Σ_{i<j} Σ_{k,k'} τ_ik τ_jk' (1-A_ij) log(1-B_kk')
        total_pairs = np.outer(tau_sum, tau_sum) - np.diag((tau ** 2).sum(axis=0))
        non_edge_term = 0.5 * np.sum(total_pairs * log1mB) - edge_term * (np.sum(log1mB) / np.sum(logB) if np.sum(logB) != 0 else 0)

        # Simpler correct computation:
        # total_term = 0.5 * [Σ_kk' (n_kk' logB + (m_kk' - n_kk') log(1-B))]
        # where n_kk' = expected edges, m_kk' = expected possible edges
        n_kk = tau.T @ At  # (K,K) expected edges (double counted for off-diag)
        m_kk = np.outer(tau_sum, tau_sum)
        np.fill_diagonal(m_kk, m_kk.diagonal() - (tau ** 2).sum(axis=0))

        layer_elbo = 0.5 * np.sum(n_kk * logB + (m_kk - n_kk) * log1mB)
        elbo += layer_elbo

    # Entropy: -E_q[log q(z)] = -Σ_i Σ_k τ_ik log τ_ik
    entropy = -np.sum(tau * np.log(np.clip(tau, eps, None)))
    elbo += entropy

    return elbo


def _compute_log_likelihood(
    tau: np.ndarray,
    adj_matrices: list,
    B_list: list,
    pi: np.ndarray,
) -> float:
    """Compute the expected complete-data log-likelihood under q(z)."""
    N, K = tau.shape
    eps = 1e-300

    ll = np.sum(tau * np.log(np.clip(pi, eps, None))[np.newaxis, :])

    for l_idx, A in enumerate(adj_matrices):
        B = B_list[l_idx]
        logB = np.log(np.clip(B, eps, None))
        log1mB = np.log(np.clip(1.0 - B, eps, None))

        tau_sum = tau.sum(axis=0)
        At = A.dot(tau)
        n_kk = tau.T @ At
        m_kk = np.outer(tau_sum, tau_sum)
        np.fill_diagonal(m_kk, m_kk.diagonal() - (tau ** 2).sum(axis=0))

        ll += 0.5 * np.sum(n_kk * logB + (m_kk - n_kk) * log1mB)

    return ll


# ---------------------------------------------------------------------------
# Main fitting routine
# ---------------------------------------------------------------------------

def fit_multiplex_sbm(
    data: MultiplexData,
    K: int,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_restarts: int = 5,
    use_metadata: bool = False,
    metadata_weight: float = 1.0,
    seed: int = 42,
    chunk_size: int = 5000,
    verbose: bool = True,
) -> SBMParams:
    """
    Fit a multiplex SBM with K communities using variational EM.

    Parameters
    ----------
    data : MultiplexData
        Pre-processed multiplex network.
    K : int
        Number of communities.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence threshold on relative ELBO change.
    n_restarts : int
        Number of random restarts; best ELBO wins.
    use_metadata : bool
        If True and data has metadata, incorporate as prior.
    metadata_weight : float
        Weight (η) for metadata prior term.
    seed : int
        Random seed for reproducibility.
    chunk_size : int
        Number of nodes per E-step chunk (controls memory).
    verbose : bool
        Print progress.

    Returns
    -------
    SBMParams with fitted model.
    """
    N = data.n_nodes
    L = data.n_layers

    logger.info("Fitting multiplex SBM: N=%d, L=%d, K=%d, restarts=%d",
                N, L, K, n_restarts)

    best_result = None
    best_elbo = -np.inf

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart * 1000)

        # Initialise
        tau = _init_tau(N, K, rng)
        pi = _m_step_pi(tau)
        B_list = [_m_step_B(tau, A, K) for A in data.adj_matrices]

        # Metadata emission matrix (if used)
        log_meta_B = None
        if use_metadata and data.metadata_labels is not None:
            M = len(data.metadata_names) if data.metadata_names else 0
            if M > 0:
                # Initialise uniformly
                meta_B = np.ones((K, M), dtype=np.float64) / M
                log_meta_B = np.log(meta_B)

        elbo_history = []
        converged = False

        for it in range(max_iter):
            t0 = time.time()

            # --- E-step (chunked) ---
            log_pi = np.log(np.clip(pi, 1e-300, None))
            tau_new = np.empty_like(tau)

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                tau_new[start:end] = _e_step_chunk(
                    tau, data.adj_matrices, B_list, log_pi,
                    (start, end),
                    data.metadata_labels if use_metadata else None,
                    log_meta_B,
                    metadata_weight if use_metadata else 0.0,
                )

            tau = tau_new

            # --- M-step ---
            pi = _m_step_pi(tau)
            B_list = [_m_step_B(tau, A, K) for A in data.adj_matrices]

            # Update metadata emission if used
            if use_metadata and data.metadata_labels is not None and log_meta_B is not None:
                M = log_meta_B.shape[1]
                meta_B = np.zeros((K, M), dtype=np.float64)
                for m in range(M):
                    mask = data.metadata_labels == m
                    if np.any(mask):
                        meta_B[:, m] = tau[mask].sum(axis=0)
                meta_B += 1e-10  # smoothing
                meta_B /= meta_B.sum(axis=1, keepdims=True)
                log_meta_B = np.log(meta_B)

            # --- ELBO ---
            elbo = _compute_elbo(tau, data.adj_matrices, B_list, pi)
            elbo_history.append(elbo)

            dt = time.time() - t0

            if verbose and (it % 5 == 0 or it < 3):
                logger.info("  Restart %d | Iter %3d | ELBO=%.4e | dt=%.1fs",
                            restart, it, elbo, dt)

            # Check convergence
            if it > 0:
                rel_change = abs(elbo - elbo_history[-2]) / (abs(elbo_history[-2]) + 1e-300)
                if rel_change < tol:
                    converged = True
                    if verbose:
                        logger.info("  Restart %d converged at iteration %d "
                                    "(rel_change=%.2e)", restart, it, rel_change)
                    break

        if elbo > best_elbo:
            best_elbo = elbo
            ll = _compute_log_likelihood(tau, data.adj_matrices, B_list, pi)
            # Number of free parameters: K-1 (pi) + L * K*(K+1)/2 (B symmetric)
            n_params = (K - 1) + L * K * (K + 1) // 2
            if use_metadata and log_meta_B is not None:
                n_params += K * (log_meta_B.shape[1] - 1)

            best_result = SBMParams(
                K=K,
                pi=pi.copy(),
                B=[b.copy() for b in B_list],
                tau=tau.copy(),
                community_assignments=np.argmax(tau, axis=1),
                log_likelihood=ll,
                n_params=n_params,
                converged=converged,
                n_iterations=it + 1,
                elbo_history=elbo_history,
                metadata_weight=metadata_weight if use_metadata else 0.0,
                metadata_B=np.exp(log_meta_B) if log_meta_B is not None else None,
            )

    logger.info("Best ELBO across restarts: %.4e (K=%d)", best_elbo, K)
    return best_result


# ---------------------------------------------------------------------------
# Parallel wrapper for model selection
# ---------------------------------------------------------------------------

def fit_single_K(args_tuple):
    """
    Wrapper for multiprocessing: fits SBM for a single K value.
    args_tuple: (data_path, registry_path, K, fit_kwargs)
    Returns (K, SBMParams_dict) — we return a dict to avoid pickling issues.
    """
    data_path, registry_path, K, fit_kwargs = args_tuple

    # Reload data in worker process to avoid pickling large arrays
    edge_types = fit_kwargs.pop("edge_types", ("ppi", "pathway", "ligand_receptor"))
    use_meta_load = fit_kwargs.get("use_metadata", False)
    metadata_col = fit_kwargs.pop("metadata_col", "cell_type")

    data = load_multiplex_data(
        data_path, registry_path,
        edge_types=edge_types,
        use_metadata=use_meta_load,
        metadata_col=metadata_col,
    )

    result = fit_multiplex_sbm(data, K=K, **fit_kwargs)

    # Convert to serialisable dict
    result_dict = {
        "K": result.K,
        "pi": result.pi,
        "B": result.B,
        "tau": result.tau,
        "community_assignments": result.community_assignments,
        "log_likelihood": result.log_likelihood,
        "n_params": result.n_params,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "elbo_history": result.elbo_history,
        "metadata_weight": result.metadata_weight,
        "metadata_B": result.metadata_B,
    }
    return (K, result_dict)


def dict_to_sbm_params(d: dict) -> SBMParams:
    """Convert a serialised dict back to SBMParams."""
    return SBMParams(**d)
