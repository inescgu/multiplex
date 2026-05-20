#!/usr/bin/env python3
"""
model_selection.py — Automatic selection of optimal K for multiplex SBM
=========================================================================

Implements three information criteria for model comparison:
  - BIC  (Bayesian Information Criterion)
  - AIC  (Akaike Information Criterion)
  - ICL  (Integrated Classification Likelihood — penalises fuzzy assignments)

Also provides a parallel sweep over K values and reporting utilities.

Author: Abacus AI Agent
"""

import logging
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sbm_core import (
    MultiplexData,
    SBMParams,
    dict_to_sbm_params,
    fit_multiplex_sbm,
    fit_single_K,
    load_multiplex_data,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------

def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """
    BIC = -2 * log_likelihood + n_params * log(n_obs)

    Lower is better.
    n_obs: number of node pairs × number of layers (total observations).
    """
    return -2.0 * log_likelihood + n_params * np.log(n_obs)


def compute_aic(log_likelihood: float, n_params: int) -> float:
    """
    AIC = -2 * log_likelihood + 2 * n_params

    Lower is better.
    """
    return -2.0 * log_likelihood + 2.0 * n_params


def compute_icl(
    log_likelihood: float,
    n_params: int,
    n_obs: int,
    tau: np.ndarray,
) -> float:
    """
    ICL = BIC + classification entropy penalty.

    The entropy term penalises fuzzy community assignments, favouring
    models where nodes are more clearly assigned to communities.

    ICL = BIC - Σ_i Σ_k τ_ik log τ_ik

    Lower is better.
    """
    bic = compute_bic(log_likelihood, n_params, n_obs)
    # Classification entropy (negative because τ log τ is negative)
    eps = 1e-300
    entropy = -np.sum(tau * np.log(np.clip(tau, eps, None)))
    # ICL = BIC + entropy (entropy >= 0, so ICL >= BIC)
    return bic + entropy


def compute_all_criteria(
    result: SBMParams,
    n_nodes: int,
    n_layers: int,
) -> dict:
    """
    Compute all information criteria for a fitted model.

    Parameters
    ----------
    result : SBMParams
        Fitted model parameters.
    n_nodes : int
        Number of nodes.
    n_layers : int
        Number of layers.

    Returns
    -------
    dict with keys: K, log_likelihood, n_params, BIC, AIC, ICL, ELBO
    """
    # Number of observations: for each layer, N*(N-1)/2 possible edges
    n_obs = n_layers * n_nodes * (n_nodes - 1) // 2

    bic = compute_bic(result.log_likelihood, result.n_params, n_obs)
    aic = compute_aic(result.log_likelihood, result.n_params)
    icl = compute_icl(result.log_likelihood, result.n_params, n_obs, result.tau)
    elbo = result.elbo_history[-1] if result.elbo_history else np.nan

    return {
        "K": result.K,
        "log_likelihood": result.log_likelihood,
        "n_params": result.n_params,
        "n_obs": n_obs,
        "BIC": bic,
        "AIC": aic,
        "ICL": icl,
        "ELBO": elbo,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
    }


# ---------------------------------------------------------------------------
# Model selection sweep
# ---------------------------------------------------------------------------

def select_K_sequential(
    data: MultiplexData,
    K_range: range,
    fit_kwargs: dict,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[int, SBMParams]]:
    """
    Sweep over K values sequentially, fit SBM for each, compute criteria.

    Returns
    -------
    comparison_df : pd.DataFrame
        Rows = K values, columns = criteria.
    results : dict
        K -> SBMParams for each fitted model.
    """
    records = []
    results = {}

    for K in K_range:
        t0 = time.time()
        logger.info("=" * 50)
        logger.info("Fitting K=%d ...", K)

        result = fit_multiplex_sbm(data, K=K, **fit_kwargs)
        results[K] = result

        criteria = compute_all_criteria(result, data.n_nodes, data.n_layers)
        criteria["fit_time_s"] = time.time() - t0
        records.append(criteria)

        if verbose:
            logger.info("  K=%d | BIC=%.4e | AIC=%.4e | ICL=%.4e | LL=%.4e | time=%.1fs",
                        K, criteria["BIC"], criteria["AIC"], criteria["ICL"],
                        criteria["log_likelihood"], criteria["fit_time_s"])

    df = pd.DataFrame(records)
    return df, results


def select_K_parallel(
    graph_path: str,
    registry_path: str,
    K_range: range,
    fit_kwargs: dict,
    n_cores: int = 4,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[int, SBMParams]]:
    """
    Sweep over K values in parallel using multiprocessing.

    Each worker reloads the data to avoid pickling large arrays.

    Returns
    -------
    comparison_df : pd.DataFrame
    results : dict of K -> SBMParams
    """
    # Prepare arguments for each K
    tasks = []
    for K in K_range:
        kw = dict(fit_kwargs)
        tasks.append((graph_path, registry_path, K, kw))

    logger.info("Launching parallel K sweep: K in [%d, %d], %d cores",
                K_range.start, K_range.stop - 1, n_cores)

    results_dict = {}
    records = []

    # Use spawn context if available for safety
    if n_cores > 1 and len(tasks) > 1:
        effective_cores = min(n_cores, len(tasks))
        with Pool(processes=effective_cores) as pool:
            for K, res_dict in pool.imap_unordered(fit_single_K, tasks):
                result = dict_to_sbm_params(res_dict)
                results_dict[K] = result
                logger.info("  Completed K=%d", K)
    else:
        # Sequential fallback
        for task in tasks:
            K, res_dict = fit_single_K(task)
            result = dict_to_sbm_params(res_dict)
            results_dict[K] = result
            logger.info("  Completed K=%d", K)

    # Now load data once to compute criteria
    edge_types = fit_kwargs.get("edge_types", ("ppi", "pathway", "ligand_receptor"))
    use_meta = fit_kwargs.get("use_metadata", False)
    meta_col = fit_kwargs.get("metadata_col", "cell_type")

    data = load_multiplex_data(
        graph_path, registry_path,
        edge_types=edge_types,
        use_metadata=use_meta,
        metadata_col=meta_col,
    )

    for K in sorted(results_dict.keys()):
        result = results_dict[K]
        criteria = compute_all_criteria(result, data.n_nodes, data.n_layers)
        records.append(criteria)
        if verbose:
            logger.info("  K=%d | BIC=%.4e | AIC=%.4e | ICL=%.4e",
                        K, criteria["BIC"], criteria["AIC"], criteria["ICL"])

    df = pd.DataFrame(records).sort_values("K").reset_index(drop=True)
    return df, results_dict


def select_best_K(
    comparison_df: pd.DataFrame,
    criterion: str = "ICL",
) -> int:
    """
    Select the best K based on the specified criterion (lower is better).

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of select_K_* functions.
    criterion : str
        One of 'BIC', 'AIC', 'ICL'. Default: 'ICL'.

    Returns
    -------
    int : optimal K value.
    """
    if criterion not in comparison_df.columns:
        raise ValueError(f"Criterion '{criterion}' not found. Available: {list(comparison_df.columns)}")

    best_idx = comparison_df[criterion].idxmin()
    best_K = int(comparison_df.loc[best_idx, "K"])
    logger.info("Best K by %s: %d (value=%.4e)",
                criterion, best_K, comparison_df.loc[best_idx, criterion])
    return best_K


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_model_comparison(
    comparison_df: pd.DataFrame,
    output_dir: str,
    filename: str = "model_comparison.csv",
) -> str:
    """Save model comparison table to CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    comparison_df.to_csv(path, index=False, float_format="%.6e")
    logger.info("Model comparison saved → %s", path)
    return str(path)


def save_best_model(
    result: SBMParams,
    data: MultiplexData,
    output_dir: str,
) -> dict:
    """
    Save the best model's community assignments and parameters.

    Returns
    -------
    dict with paths to saved files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    # --- Community assignments ---
    assign_df = data.node_attrs.copy()
    assign_df["community"] = result.community_assignments

    # Add posterior probabilities for top communities
    n_top = min(3, result.K)
    top_n = np.argsort(-result.tau, axis=1)[:, :n_top]
    for rank in range(n_top):
        assign_df[f"community_prob_rank{rank+1}"] = [
            result.tau[i, top_n[i, rank]] for i in range(data.n_nodes)
        ]
        assign_df[f"community_rank{rank+1}"] = top_n[:, rank]

    assign_path = out / "community_assignments.csv"
    assign_df.to_csv(assign_path, index=False)
    paths["assignments"] = str(assign_path)
    logger.info("Community assignments saved → %s", assign_path)

    # --- Model parameters ---
    params = {
        "K": result.K,
        "pi": result.pi.tolist(),
        "B": [b.tolist() for b in result.B],
        "log_likelihood": result.log_likelihood,
        "n_params": result.n_params,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "elbo_history": [float(x) for x in result.elbo_history],
        "metadata_weight": result.metadata_weight,
    }
    if result.metadata_B is not None:
        params["metadata_B"] = result.metadata_B.tolist()

    import json
    params_path = out / "model_parameters.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    paths["parameters"] = str(params_path)
    logger.info("Model parameters saved → %s", params_path)

    # --- Tau matrix (compressed) ---
    tau_path = out / "tau_posteriors.npz"
    np.savez_compressed(tau_path, tau=result.tau)
    paths["tau"] = str(tau_path)
    logger.info("Posterior matrix saved → %s", tau_path)

    return paths
