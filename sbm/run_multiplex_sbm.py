#!/usr/bin/env python3
"""
run_multiplex_sbm.py — Main driver for Multiplex Stochastic Block Model
==========================================================================

Command-line interface for fitting a multiplex SBM to an integrated
multiplex interactome network. Supports:

  - Automatic model selection over a range of K values
  - Fixed K fitting
  - Optional metadata incorporation (cell type prior)
  - Multi-core parallelisation
  - Comprehensive output: assignments, parameters, visualisations

Usage examples:

  # Model selection (K=2..20), 4 cores, no metadata:
  python run_multiplex_sbm.py \\
      --graph integrated_multiplex.pkl \\
      --registry node_registry.csv \\
      --output-dir sbm_results \\
      --k-min 2 --k-max 20 \\
      --n-cores 4

  # Fixed K=10 with metadata:
  python run_multiplex_sbm.py \\
      --graph integrated_multiplex.pkl \\
      --registry node_registry.csv \\
      --output-dir sbm_results \\
      --fixed-k 10 \\
      --use-metadata

  # Full sweep K=2..50, 10 cores, metadata, custom convergence:
  python run_multiplex_sbm.py \\
      --graph integrated_multiplex.pkl \\
      --registry node_registry.csv \\
      --output-dir sbm_results \\
      --k-min 2 --k-max 50 \\
      --n-cores 10 \\
      --use-metadata \\
      --max-iter 300 --tol 1e-7 \\
      --n-restarts 10

Author: Abacus AI Agent
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure this module can find sibling files
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sbm_core import MultiplexData, SBMParams, fit_multiplex_sbm, load_multiplex_data
from model_selection import (
    compute_all_criteria,
    save_best_model,
    save_model_comparison,
    select_best_K,
    select_K_parallel,
    select_K_sequential,
)
from sbm_visualization import generate_all_visualisations


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multiplex Stochastic Block Model — community detection "
                    "on integrated multiplex interactome networks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Model selection sweep:
  python run_multiplex_sbm.py --graph integrated_multiplex.pkl \\
      --registry node_registry.csv --output-dir sbm_results \\
      --k-min 2 --k-max 20 --n-cores 4

  # Fixed K with metadata:
  python run_multiplex_sbm.py --graph integrated_multiplex.pkl \\
      --registry node_registry.csv --output-dir sbm_results \\
      --fixed-k 10 --use-metadata

Edge types (ppi, pathway, ligand_receptor) are treated equally in the model.
Each edge type becomes a separate layer with its own block probability matrix,
but all layers share a single community assignment vector.
        """,
    )

    # Input paths
    io = parser.add_argument_group("Input / Output")
    io.add_argument("--graph", required=True,
                    help="Path to integrated_multiplex.pkl (NetworkX graph)")
    io.add_argument("--registry", required=True,
                    help="Path to node_registry.csv (node ID mappings)")
    io.add_argument("--output-dir", default="sbm_results",
                    help="Output directory for results (default: sbm_results)")

    # Model selection
    ms = parser.add_argument_group("Model Selection")
    ms.add_argument("--fixed-k", type=int, default=None,
                    help="Fixed number of communities (skip model selection)")
    ms.add_argument("--k-min", type=int, default=2,
                    help="Minimum K for model selection sweep (default: 2)")
    ms.add_argument("--k-max", type=int, default=20,
                    help="Maximum K for model selection sweep (default: 20)")
    ms.add_argument("--k-step", type=int, default=1,
                    help="Step size for K sweep (default: 1)")
    ms.add_argument("--criterion", choices=["BIC", "AIC", "ICL"],
                    default="ICL",
                    help="Information criterion for selecting best K (default: ICL)")

    # Metadata
    meta = parser.add_argument_group("Metadata")
    meta.add_argument("--use-metadata", action="store_true",
                      help="Incorporate node metadata (cell_type) as prior")
    meta.add_argument("--metadata-col", default="cell_type",
                      help="Node attribute column for metadata (default: cell_type)")
    meta.add_argument("--metadata-weight", type=float, default=1.0,
                      help="Weight (η) for metadata prior (default: 1.0)")

    # Inference parameters
    inf = parser.add_argument_group("Inference Parameters")
    inf.add_argument("--max-iter", type=int, default=200,
                     help="Max EM iterations per fit (default: 200)")
    inf.add_argument("--tol", type=float, default=1e-6,
                     help="Convergence threshold on relative ELBO change (default: 1e-6)")
    inf.add_argument("--n-restarts", type=int, default=5,
                     help="Random restarts per K value (default: 5)")
    inf.add_argument("--chunk-size", type=int, default=5000,
                     help="E-step chunk size for memory control (default: 5000)")
    inf.add_argument("--seed", type=int, default=42,
                     help="Random seed (default: 42)")

    # Performance
    perf = parser.add_argument_group("Performance")
    perf.add_argument("--n-cores", type=int, default=1,
                      help="Number of CPU cores for parallel K sweep (default: 1, max: 10)")
    perf.add_argument("--parallel-mode", choices=["sequential", "parallel"],
                      default="auto",
                      help="Force sequential or parallel mode (default: auto)")

    # Visualisation
    viz = parser.add_argument_group("Visualisation")
    viz.add_argument("--no-viz", action="store_true",
                     help="Skip visualisation generation")
    viz.add_argument("--max-network-nodes", type=int, default=2000,
                     help="Max nodes for network layout plot (default: 2000)")

    # Edge types
    et = parser.add_argument_group("Edge Types")
    et.add_argument("--edge-types", nargs="+",
                    default=["ppi", "pathway", "ligand_receptor"],
                    help="Edge types to include as layers (default: all three)")

    # Logging
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(args.output_dir, "sbm_run.log") if os.path.isdir(args.output_dir)
                else "sbm_run.log"
            ),
        ],
    )
    logger = logging.getLogger("multiplex_sbm")

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Re-configure file handler now that dir exists
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logging.root.removeHandler(handler)
    fh = logging.FileHandler(out_dir / "sbm_run.log")
    fh.setLevel(getattr(logging, args.log_level))
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    logging.root.addHandler(fh)

    t_start = time.time()

    # Save run configuration
    config = vars(args)
    config_path = out_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info("Run configuration saved → %s", config_path)

    # ===================================================================
    # STEP 1: Load data
    # ===================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading multiplex data")
    logger.info("=" * 60)

    edge_types = tuple(args.edge_types)
    data = load_multiplex_data(
        graph_path=args.graph,
        registry_path=args.registry,
        edge_types=edge_types,
        use_metadata=args.use_metadata,
        metadata_col=args.metadata_col,
    )

    logger.info("Loaded: %d nodes, %d layers (%s)",
                data.n_nodes, data.n_layers, ", ".join(data.layer_names))

    # ===================================================================
    # STEP 2: Fit model(s)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Fitting SBM model(s)")
    logger.info("=" * 60)

    fit_kwargs = dict(
        max_iter=args.max_iter,
        tol=args.tol,
        n_restarts=args.n_restarts,
        use_metadata=args.use_metadata,
        metadata_weight=args.metadata_weight,
        seed=args.seed,
        chunk_size=args.chunk_size,
        verbose=True,
    )

    comparison_df = None
    all_results = {}

    if args.fixed_k is not None:
        # Fixed K mode
        K = args.fixed_k
        logger.info("Fixed K=%d mode", K)
        result = fit_multiplex_sbm(data, K=K, **fit_kwargs)
        all_results[K] = result

        from model_selection import compute_all_criteria
        criteria = compute_all_criteria(result, data.n_nodes, data.n_layers)
        comparison_df = pd.DataFrame([criteria])
        best_K = K

    else:
        # Model selection sweep
        K_range = range(args.k_min, args.k_max + 1, args.k_step)
        n_cores = min(args.n_cores, 10)

        # Decide parallel vs sequential
        use_parallel = (
            (args.parallel_mode == "parallel") or
            (args.parallel_mode == "auto" and n_cores > 1 and len(K_range) > 2)
        )

        if use_parallel:
            logger.info("Parallel model selection: K in [%d, %d], step=%d, cores=%d",
                        args.k_min, args.k_max, args.k_step, n_cores)

            # For parallel mode, pass kwargs that fit_single_K needs
            par_kwargs = dict(fit_kwargs)
            par_kwargs["edge_types"] = edge_types
            par_kwargs["metadata_col"] = args.metadata_col

            comparison_df, all_results = select_K_parallel(
                graph_path=args.graph,
                registry_path=args.registry,
                K_range=K_range,
                fit_kwargs=par_kwargs,
                n_cores=n_cores,
                verbose=True,
            )
        else:
            logger.info("Sequential model selection: K in [%d, %d], step=%d",
                        args.k_min, args.k_max, args.k_step)
            comparison_df, all_results = select_K_sequential(
                data=data,
                K_range=K_range,
                fit_kwargs=fit_kwargs,
                verbose=True,
            )

        # Select best K
        best_K = select_best_K(comparison_df, criterion=args.criterion)
        logger.info("Selected best K=%d by %s", best_K, args.criterion)

    best_result = all_results[best_K]

    # ===================================================================
    # STEP 3: Save results
    # ===================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Saving results")
    logger.info("=" * 60)

    # Model comparison table
    if comparison_df is not None:
        save_model_comparison(comparison_df, str(out_dir))

    # Best model outputs
    saved_paths = save_best_model(best_result, data, str(out_dir))

    # ===================================================================
    # STEP 4: Visualisations
    # ===================================================================
    if not args.no_viz:
        logger.info("=" * 60)
        logger.info("STEP 4: Generating visualisations")
        logger.info("=" * 60)

        viz_paths = generate_all_visualisations(
            data=data,
            result=best_result,
            comparison_df=comparison_df if comparison_df is not None and len(comparison_df) > 1 else None,
            graph_path=args.graph,
            output_dir=str(out_dir),
            max_network_nodes=args.max_network_nodes,
        )
        saved_paths.update(viz_paths)

    # ===================================================================
    # STEP 5: Summary
    # ===================================================================
    elapsed = time.time() - t_start

    logger.info("=" * 60)
    logger.info("MULTIPLEX SBM — RUN COMPLETE")
    logger.info("=" * 60)
    logger.info("  Best K            : %d", best_K)
    logger.info("  Communities found  : %d", len(np.unique(best_result.community_assignments)))
    logger.info("  Converged          : %s", best_result.converged)
    logger.info("  Iterations         : %d", best_result.n_iterations)
    logger.info("  Log-likelihood     : %.4e", best_result.log_likelihood)
    logger.info("  ELBO (final)       : %.4e",
                best_result.elbo_history[-1] if best_result.elbo_history else float("nan"))

    if comparison_df is not None and len(comparison_df) > 1:
        best_row = comparison_df[comparison_df["K"] == best_K].iloc[0]
        logger.info("  BIC                : %.4e", best_row.get("BIC", float("nan")))
        logger.info("  AIC                : %.4e", best_row.get("AIC", float("nan")))
        logger.info("  ICL                : %.4e", best_row.get("ICL", float("nan")))

    # Community size summary
    comm_counts = np.bincount(best_result.community_assignments, minlength=best_K)
    logger.info("  Community sizes    : min=%d, max=%d, median=%d",
                comm_counts.min(), comm_counts.max(), int(np.median(comm_counts)))

    logger.info("  Output directory   : %s", out_dir)
    logger.info("  Total runtime      : %.1f seconds (%.1f minutes)",
                elapsed, elapsed / 60)
    logger.info("=" * 60)

    # Save final summary
    summary = {
        "best_K": int(best_K),
        "n_communities_active": int(len(np.unique(best_result.community_assignments))),
        "converged": bool(best_result.converged),
        "n_iterations": int(best_result.n_iterations),
        "log_likelihood": float(best_result.log_likelihood),
        "final_elbo": float(best_result.elbo_history[-1]) if best_result.elbo_history else None,
        "n_nodes": int(data.n_nodes),
        "n_layers": int(data.n_layers),
        "layer_names": data.layer_names,
        "community_sizes": comm_counts.tolist(),
        "runtime_seconds": elapsed,
        "output_files": saved_paths,
    }
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Run summary saved → %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
