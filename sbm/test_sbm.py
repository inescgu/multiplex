#!/usr/bin/env python3
"""
test_sbm.py — End-to-end test with a synthetic multiplex network.

Creates a small planted-partition multiplex graph, saves it in the
same format as build_integrated_multiplex.py, then runs the full
SBM pipeline to verify correctness.
"""

import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# -----------------------------------------------------------------------
# 1. Generate synthetic multiplex
# -----------------------------------------------------------------------

def make_synthetic_multiplex(
    n_per_community: int = 50,
    K: int = 4,
    n_layers: int = 3,
    p_in: float = 0.3,
    p_out: float = 0.02,
    seed: int = 42,
):
    """
    Create a planted-partition multiplex graph.
    Returns (G, node_registry_df, true_communities).
    """
    rng = np.random.default_rng(seed)
    N = n_per_community * K
    true_z = np.repeat(np.arange(K), n_per_community)

    sources = ["tabula_sapiens", "hrdg", "global_ppi"]
    cell_types = ["T_cell", "B_cell", "Macrophage"]
    edge_type_names = ["ppi", "pathway", "ligand_receptor"]

    G = nx.Graph()

    # Register nodes
    registry_rows = []
    for i in range(N):
        source = sources[i % len(sources)]
        cell_type = cell_types[i % len(cell_types)]
        gene = f"GENE{i:04d}"
        G.add_node(i, source=source, cell_type=cell_type, gene=gene)
        registry_rows.append({
            "node_id": i,
            "source": source,
            "cell_type": cell_type,
            "gene": gene,
        })

    registry_df = pd.DataFrame(registry_rows)

    # Add edges per layer
    for l in range(n_layers):
        etype = edge_type_names[l % len(edge_type_names)]
        for i in range(N):
            for j in range(i + 1, N):
                p = p_in if true_z[i] == true_z[j] else p_out
                if rng.random() < p:
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, edge_type=etype)

    print(f"Synthetic graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, K_true={K}")

    return G, registry_df, true_z


def save_synthetic(G, registry_df, tmpdir):
    """Save in the same format as the real pipeline."""
    graph_path = os.path.join(tmpdir, "integrated_multiplex.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f, protocol=4)

    reg_path = os.path.join(tmpdir, "node_registry.csv")
    registry_df.to_csv(reg_path, index=False)

    return graph_path, reg_path


# -----------------------------------------------------------------------
# 2. Run tests
# -----------------------------------------------------------------------

def test_load_and_fit():
    from sbm_core import load_multiplex_data, fit_multiplex_sbm

    tmpdir = tempfile.mkdtemp(prefix="sbm_test_")
    print(f"\nTest dir: {tmpdir}")

    G, reg_df, true_z = make_synthetic_multiplex(
        n_per_community=30, K=3, p_in=0.35, p_out=0.03)
    graph_path, reg_path = save_synthetic(G, reg_df, tmpdir)

    # Load
    data = load_multiplex_data(
        graph_path, reg_path,
        edge_types=("ppi", "pathway", "ligand_receptor"),
        use_metadata=True,
        metadata_col="cell_type",
    )
    print(f"  Loaded: N={data.n_nodes}, L={data.n_layers}, "
          f"layers={data.layer_names}")
    assert data.n_nodes == 90
    assert data.n_layers == 3

    # Fit
    result = fit_multiplex_sbm(
        data, K=3,
        max_iter=50,
        tol=1e-5,
        n_restarts=3,
        use_metadata=True,
        metadata_weight=0.5,
        seed=42,
        chunk_size=50,
    )
    print(f"  Fit: K={result.K}, converged={result.converged}, "
          f"iters={result.n_iterations}, LL={result.log_likelihood:.4e}")
    assert result.tau.shape == (90, 3)
    assert result.community_assignments.shape == (90,)
    assert len(result.B) == 3

    # Check communities are non-trivial
    from collections import Counter
    comm_counts = Counter(result.community_assignments)
    print(f"  Community sizes: {dict(comm_counts)}")
    assert len(comm_counts) >= 2, "Expected at least 2 non-empty communities"

    print("  ✓ test_load_and_fit PASSED\n")
    return data, result, graph_path, reg_path, tmpdir


def test_model_selection(data, graph_path, reg_path, tmpdir):
    from model_selection import (
        select_K_sequential, select_best_K, save_model_comparison, save_best_model
    )

    print("Testing model selection (K=2..5) ...")
    fit_kwargs = dict(
        max_iter=30, tol=1e-4, n_restarts=2,
        use_metadata=False, seed=42, chunk_size=50,
    )

    comparison_df, all_results = select_K_sequential(
        data, K_range=range(2, 6), fit_kwargs=fit_kwargs
    )
    print(comparison_df[["K", "BIC", "AIC", "ICL", "converged"]].to_string(index=False))

    best_K = select_best_K(comparison_df, criterion="ICL")
    print(f"  Best K by ICL: {best_K}")
    assert 2 <= best_K <= 5

    # Save
    save_model_comparison(comparison_df, tmpdir)
    paths = save_best_model(all_results[best_K], data, tmpdir)
    print(f"  Saved files: {list(paths.keys())}")

    assert os.path.exists(os.path.join(tmpdir, "model_comparison.csv"))
    assert os.path.exists(os.path.join(tmpdir, "community_assignments.csv"))

    print("  ✓ test_model_selection PASSED\n")
    return comparison_df, all_results[best_K]


def test_visualisations(data, result, comparison_df, graph_path, tmpdir):
    from sbm_visualization import generate_all_visualisations

    print("Testing visualisation generation ...")
    viz_paths = generate_all_visualisations(
        data=data,
        result=result,
        comparison_df=comparison_df,
        graph_path=graph_path,
        output_dir=tmpdir,
        max_network_nodes=100,
    )
    print(f"  Generated {len(viz_paths)} visualisation(s):")
    for name, path in viz_paths.items():
        exists = os.path.exists(path) if isinstance(path, str) else all(os.path.exists(p) for p in path)
        status = "✓" if exists else "✗"
        print(f"    {status} {name}: {path}")

    assert len(viz_paths) >= 4
    print("  ✓ test_visualisations PASSED\n")


def test_cli_dry_run(graph_path, reg_path, tmpdir):
    """Test CLI argument parsing (dry run)."""
    import subprocess
    out_dir = os.path.join(tmpdir, "cli_output")
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "run_multiplex_sbm.py"),
        "--graph", graph_path,
        "--registry", reg_path,
        "--output-dir", out_dir,
        "--k-min", "2", "--k-max", "4",
        "--max-iter", "20",
        "--n-restarts", "2",
        "--tol", "1e-4",
        "--chunk-size", "50",
        "--max-network-nodes", "100",
        "--n-cores", "1",
    ]
    print(f"Running CLI: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"  Return code: {proc.returncode}")
    if proc.returncode != 0:
        print(f"  STDOUT:\n{proc.stdout[-2000:]}")
        print(f"  STDERR:\n{proc.stderr[-2000:]}")
    else:
        # Check outputs exist
        expected = [
            "community_assignments.csv",
            "model_parameters.json",
            "model_comparison.csv",
            "run_summary.json",
        ]
        for fname in expected:
            path = os.path.join(out_dir, fname)
            status = "✓" if os.path.exists(path) else "✗"
            print(f"    {status} {fname}")
        print("  ✓ test_cli_dry_run PASSED\n")

    assert proc.returncode == 0, f"CLI failed: {proc.stderr[-500:]}"


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MULTIPLEX SBM — END-TO-END TESTS")
    print("=" * 60)

    data, result, graph_path, reg_path, tmpdir = test_load_and_fit()
    comparison_df, best_result = test_model_selection(data, graph_path, reg_path, tmpdir)
    test_visualisations(data, best_result, comparison_df, graph_path, tmpdir)
    test_cli_dry_run(graph_path, reg_path, tmpdir)

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print(f"Test artifacts in: {tmpdir}")
    print("=" * 60)
