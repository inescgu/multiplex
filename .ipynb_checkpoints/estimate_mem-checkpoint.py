#!/usr/bin/env python3
"""
estimate_memory.py
Estimates peak RAM for MultiXrank from your layer files.
"""
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def estimate_memory(wdir: Path, config: str = "config.yml"):
    with open(wdir / config) as f:
        cfg = yaml.safe_load(f)

    layers = cfg["multiplex"]["gene_multiplex"]["layers"]

    total_edges = 0
    total_nodes_per_layer = []
    all_nodes = set()

    for lp in layers:
        path = wdir / lp
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", header=None,
                        names=["g1","g2"], dtype=str)
        n_edges = len(df)
        nodes = set(df["g1"]) | set(df["g2"])
        total_edges += n_edges
        total_nodes_per_layer.append(len(nodes))
        all_nodes |= nodes

    n_layers = len(layers)
    n_unique_genes = len(all_nodes)
    avg_nodes = np.mean(total_nodes_per_layer)
    max_nodes = max(total_nodes_per_layer)

    print(f"\n{'='*55}")
    print(f"MEMORY ESTIMATE")
    print(f"{'='*55}")
    print(f"  Layers            : {n_layers}")
    print(f"  Total edges       : {total_edges:,}")
    print(f"  Unique genes      : {n_unique_genes:,}")
    print(f"  Avg nodes/layer   : {avg_nodes:,.0f}")
    print(f"  Max nodes/layer   : {max_nodes:,}")

    # -------------------------------------------------------
    # Component estimates (bytes)
    # -------------------------------------------------------
    bytes_per_edge  = 3 * 8        # two node indices + weight, float64
    bytes_per_score = 8            # float64

    # 1. Adjacency / edge storage (CSR sparse: ~3 arrays per matrix)
    #    Each edge stored twice (undirected) + indices
    adj_bytes = total_edges * 2 * bytes_per_edge * n_layers
    
    # 2. Transition matrices (row-normalised adjacency, same sparsity)
    trans_bytes = adj_bytes  # same structure

    # 3. Score vectors: one per layer, length = nodes in that layer
    score_bytes = sum(n * bytes_per_score for n in total_nodes_per_layer)
    score_bytes *= 2  # current + previous iteration

    # 4. NetworkX graph objects (Python overhead ~200 bytes/node + ~150 bytes/edge)
    nx_node_bytes  = n_unique_genes * 200
    nx_edge_bytes  = total_edges * 150
    nx_bytes       = nx_node_bytes + nx_edge_bytes

    # 5. Pandas DataFrames during file loading (transient, peaks then releases)
    #    ~100 bytes per row during load
    load_peak_bytes = total_edges * 100  # transient peak

    # 6. Python interpreter + numpy + libraries baseline
    baseline_bytes = 500 * 1024**2  # ~500 MB

    raw_total = (adj_bytes + trans_bytes + score_bytes + 
                 nx_bytes + baseline_bytes)
    peak_total = raw_total + load_peak_bytes

    def fmt(b):
        return f"{b/1024**3:.2f} GB"

    print(f"\n  Memory components:")
    print(f"    Adjacency matrices  : {fmt(adj_bytes)}")
    print(f"    Transition matrices : {fmt(trans_bytes)}")
    print(f"    Score vectors       : {fmt(score_bytes)}")
    print(f"    NetworkX overhead   : {fmt(nx_bytes)}")
    print(f"    Loading peak        : {fmt(load_peak_bytes)}")
    print(f"    Python baseline     : {fmt(baseline_bytes)}")
    print(f"\n  Estimated steady-state RAM : {fmt(raw_total)}")
    print(f"  Estimated peak RAM         : {fmt(peak_total)}")

    # Safety margin recommendations
    for margin in [1.5, 2.0, 2.5]:
        rec = peak_total * margin / 1024**3
        print(f"  With {margin:.1f}× safety margin    : {rec:.0f} GB  "
              f"→ request --mem={int(np.ceil(rec/4)*4)}G")

    print(f"{'='*55}")
    print(f"\nRECOMMENDATION:")

    rec_gb = int(np.ceil(peak_total * 2.0 / 1024**3 / 4) * 4)  # round up to 4GB
    print(f"  #SBATCH --mem={rec_gb}G")
    print(f"\n  (2× safety margin, rounded to nearest 4 GB)")
    print(f"  If job gets OOM-killed, double this value.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wdir", required=True)
    parser.add_argument("--config", default="config.yml")
    args = parser.parse_args()
    estimate_memory(Path(args.wdir), args.config)