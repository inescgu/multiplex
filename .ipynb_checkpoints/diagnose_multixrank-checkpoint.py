#!/usr/bin/env python3
"""
Diagnose MultiXrank input size and estimate runtime.
Usage: python diagnose_multixrank.py --wdir /path/to/multixrank_input
"""
import argparse
from pathlib import Path
import pandas as pd
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wdir", required=True)
    args = parser.parse_args()
    wdir = Path(args.wdir)
    
    with open(wdir / "config.yml") as f:
        cfg = yaml.safe_load(f)
    
    layers = cfg["multiplex"]["gene_multiplex"]["layers"]
    seeds_path = wdir / cfg["seed"]
    
    with open(seeds_path) as f:
        n_seeds = sum(1 for l in f if l.strip())
    
    print(f"\n{'='*70}")
    print(f"MULTIXRANK INPUT DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"Config r (restart)  : {cfg.get('r', '?')}")
    print(f"Config delta        : {cfg['multiplex']['gene_multiplex'].get('delta', '?')}")
    print(f"Number of seeds     : {n_seeds:,}")
    print(f"Number of layers    : {len(layers)}")
    print(f"\n{'Layer':<55} {'Edges':>10} {'Nodes':>8} {'Size MB':>8}")
    print("-" * 85)
    
    total_edges = 0
    all_nodes = set()
    intra_count = 0
    inter_count = 0
    
    for layer_path in layers:
        full_path = wdir / layer_path
        name = Path(layer_path).name
        
        if not full_path.exists():
            print(f"  {name:<55} MISSING")
            continue
        
        size_mb = full_path.stat().st_size / 1e6
        df = pd.read_csv(full_path, sep='\t', header=None, 
                        names=['g1','g2'], dtype=str)
        n_edges = len(df)
        nodes = set(df['g1']) | set(df['g2'])
        all_nodes |= nodes
        total_edges += n_edges
        
        layer_type = "INTER" if "inter" in name.lower() else "intra"
        if layer_type == "INTER":
            inter_count += 1
        else:
            intra_count += 1
        
        print(f"  [{layer_type}] {name:<50} {n_edges:>10,} {len(nodes):>8,} {size_mb:>7.1f}")
    
    print("-" * 85)
    print(f"  {'TOTAL':<55} {total_edges:>10,} {len(all_nodes):>8,}")
    
    print(f"\n{'='*70}")
    print(f"LAYER BREAKDOWN")
    print(f"  Intra-layer files : {intra_count}")
    print(f"  Inter-layer files : {inter_count}")
    print(f"  Total             : {intra_count + inter_count}")
    
    print(f"\n{'='*70}")
    print(f"RUNTIME ESTIMATE")
    
    # Empirical: ~1M edge-iterations per second in MultiXrank pure Python
    iterations = 100
    edge_iters = total_edges * len(layers) * iterations
    est_seconds = edge_iters / 1_000_000
    
    print(f"  Estimated iterations     : {iterations}")
    print(f"  Total edge-iterations    : {edge_iters:>15,}")
    print(f"  Estimated runtime        : {est_seconds/60:.1f} minutes")
    print(f"  (assumes 1M edge-iter/s — run with profiler to calibrate)")
    
    if inter_count > 20:
        print(f"\n  ⚠ WARNING: {inter_count} inter-layer files is high.")
        print(f"    Consider merging into 1-3 files grouped by edge type.")
        intra_only_est = total_edges * intra_count * iterations / 1_000_000
        print(f"    With merged inter-layers: ~{intra_only_est/60:.1f} min estimated")
    
    if total_edges > 5_000_000:
        print(f"\n  ⚠ WARNING: {total_edges:,} total edges is very large.")
        print(f"    Consider filtering low-confidence edges.")

if __name__ == "__main__":
    main()