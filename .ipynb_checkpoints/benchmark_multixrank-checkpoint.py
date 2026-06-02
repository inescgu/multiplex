#!/usr/bin/env python3
"""
benchmark_multixrank.py
Runs MultiXrank on a small subset to measure actual iterations/second.
Then gives a calibrated runtime estimate for your full multiplex.
"""
import argparse
import time
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import multixrank

def sample_layers(wdir: Path, config_path: Path, 
                  n_layers: int = 10, 
                  max_edges_per_layer: int = 5000,
                  output_dir: Path = None) -> Path:
    """
    Create a small benchmark config using a subset of layers
    and a random sample of edges from each.
    """
    if output_dir is None:
        output_dir = wdir / "_benchmark_tmp"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "multiplex" / "gene_multiplex").mkdir(parents=True, exist_ok=True)
    (output_dir / "seeds").mkdir(exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    all_layers = cfg["multiplex"]["gene_multiplex"]["layers"]
    
    # Pick n_layers evenly spaced (to sample small + large)
    indices = np.linspace(0, len(all_layers)-1, 
                          min(n_layers, len(all_layers)), 
                          dtype=int)
    sampled = [all_layers[i] for i in indices]

    new_layer_paths = []
    all_nodes = set()

    for layer_path in sampled:
        src = wdir / layer_path
        if not src.exists():
            continue
        df = pd.read_csv(src, sep="\t", header=None, 
                        names=["g1","g2"], dtype=str)
        # Sample edges
        if len(df) > max_edges_per_layer:
            df = df.sample(n=max_edges_per_layer, random_state=42)
        
        dest_name = Path(layer_path).name
        dest = output_dir / "multiplex" / "gene_multiplex" / dest_name
        df.to_csv(dest, sep="\t", index=False, header=False)
        new_layer_paths.append(
            str(Path("multiplex") / "gene_multiplex" / dest_name)
        )
        all_nodes |= set(df["g1"]) | set(df["g2"])

    # Write seeds (use nodes that exist in benchmark graph)
    seeds_src = wdir / cfg["seed"]
    with open(seeds_src) as f:
        all_seeds = [l.strip() for l in f if l.strip()]
    
    seeds_in = [s for s in all_seeds if s in all_nodes][:50]  # cap at 50
    if not seeds_in:
        seeds_in = list(all_nodes)[:10]  # fallback
    
    with open(output_dir / "seeds" / "seeds.txt", "w") as f:
        for s in seeds_in:
            f.write(s + "\n")

    n = len(new_layer_paths)
    tau = round(1.0 / n, 6)
    bench_cfg = {
        "seed": "seeds/seeds.txt",
        "self_loops": 0,
        "r": cfg.get("r", 0.7),
        "multiplex": {
            "gene_multiplex": {
                "layers": new_layer_paths,
                "graph_type": ["00"] * n,
                "delta": cfg["multiplex"]["gene_multiplex"].get("delta", 0.5),
                "tau": [tau] * n,
            }
        },
    }
    bench_config = output_dir / "config.yml"
    with open(bench_config, "w") as f:
        yaml.safe_dump(bench_cfg, f, default_flow_style=False, sort_keys=False)

    total_bench_edges = sum(
        len(pd.read_csv(output_dir / p, sep="\t", header=None))
        for p in new_layer_paths
        if (output_dir / p).exists()
    )

    print(f"Benchmark config: {n} layers, "
          f"{total_bench_edges:,} edges, "
          f"{len(seeds_in)} seeds")
    print(f"Written to: {output_dir}")
    
    return output_dir, total_bench_edges

def run_benchmark(bench_wdir: Path, total_bench_edges: int,
                  full_total_edges: int, full_n_layers: int):
    """Run MultiXrank on benchmark and extrapolate to full problem."""
    
    print("\nRunning benchmark RWR...")
    t0 = time.time()
    
    mxr = multixrank.Multixrank(
        config=str(bench_wdir / "config.yml"),
        wdir=str(bench_wdir)
    )
    result = mxr.random_walk_rank()
    
    elapsed = time.time() - t0
    print(f"Benchmark completed in {elapsed:.1f}s")

    # Count actual iterations from result if possible
    # MultiXrank doesn't expose iteration count directly,
    # so we estimate from timing
    
    with open(bench_wdir / "config.yml") as f:
        cfg = yaml.safe_load(f)
    bench_layers = len(cfg["multiplex"]["gene_multiplex"]["layers"])

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Benchmark layers     : {bench_layers}")
    print(f"  Benchmark edges      : {total_bench_edges:,}")
    print(f"  Wall-clock time      : {elapsed:.2f}s")
    
    # Actual edge-iterations per second
    # Assume ~50 iterations (typical for r=0.7-0.85)
    assumed_iters = 50
    actual_edge_iters = total_bench_edges * bench_layers * assumed_iters
    actual_speed = actual_edge_iters / elapsed
    
    print(f"\n  Assumed RWR iterations: {assumed_iters}")
    print(f"  Edge-iterations done  : {actual_edge_iters:,}")
    print(f"  Actual speed          : {actual_speed/1e6:.2f}M edge-iter/s")
    print(f"  (diagnostic assumed 1.0M edge-iter/s)")

    # Extrapolate to full problem
    full_edge_iters = full_total_edges * full_n_layers * assumed_iters
    est_full_seconds = full_edge_iters / actual_speed
    
    print(f"\n{'='*60}")
    print(f"FULL RUN ESTIMATE (calibrated)")
    print(f"{'='*60}")
    print(f"  Full total edges     : {full_total_edges:,}")
    print(f"  Full layers          : {full_n_layers}")
    print(f"  Full edge-iterations : {full_edge_iters:,}")
    print(f"  Estimated runtime    : {est_full_seconds/60:.0f} minutes")
    print(f"  Estimated runtime    : {est_full_seconds/3600:.1f} hours")
    
    # Targets
    targets = {
        "< 30 min (ideal)":    30*60,
        "< 2 hours (ok)":     2*3600,
        "< 8 hours (slurm)":  8*3600,
        "< 24 hours (slurm)": 24*3600,
    }
    print(f"\n  To achieve each target, max total edges:")
    for label, target_s in targets.items():
        max_edges = (target_s * actual_speed) / (full_n_layers * assumed_iters)
        reduction = max_edges / full_total_edges * 100
        print(f"    {label:<25} "
              f"max {max_edges:>12,.0f} edges "
              f"({reduction:.0f}% of current)")

    return actual_speed, est_full_seconds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wdir", required=True)
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--n-layers", type=int, default=10,
                        help="Layers to include in benchmark (default 10)")
    parser.add_argument("--max-edges", type=int, default=5000,
                        help="Max edges per layer in benchmark (default 5000)")
    args = parser.parse_args()

    wdir = Path(args.wdir)
    config_path = wdir / args.config

    # Get full problem size
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    full_layers = cfg["multiplex"]["gene_multiplex"]["layers"]
    full_n_layers = len(full_layers)
    full_total_edges = sum(
        len(pd.read_csv(wdir / p, sep="\t", header=None))
        for p in full_layers
        if (wdir / p).exists()
    )
    print(f"Full problem: {full_n_layers} layers, {full_total_edges:,} edges")

    # Build and run benchmark
    bench_wdir, bench_edges = sample_layers(
        wdir, config_path,
        n_layers=args.n_layers,
        max_edges_per_layer=args.max_edges,
    )
    
    actual_speed, est_seconds = run_benchmark(
        bench_wdir, bench_edges,
        full_total_edges, full_n_layers
    )

    print(f"\nNext steps based on estimate:")
    if est_seconds < 2*3600:
        print("  ✓ Should be fine — run directly on SLURM")
    elif est_seconds < 8*3600:
        print("  ⚠ Long but feasible — request 8h SLURM job")
        print("  Consider: r=0.85 to reduce iterations ~40%")
    elif est_seconds < 24*3600:
        print("  ⚠ Very long — need edge filtering")
        print("  See edge confidence filtering below")
    else:
        print("  ✗ Too slow — aggressive filtering required")

if __name__ == "__main__":
    main()