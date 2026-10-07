#!/usr/bin/env python3
"""
Step 11b: Multi-resolution (and, optionally, multi-coupling) partition
sweep over the heterogeneous supra-graph built in step 11a, using leidenalg.

This is the direct generalization of HiDeF Phase I (log-spaced resolution
sweep + modularity maximization) to your heterogeneous multiplex: same
Reichardt-Bornholdt resolution parameter gamma HiDeF itself uses (so this
is directly comparable to your prior monoplex HiDeF run's resolution
range), plus a SECOND optional sweep axis, omega, the coupling weight
multiplier applied to every non-intra edge (bipartite LR/Reactome/
scored-LR/global_ppi-identity edges from step 2, plus any extra identity
edges from step 11a's --add-identity-coupling). This is exactly the
interlayer coupling term from Mucha et al.'s multislice modularity
(Science 2010): omega=0 clusters every cell type's PPI network
independently; omega=1 (the default here) treats biological cross-
cell-type edges the same as intra-layer PPI edges; omega>>1 increasingly
forces communities to agree across cell types.

The output is a raw ENSEMBLE of partitions -- one membership vector per
(gamma, omega, restart) triple. HiDeF's persistence/DAG logic (step 11c) is
documented as agnostic to how its input partitions were generated ("the
construction of DAG is independent of the upstream clustering algorithm"),
so this script's only job is to produce a good, diverse ensemble; step 11c
does the robustness/persistence analysis on top of it.
"""
import argparse
import pickle
import time
from pathlib import Path
from provenance import write_manifest
import numpy as np
import igraph as ig
import leidenalg as la

PARTITION_CLASSES = {
    "RB": la.RBConfigurationVertexPartition,
    "CPM": la.CPMVertexPartition,
}


def make_resolution_grid(minres: float, maxres: float, n: int) -> np.ndarray:
    return np.geomspace(minres, maxres, num=n)


def run_sweep(g: ig.Graph, gammas, omegas, n_restarts: int, partition_type: str, base_seed: int):
    etypes = np.array(g.es["etype"])
    base_weight = np.array(g.es["weight"], dtype=float)
    is_coupling = etypes != "intra"

    PartitionClass = PARTITION_CLASSES[partition_type]

    partitions = []
    node_names = g.vs["name"]
    total_runs = len(gammas) * len(omegas) * n_restarts
    run_i = 0
    t0 = time.time()

    for omega in omegas:
        w = base_weight.copy()
        w[is_coupling] *= omega
        g.es["w_scaled"] = w.tolist()

        for gamma in gammas:
            for r in range(n_restarts):
                run_i += 1
                seed = base_seed + run_i
                part = la.find_partition(
                    g, PartitionClass,
                    weights="w_scaled",
                    resolution_parameter=gamma,
                    seed=seed,
                    n_iterations=-1,
                )
                partitions.append({
                    "membership": list(part.membership),
                    "gamma": float(gamma),
                    "omega": float(omega),
                    "restart": r,
                    "n_communities": len(set(part.membership)),
                    "quality": part.quality(),
                })
                if run_i % 10 == 0 or run_i == total_runs:
                    elapsed = time.time() - t0
                    print(f"  [{run_i}/{total_runs}] gamma={gamma:.4g} omega={omega:.4g} "
                          f"restart={r} -> {partitions[-1]['n_communities']} communities "
                          f"({elapsed:.1f}s elapsed)")

    return partitions, node_names


def main():
    ap = argparse.ArgumentParser(description="Step 11b: resolution/coupling sweep over the "
                                               "heterogeneous supra-graph (step 11a output)")
    ap.add_argument("--supra-graph", required=True, help="Pickle from step 11a (igraph.Graph)")
    ap.add_argument("--output", required=True, help="Path to write the partition ensemble (.pkl)")
    ap.add_argument("--min-resolution", type=float, default=0.01)
    ap.add_argument("--max-resolution", type=float, default=10.0)
    ap.add_argument("--n-resolutions", type=int, default=30,
                     help="Number of log-spaced gamma values to sample (HiDeF-style pan-resolution sweep)")
    ap.add_argument("--omegas", type=float, nargs="+", default=[1.0],
                     help="Interlayer/cross-cell-type coupling weight(s) to sweep. Default: [1.0] "
                          "(bipartite/identity edges weighted the same as intra-layer PPI edges -- "
                          "equivalent to a plain resolution sweep on the full supra-graph). Pass e.g. "
                          "--omegas 0.1 0.5 1 2 5 for a full Mucha-style multislice sweep.")
    ap.add_argument("--n-restarts", type=int, default=1,
                     help="Random restarts per (gamma, omega) -- adds robustness signal for step 11c's "
                          "persistence analysis, at proportional runtime cost.")
    ap.add_argument("--partition-type", choices=["RB", "CPM"], default="RB",
                     help="RBConfigurationVertexPartition (matches HiDeF's own Reichardt-Bornholdt "
                          "modularity -- recommended for continuity with a prior monoplex HiDeF run) "
                          "or CPMVertexPartition (resolution has an absolute density interpretation, "
                          "sometimes preferable when edge density varies a lot across cell types).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    write_manifest(Path(args.output).parent, args, inputs=[args.supra_graph])
    with open(args.supra_graph, "rb") as f:
        g = pickle.load(f)
    print(f"Loaded supra-graph: {g.vcount():,} nodes, {g.ecount():,} edges\n")

    gammas = make_resolution_grid(args.min_resolution, args.max_resolution, args.n_resolutions)
    print(f"Resolution grid ({len(gammas)} values): {gammas[0]:.4g} ... {gammas[-1]:.4g}")
    print(f"Coupling (omega) grid: {args.omegas}")
    print(f"Restarts per setting: {args.n_restarts}")
    print(f"Partition type: {args.partition_type}\n")

    partitions, node_names = run_sweep(
        g, gammas, args.omegas, args.n_restarts, args.partition_type, args.seed
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"partitions": partitions, "node_names": node_names,
                     "cell_type": g.vs["cell_type"], "gene": g.vs["gene"]}, f)

    print(f"\n{'='*60}")
    print(f"Step 11b complete: {len(partitions)} partitions across "
          f"{len(gammas)} resolution(s) x {len(args.omegas)} coupling weight(s) x {args.n_restarts} restart(s)")
    print(f"  -> {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()