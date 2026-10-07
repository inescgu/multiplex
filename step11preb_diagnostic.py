#!/usr/bin/env python3
"""
Step 11b-diagnostic: cheap resolution PROFILE over a wide gamma range,
before committing to a full, expensive step 11b sweep.

WHY THIS EXISTS
---------------
Your actual step 11b sweep (--min-resolution 0.01 --max-resolution 10.0,
30 log-spaced gammas) produced nearly IDENTICAL partitions across almost
the entire range:

    gamma=0.085  -> 6,445 clusters (size >= 4)
    gamma=0.924  -> 6,458 clusters (size >= 4)
    gamma=10.0   -> 6,591 clusters (size >= 4)

...which is why persistence ended up concentrated at the maximum (99.5%
of clusters at persistence=30) once the windowed-neighbor fix was
verified correct: if the 30 partitions in the ensemble aren't meaningfully
different from each other, of course nearly every cluster looks
"persistent" across all of them. That's not a bug in persistence
computation -- it means 0.01-10.0 is sitting inside a resolution
"plateau" for THIS graph (23 mostly-separate, internally dense
per-cell-type PPI blocks bridged by comparatively sparse bipartite edges
can easily produce a wide plateau where the partition barely changes
until gamma crosses some threshold).

This script runs a SINGLE Leiden partition (no restarts, no omega sweep)
at each of a small number of gamma values spread across a MUCH wider
range than your production sweep, and plots n_communities vs. gamma on
log-log axes -- so you can see where (if anywhere) the partition actually
changes, and pick a --min-resolution/--max-resolution for the real sweep
that actually covers a transition instead of sitting inside a plateau.
Also supports comparing RBConfiguration against CPM, since CPM's
resolution parameter has an absolute density interpretation that can
behave differently across a graph combining very different edge densities
per cell-type block (as your --partition-type help text already flags).

This is deliberately cheap: single restart, no persistence/consensus
machinery, just "how many communities does Leiden find at this gamma."
Run this BEFORE re-submitting the expensive 24h step 11b job.

USAGE
-----
    python step11b_resolution_profile.py \
        --supra-graph 11_hidef_hetero/supra_graph.pkl \
        --output-prefix 11_hidef_hetero/resolution_profile \
        --min-resolution 1e-5 \
        --max-resolution 1000 \
        --n-resolutions 25 \
        --partition-types RB CPM
"""
import argparse
import pickle
import time
from pathlib import Path
from provenance import write_manifest
import igraph as ig
import leidenalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARTITION_CLASSES = {
    "RB": la.RBConfigurationVertexPartition,
    "CPM": la.CPMVertexPartition,
}


def profile_partition_type(g, gammas, partition_type, seed=0):
    PartitionClass = PARTITION_CLASSES[partition_type]
    rows = []
    t0 = time.time()
    for i, gamma in enumerate(gammas, 1):
        part = la.find_partition(g, PartitionClass, weights="weight",
                                  resolution_parameter=gamma, seed=seed, n_iterations=-1)
        n_comm = len(set(part.membership))
        rows.append({"partition_type": partition_type, "gamma": float(gamma),
                     "n_communities": n_comm, "quality": part.quality()})
        elapsed = time.time() - t0
        print(f"  [{partition_type} {i}/{len(gammas)}] gamma={gamma:.4g} -> "
              f"{n_comm:,} communities ({elapsed:.1f}s elapsed)")
    return rows


def plot_profile(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for ptype, sub in df.groupby("partition_type"):
        sub = sub.sort_values("gamma")
        ax.plot(sub["gamma"], sub["n_communities"], marker="o", label=ptype)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Resolution parameter (gamma, log scale)")
    ax.set_ylabel("# communities found (log scale)")
    ax.set_title("Resolution profile -- where does the partition actually change?")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved profile plot -> {out_path}")


def suggest_range(df):
    """Flag the steepest jumps in n_communities as candidate transition
    points -- a sensible resolution sweep should straddle these rather
    than sitting entirely inside a flat plateau."""
    print("\nTransition analysis (per partition type):")
    for ptype, sub in df.groupby("partition_type"):
        sub = sub.sort_values("gamma").reset_index(drop=True)
        if len(sub) < 2:
            continue
        log_gamma = np.log10(sub["gamma"].values)
        log_n = np.log10(sub["n_communities"].values.clip(min=1))
        d_log_n = np.abs(np.diff(log_n))
        d_log_gamma = np.diff(log_gamma)
        steepness = d_log_n / np.where(d_log_gamma == 0, np.nan, d_log_gamma)
        if len(steepness) == 0 or np.all(np.isnan(steepness)):
            print(f"  {ptype}: not enough variation to assess")
            continue
        top = np.nanargmax(steepness)
        print(f"  {ptype}: steepest change between gamma={sub['gamma'][top]:.4g} "
              f"({sub['n_communities'][top]:,} communities) and "
              f"gamma={sub['gamma'][top+1]:.4g} ({sub['n_communities'][top+1]:,} communities)")
        ratio = sub["n_communities"].max() / max(sub["n_communities"].min(), 1)
        print(f"  {ptype}: overall n_communities ratio across swept range = {ratio:.2f}x "
              f"({'looks flat -- widen the range' if ratio < 3 else 'shows real variation'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--supra-graph", required=True, help="Pickle from step 11a")
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--min-resolution", type=float, default=1e-5,
                     help="Much wider than your production sweep's 0.01, deliberately, to find "
                          "where the plateau actually ends")
    ap.add_argument("--max-resolution", type=float, default=1000.0)
    ap.add_argument("--n-resolutions", type=int, default=25,
                     help="Kept small -- this is a diagnostic, not the real sweep")
    ap.add_argument("--partition-types", nargs="+", choices=["RB", "CPM"], default=["RB", "CPM"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.supra_graph, "rb") as f:
        g = pickle.load(f)
    print(f"Loaded supra-graph: {g.vcount():,} nodes, {g.ecount():,} edges\n")

    gammas = np.geomspace(args.min_resolution, args.max_resolution, args.n_resolutions)
    print(f"Profiling {len(gammas)} gamma values from {gammas[0]:.4g} to {gammas[-1]:.4g} "
          f"x {len(args.partition_types)} partition type(s)\n")

    all_rows = []
    for ptype in args.partition_types:
        print(f"--- {ptype} ---")
        all_rows += profile_partition_type(g, gammas, ptype, seed=args.seed)
        print()

    df = pd.DataFrame(all_rows)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{out_prefix}.csv")
    df.to_csv(csv_path, index=False)
    plot_profile(df, Path(f"{out_prefix}.png"))
    suggest_range(df)

    print(f"\n✓ Saved -> {csv_path}")
    print("\nNext: pick --min-resolution/--max-resolution for the real step 11b sweep so it "
          "straddles a real transition region shown above, rather than sitting entirely inside "
          "a flat plateau -- then re-run steps 11b -> 11c -> 11e and check the persistence "
          "histogram again.")


if __name__ == "__main__":
    main()