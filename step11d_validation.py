#!/usr/bin/env python3
"""
Step 11d (optional validation): Cross-check the persistent
communities from step 11c against an independent, flow-based clustering of
the same heterogeneous supra-graph using Infomap.

CHANGELOG
---------
- v3: --communities is now OPTIONAL. This script can read the .nodes file
  from step 11c Phase 2 directly (via --nodes-file), the same parser used
  by step 11e, instead of requiring a separately-recovered
  heterogeneous_hidef_communities_FIXED.pkl. That recovery pickle was tied
  to a specific (now superseded) run's extract_communities() bug fix --
  every time you re-run the pipeline with new sweep parameters (e.g. the
  CPM-based resolution sweep replacing the original flat-persistence RB
  sweep), regenerating that pickle by hand is an easy step to forget,
  which silently validates STALE clusters against a fresh Infomap run
  instead of erroring out. Reading .nodes directly removes that failure
  mode. --communities is kept for backward compatibility with any older
  recovery pickles you still want to cross-check.
- --min-size (default 2): entries with size < min-size are dropped before
  computing purity, whichever source they came from. Without this, a
  large number of legitimately-trivial single-node leaf entries dominate
  the report and drive purity to ~1.0 trivially (a singleton is always
  "pure" against any module assignment), which is meaningless as a
  cross-validation of the actual hierarchy against Infomap.
- Report records how many communities were dropped by the size filter,
  and both the filtered and unfiltered counts.

SIMPLIFICATION, stated up front: rather than reconstructing the full
interconnected/multilayer Infomap formulation (De Domenico et al. 2015,
Phys. Rev. X 5:011027) via explicit state-node/multilayer link calls --
whose exact Python API differs across `infomap` package versions and would
need verifying against whatever version you have installed -- this runs
standard flow-based Infomap directly on the SAME weighted supra-graph built
in step 11a. Every "<cell_type>::GENE" node is still kept distinct and every
intra-layer PPI + bipartite/identity coupling edge is still included --
nothing is flattened or merged across cell types -- this just uses
Infomap's ordinary single-network solver rather than its multilayer-
specific link API. If you later want the fully explicit multilayer
formulation (e.g. to weight intra- vs inter-layer flow differently, or
model a relax-rate), look at `add_multilayer_intra_link` /
`add_multilayer_inter_link` (or your installed version's equivalents) in
the infomap docs and adapt run_infomap() below.

Reports:
  1. How many modules Infomap found on the supra-graph.
  2. Purity of each persistent HiDeF community (step 3c output, filtered
     to size >= --min-size) against Infomap's module assignment -- what
     fraction of a community's members fall into Infomap's single most
     common module for that community.
  3. NMI between Infomap's partition and whichever leidenalg partition in
     the step 3b ensemble has the closest number of communities -- a rough
     single-number concordance score between the two engines.
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

try:
    from infomap import Infomap
except ImportError:
    print("ERROR: could not import infomap -- pip install infomap")
    raise

from sklearn.metrics import normalized_mutual_info_score


def run_infomap(g, weight_attr="weight", seed=1):
    # seed=1, not 0: infomap 2.9.2 has a bug in its args-string bridge that
    # fails to parse "--seed 0" specifically (fixed in later releases,
    # which require Python >=3.11 and aren't installable here). See step
    # 11d troubleshooting notes.
    im = Infomap(two_level=True, seed=seed, silent=True)

    has_weight = weight_attr in g.es.attributes()
    for e in g.es:
        u, v = e.tuple
        w = e[weight_attr] if has_weight else 1.0
        im.add_link(u, v, w)

    im.run()

    membership = np.full(g.vcount(), -1, dtype=int)
    for node in im.tree:
        if node.is_leaf:
            membership[node.node_id] = node.module_id
    n_unassigned = int((membership == -1).sum())
    if n_unassigned:
        print(f"  ⚠ {n_unassigned} node(s) had no module assignment (isolated in the network?)")
    return membership


def closest_leidenalg_partition(partitions, target_n_modules):
    best, best_diff = None, None
    for p in partitions:
        diff = abs(p["n_communities"] - target_n_modules)
        if best_diff is None or diff < best_diff:
            best, best_diff = p, diff
    return best


def community_purity(members, node_name_to_idx, infomap_membership):
    idxs = [node_name_to_idx[m] for m in members if m in node_name_to_idx]
    if not idxs:
        return None, None
    modules = infomap_membership[idxs]
    vals, counts = np.unique(modules, return_counts=True)
    top_module = vals[np.argmax(counts)]
    purity = counts.max() / len(idxs)
    return purity, int(top_module)


# ============================================================================
# COMMUNITY LOADING -- either the recovered pickle (legacy) or .nodes directly
# ============================================================================

def load_communities_from_pickle(path: Path):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    communities = {}
    for cid, c in raw.items():
        members = c["members"]
        communities[str(cid)] = {
            "members": list(members),
            "size": c.get("size", len(members)),
            "persistence": c.get("persistence", None),
        }
    print(f"Loaded {len(communities):,} communities from recovery pickle {path}\n")
    return communities


def load_communities_from_nodes_file(nodes_path: Path, node_names: list):
    """Same parser as step 11e: community, size, members, persistence
    (tab-separated, no header). 'members' may be whitespace- or comma-
    separated, and may contain either 'cell_type::gene' names or integer
    vertex indices -- both are handled. Rows with an empty/NaN members
    field are skipped rather than crashing downstream."""
    import pandas as pd

    df = pd.read_csv(nodes_path, sep="\t", header=None,
                      names=["cluster_id", "size", "members", "persistence"],
                      dtype={"members": str})
    communities = {}
    n_skipped_empty = 0
    for _, row in df.iterrows():
        raw_field = row["members"]
        if pd.isna(raw_field) or not str(raw_field).strip():
            n_skipped_empty += 1
            continue
        raw = [tok for tok in re.split(r"[,\s]+", str(raw_field).strip()) if tok]
        if not raw:
            n_skipped_empty += 1
            continue

        looks_like_indices = all(tok.lstrip("-").isdigit() for tok in raw[:5])
        if looks_like_indices:
            try:
                members = [node_names[int(tok)] for tok in raw]
            except (IndexError, ValueError):
                members = raw
        else:
            members = raw

        communities[str(row["cluster_id"])] = {
            "members": members,
            "size": int(row["size"]),
            "persistence": float(row["persistence"]),
        }

    print(f"Loaded {len(communities):,} communities from {nodes_path}")
    if n_skipped_empty:
        print(f"  ⚠ skipped {n_skipped_empty} row(s) with empty/NaN members field")
    print()
    return communities


def filter_trivial_communities(communities: dict, min_size: int):
    """Drop terminal-leaf / trivially small entries before computing
    purity. See CHANGELOG at top of file for why this matters."""
    n_before = len(communities)
    filtered = {cid: c for cid, c in communities.items()
                if c.get("size", len(c.get("members", []))) >= min_size}
    n_after = len(filtered)
    print(f"Community size filter (--min-size {min_size}): "
          f"{n_before:,} -> {n_after:,} communities "
          f"({n_before - n_after:,} dropped as trivial)")
    if n_after == 0:
        print("  ⚠ Filter removed everything -- check --nodes-file/--communities points "
              "at the right output and --min-size isn't too strict.")
    return filtered, n_before, n_after


def main():
    ap = argparse.ArgumentParser(description="Step 3d: cross-check step 3c's persistent "
                                               "communities against Infomap on the same supra-graph")
    ap.add_argument("--supra-graph", required=True, help="Pickle from step 3a")
    ap.add_argument("--ensemble", required=True, help="Pickle from step 3b (for the closest-N comparison)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--nodes-file",
                      help="hidef_finder .nodes output from step 11c Phase 2 -- read directly, "
                           "no recovery-pickle step needed (preferred)")
    src.add_argument("--communities",
                      help="Legacy: a {comm_id: {'members', 'size', 'persistence'}} recovery "
                           "pickle from an earlier extract_communities() bug-fix pass")
    ap.add_argument("--min-size", type=int, default=2,
                     help="Drop communities smaller than this before computing purity "
                          "(default 2 -- excludes trivial single-node leaf entries)")
    ap.add_argument("--output", required=True, help="Where to write the cross-check report (.json)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    with open(args.supra_graph, "rb") as f:
        g = pickle.load(f)
    with open(args.ensemble, "rb") as f:
        ensemble = pickle.load(f)

    if args.nodes_file:
        communities = load_communities_from_nodes_file(Path(args.nodes_file), g.vs["name"])
    else:
        communities = load_communities_from_pickle(Path(args.communities))

    communities, n_before_filter, n_after_filter = filter_trivial_communities(
        communities, args.min_size
    )

    print(f"\nRunning Infomap on supra-graph ({g.vcount():,} nodes, {g.ecount():,} edges) ...")
    infomap_membership = run_infomap(g, seed=args.seed)
    n_infomap_modules = len(set(infomap_membership.tolist()))
    print(f"  Infomap found {n_infomap_modules} modules\n")

    node_name_to_idx = {n: i for i, n in enumerate(g.vs["name"])}

    print("Purity of persistent HiDeF communities against Infomap modules "
          f"(size >= {args.min_size} only):")
    purities = []
    per_community = {}
    for cid, c in communities.items():
        purity, top_module = community_purity(c["members"], node_name_to_idx, infomap_membership)
        if purity is not None:
            purities.append(purity)
            per_community[cid] = {"purity": purity, "top_infomap_module": top_module, "size": c["size"]}
    if purities:
        print(f"  mean purity: {np.mean(purities):.3f}  median: {np.median(purities):.3f}  "
              f"(n={len(purities)} communities)\n")
    else:
        print("  ⚠ no communities had any members present in the supra-graph -- check inputs match.\n")

    closest = closest_leidenalg_partition(ensemble["partitions"], n_infomap_modules)
    nmi = normalized_mutual_info_score(closest["membership"], infomap_membership.tolist())
    print(f"Closest leidenalg partition: gamma={closest['gamma']:.4g}, omega={closest['omega']:.4g}, "
          f"{closest['n_communities']} communities")
    print(f"NMI(Infomap, closest leidenalg partition) = {nmi:.3f}\n")

    report = {
        "source": "nodes_file" if args.nodes_file else "communities_pickle",
        "min_size_filter": args.min_size,
        "n_communities_before_filter": n_before_filter,
        "n_communities_after_filter": n_after_filter,
        "n_infomap_modules": n_infomap_modules,
        "mean_community_purity": float(np.mean(purities)) if purities else None,
        "median_community_purity": float(np.median(purities)) if purities else None,
        "closest_leidenalg_partition": {
            "gamma": closest["gamma"], "omega": closest["omega"],
            "n_communities": closest["n_communities"],
        },
        "nmi_infomap_vs_closest_leidenalg": float(nmi),
        "per_community": per_community,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved cross-check report -> {out_path}")


if __name__ == "__main__":
    main()