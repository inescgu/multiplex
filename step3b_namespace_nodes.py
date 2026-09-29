#!/usr/bin/env python3
"""
Step 3.5 -- RUN THIS AFTER STEP 3, NOT BEFORE.

Why after, not before
----------------------
Step 3 rebuilds seeds/seeds.txt (filtered against celltype_gene_sets.json,
which uses plain gene symbols) and gene_reference_graph.pkl EVERY time it
runs. If you namespace nodes before step 3, step 3 just overwrites your
namespaced seeds.txt with a fresh plain-gene version -- which is exactly
what caused "0/372 seeds in LCC" / "seed nodes not found in network".

Correct order: step1 -> step2 -> step3 -> step2.5 (this script) -> step4.
If you ever re-run step 3 for any reason, you MUST re-run this script
again afterward, or your seeds/reference-graph will silently revert to
plain (unnamespaced) form again.

What this does
---------------
1. Renames every node GENE -> "<celltype>::GENE" in each
   multiplex/<ct>/ppi_layer.tsv and in every bipartite/<a>__<b>.tsv.
2. Re-expands the (plain-gene) seeds.txt that step 3 just produced into
   namespaced seeds: one "<celltype>::GENE" entry per cell type where that
   gene is actually a node -- not just one.
3. Rebuilds gene_reference_graph.pkl / gene_ppi_reference_graph.pkl from
   the now-namespaced files, so step 4's LCC sanity check uses node names
   that actually match the namespaced network and seeds.

Idempotent: originals are preserved as *.tsv.orig on first run, and
re-running re-derives from those backups.

NEW: --exclude-seed-celltypes / --exclude-ppi-seeds
------------------------------------------------------
Previously, step (2) above (re-expanding plain seeds into namespaced
seeds) had no exclusion logic at all: a seed gene present in ANY cell
type's gene set -- including a --base-ppi backbone like "global_ppi",
added in step 1 and wired to every other cell type via identity bipartite
edges in step 2 -- got a "<celltype>::GENE" seed entry for THAT cell type
too. Since a backbone like global_ppi is by construction comprehensive,
this meant nearly every seed silently grew a "global_ppi::GENE" seed
entry, with no way to stop it here -- the --exclude-ppi-seeds /
--exclude-seed-celltypes flags on step 5 (preflight) and step 6 (build
disease module) only filter THEIR OWN in-memory seed lists; they never
touch seeds.txt on disk, so they have zero effect on the actual MultiXrank
random walk in step 4, which just reads seeds.txt as-is. In practice this
meant a pipeline run with --exclude-ppi-seeds passed at steps 5/6 could
still have had the backbone anchoring the RWR itself in step 4, silently.

This flag closes that gap at the source: pass

    --exclude-seed-celltypes global_ppi
    --exclude-ppi-seeds                 # shorthand for the line above

and the namespaced expansion in step (2) will skip creating a
"<celltype>::GENE" seed entry for any excluded cell type, for every seed,
before seeds.txt is ever written to disk -- so step 4's random walk never
sees those entries as seeds in the first place. The cell type's NODES and
its identity bipartite edges are untouched (this only affects the SEED
list): global_ppi (or whatever cell type you exclude) still fully
participates in the network and can still be reached as a bridging
resource downstream, it just never gets restart probability mass of its
own. This mirrors the "seed-only" exclusion semantics already used by
step 5's --exclude-seed-celltypes/--exclude-ppi-seeds and step 6's option
of the same name -- now it's enforced where it actually matters, before
step 4 runs, not just in the two scripts that run after.

If you want the excluded cell type gone from the network entirely (not
just from the seed list), that's step 5/6's separate --exclude-celltypes /
--exclude-ppi-layer option -- this script does not touch multiplex or
bipartite structure, only which namespaced entries seeds.txt gets.
"""
import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

SEP = "::"  # keep in sync with the split used in step4's parse_multixrank_result


def namespace(gene: str, ct: str) -> str:
    return f"{ct}{SEP}{gene}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wdir", required=True)
    ap.add_argument("--exclude-seed-celltypes", nargs="+", default=[], metavar="CELLTYPE",
                     help="Cell type(s) (the '<celltype>::' namespace prefix) to skip when "
                          "re-expanding plain seeds into namespaced seeds -- i.e. these cell "
                          "types will NEVER get a '<celltype>::GENE' seed entry, even if the "
                          "gene is present in that cell type's gene set. The cell type's nodes "
                          "and bipartite edges are untouched; only seeds.txt is affected. "
                          "Example: --exclude-seed-celltypes global_ppi. Case-insensitive.")
    ap.add_argument("--exclude-ppi-seeds", action="store_true",
                     help="Shorthand for --exclude-seed-celltypes global_ppi -- stops the "
                          "base/comprehensive PPI pseudo-cell-type added in step 1 via "
                          "--base-ppi from ever anchoring the MultiXrank random walk as a "
                          "seed, while leaving it fully in place as a connectivity resource.")
    args = ap.parse_args()
    wdir = Path(args.wdir)

    exclude_seed_celltypes = list(args.exclude_seed_celltypes)
    if args.exclude_ppi_seeds and "global_ppi" not in {c.lower() for c in exclude_seed_celltypes}:
        exclude_seed_celltypes.append("global_ppi")
    exclude_seed_ct_upper = {c.strip().upper() for c in exclude_seed_celltypes}
    if exclude_seed_ct_upper:
        print(f"Seed-only exclusion active: {sorted(exclude_seed_ct_upper)} will NEVER receive "
              f"a namespaced seed entry (nodes/bipartite edges are unaffected)\n")

    mul_root = wdir / "multiplex"
    cell_types = sorted(p.name for p in mul_root.iterdir() if p.is_dir())
    print(f"Found {len(cell_types)} cell-type multiplex(es)\n")

    if exclude_seed_ct_upper:
        unmatched = exclude_seed_ct_upper - {ct.upper() for ct in cell_types}
        if unmatched:
            print(f"  \u26a0 --exclude-seed-celltypes/--exclude-ppi-seeds named {sorted(unmatched)}, "
                  f"which doesn't match any discovered multiplex folder name ({cell_types}) -- "
                  f"check spelling/case; as given, it will simply never match and have no effect.\n")

    # ------------------------------------------------------------------
    # 1. Namespace nodes in each ppi_layer.tsv; remember gene sets per ct
    # ------------------------------------------------------------------
    gene_sets: dict[str, set[str]] = {}
    for ct in cell_types:
        layer_path = mul_root / ct / "ppi_layer.tsv"
        backup = layer_path.with_suffix(".tsv.orig")
        if not backup.exists():
            layer_path.rename(backup)
        df = pd.read_csv(backup, sep="\t", header=None, names=["g1", "g2"], dtype=str)
        genes = set(df["g1"]) | set(df["g2"])
        gene_sets[ct] = genes
        df["g1"] = df["g1"].apply(lambda g, ct=ct: namespace(g, ct))
        df["g2"] = df["g2"].apply(lambda g, ct=ct: namespace(g, ct))
        df.to_csv(layer_path, sep="\t", header=False, index=False)
        print(f"  {ct:35s}: {len(genes):5,} genes -> namespaced")

    # ------------------------------------------------------------------
    # 2. Namespace bipartite edge files
    # ------------------------------------------------------------------
    manifest_path = wdir / "bipartite_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    for m in manifest:
        bip_path = wdir / m["file"]
        backup = bip_path.with_suffix(".tsv.orig")
        if not backup.exists():
            bip_path.rename(backup)
        df = pd.read_csv(backup, sep="\t", header=None, names=["g1", "g2"], dtype=str)
        df["g1"] = df["g1"].apply(lambda g, ct=m["source"]: namespace(g, ct))
        df["g2"] = df["g2"].apply(lambda g, ct=m["target"]: namespace(g, ct))
        df.to_csv(bip_path, sep="\t", header=False, index=False)
    print(f"\n  Namespaced {len(manifest)} bipartite file(s)")

    # ------------------------------------------------------------------
    # 3. Re-expand seeds.txt (currently plain, freshly written by step 3)
    # ------------------------------------------------------------------
    seeds_dir = wdir / "seeds"
    current_seeds_path = seeds_dir / "seeds.txt"
    plain_backup_path = seeds_dir / "seeds_plain_from_step3.txt"

    # Always re-derive from a fresh copy of whatever step 3 just wrote,
    # so re-running this script after re-running step 3 works correctly.
    # NOTE: this also means re-running this script with a DIFFERENT
    # --exclude-seed-celltypes value (or none at all) correctly re-expands
    # from the same plain base each time -- exclusion is applied fresh at
    # expansion time below, never baked into the plain backup itself.
    with open(current_seeds_path) as f:
        current_lines = [s.strip() for s in f if s.strip()]
    already_namespaced = any(SEP in s for s in current_lines)
    if already_namespaced:
        if not plain_backup_path.exists():
            raise SystemExit(
                "seeds.txt already looks namespaced but no seeds_plain_from_step3.txt "
                "backup exists -- refusing to guess. If you re-ran step 3, its fresh "
                "plain seeds.txt should be in place before running this script."
            )
        with open(plain_backup_path) as f:
            base_seeds = [s.strip().upper() for s in f if s.strip()]
    else:
        base_seeds = [s.strip().upper() for s in current_lines]
        with open(plain_backup_path, "w") as f:
            for s in base_seeds:
                f.write(s + "\n")

    expanded = set()
    per_seed_hits = {}          # ct's the seed matched AND was expanded into
    per_seed_excluded_hits = {} # ct's the seed matched but was excluded from expansion
    for s in base_seeds:
        all_hits = [ct for ct, genes in gene_sets.items() if s in genes]
        kept_hits = [ct for ct in all_hits if ct.upper() not in exclude_seed_ct_upper]
        excluded_hits = [ct for ct in all_hits if ct.upper() in exclude_seed_ct_upper]
        per_seed_hits[s] = kept_hits
        if excluded_hits:
            per_seed_excluded_hits[s] = excluded_hits
        expanded.update(namespace(s, ct) for ct in kept_hits)

    with open(current_seeds_path, "w") as f:
        for s in sorted(expanded):
            f.write(s + "\n")

    zero_hit = [s for s, hits in per_seed_hits.items() if not hits]
    n_excluded_entries = sum(len(v) for v in per_seed_excluded_hits.values())
    print(f"\nSeeds: {len(base_seeds)} plain -> {len(expanded)} namespaced seed entries")
    if exclude_seed_ct_upper:
        print(f"  [exclude-seed-celltypes] Skipped {n_excluded_entries} would-be seed "
              f"entr{'y' if n_excluded_entries == 1 else 'ies'} under {sorted(exclude_seed_ct_upper)} "
              f"across {len(per_seed_excluded_hits)} seed gene(s) -- those cell type(s) remain "
              f"in the network and can still be reached while bridging, they just never start "
              f"with seed restart mass of their own.")
        newly_zero_hit = [s for s in zero_hit if s in per_seed_excluded_hits]
        if newly_zero_hit:
            print(f"  \u26a0 {len(newly_zero_hit)} seed gene(s) matched ONLY excluded cell type(s) "
                  f"and now have NO namespaced seed entry at all (fell out of the seed list "
                  f"entirely, not just out of that one cell type): {newly_zero_hit[:10]}")
    if zero_hit:
        print(f"  \u26a0 {len(zero_hit)} seed genes matched NO (non-excluded) cell-type layer "
              f"(will contribute nothing): {zero_hit[:10]}")
    print(f"  Saved -> {current_seeds_path}")

    # ------------------------------------------------------------------
    # 4. Rebuild reference graphs from the NOW-namespaced files
    #    (mirrors step 3's reference-graph block, but must run after
    #    namespacing so node names in the pkl match seeds.txt and the
    #    actual MultiXrank network)
    # ------------------------------------------------------------------
    print("\nRebuilding reference graphs from namespaced files ...")
    ref = nx.Graph()
    ppi_only = nx.Graph()

    for ct in cell_types:
        layer_path = mul_root / ct / "ppi_layer.tsv"
        edges_df = pd.read_csv(layer_path, sep="\t", header=None, names=["g1", "g2"], dtype=str)
        for a, b in zip(edges_df["g1"], edges_df["g2"]):
            if a == b:
                continue
            if not ref.has_edge(a, b):
                ref.add_edge(a, b, cell_types=set(), inter_cell_types=set(), edge_type=None)
            ref[a][b]["cell_types"].add(ct)
            ref[a][b]["edge_type"] = "both" if ref[a][b]["edge_type"] == "inter" else "ppi"
            ppi_only.add_edge(a, b)

    for m in manifest:
        edges_df = pd.read_csv(wdir / m["file"], sep="\t", header=None, names=["g1", "g2"], dtype=str)
        pair_label = (m["source"], m["target"])
        for a, b in zip(edges_df["g1"], edges_df["g2"]):
            if a == b:
                continue
            if not ref.has_edge(a, b):
                ref.add_edge(a, b, cell_types=set(), inter_cell_types=set(), edge_type=None)
            ref[a][b]["inter_cell_types"].add(pair_label)
            ref[a][b]["edge_type"] = "both" if ref[a][b]["edge_type"] == "ppi" else "inter"

    with open(wdir / "gene_reference_graph.pkl", "wb") as f:
        pickle.dump(ref, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(wdir / "gene_ppi_reference_graph.pkl", "wb") as f:
        pickle.dump(ppi_only, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Reference graph : {ref.number_of_nodes():,} nodes, {ref.number_of_edges():,} edges")
    print(f"  PPI-only graph  : {ppi_only.number_of_nodes():,} nodes, {ppi_only.number_of_edges():,} edges")

    print("\nDone. Ready for step 4. Do NOT re-run step 3 without re-running this "
          "script again afterward.")


if __name__ == "__main__":
    main()