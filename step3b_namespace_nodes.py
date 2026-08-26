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
    args = ap.parse_args()
    wdir = Path(args.wdir)

    mul_root = wdir / "multiplex"
    cell_types = sorted(p.name for p in mul_root.iterdir() if p.is_dir())
    print(f"Found {len(cell_types)} cell-type multiplex(es)\n")

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
    per_seed_hits = {}
    for s in base_seeds:
        hits = [ct for ct, genes in gene_sets.items() if s in genes]
        per_seed_hits[s] = hits
        expanded.update(namespace(s, ct) for ct in hits)

    with open(current_seeds_path, "w") as f:
        for s in sorted(expanded):
            f.write(s + "\n")

    zero_hit = [s for s, hits in per_seed_hits.items() if not hits]
    print(f"\nSeeds: {len(base_seeds)} plain -> {len(expanded)} namespaced seed entries")
    if zero_hit:
        print(f"  \u26a0 {len(zero_hit)} seed genes matched NO cell-type layer "
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