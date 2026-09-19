#!/usr/bin/env python3
"""
Step 7d (leftover, standalone): quick seed-vs-layer sanity report you can run
right after step 1 -- before step4/step5/step6 have produced anything.

Almost everything the original draft did is now
better homes:
  - per-layer topology, pairwise layer similarity, union graph, hub genes,
    network pictures                -> folded into step7 (step7_layer_topology.py)
  - seed presence/degree in every layer, final-module cross-reference,
    expression specificity          -> folded into step6b (step6b_module_diagnostics.py)

What's left is genuinely a different use case: a fast, standalone check of
how your seed genes sit in the raw per-cell-type PPI layers, usable the
moment step 1 has produced multiplex/<ct>/ppi_layer.tsv + celltype_gene_sets.json
-- long before MultiXrank has run or a disease module exists. Useful as an
early smoke test (e.g. right after changing --layers-csv/--merge-by in
step 1, or before deciding on --exclude-seeds for step 6) without waiting
on step4/step5/step6.

If you already have a disease module and want the fuller picture (final-
module cross-reference, expression specificity), use step6b instead -- this
script does not read disease_module.csv or cell_type_markers.csv at all.

Outputs (under --output-dir):
  seed_layer_presence.csv     long format: gene, cell_type, present, degree
  seed_layer_summary.csv      one row per seed gene: n_layers_present,
                               total_degree_across_layers, dominant_layer,
                               dominant_layer_degree
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def _gene_of(node: str) -> str:
    s = str(node).strip()
    return s.rsplit("::", 1)[-1].upper() if "::" in s else s.upper()


def load_seeds(wdir: Path) -> list[str]:
    orig = wdir / "seeds" / "seeds_original.txt"
    plain = wdir / "seeds" / "seeds_plain_from_step3.txt"
    fallback = wdir / "seeds" / "seeds.txt"
    seed_csv = None  # allow passing a raw seed CSV directly too, see main()
    for path in (orig, plain):
        if path.exists():
            genes = sorted({line.strip().upper() for line in open(path) if line.strip()})
            print(f"Loaded {len(genes)} seed genes from {path}")
            return genes
    genes = sorted({_gene_of(line) for line in open(fallback) if line.strip()})
    print(f"Loaded {len(genes)} seed genes from {fallback} (namespace stripped, deduped)")
    return genes


def load_gene_sets(path: str) -> dict[str, set]:
    with open(path) as f:
        raw = json.load(f)
    return {ct: {_gene_of(g) for g in genes} for ct, genes in raw.items()}


def compute_layer_degrees(wdir: Path, gene_sets: dict, genes_of_interest: set) -> dict[str, dict[str, int]]:
    degrees: dict[str, dict[str, int]] = {g: {} for g in genes_of_interest}
    for ct in gene_sets:
        layer_path = wdir / "multiplex" / ct / "ppi_layer.tsv"
        if not layer_path.exists():
            layer_path = wdir / "multiplex" / ct / "ppi_layer.tsv.orig"
        if not layer_path.exists():
            continue
        df = pd.read_csv(layer_path, sep="\t", header=None, names=["g1", "g2"], dtype=str)
        df["g1"] = df["g1"].map(_gene_of)
        df["g2"] = df["g2"].map(_gene_of)
        deg_counter: dict[str, int] = {}
        for a, b in zip(df["g1"], df["g2"]):
            if a == b:
                continue
            deg_counter[a] = deg_counter.get(a, 0) + 1
            deg_counter[b] = deg_counter.get(b, 0) + 1
        for g in genes_of_interest:
            if g in deg_counter:
                degrees[g][ct] = deg_counter[g]
    return degrees


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wdir", required=True, help="MultiXrank input dir (has multiplex/, celltype_gene_sets.json)")
    ap.add_argument("--celltype-gene-sets", default=None, help="Defaults to <wdir>/celltype_gene_sets.json")
    ap.add_argument("--seed-csv", default=None,
                     help="Optional: read seeds from this CSV ('gene' column) instead of <wdir>/seeds/")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    wdir = Path(args.wdir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_sets_path = args.celltype_gene_sets or (wdir / "celltype_gene_sets.json")

    print(f"Loading gene sets from {gene_sets_path} ...")
    gene_sets = load_gene_sets(gene_sets_path)

    if args.seed_csv:
        df = pd.read_csv(args.seed_csv)
        seeds = sorted({str(g).strip().upper() for g in df["gene"].dropna()})
        print(f"Loaded {len(seeds)} seed genes from {args.seed_csv}")
    else:
        seeds = load_seeds(wdir)

    print(f"\nComputing per-layer degree for {len(seeds):,} seed genes ...")
    degrees = compute_layer_degrees(wdir, gene_sets, set(seeds))

    rows = []
    for g in seeds:
        for ct, genes in gene_sets.items():
            present = g in genes
            rows.append({"gene": g, "cell_type": ct, "present": present,
                         "degree": degrees.get(g, {}).get(ct, 0) if present else None})
    presence_df = pd.DataFrame(rows)
    presence_path = out_dir / "seed_layer_presence.csv"
    presence_df.to_csv(presence_path, index=False)
    print(f"Saved -> {presence_path}")

    summary_rows = []
    for g in seeds:
        layer_degs = degrees.get(g, {})
        dominant_layer = max(layer_degs, key=layer_degs.get) if layer_degs else None
        summary_rows.append({
            "gene": g,
            "n_layers_present": len(layer_degs),
            "total_degree_across_layers": sum(layer_degs.values()),
            "dominant_layer": dominant_layer,
            "dominant_layer_degree": layer_degs.get(dominant_layer, 0) if dominant_layer else 0,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("total_degree_across_layers", ascending=False)
    summary_path = out_dir / "seed_layer_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved -> {summary_path}")

    n_absent = int((summary_df["n_layers_present"] == 0).sum())
    print(f"\n{n_absent}/{len(seeds)} seed gene(s) absent from every cell-type layer entirely")
    if n_absent:
        print("  " + ", ".join(summary_df.loc[summary_df["n_layers_present"] == 0, "gene"].tolist()[:20]))


if __name__ == "__main__":
    main()