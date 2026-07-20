#!/usr/bin/env python3
"""
Quick diagnostic — run this against your EXISTING integrated_multiplex.pkl
(no need to rebuild anything first). Confirms whether pathway edges are
missing everywhere (not just for neurons), and whether the CSV format
matches the uniprot-symbol-mapping theory.

Usage:
    python diagnose_pathway_bug.py \
        --multiplex-pkl 03_integrated/integrated_multiplex_output/integrated_multiplex.pkl \
        --reactome-csv pathway_prep/reactome_interactions_with_pathways_v2.csv
"""
import argparse
import pickle
from collections import Counter

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiplex-pkl", required=True)
    ap.add_argument("--reactome-csv", required=True)
    args = ap.parse_args()

    # 1) Global edge_type distribution in the whole graph (all cell types)
    print("Loading multiplex ...")
    with open(args.multiplex_pkl, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")

    print("Global edge_type counts (ALL cell types, not just neurons):")
    counts = Counter(d.get("edge_type", "unknown") for _, _, d in G.edges(data=True))
    for et, c in counts.most_common():
        print(f"  {et:<20} {c:>12,}  ({100*c/G.number_of_edges():.3f}%)")
    print()

    # 2) Sample the reactome CSV's uniprot1/uniprot2 columns
    print(f"Sampling {args.reactome_csv} ...")
    df = pd.read_csv(args.reactome_csv, nrows=10)
    print(f"  Columns: {list(df.columns)}")
    print(df[[c for c in ["uniprot1", "uniprot2", "common_pathway"] if c in df.columns]].head(10).to_string())
    print()

    # 3) Do those raw uniprot1/uniprot2 values match any gene symbol actually in the graph?
    graph_genes = {d.get("gene") for _, d in G.nodes(data=True) if d.get("gene")}
    df_full = pd.read_csv(args.reactome_csv, usecols=["uniprot1", "uniprot2"])
    raw_ids = set(df_full["uniprot1"].astype(str).str.strip().str.upper()) | \
              set(df_full["uniprot2"].astype(str).str.strip().str.upper())

    overlap = raw_ids & graph_genes
    print(f"Distinct uniprot1/uniprot2 values in CSV : {len(raw_ids):,}")
    print(f"Distinct gene symbols in graph            : {len(graph_genes):,}")
    print(f"Values that match a graph gene symbol     : {len(overlap):,} "
          f"({100*len(overlap)/max(len(raw_ids),1):.2f}%)")
    print()
    if len(overlap) / max(len(raw_ids), 1) < 0.05:
        print("  -> Near-zero overlap. Strongly suggests uniprot1/uniprot2 are raw UniProt")
        print("     accessions (e.g. 'P30926') being compared against gene symbols (e.g. 'TAC3')")
        print("     with no --uniprot-map translation applied in step3. This is why pathway")
        print("     edges are (almost) entirely missing from the graph.")
    else:
        print("  -> Meaningful overlap found. The uniprot-map theory may not be the (only) issue —")
        print("     look elsewhere (e.g. --cross-source-only filtering, --max-pathway-size).")


if __name__ == "__main__":
    main()