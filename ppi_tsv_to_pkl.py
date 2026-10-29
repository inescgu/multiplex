#!/usr/bin/env python3
"""
ppi_tsv_to_pkl.py -- turn a PPI_builder consensus TSV into the networkx pickle
the pipeline reads as 00_inputs/ppi_network.pkl.

Input  (PPI_builder outputs/consensus_ppi_symbol_combined_lcc_<date>.tsv):
    gene_a  gene_b  sources                        n_sources
    A1BG    KCMF1   BioPlex_293T,BioPlex_HCT116    2
Output: undirected nx.Graph on gene symbols, edge attributes
    sources       list[str]   e.g. ['BioPlex_293T', 'BioPlex_HCT116']
    source_count  int
-- the same format as the previous network (og_gene_set combined26), so every
step that reads --ppi-pkl / --base-ppi works unchanged. Self-loops are dropped
and only the largest connected component is kept.

With --compare, prints node/edge overlap with the old network and seed coverage.

Usage:
    python ppi_tsv_to_pkl.py --tsv <consensus_..._lcc_YYYYMMDD.tsv> \
        --out 00_inputs/ppi_network_YYYYMMDD.pkl \
        --compare ../og_gene_set/temp/combined26/full_network.pkl \
        --seeds 00_inputs/seed_genes.csv
"""
import argparse
import ast
import pickle
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd


def build_graph(tsv: Path) -> nx.Graph:
    df = pd.read_csv(tsv, sep="\t", dtype=str)
    missing = {"gene_a", "gene_b", "sources"} - set(df.columns)
    if missing:
        raise SystemExit(f"{tsv}: missing column(s) {sorted(missing)}; got {list(df.columns)}")
    df = df.dropna(subset=["gene_a", "gene_b"])
    df = df[df["gene_a"] != df["gene_b"]]

    G = nx.Graph()
    for a, b, s in zip(df["gene_a"], df["gene_b"], df["sources"].fillna("")):
        # newer PPI_builder writes a list repr "['BioGRID', 'IntAct']", older ones "BioGRID,IntAct"
        srcs = ast.literal_eval(s) if s.startswith("[") else [x for x in s.split(",") if x]
        if G.has_edge(a, b):  # same pair listed twice -> merge sources
            srcs = sorted(set(G[a][b]["sources"]) | set(srcs))
        G.add_edge(a, b, sources=srcs, source_count=len(srcs))
    return G


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="old network pickle to compare against")
    ap.add_argument("--seeds", help="seed_genes.csv (column 'gene') for a coverage check")
    args = ap.parse_args()

    G = build_graph(Path(args.tsv))
    n_cc = nx.number_connected_components(G)
    if n_cc > 1:
        lcc = max(nx.connected_components(G), key=len)
        print(f"{n_cc} components; keeping LCC ({len(lcc):,} of {G.number_of_nodes():,} nodes)")
        G = G.subgraph(lcc).copy()

    print(f"New network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    sc = Counter(d["source_count"] for *_, d in G.edges(data=True))
    print("  source_count:", dict(sorted(sc.items())))
    print("  sources:", dict(Counter(s for *_, d in G.edges(data=True) for s in d["sources"]).most_common()))

    if args.compare:
        with open(args.compare, "rb") as f:
            H = pickle.load(f)
        nG, nH = set(G.nodes()), set(H.nodes())
        eG = {frozenset(e) for e in G.edges()}
        eH = {frozenset(e) for e in H.edges()}
        print(f"Old network: {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")
        print(f"  nodes: {len(nG & nH):,} shared, {len(nG - nH):,} new only, {len(nH - nG):,} old only")
        print(f"  edges: {len(eG & eH):,} shared, {len(eG - eH):,} new only, {len(eH - eG):,} old only")
    if args.seeds:
        seeds = set(pd.read_csv(args.seeds)["gene"].str.upper())
        line = f"Seeds in network: {len(seeds & set(G.nodes())):,} / {len(seeds):,}"
        if args.compare:
            line += f" (old network: {len(seeds & set(H.nodes())):,})"
        print(line)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
