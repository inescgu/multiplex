#!/usr/bin/env python3
"""
validate_external_breadth.py

FIX 2 -- Test "high persistence = broad/general biology" against a
generality measure that has NOTHING to do with your own network, RWR, or
seed list. Enrichment against your own phenotype/pain-gene CSVs is
circular (you built the module by walking outward from pain seeds, of
course its neighborhood looks pain-related) -- this script asks instead:
do high-persistence clusters contain genes that are independently known to
be "broad" (many diseases, many tissues, many publications), while
low-persistence clusters contain genes that are independently "narrow"?

You need to supply at least one external annotation file. Good, cheap
options:
  - DisGeNET "gene_disease associations" export -> count distinct diseases
    per gene (gene, n_diseases)
  - GTEx median TPM table -> count tissues with expression above some
    threshold per gene (gene, n_tissues)
  - A PubMed count per gene symbol (e.g. via NCBI gene2pubmed) -> literature
    degree, useful as a CONTROL variable for study bias (see the "biologist"
    critique: streetlight effect). If persistence correlates with PubMed
    count as strongly as with disease/tissue breadth, that's a red flag
    your "broad" clusters are just "famous," not "biologically general."

Each external file just needs two columns: 'gene' and a numeric breadth
score. Pass as many as you have with repeated --external-annotation.

USAGE
-----
    python validate_external_breadth.py \
        --nodes-file heterogeneous_hidef_faithful_cpm.nodes \
        --supra-graph supra_graph.pkl \
        --persistence-threshold 8 \
        --external-annotation disgenet_disease_count.csv:n_diseases \
        --external-annotation gtex_tissue_count.csv:n_tissues \
        --external-annotation pubmed_count.csv:n_pubmed \
        --output-prefix characterized/heterogeneous_hidef
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hidef_multiplex_adapter import load_multiplex_clusters


def load_external(spec: str):
    """spec = 'path/to/file.csv:column_name'. File must have a 'gene' column
    plus the named numeric column."""
    path, col = spec.split(":", 1)
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        raise ValueError(f"{path} has no 'gene' column")
    df["gene"] = df["gene"].astype(str).str.upper().str.strip()
    return dict(zip(df["gene"], df[col])), col


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes-file", required=True)
    ap.add_argument("--supra-graph", default=None)
    ap.add_argument("--persistence-threshold", type=int, default=5)
    ap.add_argument("--external-annotation", action="append", required=True,
                     help="path.csv:column_name -- repeatable")
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    node_names = None
    if args.supra_graph:
        with open(args.supra_graph, "rb") as f:
            g = pickle.load(f)
        node_names = g.vs["name"]

    clusters, _, extra = load_multiplex_clusters(
        args.nodes_file, node_names=node_names,
        persistence_threshold=args.persistence_threshold,
    )
    if not clusters:
        raise SystemExit("No clusters passed the persistence threshold -- nothing to test.")

    external_maps = [load_external(spec) for spec in args.external_annotation]

    rows = []
    for cname, genes in clusters.items():
        row = {"cluster_id": cname, "size": len(genes),
               "persistence": extra["persistence"][cname]}
        for mapping, colname in external_maps:
            vals = [mapping[g] for g in genes if g in mapping]
            row[f"mean_{colname}"] = float(np.mean(vals)) if vals else np.nan
            row[f"frac_annotated_{colname}"] = len(vals) / len(genes) if genes else 0.0
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("persistence", ascending=False)
    summary_path = f"{args.output_prefix}_external_breadth.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved per-cluster external-breadth table -> {summary_path}")

    print("\nAnnotation coverage (fraction of cluster genes with an external value):")
    for _, colname in external_maps:
        frac_col = f"frac_annotated_{colname}"
        print(f"  {colname:<20} median coverage = {df[frac_col].median():.2f} "
              f"(low coverage weakens any correlation you find/don't find)")

    lines = ["=" * 70, "FIX 2: EXTERNAL BREADTH VALIDATION", "=" * 70,
              f"Clusters tested: {len(df)}", ""]

    fig, axes = plt.subplots(1, len(external_maps), figsize=(5 * len(external_maps), 4.5),
                              squeeze=False)
    for i, (_, colname) in enumerate(external_maps):
        mean_col = f"mean_{colname}"
        rho, p = stats.spearmanr(df["persistence"], df[mean_col], nan_policy="omit")
        line = (f"persistence vs mean({colname}) across cluster genes: "
                f"rho={rho:+.3f}  p={p:.2e}  n={df[mean_col].notna().sum()}")
        print(line)
        lines.append(line)

        ax = axes[0][i]
        ax.scatter(df["persistence"], df[mean_col], s=16, alpha=0.55, edgecolors="none")
        ax.set_xlabel("persistence")
        ax.set_ylabel(f"mean {colname} per cluster")
        ax.set_title(f"rho={rho:+.2f}, p={p:.1e}")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = f"{args.output_prefix}_external_breadth_scatter.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved scatter plots -> {plot_path}")

    lines.append("")
    lines.append("How to read this: a POSITIVE, significant correlation between persistence "
                 "and an independent breadth measure (disease count, tissue count) that was "
                 "never used to build your network or seeds is the strongest single piece of "
                 "non-circular support for your hypothesis. If PubMed/literature-degree shows "
                 "the SAME strength of correlation as disease/tissue breadth, treat that as "
                 "evidence the pattern may partly reflect study bias (well-studied genes end "
                 "up everywhere) rather than genuine biological generality -- report both.")
    Path(f"{args.output_prefix}_external_breadth_report.txt").write_text("\n".join(lines))
    print(f"Saved report -> {args.output_prefix}_external_breadth_report.txt")


if __name__ == "__main__":
    main()