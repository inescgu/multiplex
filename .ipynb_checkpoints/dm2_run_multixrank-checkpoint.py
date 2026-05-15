#!/usr/bin/env python3
"""
Run MultiXrank
===============
Executes MultiXrank RWR on the prepared inputs and saves gene-level scores.

Outputs
-------
<output-dir>/
├── gene_rwr_scores.csv       # gene, score (sorted descending)
└── multixrank_raw_output/    # raw MultiXrank output directory
"""

import os
import argparse
import time
from pathlib import Path

import pandas as pd
import multixrank


def main():
    parser = argparse.ArgumentParser(description="Run MultiXrank RWR")
    parser.add_argument("--wdir", required=True,
                        help="MultiXrank working directory (from script 1 output)")
    parser.add_argument("--config", default="config.yml",
                        help="Config filename relative to wdir (default config.yml)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save processed RWR results")
    args = parser.parse_args()

    wdir = Path(args.wdir).resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_out = out / "multixrank_raw_output"
    raw_out.mkdir(exist_ok=True)

    print("=" * 60)
    print("Running MultiXrank")
    print("=" * 60)
    print(f"Working dir : {wdir}")
    print(f"Config file : {args.config}")
    print(f"Output dir  : {out}")
    print("=" * 60 + "\n")

    t0 = time.time()
    mxr = multixrank.Multixrank(config=args.config, wdir=str(wdir))
    rwr_result = mxr.random_walk_rank()
    print(f"\nRWR complete in {time.time() - t0:.1f}s\n")

    # Persist raw output via MultiXrank's own writer
    mxr.write_ranking(rwr_result, path=str(raw_out))
    print(f"Raw MultiXrank output saved → {raw_out}")
    
    # Each Series has the gene scores indexed by node name
    df_list = []
    for mplex_name, series in rwr_result.items():
        if mplex_name != 'multiplex':  # Skip the first series
            df = series.to_frame()
            df_list.append(df)

    full = pd.concat(df_list, axis=1)
    full["score"] = pd.to_numeric(full["score"], errors="coerce")
    print(full)

    # Now proceed with the rest
    if "node" not in full.columns or "score" not in full.columns:
        raise RuntimeError(
            f"Unexpected MultiXrank output columns: {list(full.columns)}. "
            "Expected at least 'node' and 'score'."
        )

    gene_scores = (
        full.groupby("node", as_index=False)["score"]
        .max()
        .rename(columns={"node": "gene", "score": "rwr_score"})
        .sort_values("rwr_score", ascending=False)
        .reset_index(drop=True)
    )
    scores_path = out / "gene_rwr_scores.csv"
    gene_scores.to_csv(scores_path, index=False)

    print(f"\nSaved gene-level RWR scores → {scores_path}")
    print(f"  Genes ranked     : {len(gene_scores):,}")
    print(f"  Top score        : {gene_scores['rwr_score'].iloc[0]:.6e}")
    print(f"  Median score     : {gene_scores['rwr_score'].median():.6e}")
    print(f"  Non-zero scores  : {(gene_scores['rwr_score'] > 0).sum():,}")
    print("\nNext step: run script 3 (build_disease_module.py)")


if __name__ == "__main__":
    main()