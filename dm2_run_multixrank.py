#!/usr/bin/env python3
"""
Run MultiXrank (Cell-Type-Aware)
=================================
Executes MultiXrank RWR on cell-type-layer inputs and saves both
per-cell-type gene scores and an aggregated score across all layers.

Outputs
-------
<output-dir>/
├── gene_rwr_scores.csv            # gene, rwr_score  (aggregated, sorted desc)
├── per_celltype_scores.csv        # gene × cell_type score matrix
└── multixrank_raw_output/         # raw MultiXrank output
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import multixrank


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

AGG_FUNCTIONS = {
    "max":  lambda df: df.max(axis=1),
    "mean": lambda df: df.mean(axis=1),
    "sum":  lambda df: df.sum(axis=1),
}


def load_layer_order(wdir: Path) -> list[str]:
    """
    Read cell_type_metadata.json written by script 1 to recover the ordered
    list of cell-type layer names.  Fallback to alphabetical if missing.
    """
    meta_path = wdir / "cell_type_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("cell_types", [])
    return []


def parse_multixrank_result(
    rwr_result: dict,
    layer_names: list[str],
) -> pd.DataFrame:
    """
    Convert MultiXrank's output dict into a (gene × cell_type) DataFrame.

    MultiXrank returns:
        rwr_result["multiplex"]["gene_multiplex"]  → DataFrame with cols
            [multiplex, layer, node, score]
    or a flat DataFrame/Series depending on version.  We handle both.
    """
    # ---- locate the gene_multiplex table ----
    raw = rwr_result.get("multiplex", rwr_result)
    if isinstance(raw, dict):
        raw = raw.get("gene_multiplex", next(iter(raw.values())))

    if hasattr(raw, "to_frame"):          # it's a Series
        raw = raw.reset_index()
        raw.columns = ["node", "score"]
        raw["layer"] = "layer_0"

    # Normalise column names (MultiXrank versions differ)
    raw = raw.reset_index(drop=True)
    col_map = {}
    for c in raw.columns:
        lc = str(c).lower()
        if "node" in lc or "gene" in lc:
            col_map[c] = "node"
        elif "score" in lc or "rank" in lc:
            col_map[c] = "score"
        elif "layer" in lc:
            col_map[c] = "layer"
    raw = raw.rename(columns=col_map)
    raw["score"] = pd.to_numeric(raw["score"], errors="coerce").fillna(0.0)
    raw["node"] = raw["node"].astype(str).str.strip().str.upper()

    # ---- pivot to gene × layer ----
    if "layer" in raw.columns:
        pivot = raw.pivot_table(index="node", columns="layer",
                                values="score", aggfunc="max")
    else:
        # single-layer fallback
        pivot = raw.set_index("node")[["score"]]
        pivot.columns = ["layer_0"]

    pivot.index.name = "gene"
    pivot.columns.name = None

    # ---- rename columns to cell-type names if we have them ----
    if layer_names and len(layer_names) == len(pivot.columns):
        # MultiXrank orders layers as they appear in config
        pivot.columns = layer_names
    else:
        # best-effort: keep whatever MultiXrank assigned
        pass

    return pivot.fillna(0.0)


def aggregate_scores(
    pivot: pd.DataFrame,
    method: str,
) -> pd.Series:
    agg_fn = AGG_FUNCTIONS.get(method)
    if agg_fn is None:
        raise ValueError(f"Unknown aggregation method '{method}'. "
                         f"Choose from: {list(AGG_FUNCTIONS)}")
    return agg_fn(pivot).rename("rwr_score")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run MultiXrank RWR and emit per-cell-type + aggregated scores"
    )
    parser.add_argument("--wdir", required=True,
                        help="MultiXrank working directory (output of script 1)")
    parser.add_argument("--config", default="config.yml",
                        help="Config filename relative to wdir (default: config.yml)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save processed RWR results")
    parser.add_argument("--aggregation", choices=list(AGG_FUNCTIONS), default="max",
                        help="How to collapse per-layer scores into one ranking "
                             "(default: max)")
    args = parser.parse_args()

    wdir = Path(args.wdir).resolve()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_out = out / "multixrank_raw_output"
    raw_out.mkdir(exist_ok=True)

    print("=" * 60)
    print("Running MultiXrank (cell-type layers)")
    print("=" * 60)
    print(f"Working dir   : {wdir}")
    print(f"Config file   : {args.config}")
    print(f"Output dir    : {out}")
    print(f"Aggregation   : {args.aggregation}")
    print("=" * 60 + "\n")

    # Read layer order from metadata written by script 1
    layer_names = load_layer_order(wdir)
    if layer_names:
        print(f"Cell-type layers ({len(layer_names)}): {layer_names}\n")
    else:
        print("⚠ cell_type_metadata.json not found — layer columns will use "
              "MultiXrank's default naming.\n")

    # Run RWR
    t0 = time.time()
    mxr = multixrank.Multixrank(config=args.config, wdir=str(wdir))
    rwr_result = mxr.random_walk_rank()
    print(f"\nRWR complete in {time.time() - t0:.1f}s\n")

    mxr.write_ranking(rwr_result, path=str(raw_out))
    print(f"Raw MultiXrank output saved → {raw_out}\n")

    # Parse into gene × cell_type matrix
    pivot = parse_multixrank_result(rwr_result, layer_names)
    print("Per-cell-type score matrix:")
    print(f"  Genes    : {len(pivot):,}")
    print(f"  Layers   : {list(pivot.columns)}")
    print(pivot.head(5).to_string())
    print()

    # Save per-cell-type scores
    ct_path = out / "per_celltype_scores.csv"
    pivot.reset_index().to_csv(ct_path, index=False)
    print(f"Saved per-cell-type scores → {ct_path}")

    # Aggregate
    agg_scores = aggregate_scores(pivot, args.aggregation)
    gene_scores = (
        agg_scores
        .reset_index()
        .rename(columns={"index": "gene"})   # reset_index keeps old name
        .sort_values("rwr_score", ascending=False)
        .reset_index(drop=True)
    )
    # Tidy: make sure gene column has the right name regardless of pivot state
    if "gene" not in gene_scores.columns:
        gene_scores = gene_scores.rename(columns={gene_scores.columns[0]: "gene"})

    scores_path = out / "gene_rwr_scores.csv"
    gene_scores.to_csv(scores_path, index=False)

    print(f"Saved aggregated RWR scores → {scores_path}")
    print(f"  Aggregation method : {args.aggregation}")
    print(f"  Genes ranked       : {len(gene_scores):,}")
    print(f"  Top score          : {gene_scores['rwr_score'].iloc[0]:.6e}")
    print(f"  Median score       : {gene_scores['rwr_score'].median():.6e}")
    print(f"  Non-zero scores    : {(gene_scores['rwr_score'] > 0).sum():,}")
    print("\nNext step: run script 3 (build_disease_module.py)")
    print(f"  --scores {scores_path}")
    print(f"  --per-celltype-scores {ct_path}")


if __name__ == "__main__":
    main()