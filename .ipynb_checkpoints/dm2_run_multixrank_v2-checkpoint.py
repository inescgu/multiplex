#!/usr/bin/env python3
"""
Run MultiXrank (Edge-Driven Inter-Layer Architecture)
=======================================================
Executes MultiXrank RWR on edge-driven inter-layer inputs and produces:
  1. Per-cell-type gene scores (from intra-layer contributions)
  2. Aggregated gene scores across all layers

Outputs
-------
<output-dir>/
├── gene_rwr_scores.csv            # gene, rwr_score  (aggregated)
├── per_celltype_scores.csv        # gene × cell_type score matrix
├── per_celltype_scores_detailed.csv  # includes inter-layer contributions
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


def load_layer_info(wdir: Path) -> dict:
    """
    Read layer_info.json to understand which layers are intra vs inter.

    Returns
    -------
    dict mapping layer_name -> {"type": "intra"|"inter", "cell_types": [...], ...}
    """
    info_path = wdir / "layer_info.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)
    return {}


def load_cell_type_metadata(wdir: Path) -> dict:
    """Read cell_type_metadata.json for cell type list."""
    meta_path = wdir / "cell_type_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def parse_multixrank_result(
    rwr_result: dict,
    layer_info: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert MultiXrank's output into:
      1. per_celltype_scores: gene × cell_type matrix (intra-layer only)
      2. all_layer_scores: gene × all_layers matrix (for debugging)

    MultiXrank returns:
        rwr_result["multiplex"]["gene_multiplex"]  → DataFrame with cols
            [multiplex, layer, node, score]
    """
    # ---- locate the gene_multiplex table ----
    raw = rwr_result.get("multiplex", rwr_result)
    if isinstance(raw, dict):
        raw = raw.get("gene_multiplex", next(iter(raw.values())))

    if hasattr(raw, "to_frame"):          # it's a Series
        raw = raw.reset_index()
        raw.columns = ["node", "score"]
        raw["layer"] = "layer_0"

    # Normalise column names
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
        all_layers_pivot = raw.pivot_table(index="node", columns="layer",
                                           values="score", aggfunc="max")
    else:
        all_layers_pivot = raw.set_index("node")[["score"]]
        all_layers_pivot.columns = ["layer_0"]

    all_layers_pivot.index.name = "gene"
    all_layers_pivot.columns.name = None

    # ---- extract intra-layer scores only ----
    # Map layer names to cell types
    layer_to_celltype = {}
    for layer_name, info in layer_info.items():
        if info.get("type") == "intra":
            cell_types = info.get("cell_types", [])
            if cell_types:
                layer_to_celltype[layer_name] = cell_types[0]

    # Filter to intra-layer columns only
    intra_cols = [col for col in all_layers_pivot.columns if col in layer_to_celltype]
    if intra_cols:
        celltype_pivot = all_layers_pivot[intra_cols].copy()
        celltype_pivot.columns = [layer_to_celltype[col] for col in intra_cols]
    else:
        # Fallback: use all columns
        celltype_pivot = all_layers_pivot.copy()

    return celltype_pivot.fillna(0.0), all_layers_pivot.fillna(0.0)


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
        description="Run MultiXrank RWR (edge-driven inter-layer) and emit per-cell-type + aggregated scores"
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
    print("Running MultiXrank (edge-driven inter-layer)")
    print("=" * 60)
    print(f"Working dir   : {wdir}")
    print(f"Config file   : {args.config}")
    print(f"Output dir    : {out}")
    print(f"Aggregation   : {args.aggregation}")
    print("=" * 60 + "\n")

    # Read layer info
    layer_info = load_layer_info(wdir)
    cell_type_meta = load_cell_type_metadata(wdir)

    if layer_info:
        intra_count = sum(1 for info in layer_info.values() if info.get("type") == "intra")
        inter_count = sum(1 for info in layer_info.values() if info.get("type") == "inter")
        print(f"Layer structure:")
        print(f"  Intra-layers (PPI within cell types) : {intra_count}")
        print(f"  Inter-layers (between cell types)    : {inter_count}")
        print(f"  Total layers                         : {intra_count + inter_count}\n")

    if cell_type_meta:
        cell_types = cell_type_meta.get("cell_types", [])
        print(f"Cell types: {cell_types}\n")

    # Run RWR
    t0 = time.time()
    config_path = wdir / args.config if not Path(args.config).is_absolute() else args.config
    mxr = multixrank.Multixrank(config=str(config_path), wdir=str(wdir))
    rwr_result = mxr.random_walk_rank()
    print(f"\nRWR complete in {time.time() - t0:.1f}s\n")

    mxr.write_ranking(rwr_result, path=str(raw_out))
    print(f"Raw MultiXrank output saved → {raw_out}\n")

    # Parse into per-cell-type and all-layers matrices
    celltype_pivot, all_layers_pivot = parse_multixrank_result(rwr_result, layer_info)

    print("Per-cell-type score matrix (intra-layer only):")
    print(f"  Genes    : {len(celltype_pivot):,}")
    print(f"  Cell types : {list(celltype_pivot.columns)}")
    print(celltype_pivot.head(5).to_string())
    print()

    # Save per-cell-type scores
    ct_path = out / "per_celltype_scores.csv"
    celltype_pivot.reset_index().to_csv(ct_path, index=False)
    print(f"Saved per-cell-type scores → {ct_path}")

    # Save all-layer scores (for debugging/reference)
    all_path = out / "per_celltype_scores_detailed.csv"
    all_layers_pivot.reset_index().to_csv(all_path, index=False)
    print(f"Saved all-layer scores (detailed) → {all_path}")

    # Aggregate intra-layer scores only
    agg_scores = aggregate_scores(celltype_pivot, args.aggregation)
    gene_scores = (
        agg_scores
        .reset_index()
        .rename(columns={"index": "gene"})
        .sort_values("rwr_score", ascending=False)
        .reset_index(drop=True)
    )

    # Tidy: make sure gene column has the right name
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
