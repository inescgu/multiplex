#!/usr/bin/env python3
"""
Run MultiXrank (Edge-Driven Inter-Layer Architecture)
"""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import multixrank

AGG_FUNCTIONS = {
    "max":  lambda df: df.max(axis=1),
    "mean": lambda df: df.mean(axis=1),
    "sum":  lambda df: df.sum(axis=1),
}

def load_layer_info(wdir: Path) -> dict:
    info_path = wdir / "layer_info.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)
    return {}

def load_cell_type_metadata(wdir: Path) -> dict:
    meta_path = wdir / "cell_type_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}

def parse_multixrank_result(
    rwr_result,
    layer_info: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse MultiXrank output into per-layer pivot table.
    
    MultiXrank returns a LIST of DataFrames (one per multiplex network).
    Each DataFrame has columns: multiplex, layer, node, score
    """
    # -----------------------------------------------------------
    # Step 1: extract the raw DataFrame
    # -----------------------------------------------------------
    if isinstance(rwr_result, list):
        # Standard MultiXrank output: list of DataFrames
        if len(rwr_result) == 0:
            raise ValueError("MultiXrank returned empty list!")
        raw = pd.concat(rwr_result, ignore_index=True)
        print(f"  rwr_result is a list of {len(rwr_result)} DataFrame(s)")
    elif isinstance(rwr_result, pd.DataFrame):
        raw = rwr_result.copy()
        print("  rwr_result is a single DataFrame")
    elif isinstance(rwr_result, dict):
        # Fallback: old dict-based format
        frames = []
        for key, val in rwr_result.items():
            if isinstance(val, pd.DataFrame):
                frames.append(val)
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, pd.DataFrame):
                        frames.append(v2)
        if not frames:
            raise ValueError(f"Could not extract DataFrame from dict keys: "
                             f"{list(rwr_result.keys())}")
        raw = pd.concat(frames, ignore_index=True)
        print(f"  rwr_result is dict, extracted {len(frames)} DataFrame(s)")
    else:
        raise TypeError(f"Unexpected rwr_result type: {type(rwr_result)}")

    print(f"  Raw shape: {raw.shape}")
    print(f"  Raw columns: {list(raw.columns)}")
    print(f"  Raw head:\n{raw.head(3).to_string()}")

    # -----------------------------------------------------------
    # Step 2: normalise column names
    # -----------------------------------------------------------
    col_map = {}
    for c in raw.columns:
        lc = str(c).lower().strip()
        if lc in ("node", "gene", "gene_name"):
            col_map[c] = "node"
        elif lc in ("score", "rwr_score", "rank_score"):
            col_map[c] = "score"
        elif lc in ("layer", "layer_name", "layer_id"):
            col_map[c] = "layer"
        elif lc in ("multiplex", "network", "multiplex_name"):
            col_map[c] = "multiplex"
    raw = raw.rename(columns=col_map)

    # If still no 'node' column, try positional assignment
    if "node" not in raw.columns:
        print(f"  WARNING: no 'node' column found, columns are: {list(raw.columns)}")
        # MultiXrank v1 format: columns may be positional
        if raw.shape[1] >= 3:
            raw.columns = ["multiplex", "layer", "node", "score"] + \
                          list(raw.columns[4:])
        elif raw.shape[1] == 2:
            raw.columns = ["node", "score"]

    raw["score"] = pd.to_numeric(raw["score"], errors="coerce").fillna(0.0)
    raw["node"]  = raw["node"].astype(str).str.strip().str.upper()

    # -----------------------------------------------------------
    # Step 3: sanity check scores
    # -----------------------------------------------------------
    n_nonzero = (raw["score"] > 0).sum()
    print(f"  Non-zero scores in raw: {n_nonzero:,} / {len(raw):,}")
    if n_nonzero == 0:
        print("  *** ALL SCORES ARE ZERO — check seeds file and config! ***")

    # -----------------------------------------------------------
    # Step 4: pivot to gene × layer
    # -----------------------------------------------------------
    if "layer" in raw.columns:
        # Extract just the stem of the layer path for matching
        # e.g. "multiplex/gene_multiplex/T_cell_merged.tsv" -> "T_cell_merged"
        raw["layer_stem"] = (
            raw["layer"]
            .astype(str)
            .apply(lambda x: Path(x).stem if "/" in x or "\\" in x else x)
        )
        all_layers_pivot = raw.pivot_table(
            index="node",
            columns="layer_stem",
            values="score",
            aggfunc="max",
        )
    else:
        print("  WARNING: no 'layer' column — single-column pivot")
        all_layers_pivot = raw.set_index("node")[["score"]]
        all_layers_pivot.columns = ["all_layers"]

    all_layers_pivot.index.name = "gene"
    all_layers_pivot.columns.name = None
    all_layers_pivot = all_layers_pivot.fillna(0.0)

    print(f"  Pivot shape: {all_layers_pivot.shape}")
    print(f"  Pivot columns (first 5): {list(all_layers_pivot.columns)[:5]}")
    print(f"  Non-zero cells in pivot: "
          f"{(all_layers_pivot > 0).any(axis=1).sum():,} genes with any score")

    # -----------------------------------------------------------
    # Step 5: map layer stems to cell types via layer_info
    # -----------------------------------------------------------
    # layer_info keys are stems like "T_cell_merged"
    # pivot columns are also stems after our normalisation above
    excluded_types = {"special", "inter"}
    layer_to_celltype = {}

    for layer_name, info in layer_info.items():
        if info.get("type") not in excluded_types:
            cell_types = info.get("cell_types", [])
            if cell_types:
                layer_to_celltype[layer_name] = cell_types[0]

    print(f"  layer_to_celltype entries: {len(layer_to_celltype)}")

    intra_cols = [col for col in all_layers_pivot.columns
                  if col in layer_to_celltype]

    print(f"  Matched intra columns: {len(intra_cols)} / "
          f"{len(all_layers_pivot.columns)}")

    if not intra_cols:
        # Fallback: use ALL columns (better than nothing)
        print("  WARNING: no layer_info matches — using ALL layers as cell types")
        print(f"  Pivot cols: {list(all_layers_pivot.columns)[:10]}")
        print(f"  layer_info keys: {list(layer_info.keys())[:10]}")
        celltype_pivot = all_layers_pivot.copy()
    else:
        celltype_pivot = all_layers_pivot[intra_cols].copy()
        celltype_pivot.columns = [layer_to_celltype[col] for col in intra_cols]

    return celltype_pivot.fillna(0.0), all_layers_pivot.fillna(0.0)


def aggregate_scores(pivot: pd.DataFrame, method: str) -> pd.Series:
    agg_fn = AGG_FUNCTIONS.get(method)
    if agg_fn is None:
        raise ValueError(f"Unknown aggregation '{method}'. "
                         f"Choose: {list(AGG_FUNCTIONS)}")
    return agg_fn(pivot).rename("rwr_score")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wdir",        required=True)
    parser.add_argument("--config",      default="config.yml")
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--aggregation", choices=list(AGG_FUNCTIONS),
                        default="max")
    args = parser.parse_args()

    wdir    = Path(args.wdir).resolve()
    out     = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_out = out / "multixrank_raw_output"
    raw_out.mkdir(exist_ok=True)

    print("=" * 60)
    print("Running MultiXrank (edge-driven inter-layer)")
    print("=" * 60)
    print(f"Working dir : {wdir}")
    print(f"Config      : {args.config}")
    print(f"Output dir  : {out}")
    print(f"Aggregation : {args.aggregation}")
    print("=" * 60 + "\n")

    layer_info     = load_layer_info(wdir)
    cell_type_meta = load_cell_type_metadata(wdir)

    if layer_info:
        intra = sum(1 for v in layer_info.values() if v.get("type") == "intra")
        inter = sum(1 for v in layer_info.values() if v.get("type") == "inter")
        print(f"Layer structure: {intra} intra, {inter} inter, "
              f"{len(layer_info)} total\n")

    # ------------------------------------------------------------------
    # Run RWR
    # ------------------------------------------------------------------
    t0 = time.time()
    config_path = (wdir / args.config
                   if not Path(args.config).is_absolute()
                   else Path(args.config))

    print(f"Config path : {config_path}")
    print(f"Config exists: {config_path.exists()}\n")

    mxr        = multixrank.Multixrank(config=str(config_path), wdir=str(wdir))
    rwr_result = mxr.random_walk_rank()

    print(f"\nRWR complete in {time.time() - t0:.1f}s")
    print(f"Result type : {type(rwr_result)}")
    if isinstance(rwr_result, list):
        print(f"Result len  : {len(rwr_result)}")
        for i, df in enumerate(rwr_result):
            if hasattr(df, "shape"):
                print(f"  [{i}] shape={df.shape}, "
                      f"cols={list(df.columns)}, "
                      f"nonzero={(df.select_dtypes('number') > 0).any(axis=1).sum()}")
    print()

    mxr.write_ranking(rwr_result, path=str(raw_out))
    print(f"Raw output saved → {raw_out}\n")

    # ------------------------------------------------------------------
    # Parse & save
    # ------------------------------------------------------------------
    celltype_pivot, all_layers_pivot = parse_multixrank_result(
        rwr_result, layer_info
    )

    print(f"\nPer-cell-type matrix: "
          f"{len(celltype_pivot):,} genes × {len(celltype_pivot.columns)} cell types")
    print(f"Non-zero genes: "
          f"{(celltype_pivot > 0).any(axis=1).sum():,}")
    print(celltype_pivot[celltype_pivot.max(axis=1) > 0].head(5).to_string())
    print()

    ct_path  = out / "per_celltype_scores.csv"
    all_path = out / "per_celltype_scores_detailed.csv"
    celltype_pivot.reset_index().to_csv(ct_path,  index=False)
    all_layers_pivot.reset_index().to_csv(all_path, index=False)
    print(f"Saved per-celltype  → {ct_path}")
    print(f"Saved all-layers    → {all_path}")

    agg_scores = aggregate_scores(celltype_pivot, args.aggregation)
    gene_scores = (
        agg_scores
        .reset_index()
        .rename(columns={"node": "gene", "index": "gene"})
        .sort_values("rwr_score", ascending=False)
        .reset_index(drop=True)
    )
    if "gene" not in gene_scores.columns:
        gene_scores.columns = ["gene", "rwr_score"]

    scores_path = out / "gene_rwr_scores.csv"
    gene_scores.to_csv(scores_path, index=False)

    print(f"\nAggregated scores   → {scores_path}")
    print(f"  Method      : {args.aggregation}")
    print(f"  Genes ranked: {len(gene_scores):,}")
    print(f"  Non-zero    : {(gene_scores['rwr_score'] > 0).sum():,}")
    if len(gene_scores) > 0:
        print(f"  Top score   : {gene_scores['rwr_score'].iloc[0]:.6e}")
        print(f"  Median      : {gene_scores['rwr_score'].median():.6e}")
    print(f"\nTop 20 genes:\n"
          f"{gene_scores.head(20).to_string(index=False)}")


if __name__ == "__main__":
    main()