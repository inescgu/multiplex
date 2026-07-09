#!/usr/bin/env python3
"""
Run MultiXrank (Edge-Driven Inter-Layer Architecture)
"""
import argparse
import json
import time
import sys
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

def check_seeds_in_network(wdir: Path, seeds_file: Path = None) -> dict:
    """
    Check which seed nodes are present in the network LCC.
    Returns stats on missing seeds.
    """
    import networkx as nx
    from pathlib import Path

    # Load seeds
    if seeds_file is None:
        seeds_file = wdir / "seeds" / "seeds.txt"  # Adjusted to correct path

    if not seeds_file.exists():
        print(f"Seeds file not found: {seeds_file}")
        return {}

    with open(seeds_file) as f:
        seeds = set(line.strip().upper() for line in f if line.strip())

    print(f"\nChecking {len(seeds)} seeds against network LCC...")
    print("=" * 60)

    # Check each layer
    network_dir = wdir / "multiplex" / "gene_multiplex"
    if not network_dir.exists():
        print(f"Network directory not found: {network_dir}")
        return {}

    layer_files = list(network_dir.glob("*.tsv"))
    print(f"Found {len(layer_files)} layer files\n")

    seeds_found = set()
    seeds_missing_by_layer = {layer.stem: set() for layer in layer_files}

    for layer_file in sorted(layer_files):
        # Load layer
        df = pd.read_csv(layer_file, sep='\t', header=None)

        # Get nodes in this layer
        nodes_in_layer = set(
            pd.concat([df.iloc[:, 0], df.iloc[:, 1]])
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # Build graph and find LCC
        G = nx.from_pandas_edgelist(
            df.iloc[:, [0, 1]],
            source=0,
            target=1,
            create_using=nx.Graph()
        )

        # Convert node names to uppercase for comparison
        G = nx.relabel_nodes(G, {n: str(n).upper() for n in G.nodes()})

        if len(G) == 0:
            print(f"  {layer_file.stem:30s}: EMPTY GRAPH")
            continue

        # Get LCC
        lcc = max(nx.connected_components(G), key=len)
        lcc_nodes = set(lcc)

        # Check seeds
        seeds_in_lcc = seeds & lcc_nodes
        seeds_not_in_lcc = seeds - lcc_nodes

        seeds_found.update(seeds_in_lcc)
        seeds_missing_by_layer[layer_file.stem] = seeds_not_in_lcc

        found_pct = 100 * len(seeds_in_lcc) / len(seeds) if seeds else 0
        in_graph_pct = 100 * len(seeds & nodes_in_layer) / len(seeds) if seeds else 0

        print(f"  {layer_file.stem:30s}: "
              f"{len(seeds_in_lcc):3d}/{len(seeds)} in LCC ({found_pct:5.1f}%) | "
              f"{len(seeds & nodes_in_layer):3d} in full graph ({in_graph_pct:5.1f}%)")

        if seeds_not_in_lcc:
            print(f"      → Not in LCC: {sorted(seeds_not_in_lcc)[:5]}"
                  f"{'...' if len(seeds_not_in_lcc) > 5 else ''}")

    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"  Total seeds:        {len(seeds)}")
    print(f"  Found in ≥1 LCC:    {len(seeds_found)} ({100*len(seeds_found)/len(seeds):.1f}%)")
    print(f"  Missing from all:   {len(seeds - seeds_found)}")

    if seeds - seeds_found:
        print(f"\n  *** COMPLETELY MISSING SEEDS (not in ANY layer) ***")
        for seed in sorted(seeds - seeds_found):
            print(f"      {seed}")

    print("=" * 60 + "\n")

    return {
        "total_seeds": seeds,
        "found_seeds": seeds_found,
        "missing_seeds": seeds - seeds_found,
        "seeds_missing_by_layer": seeds_missing_by_layer,
    }

def parse_multixrank_result(
    rwr_result,
    layer_info: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse MultiXrank output into per-layer pivot table.
    """
    if isinstance(rwr_result, list):
        if len(rwr_result) == 0:
            raise ValueError("MultiXrank returned empty list!")
        raw = pd.concat(rwr_result, ignore_index=True)
        print(f"  rwr_result is a list of {len(rwr_result)} DataFrame(s)")
    elif isinstance(rwr_result, pd.DataFrame):
        raw = rwr_result.copy()
        print("  rwr_result is a single DataFrame")
    elif isinstance(rwr_result, dict):
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

    if "node" not in raw.columns:
        print(f"  WARNING: no 'node' column found, columns are: {list(raw.columns)}")
        if raw.shape[1] >= 3:
            raw.columns = ["multiplex", "layer", "node", "score"] + list(raw.columns[4:])
        elif raw.shape[1] == 2:
            raw.columns = ["node", "score"]

    raw["score"] = pd.to_numeric(raw["score"], errors="coerce").fillna(0.0)
    raw["node"]  = raw["node"].astype(str).str.strip().str.upper()

    n_nonzero = (raw["score"] > 0).sum()
    print(f"  Non-zero scores in raw: {n_nonzero:,} / {len(raw):,}")
    if n_nonzero == 0:
        print("  *** ALL SCORES ARE ZERO — check seeds file and config! ***")

    if "layer" in raw.columns:
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

    excluded_types = {"special", "inter"}
    layer_to_celltype = {}
    for layer_name, info in layer_info.items():
        if info.get("type") not in excluded_types:
            cell_types = info.get("cell_types", [])
            if cell_types:
                layer_to_celltype[layer_name] = cell_types[0]

    print(f"  layer_to_celltype entries: {len(layer_to_celltype)}")
    intra_cols = [col for col in all_layers_pivot.columns if col in layer_to_celltype]
    print(f"  Matched intra columns: {len(intra_cols)} / {len(all_layers_pivot.columns)}")

    if not intra_cols:
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
        raise ValueError(f"Unknown aggregation '{method}'. Choose: {list(AGG_FUNCTIONS)}")
    return agg_fn(pivot).rename("rwr_score")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wdir",        required=True)
    parser.add_argument("--config",      default="config.yml")
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--aggregation", choices=list(AGG_FUNCTIONS), default="max")
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

    # Perform Diagnostic Seed Check
    seed_check = check_seeds_in_network(wdir)

    if seed_check:
        total_len = len(seed_check["total_seeds"])
        found_len = len(seed_check["found_seeds"])
        missing_pct = 100 * (total_len - found_len) / total_len if total_len else 0
        if missing_pct > 50:
            print(f"\n⚠️  WARNING: {missing_pct:.0f}% of seeds not found in LCC!")
            print("   Consider filtering your seeds file or checking gene name formats.\n")

    if seed_check and seed_check["missing_seeds"]:
        print(f"\n⚠️  Filtering out {len(seed_check['missing_seeds'])} seeds not in LCC...")
        filtered_seeds_file = wdir / "seeds" / "seeds_filtered.txt"
        with open(filtered_seeds_file, 'w') as f:
            for seed in sorted(seed_check["total_seeds"] - seed_check["missing_seeds"]):
                f.write(f"{seed}\n")
        print(f"   Saved filtered seeds → {filtered_seeds_file}")
        print(f"   Update config.yml to use: seeds_filtered.txt\n")
        sys.exit(1)  

    layer_info     = load_layer_info(wdir)
    cell_type_meta = load_cell_type_metadata(wdir)

    if layer_info:
        intra = sum(1 for v in layer_info.values() if v.get("type") == "intra")
        inter = sum(1 for v in layer_info.values() if v.get("type") == "inter")
        print(f"Layer structure: {intra} intra, {inter} inter, {len(layer_info)} total\n")

    # ------------------------------------------------------------------
    # Run RWR
    # ------------------------------------------------------------------
    t0 = time.time()
    config_path = (wdir / args.config if not Path(args.config).is_absolute() else Path(args.config))
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
    # Parse & Save Outputs
    # ------------------------------------------------------------------
    celltype_pivot, all_layers_pivot = parse_multixrank_result(rwr_result, layer_info)
    print(f"\nPer-cell-type matrix: {len(celltype_pivot):,} genes × {len(celltype_pivot.columns)} cell types")
    print(f"Non-zero genes: {(celltype_pivot > 0).any(axis=1).sum():,}")
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

    print(f"\nTop 20 genes:\n{gene_scores.head(20).to_string(index=False)}")

if __name__ == "__main__":
    main()