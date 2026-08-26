#!/usr/bin/env python3
"""
Step 4 (replaces old step8): Run MultiXrank on the heterogeneous
multiplex+bipartite structure and parse output.

Why this is so much shorter than the old run script
-----------------------------------------------------
The old script had to reconstruct which "layer" belonged to which cell type
via layer_info.json, because in the homogeneous design every cell type was
just a same-named *layer* inside one giant multiplex, and the RWR output's
'layer' column was the only thing distinguishing them.

Here, every cell type IS its own multiplex, so MultiXrank's own output
column 'multiplex' already tells you the cell type directly — no lookup
table, no fuzzy matching needed.
"""
import argparse
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import multixrank

AGG_FUNCTIONS = {
    "max": lambda df: df.max(axis=1),
    "mean": lambda df: df.mean(axis=1),
    "sum": lambda df: df.sum(axis=1),
}


def check_seeds_in_network(wdir: Path) -> dict:
    """Lightweight LCC sanity check across the combined reference graph
    (multiplex PPI edges + bipartite edges) written by step 3."""
    import pickle
    seeds_path = wdir / "seeds" / "seeds.txt"
    ref_path = wdir / "gene_reference_graph.pkl"
    if not seeds_path.exists() or not ref_path.exists():
        return {}

    def _norm(s: str) -> str:
        s = s.strip()
        if "::" in s:
            ct, gene = s.rsplit("::", 1)
            return f"{ct}::{gene.upper()}"
        return s.upper()

    seeds = {_norm(s) for s in seeds_path.read_text().splitlines() if s.strip()}
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)

    print(f"\nChecking {len(seeds)} seeds against combined reference graph LCC ...")
    if ref.number_of_nodes() == 0:
        print("  Reference graph is empty — skipping LCC check.\n")
        return {}
    lcc = max(nx.connected_components(ref), key=len)
    found = seeds & lcc
    missing = seeds - lcc
    print(f"  Seeds in LCC: {len(found)}/{len(seeds)} ({100*len(found)/max(len(seeds),1):.1f}%)")
    if missing:
        print(f"  Not in LCC (first 10): {sorted(missing)[:10]}")
    print()
    return {"total_seeds": seeds, "found_seeds": found, "missing_seeds": missing}


def parse_multixrank_result(rwr_result) -> pd.DataFrame:
    """rwr_result columns per MultiXrank docs: multiplex, node, layer, score.
    Since every multiplex here has exactly one layer, 'multiplex' IS the
    cell type — pivot straight on it. Node IDs are namespaced as
    "<celltype>::GENE" (see step 2.5), so strip the prefix back off before
    pivoting — the 'multiplex' column already gives the cell type."""
    if isinstance(rwr_result, list):
        raw = pd.concat(rwr_result, ignore_index=True)
    elif isinstance(rwr_result, dict):
        frames = [v for v in rwr_result.values() if isinstance(v, pd.DataFrame)]
        raw = pd.concat(frames, ignore_index=True)
    else:
        raw = rwr_result.copy()

    raw.columns = [str(c).lower().strip() for c in raw.columns]
    raw["score"] = pd.to_numeric(raw["score"], errors="coerce").fillna(0.0)
    raw["node"] = raw["node"].astype(str).str.strip().str.rsplit("::", n=1).str[-1].str.upper()

    pivot = raw.pivot_table(index="node", columns="multiplex", values="score", aggfunc="max")
    pivot.index.name = "gene"
    pivot.columns.name = None
    return pivot.fillna(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wdir", required=True)
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--aggregation", choices=list(AGG_FUNCTIONS), default="max")
    args = ap.parse_args()

    wdir = Path(args.wdir).resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_out = out / "multixrank_raw_output"
    raw_out.mkdir(exist_ok=True)

    print("=" * 60)
    print("Running MultiXrank (heterogeneous multiplex + bipartite)")
    print("=" * 60)
    print(f"Working dir : {wdir}")
    print(f"Config      : {args.config}")
    print(f"Output dir  : {out}\n")

    seed_check = check_seeds_in_network(wdir)
    if seed_check and seed_check["missing_seeds"]:
        pct = 100 * len(seed_check["missing_seeds"]) / max(len(seed_check["total_seeds"]), 1)
        print(f"⚠ {pct:.0f}% of seeds not in the LCC. Continuing anyway — "
              f"MultiXrank will just get no signal from those seeds.\n")

    t0 = time.time()
    config_path = wdir / args.config if not Path(args.config).is_absolute() else Path(args.config)
    mxr = multixrank.Multixrank(config=str(config_path), wdir=str(wdir))
    rwr_result = mxr.random_walk_rank()
    print(f"RWR complete in {time.time() - t0:.1f}s\n")

    mxr.write_ranking(rwr_result, path=str(raw_out))
    print(f"Raw output saved -> {raw_out}\n")

    pivot = parse_multixrank_result(rwr_result)
    print(f"Per-cell-type matrix: {len(pivot):,} genes x {len(pivot.columns)} cell types")
    print(f"Non-zero genes: {(pivot > 0).any(axis=1).sum():,}\n")

    ct_path = out / "per_celltype_scores.csv"
    pivot.reset_index().to_csv(ct_path, index=False)
    print(f"Saved per-celltype scores -> {ct_path}")

    agg_fn = AGG_FUNCTIONS[args.aggregation]
    gene_scores = (
        agg_fn(pivot).rename("rwr_score").reset_index()
        .rename(columns={"index": "gene"})
        .sort_values("rwr_score", ascending=False)
        .reset_index(drop=True)
    )
    scores_path = out / "gene_rwr_scores.csv"
    gene_scores.to_csv(scores_path, index=False)

    print(f"Saved aggregated scores -> {scores_path}")
    print(f"  Method       : {args.aggregation}")
    print(f"  Genes ranked : {len(gene_scores):,}")
    print(f"  Non-zero     : {(gene_scores['rwr_score'] > 0).sum():,}")
    if len(gene_scores):
        print(f"  Top score    : {gene_scores['rwr_score'].iloc[0]:.6e}")
    print(f"\nTop 20 genes:\n{gene_scores.head(20).to_string(index=False)}")


if __name__ == "__main__":
    main()