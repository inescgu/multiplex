#!/usr/bin/env python3
"""
Build Disease Module from MultiXrank Output (Cell-Type-Aware)
=============================================================
Given aggregated gene-level RWR scores (and optionally per-cell-type scores),
build a connected (or LCC fallback) disease module.

The per-cell-type scores are used to annotate each module gene with its
dominant cell type — i.e. the layer in which it scored highest.

Outputs
-------
<output-dir>/
├── disease_module.csv          # gene, is_seed, rwr_score, dominant_cell_type,
│                               #   degree_in_module, <cell_type_1>, ...
├── module_graph.pkl
└── module_diagnostics.txt
"""

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_inputs(scores_csv, ref_graph_pkl, seeds_txt,
                input_dir=None, per_ct_csv=None):
    print(f"Loading RWR scores from {scores_csv} ...")
    scores = pd.read_csv(scores_csv)
    print(f"  {len(scores):,} ranked genes")

    print(f"Loading reference graph from {ref_graph_pkl} ...")
    with open(ref_graph_pkl, "rb") as f:
        H = pickle.load(f)
    print(f"  {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

    # Cell-type metadata
    cell_types = []
    if input_dir:
        meta_path = Path(input_dir) / "cell_type_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            cell_types = meta.get("cell_types", [])
            print(f"  Cell types: {cell_types}")
            H.graph["cell_type_metadata"] = meta

    # Per-cell-type score matrix (optional)
    per_ct_df = None
    if per_ct_csv and Path(per_ct_csv).exists():
        per_ct_df = pd.read_csv(per_ct_csv, index_col="gene")
        print(f"  Per-cell-type scores loaded: {per_ct_df.shape}")

    print(f"Loading seeds from {seeds_txt} ...")
    with open(seeds_txt) as f:
        seeds = [line.strip().upper() for line in f if line.strip()]
    print(f"  {len(seeds)} seed genes\n")

    return scores, H, seeds, cell_types, per_ct_df


# ---------------------------------------------------------------------------
# Module construction (unchanged logic, seeds → ranked expansion)
# ---------------------------------------------------------------------------

def module_subgraph(genes, H):
    nodes_in = [g for g in genes if g in H.nodes()]
    return H.subgraph(nodes_in).copy()


def build_module(scores_df, H, seeds, k, max_bridge):
    seeds_in_h  = [s for s in seeds if s in H.nodes()]
    seeds_miss  = [s for s in seeds if s not in H.nodes()]
    seed_set    = set(seeds)
    ranked      = [g for g in scores_df["gene"].tolist()
                   if g not in seed_set and g in H.nodes()]

    module = list(seeds_in_h)
    print(f"Initial module: {len(seeds_in_h)} seeds in reference graph")

    sub = module_subgraph(module, H)
    if nx.is_connected(sub) and sub.number_of_nodes() > 0:
        print("Seeds alone are connected!")
        return module, 0, True, seeds_in_h, seeds_miss

    print("Seeds not fully connected — expanding with ranked genes ...")
    added = 0
    for g in ranked:
        if added >= k + max_bridge:
            break
        module.append(g)
        added += 1
        sub = module_subgraph(module, H)
        is_conn = nx.is_connected(sub) if sub.number_of_nodes() > 0 else False
        if added % 100 == 0 or is_conn:
            ncc = nx.number_connected_components(sub)
            print(f"  +{added:>5} genes, {ncc} component(s), connected={is_conn}")
        if is_conn:
            print(f"Connected after adding {added} genes!")
            return module, added, True, seeds_in_h, seeds_miss

    sub = module_subgraph(module, H)
    fc  = nx.is_connected(sub) if sub.number_of_nodes() > 0 else False
    print(f"Reached limit of {added} genes without full connectivity")
    return module, added, fc, seeds_in_h, seeds_miss


def extract_lcc(module, H):
    sub = module_subgraph(module, H)
    if sub.number_of_nodes() == 0:
        return [], sub
    largest = max(nx.connected_components(sub), key=len)
    lcc_sub = sub.subgraph(largest).copy()
    return sorted(largest), lcc_sub


def diagnose_seed_connectivity(seeds, H):
    sub = module_subgraph(seeds, H)
    if sub.number_of_nodes() == 0:
        return {"seed_components": 0, "largest_seed_cc_size": 0,
                "isolated_seeds": [], "seed_subgraph_edges": 0}
    comps    = list(nx.connected_components(sub))
    largest  = max(comps, key=len) if comps else set()
    isolated = [n for n in sub.nodes() if sub.degree(n) == 0]
    return {
        "seed_components": len(comps),
        "largest_seed_cc_size": len(largest),
        "isolated_seeds": sorted(isolated),
        "seed_subgraph_edges": sub.number_of_edges(),
    }


# ---------------------------------------------------------------------------
# Cell-type annotation
# ---------------------------------------------------------------------------

def annotate_cell_types(final_module: list, per_ct_df: pd.DataFrame | None,
                        cell_types: list) -> pd.DataFrame:
    """
    For each gene in the module, find the cell type where it scored highest
    and include individual per-cell-type score columns.

    Returns a DataFrame indexed by gene with columns:
        dominant_cell_type, <ct1>, <ct2>, ...
    """
    if per_ct_df is None or per_ct_df.empty:
        return pd.DataFrame({"dominant_cell_type": [None] * len(final_module)},
                            index=final_module)

    subset = per_ct_df.reindex(final_module).fillna(0.0)

    # Dominant cell type = column with the highest score
    subset["dominant_cell_type"] = subset.idxmax(axis=1)

    return subset


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def write_diagnostics(path, info):
    lines = []
    lines.append("=" * 70)
    lines.append("DISEASE MODULE DIAGNOSTICS  (cell-type-aware)")
    lines.append("=" * 70)
    lines.append(f"Connectivity mode      : {info['connectivity_mode']}")
    lines.append(f"Reference graph        : {info['ref_path']}")
    lines.append(f"  nodes                : {info['ref_nodes']:,}")
    lines.append(f"  edges                : {info['ref_edges']:,}")
    lines.append(f"Cell types             : {info['cell_types']}")
    lines.append("")
    lines.append("--- SEEDS ---")
    lines.append(f"Seeds provided         : {info['n_seeds_provided']}")
    lines.append(f"Seeds in reference     : {info['n_seeds_in_ref']}")
    lines.append(f"Seeds missing          : {len(info['seeds_missing'])}")
    if info["seeds_missing"]:
        lines.append(f"  (first 20): {info['seeds_missing'][:20]}")
    lines.append("")
    lines.append("--- SEED-ONLY SUBGRAPH ---")
    sd = info["seed_diag"]
    lines.append(f"Connected components : {sd['seed_components']}")
    lines.append(f"Largest seed CC size : {sd['largest_seed_cc_size']}")
    lines.append(f"Seed subgraph edges  : {sd['seed_subgraph_edges']}")
    lines.append(f"Isolated seeds       : {len(sd['isolated_seeds'])}")
    if sd["isolated_seeds"]:
        lines.append(f"  Examples: {sd['isolated_seeds'][:20]}")
    lines.append("")
    lines.append("--- MODULE CONSTRUCTION ---")
    lines.append(f"Top-k requested            : {info['k']}")
    lines.append(f"Max bridging genes allowed : {info['max_bridge']}")
    lines.append(f"Genes added                : {info['n_bridges']}")
    lines.append(f"Full module size           : {info['full_module_size']}")
    lines.append(f"Fully connected            : {info['fully_connected']}")
    if not info["fully_connected"]:
        lines.append(f"  Components in full module: {info['full_module_components']}")
    lines.append("")
    lines.append("--- FINAL MODULE ---")
    lines.append(f"Final module size       : {info['final_module_size']}")
    lines.append(f"Final module edges      : {info['final_module_edges']}")
    lines.append(f"Used LCC fallback       : {info['used_lcc_fallback']}")
    lines.append(f"Seeds in final module   : {info['seeds_in_final']} / {info['n_seeds_in_ref']}")
    lines.append(f"Seeds dropped by LCC    : {info['seeds_dropped_by_lcc']}")
    if info["seeds_dropped_by_lcc"]:
        lines.append(f"  Dropped: {info['dropped_seed_list'][:20]}")
    lines.append("")
    if info["dominant_ct_counts"]:
        lines.append("--- DOMINANT CELL TYPE DISTRIBUTION (module genes) ---")
        for ct, n in sorted(info["dominant_ct_counts"].items(),
                             key=lambda x: -x[1]):
            lines.append(f"  {ct:<35} {n:>6,} genes")
        lines.append("")

    # Suggestions when not fully connected
    if not info["fully_connected"]:
        lines.append("=" * 70)
        lines.append("⚠ MODULE NOT FULLY CONNECTED — SUGGESTIONS")
        lines.append("=" * 70)
        suggestions = []
        if info["connectivity_mode"] == "ppi":
            suggestions.append(
                "• Used --connectivity ppi (strict). Try --connectivity all to include "
                "all cell-type layer edges in the connectivity check."
            )
        if sd["seed_components"] > 1:
            suggestions.append(
                f"• Seeds split into {sd['seed_components']} components on the reference graph. "
                "Consider (a) reviewing whether all seeds belong to the same module, "
                "(b) running RWR per seed subgroup, or (c) increasing --k."
            )
        if info["n_bridges"] >= info["max_bridge"]:
            suggestions.append(
                f"• Bridging hit the cap (--max-bridge={info['max_bridge']}). "
                "Try increasing it (e.g., --max-bridge 2000)."
            )
        if info["used_lcc_fallback"]:
            suggestions.append(
                "• LCC fallback used — non-LCC components discarded. "
                "Set --no-lcc-fallback to keep all genes."
            )
        suggestions.append(
            "• Cell-type layers may be sparse. Check that each layer has enough edges "
            "to allow the walker to bridge between seed clusters."
        )
        for s in suggestions:
            lines.append(s)
        lines.append("")

    lines.append("=" * 70)
    text = "\n".join(lines)
    print(text)
    Path(path).write_text(text)
    print(f"\nDiagnostics saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build disease module from MultiXrank scores (cell-type-aware)"
    )
    parser.add_argument("--scores", required=True,
                        help="gene_rwr_scores.csv (aggregated) from script 2")
    parser.add_argument("--per-celltype-scores",
                        help="per_celltype_scores.csv from script 2 (optional but recommended)")
    parser.add_argument("--input-dir", required=True,
                        help="Output dir from script 1 (contains reference graphs + seeds)")
    parser.add_argument("--output-dir", required=True,
                        help="Where to save module outputs")
    parser.add_argument("--connectivity", choices=["all", "ppi"], default="all",
                        help="Reference for connectivity checks (default: all)")
    parser.add_argument("--k", type=int, default=200,
                        help="Top-k non-seed genes to include (default 200)")
    parser.add_argument("--max-bridge", type=int, default=1000,
                        help="Max bridging genes when chasing connectivity (default 1000)")
    parser.add_argument("--no-lcc-fallback", action="store_true",
                        help="Keep disconnected module instead of extracting LCC")
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_path = (inp / "gene_reference_graph.pkl"
                if args.connectivity == "all"
                else inp / "gene_ppi_reference_graph.pkl")
    seeds_path = inp / "seeds" / "seeds.txt"

    scores, H, seeds, cell_types, per_ct_df = load_inputs(
        args.scores, ref_path, seeds_path,
        input_dir=args.input_dir,
        per_ct_csv=args.per_celltype_scores,
    )

    seed_diag = diagnose_seed_connectivity(seeds, H)
    print(f"Seed-only subgraph: {seed_diag['seed_components']} components, "
          f"largest CC = {seed_diag['largest_seed_cc_size']}\n")

    module, n_bridges, fully_connected, seeds_in_h, seeds_missing = build_module(
        scores, H, seeds, args.k, args.max_bridge
    )

    full_sub = module_subgraph(module, H)
    full_cc  = (nx.number_connected_components(full_sub)
                if full_sub.number_of_nodes() else 0)

    used_lcc = False
    if not fully_connected and not args.no_lcc_fallback:
        print("Applying LCC fallback ...")
        module_lcc, lcc_sub = extract_lcc(module, H)
        print(f"  LCC: {len(module_lcc)} genes (dropped {len(module) - len(module_lcc)})")
        final_module = module_lcc
        final_sub    = lcc_sub
        used_lcc     = True
    else:
        final_module = module
        final_sub    = full_sub

    seeds_in_final = sum(1 for s in seeds_in_h if s in set(final_module))
    dropped_seeds  = sorted(set(seeds_in_h) - set(final_module))

    # Cell-type annotation
    ct_annotation = annotate_cell_types(final_module, per_ct_df, cell_types)
    dominant_ct_counts = (
        ct_annotation["dominant_cell_type"].value_counts().to_dict()
        if "dominant_cell_type" in ct_annotation.columns
        else {}
    )

    # Build output CSV
    score_lookup = dict(zip(scores["gene"], scores["rwr_score"]))
    seed_set     = set(seeds)
    rows = [{
        "gene":              g,
        "is_seed":           g in seed_set,
        "rwr_score":         score_lookup.get(g, 0.0),
        "degree_in_module":  final_sub.degree(g) if g in final_sub.nodes() else 0,
    } for g in final_module]
    module_df = pd.DataFrame(rows).sort_values(
        ["is_seed", "rwr_score"], ascending=[False, False]
    ).reset_index(drop=True)

    # Merge cell-type annotation
    if not ct_annotation.empty:
        module_df = module_df.merge(
            ct_annotation.reset_index().rename(columns={"index": "gene"}),
            on="gene", how="left"
        )

    module_df.to_csv(out / "disease_module.csv", index=False)
    print(f"Saved disease module CSV → {out / 'disease_module.csv'}")
    if "dominant_cell_type" in module_df.columns:
        print("  dominant_cell_type distribution:")
        for ct, n in sorted(dominant_ct_counts.items(), key=lambda x: -x[1]):
            print(f"    {ct:<35} {n:>6,} genes")

    with open(out / "module_graph.pkl", "wb") as f:
        pickle.dump(final_sub, f, protocol=4)
    print(f"Saved module graph → {out / 'module_graph.pkl'}")

    info = {
        "connectivity_mode":    args.connectivity,
        "ref_path":             str(ref_path),
        "ref_nodes":            H.number_of_nodes(),
        "ref_edges":            H.number_of_edges(),
        "cell_types":           cell_types,
        "n_seeds_provided":     len(seeds),
        "n_seeds_in_ref":       len(seeds_in_h),
        "seeds_missing":        seeds_missing,
        "seed_diag":            seed_diag,
        "k":                    args.k,
        "max_bridge":           args.max_bridge,
        "n_bridges":            n_bridges,
        "full_module_size":     len(module),
        "fully_connected":      fully_connected,
        "full_module_components": full_cc,
        "final_module_size":    len(final_module),
        "final_module_edges":   final_sub.number_of_edges(),
        "used_lcc_fallback":    used_lcc,
        "seeds_in_final":       seeds_in_final,
        "seeds_dropped_by_lcc": len(dropped_seeds),
        "dropped_seed_list":    dropped_seeds,
        "dominant_ct_counts":   dominant_ct_counts,
    }
    write_diagnostics(out / "module_diagnostics.txt", info)


if __name__ == "__main__":
    main()