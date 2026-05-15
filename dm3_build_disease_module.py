#!/usr/bin/env python3
"""
Build Disease Module from MultiXrank Output
============================================
Given gene-level RWR scores and a reference graph, build a connected
(or LCC fallback) disease module.

Outputs
-------
<output-dir>/
├── disease_module.csv
├── module_graph.pkl
└── module_diagnostics.txt
"""

import os
import argparse
import pickle
from pathlib import Path
import json
import pandas as pd
import networkx as nx


def load_inputs(scores_csv, ref_graph_pkl, seeds_txt, input_dir=None):
    print(f"Loading RWR scores from {scores_csv} ...")
    scores = pd.read_csv(scores_csv)
    print(f"  {len(scores):,} ranked genes")

    print(f"Loading reference graph from {ref_graph_pkl} ...")
    with open(ref_graph_pkl, "rb") as f:
        H = pickle.load(f)
    print(f"  {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

    # Load edge type metadata if available
    if input_dir:
        metadata_path = Path(input_dir) / "edge_type_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            print(f"  Edge types: {', '.join(metadata['edge_types'])}")
            H.graph["edge_type_metadata"] = metadata

    print(f"Loading seeds from {seeds_txt} ...")
    with open(seeds_txt) as f:
        seeds = [line.strip().upper() for line in f if line.strip()]
    print(f"  {len(seeds)} seed genes\n")

    return scores, H, seeds


def module_subgraph(genes, H):
    """Return subgraph of H induced by genes that exist in H."""
    nodes_in = [g for g in genes if g in H.nodes()]
    return H.subgraph(nodes_in).copy()


def build_module(scores_df, H, seeds, k, max_bridge):
    """
    Build module by adding ranked genes one by one and checking connectivity after each.

    Returns
    -------
    module_genes : list[str]   final module (full set, may be disconnected)
    n_bridges    : int          number of bridging genes added
    fully_connected : bool
    """
    seeds_in_h = [s for s in seeds if s in H.nodes()]
    seeds_missing_from_ref = [s for s in seeds if s not in H.nodes()]

    # Ranked non-seed pool
    seed_set = set(seeds)
    ranked = [g for g in scores_df["gene"].tolist() if g not in seed_set and g in H.nodes()]

    # Start with seeds
    module = list(seeds_in_h)
    print(f"Initial module: {len(seeds_in_h)} seeds")

    sub = module_subgraph(module, H)
    if nx.is_connected(sub) and sub.number_of_nodes() > 0:
        print("Seeds alone are connected!")
        return module, 0, True, seeds_in_h, seeds_missing_from_ref

    # Add ranked genes one by one
    print("Seeds not connected — adding ranked genes one by one...")
    added = 0
    for i, g in enumerate(ranked):
        if added >= k + max_bridge:
            break
        
        module.append(g)
        added += 1
        
        sub = module_subgraph(module, H)
        is_connected = nx.is_connected(sub) if sub.number_of_nodes() > 0 else False
        
        if added % 100 == 0 or is_connected:
            ncc = nx.number_connected_components(sub)
            print(f"  +{added} genes, {ncc} components, connected={is_connected}")
        
        if is_connected:
            print(f"Connected after adding {added} genes!")
            return module, added, True, seeds_in_h, seeds_missing_from_ref

    sub = module_subgraph(module, H)
    fully_connected = nx.is_connected(sub) if sub.number_of_nodes() > 0 else False
    print(f"Reached limit of {added} genes without full connectivity")
    return module, added, fully_connected, seeds_in_h, seeds_missing_from_ref


def extract_lcc(module, H):
    """Return the largest connected component of the module subgraph."""
    sub = module_subgraph(module, H)
    if sub.number_of_nodes() == 0:
        return [], sub
    components = list(nx.connected_components(sub))
    largest = max(components, key=len)
    lcc_sub = sub.subgraph(largest).copy()
    return sorted(largest), lcc_sub


def diagnose_seed_connectivity(seeds, H):
    """How are seeds distributed across components of H restricted to seeds?"""
    sub = module_subgraph(seeds, H)
    if sub.number_of_nodes() == 0:
        return {"seed_components": 0, "largest_seed_cc_size": 0,
                "isolated_seeds": [], "seed_subgraph_edges": 0}
    components = list(nx.connected_components(sub))
    largest = max(components, key=len) if components else set()
    isolated = [n for n in sub.nodes() if sub.degree(n) == 0]
    return {
        "seed_components": len(components),
        "largest_seed_cc_size": len(largest),
        "isolated_seeds": sorted(isolated),
        "seed_subgraph_edges": sub.number_of_edges(),
    }


def write_diagnostics(path, info):
    lines = []
    lines.append("=" * 70)
    lines.append("DISEASE MODULE DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append(f"Connectivity mode      : {info['connectivity_mode']}")
    lines.append(f"Reference graph        : {info['ref_path']}")
    lines.append(f"  nodes                : {info['ref_nodes']:,}")
    lines.append(f"  edges                : {info['ref_edges']:,}")
    lines.append("")
    lines.append("--- SEEDS ---")
    lines.append(f"Seeds provided         : {info['n_seeds_provided']}")
    lines.append(f"Seeds in reference     : {info['n_seeds_in_ref']}")
    lines.append(f"Seeds missing from ref : {len(info['seeds_missing'])}")
    if info["seeds_missing"]:
        lines.append(f"  Missing seeds (up to 20): {info['seeds_missing'][:20]}")
    lines.append("")
    lines.append("--- SEED-ONLY SUBGRAPH (before RWR-driven expansion) ---")
    sd = info["seed_diag"]
    lines.append(f"Connected components among seeds : {sd['seed_components']}")
    lines.append(f"Largest seed-only CC size        : {sd['largest_seed_cc_size']}")
    lines.append(f"Seed-only subgraph edges         : {sd['seed_subgraph_edges']}")
    lines.append(f"Isolated seeds (in ref but no edges to other seeds): {len(sd['isolated_seeds'])}")
    if sd["isolated_seeds"]:
        lines.append(f"  Examples: {sd['isolated_seeds'][:20]}")
    lines.append("")
    lines.append("--- MODULE CONSTRUCTION ---")
    lines.append(f"Top-k requested            : {info['k']}")
    lines.append(f"Max bridging genes allowed : {info['max_bridge']}")
    lines.append(f"Genes added                : {info['n_bridges']}")
    lines.append(f"Full module size           : {info['full_module_size']}")
    lines.append(f"Full module fully connected: {info['fully_connected']}")
    if not info["fully_connected"]:
        lines.append(f"  Components in full module: {info['full_module_components']}")
    lines.append("")
    lines.append("--- FINAL MODULE (after LCC fallback if needed) ---")
    lines.append(f"Final module size       : {info['final_module_size']}")
    lines.append(f"Final module edges      : {info['final_module_edges']}")
    lines.append(f"Used LCC fallback       : {info['used_lcc_fallback']}")
    lines.append(f"Seeds in final module   : {info['seeds_in_final']} / {info['n_seeds_in_ref']}")
    lines.append(f"Seeds dropped by LCC    : {info['seeds_dropped_by_lcc']}")
    if info["seeds_dropped_by_lcc"] > 0:
        lines.append(f"  Dropped seeds: {info['dropped_seed_list'][:20]}")
    lines.append("")

    # Suggestions
    if not info["fully_connected"]:
        lines.append("=" * 70)
        lines.append("⚠ MODULE NOT FULLY CONNECTED — SUGGESTIONS")
        lines.append("=" * 70)
        suggestions = []
        if info["connectivity_mode"] == "ppi":
            suggestions.append(
                "• You used --connectivity ppi. PPI-only is strict: try --connectivity all "
                "to include pathway and LR edges in the connectivity check."
            )
        if sd["seed_components"] > 1:
            suggestions.append(
                f"• Your seeds split into {sd['seed_components']} components even on the full "
                "reference graph. This suggests biologically distinct seed subgroups. Consider: "
                "(a) reviewing whether all seeds belong to the same disease module, "
                "(b) running RWR separately per seed subgroup, or "
                "(c) increasing --k to give the walker more room to bridge them."
            )
        if sd["isolated_seeds"]:
            suggestions.append(
                f"• {len(sd['isolated_seeds'])} seeds have no edges to other seeds in the "
                "reference graph. These are likely poorly connected genes — check whether "
                "they should remain as seeds or be moved to a validation set."
            )
        if info["n_bridges"] >= info["max_bridge"]:
            suggestions.append(
                f"• Bridging hit the cap (--max-bridge={info['max_bridge']}). "
                "Try increasing it (e.g., --max-bridge 2000)."
            )
        if info["used_lcc_fallback"]:
            suggestions.append(
                "• LCC fallback was used — non-LCC components were discarded. "
                "If you want to preserve all genes, set --no-lcc-fallback and report the "
                "module as multi-component."
            )
        suggestions.append(
            "• Consider lowering MultiXrank's restart probability `r` (more diffusion) or "
            "tuning `delta` (more inter-layer jumping) in script 1, then re-running."
        )
        suggestions.append(
            "• Check that pathway and LR layers are well-populated: sparse inter-layer "
            "edges limit the walker's ability to bridge disconnected seed clusters."
        )
        for s in suggestions:
            lines.append(s)
        lines.append("")

    lines.append("=" * 70)
    text = "\n".join(lines)
    print(text)
    Path(path).write_text(text)
    print(f"\nDiagnostics saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Build disease module from MultiXrank scores")
    parser.add_argument("--scores", required=True, help="gene_rwr_scores.csv from script 2")
    parser.add_argument("--input-dir", required=True,
                        help="Output dir from script 1 (contains reference graphs + seeds)")
    parser.add_argument("--output-dir", required=True, help="Where to save module + diagnostics")
    parser.add_argument("--connectivity", choices=["all", "ppi"], default="all",
                        help="Reference for connectivity: 'all'=PPI∪pathway∪LR, 'ppi'=PPI only")
    parser.add_argument("--k", type=int, default=200, help="Top-k non-seed genes to include (default 200)")
    parser.add_argument("--max-bridge", type=int, default=1000,
                        help="Max bridging genes to add when chasing connectivity (default 1000)")
    parser.add_argument("--no-lcc-fallback", action="store_true",
                        help="Disable LCC fallback (keep disconnected module as-is)")
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Pick reference graph
    if args.connectivity == "all":
        ref_path = inp / "gene_reference_graph.pkl"
    else:
        ref_path = inp / "gene_ppi_reference_graph.pkl"

    seeds_path = inp / "seeds" / "seeds.txt"

    scores, H, seeds = load_inputs(args.scores, ref_path, seeds_path, input_dir=args.input_dir)

    # Diagnose seed connectivity before doing anything
    seed_diag = diagnose_seed_connectivity(seeds, H)
    print(f"Seed-only subgraph: {seed_diag['seed_components']} components, "
          f"largest CC = {seed_diag['largest_seed_cc_size']}\n")

    # Build module
    module, n_bridges, fully_connected, seeds_in_h, seeds_missing = build_module(
        scores, H, seeds, args.k, args.max_bridge
    )

    full_sub = module_subgraph(module, H)
    full_components = nx.number_connected_components(full_sub) if full_sub.number_of_nodes() else 0

    # LCC fallback
    used_lcc = False
    if not fully_connected and not args.no_lcc_fallback:
        print("Applying LCC fallback...")
        module_lcc, lcc_sub = extract_lcc(module, H)
        print(f"  LCC has {len(module_lcc)} genes (dropped {len(module) - len(module_lcc)})")
        final_module = module_lcc
        final_sub = lcc_sub
        used_lcc = True
    else:
        final_module = module
        final_sub = full_sub

    seeds_in_final = sum(1 for s in seeds_in_h if s in set(final_module))
    dropped_seeds = sorted(set(seeds_in_h) - set(final_module))

    # Save module CSV
    score_lookup = dict(zip(scores["gene"], scores["rwr_score"]))
    seed_set = set(seeds)
    rows = [{
        "gene": g,
        "is_seed": g in seed_set,
        "rwr_score": score_lookup.get(g, 0.0),
        "degree_in_module": final_sub.degree(g) if g in final_sub.nodes() else 0,
    } for g in final_module]
    module_df = pd.DataFrame(rows).sort_values(
        ["is_seed", "rwr_score"], ascending=[False, False]
    )
    module_df.to_csv(out / "disease_module.csv", index=False)
    print(f"Saved disease module CSV → {out / 'disease_module.csv'}")

    # Save module subgraph
    with open(out / "module_graph.pkl", "wb") as f:
        pickle.dump(final_sub, f, protocol=4)
    print(f"Saved module graph → {out / 'module_graph.pkl'}")

    # Diagnostics
    info = {
        "connectivity_mode": args.connectivity,
        "ref_path": str(ref_path),
        "ref_nodes": H.number_of_nodes(),
        "ref_edges": H.number_of_edges(),
        "n_seeds_provided": len(seeds),
        "n_seeds_in_ref": len(seeds_in_h),
        "seeds_missing": seeds_missing,
        "seed_diag": seed_diag,
        "k": args.k,
        "max_bridge": args.max_bridge,
        "n_bridges": n_bridges,
        "full_module_size": len(module),
        "fully_connected": fully_connected,
        "full_module_components": full_components,
        "final_module_size": len(final_module),
        "final_module_edges": final_sub.number_of_edges(),
        "used_lcc_fallback": used_lcc,
        "seeds_in_final": seeds_in_final,
        "seeds_dropped_by_lcc": len(dropped_seeds),
        "dropped_seed_list": dropped_seeds,
    }
    write_diagnostics(out / "module_diagnostics.txt", info)


if __name__ == "__main__":
    main()