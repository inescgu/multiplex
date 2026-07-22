#!/usr/bin/env python3
"""
step9b_module_diagnostics.py

Exploratory diagnostics for a disease module built by step9_build_disease_module.py.

Answers:
  - How many layers (cell types) are represented in the final module?
  - How many nodes / edges per layer?
  - How many seeds are in the module, broken down by layer?
  - How many seeds are isolated (degree 0) in the final module, overall and per layer?
  - How many seeds are duplicated (literal duplicate lines in seeds.txt, and/or
    seeds whose edges span more than one layer)?
  - Are some layers over/under-represented relative to their size in the module?

Reads:
  <output-dir>/module_graph.pkl      (networkx graph, edges carry a 'cell_types' set
                                       attribute — this is the multiplex layer tag)
  <output-dir>/disease_module.csv    (per-gene table incl. is_seed, dominant_cell_type)
  <input-dir>/seeds/seeds.txt        (original seed list, for duplicate detection)
  <input-dir>/layer_info.json        (optional, for human-readable layer names)

Writes (to --out-prefix, default "module_exploratory"):
  <prefix>_per_layer.csv
  <prefix>_seed_membership.csv
  <prefix>_summary.txt

Usage:
  python step9b_module_diagnostics.py \
      --output-dir 06_disease_module/disease_module_output_consolidated \
      --input-dir  04_multixrank_prep/multixrank_input_consolidated
"""
import argparse
import json
import pickle
from pathlib import Path
from collections import Counter

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load(output_dir, input_dir):
    output_dir = Path(output_dir)
    input_dir = Path(input_dir) if input_dir else None

    graph_path = output_dir / "module_graph.pkl"
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded module graph  : {graph_path}")
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    module_csv = output_dir / "disease_module.csv"
    module_df = pd.read_csv(module_csv)
    module_df["gene"] = module_df["gene"].astype(str).str.strip().str.upper()
    print(f"Loaded module table   : {module_csv} ({len(module_df):,} rows)")

    raw_seeds = []
    layer_info = {}
    if input_dir:
        seeds_path = input_dir / "seeds" / "seeds.txt"
        if seeds_path.exists():
            raw_seeds = [
                line.strip().upper() for line in open(seeds_path) if line.strip()
            ]
            print(f"Loaded seed list      : {seeds_path} ({len(raw_seeds)} lines)")
        else:
            print(f"  [warn] seeds.txt not found at {seeds_path}; "
                  f"falling back to is_seed column in disease_module.csv")

        li_path = input_dir / "layer_info.json"
        if li_path.exists():
            layer_info = json.loads(li_path.read_text())
            print(f"Loaded layer_info.json: {li_path} ({len(layer_info)} layer stems)")

    if not raw_seeds and "is_seed" in module_df.columns:
        raw_seeds = module_df.loc[module_df["is_seed"], "gene"].tolist()

    return G, module_df, raw_seeds, layer_info


def layer_display_name(stem, layer_info):
    info = layer_info.get(stem, {})
    ct = info.get("cell_types", [])
    return ct[0] if ct else stem


# ---------------------------------------------------------------------------
# Layer extraction
# ---------------------------------------------------------------------------
def get_layers(G):
    layers = set()
    for _, _, d in G.edges(data=True):
        layers |= set(d.get("cell_types", set()))
    return sorted(layers)


def per_layer_stats(G, layers, seed_set, layer_info):
    total_nodes = G.number_of_nodes()
    rows = []
    for layer in layers:
        layer_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if layer in d.get("cell_types", set())
        ]
        layer_nodes = set(n for e in layer_edges for n in e)
        n_seeds = len(layer_nodes & seed_set)
        rows.append({
            "layer_stem": layer,
            "layer_display_name": layer_display_name(layer, layer_info),
            "n_nodes": len(layer_nodes),
            "n_edges": len(layer_edges),
            "pct_of_module_nodes": round(100 * len(layer_nodes) / total_nodes, 2)
                                    if total_nodes else 0.0,
            "n_seeds": n_seeds,
            "seed_fraction_of_layer": round(n_seeds / len(layer_nodes), 4)
                                       if layer_nodes else 0.0,
        })
    df = pd.DataFrame(rows).sort_values("n_nodes", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Seed-level diagnostics
# ---------------------------------------------------------------------------
def seed_membership(G, raw_seeds, layer_info):
    module_nodes = set(G.nodes())
    dup_in_file = Counter(raw_seeds)
    duplicated_in_file = {g: c for g, c in dup_in_file.items() if c > 1}

    unique_seeds = sorted(set(raw_seeds))
    rows = []
    for s in unique_seeds:
        in_module = s in module_nodes
        deg = G.degree(s) if in_module else 0
        layers_present = set()
        if in_module:
            for _, _, d in G.edges(s, data=True):
                layers_present |= set(d.get("cell_types", set()))
        rows.append({
            "seed": s,
            "n_times_in_seeds_txt": dup_in_file.get(s, 0),
            "in_final_module": in_module,
            "degree_in_module": deg,
            "isolated_in_module": in_module and deg == 0,
            "n_layers_present_in": len(layers_present),
            "spans_multiple_layers": len(layers_present) > 1,
            "layers": ", ".join(sorted(
                layer_display_name(l, layer_info) for l in layers_present
            )),
        })
    df = pd.DataFrame(rows)
    return df, duplicated_in_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True,
                     help="Directory containing module_graph.pkl / disease_module.csv")
    ap.add_argument("--input-dir", default=None,
                     help="Directory containing seeds/seeds.txt and layer_info.json "
                          "(same --input-dir passed to step9)")
    ap.add_argument("--out-prefix", default="module_exploratory")
    args = ap.parse_args()

    G, module_df, raw_seeds, layer_info = load(args.output_dir, args.input_dir)
    seed_set = set(raw_seeds)

    layers = get_layers(G)

    print("\n" + "=" * 70)
    print("MODULE OVERVIEW")
    print("=" * 70)
    print(f"Total nodes           : {G.number_of_nodes():,}")
    print(f"Total edges           : {G.number_of_edges():,}")
    print(f"Layers represented    : {len(layers)}")
    for l in layers:
        print(f"  - {l}  ({layer_display_name(l, layer_info)})")

    isolated_all = [n for n in G.nodes() if G.degree(n) == 0]
    print(f"\nIsolated nodes (degree 0, whole module): {len(isolated_all):,}")
    isolated_seeds_all = sorted(set(isolated_all) & seed_set)
    print(f"  of which are seeds                    : {len(isolated_seeds_all)}")
    if isolated_seeds_all:
        print(f"    {isolated_seeds_all[:30]}"
              f"{' ...' if len(isolated_seeds_all) > 30 else ''}")

    # --- per layer ---
    pl = per_layer_stats(G, layers, seed_set, layer_info)
    print("\n" + "-" * 70)
    print("PER-LAYER STATS (sorted by node count, largest first)")
    print("-" * 70)
    print(pl.to_string(index=False))
    pl_path = f"{args.out_prefix}_per_layer.csv"
    pl.to_csv(pl_path, index=False)

    if not pl.empty:
        node_share = pl["n_nodes"]
        cv = node_share.std() / node_share.mean() if node_share.mean() else 0
        top = pl.iloc[0]
        print(f"\nLargest layer: {top['layer_display_name']} "
              f"({top['n_nodes']:,} nodes, {top['pct_of_module_nodes']}% of module)")
        print(f"Coefficient of variation of layer sizes: {cv:.2f} "
              f"({'uneven' if cv > 0.5 else 'fairly even'} representation across layers)")

    # --- seed membership ---
    sm, dup_in_file = seed_membership(G, raw_seeds, layer_info)
    print("\n" + "-" * 70)
    print("SEED DIAGNOSTICS")
    print("-" * 70)
    print(f"Seed lines in seeds.txt      : {len(raw_seeds)}")
    print(f"Unique seed genes            : {len(set(raw_seeds))}")
    if dup_in_file:
        print(f"Duplicate lines in seeds.txt : {len(dup_in_file)} gene(s) -> {dup_in_file}")
    else:
        print("Duplicate lines in seeds.txt : none")

    print(f"\nSeeds present in final module     : {sm['in_final_module'].sum()} "
          f"/ {len(sm)}")
    print(f"Seeds missing from final module   : {(~sm['in_final_module']).sum()}")
    print(f"Seeds isolated in final module    : {sm['isolated_in_module'].sum()}")
    print(f"Seeds spanning >1 layer in module : {sm['spans_multiple_layers'].sum()}")

    print("\nSeed counts by layer (seeds present & connected within that layer):")
    for _, row in pl.iterrows():
        print(f"  {row['layer_display_name']:<30} {row['n_seeds']:>4} seeds "
              f"({row['seed_fraction_of_layer']*100:.1f}% of layer's nodes)")

    sm_path = f"{args.out_prefix}_seed_membership.csv"
    sm.to_csv(sm_path, index=False)

    # --- summary file ---
    summary_lines = [
        "=" * 70,
        "MODULE EXPLORATORY DIAGNOSTICS",
        "=" * 70,
        f"Total nodes: {G.number_of_nodes():,}   Total edges: {G.number_of_edges():,}",
        f"Layers represented: {len(layers)}",
        "",
        "Per-layer breakdown:",
        pl.to_string(index=False),
        "",
        f"Isolated nodes overall: {len(isolated_all):,} "
        f"(of which seeds: {len(isolated_seeds_all)})",
        f"Seed lines in seeds.txt: {len(raw_seeds)}  "
        f"| unique: {len(set(raw_seeds))}  "
        f"| duplicate lines: {dup_in_file if dup_in_file else 'none'}",
        f"Seeds in final module: {sm['in_final_module'].sum()} / {len(sm)}",
        f"Seeds isolated in module: {sm['isolated_in_module'].sum()}",
        f"Seeds spanning >1 layer: {sm['spans_multiple_layers'].sum()}",
        "",
        f"Full seed-level table -> {sm_path}",
        f"Full per-layer table  -> {pl_path}",
        "=" * 70,
    ]
    summary_path = f"{args.out_prefix}_summary.txt"
    Path(summary_path).write_text("\n".join(summary_lines))

    print(f"\nSaved: {pl_path}")
    print(f"Saved: {sm_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()