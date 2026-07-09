#!/usr/bin/env python3
"""
Build Disease Module from MultiXrank Output (Cell-Type-Aware)
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
                input_dir=None, per_ct_csv=None,
                exclude_seeds=None):
    print(f"Loading RWR scores from {scores_csv} ...")
    scores = pd.read_csv(scores_csv)
    scores["gene"] = scores["gene"].astype(str).str.strip().str.upper()
    print(f"  {len(scores):,} ranked genes")

    print(f"\nLoading reference graph from {ref_graph_pkl} ...")
    with open(ref_graph_pkl, "rb") as f:
        H = pickle.load(f)
    mapping = {n: str(n).strip().upper() for n in H.nodes()}
    H = nx.relabel_nodes(H, mapping)
    print(f"  {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

    cell_types = []
    layer_info = {}
    if input_dir:
        meta_path = Path(input_dir) / "cell_type_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            cell_types = meta.get("cell_types", [])
            H.graph["cell_type_metadata"] = meta

        li_path = Path(input_dir) / "layer_info.json"
        if li_path.exists():
            with open(li_path) as f:
                layer_info = json.load(f)

    per_ct_df = None
    if per_ct_csv and Path(per_ct_csv).exists():
        per_ct_df = pd.read_csv(per_ct_csv)
        gene_col = next(
            (c for c in per_ct_df.columns
             if c.lower() in ("gene", "node", "index")),
            per_ct_df.columns[0]
        )
        per_ct_df = per_ct_df.rename(columns={gene_col: "gene"})
        per_ct_df["gene"] = per_ct_df["gene"].astype(str).str.strip().str.upper()
        per_ct_df = per_ct_df.set_index("gene")
        print(f"  Per-cell-type scores: {per_ct_df.shape[0]:,} genes "
              f"× {per_ct_df.shape[1]} layers")

    print(f"\nLoading seeds from {seeds_txt} ...")
    with open(seeds_txt) as f:
        raw_seeds = [line.strip().upper() for line in f if line.strip()]

    # Apply exclusions
    exclude_set = set(g.strip().upper() for g in (exclude_seeds or []))
    if exclude_set:
        excluded_found = [s for s in raw_seeds if s in exclude_set]
        seeds = [s for s in raw_seeds if s not in exclude_set]
        print(f"  {len(raw_seeds)} seeds loaded")
        print(f"  Excluding {len(excluded_found)} seed(s): {excluded_found}")
        print(f"  {len(seeds)} seeds remaining after exclusion")
    else:
        seeds = raw_seeds
        print(f"  {len(seeds)} seed genes")

    print()
    return scores, H, seeds, cell_types, per_ct_df, layer_info

def filter_to_lcc_per_layer(H):
    """
    For each layer (identified via the 'cell_types' set on edges),
    keep only nodes in that layer's LCC. A node is kept in H if it
    belongs to the LCC of at least one layer it participates in.
    """
    # Collect all layer names across all edges
    layers = set()
    for _, _, d in H.edges(data=True):
        layers |= d.get("cell_types", set())

    if not layers:
        print("  [lcc-per-layer] No 'cell_types' edge attribute found — skipping.")
        return H

    kept_nodes = set()
    for layer in sorted(layers):
        # Nodes and edges belonging to this layer
        layer_edges = [
            (u, v) for u, v, d in H.edges(data=True)
            if layer in d.get("cell_types", set())
        ]
        layer_nodes = set(n for e in layer_edges for n in e)
        Lsub = H.subgraph(layer_nodes)

        if Lsub.number_of_nodes() == 0:
            print(f"  [lcc-per-layer] Layer '{layer}': empty, skipping.")
            continue

        lcc = max(nx.connected_components(Lsub), key=len)
        print(f"  [lcc-per-layer] Layer '{layer}': "
              f"{Lsub.number_of_nodes():,} → {len(lcc):,} nodes "
              f"(dropped {Lsub.number_of_nodes() - len(lcc):,})")
        kept_nodes |= lcc

    H_filtered = H.subgraph(kept_nodes).copy()
    print(f"\n  [lcc-per-layer] Total: {H.number_of_nodes():,} → "
          f"{H_filtered.number_of_nodes():,} nodes, "
          f"{H.number_of_edges():,} → "
          f"{H_filtered.number_of_edges():,} edges")
    return H_filtered

# ---------------------------------------------------------------------------
# Module construction
# ---------------------------------------------------------------------------
def module_subgraph(genes, H):
    nodes_in = [g for g in genes if g in H.nodes()]
    return H.subgraph(nodes_in).copy()


def build_module(scores_df, H, seeds, k, max_bridge, exclude_set=None):
    exclude_set = exclude_set or set()
    seeds_in_h = [s for s in seeds if s in H.nodes()]
    seeds_miss = [s for s in seeds if s not in H.nodes()]
    seed_set   = set(seeds)

    ranked = [g for g in scores_df["gene"].tolist()
              if g not in seed_set
              and g not in exclude_set          # ← ADD THIS
              and g in H.nodes()]
    print(f"Ranked non-seed genes available: {len(ranked):,}")

    module = list(seeds_in_h)
    sub = module_subgraph(module, H)

    if sub.number_of_nodes() > 0 and nx.is_connected(sub):
        print("Seeds alone are connected!")
        return module, 0, True, seeds_in_h, seeds_miss

    print("Seeds not fully connected — expanding with ranked genes ...")
    added = 0
    for g in ranked:
        if added >= k + max_bridge:
            break

        module.append(g)
        added += 1

        # Check connectivity after EVERY gene added
        sub = module_subgraph(module, H)
        if nx.is_connected(sub):
            print(f"  Connected after adding {added} gene(s)!")
            return module, added, True, seeds_in_h, seeds_miss
        
        # Faster alternative: only check if the new gene
        # has neighbours in the current module
        # (only possible connectivity change is reducing component count)
        # new_gene_neighbours = set(H.neighbors(g)) & set(module[:-1])
        # if new_gene_neighbours:
        #     sub = module_subgraph(module, H)
        #     if nx.is_connected(sub):
        #         print(f"  Connected after adding {added} gene(s)!")
        #         return module, added, True, seeds_in_h, seeds_miss

        # Progress report every 200 genes (print only, no connectivity logic)
        if added % 200 == 0:
            ncc = nx.number_connected_components(sub)
            print(f"  +{added:>5} genes → {sub.number_of_nodes()} nodes, "
                  f"{ncc} component(s)")

    sub = module_subgraph(module, H)
    fc  = nx.is_connected(sub) if sub.number_of_nodes() > 0 else False
    ncc = nx.number_connected_components(sub) if sub.number_of_nodes() > 0 else 0
    print(f"Reached limit of {added} genes. Connected={fc}, components={ncc}")
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
        "seed_components":      len(comps),
        "largest_seed_cc_size": len(largest),
        "isolated_seeds":       sorted(isolated),
        "seed_subgraph_edges":  sub.number_of_edges(),
    }


# ---------------------------------------------------------------------------
# Cell-type annotation
# ---------------------------------------------------------------------------
def annotate_cell_types(final_module, per_ct_df, cell_types, layer_info):
    if per_ct_df is None or per_ct_df.empty:
        return pd.DataFrame(
            {"dominant_cell_type": [None] * len(final_module)},
            index=final_module,
        )

    subset = per_ct_df.reindex(final_module).fillna(0.0)

    stem_to_display = {}
    for stem, info in layer_info.items():
        ct_list = info.get("cell_types", [])
        stem_to_display[stem] = ct_list[0] if ct_list else stem

    rename_map = {col: stem_to_display.get(col, col) for col in subset.columns}
    subset = subset.rename(columns=rename_map)

    score_cols = list(subset.columns)
    subset["dominant_cell_type"] = subset[score_cols].idxmax(axis=1)

    all_zero = (subset[score_cols] == 0).all(axis=1)
    subset.loc[all_zero, "dominant_cell_type"] = "unknown"

    print(f"\nCell-type annotation:")
    print(f"  Genes annotated : {(~all_zero).sum():,}")
    print(f"  All-zero genes  : {all_zero.sum():,}")
    print(f"  Unique dominant : {subset['dominant_cell_type'].nunique()}")

    return subset


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def write_diagnostics(path, info):
    lines = ["=" * 70,
             "DISEASE MODULE DIAGNOSTICS  (cell-type-aware)",
             "=" * 70,
             f"Connectivity mode      : {info['connectivity_mode']}",
             f"Reference graph        : {info['ref_path']}",
             f"  nodes                : {info['ref_nodes']:,}",
             f"  edges                : {info['ref_edges']:,}",
             "",
             "--- SEEDS ---",
             f"Seeds provided         : {info['n_seeds_provided']}",
             f"Seeds excluded         : {info['n_seeds_excluded']}",  # NEW
             f"  Excluded list        : {info['excluded_seeds']}",    # NEW
             f"Seeds after exclusion  : {info['n_seeds_after_exclusion']}",  # NEW
             f"Seeds in reference     : {info['n_seeds_in_ref']}",
             f"Seeds missing          : {len(info['seeds_missing'])}",
             ]
    if info["seeds_missing"]:
        lines.append(f"  Missing: {info['seeds_missing']}")

    sd = info["seed_diag"]
    lines += ["",
              "--- SEED-ONLY SUBGRAPH ---",
              f"Connected components   : {sd['seed_components']}",
              f"Largest seed CC size   : {sd['largest_seed_cc_size']}",
              f"Seed subgraph edges    : {sd['seed_subgraph_edges']}",
              f"Isolated seeds         : {len(sd['isolated_seeds'])}",
              ]
    if sd["isolated_seeds"]:
        lines.append(f"  Examples: {sd['isolated_seeds'][:20]}")

    lines += ["",
              "--- MODULE CONSTRUCTION ---",
              f"Top-k requested            : {info['k']}",
              f"Max bridging genes allowed : {info['max_bridge']}",
              f"Genes added                : {info['n_bridges']}",
              f"Full module size           : {info['full_module_size']}",
              f"Fully connected            : {info['fully_connected']}",
              ]
    if not info["fully_connected"]:
        lines.append(
            f"  Components in full module: {info['full_module_components']}"
        )

    lines += ["",
              "--- FINAL MODULE ---",
              f"Final module size       : {info['final_module_size']}",
              f"Final module edges      : {info['final_module_edges']}",
              f"Used LCC fallback       : {info['used_lcc_fallback']}",
              f"Seeds in final module   : {info['seeds_in_final']} / "
              f"{info['n_seeds_in_ref']}",
              f"Seeds dropped by LCC    : {info['seeds_dropped_by_lcc']}",
              ]
    if info["seeds_dropped_by_lcc"]:
        lines.append(f"  Dropped: {info['dropped_seed_list'][:20]}")

    if info["dominant_ct_counts"]:
        lines += ["", "--- DOMINANT CELL TYPE DISTRIBUTION ---"]
        for ct, n in sorted(info["dominant_ct_counts"].items(),
                             key=lambda x: -x[1]):
            lines.append(f"  {ct:<40} {n:>6,} genes")

    if not info["fully_connected"]:
        lines += ["", "=" * 70,
                  "⚠  MODULE NOT FULLY CONNECTED — SUGGESTIONS",
                  "=" * 70]
        if sd["seed_components"] > 1:
            lines.append(
                f"• Seeds split into {sd['seed_components']} components. "
                "Consider increasing --k or reviewing seed list."
            )
        if info["n_bridges"] >= info["max_bridge"]:
            lines.append(
                f"• Hit --max-bridge cap ({info['max_bridge']}). "
                "Try increasing it."
            )
        if info["used_lcc_fallback"]:
            lines.append(
                "• LCC fallback used. Use --no-lcc-fallback to keep all genes."
            )

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
    parser.add_argument("--scores",              required=True)
    parser.add_argument("--per-celltype-scores", default=None)
    parser.add_argument("--input-dir",           required=True)
    parser.add_argument("--output-dir",          required=True)
    parser.add_argument("--connectivity",
                        choices=["all", "ppi"], default="all")
    parser.add_argument("--k",          type=int, default=300)
    parser.add_argument("--max-bridge", type=int, default=2000)
    parser.add_argument("--no-lcc-fallback", action="store_true")
    parser.add_argument(
        "--lcc-per-layer",
        action="store_true",
        help="Restrict each layer to its largest connected component before building the module.",
    )
    # NEW: exclusion argument — accepts one or more gene symbols
    parser.add_argument(
        "--exclude-seeds",
        nargs="+",
        default=[],
        metavar="GENE",
        help=(
            "Seed gene(s) to exclude from module building. "
            "Useful when one seed is an outlier requiring many bridging genes. "
            "Example: --exclude-seeds IAPP SLC12A5"
        ),
    )
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_path   = (inp / "gene_reference_graph.pkl"
                  if args.connectivity == "all"
                  else inp / "gene_ppi_reference_graph.pkl")
    seeds_path = inp / "seeds" / "seeds.txt"

    # Normalise exclusion list
    exclude_seeds = [g.strip().upper() for g in args.exclude_seeds]

    scores, H, seeds, cell_types, per_ct_df, layer_info = load_inputs(
        args.scores, ref_path, seeds_path,
        input_dir=args.input_dir,
        per_ct_csv=args.per_celltype_scores,
        exclude_seeds=exclude_seeds,
    )
    # ── NEW: optionally restrict each layer to its LCC ───────────────────
    if args.lcc_per_layer:
        print("\nFiltering each layer to its largest connected component ...")
        H = filter_to_lcc_per_layer(H)
        print()
    # ─────────────────────────────────────────────────────────────────────

    seed_diag = diagnose_seed_connectivity(seeds, H)
    print(f"Seed-only subgraph: {seed_diag['seed_components']} components, "
          f"largest CC = {seed_diag['largest_seed_cc_size']}\n")

    module, n_bridges, fully_connected, seeds_in_h, seeds_missing = (
        build_module(scores, H, seeds, args.k, args.max_bridge,
                     exclude_set=set(exclude_seeds))   # ← ADD THIS
    )
    
    # ── NEW: show component breakdown before LCC ──────────────────────────
    if not fully_connected:
        full_sub_diag = module_subgraph(module, H)
        components = sorted(nx.connected_components(full_sub_diag),
                            key=len, reverse=True)
        seed_set_diag = set(seeds_in_h)
        print("\nComponent size breakdown (before LCC):")
        for i, comp in enumerate(components):
            seeds_in_comp = [g for g in comp if g in seed_set_diag]
            non_seeds_in_comp = [g for g in comp if g not in seed_set_diag]
            line = (f"  Component {i+1}: {len(comp):>6,} genes, "
                    f"{len(seeds_in_comp):>3} seeds")
            if seeds_in_comp:
                line += f"  → seeds: {seeds_in_comp}"
            if non_seeds_in_comp and len(non_seeds_in_comp) < 10:
                line += f"  → non-seeds: {non_seeds_in_comp}"
            print(line)
        print()
    # ─────────────────────────────────────────────────────────────────────
    
    full_sub = module_subgraph(module, H)
    full_cc  = (nx.number_connected_components(full_sub)
                if full_sub.number_of_nodes() else 0)
    
    used_lcc = False
    if not fully_connected and not args.no_lcc_fallback:
        print("Applying LCC fallback ...")
        module_lcc, lcc_sub = extract_lcc(module, H)
        print(f"  LCC: {len(module_lcc)} genes "
              f"(dropped {len(module) - len(module_lcc)})")
        final_module = module_lcc
        final_sub    = lcc_sub
        used_lcc     = True
    else:
        final_module = module
        final_sub    = full_sub

    seeds_in_final = sum(1 for s in seeds_in_h if s in set(final_module))
    dropped_seeds  = sorted(set(seeds_in_h) - set(final_module))

    ct_annotation = annotate_cell_types(
        final_module, per_ct_df, cell_types, layer_info
    )
    dominant_ct_counts = (
        ct_annotation["dominant_cell_type"].value_counts().to_dict()
        if "dominant_cell_type" in ct_annotation.columns
        else {}
    )

    score_lookup = dict(zip(scores["gene"], scores["rwr_score"]))
    # Original full seed list (before exclusion) for is_seed flag
    original_seeds = set(
        line.strip().upper()
        for line in open(seeds_path)
        if line.strip()
    )
    rows = [{
        "gene":             g,
        "is_seed":          g in original_seeds,  # reflects original seeds
        "is_excluded_seed": g in set(exclude_seeds),
        "rwr_score":        score_lookup.get(g, 0.0),
        "degree_in_module": (final_sub.degree(g)
                             if g in final_sub.nodes() else 0),
    } for g in final_module]

    module_df = (
        pd.DataFrame(rows)
        .sort_values(["is_seed", "rwr_score"], ascending=[False, False])
        .reset_index(drop=True)
    )

    if not ct_annotation.empty:
        ct_reset = ct_annotation.reset_index()
        if "index" in ct_reset.columns:
            ct_reset = ct_reset.rename(columns={"index": "gene"})
        elif ct_reset.columns[0] != "gene":
            ct_reset = ct_reset.rename(columns={ct_reset.columns[0]: "gene"})
        ct_reset["gene"] = ct_reset["gene"].astype(str).str.strip().str.upper()
        module_df = module_df.merge(ct_reset, on="gene", how="left")

    module_csv = out / "disease_module.csv"
    module_df.to_csv(module_csv, index=False)
    print(f"\nSaved disease module → {module_csv}")
    print(f"  {len(module_df):,} genes, {module_df['is_seed'].sum()} seeds")
    if exclude_seeds:
        print(f"  Excluded seeds    : {exclude_seeds}")

    with open(out / "module_graph.pkl", "wb") as f:
        pickle.dump(final_sub, f, protocol=4)
    print(f"Saved module graph → {out / 'module_graph.pkl'}")

    # Load original seed count for diagnostics
    all_original_seeds = [
        line.strip().upper()
        for line in open(seeds_path)
        if line.strip()
    ]

    info = {
        "connectivity_mode":        args.connectivity,
        "ref_path":                 str(ref_path),
        "ref_nodes":                H.number_of_nodes(),
        "ref_edges":                H.number_of_edges(),
        "cell_types":               cell_types,
        "n_seeds_provided":         len(all_original_seeds),
        "n_seeds_excluded":         len(exclude_seeds),           # NEW
        "excluded_seeds":           exclude_seeds,                 # NEW
        "n_seeds_after_exclusion":  len(seeds),                   # NEW
        "n_seeds_in_ref":           len(seeds_in_h),
        "seeds_missing":            seeds_missing,
        "seed_diag":                seed_diag,
        "k":                        args.k,
        "max_bridge":               args.max_bridge,
        "n_bridges":                n_bridges,
        "full_module_size":         len(module),
        "fully_connected":          fully_connected,
        "full_module_components":   full_cc,
        "final_module_size":        len(final_module),
        "final_module_edges":       final_sub.number_of_edges(),
        "used_lcc_fallback":        used_lcc,
        "seeds_in_final":           seeds_in_final,
        "seeds_dropped_by_lcc":     len(dropped_seeds),
        "dropped_seed_list":        dropped_seeds,
        "dominant_ct_counts":       dominant_ct_counts,
    }
    write_diagnostics(out / "module_diagnostics.txt", info)


if __name__ == "__main__":
    main()