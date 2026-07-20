#!/usr/bin/env python3
"""
Step 14 (v3): Analyze Inter-Layer Edges — fast + layer-correct + DRG-hierarchy-aware
======================================================================================
Builds on v2 (correct layer=(source,cell_type) definition, single-pass edge scan).
Two new things in v3:

1) --exclude-sources
   --cell-types-csv only filters on the `cell_type` node attribute. If a
   cell-type label in your filter CSV happens to also appear (even loosely,
   or via case/whitespace) on Tabula Sapiens nodes, ts::<celltype> layers
   can survive the cell-type filter. --exclude-sources lets you hard-drop
   every node belonging to a given `source` value (e.g. "ts") BEFORE any
   other filtering happens, so no Tabula Sapiens node/edge can appear in
   the analysis, full stop.

   Usage: --exclude-sources ts
          --exclude-sources ts,some_other_source

2) DRG-hierarchy-aware plots
   The old stacked bar chart plotted every single (layer_1,layer_2) pair,
   which is unreadable once you have a few dozen fine-grained DRG neuron
   subtypes. v3 hard-codes your A-fiber / C-fiber / Non-classified
   hierarchy and produces:

     - interlayer_heatmap_celltype.png     fine cell-type x cell-type heatmap,
                                            ordered by fiber -> subcategory,
                                            tick labels colored by fiber,
                                            thin dashed lines = subcategory
                                            boundaries, thick solid lines =
                                            fiber-type boundaries.
     - interlayer_heatmap_subcategory.png  rolled up to subcategory level
                                            (e.g. "A-beta LTMR" vs "C-LTMR")
     - interlayer_heatmap_fibertype.png    rolled up to A-fiber/C-fiber/
                                            Non-classified level (3x3)
     - interlayer_edgetype_breakdown_subcategory.png  small stacked bar,
                                            edge-type composition per
                                            subcategory pair (few enough
                                            bars to actually read)
     - interlayer_edgetype_breakdown_fibertype.png     same, at fiber level
     - interlayer_edge_counts_by_subcategory.csv / _by_fibertype.csv

   Any cell type present in the data but NOT in DRG_HIERARCHY is bucketed
   into "Unclassified" (grey) at the subcategory/fiber level so nothing is
   silently dropped; a warning is printed listing which cell types those are.

Output Files
------------
  analysis/
    ├── interlayer_edge_counts.csv                    (per source::cell_type layer pair)
    ├── interlayer_edge_counts_by_celltype.csv         (rolled up across source)
    ├── interlayer_edge_counts_by_subcategory.csv      (NEW)
    ├── interlayer_edge_counts_by_fibertype.csv        (NEW)
    ├── interlayer_heatmap_celltype.png                (NEW, replaces old bar chart)
    ├── interlayer_heatmap_subcategory.png             (NEW)
    ├── interlayer_heatmap_fibertype.png                (NEW)
    ├── interlayer_edgetype_breakdown_subcategory.png  (NEW)
    ├── interlayer_edgetype_breakdown_fibertype.png    (NEW)
    ├── interlayer_annotations.csv
    ├── interlayer_density.csv
    ├── density_comparison.png
    ├── edge_annotation_heatmap.png
    ├── top_pathways_by_edges.csv
    └── interlayer_analysis_report.txt
"""

import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import networkx as nx
import numpy as np

# ============================================================================
# DRG NEURON HIERARCHY DEFINITION
# ============================================================================

DRG_HIERARCHY = {
    "A-fiber": {
        "color": "#1f77b4",  # blue
        "subcategories": {
            "Proprioceptors": {
                "marker": "o",
                "cell_types": [
                    "A-Propr.EPHA3",
                    "A-Propr.HAPLN4",
                    "A-Propr.PCDH8",
                ],
            },
            "A-beta LTMR": {
                "marker": "s",
                "cell_types": [
                    "Ab-LTMR.CCKAR",
                    "Ab-LTMR.ETV1",
                    "Ab-LTMR.LGI2",
                    "Ab-LTMR.NSG2",
                ],
            },
            "A-LTMR": {
                "marker": "^",
                "cell_types": [
                    "A-LTMR.TAC3",
                ],
            },
            "A-peptidergic": {
                "marker": "D",
                "cell_types": [
                    "A-PEP.CHRNA7_SLC18A3",
                    "A-PEP.KIT",
                    "A-PEP.NTRK3_S100A16",
                    "A-PEP.SCGN_ADRA2C",
                ],
            },
        },
    },
    "C-fiber": {
        "color": "#ff7f0e",  # orange
        "subcategories": {
            "C-LTMR": {
                "marker": "o",
                "cell_types": [
                    "C-LTMR.CDH9",
                ],
            },
            "C-peptidergic": {
                "marker": "D",
                "cell_types": [
                    "C-PEP.ADORA2B",
                    "C-PEP.TAC1_CACNG5",
                    "C-PEP.TAC1_CHRNA3",
                ],
            },
            "MRGPRX nociceptors": {
                "marker": "s",
                "cell_types": [
                    "C-NP.MRGPRX1_GFRA2",
                    "C-NP.MRGPRX1_MRGPRX4",
                ],
            },
            "C-thermosensitive": {
                "marker": "*",
                "cell_types": [
                    "C-Thermo.RXFP1",
                    "C-Thermo.TRPM8",
                ],
            },
            "C-SST nociceptor": {
                "marker": "v",
                "cell_types": [
                    "C-NP.SST",
                ],
            },
        },
    },
    "Non-classified": {
        "color": "#d62728",  # red
        "subcategories": {
            "ATF3 (injured)": {
                "marker": "x",
                "cell_types": [
                    "ATF3",
                ],
            },
        },
    },
}

UNCLASSIFIED_COLOR = "#7f7f7f"  # grey, for cell types not in DRG_HIERARCHY


def build_hierarchy_lookup(hierarchy: dict) -> dict:
    """Flatten DRG_HIERARCHY into lookups + orderings used by the plots."""
    ct_to_fiber, ct_to_subcat, subcat_to_fiber, marker_lookup = {}, {}, {}, {}
    fiber_colors, fiber_order, subcat_order, fine_order = {}, [], [], []

    for fiber, fdata in hierarchy.items():
        fiber_order.append(fiber)
        fiber_colors[fiber] = fdata["color"]
        for subcat, sdata in fdata["subcategories"].items():
            subcat_order.append(subcat)
            subcat_to_fiber[subcat] = fiber
            for ct in sdata["cell_types"]:
                fine_order.append(ct)
                ct_to_fiber[ct] = fiber
                ct_to_subcat[ct] = subcat
                marker_lookup[ct] = sdata["marker"]

    return {
        "ct_to_fiber": ct_to_fiber,
        "ct_to_subcat": ct_to_subcat,
        "subcat_to_fiber": subcat_to_fiber,
        "fiber_colors": fiber_colors,
        "fiber_order": fiber_order,
        "subcat_order": subcat_order,
        "fine_order": fine_order,
        "marker_lookup": marker_lookup,
    }


def compute_fine_boundaries(hierarchy: dict):
    """Cumulative cell-type counts -> gridline positions for the fine heatmap."""
    fiber_bounds, subcat_bounds = [], []
    idx = 0
    for fiber, fdata in hierarchy.items():
        for subcat, sdata in fdata["subcategories"].items():
            idx += len(sdata["cell_types"])
            subcat_bounds.append(idx)
        fiber_bounds.append(idx)
    return fiber_bounds[:-1], subcat_bounds[:-1]


def compute_subcat_boundaries(hierarchy: dict):
    """Cumulative subcategory counts -> gridline positions for the subcat heatmap."""
    bounds, idx = [], 0
    for fiber, fdata in hierarchy.items():
        idx += len(fdata["subcategories"])
        bounds.append(idx)
    return bounds[:-1]


# ============================================================================
# HELPERS
# ============================================================================

def load_multiplex(pkl_path: str) -> nx.Graph:
    print(f"Loading multiplex from {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")
    return G


def load_node_registry(csv_path: str) -> dict:
    print(f"Loading node registry from {csv_path} ...")
    df = pd.read_csv(csv_path, index_col="node_id")
    print(f"  {len(df):,} nodes registered\n")
    return df.to_dict(orient="index")


def layer_id(d: dict) -> str:
    """A true 'layer' is (source, cell_type), not cell_type alone."""
    return f"{d.get('source', 'unknown')}::{d.get('cell_type', 'unknown')}"


def load_allowed_cell_types(csv_path: str) -> set:
    """
    Read a cell-type filter CSV using the same schema as
    step5_consolidate_layers.py's layers_categories.csv / neurons_only.csv:
        Cell Type | Broad Category - ai | category
    Returns the set of 'Cell Type' values to keep (matches the cell_type
    node attribute in integrated_multiplex.pkl directly).
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0].strip() in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]
    if "Cell Type" not in df.columns:
        raise ValueError(
            f"{csv_path} is missing a 'Cell Type' column. Found: {list(df.columns)}"
        )
    cell_types = set(df["Cell Type"].astype(str).str.strip())
    print(f"Loaded {len(cell_types)} cell types to keep from {csv_path}")
    return cell_types


def filter_graph_by_cell_types(G: nx.Graph, allowed_cell_types: set) -> nx.Graph:
    """Restrict G to nodes whose cell_type is in allowed_cell_types."""
    all_cts_in_graph = {d.get("cell_type") for _, d in G.nodes(data=True)}
    matched = all_cts_in_graph & allowed_cell_types
    unmatched_csv = allowed_cell_types - all_cts_in_graph

    print(f"  Cell types in graph          : {len(all_cts_in_graph)}")
    print(f"  Cell types in filter CSV     : {len(allowed_cell_types)}")
    print(f"  Matched (kept)               : {len(matched)}")
    if unmatched_csv:
        print(f"  ⚠ In CSV but not found in graph ({len(unmatched_csv)}): "
              f"{sorted(unmatched_csv)[:10]}{'...' if len(unmatched_csv) > 10 else ''}")

    keep_nodes = [n for n, d in G.nodes(data=True) if d.get("cell_type") in allowed_cell_types]
    Gf = G.subgraph(keep_nodes).copy()
    print(f"  Filtered graph               : {Gf.number_of_nodes():,} nodes, "
          f"{Gf.number_of_edges():,} edges (from {G.number_of_nodes():,} / {G.number_of_edges():,})\n")
    return Gf


def filter_graph_exclude_sources(G: nx.Graph, excluded_sources: set) -> nx.Graph:
    """Hard-drop every node whose `source` attribute is in excluded_sources."""
    all_sources = {d.get("source") for _, d in G.nodes(data=True)}
    print(f"  Sources present in graph     : {sorted(s for s in all_sources if s is not None)}")
    print(f"  Excluding sources            : {sorted(excluded_sources)}")

    keep_nodes = [n for n, d in G.nodes(data=True) if d.get("source") not in excluded_sources]
    Gf = G.subgraph(keep_nodes).copy()
    print(f"  Filtered graph               : {Gf.number_of_nodes():,} nodes, "
          f"{Gf.number_of_edges():,} edges (from {G.number_of_nodes():,} / {G.number_of_edges():,})\n")
    return Gf


# ============================================================================
# SINGLE-PASS CORE: edge counts + annotations + density numerators
# ============================================================================

def analyze_edges(G: nx.Graph):
    """
    One pass over G.edges(data=True). Returns everything downstream
    sections need, instead of re-scanning the edge list per layer pair.
    """
    print("Scanning all edges (single pass) ...")
    t0 = time.time()

    edge_counts = defaultdict(lambda: defaultdict(int))
    celltype_rollup = defaultdict(lambda: defaultdict(int))

    annotations = []
    pathway_involvement = defaultdict(set)
    gene_involvement = defaultdict(set)

    n_total = 0
    n_inter = 0

    for u, v, d in G.edges(data=True):
        n_total += 1
        du = G.nodes[u]
        dv = G.nodes[v]

        lu, lv = layer_id(du), layer_id(dv)
        if lu == lv:
            continue  # true intra-layer (same source AND same cell_type)

        n_inter += 1
        pair = tuple(sorted([lu, lv]))
        edge_type = d.get("edge_type", "other")
        edge_counts[pair][edge_type] += 1

        ct_u, ct_v = du.get("cell_type"), dv.get("cell_type")
        if ct_u is not None and ct_v is not None:
            ct_pair = tuple(sorted([ct_u, ct_v]))
            celltype_rollup[ct_pair][edge_type] += 1

        gu, gv = du.get("gene"), dv.get("gene")

        pathways = d.get("pathways", set())
        pathway_str = "; ".join(f"{p[0]} ({p[1]})" for p in sorted(pathways)) if pathways else ""
        if edge_type == "pathway" and "pathway" in d:
            pathway_str = f"{d.get('pathway')} ({d.get('pathway_db', 'unknown')})"

        lr_gene = ""
        if edge_type == "ligand_receptor":
            lr_gene = f"{d.get('ligand_gene', '')} -> {d.get('receptor_gene', '')}"

        annotations.append({
            "layer_1": pair[0],
            "layer_2": pair[1],
            "gene_1": str(gu),
            "gene_2": str(gv),
            "edge_type": edge_type,
            "pathways": pathway_str,
            "ligand_receptor": lr_gene,
        })

        if pathway_str:
            for pway in pathway_str.split("; "):
                pathway_involvement[pway].add(pair)
        gene_involvement[str(gu)].add(pair)
        gene_involvement[str(gv)].add(pair)

    print(f"  {n_total:,} edges scanned, {n_inter:,} inter-layer ({100*n_inter/max(n_total,1):.1f}%)")
    print(f"  Done in {time.time()-t0:.1f}s\n")

    return edge_counts, celltype_rollup, annotations, pathway_involvement, gene_involvement


def layer_sizes(G: nx.Graph) -> dict:
    """gene set per (source,cell_type) layer, single pass over nodes."""
    sizes = defaultdict(set)
    for _, d in G.nodes(data=True):
        gene = d.get("gene")
        if gene:
            sizes[layer_id(d)].add(gene)
    return sizes


# ============================================================================
# SECTION 1: EDGE COUNT TABLES
# ============================================================================

EDGE_TYPE_COLS = ["ppi", "pathway", "ligand_receptor"]


def edge_counts_to_df(edge_counts: dict, col1: str, col2: str) -> pd.DataFrame:
    rows = []
    for (a, b), type_counts in sorted(edge_counts.items()):
        total = sum(type_counts.values())
        row = {col1: a, col2: b}
        for et in EDGE_TYPE_COLS:
            row[f"n_{et}"] = type_counts.get(et, 0)
        row["n_other"] = sum(v for k, v in type_counts.items() if k not in EDGE_TYPE_COLS)
        row["total"] = total
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("total", ascending=False).reset_index(drop=True)
    return df


def plot_interlayer_edge_counts(df: pd.DataFrame, col1: str, col2: str, output_path: Path,
                                 title: str = "Inter-Layer Edge Counts") -> None:
    """Stacked bar chart. Only use this for SHORT dfs (subcategory/fiber level) —
    at fine cell-type level this becomes unreadable, use the heatmap instead."""
    if len(df) == 0:
        print("  No inter-layer edges to plot.")
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(df) + 2), 7))
    x_labels = [f"{row[col1]}\n<->\n{row[col2]}" for _, row in df.iterrows()]
    x = np.arange(len(x_labels))
    width = 0.6
    bottom = np.zeros(len(df))
    colors = {"n_ppi": "#1f77b4", "n_pathway": "#ff7f0e",
              "n_ligand_receptor": "#2ca02c", "n_other": "#d62728"}

    for et in ["n_ppi", "n_pathway", "n_ligand_receptor", "n_other"]:
        values = df[et].values
        ax.bar(x, values, width, label=et.replace("n_", "").replace("_", " ").title(),
               bottom=bottom, color=colors.get(et, "#999"))
        bottom += values

    ax.set_xlabel("Group Pair", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Edges", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="right")
    ax.legend(title="Edge Type", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  Saved -> {output_path}")
    plt.close()


# ============================================================================
# SECTION 1b (NEW): HIERARCHY-AWARE HEATMAPS + ROLLUPS
# ============================================================================

def pivot_symmetric(df: pd.DataFrame, col1: str, col2: str, order: list,
                     value_col: str = "total") -> pd.DataFrame:
    mat = pd.DataFrame(0.0, index=order, columns=order)
    if len(df) == 0:
        return mat
    order_set = set(order)
    for _, row in df.iterrows():
        a, b = row[col1], row[col2]
        if a in order_set and b in order_set:
            mat.loc[a, b] += row[value_col]
            if a != b:
                mat.loc[b, a] += row[value_col]
    return mat


def build_group_rollup(annotations_df: pd.DataFrame, group_lookup: dict) -> pd.DataFrame:
    """
    Aggregate the fine-grained annotations table up to an arbitrary group
    level (subcategory or fiber-type), using group_lookup: cell_type -> group
    (any cell type not in group_lookup is bucketed as 'Unclassified').
    Returns a df shaped like edge_counts_to_df's output (group_1, group_2,
    n_ppi, n_pathway, n_ligand_receptor, n_other, total).
    """
    if len(annotations_df) == 0:
        return pd.DataFrame(columns=["group_1", "group_2"] + [f"n_{e}" for e in EDGE_TYPE_COLS]
                             + ["n_other", "total"])

    def parse_ct(layer_str):
        return layer_str.split("::", 1)[-1] if "::" in layer_str else layer_str

    df = annotations_df.copy()
    df["g1"] = df["layer_1"].apply(lambda x: group_lookup.get(parse_ct(x), "Unclassified"))
    df["g2"] = df["layer_2"].apply(lambda x: group_lookup.get(parse_ct(x), "Unclassified"))
    df["pair"] = list(zip(df[["g1", "g2"]].min(axis=1), df[["g1", "g2"]].max(axis=1)))

    counts = df.groupby(["pair", "edge_type"]).size().unstack(fill_value=0)
    rows = []
    for pair, row in counts.iterrows():
        g1, g2 = pair
        d = {"group_1": g1, "group_2": g2}
        for et in EDGE_TYPE_COLS:
            d[f"n_{et}"] = int(row.get(et, 0))
        d["n_other"] = int(sum(v for k, v in row.items() if k not in EDGE_TYPE_COLS))
        d["total"] = int(row.sum())
        rows.append(d)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("total", ascending=False).reset_index(drop=True)
    return out


def plot_hierarchy_heatmap(mat: pd.DataFrame, order: list, tick_colors: dict,
                            output_path: Path, title: str,
                            major_boundaries=None, minor_boundaries=None,
                            annot=None) -> None:
    if mat.values.sum() == 0:
        print(f"  Skipping {output_path.name}: no data.")
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(order)
    if annot is None:
        annot = n <= 18
    size = max(6, min(0.55 * n + 3, 22))

    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        mat, xticklabels=order, yticklabels=order, cmap="rocket_r",
        annot=annot, fmt=".0f", annot_kws={"size": 7},
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Inter-layer edges"}, square=True, ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    for label in ax.get_xticklabels():
        label.set_color(tick_colors.get(label.get_text(), "black"))
        label.set_rotation(90 if n > 6 else 30)
        label.set_ha("center" if n > 6 else "right")
        label.set_fontsize(8 if n < 25 else 6)
    for label in ax.get_yticklabels():
        label.set_color(tick_colors.get(label.get_text(), "black"))
        label.set_fontsize(8 if n < 25 else 6)

    for b in (minor_boundaries or []):
        ax.axhline(b, color="grey", linewidth=0.9, linestyle="--")
        ax.axvline(b, color="grey", linewidth=0.9, linestyle="--")
    for b in (major_boundaries or []):
        ax.axhline(b, color="black", linewidth=2.4)
        ax.axvline(b, color="black", linewidth=2.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  Saved -> {output_path}")
    plt.close()


def hierarchy_breakdown_section(rollup_df: pd.DataFrame, annotations_df: pd.DataFrame,
                                 out: Path, no_plots: bool) -> dict:
    """
    Produces the fine/subcategory/fibertype heatmaps + subcategory/fibertype
    bar charts + CSVs. Returns a dict of dataframes for use in the report.
    """
    hl = build_hierarchy_lookup(DRG_HIERARCHY)

    all_cts_seen = set()
    if len(rollup_df):
        all_cts_seen |= set(rollup_df["cell_type_1"]) | set(rollup_df["cell_type_2"])
    unmapped = sorted(all_cts_seen - set(hl["ct_to_fiber"].keys()))
    if unmapped:
        print(f"  ⚠ {len(unmapped)} cell type(s) present in data but not in DRG_HIERARCHY "
              f"(bucketed as 'Unclassified' at subcategory/fiber level, excluded from the "
              f"fine cell-type heatmap): {unmapped}")

    # ---- fine cell-type level heatmap (from the existing celltype rollup) ----
    fiber_bounds_fine, subcat_bounds_fine = compute_fine_boundaries(DRG_HIERARCHY)
    fine_order = hl["fine_order"]
    fine_tick_colors = {ct: hl["fiber_colors"][hl["ct_to_fiber"][ct]] for ct in fine_order}
    fine_mat = pivot_symmetric(rollup_df, "cell_type_1", "cell_type_2", fine_order, "total")

    if not no_plots:
        plot_hierarchy_heatmap(
            fine_mat, fine_order, fine_tick_colors,
            out / "interlayer_heatmap_celltype.png",
            "Inter-Layer Edges — Fine Cell Type\n(grouped by fiber type / subcategory; "
            "thick lines = fiber-type boundary, dashed = subcategory boundary)",
            major_boundaries=fiber_bounds_fine, minor_boundaries=subcat_bounds_fine,
        )

    # ---- subcategory-level rollup ----
    subcat_df = build_group_rollup(annotations_df, hl["ct_to_subcat"])
    subcat_df.to_csv(out / "interlayer_edge_counts_by_subcategory.csv", index=False)

    subcat_order = list(hl["subcat_order"])
    subcat_tick_colors = {s: hl["fiber_colors"][hl["subcat_to_fiber"][s]] for s in subcat_order}
    if "Unclassified" in set(subcat_df["group_1"]).union(subcat_df["group_2"]):
        subcat_order = subcat_order + ["Unclassified"]
        subcat_tick_colors["Unclassified"] = UNCLASSIFIED_COLOR
    subcat_bounds = compute_subcat_boundaries(DRG_HIERARCHY)
    subcat_mat = pivot_symmetric(subcat_df, "group_1", "group_2", subcat_order, "total")

    if not no_plots:
        plot_hierarchy_heatmap(
            subcat_mat, subcat_order, subcat_tick_colors,
            out / "interlayer_heatmap_subcategory.png",
            "Inter-Layer Edges — DRG Subcategory Level",
            major_boundaries=subcat_bounds,
        )
        plot_interlayer_edge_counts(
            subcat_df, "group_1", "group_2",
            out / "interlayer_edgetype_breakdown_subcategory.png",
            title="Inter-Layer Edge Type Breakdown — Subcategory Level",
        )

    # ---- fiber-type level rollup ----
    fiber_df = build_group_rollup(annotations_df, hl["ct_to_fiber"])
    fiber_df.to_csv(out / "interlayer_edge_counts_by_fibertype.csv", index=False)

    fiber_order = list(hl["fiber_order"])
    fiber_tick_colors = {f: hl["fiber_colors"][f] for f in fiber_order}
    if "Unclassified" in set(fiber_df["group_1"]).union(fiber_df["group_2"]):
        fiber_order = fiber_order + ["Unclassified"]
        fiber_tick_colors["Unclassified"] = UNCLASSIFIED_COLOR
    fiber_mat = pivot_symmetric(fiber_df, "group_1", "group_2", fiber_order, "total")

    if not no_plots:
        plot_hierarchy_heatmap(
            fiber_mat, fiber_order, fiber_tick_colors,
            out / "interlayer_heatmap_fibertype.png",
            "Inter-Layer Edges — Fiber Type Level",
        )
        plot_interlayer_edge_counts(
            fiber_df, "group_1", "group_2",
            out / "interlayer_edgetype_breakdown_fibertype.png",
            title="Inter-Layer Edge Type Breakdown — Fiber Type Level",
        )

    return {"subcat_df": subcat_df, "fiber_df": fiber_df, "unmapped": unmapped}


# ============================================================================
# SECTION 2: ANNOTATION SUMMARY
# ============================================================================

def create_annotation_summary(df: pd.DataFrame, pathway_involvement: dict, gene_involvement: dict) -> dict:
    top_pathways = sorted(((p, len(pairs)) for p, pairs in pathway_involvement.items()),
                           key=lambda x: -x[1])[:20]
    top_genes = sorted(((g, len(pairs)) for g, pairs in gene_involvement.items()),
                        key=lambda x: -x[1])[:20]
    edge_type_counts = df["edge_type"].value_counts().to_dict() if len(df) else {}
    return {
        "top_pathways": top_pathways,
        "top_genes": top_genes,
        "edge_type_counts": edge_type_counts,
        "total_annotations": len(df),
    }


# ============================================================================
# SECTION 3: DENSITY METRICS (uses edge_counts already computed — O(1) lookup)
# ============================================================================

def calculate_interlayer_density(edge_counts: dict, sizes: dict, gene_to_pathways: dict) -> pd.DataFrame:
    print("Calculating inter-layer density metrics ...")
    rows = []
    for (l1, l2), type_counts in sorted(edge_counts.items()):
        genes1 = sizes.get(l1, set())
        genes2 = sizes.get(l2, set())
        n1, n2 = len(genes1), len(genes2)
        n_possible = n1 * n2
        n_actual = sum(type_counts.values())
        density = n_actual / n_possible if n_possible > 0 else 0

        gp1 = sum(1 for g in genes1 if g in gene_to_pathways)
        gp2 = sum(1 for g in genes2 if g in gene_to_pathways)
        n_pathway_pairs = gp1 * gp2
        pathway_density = n_actual / n_pathway_pairs if n_pathway_pairs > 0 else 0

        rows.append({
            "layer_1": l1, "layer_2": l2,
            "n_genes_1": n1, "n_genes_2": n2,
            "n_possible_edges": n_possible,
            "n_actual_edges": n_actual,
            "density_percent": 100 * density,
            "n_pathway_pairs": n_pathway_pairs,
            "pathway_density_percent": 100 * pathway_density,
        })
    return pd.DataFrame(rows)


def plot_density_comparison(df: pd.DataFrame, output_path: Path) -> None:
    if len(df) == 0:
        print("  No density data to plot.")
        return
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x_labels = [f"{row['layer_1']}\n<->\n{row['layer_2']}" for _, row in df.iterrows()]
    x = np.arange(len(x_labels))
    width = 0.6

    ax1.bar(x, df["density_percent"], width, color="#1f77b4", alpha=0.8)
    ax1.set_title("Inter-Layer Density\n(Observed / All Possible Pairs)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, df["pathway_density_percent"], width, color="#ff7f0e", alpha=0.8)
    ax2.set_title("Pathway-Weighted Density", fontsize=12, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  Saved -> {output_path}")
    plt.close()


# ============================================================================
# SECTION 4: PATHWAY HEATMAP
# ============================================================================

def create_pathway_heatmap(annotations_df: pd.DataFrame, pathway_involvement: dict, output_path: Path) -> None:
    if not pathway_involvement or len(annotations_df) == 0:
        print("  No pathways to plot.")
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_pathways = sorted(((p, len(pairs)) for p, pairs in pathway_involvement.items()),
                           key=lambda x: -x[1])[:15]
    layer_pairs = sorted(annotations_df[["layer_1", "layer_2"]].drop_duplicates().values.tolist())

    matrix, pathway_labels = [], []
    for pathway, _ in top_pathways:
        pathway_labels.append(pathway[:50])
        row = []
        for l1, l2 in layer_pairs:
            pair = tuple(sorted([l1, l2]))
            row.append(1 if pair in pathway_involvement.get(pathway, set()) else 0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 8))
    pair_labels = [f"{l1}\n<->\n{l2}" for l1, l2 in layer_pairs]
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=pair_labels, yticklabels=pathway_labels,
                cbar_kws={"label": "Pathway Active"}, ax=ax)
    ax.set_title("Top 15 Pathways by Layer Pair Involvement", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  Saved -> {output_path}")
    plt.close()


# ============================================================================
# SECTION 5: REPORT
# ============================================================================

def write_report(edge_counts_df, density_df, annotation_summary, hierarchy_info, output_path: Path) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append("INTER-LAYER EDGE ANALYSIS REPORT (layer = source::cell_type)")
    lines.append("=" * 80)

    total = edge_counts_df["total"].sum() if len(edge_counts_df) else 0
    n_pairs = len(edge_counts_df)

    lines.append("\nOVERALL SUMMARY")
    lines.append("-" * 80)
    lines.append(f"  Total inter-layer layer-pairs    : {n_pairs}")
    lines.append(f"  Total inter-layer edges          : {total:,}")
    lines.append("  Edge types:")
    for etype, count in sorted(annotation_summary["edge_type_counts"].items(), key=lambda x: -x[1]):
        pct = 100 * count / annotation_summary["total_annotations"] if annotation_summary["total_annotations"] else 0
        lines.append(f"    {etype:<25} {count:>10,} ({pct:>6.1f}%)")

    lines.append("\nTOP 10 LAYER PAIRS BY VOLUME")
    lines.append("-" * 80)
    for _, row in edge_counts_df.head(10).iterrows():
        lines.append(f"  {row['layer_1']:<35} <-> {row['layer_2']:<35}")
        lines.append(f"    PPI: {row['n_ppi']:>6,}  |  Pathway: {row['n_pathway']:>6,}  |  "
                      f"L-R: {row['n_ligand_receptor']:>6,}  |  Total: {row['total']:>8,}")

    if hierarchy_info is not None:
        fiber_df = hierarchy_info["fiber_df"]
        subcat_df = hierarchy_info["subcat_df"]
        lines.append("\nFIBER-TYPE LEVEL BREAKDOWN (A-fiber / C-fiber / Non-classified)")
        lines.append("-" * 80)
        if len(fiber_df):
            for _, row in fiber_df.sort_values("total", ascending=False).iterrows():
                lines.append(f"  {row['group_1']:<20} <-> {row['group_2']:<20}  total: {row['total']:>8,}"
                              f"   (PPI {row['n_ppi']:,} | Pathway {row['n_pathway']:,} | "
                              f"L-R {row['n_ligand_receptor']:,})")
        lines.append("\nTOP 10 SUBCATEGORY PAIRS BY VOLUME")
        lines.append("-" * 80)
        if len(subcat_df):
            for _, row in subcat_df.sort_values("total", ascending=False).head(10).iterrows():
                lines.append(f"  {row['group_1']:<22} <-> {row['group_2']:<22}  total: {row['total']:>8,}")
        if hierarchy_info["unmapped"]:
            lines.append(f"\n  ⚠ Cell types present in data but not in DRG_HIERARCHY "
                          f"(bucketed as 'Unclassified'): {hierarchy_info['unmapped']}")

    lines.append("\nTOP 10 PATHWAYS BY INVOLVEMENT")
    lines.append("-" * 80)
    for i, (pathway, n) in enumerate(annotation_summary["top_pathways"][:10], 1):
        lines.append(f"  {i:2}. {pathway:<60} ({n} layer pairs)")

    lines.append("\nTOP 10 GENES BY INTER-LAYER INVOLVEMENT")
    lines.append("-" * 80)
    for i, (gene, n) in enumerate(annotation_summary["top_genes"][:10], 1):
        lines.append(f"  {i:2}. {gene:<60} ({n} layer pairs)")

    if len(density_df):
        lines.append("\nDENSITY METRICS (Top 5 by Edge Count)")
        lines.append("-" * 80)
        for _, row in density_df.sort_values("n_actual_edges", ascending=False).head(5).iterrows():
            lines.append(f"  {row['layer_1']:<35} <-> {row['layer_2']:<35}")
            lines.append(f"    Actual edges: {row['n_actual_edges']:>6,} / {row['n_possible_edges']:>10,} possible")
            lines.append(f"    Density: {row['density_percent']:>6.3f}%   "
                          f"Pathway density: {row['pathway_density_percent']:>6.3f}% "
                          f"(of {row['n_pathway_pairs']:,} pathway-annotated pairs)")

    lines.append("\n" + "=" * 80)
    lines.append("End of report")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    output_path.write_text(report_text)
    print(f"\nSaved report -> {output_path}")
    print(report_text)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze inter-layer edges (fast, layer-aware, DRG-hierarchy-aware)")
    parser.add_argument("--multiplex-pkl", required=True)
    parser.add_argument("--node-registry-csv", required=True)
    parser.add_argument("--pathway-csv", default=None)
    parser.add_argument("--cell-types-csv", default=None,
                         help="Optional CSV (same schema as step5's neurons_only.csv, "
                              "needs a 'Cell Type' column) to restrict the analysis to a "
                              "subset of cell types, e.g. neurons only.")
    parser.add_argument("--exclude-sources", default=None,
                         help="Comma-separated list of `source` values to hard-exclude "
                              "BEFORE any other filtering, e.g. 'ts' to remove all "
                              "Tabula Sapiens nodes/edges entirely.")
    parser.add_argument("--output-dir", default="analysis")
    parser.add_argument("--no-plots", action="store_true",
                         help="Skip matplotlib/seaborn plots for fastest possible run — just CSVs + report")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STEP 14 (v3): INTER-LAYER EDGE ANALYSIS")
    print("=" * 80 + "\n")

    t_start = time.time()

    G = load_multiplex(args.multiplex_pkl)
    _ = load_node_registry(args.node_registry_csv)

    if args.exclude_sources:
        excluded = {s.strip() for s in args.exclude_sources.split(",") if s.strip()}
        print(f"Excluding sources {sorted(excluded)} ...")
        G = filter_graph_exclude_sources(G, excluded)

    if args.cell_types_csv:
        print(f"Restricting analysis to cell types from {args.cell_types_csv} ...")
        allowed_cell_types = load_allowed_cell_types(args.cell_types_csv)
        G = filter_graph_by_cell_types(G, allowed_cell_types)

    gene_to_pathways = {}
    if args.pathway_csv and Path(args.pathway_csv).exists():
        print(f"Loading pathway data from {args.pathway_csv} ...")
        df_path = pd.read_csv(args.pathway_csv)
        for _, row in df_path.iterrows():
            gene = str(row.get("gene", "")).strip().upper()
            pathway = str(row.get("pathway", "")).strip()
            if gene and pathway:
                gene_to_pathways.setdefault(gene, set()).add(pathway)
        print(f"  {len(gene_to_pathways):,} genes with pathway annotations\n")

    # ---- single pass over all edges ----
    edge_counts, celltype_rollup, annotations, pathway_involvement, gene_involvement = analyze_edges(G)
    sizes = layer_sizes(G)

    # ---- Section 1: edge counts ----
    print("=" * 80); print("SECTION 1: INTER-LAYER EDGE COUNTS"); print("=" * 80 + "\n")
    edge_counts_df = edge_counts_to_df(edge_counts, "layer_1", "layer_2")
    edge_counts_df.to_csv(out / "interlayer_edge_counts.csv", index=False)
    print(edge_counts_df.to_string(index=False))

    rollup_df = edge_counts_to_df(celltype_rollup, "cell_type_1", "cell_type_2")
    rollup_df.to_csv(out / "interlayer_edge_counts_by_celltype.csv", index=False)
    print(f"\n(cell-type rollup, pooled across source, saved separately)")

    # ---- Section 2: annotations ----
    print("\n" + "=" * 80); print("SECTION 2: EDGE ANNOTATIONS"); print("=" * 80 + "\n")
    annotations_df = pd.DataFrame(annotations)
    annotations_df.to_csv(out / "interlayer_annotations.csv", index=False)
    annotation_summary = create_annotation_summary(annotations_df, pathway_involvement, gene_involvement)

    print("Top 10 pathways by layer-pair involvement:")
    for i, (pathway, n) in enumerate(annotation_summary["top_pathways"][:10], 1):
        print(f"  {i:2}. {pathway:<60} ({n} layer pairs)")

    pd.DataFrame(annotation_summary["top_pathways"], columns=["pathway", "n_layer_pairs"]).to_csv(
        out / "top_pathways_by_edges.csv", index=False)

    if not args.no_plots:
        create_pathway_heatmap(annotations_df, pathway_involvement, out / "edge_annotation_heatmap.png")

    # ---- Section 1b (NEW): DRG hierarchy breakdown ----
    print("\n" + "=" * 80); print("SECTION 1b: DRG HIERARCHY BREAKDOWN"); print("=" * 80 + "\n")
    hierarchy_info = hierarchy_breakdown_section(rollup_df, annotations_df, out, args.no_plots)

    # ---- Section 3: density (O(1) lookups from edge_counts, no re-scan) ----
    print("\n" + "=" * 80); print("SECTION 3: DENSITY METRICS"); print("=" * 80 + "\n")
    density_df = calculate_interlayer_density(edge_counts, sizes, gene_to_pathways)
    density_df.to_csv(out / "interlayer_density.csv", index=False)
    print(density_df.to_string(index=False))

    if not args.no_plots:
        plot_density_comparison(density_df, out / "density_comparison.png")

    # ---- Section 4: report ----
    print("\n" + "=" * 80); print("SECTION 4: SUMMARY REPORT"); print("=" * 80 + "\n")
    write_report(edge_counts_df, density_df, annotation_summary, hierarchy_info,
                 out / "interlayer_analysis_report.txt")

    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE in {time.time()-t_start:.1f}s")
    print("=" * 80)
    print(f"All outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()