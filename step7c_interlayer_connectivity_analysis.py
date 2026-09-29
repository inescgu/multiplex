#!/usr/bin/env python3
"""
Step 7c (replaces step10_pathway_connection_analysis.py AND
step14_analyze_interlayer_edges.py — merged into one script): inter-cell-type
(bipartite) connectivity analysis.

Why these two got merged
---------------------------
Both old scripts existed to answer the same underlying question — "which
cell types are connected to which, via what, and how much?" — but got
there two different ways because of what the old homogeneous structure
made easy vs. hard:

  - step14 scanned integrated_multiplex.pkl's edges directly (fast, one
    pass) but its DRG-hierarchy-aware heatmaps were the more polished half.
  - step10's pathway-connectivity script instead RE-DERIVED cell-type
    connectivity from scratch — cross-referencing separate LR/pathway edge
    lists against per-cell-type gene TSVs — because the old special
    ppi_layer/pathway_layer/ligand_receptor_layer files didn't carry that
    mapping directly. That reconstruction is where its extra value lived:
    gene-overlap-across-cell-types analysis and top-pathway/gene-label
    tables.

In the new heterogeneous structure, "which cell types are connected, via
what" is no longer something to reconstruct at all — it's exactly what
bipartite/<ct_a>__<ct_b>.tsv + bipartite_manifest.json already ARE, written
directly by step 2. So this script reads those files as the source of
truth for connectivity and counts, borrows step14's DRG-hierarchy heatmap
code (more general than step10's hardcoded NEURON_GROUPS) for the visuals,
and borrows step10's gene-overlap analysis (which is genuinely
cell-type-set-driven, not graph-edge-driven, so it transfers over cleanly
using celltype_gene_sets.json instead of re-reading TSVs).

Dropped from the old scripts (say the word if you want these back):
  - step10's hardcoded NEURON_GROUPS grouping and its per-group heatmap
    grid (the DRG_HIERARCHY subcategory level below is the same idea,
    more general, and already covered by this script).
  - step10's networkx spring-layout network diagrams (nice to look at,
    lowest analytical value per line of code, easy to bolt back on).

What edge_type labeling means here
-------------------------------------
Because step 2 merges LR and Reactome edges into ONE bipartite file per
cell-type pair (MultiXrank only allows one bipartite file per pair — see
the pipeline README), a raw bipartite edge doesn't carry its own source
label on disk. This script re-derives it by re-checking each edge against
--lr-csv / --reactome-csv (same idea step10 already used, just pointed at
the new files) — pass both to get a full ligand_receptor/reactome/other
breakdown; skip either flag and those edges are labeled 'other'.

NEW: LR interaction-type breakdown (method/source + NeuronChat ligand_type)
------------------------------------------------------------------------------
Two things changed with this update:

1. Edge labeling now also checks each raw bipartite edge against
   scored_lr_edge_provenance.csv (written by step2_build_bipartite_edges.py
   whenever it was run with --scored-lr-csv), in addition to --lr-csv.
   Previously, an edge that came ONLY from LIANA/NeuronChat's scored-LR
   output (i.e. not also present in the flat --lr-csv resource) had no way
   to be recognized as 'ligand_receptor' at all here -- it silently fell
   into the 'other' bucket. Matching is done on the unordered
   (cell-type-pair, gene-pair) combination, since a raw bipartite edge's
   g1/g2 column order doesn't reliably reflect ligand-vs-receptor
   direction once it's mixed with LR/Reactome/identity edges in the same
   file (see FIX note in step2's docstring).

2. Every bipartite edge matched against the provenance file is now
   attributed to whichever method(s) (cellphonedb, natmi, connectome, sca,
   logfc, liana_consensus, neuronchat) and, for NeuronChat rows, ligand_type
   categor(y/ies) (Neurotransmitter, Neuropeptide, Gas, Gap junction
   protein) proposed it. Two new histograms are produced network-wide (not
   restricted to any particular disease module -- see
   step6b_module_diagnostics.py for the module-restricted version of the
   same breakdown):
       lr_edges_by_source.csv / .png
       lr_edges_by_ligand_type.csv / .png
   Edges labeled 'ligand_receptor' purely via --lr-csv (i.e. with no
   matching provenance row -- the flat resource carries no per-method
   information) simply don't appear in these two breakdowns; they're still
   counted normally everywhere else (edge_counts_by_celltype_pair.csv,
   the heatmaps, etc).

Only generated if scored_lr_edge_provenance.csv exists under --wdir (i.e.
step2 was run with --scored-lr-csv). Auto-discovered by default; override
with --scored-lr-provenance-csv if it lives somewhere else.

NEW: density-normalized heatmaps (this update)
------------------------------------------------
Raw edge-count heatmaps (heatmap_celltype.png / heatmap_subcategory.png /
heatmap_fibertype.png, kept exactly as before) naturally favor whichever
cell type or group has the most genes to draw edges from -- most visibly
'Unclassified', which is just the leftover bucket for every cell type not
in DRG_HIERARCHY (in practice, PPI-linked types with far larger gene sets
than any DRG neuron subtype). Raw counts alone therefore make the biggest
layer look like the "best connected" one almost by construction, not
because it's actually more densely wired.

To make cross-group comparisons fair, every hierarchy heatmap now also
gets a density-normalized twin (heatmap_*_density.png, plus matching
edge_density_by_*.csv): edges / (n_genes_group_a * n_genes_group_b), i.e.
what fraction of all possible gene-gene pairs between the two groups are
actually realized as bipartite edges. A group's gene pool is the UNION of
its member cell types' gene sets (celltype_gene_sets.json), not a sum, so
a gene shared across several cell types in the same group isn't
double-counted. This is an approximation at the group level (it doesn't
try to subtract out gene pairs that could never appear as a
between-cell-type bipartite edge in the first place), but it's enough to
stop the biggest layer from automatically looking like the best-connected
one. The absolute-count plots are unchanged and still produced -- density
is an additional lens, not a replacement.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# DRG NEURON HIERARCHY (same constant as step 7 — keep in sync if edited)
# ============================================================================
DRG_HIERARCHY = {
    "A-fiber": {"color": "#1f77b4", "subcategories": {
        "Proprioceptors": {"cell_types": ["A-Propr.EPHA3", "A-Propr.HAPLN4", "A-Propr.PCDH8"]},
        "A-beta LTMR": {"cell_types": ["Ab-LTMR.CCKAR", "Ab-LTMR.ETV1", "Ab-LTMR.LGI2", "Ab-LTMR.NSG2"]},
        "A-LTMR": {"cell_types": ["A-LTMR.TAC3"]},
        "A-peptidergic": {"cell_types": ["A-PEP.CHRNA7_SLC18A3", "A-PEP.KIT", "A-PEP.NTRK3_S100A16", "A-PEP.SCGN_ADRA2C"]},
    }},
    "C-fiber": {"color": "#ff7f0e", "subcategories": {
        "C-LTMR": {"cell_types": ["C-LTMR.CDH9"]},
        "C-peptidergic": {"cell_types": ["C-PEP.ADORA2B", "C-PEP.TAC1_CACNG5", "C-PEP.TAC1_CHRNA3"]},
        "MRGPRX nociceptors": {"cell_types": ["C-NP.MRGPRX1_GFRA2", "C-NP.MRGPRX1_MRGPRX4"]},
        "C-thermosensitive": {"cell_types": ["C-Thermo.RXFP1", "C-Thermo.TRPM8"]},
        "C-SST nociceptor": {"cell_types": ["C-NP.SST"]},
    }},
    "ATF3": {"color": "#d62728", "subcategories": {
        "ATF3 (injured)": {"cell_types": ["ATF3"]},
    }},
}
UNCLASSIFIED_COLOR = "#7f7f7f"
EDGE_TYPE_COLS = ["ligand_receptor", "reactome"]


def build_hierarchy_lookup(hierarchy: dict) -> dict:
    ct_to_fiber, ct_to_subcat, subcat_to_fiber = {}, {}, {}
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
    return {"ct_to_fiber": ct_to_fiber, "ct_to_subcat": ct_to_subcat, "subcat_to_fiber": subcat_to_fiber,
            "fiber_colors": fiber_colors, "fiber_order": fiber_order, "subcat_order": subcat_order, "fine_order": fine_order}


def compute_fine_boundaries(hierarchy: dict):
    fiber_bounds, subcat_bounds, idx = [], [], 0
    for fiber, fdata in hierarchy.items():
        for subcat, sdata in fdata["subcategories"].items():
            idx += len(sdata["cell_types"])
            subcat_bounds.append(idx)
        fiber_bounds.append(idx)
    return fiber_bounds[:-1], subcat_bounds[:-1]


def compute_subcat_boundaries(hierarchy: dict):
    bounds, idx = [], 0
    for fiber, fdata in hierarchy.items():
        idx += len(fdata["subcategories"])
        bounds.append(idx)
    return bounds[:-1]


# ============================================================================
# Cell-type name sanitizing (kept in sync with step1/step2's sanitize_name())
# ============================================================================
def sanitize_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").replace(",", "")


# ============================================================================
# LOADING: bipartite files are the source of truth for connectivity
# ============================================================================
def load_celltype_gene_sets(wdir: Path) -> dict[str, set]:
    with open(wdir / "celltype_gene_sets.json") as f:
        raw = json.load(f)
    return {ct: set(genes) for ct, genes in raw.items()}


def load_bipartite_edges(wdir: Path) -> pd.DataFrame:
    """One row per raw bipartite edge, across every cell-type pair."""
    manifest_path = wdir / "bipartite_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"ERROR: {manifest_path} not found — run step 2 first.")
    with open(manifest_path) as f:
        manifest = json.load(f)

    rows = []
    for m in manifest:
        df = pd.read_csv(wdir / m["file"], sep="\t", header=None, names=["gene_a", "gene_b"], dtype=str)
        for ga, gb in zip(df["gene_a"], df["gene_b"]):
            ga, gb = str(ga).strip().upper(), str(gb).strip().upper()
            if not ga or not gb or ga == gb:
                continue
            ct1, ct2 = sorted([m["source"], m["target"]])
            rows.append({"cell_type_1": ct1, "cell_type_2": ct2, "gene_a": ga, "gene_b": gb})
    print(f"Loaded {len(rows):,} raw bipartite edges across {len(manifest)} cell-type pair(s)\n")
    return pd.DataFrame(rows)


def load_lr_set(lr_csv: str) -> set:
    df = pd.read_csv(lr_csv)
    if "ligand_gname" not in df.columns:
        df = pd.read_csv(lr_csv, sep="\t")
    df["ligand_gname"] = df["ligand_gname"].str.strip().str.upper()
    df["receptor_gname"] = df["receptor_gname"].str.strip().str.upper()
    return set(zip(df["ligand_gname"], df["receptor_gname"]))


def load_pathway_map(reactome_csv: str, uniprot_map_csv: str | None, restrict_to: set) -> dict[tuple, list]:
    """gene-pair -> list of pathway names, for top-pathway reporting."""
    u2s = {}
    if uniprot_map_csv:
        u2s_df = pd.read_csv(uniprot_map_csv)
        u2s = dict(zip(u2s_df["uniprotid"], u2s_df["symbol"]))

    df_temp = pd.read_csv(reactome_csv, nrows=5)
    use_symbol_cols = {"symbol1", "symbol2"}.issubset(df_temp.columns)
    id_cols = ["symbol1", "symbol2"] if use_symbol_cols else ["uniprot1", "uniprot2"]
    usecols = id_cols + ["common_pathway"]
    df = pd.read_csv(reactome_csv, low_memory=False, usecols=usecols).dropna(subset=["common_pathway"])

    c1 = df[id_cols[0]].astype(str).str.strip().str.upper()
    c2 = df[id_cols[1]].astype(str).str.strip().str.upper()
    if not use_symbol_cols and u2s:
        lookup = {str(k).upper(): v for k, v in u2s.items()}
        c1 = c1.map(lookup).fillna(c1).astype(str).str.upper()
        c2 = c2.map(lookup).fillna(c2).astype(str).str.upper()

    pathway = df["common_pathway"].astype(str).str.strip()
    valid = (c1 != c2) & (c1.isin(restrict_to)) & (c2.isin(restrict_to))
    keys = list(zip(c1[valid], c2[valid]))

    pw_map = defaultdict(set)
    for (a, b), p in zip(keys, pathway[valid]):
        key = (min(a, b), max(a, b))
        pw_map[key].add(p)
    return {k: sorted(v) for k, v in pw_map.items()}


# ============================================================================
# NEW: scored-LR provenance loading + lookup (method/source + ligand_type)
# ============================================================================
def load_lr_provenance(wdir: Path, override_path: str | None) -> pd.DataFrame | None:
    """Loads scored_lr_edge_provenance.csv written by
    step2_build_bipartite_edges.py (only present if step2 was run with
    --scored-lr-csv). Returns None if not found, so callers can skip the
    interaction-type breakdown gracefully."""
    path = Path(override_path) if override_path else (wdir / "scored_lr_edge_provenance.csv")
    if not path.exists():
        print(f"[info] {path} not found -- skipping LR interaction-type breakdown. "
              f"(Re-run step2_build_bipartite_edges.py with --scored-lr-csv to generate it.)\n")
        return None

    df = pd.read_csv(path)
    required = {"cell_type_source", "cell_type_target", "ligand", "receptor", "source"}
    missing = required - set(df.columns)
    if missing:
        print(f"[warn] {path} is missing columns {missing} -- skipping LR interaction-type breakdown.\n")
        return None

    df["cell_type_source"] = df["cell_type_source"].astype(str).str.strip()
    df["cell_type_target"] = df["cell_type_target"].astype(str).str.strip()
    df["ligand"] = df["ligand"].astype(str).str.strip().str.upper()
    df["receptor"] = df["receptor"].astype(str).str.strip().str.upper()
    df["source"] = df["source"].astype(str).str.strip()
    if "ligand_type" not in df.columns:
        df["ligand_type"] = pd.NA
    else:
        df["ligand_type"] = df["ligand_type"].astype(str).str.strip()
        df.loc[df["ligand_type"].isin(["", "nan", "None", "<NA>"]), "ligand_type"] = pd.NA

    print(f"Loaded scored-LR edge provenance: {path} ({len(df):,} rows)")
    print(f"  Sources present     : {sorted(df['source'].dropna().unique())}")
    lt_present = sorted(df['ligand_type'].dropna().unique())
    print(f"  Ligand types present: {lt_present if lt_present else '(none)'}\n")
    return df


def build_provenance_lookup(prov_df: pd.DataFrame | None) -> dict:
    """Maps (frozenset({ct1, ct2}), frozenset({gene1, gene2})) -> list of
    {'source': ..., 'ligand_type': ...} dicts. Deliberately unordered/
    direction-agnostic on both cell types and genes: a raw bipartite edge's
    g1/g2 column order doesn't reliably preserve ligand-vs-receptor
    direction once LR/Reactome/scored-LR/identity edges are merged into one
    file per cell-type pair (see step2's FIX note), so matching a bipartite
    edge against this lookup has to be direction-agnostic too."""
    lookup = defaultdict(list)
    if prov_df is None or prov_df.empty:
        return lookup
    for row in prov_df.itertuples(index=False):
        ct_key = frozenset((row.cell_type_source, row.cell_type_target))
        gene_key = frozenset((row.ligand, row.receptor))
        lookup[(ct_key, gene_key)].append({
            "source": row.source,
            "ligand_type": row.ligand_type if pd.notna(row.ligand_type) else None,
        })
    return lookup


# ============================================================================
# Edge labeling + counts
# ============================================================================
def label_and_count_edges(bip_df: pd.DataFrame, lr_set: set | None, pw_map: dict | None,
                           prov_lookup: dict | None) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Returns (annotated per-edge df, edge_counts[pair][type]=count,
    pathway_involvement[pathway]=set(pairs), gene_involvement[gene]=set(pairs)).

    The annotated df also carries an 'lr_sources' / 'lr_ligand_types' column
    (semicolon-joined) for every edge that matched the scored-LR provenance
    lookup, whether or not it also matched --lr-csv -- this is what feeds
    the network-wide interaction-type histograms in main().
    """
    edge_counts = defaultdict(lambda: defaultdict(int))
    pathway_involvement = defaultdict(set)
    gene_involvement = defaultdict(set)
    rows = []
    prov_lookup = prov_lookup or {}

    for _, row in bip_df.iterrows():
        ga, gb, ct1, ct2 = row["gene_a"], row["gene_b"], row["cell_type_1"], row["cell_type_2"]
        pair = (ct1, ct2)
        key = (min(ga, gb), max(ga, gb))

        prov_matches = prov_lookup.get((frozenset((ct1, ct2)), frozenset((ga, gb))), [])
        matched_lr_csv = lr_set is not None and ((ga, gb) in lr_set or (gb, ga) in lr_set)

        etype = "other"
        pw_names = []
        if matched_lr_csv or prov_matches:
            etype = "ligand_receptor"
        if pw_map is not None and key in pw_map:
            pw_names = pw_map[key]
            etype = "reactome" if etype == "other" else "ligand_receptor+reactome"

        edge_counts[pair][etype if "+" not in etype else "ligand_receptor"] += 1
        if "reactome" in etype:
            edge_counts[pair]["reactome"] += 1

        for p in pw_names:
            pathway_involvement[p].add(pair)
        gene_involvement[ga].add(pair)
        gene_involvement[gb].add(pair)

        lr_sources = sorted({m["source"] for m in prov_matches})
        lr_ligand_types = sorted({m["ligand_type"] for m in prov_matches if m["ligand_type"]})

        rows.append({"cell_type_1": ct1, "cell_type_2": ct2, "gene_a": ga, "gene_b": gb,
                      "edge_type": etype, "pathways": "; ".join(pw_names),
                      "lr_sources": "; ".join(lr_sources), "lr_ligand_types": "; ".join(lr_ligand_types)})

    return pd.DataFrame(rows), edge_counts, pathway_involvement, gene_involvement


def edge_counts_to_df(edge_counts: dict) -> pd.DataFrame:
    rows = []
    for (a, b), type_counts in sorted(edge_counts.items()):
        row = {"cell_type_1": a, "cell_type_2": b}
        for et in EDGE_TYPE_COLS:
            row[f"n_{et}"] = type_counts.get(et, 0)
        row["n_other"] = type_counts.get("other", 0)
        row["total"] = sum(v for k, v in type_counts.items() if k != "ligand_receptor+reactome")
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("total", ascending=False).reset_index(drop=True)
    return df


# ============================================================================
# NEW: network-wide LR interaction-type histograms (method/source + ligand_type)
# ============================================================================
def lr_interaction_type_counts(annotations_df: pd.DataFrame, column: str) -> pd.Series:
    """Unique-edge counts grouped by `column` ('lr_sources' or
    'lr_ligand_types', which are semicolon-joined multi-value strings --
    exploded to one row per value first). Rows with an empty value are
    dropped (e.g. edges labeled ligand_receptor purely via --lr-csv, with
    no matching provenance row, have nothing here)."""
    sub = annotations_df[annotations_df["edge_type"].isin(["ligand_receptor", "ligand_receptor+reactome"])]
    sub = sub[sub[column] != ""]
    if sub.empty:
        return pd.Series(dtype=int)
    exploded = sub.assign(**{column: sub[column].str.split("; ")}).explode(column)
    # Dedup on a gene-order-INDEPENDENT key: the same conceptual edge can
    # appear in the annotated table with (gene_a, gene_b) in either order
    # (e.g. once via --lr-csv's undirected pair, once via a scored-LR row
    # whose ligand/receptor happened to land in the opposite gene_a/gene_b
    # slot) -- without sorting here, that single edge would be counted
    # twice for the same source/ligand_type.
    gene_lo = exploded[["gene_a", "gene_b"]].min(axis=1)
    gene_hi = exploded[["gene_a", "gene_b"]].max(axis=1)
    exploded = exploded.assign(_gene_lo=gene_lo, _gene_hi=gene_hi)
    key_cols = ["cell_type_1", "cell_type_2", "_gene_lo", "_gene_hi", column]
    exploded = exploded.drop_duplicates(subset=key_cols)
    return exploded.groupby(column).size().sort_values(ascending=False)


def plot_lr_interaction_bar(counts: pd.Series, title: str, out_path: Path, color: str) -> bool:
    if counts.empty:
        print(f"  [plot] No data for '{title}' -- skipping.")
        return False
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(counts) + 2), 5.5))
    ax.bar(range(len(counts)), counts.values, color=color, edgecolor="white")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index.tolist(), rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of unique LR edges")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved -> {out_path}")
    plt.close(fig)
    return True


def lr_interaction_type_section(annotations_df: pd.DataFrame, out: Path, no_plots: bool) -> dict:
    print("\nSECTION: LR INTERACTION-TYPE BREAKDOWN (method/source + NeuronChat ligand_type)")
    by_source = lr_interaction_type_counts(annotations_df, "lr_sources")
    by_ligand_type = lr_interaction_type_counts(annotations_df, "lr_ligand_types")

    by_source.rename("n_edges").rename_axis("source").reset_index().to_csv(out / "lr_edges_by_source.csv", index=False)
    by_ligand_type.rename("n_edges").rename_axis("ligand_type").reset_index().to_csv(
        out / "lr_edges_by_ligand_type.csv", index=False)

    print("\nLR edges by method/source (network-wide):")
    print(by_source.to_string() if not by_source.empty else "  (no provenance-matched edges)")
    print("\nLR edges by NeuronChat ligand_type (network-wide):")
    print(by_ligand_type.to_string() if not by_ligand_type.empty else "  (no provenance-matched edges)")

    source_plot = ligand_type_plot = None
    if not no_plots:
        if plot_lr_interaction_bar(by_source, "Network-wide LR edges by method/source",
                                    out / "lr_edges_by_source.png", "#8172B2"):
            source_plot = out / "lr_edges_by_source.png"
        if plot_lr_interaction_bar(by_ligand_type, "Network-wide LR edges by NeuronChat ligand_type",
                                    out / "lr_edges_by_ligand_type.png", "#55A868"):
            ligand_type_plot = out / "lr_edges_by_ligand_type.png"

    return {"by_source": by_source, "by_ligand_type": by_ligand_type,
            "source_plot": source_plot, "ligand_type_plot": ligand_type_plot}


# ============================================================================
# Hierarchy-aware heatmaps (from old step14, generalized: no more
# layer_id()/"source::celltype" parsing needed — cell-type names are direct)
# ============================================================================
def pivot_symmetric(df: pd.DataFrame, col1: str, col2: str, order: list, value_col: str = "total") -> pd.DataFrame:
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
    if len(annotations_df) == 0:
        return pd.DataFrame(columns=["group_1", "group_2"] + [f"n_{e}" for e in EDGE_TYPE_COLS] + ["n_other", "total"])
    df = annotations_df.copy()
    df["g1"] = df["cell_type_1"].map(lambda x: group_lookup.get(x, "Unclassified"))
    df["g2"] = df["cell_type_2"].map(lambda x: group_lookup.get(x, "Unclassified"))
    df["pair"] = list(zip(df[["g1", "g2"]].min(axis=1), df[["g1", "g2"]].max(axis=1)))
    counts = df.groupby(["pair", "edge_type"]).size().unstack(fill_value=0)
    rows = []
    for pair, row in counts.iterrows():
        g1, g2 = pair
        d = {"group_1": g1, "group_2": g2}
        for et in EDGE_TYPE_COLS:
            d[f"n_{et}"] = int(row.get(et, 0)) + int(row.get("ligand_receptor+reactome", 0) if et == "ligand_receptor" else 0)
        d["n_other"] = int(row.get("other", 0))
        d["total"] = int(row.sum())
        rows.append(d)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("total", ascending=False).reset_index(drop=True)
    return out


# ============================================================================
# NEW: group-level gene pools + density, for normalizing group heatmaps
# ============================================================================
def build_group_gene_universe(gene_sets: dict, group_lookup: dict) -> dict:
    """Union (NOT sum) of every group's member cell types' gene sets --
    the relevant 'pool size' for that group when normalizing edge counts
    into a density. Using union rather than sum means a gene shared by
    several cell types in the same group is only counted once, so the
    density isn't artificially deflated for groups with lots of gene
    overlap between their own member cell types.

    Cell types absent from group_lookup fall into 'Unclassified', the
    same catch-all bucket build_group_rollup() already uses -- in
    practice this is usually a much larger gene pool than any single DRG
    fiber/subcategory, which is exactly why its raw edge counts dwarf
    everything else and why this normalization matters."""
    group_genes = defaultdict(set)
    for ct, genes in gene_sets.items():
        group_genes[group_lookup.get(ct, "Unclassified")] |= genes
    return {g: len(s) for g, s in group_genes.items()}


def calculate_group_density(group_df: pd.DataFrame, group_gene_sizes: dict) -> pd.DataFrame:
    """Same idea as calculate_density() below, but at the subcategory/
    fiber-type group level: density_percent = 100 * actual edges /
    (n_genes_group_1 * n_genes_group_2). This is an approximation for
    within-group (diagonal) entries -- it doesn't subtract out gene pairs
    that could never appear as a between-cell-type bipartite edge (e.g.
    a gene paired with itself, or two genes belonging to the same single
    cell type) -- but it's a consistent yardstick across every group pair,
    which raw counts alone are not."""
    rows = []
    for _, row in group_df.iterrows():
        g1, g2 = row["group_1"], row["group_2"]
        n1, n2 = group_gene_sizes.get(g1, 0), group_gene_sizes.get(g2, 0)
        n_possible = n1 * n2
        n_actual = row["total"]
        density = 100 * n_actual / n_possible if n_possible else 0
        rows.append({"group_1": g1, "group_2": g2, "n_genes_1": n1, "n_genes_2": n2,
                      "n_possible_edges": n_possible, "n_actual_edges": n_actual, "density_percent": density})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("n_actual_edges", ascending=False).reset_index(drop=True)
    return df


def plot_hierarchy_heatmap(mat, order, tick_colors, output_path, title, major_boundaries=None, minor_boundaries=None,
                            annot=None, cbar_label="Inter-cell-type edges", fmt=".0f"):
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
    sns.heatmap(mat, xticklabels=order, yticklabels=order, cmap="rocket_r", annot=annot, fmt=fmt,
                annot_kws={"size": 7}, linewidths=0.4, linecolor="white",
                cbar_kws={"label": cbar_label}, square=True, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    for label in ax.get_xticklabels():
        label.set_color(tick_colors.get(label.get_text(), "black"))
        label.set_rotation(90 if n > 6 else 30)
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


def plot_edgetype_breakdown(df, col1, col2, output_path, title):
    if len(df) == 0:
        print("  No data to plot."); return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(df) + 2), 7))
    x_labels = [f"{row[col1]}\n<->\n{row[col2]}" for _, row in df.iterrows()]
    x = np.arange(len(x_labels))
    bottom = np.zeros(len(df))
    colors = {"n_ligand_receptor": "#2ca02c", "n_reactome": "#ff7f0e", "n_other": "#d62728"}
    for et in ["n_ligand_receptor", "n_reactome", "n_other"]:
        values = df[et].values
        ax.bar(x, values, 0.6, label=et.replace("n_", "").replace("_", " ").title(), bottom=bottom, color=colors.get(et, "#999"))
        bottom += values
    ax.set_ylabel("Number of edges", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="right")
    ax.legend(title="Edge Type"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  Saved -> {output_path}")
    plt.close()


def hierarchy_breakdown_section(edge_counts_df, annotations_df, gene_sets, density_df, out, no_plots) -> dict:
    hl = build_hierarchy_lookup(DRG_HIERARCHY)
    all_cts_seen = set(edge_counts_df["cell_type_1"]) | set(edge_counts_df["cell_type_2"]) if len(edge_counts_df) else set()
    unmapped = sorted(all_cts_seen - set(hl["ct_to_fiber"].keys()))
    if unmapped:
        print(f"  ⚠ {len(unmapped)} cell type(s) not in DRG_HIERARCHY (bucketed 'Unclassified' at "
              f"subcategory/fiber level, excluded from fine heatmap): {unmapped}")

    fiber_bounds_fine, subcat_bounds_fine = compute_fine_boundaries(DRG_HIERARCHY)
    fine_order = hl["fine_order"]
    fine_tick_colors = {ct: hl["fiber_colors"][hl["ct_to_fiber"][ct]] for ct in fine_order}
    fine_mat = pivot_symmetric(edge_counts_df, "cell_type_1", "cell_type_2", fine_order, "total")
    if not no_plots:
        plot_hierarchy_heatmap(fine_mat, fine_order, fine_tick_colors, out / "heatmap_celltype.png",
                                "Inter-Cell-Type Edges — Fine Cell Type\n(thick=fiber boundary, dashed=subcategory boundary)",
                                major_boundaries=fiber_bounds_fine, minor_boundaries=subcat_bounds_fine)
        # NEW: density-normalized twin — same edges, divided by
        # n_genes(ct1) * n_genes(ct2), so a cell-type pair with huge gene
        # sets doesn't just look "more connected" by virtue of having more
        # genes available to form edges from.
        fine_density_mat = pivot_symmetric(density_df, "cell_type_1", "cell_type_2", fine_order, "density_percent")
        plot_hierarchy_heatmap(fine_density_mat, fine_order, fine_tick_colors, out / "heatmap_celltype_density.png",
                                "Inter-Cell-Type Edge DENSITY — Fine Cell Type\n"
                                "(% of possible gene-gene pairs realized; thick=fiber, dashed=subcategory)",
                                major_boundaries=fiber_bounds_fine, minor_boundaries=subcat_bounds_fine,
                                cbar_label="Edge density (% of possible gene pairs)", fmt=".4f")

    subcat_df = build_group_rollup(annotations_df, hl["ct_to_subcat"])
    subcat_df.to_csv(out / "edge_counts_by_subcategory.csv", index=False)
    subcat_gene_sizes = build_group_gene_universe(gene_sets, hl["ct_to_subcat"])
    subcat_density_df = calculate_group_density(subcat_df, subcat_gene_sizes)
    subcat_density_df.to_csv(out / "edge_density_by_subcategory.csv", index=False)
    subcat_order = list(hl["subcat_order"])
    subcat_tick_colors = {s: hl["fiber_colors"][hl["subcat_to_fiber"][s]] for s in subcat_order}
    if len(subcat_df) and "Unclassified" in set(subcat_df["group_1"]).union(subcat_df["group_2"]):
        subcat_order = subcat_order + ["Unclassified"]
        subcat_tick_colors["Unclassified"] = UNCLASSIFIED_COLOR
    subcat_bounds = compute_subcat_boundaries(DRG_HIERARCHY)
    subcat_mat = pivot_symmetric(subcat_df, "group_1", "group_2", subcat_order, "total")
    if not no_plots:
        plot_hierarchy_heatmap(subcat_mat, subcat_order, subcat_tick_colors, out / "heatmap_subcategory.png",
                                "Inter-Cell-Type Edges — DRG Subcategory Level", major_boundaries=subcat_bounds)
        subcat_density_mat = pivot_symmetric(subcat_density_df, "group_1", "group_2", subcat_order, "density_percent")
        plot_hierarchy_heatmap(subcat_density_mat, subcat_order, subcat_tick_colors, out / "heatmap_subcategory_density.png",
                                "Inter-Cell-Type Edge DENSITY — DRG Subcategory Level\n(% of possible gene-gene pairs realized)",
                                major_boundaries=subcat_bounds,
                                cbar_label="Edge density (% of possible gene pairs)", fmt=".4f")
        plot_edgetype_breakdown(subcat_df, "group_1", "group_2", out / "edgetype_breakdown_subcategory.png",
                                 "Edge Type Breakdown — Subcategory Level")

    fiber_df = build_group_rollup(annotations_df, hl["ct_to_fiber"])
    fiber_df.to_csv(out / "edge_counts_by_fibertype.csv", index=False)
    fiber_gene_sizes = build_group_gene_universe(gene_sets, hl["ct_to_fiber"])
    fiber_density_df = calculate_group_density(fiber_df, fiber_gene_sizes)
    fiber_density_df.to_csv(out / "edge_density_by_fibertype.csv", index=False)
    fiber_order = list(hl["fiber_order"])
    fiber_tick_colors = {f: hl["fiber_colors"][f] for f in fiber_order}
    if len(fiber_df) and "Unclassified" in set(fiber_df["group_1"]).union(fiber_df["group_2"]):
        fiber_order = fiber_order + ["Unclassified"]
        fiber_tick_colors["Unclassified"] = UNCLASSIFIED_COLOR
    fiber_mat = pivot_symmetric(fiber_df, "group_1", "group_2", fiber_order, "total")
    if not no_plots:
        plot_hierarchy_heatmap(fiber_mat, fiber_order, fiber_tick_colors, out / "heatmap_fibertype.png",
                                "Inter-Cell-Type Edges — Fiber Type Level")
        fiber_density_mat = pivot_symmetric(fiber_density_df, "group_1", "group_2", fiber_order, "density_percent")
        plot_hierarchy_heatmap(fiber_density_mat, fiber_order, fiber_tick_colors, out / "heatmap_fibertype_density.png",
                                "Inter-Cell-Type Edge DENSITY — Fiber Type Level\n(% of possible gene-gene pairs realized)",
                                cbar_label="Edge density (% of possible gene pairs)", fmt=".4f")
        plot_edgetype_breakdown(fiber_df, "group_1", "group_2", out / "edgetype_breakdown_fibertype.png",
                                 "Edge Type Breakdown — Fiber Type Level")

    return {"subcat_df": subcat_df, "fiber_df": fiber_df, "unmapped": unmapped,
            "subcat_density_df": subcat_density_df, "fiber_density_df": fiber_density_df}


# ============================================================================
# Density metrics (fine cell-type-pair level)
# ============================================================================
def calculate_density(edge_counts_df: pd.DataFrame, gene_sets: dict) -> pd.DataFrame:
    rows = []
    for _, row in edge_counts_df.iterrows():
        ct1, ct2 = row["cell_type_1"], row["cell_type_2"]
        n1, n2 = len(gene_sets.get(ct1, set())), len(gene_sets.get(ct2, set()))
        n_possible = n1 * n2
        n_actual = row["total"]
        density = 100 * n_actual / n_possible if n_possible else 0
        rows.append({"cell_type_1": ct1, "cell_type_2": ct2, "n_genes_1": n1, "n_genes_2": n2,
                      "n_possible_edges": n_possible, "n_actual_edges": n_actual, "density_percent": density})
    return pd.DataFrame(rows)


# ============================================================================
# Gene overlap across cell types (from old step10 — now trivial: reuse the
# gene_to_celltypes reverse index built straight from celltype_gene_sets.json)
# ============================================================================
def analyze_gene_overlap(gene_sets: dict) -> dict:
    gene_to_celltypes = defaultdict(set)
    for ct, genes in gene_sets.items():
        for g in genes:
            gene_to_celltypes[g].add(ct)

    multi = {g: cts for g, cts in gene_to_celltypes.items() if len(cts) > 1}
    single = {g: cts for g, cts in gene_to_celltypes.items() if len(cts) == 1}
    overlap_counts = defaultdict(int)
    for g, cts in multi.items():
        overlap_counts[len(cts)] += 1

    return {"total_unique_genes": len(gene_to_celltypes), "genes_in_single_celltype": len(single),
            "genes_in_multiple_celltypes": len(multi), "overlap_distribution": dict(overlap_counts),
            "gene_celltype_map": dict(gene_to_celltypes), "multi_celltype_genes": multi}


def save_gene_overlap_outputs(overlap: dict, out: Path):
    rows = [{"gene": g, "n_celltypes": len(cts), "cell_types": "|".join(sorted(cts))}
            for g, cts in overlap["multi_celltype_genes"].items()]
    df = pd.DataFrame(rows).sort_values("n_celltypes", ascending=False) if rows else pd.DataFrame(columns=["gene", "n_celltypes", "cell_types"])
    df.to_csv(out / "genes_shared_across_celltypes.csv", index=False)

    if not overlap["total_unique_genes"]:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dist = overlap["overlap_distribution"]
    if dist:
        axes[0].bar(dist.keys(), dist.values(), color="#1f77b4", edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Number of cell types sharing gene"); axes[0].set_ylabel("Count")
    axes[0].set_title("Gene sharing distribution", fontweight="bold")
    labels = ["Single cell type", "Multiple cell types"]
    values = [overlap["genes_in_single_celltype"], overlap["genes_in_multiple_celltypes"]]
    axes[1].pie(values, labels=labels, autopct="%1.1f%%", colors=["#d62728", "#2ca02c"], startangle=90)
    axes[1].set_title("Gene categorization", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "gene_overlap_distribution.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved -> {out / 'gene_overlap_distribution.png'}")


# ============================================================================
# Report
# ============================================================================
def write_report(edge_counts_df, density_df, pathway_involvement, gene_involvement, overlap, hierarchy_info,
                  lr_interaction_info, output_path: Path):
    lines = ["=" * 80, "INTER-CELL-TYPE CONNECTIVITY ANALYSIS REPORT", "=" * 80]
    total = edge_counts_df["total"].sum() if len(edge_counts_df) else 0
    lines.append(f"\nTotal cell-type pairs connected : {len(edge_counts_df)}")
    lines.append(f"Total inter-cell-type edges     : {total:,}")

    lines.append("\nTOP 10 CELL-TYPE PAIRS BY VOLUME\n" + "-" * 80)
    for _, row in edge_counts_df.head(10).iterrows():
        lines.append(f"  {row['cell_type_1']:<30} <-> {row['cell_type_2']:<30}  "
                      f"L-R: {row['n_ligand_receptor']:>6,} | Reactome: {row['n_reactome']:>6,} | Total: {row['total']:>8,}")

    fiber_df, subcat_df = hierarchy_info["fiber_df"], hierarchy_info["subcat_df"]
    lines.append("\nFIBER-TYPE LEVEL BREAKDOWN — RAW COUNTS\n" + "-" * 80)
    for _, row in fiber_df.sort_values("total", ascending=False).iterrows():
        lines.append(f"  {row['group_1']:<20} <-> {row['group_2']:<20}  total: {row['total']:>8,}")
    if hierarchy_info["unmapped"]:
        lines.append(f"\n  ⚠ Cell types not in DRG_HIERARCHY (bucketed 'Unclassified'): {hierarchy_info['unmapped']}")

    # NEW: density-normalized ranking, right next to the raw-count one
    # above, so re-ranking (e.g. 'Unclassified' dropping out of the top
    # spot once it's not just winning by gene-pool size) is easy to see.
    fiber_density_df = hierarchy_info.get("fiber_density_df")
    lines.append("\nFIBER-TYPE LEVEL BREAKDOWN — DENSITY-NORMALIZED (% of possible gene pairs realized)\n" + "-" * 80)
    if fiber_density_df is not None and len(fiber_density_df):
        for _, row in fiber_density_df.sort_values("density_percent", ascending=False).iterrows():
            lines.append(f"  {row['group_1']:<20} <-> {row['group_2']:<20}  "
                          f"density: {row['density_percent']:>10.4f}%  "
                          f"({row['n_actual_edges']:>8,} / {row['n_possible_edges']:>14,} possible)")
        lines.append("  (a group's raw edge count naturally scales with how many genes it has -- density is what "
                      "tells you if a group is actually more densely wired than another, not just bigger)")
    else:
        lines.append("  (no data)")

    if pathway_involvement:
        top_pathways = sorted(((p, len(pairs)) for p, pairs in pathway_involvement.items()), key=lambda x: -x[1])[:10]
        lines.append("\nTOP 10 PATHWAYS BY CELL-TYPE-PAIR INVOLVEMENT\n" + "-" * 80)
        for i, (p, n) in enumerate(top_pathways, 1):
            lines.append(f"  {i:2}. {p:<60} ({n} pairs)")

    top_genes = sorted(((g, len(pairs)) for g, pairs in gene_involvement.items()), key=lambda x: -x[1])[:10]
    lines.append("\nTOP 10 GENES BY INTER-CELL-TYPE INVOLVEMENT\n" + "-" * 80)
    for i, (g, n) in enumerate(top_genes, 1):
        lines.append(f"  {i:2}. {g:<20} ({n} pairs)")

    lines.append("\nGENE OVERLAP ACROSS CELL TYPES\n" + "-" * 80)
    lines.append(f"  Total unique genes (across all cell types) : {overlap['total_unique_genes']:,}")
    lines.append(f"  Genes in exactly 1 cell type                : {overlap['genes_in_single_celltype']:,}")
    lines.append(f"  Genes in >1 cell type                       : {overlap['genes_in_multiple_celltypes']:,}")

    if len(density_df):
        lines.append("\nDENSITY METRICS — FINE CELL-TYPE PAIRS (Top 5 by Edge Count)\n" + "-" * 80)
        for _, row in density_df.sort_values("n_actual_edges", ascending=False).head(5).iterrows():
            lines.append(f"  {row['cell_type_1']:<30} <-> {row['cell_type_2']:<30}  "
                          f"{row['n_actual_edges']:>6,} / {row['n_possible_edges']:>10,} possible "
                          f"({row['density_percent']:.3f}%)")
        lines.append("\nDENSITY METRICS — FINE CELL-TYPE PAIRS (Top 5 by DENSITY, min 1 edge)\n" + "-" * 80)
        nonzero = density_df[density_df["n_actual_edges"] > 0]
        for _, row in nonzero.sort_values("density_percent", ascending=False).head(5).iterrows():
            lines.append(f"  {row['cell_type_1']:<30} <-> {row['cell_type_2']:<30}  "
                          f"{row['n_actual_edges']:>6,} / {row['n_possible_edges']:>10,} possible "
                          f"({row['density_percent']:.3f}%)")

    lines.append("\nLR INTERACTION-TYPE BREAKDOWN (method/source + NeuronChat ligand_type)\n" + "-" * 80)
    if lr_interaction_info is None:
        lines.append("  (skipped -- no scored_lr_edge_provenance.csv found)")
    else:
        by_source, by_ligand_type = lr_interaction_info["by_source"], lr_interaction_info["by_ligand_type"]
        lines.append("  By method/source:")
        if by_source.empty:
            lines.append("    (no provenance-matched edges)")
        else:
            for src, n in by_source.items():
                lines.append(f"    {src:<20} {n:>8,}")
        lines.append("  By NeuronChat ligand_type:")
        if by_ligand_type.empty:
            lines.append("    (no provenance-matched edges)")
        else:
            for lt, n in by_ligand_type.items():
                lines.append(f"    {lt:<20} {n:>8,}")

    lines.append("\n" + "=" * 80 + "\nEnd of report\n" + "=" * 80)
    text = "\n".join(lines)
    output_path.write_text(text)
    print(f"\nSaved report -> {output_path}")
    print(text)


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Inter-cell-type connectivity analysis (heterogeneous structure)")
    ap.add_argument("--wdir", required=True, help="MultiXrank wdir (needs bipartite_manifest.json + celltype_gene_sets.json)")
    ap.add_argument("--lr-csv", default=None, help="Optional, to label edges as ligand_receptor")
    ap.add_argument("--reactome-csv", default=None, help="Optional, to label edges as reactome + get pathway names")
    ap.add_argument("--uniprot-map", default=None)
    ap.add_argument("--scored-lr-provenance-csv", default=None,
                     help="Optional override for scored_lr_edge_provenance.csv's location "
                          "(default: auto-discovered at <wdir>/scored_lr_edge_provenance.csv, "
                          "written by step2_build_bipartite_edges.py when run with --scored-lr-csv)")
    ap.add_argument("--output-dir", default="interlayer_analysis")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    wdir = Path(args.wdir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80 + "\nSTEP 7c: INTER-CELL-TYPE CONNECTIVITY ANALYSIS\n" + "=" * 80 + "\n")

    gene_sets = load_celltype_gene_sets(wdir)
    bip_df = load_bipartite_edges(wdir)

    lr_set = load_lr_set(args.lr_csv) if args.lr_csv else None
    pw_map = None
    if args.reactome_csv:
        gene_universe = set().union(*gene_sets.values()) if gene_sets else set()
        pw_map = load_pathway_map(args.reactome_csv, args.uniprot_map, gene_universe)
        print(f"Loaded pathway map: {len(pw_map):,} gene pairs with shared-pathway annotation\n")

    prov_df = load_lr_provenance(wdir, args.scored_lr_provenance_csv)
    prov_lookup = build_provenance_lookup(prov_df)

    annotations_df, edge_counts, pathway_involvement, gene_involvement = label_and_count_edges(
        bip_df, lr_set, pw_map, prov_lookup)
    annotations_df.to_csv(out / "annotations.csv", index=False)

    edge_counts_df = edge_counts_to_df(edge_counts)
    edge_counts_df.to_csv(out / "edge_counts_by_celltype_pair.csv", index=False)
    print("SECTION 1: EDGE COUNTS BY CELL-TYPE PAIR")
    print(edge_counts_df.to_string(index=False))

    # Computed up front (moved ahead of the hierarchy section) since the
    # hierarchy heatmaps now need it too, for the fine-level density plot.
    density_df = calculate_density(edge_counts_df, gene_sets)
    density_df.to_csv(out / "density.csv", index=False)

    print("\nSECTION 2: DRG HIERARCHY BREAKDOWN (raw counts + density-normalized)")
    hierarchy_info = hierarchy_breakdown_section(edge_counts_df, annotations_df, gene_sets, density_df, out, args.no_plots)

    print("\nSECTION 3: DENSITY METRICS")
    print(density_df.to_string(index=False))

    print("\nSECTION 4: GENE OVERLAP ACROSS CELL TYPES")
    overlap = analyze_gene_overlap(gene_sets)
    if not args.no_plots:
        save_gene_overlap_outputs(overlap, out)
    else:
        rows = [{"gene": g, "n_celltypes": len(cts), "cell_types": "|".join(sorted(cts))}
                for g, cts in overlap["multi_celltype_genes"].items()]
        pd.DataFrame(rows).sort_values("n_celltypes", ascending=False).to_csv(
            out / "genes_shared_across_celltypes.csv", index=False)
    print(f"  Total unique genes: {overlap['total_unique_genes']:,} | "
          f"in >1 cell type: {overlap['genes_in_multiple_celltypes']:,}")

    lr_interaction_info = None
    if prov_df is not None:
        lr_interaction_info = lr_interaction_type_section(annotations_df, out, args.no_plots)

    print("\nSECTION 5: REPORT")
    write_report(edge_counts_df, density_df, pathway_involvement, gene_involvement, overlap,
                 hierarchy_info, lr_interaction_info, out / "interlayer_analysis_report.txt")

    print(f"\nAll outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()