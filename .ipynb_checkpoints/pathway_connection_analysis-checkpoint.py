#!/usr/bin/env python3
"""
pathway_connectivity_analysis.py
=================================
Analyses which pathways and ligand-receptor pairs connect which cell types
and cell-type groups in a MultiXrank multiplex.

Changes vs previous version
----------------------------
  1. All group-level plots / tables use COL_AI ("category") exclusively.
     The COL_BROAD column is still loaded but not used for any outputs.
  2. Heatmaps are normalised by the total number of genes in the
     connecting cell type (edges per gene in that cell type).
  3. Heatmap tick-labels show the "category" value for each cell type,
     not the raw cell-type name.
"""
import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

DIVIDER    = "=" * 72
SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
COL_CELL_TYPE = "Cell Type"
COL_BROAD     = "Broad Category - ai"   # kept for loading; NOT used in outputs
COL_AI        = "category"              # PRIMARY group column throughout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_block(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def load_categories(
    csv_path: Path,
    exclude_ai: list[str] | None = None,
) -> pd.DataFrame:
    """Load layers_categories.csv.  Only COL_AI exclusions are applied."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]

    required = {COL_CELL_TYPE, COL_BROAD, COL_AI}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"layers_categories.csv missing columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    before = len(df)
    if exclude_ai:
        mask = df[COL_AI].str.lower().isin([e.lower() for e in exclude_ai])
        df   = df[~mask]
        print(f"Excluded {before - len(df)} rows by '{COL_AI}' ({exclude_ai})")

    print(f"Retained {len(df)} cell types")
    return df.reset_index(drop=True)


def stem_from_cell_type(cell_type: str) -> str:
    return cell_type.replace(" ", "_") + "_layer"


def load_layer_genes(path: Path) -> set[str]:
    """Return all gene symbols appearing in a tab-separated layer TSV."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(
            path, sep="\t", header=None,
            names=["g1", "g2"], dtype=str,
            on_bad_lines="skip",
        )
    except Exception:
        return set()
    genes: set[str] = set()
    for col in ("g1", "g2"):
        genes.update(
            df[col].dropna()
                   .str.strip()
                   .str.upper()
                   .replace("", pd.NA)
                   .dropna()
        )
    genes.discard("NAN")
    return genes


def load_edge_list(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(
            path, sep="\t", header=None,
            names=["g1", "g2"], dtype=str,
            on_bad_lines="skip",
        )
    except Exception:
        return []
    edges = []
    for _, row in df.iterrows():
        a = str(row["g1"]).strip().upper()
        b = str(row["g2"]).strip().upper()
        if a and b and a != "NAN" and b != "NAN" and a != b:
            edges.append((a, b))
    return edges


# ---------------------------------------------------------------------------
# LR CSV loading
# ---------------------------------------------------------------------------

def load_lr_csv(lr_csv_path: Path) -> dict[str, str]:
    """receptor_gname.upper() -> human_receptor description."""
    if not lr_csv_path.exists():
        print(f"  ⚠  LR CSV not found: {lr_csv_path}")
        return {}
    df = pd.read_csv(lr_csv_path)
    if df.columns[0].strip() in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]
    df.columns = df.columns.str.strip().str.lower()

    rec_gname_col = next(
        (c for c in df.columns if c in ("human_receptor", "receptor_gene")), None
    )
    human_rec_col = next(
        (c for c in df.columns if c in ("receptor_description", "receptor_name")), None
    )
    if not rec_gname_col or not human_rec_col:
        print(
            f"  ⚠  LR CSV: could not find receptor_gname / human_receptor columns.\n"
            f"     Found: {list(df.columns)}"
        )
        return {}

    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        gname = str(row[rec_gname_col]).strip().upper()
        hname = str(row[human_rec_col]).strip()
        if gname and hname and gname != "NAN" and hname != "NAN":
            lookup[gname] = hname
    print(f"  Loaded {len(lookup):,} receptor_gname → human_receptor mappings")
    return lookup


# ---------------------------------------------------------------------------
# Step 1 — Build gene → cell-type index  +  gene-count per cell type
# ---------------------------------------------------------------------------

def build_gene_celltype_index(
    multiplex_dir: Path,
    categories_df: pd.DataFrame,
) -> tuple[
    dict[str, set[str]],   # gene  -> set of cell types
    dict[str, str],         # cell type -> COL_BROAD  (kept for completeness)
    dict[str, str],         # cell type -> COL_AI
    dict[str, int],         # cell type -> n_genes  (NEW)
]:
    _print_block("Building gene → cell-type index")

    celltype_to_broad  = dict(zip(categories_df[COL_CELL_TYPE], categories_df[COL_BROAD]))
    celltype_to_ai     = dict(zip(categories_df[COL_CELL_TYPE], categories_df[COL_AI]))
    celltype_gene_count: dict[str, int] = {}

    gene_to_celltypes: dict[str, set[str]] = defaultdict(set)
    found = missing = 0

    for _, row in categories_df.iterrows():
        cell_type = row[COL_CELL_TYPE]
        tsv_path  = multiplex_dir / f"{stem_from_cell_type(cell_type)}.tsv"
        genes     = load_layer_genes(tsv_path)
        if not genes:
            missing += 1
            celltype_gene_count[cell_type] = 0
            continue
        found += 1
        celltype_gene_count[cell_type] = len(genes)
        for g in genes:
            gene_to_celltypes[g].add(cell_type)

    print(f"  Cell types found    : {found}")
    print(f"  Cell types missing  : {missing}")
    print(f"  Unique genes indexed: {len(gene_to_celltypes):,}")
    return (
        dict(gene_to_celltypes),
        celltype_to_broad,
        celltype_to_ai,
        celltype_gene_count,
    )


# ---------------------------------------------------------------------------
# Step 2 — Load pathway + LR edges with labels
# ---------------------------------------------------------------------------

def load_all_edges(
    multiplex_dir: Path,
    pathway_stems: list[str],
    lr_stems: list[str],
    receptor_lookup: dict[str, str],
) -> list[tuple[str, str, str]]:
    _print_block("Loading pathway / LR edges")
    all_edges: list[tuple[str, str, str]] = []

    for stem in pathway_stems:
        path  = multiplex_dir / f"{stem}.tsv"
        edges = load_edge_list(path)
        if not edges:
            print(f"  ⚠  {stem}.tsv — not found or empty")
            continue
        for a, b in edges:
            all_edges.append((a, b, "pathway"))
        print(f"  {stem}.tsv (pathway) : {len(edges):,} edges")

    for stem in lr_stems:
        path  = multiplex_dir / f"{stem}.tsv"
        edges = load_edge_list(path)
        if not edges:
            print(f"  ⚠  {stem}.tsv — not found or empty")
            continue
        labelled = unlabelled = 0
        for a, b in edges:
            label = receptor_lookup.get(b) or receptor_lookup.get(a)
            if label:
                labelled += 1
            else:
                label      = "ligand_receptor"
                unlabelled += 1
            all_edges.append((a, b, label))
        print(
            f"  {stem}.tsv (LR)      : {len(edges):,} edges  "
            f"| labelled={labelled:,}  unlabelled={unlabelled:,}"
        )

    unique_labels = {e[2] for e in all_edges}
    print(f"\n  Total edges loaded  : {len(all_edges):,}")
    print(f"  Unique edge labels  : {len(unique_labels):,}")
    return all_edges


# ---------------------------------------------------------------------------
# Step 3 — Map edges to cell-type pairs
# ---------------------------------------------------------------------------

def map_edges_to_celltypes(
    edges: list[tuple[str, str, str]],
    gene_to_celltypes: dict[str, set[str]],
    celltype_to_broad: dict[str, str],
    celltype_to_ai:    dict[str, str],
) -> pd.DataFrame:
    _print_block("Mapping edges to cell-type pairs")
    rows = []
    skipped_no_gene = skipped_same_ct = 0

    for gene_a, gene_b, label in edges:
        cts_a = gene_to_celltypes.get(gene_a, set())
        cts_b = gene_to_celltypes.get(gene_b, set())
        if not cts_a or not cts_b:
            skipped_no_gene += 1
            continue
        for ct_a in cts_a:
            for ct_b in cts_b:
                ct1, ct2 = sorted([ct_a, ct_b])
                if ct1 == ct2:
                    skipped_same_ct += 1
                    continue
                rows.append({
                    "gene_a":     gene_a,
                    "gene_b":     gene_b,
                    "label":      label,
                    "edge_type":  "pathway" if label == "pathway" else "ligand_receptor",
                    "celltype_a": ct1,
                    "celltype_b": ct2,
                    # broad kept for internal use / tables only
                    "broad_a":    celltype_to_broad.get(ct1, "Unknown"),
                    "broad_b":    celltype_to_broad.get(ct2, "Unknown"),
                    # ai = primary group
                    "ai_a":       celltype_to_ai.get(ct1, "Unknown"),
                    "ai_b":       celltype_to_ai.get(ct2, "Unknown"),
                })

    df = pd.DataFrame(rows)
    print(f"  Skipped (gene not in any layer) : {skipped_no_gene:,}")
    print(f"  Skipped (same cell type)        : {skipped_same_ct:,}")
    print(f"  Retained connection rows        : {len(df):,}")
    if not df.empty:
        print(f"  Unique cell-type pairs  : "
              f"{df.groupby(['celltype_a','celltype_b']).ngroups:,}")
        print(f"  Unique labels           : {df['label'].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Aggregate tables  (group_col forced to "ai")
# ---------------------------------------------------------------------------

def build_summary_tables(
    conn_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Always aggregates by the ai / category column."""
    _print_block("Building summary tables (group = 'category')")

    a_col, b_col = "ai_a", "ai_b"

    ct_pair_df = (
        conn_df
        .groupby(["celltype_a", "celltype_b", "label", "edge_type"])
        .agg(n_edges=("gene_a", "count"))
        .reset_index()
        .sort_values("n_edges", ascending=False)
    )

    group_pair_df = (
        conn_df
        .groupby([a_col, b_col, "label", "edge_type"])
        .agg(n_edges=("gene_a", "count"))
        .reset_index()
        .rename(columns={a_col: "group_a", b_col: "group_b"})
        .sort_values("n_edges", ascending=False)
    )

    combined = pd.concat([
        conn_df[[a_col, "label"]].rename(columns={a_col: "group"}),
        conn_df[[b_col, "label"]].rename(columns={b_col: "group"}),
    ])
    pw_group_matrix = (
        combined
        .groupby(["label", "group"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    top_pw_df = (
        group_pair_df
        .sort_values(["group_a", "group_b", "n_edges"],
                     ascending=[True, True, False])
        .groupby(["group_a", "group_b"])
        .head(10)
        .reset_index(drop=True)
    )

    print(f"  Cell-type pair rows  : {len(ct_pair_df):,}")
    print(f"  Group pair rows      : {len(group_pair_df):,}")
    print(f"  Label × group matrix : {pw_group_matrix.shape}")
    return ct_pair_df, group_pair_df, pw_group_matrix, top_pw_df


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _norm_value(
    raw: float,
    ct_a: str,
    ct_b: str,
    celltype_gene_count: dict[str, int],
) -> float:
    """
    Divide edge count by the geometric mean of gene counts of the two
    cell types, giving edges-per-gene (×1000 for readability).
    Falls back to raw count if either gene count is zero/missing.
    """
    n_a = celltype_gene_count.get(ct_a, 0)
    n_b = celltype_gene_count.get(ct_b, 0)
    if n_a > 0 and n_b > 0:
        return raw / np.sqrt(n_a * n_b) * 1_000
    return raw


# ---------------------------------------------------------------------------
# Step 5 — Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmap_celltype_pairs(
    conn_df: pd.DataFrame,
    celltype_gene_count: dict[str, int],
    celltype_to_ai: dict[str, str],
    out: Path,
    top_n: int = 40,
) -> None:
    """
    Heatmap: cell type × cell type.
    • Axes are labelled with the "category" value for each cell type.
    • Values are normalised by √(n_genes_A × n_genes_B) × 1000.
    """
    pivot_total = (
        conn_df
        .groupby(["celltype_a", "celltype_b"])
        .agg(n=("gene_a", "count"))
        .reset_index()
    )

    # Pick the top-N most-connected cell types
    top_cts = (
        pd.concat([pivot_total["celltype_a"], pivot_total["celltype_b"]])
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    pivot_total = pivot_total[
        pivot_total["celltype_a"].isin(top_cts) &
        pivot_total["celltype_b"].isin(top_cts)
    ]

    # Category labels for axes (cell type → category string)
    cat_labels = [celltype_to_ai.get(ct, ct) for ct in top_cts]

    def _symmetric_norm_matrix(piv: pd.DataFrame) -> pd.DataFrame:
        """Build symmetric, normalised matrix indexed by top_cts."""
        mat = (
            piv.pivot(index="celltype_a", columns="celltype_b", values="n")
               .reindex(index=top_cts, columns=top_cts, fill_value=0)
               .fillna(0)
        )
        # Symmetrise
        mat = mat + mat.T
        vals = mat.to_numpy().astype(float).copy()
        np.fill_diagonal(vals, 0.0)
        # Normalise each cell by √(genes_row × genes_col)
        for i, cta in enumerate(top_cts):
            for j, ctb in enumerate(top_cts):
                if i != j:
                    vals[i, j] = _norm_value(
                        vals[i, j], cta, ctb, celltype_gene_count
                    )
        return pd.DataFrame(vals, index=cat_labels, columns=cat_labels)

    edge_types = conn_df["edge_type"].unique().tolist()
    n_panels   = 1 + len(edge_types)
    fig_h      = max(8, len(top_cts) * 0.28)
    fig_w      = fig_h * n_panels * 0.9

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = [axes]

    cmaps = ["YlOrRd", "Blues", "Greens"]

    mat_all = _symmetric_norm_matrix(pivot_total)
    sns.heatmap(
        mat_all, ax=axes[0], cmap=cmaps[0],
        linewidths=0.2, linecolor="white",
        cbar_kws={"label": "edges / √(genes_A·genes_B) ×1k", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    axes[0].set_title("All edges (normalised)", fontsize=8, fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=5)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0,  fontsize=5)

    for i, et in enumerate(edge_types, start=1):
        sub = conn_df[conn_df["edge_type"] == et]
        piv = (
            sub.groupby(["celltype_a", "celltype_b"])
               .agg(n=("gene_a", "count"))
               .reset_index()
        )
        piv = piv[
            piv["celltype_a"].isin(top_cts) &
            piv["celltype_b"].isin(top_cts)
        ]
        mat = (
            _symmetric_norm_matrix(piv) if not piv.empty
            else pd.DataFrame(0.0, index=cat_labels, columns=cat_labels)
        )
        sns.heatmap(
            mat, ax=axes[i], cmap=cmaps[i % len(cmaps)],
            linewidths=0.2, linecolor="white",
            cbar_kws={"label": "edges / √(genes_A·genes_B) ×1k", "shrink": 0.6},
            xticklabels=True, yticklabels=True,
        )
        axes[i].set_title(f"{et} (normalised)", fontsize=8, fontweight="bold")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90, fontsize=5)
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0,  fontsize=5)

    fig.suptitle(
        f"Cell-type connectivity — top {top_n} (category labels, normalised)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"heatmap_celltype_connections.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


def plot_heatmap_group_pairs(
    conn_df: pd.DataFrame,
    celltype_gene_count: dict[str, int],
    out: Path,
) -> None:
    """
    Heatmap: category × category.
    Values normalised: edges / √(total_genes_groupA × total_genes_groupB) × 1000.
    """
    a_col, b_col = "ai_a", "ai_b"
    all_groups   = sorted(set(conn_df[a_col]) | set(conn_df[b_col]))

    # Pre-compute total gene count per category group
    # (sum of all cell-type gene counts in that category)
    # We need celltype_to_ai mapping — reconstruct from conn_df columns
    ct_ai = (
        pd.concat([
            conn_df[["celltype_a", "ai_a"]].rename(
                columns={"celltype_a": "ct", "ai_a": "ai"}),
            conn_df[["celltype_b", "ai_b"]].rename(
                columns={"celltype_b": "ct", "ai_b": "ai"}),
        ])
        .drop_duplicates("ct")
        .set_index("ct")["ai"]
        .to_dict()
    )
    group_gene_count: dict[str, int] = defaultdict(int)
    for ct, n in celltype_gene_count.items():
        grp = ct_ai.get(ct, "Unknown")
        group_gene_count[grp] += n

    def _group_norm_matrix(subset: pd.DataFrame) -> pd.DataFrame:
        piv = (
            subset.groupby([a_col, b_col])
                  .agg(n=("gene_a", "count"))
                  .reset_index()
        )
        mat = pd.DataFrame(0.0, index=all_groups, columns=all_groups)
        for _, row in piv.iterrows():
            mat.loc[row[a_col], row[b_col]] += row["n"]
            mat.loc[row[b_col], row[a_col]] += row["n"]
        vals = mat.to_numpy().astype(float).copy()
        np.fill_diagonal(vals, 0.0)
        # Normalise
        for i, ga in enumerate(all_groups):
            for j, gb in enumerate(all_groups):
                if i != j:
                    na = group_gene_count.get(ga, 0)
                    nb = group_gene_count.get(gb, 0)
                    if na > 0 and nb > 0:
                        vals[i, j] = vals[i, j] / np.sqrt(na * nb) * 1_000
        return pd.DataFrame(vals, index=all_groups, columns=all_groups)

    edge_types = conn_df["edge_type"].unique().tolist()
    n_panels   = 1 + len(edge_types)
    annot      = len(all_groups) <= 20
    fig_h      = max(6, len(all_groups) * 0.45)
    fig_w      = fig_h * n_panels * 0.85

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = [axes]

    cmaps = ["Blues", "Purples", "Greens"]

    sns.heatmap(
        _group_norm_matrix(conn_df), ax=axes[0], cmap=cmaps[0],
        annot=annot, fmt=".1f" if annot else "",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "edges / √(genes_A·genes_B) ×1k", "shrink": 0.7},
        xticklabels=True, yticklabels=True,
    )
    axes[0].set_title("All edges (normalised)", fontsize=8, fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=7)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=7)

    for i, et in enumerate(edge_types, start=1):
        sub = conn_df[conn_df["edge_type"] == et]
        mat = (
            _group_norm_matrix(sub) if not sub.empty
            else pd.DataFrame(0.0, index=all_groups, columns=all_groups)
        )
        sns.heatmap(
            mat, ax=axes[i], cmap=cmaps[i % len(cmaps)],
            annot=annot, fmt=".1f" if annot else "",
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": "edges / √(genes_A·genes_B) ×1k", "shrink": 0.7},
            xticklabels=True, yticklabels=True,
        )
        axes[i].set_title(f"{et} (normalised)", fontsize=8, fontweight="bold")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right", fontsize=7)
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0, fontsize=7)

    fig.suptitle(
        f"Category-level connectivity (normalised by gene count)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"heatmap_category_group_connections.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


def plot_heatmap_label_vs_group(
    pw_group_matrix: pd.DataFrame,
    out: Path,
    top_n_labels: int = 50,
) -> None:
    """Heatmap: edge label × category group."""
    row_sums   = pw_group_matrix.sum(axis=1).sort_values(ascending=False)
    top_labels = row_sums.head(top_n_labels).index
    mat        = pw_group_matrix.loc[top_labels].copy()

    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        if len(mat) > 2:
            Z = linkage(mat.values, method="ward")
            d = dendrogram(Z, no_plot=True)
            mat = mat.iloc[d["leaves"]]
    except Exception:
        pass

    fig_h = max(8,  len(mat)         * 0.22)
    fig_w = max(10, len(mat.columns) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        mat, ax=ax, cmap="viridis",
        linewidths=0.15, linecolor="white",
        cbar_kws={"label": "Edge count"},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title(
        f"Top {top_n_labels} edge labels × category groups",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"heatmap_label_vs_category_group.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 6 — Bar charts  (all use "category" / ai)
# ---------------------------------------------------------------------------

def plot_top_labels_overall(
    conn_df: pd.DataFrame,
    out: Path,
    top_n: int = 30,
) -> None:
    agg = (
        conn_df
        .groupby(["label", "edge_type"])
        .agg(
            n_edges=("gene_a", "count"),
            n_ct_pairs=(
                "celltype_a",
                lambda x: len(set(zip(x, conn_df.loc[x.index, "celltype_b"]))),
            ),
        )
        .reset_index()
    )
    top_by_edges = agg.sort_values("n_edges",    ascending=False).head(top_n)
    top_by_pairs = agg.sort_values("n_ct_pairs", ascending=False).head(top_n)
    palette      = {"pathway": "#1982c4", "ligand_receptor": "#ff595e"}

    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, top_n * 0.32)))
    for ax, df_top, col, xlabel in [
        (axes[0], top_by_edges, "n_edges",    "Total edges"),
        (axes[1], top_by_pairs, "n_ct_pairs", "Unique cell-type pairs"),
    ]:
        colors = [palette.get(et, "#888") for et in df_top["edge_type"]]
        ax.barh(df_top["label"], df_top[col],
                color=colors, edgecolor="white", linewidth=0.4)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_yticklabels(df_top["label"], fontsize=7)
        ax.set_title(xlabel, fontsize=9, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=et) for et, c in palette.items()]
    axes[0].legend(handles=patches, fontsize=8, loc="lower right")
    fig.suptitle(f"Top {top_n} edge labels by connectivity",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"bar_top_labels_overall.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


def plot_top_labels_per_group_pair(
    top_pw_df: pd.DataFrame,
    out: Path,
    max_pairs: int = 12,
) -> None:
    pairs = (
        top_pw_df
        .groupby(["group_a", "group_b"])["n_edges"]
        .sum()
        .sort_values(ascending=False)
        .head(max_pairs)
        .index.tolist()
    )
    n_cols = 3
    n_rows = int(np.ceil(len(pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axes = np.array(axes).flatten()

    palette = {"pathway": "#1982c4", "ligand_receptor": "#ff595e"}
    for i, (ga, gb) in enumerate(pairs):
        ax  = axes[i]
        sub = top_pw_df[
            (top_pw_df["group_a"] == ga) &
            (top_pw_df["group_b"] == gb)
        ].head(8)
        colors = [palette.get(et, "#888") for et in sub["edge_type"]]
        ax.barh(sub["label"], sub["n_edges"], color=colors, edgecolor="white")
        ax.invert_yaxis()
        ax.set_title(f"{ga}\n↔ {gb}", fontsize=7.5, fontweight="bold")
        ax.set_xlabel("Edges", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=6)

    for j in range(len(pairs), len(axes)):
        axes[j].set_visible(False)

    patches = [mpatches.Patch(color=c, label=et) for et, c in palette.items()]
    fig.legend(handles=patches, loc="lower right", fontsize=8, ncol=2)
    fig.suptitle("Top edge labels per category group pair",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"bar_top_labels_per_group_pair.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


def plot_connections_per_group(
    conn_df: pd.DataFrame,
    out: Path,
) -> None:
    """Stacked bar: edges per category group, split by partner."""
    a_col, b_col = "ai_a", "ai_b"
    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, max(5, conn_df[a_col].nunique() * 0.45 + 3)),
    )
    for ax, et, title_suffix in [
        (axes[0], "pathway",        "pathway edges"),
        (axes[1], "ligand_receptor","LR edges"),
    ]:
        sub = conn_df[conn_df["edge_type"] == et]
        if sub.empty:
            ax.set_visible(False)
            continue
        df_a = sub[[a_col, b_col]].rename(columns={a_col: "group", b_col: "partner"})
        df_b = sub[[a_col, b_col]].rename(columns={b_col: "group", a_col: "partner"})
        combined = pd.concat([df_a, df_b])
        combined = combined[combined["group"] != combined["partner"]]
        agg   = combined.groupby(["group","partner"]).size().reset_index(name="n")
        pivot = (
            agg.pivot(index="group", columns="partner", values="n").fillna(0)
        )
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        pivot.plot(kind="bar", stacked=True, ax=ax,
                   colormap="tab20", edgecolor="white", linewidth=0.3)
        ax.set_xlabel(COL_AI, fontsize=9)
        ax.set_ylabel("Edges", fontsize=9)
        ax.set_title(title_suffix.capitalize(), fontsize=9, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6, ncol=2)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle(
        "Connectivity per category group (stacked by partner)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fp = out / f"bar_connections_per_category_group.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 7 — Network visualisations  (node colour = category)
# ---------------------------------------------------------------------------

def _build_connection_graph(
    conn_df: pd.DataFrame,
    node_col_a: str,
    node_col_b: str,
    edge_type_filter: str | None = None,
) -> nx.Graph:
    sub = (
        conn_df[conn_df["edge_type"] == edge_type_filter]
        if edge_type_filter else conn_df
    )
    agg = (
        sub.groupby([node_col_a, node_col_b])
           .agg(weight=("gene_a", "count"))
           .reset_index()
    )
    G = nx.Graph()
    for _, row in agg.iterrows():
        a, b, w = row[node_col_a], row[node_col_b], int(row["weight"])
        if a != b:
            if G.has_edge(a, b):
                G[a][b]["weight"] += w
            else:
                G.add_edge(a, b, weight=w)
    return G


def plot_network(
    G: nx.Graph,
    title: str,
    out_stem: str,
    out: Path,
    node_color_map: dict | None = None,
    max_nodes: int = 60,
    min_edge_weight: int = 1,
) -> None:
    if G.number_of_nodes() == 0:
        print(f"  ⚠  {out_stem}: empty graph, skipping")
        return

    if G.number_of_nodes() > max_nodes:
        strength = {
            n: sum(d["weight"] for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }
        top_nodes = sorted(strength, key=strength.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    G.remove_edges_from([
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("weight", 1) < min_edge_weight
    ])
    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_nodes() == 0:
        print(f"  ⚠  {out_stem}: no nodes after pruning, skipping")
        return

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w   = max(weights) if weights else 1
    widths  = [1 + 4 * (w / max_w) for w in weights]
    alphas  = [0.3 + 0.65 * (w / max_w) for w in weights]
    node_sizes = [
        200 + 1500 * (
            sum(d["weight"] for _, _, d in G.edges(n, data=True))
            / (max_w * max(G.degree(n), 1))
        )
        for n in G.nodes()
    ]

    palette = plt.cm.tab20.colors
    if node_color_map:
        unique_grps    = sorted(set(node_color_map.values()))
        grp_color      = {g: palette[i % 20] for i, g in enumerate(unique_grps)}
        node_colors    = [
            grp_color.get(node_color_map.get(n, "Unknown"), "grey")
            for n in G.nodes()
        ]
        legend_patches = [
            mpatches.Patch(color=grp_color[g], label=g)
            for g in unique_grps
            if any(node_color_map.get(n) == g for n in G.nodes())
        ]
    else:
        node_colors    = ["#4ecdc4"] * G.number_of_nodes()
        legend_patches = []

    pos = nx.spring_layout(G, weight="weight", seed=42, k=2.5)
    fig, ax = plt.subplots(figsize=(16, 12))

    for (u, v), w, a in zip(G.edges(), widths, alphas):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            width=w, alpha=a, edge_color="#888888",
        )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes, node_color=node_colors, alpha=0.85,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: (n if len(n) <= 28 else n[:26] + "…") for n in G.nodes()},
        font_size=5.5, font_weight="bold",
    )
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower left",
                  fontsize=7, ncol=2, framealpha=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        fp = out / f"{out_stem}.{ext}"
        plt.savefig(fp, bbox_inches="tight", dpi=150)
        print(f"  Saved → {fp}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 8 — Save tables
# ---------------------------------------------------------------------------

def save_tables(
    ct_pair_df: pd.DataFrame,
    group_pair_df: pd.DataFrame,
    pw_group_matrix: pd.DataFrame,
    top_pw_df: pd.DataFrame,
    conn_df: pd.DataFrame,
    out: Path,
) -> None:
    _print_block("Saving tables")

    ct_pair_df.to_csv(out / "table_celltype_pair_labels.csv", index=False)
    print(f"  → table_celltype_pair_labels.csv   ({len(ct_pair_df):,} rows)")

    group_pair_df.to_csv(out / "table_group_pair_labels.csv", index=False)
    print(f"  → table_group_pair_labels.csv      ({len(group_pair_df):,} rows)")

    pw_group_matrix.to_csv(out / "table_label_group_matrix.csv")
    print(f"  → table_label_group_matrix.csv     {pw_group_matrix.shape}")

    top_pw_df.to_csv(out / "table_top_labels_per_group_pair.csv", index=False)
    print(f"  → table_top_labels_per_group_pair.csv")

    summary = (
        group_pair_df
        .groupby(["group_a", "group_b", "edge_type"])
        .agg(
            total_edges=("n_edges", "sum"),
            n_unique_labels=("label", "nunique"),
            top_label=(
                "label",
                lambda x: group_pair_df.loc[x.index]
                          .sort_values("n_edges", ascending=False)["label"].iloc[0],
            ),
        )
        .reset_index()
        .sort_values("total_edges", ascending=False)
    )
    summary.to_csv(out / "table_group_pair_summary.csv", index=False)
    print(f"  → table_group_pair_summary.csv     ({len(summary):,} rows)")

    label_summary = (
        conn_df
        .groupby(["label", "edge_type"])
        .agg(
            n_edges=("gene_a", "count"),
            n_celltype_pairs=(
                "celltype_a",
                lambda x: len(set(zip(x, conn_df.loc[x.index, "celltype_b"]))),
            ),
            # Use ai_a / ai_b (category) — NOT broad
            category_groups_connected=(
                "ai_a",
                lambda x: "|".join(
                    sorted(set(x) | set(conn_df.loc[x.index, "ai_b"]))
                ),
            ),
        )
        .reset_index()
        .sort_values("n_edges", ascending=False)
    )
    label_summary.to_csv(out / "table_label_summary.csv", index=False)
    print(f"  → table_label_summary.csv          ({len(label_summary):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse pathway/LR connectivity between cell types"
    )
    parser.add_argument("--wdir", required=True,
        help="MultiXrank working directory (contains multiplex/gene_multiplex/)")
    parser.add_argument("--csv", default=None,
        help="Path to layers_categories.csv")
    parser.add_argument("--lr-csv", default=None,
        help="Path to LR CSV")
    parser.add_argument("--pathway-stems", nargs="+", default=["pathway_layer"])
    parser.add_argument("--lr-stems",      nargs="+", default=["ligand_receptor_layer"])
    parser.add_argument("--output-dir",    default="pathway_connectivity_output")
    parser.add_argument("--top-n-labels",  type=int, default=50)
    parser.add_argument("--min-edge-weight", type=int, default=2)
    parser.add_argument("--exclude-ai",    nargs="+", default=[])
    args = parser.parse_args()

    wdir          = Path(args.wdir).resolve()
    out           = Path(args.output_dir)
    multiplex_dir = wdir / "multiplex" / "gene_multiplex"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = (
        Path(args.csv) if args.csv else SCRIPT_DIR / "layers_categories.csv"
    )
    lr_csv_path = (
        Path(args.lr_csv) if args.lr_csv else SCRIPT_DIR / "lr_pairs.csv"
    )

    # ── Load ──────────────────────────────────────────────────────────────
    categories_df   = load_categories(csv_path, exclude_ai=args.exclude_ai or None)
    receptor_lookup = load_lr_csv(lr_csv_path)

    gene_to_celltypes, celltype_to_broad, celltype_to_ai, celltype_gene_count = (
        build_gene_celltype_index(multiplex_dir, categories_df)
    )

    all_edges = load_all_edges(
        multiplex_dir,
        pathway_stems=args.pathway_stems,
        lr_stems=args.lr_stems,
        receptor_lookup=receptor_lookup,
    )
    if not all_edges:
        print("\n⚠  No edges loaded.")
        return

    conn_df = map_edges_to_celltypes(
        all_edges, gene_to_celltypes, celltype_to_broad, celltype_to_ai
    )
    if conn_df.empty:
        print("\n⚠  No connections found.")
        return

    # ── Tables ────────────────────────────────────────────────────────────
    ct_pair_df, group_pair_df, pw_group_matrix, top_pw_df = (
        build_summary_tables(conn_df)
    )

    # ── Heatmaps ──────────────────────────────────────────────────────────
    _print_block("Generating heatmaps")

    plot_heatmap_celltype_pairs(
        conn_df, celltype_gene_count, celltype_to_ai, out
    )
    plot_heatmap_group_pairs(
        conn_df, celltype_gene_count, out
    )
    plot_heatmap_label_vs_group(
        pw_group_matrix, out, top_n_labels=args.top_n_labels
    )

    # ── Bar charts ────────────────────────────────────────────────────────
    _print_block("Generating bar charts")
    plot_top_labels_overall(conn_df, out, top_n=args.top_n_labels)
    plot_top_labels_per_group_pair(top_pw_df, out)
    plot_connections_per_group(conn_df, out)

    # ── Networks  (node colour = category) ───────────────────────────────
    _print_block("Generating networks")
    for et, suffix in [
        (None,              "all"),
        ("pathway",         "pathway"),
        ("ligand_receptor", "LR"),
    ]:
        # Cell-type level — coloured by category
        plot_network(
            _build_connection_graph(conn_df, "celltype_a", "celltype_b", et),
            title=f"Cell-type connectivity — {suffix} edges",
            out_stem=f"network_celltype_{suffix}",
            out=out,
            node_color_map=celltype_to_ai,          # ← category colours
            min_edge_weight=args.min_edge_weight,
        )
        # Category-group level
        plot_network(
            _build_connection_graph(conn_df, "ai_a", "ai_b", et),
            title=f"Category group connectivity — {suffix} edges",
            out_stem=f"network_category_{suffix}",
            out=out,
            min_edge_weight=args.min_edge_weight,
        )

    # ── Save tables ───────────────────────────────────────────────────────
    save_tables(
        ct_pair_df, group_pair_df, pw_group_matrix,
        top_pw_df, conn_df, out,
    )

    # ── Done ──────────────────────────────────────────────────────────────
    _print_block("DONE")
    print(f"  Output directory       : {out.resolve()}")
    print(f"  Cell types analysed    : {categories_df[COL_CELL_TYPE].nunique()}")
    print(f"  Total edges loaded     : {len(all_edges):,}")
    print(f"  Connection rows        : {len(conn_df):,}")
    print(f"  Unique labels          : {conn_df['label'].nunique():,}")
    print(f"  Unique cell-type pairs : "
          f"{conn_df.groupby(['celltype_a','celltype_b']).ngroups:,}")
    print(f"  Unique category pairs  : "
          f"{conn_df.groupby(['ai_a','ai_b']).ngroups:,}")
    print()
    print("  Files generated:")
    for fp in sorted(out.glob("*")):
        if fp.is_file():
            print(f"    {fp.name}")


if __name__ == "__main__":
    main()