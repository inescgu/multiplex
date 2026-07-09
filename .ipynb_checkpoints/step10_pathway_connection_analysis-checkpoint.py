#!/usr/bin/env python3
"""
pathway_connectivity_analysis_v3.py
=====================================
Enhanced version with:
  • Cell-specific graphs labeled with full cell type names
  • Group categorization for all visualizations
  • Separate panels for each neuron group
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
# Cell type grouping (neuron-specific)
# ---------------------------------------------------------------------------
NEURON_GROUPS = {
    "Proprioceptors": [
        "PVALB+ RUNX3+ proprioceptor 1",
        "PVALB+ RUNX3+ proprioceptor 2",
        "PVALB+ RUNX3+ proprioceptor 3",
    ],
    "A-fiber LTMRs": [
        "NTRK3+ NTRK2+ A-beta LTMR subtype 1",
        "NTRK3+ NTRK2+ A-beta LTMR subtype 2",
        "NTRK3+ NTRK2+ A-beta LTMR subtype 3",
        "NTRK3+ NTRK2+ A-beta LTMR subtype 4",
        "A-LTMR",
    ],
    "Peptidergic nociceptors": [
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 1",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 2",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 3",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 4",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 5",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 6",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 7",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 8",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 9",
        "NP SST positive subtype",
    ],
    "C-fiber nociceptors": [
        "MRGPRX polymodal non-canonical peptidergic nociceptor 1",
        "MRGPRX polymodal non-canonical peptidergic nociceptor 2",
    ],
    "C-fiber LTMRs": [
        "C-LTMR subtype",
    ],
    "Thermoreceptors": [
        "Thermo sensitive population 1",
        "Thermo sensitive population 2",
    ],
    "Other": [
        "ATF3",
    ]
}

CELLTYPE_TO_GROUP = {
    cell: group for group, cells in NEURON_GROUPS.items() for cell in cells
}

GROUP_COLORS = dict(zip(
    NEURON_GROUPS.keys(),
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"]
))

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
COL_CELL_TYPE = "Cell Type"
COL_BROAD     = "Broad Category - ai"
COL_AI        = "category"
SOURCE_PRIORITY = ["Reactome", "KEGG"] 

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_block(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

def load_categories(csv_path: Path, exclude_ai: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]
    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
            
    before = len(df)
    if exclude_ai:
        mask = df[COL_AI].str.lower().isin([e.lower() for e in exclude_ai])
        df = df[~mask]
        print(f"Excluded {before - len(df)} rows by '{COL_AI}' ({exclude_ai})")
    print(f"Retained {len(df)} cell types")
    return df.reset_index(drop=True)

def stem_from_cell_type(cell_type: str) -> str:
    return cell_type.replace(" ", "_") + "_layer"

def load_layer_genes(path: Path) -> set[str]:
    if not path.exists(): return set()
    try:
        df = pd.read_csv(path, sep="\t", header=None, names=["g1", "g2"], dtype=str, on_bad_lines="skip")
    except Exception:
        return set()
    genes = set()
    for col in ("g1", "g2"):
        genes.update(df[col].dropna().str.strip().str.upper().replace("", pd.NA).dropna())
    genes.discard("NAN")
    return genes

def load_edge_list(path: Path) -> list[tuple[str, str]]:
    if not path.exists(): return []
    try:
        df = pd.read_csv(path, sep="\t", header=None, names=["g1", "g2"], dtype=str, on_bad_lines="skip")
    except Exception:
        return []
    edges = []
    for _, row in df.iterrows():
        a, b = str(row["g1"]).strip().upper(), str(row["g2"]).strip().upper()
        if a and b and a != "NAN" and b != "NAN" and a != b:
            edges.append((a, b))
    return edges

def load_uniprot_symbol_map(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists(): return {}
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    uid_col = next((c for c in df.columns if c in ("uniprotid", "uniprot_id", "uniprot")), None)
    sym_col = next((c for c in df.columns if c in ("symbol", "gene_symbol", "genesymbol")), None)
    
    mapping = {}
    if uid_col and sym_col:
        for _, row in df.iterrows():
            uid, sym = str(row[uid_col]).strip().upper(), str(row[sym_col]).strip().upper()
            if uid and sym and uid != "NAN" and sym != "NAN":
                mapping[uid] = sym
    return mapping

def load_reactome_pathway_map(reactome_csv: Path, uniprot_symbol_map: dict[str, str]) -> dict[tuple[str, str], list[str]]:
    if not reactome_csv.exists(): return {}
    df = pd.read_csv(reactome_csv, low_memory=False)
    df.columns = df.columns.str.strip()
    
    sym1_col = _find_col(df, ("symbol1",))
    sym2_col = _find_col(df, ("symbol2",))
    uid1_col = _find_col(df, ("uniprot1",))
    uid2_col = _find_col(df, ("uniprot2",))
    pathway_col = _find_col(df, ("common_pathway",))
    p1_col = _find_col(df, ("reactome_p1",))
    p2_col = _find_col(df, ("reactome_p2",))
    
    use_direct = (sym1_col is not None and sym2_col is not None)
    pw_map = {}
    
    for _, row in df.iterrows():
        if use_direct:
            raw1, raw2 = str(row[sym1_col]).strip().upper(), str(row[sym2_col]).strip().upper()
            sym1 = raw1 if raw1 not in ("", "NAN", "NONE", "NAT") else None
            sym2 = raw2 if raw2 not in ("", "NAN", "NONE", "NAT") else None
        else:
            u1, u2 = str(row[uid1_col]).strip().upper(), str(row[uid2_col]).strip().upper()
            sym1, sym2 = uniprot_symbol_map.get(u1), uniprot_symbol_map.get(u2)
            
        if not sym1 or not sym2 or sym1 == sym2: continue
        key = (min(sym1, sym2), max(sym1, sym2))
        
        pw_raw = row[pathway_col]
        if pd.notna(pw_raw) and str(pw_raw).strip() not in ("", "nan", "NAN"):
            pathways = [p.strip() for p in str(pw_raw).split("|") if p.strip()]
            if pathways:
                pw_map[key] = list(set(pw_map.get(key, []) + pathways))
                continue
                
        if p1_col and p2_col:
            p1_set = {p.strip() for p in str(row[p1_col]).split("|") if p.strip() and pd.notna(row[p1_col])}
            p2_set = {p.strip() for p in str(row[p2_col]).split("|") if p.strip() and pd.notna(row[p2_col])}
            union = p1_set | p2_set
            if union:
                pw_map[key] = list(set(pw_map.get(key, []) + list(union)))
                continue
                
        if key not in pw_map:
            pw_map[key] = ["Reactome_interaction"]
            
    return pw_map

def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower: return lower[cand.lower()]
    return None

def load_lr_pathway_map(lr_pathways_csv: Path) -> dict[tuple[str, str], list[str]]:
    if not lr_pathways_csv.exists(): return {}
    df = pd.read_csv(lr_pathways_csv, low_memory=False)
    df.columns = df.columns.str.strip()
    
    lig_col = _find_col(df, ("ligand_gname",))
    rec_col = _find_col(df, ("receptor_gname",))
    pw_col  = _find_col(df, ("common_pathways",))
    lr_map = {}
    
    if lig_col and rec_col:
        for _, row in df.iterrows():
            lig, rec = str(row[lig_col]).strip().upper(), str(row[rec_col]).strip().upper()
            if not lig or not rec or lig in ("NAN", "NONE") or rec in ("NAN", "NONE"): continue
            key = (min(lig, rec), max(lig, rec))
            
            if pw_col and pd.notna(row[pw_col]):
                pathways = [p.strip() for p in str(row[pw_col]).split("|") if p.strip() and str(row[pw_col]) != "nan"]
                if pathways:
                    lr_map[key] = list(set(lr_map.get(key, []) + pathways))
                    continue
                    
            if key not in lr_map:
                lr_map[key] = ["LR_interaction"]
    return lr_map

def merge_pathway_maps(reactome_map: dict, lr_map: dict) -> dict:
    merged = dict(reactome_map)
    for key, lr_names in lr_map.items():
        if key not in merged:
            merged[key] = lr_names
        else:
            merged[key] = list(set(merged[key] + [n for n in lr_names if n != "LR_interaction"]))
    return merged

def load_lr_csv(lr_csv_path: Path) -> dict[str, str]:
    if not lr_csv_path.exists(): return {}
    df = pd.read_csv(lr_csv_path)
    if df.columns[0].strip() in ("", "Unnamed: 0"): df = df.iloc[:, 1:]
    df.columns = df.columns.str.strip().str.lower()
    
    rec_gname_col = next((c for c in df.columns if c in ("human_receptor", "receptor_gene")), None)
    human_rec_col = next((c for c in df.columns if c in ("receptor_description", "receptor_name")), None)
    
    lookup = {}
    if rec_gname_col and human_rec_col:
        for _, row in df.iterrows():
            gname, hname = str(row[rec_gname_col]).strip().upper(), str(row[human_rec_col]).strip()
            if gname and hname and gname != "NAN" and hname != "NAN":
                lookup[gname] = hname
    return lookup

# ---------------------------------------------------------------------------
# Data Processing Steps
# ---------------------------------------------------------------------------
def build_gene_celltype_index(multiplex_dir: Path, categories_df: pd.DataFrame):
    celltype_to_broad = dict(zip(categories_df[COL_CELL_TYPE], categories_df[COL_BROAD]))
    celltype_to_ai    = dict(zip(categories_df[COL_CELL_TYPE], categories_df[COL_AI]))
    celltype_gene_count = {}
    gene_to_celltypes = defaultdict(set)
    
    for _, row in categories_df.iterrows():
        cell_type = row[COL_CELL_TYPE]
        genes = load_layer_genes(multiplex_dir / f"{stem_from_cell_type(cell_type)}.tsv")
        celltype_gene_count[cell_type] = len(genes)
        for g in genes:
            gene_to_celltypes[g].add(cell_type)
            
    return dict(gene_to_celltypes), celltype_to_broad, celltype_to_ai, celltype_gene_count

def load_all_edges(multiplex_dir: Path, pathway_stems: list[str], lr_stems: list[str], reactome_pw_map: dict, receptor_lookup: dict):
    all_edges = []
    for stem in pathway_stems:
        raw = load_edge_list(multiplex_dir / f"{stem}.tsv")
        for sym_a, sym_b in raw:
            key = (min(sym_a, sym_b), max(sym_a, sym_b))
            pw_list = reactome_pw_map.get(key, [])
            label = pw_list[0] if pw_list else "pathway"
            all_edges.append((sym_a, sym_b, label, "pathway"))
            
    for stem in lr_stems:
        raw = load_edge_list(multiplex_dir / f"{stem}.tsv")
        for sym_a, sym_b in raw:
            label = receptor_lookup.get(sym_b) or receptor_lookup.get(sym_a) or "ligand_receptor"
            all_edges.append((sym_a, sym_b, label, "ligand_receptor"))
            
    return all_edges

def map_edges_to_celltypes(edges: list, gene_to_celltypes: dict, celltype_to_broad: dict, celltype_to_ai: dict):
    rows = []
    for gene_a, gene_b, label, edge_type in edges:
        cts_a = gene_to_celltypes.get(gene_a, set())
        cts_b = gene_to_celltypes.get(gene_b, set())
        for ct_a in cts_a:
            for ct_b in cts_b:
                ct1, ct2 = sorted([ct_a, ct_b])
                if ct1 != ct2:
                    rows.append({
                        "gene_a": gene_a, "gene_b": gene_b, "label": label, "edge_type": edge_type,
                        "celltype_a": ct1, "celltype_b": ct2,
                        "broad_a": celltype_to_broad.get(ct1, "Unknown"), "broad_b": celltype_to_broad.get(ct2, "Unknown"),
                        "ai_a": celltype_to_ai.get(ct1, "Unknown"), "ai_b": celltype_to_ai.get(ct2, "Unknown"),
                    })
    return pd.DataFrame(rows)

def build_summary_tables(conn_df: pd.DataFrame):
    ct_pair_df = conn_df.groupby(["celltype_a", "celltype_b", "label", "edge_type"]).agg(n_edges=("gene_a", "count")).reset_index().sort_values("n_edges", ascending=False)
    group_pair_df = conn_df.groupby(["ai_a", "ai_b", "label", "edge_type"]).agg(n_edges=("gene_a", "count")).reset_index().rename(columns={"ai_a": "group_a", "ai_b": "group_b"}).sort_values("n_edges", ascending=False)
    
    combined = pd.concat([conn_df[["ai_a", "label"]].rename(columns={"ai_a": "group"}), conn_df[["ai_b", "label"]].rename(columns={"ai_b": "group"})])
    pw_group_matrix = combined.groupby(["label", "group"]).size().unstack(fill_value=0).sort_index()
    
    top_pw_df = group_pair_df.sort_values(["group_a", "group_b", "n_edges"], ascending=[True, True, False]).groupby(["group_a", "group_b"]).head(10).reset_index(drop=True)
    return ct_pair_df, group_pair_df, pw_group_matrix, top_pw_df

def build_pathway_source_table(conn_df: pd.DataFrame, reactome_pw_map: dict, out: Path):
    label_summary = conn_df.groupby(["label", "edge_type"]).agg(
        n_connection_rows=("gene_a", "count"),
        n_celltype_pairs=("celltype_a", lambda x: len(set(zip(x, conn_df.loc[x.index, "celltype_b"])))),
    ).reset_index().sort_values("n_connection_rows", ascending=False)
    
    label_summary["source_tier"] = ["LR" if r["edge_type"] == "ligand_receptor" else ("unnamed_pathway" if r["label"] == "pathway" else "named_pathway") for _, r in label_summary.iterrows()]
    label_summary.to_csv(out / "table_label_summary_with_source.csv", index=False)
    label_summary[label_summary["source_tier"] == "named_pathway"].to_csv(out / "table_named_pathways_overview.csv", index=False)

def _norm_value(raw: float, ct_a: str, ct_b: str, celltype_gene_count: dict) -> float:
    n_a = celltype_gene_count.get(ct_a, 0)
    n_b = celltype_gene_count.get(ct_b, 0)
    if n_a > 0 and n_b > 0:
        return raw / np.sqrt(n_a * n_b) * 1000
    return raw

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_heatmap_celltype_pairs_with_names(conn_df: pd.DataFrame, celltype_gene_count: dict, celltype_to_ai: dict, out: Path, top_n: int = 40):
    all_celltypes = sorted(set(conn_df["celltype_a"]) | set(conn_df["celltype_b"]))
    ct_by_group = defaultdict(list)
    for ct in all_celltypes:
        ct_by_group[CELLTYPE_TO_GROUP.get(ct, "Other")].append(ct)

    for group_name in sorted(ct_by_group.keys()):
        group_cts = ct_by_group[group_name]
        if not group_cts: continue

        group_conn = conn_df[(conn_df["celltype_a"].isin(group_cts)) | (conn_df["celltype_b"].isin(group_cts))]
        if group_conn.empty: continue

        pivot_data = group_conn.groupby(["celltype_a", "celltype_b"]).agg(n=("gene_a", "count")).reset_index()
        mat = pd.DataFrame(0.0, index=group_cts, columns=group_cts)
        for _, row in pivot_data.iterrows():
            ct_a, ct_b, n = row["celltype_a"], row["celltype_b"], row["n"]
            n_norm = _norm_value(n, ct_a, ct_b, celltype_gene_count)
            mat.loc[ct_a, ct_b] = n_norm
            mat.loc[ct_b, ct_a] = n_norm

        fig_h = max(4, len(group_cts) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_h * 1.1, fig_h))
        sns.heatmap(
            mat, ax=ax, cmap="YlOrRd", linewidths=0.5, linecolor="white",
            cbar_kws={"label": "edges / √(genes_A·genes_B) ×1k", "shrink": 0.8},
            annot=(len(group_cts) <= 12), fmt=".0f" if len(group_cts) <= 12 else "",
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_title(f"{group_name} — Cell-type connectivity (normalised)", fontsize=10, fontweight="bold")
        
        plt.tight_layout()
        safe_name = group_name.replace(" ", "_").replace("+", "plus").lower()
        plt.savefig(out / f"heatmap_celltype_{safe_name}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

def plot_heatmap_group_pairs_enhanced(conn_df: pd.DataFrame, celltype_gene_count: dict, out: Path):
    conn_df_copy = conn_df.copy()

    conn_df_copy["group_a"] = [CELLTYPE_TO_GROUP.get(x, "Other") for x in conn_df_copy["celltype_a"]]
    conn_df_copy["group_b"] = [CELLTYPE_TO_GROUP.get(x, "Other") for x in conn_df_copy["celltype_b"]]

    all_groups = sorted(set(conn_df_copy["group_a"]) | set(conn_df_copy["group_b"]))
    group_gene_count = defaultdict(int)
    for ct in (set(conn_df["celltype_a"]) | set(conn_df["celltype_b"])):
        group_gene_count[CELLTYPE_TO_GROUP.get(ct, "Other")] += celltype_gene_count.get(ct, 0)

    def _make_group_matrix(subset_df):
        if subset_df.empty:
            return pd.DataFrame(0.0, index=all_groups, columns=all_groups)
        subset_df = subset_df.copy()
        subset_df["group_a"] = [CELLTYPE_TO_GROUP.get(x, "Other") for x in subset_df["celltype_a"]]
        subset_df["group_b"] = [CELLTYPE_TO_GROUP.get(x, "Other") for x in subset_df["celltype_b"]]
        piv = subset_df.groupby(["group_a", "group_b"]).agg(n=("gene_a", "count")).reset_index()
        mat = pd.DataFrame(0.0, index=all_groups, columns=all_groups)
        for _, row in piv.iterrows():
            ga, gb, n = row["group_a"], row["group_b"], row["n"]
            n_norm = n / np.sqrt(group_gene_count.get(ga, 1) * group_gene_count.get(gb, 1)) * 1000
            mat.loc[ga, gb] += n_norm
            mat.loc[gb, ga] += n_norm
        # Extract array copy before fill_diagonal to avoid read-only error
        mat_arr = mat.values.copy()
        np.fill_diagonal(mat_arr, 0.0)
        mat = pd.DataFrame(mat_arr, index=mat.index, columns=mat.columns)
        return mat

    edge_types = conn_df["edge_type"].unique().tolist()
    n_panels = 1 + len(edge_types)
    fig_h = max(6, len(all_groups) * 0.6)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_h * n_panels * 0.9, fig_h))  # <-- FIX HERE
    if n_panels == 1:
        axes = [axes]
    cmaps = ["Blues", "Purples", "Greens"]

    sns.heatmap(
        _make_group_matrix(conn_df_copy), ax=axes[0], cmap=cmaps[0],
        annot=True, fmt=".1f", linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}
    )
    axes[0].set_title("All edges", fontsize=9, fontweight="bold")

    for i, et in enumerate(edge_types, start=1):
        sns.heatmap(
            _make_group_matrix(conn_df_copy[conn_df_copy["edge_type"] == et]),
            ax=axes[i], cmap=cmaps[i % len(cmaps)],
            annot=True, fmt=".1f", linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8}
        )
        axes[i].set_title(f"{et}", fontsize=9, fontweight="bold")

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(out / "heatmap_neuron_group_connections.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_heatmap_label_vs_group(pw_group_matrix: pd.DataFrame, out: Path, top_n_labels: int = 50):
    top_labels = pw_group_matrix.sum(axis=1).sort_values(ascending=False).head(top_n_labels).index
    mat = pw_group_matrix.loc[top_labels].copy()
    
    fig, ax = plt.subplots(figsize=(max(10, len(mat.columns) * 0.55), max(8, len(mat) * 0.22)))
    sns.heatmap(mat, ax=ax, cmap="viridis", linewidths=0.15, linecolor="white", cbar_kws={"label": "Edge count"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Top {top_n_labels} edge labels × category groups", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "heatmap_label_vs_category_group.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_top_labels_overall(conn_df: pd.DataFrame, out: Path, top_n: int = 30):
    agg = conn_df.groupby(["label", "edge_type"]).agg(n_edges=("gene_a", "count"), n_ct_pairs=("celltype_a", lambda x: len(set(zip(x, conn_df.loc[x.index, "celltype_b"]))))).reset_index()
    palette = {"pathway": "#1982c4", "ligand_receptor": "#ff595e"}
    
    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, top_n * 0.32)))
    for ax, df_top, col, xlabel in [(axes[0], agg.sort_values("n_edges", ascending=False).head(top_n), "n_edges", "Total edges"), (axes[1], agg.sort_values("n_ct_pairs", ascending=False).head(top_n), "n_ct_pairs", "Unique cell-type pairs")]:
        colors = [palette.get(et, "#888") for et in df_top["edge_type"]]
        ax.barh(df_top["label"], df_top[col], color=colors, edgecolor="white", linewidth=0.4)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_yticklabels(df_top["label"], fontsize=7)
    
    axes[0].legend(handles=[mpatches.Patch(color=c, label=et) for et, c in palette.items()], fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "bar_top_labels_overall.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_top_labels_per_group_pair(top_pw_df: pd.DataFrame, out: Path, max_pairs: int = 12):
    pairs = top_pw_df.groupby(["group_a", "group_b"])["n_edges"].sum().sort_values(ascending=False).head(max_pairs).index.tolist()
    n_cols = 3
    n_rows = int(np.ceil(len(pairs) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axes = np.array(axes).flatten()
    palette = {"pathway": "#1982c4", "ligand_receptor": "#ff595e"}
    
    for i, (ga, gb) in enumerate(pairs):
        sub = top_pw_df[(top_pw_df["group_a"] == ga) & (top_pw_df["group_b"] == gb)].head(8)
        axes[i].barh(sub["label"], sub["n_edges"], color=[palette.get(et, "#888") for et in sub["edge_type"]], edgecolor="white")
        axes[i].invert_yaxis()
        axes[i].set_title(f"{ga}\n↔ {gb}", fontsize=7.5, fontweight="bold")
        
    for j in range(len(pairs), len(axes)): axes[j].set_visible(False)
    fig.legend(handles=[mpatches.Patch(color=c, label=et) for et, c in palette.items()], loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out / "bar_top_labels_per_group_pair.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_connections_per_group(conn_df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, conn_df["ai_a"].nunique() * 0.45 + 3)))
    for ax, et, title_suffix in [(axes[0], "pathway", "pathway edges"), (axes[1], "ligand_receptor", "LR edges")]:
        sub = conn_df[conn_df["edge_type"] == et]
        if sub.empty:
            ax.set_visible(False)
            continue

        df_a = sub[["ai_a", "ai_b"]].rename(columns={"ai_a": "group", "ai_b": "partner"})
        df_b = sub[["ai_a", "ai_b"]].rename(columns={"ai_b": "group", "ai_a": "partner"})
        combined = pd.concat([df_a, df_b])
        combined = combined[combined["group"] != combined["partner"]]

        if combined.empty:
            ax.set_visible(False)
            continue

        counts = combined.groupby(["group", "partner"]).size().reset_index(name="n")
        pivot = counts.pivot(index="group", columns="partner", values="n").fillna(0)

        # Ensure all values are numeric
        pivot = pivot.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

        if pivot.empty or pivot.shape[0] == 0:
            ax.set_visible(False)
            continue

        # Sort rows by total count descending
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_title(title_suffix.capitalize(), fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(out / "bar_connections_per_category_group.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_network_celltype_with_groups(conn_df: pd.DataFrame, out: Path, edge_type_filter: str | None = None, max_nodes: int = 80, min_edge_weight: int = 1):
    sub = conn_df[conn_df["edge_type"] == edge_type_filter] if edge_type_filter else conn_df
    if sub.empty: return
    
    agg = sub.groupby(["celltype_a", "celltype_b"]).agg(weight=("gene_a", "count")).reset_index()
    G = nx.Graph()
    for _, row in agg.iterrows():
        a, b, w = row["celltype_a"], row["celltype_b"], int(row["weight"])
        if a != b:
            if G.has_edge(a, b): G[a][b]["weight"] += w
            else: G.add_edge(a, b, weight=w)
            
    if G.number_of_nodes() > max_nodes:
        strength = {n: sum(d["weight"] for _,_, d in G.edges(n, data=True)) for n in G.nodes()}
        G = G.subgraph(sorted(strength, key=strength.get, reverse=True)[:max_nodes]).copy()
        
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 1) < min_edge_weight])
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0: return

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [0.5 + 3 * (w / max_w) for w in weights]

    node_groups = [CELLTYPE_TO_GROUP.get(n, "Other") for n in G.nodes()]
    node_colors = [GROUP_COLORS.get(g, "#cccccc") for g in node_groups]
    node_sizes = [300 + 2000 * (sum(d["weight"] for _,_, d in G.edges(n, data=True)) / (max_w * max(G.degree(n), 1))) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(18, 14))
    pos = nx.spring_layout(G, weight="weight", seed=42, k=2.5, iterations=50)
    
    for (u, v), w in zip(G.edges(), widths):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, width=w, alpha=0.5, edge_color="#555555")
        
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.8, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: (n if len(n) <= 30 else n[:28] + "…") for n in G.nodes()}, font_size=6, font_weight="bold")
    
    ax.legend(handles=[mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in sorted(set(node_groups))], loc="upper left", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / f"network_celltype_by_group_{edge_type_filter or 'all'}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_network_group_with_labels(conn_df: pd.DataFrame, out: Path, edge_type_filter: str | None = None):
    sub = conn_df[conn_df["edge_type"] == edge_type_filter] if edge_type_filter else conn_df.copy()
    if sub.empty: return
    
    sub["group_a"] = [CELLTYPE_TO_GROUP.get(x, "Other") for x in sub["celltype_a"]]
    sub["group_b"] = [CELLTYPE_TO_GROUP.get(x, "Other") for x in sub["celltype_b"]]
    
    agg = sub.groupby(["group_a", "group_b"]).agg(weight=("gene_a", "count")).reset_index()
    
    G = nx.Graph()
    for _, row in agg.iterrows():
        a, b, w = row["group_a"], row["group_b"], int(row["weight"])
        if a != b:
            if G.has_edge(a, b): G[a][b]["weight"] += w
            else: G.add_edge(a, b, weight=w)
            
    if G.number_of_nodes() == 0: return

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    
    fig, ax = plt.subplots(figsize=(14, 12))
    pos = nx.spring_layout(G, weight="weight", seed=42, k=3, iterations=100)
    
    for (u, v), w in zip(G.edges(), [1 + 5 * (w / max_w) for w in weights]):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, width=w, alpha=0.6, edge_color="#333333")
        
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[1500 + 3000 * (sum(d["weight"] for _,_, d in G.edges(n, data=True)) / (max_w * max(G.degree(n), 1))) for n in G.nodes()], node_color=[GROUP_COLORS.get(n, "#cccccc") for n in G.nodes()], alpha=0.85, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / f"network_neuron_group_{edge_type_filter or 'all'}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Save Tables
# ---------------------------------------------------------------------------
def save_tables(ct_pair_df: pd.DataFrame, group_pair_df: pd.DataFrame, pw_group_matrix: pd.DataFrame, top_pw_df: pd.DataFrame, conn_df: pd.DataFrame, out: Path):
    ct_pair_df.to_csv(out / "table_celltype_pair_labels.csv", index=False)
    group_pair_df.to_csv(out / "table_group_pair_labels.csv", index=False)
    pw_group_matrix.to_csv(out / "table_label_group_matrix.csv")
    top_pw_df.to_csv(out / "table_top_labels_per_group_pair.csv", index=False)
    
    conn_df.groupby(["label", "edge_type"]).agg(n_edges=("gene_a", "count")).reset_index().sort_values("n_edges", ascending=False).to_csv(out / "table_label_summary.csv", index=False)

def save_celltype_group_tables(conn_df: pd.DataFrame, celltype_gene_count: dict, out: Path):
    conn_df_copy = conn_df.copy()
    conn_df_copy["group_a"] = [CELLTYPE_TO_GROUP.get(x, "Other") for x in conn_df_copy["celltype_a"]]
    conn_df_copy["group_b"] = [CELLTYPE_TO_GROUP.get(x, "Other") for x in conn_df_copy["celltype_b"]]

    pd.DataFrame([{"cell_type": ct, "group": CELLTYPE_TO_GROUP.get(ct, "Other"), "n_genes": celltype_gene_count.get(ct, 0)} for ct in sorted(set(conn_df["celltype_a"]) | set(conn_df["celltype_b"]))]).to_csv(out / "table_celltype_group_mapping.csv", index=False)
    conn_df_copy.groupby(["celltype_a", "celltype_b", "group_a", "group_b", "edge_type"]).agg(n_edges=("gene_a", "count")).reset_index().sort_values("n_edges", ascending=False).to_csv(out / "table_celltype_pair_connectivity.csv", index=False)
    conn_df_copy.groupby(["group_a", "group_b", "edge_type"]).agg(n_edges=("gene_a", "count")).reset_index().sort_values("n_edges", ascending=False).to_csv(out / "table_neuron_group_connectivity.csv", index=False)

# ---------------------------------------------------------------------------
# NEW: Gene Overlap Analysis
# ---------------------------------------------------------------------------
def analyze_gene_overlap_across_layers(multiplex_dir: Path, categories_df: pd.DataFrame) -> dict:
    """Analyze which genes appear in multiple cell types (layers)"""
    celltype_genes = {}
    
    for _, row in categories_df.iterrows():
        cell_type = row[COL_CELL_TYPE]
        genes = load_layer_genes(multiplex_dir / f"{stem_from_cell_type(cell_type)}.tsv")
        celltype_genes[cell_type] = genes
    
    # Gene to cell types mapping
    gene_celltype_map = defaultdict(set)
    for ct, genes in celltype_genes.items():
        for g in genes:
            gene_celltype_map[g].add(ct)
    
    # Genes shared across multiple cell types
    multi_celltype_genes = {g: cts for g, cts in gene_celltype_map.items() if len(cts) > 1}
    single_celltype_genes = {g: cts for g, cts in gene_celltype_map.items() if len(cts) == 1}
    
    # Summary statistics
    overlap_counts = defaultdict(int)
    for g, cts in multi_celltype_genes.items():
        overlap_counts[len(cts)] += 1
    
    summary = {
        "total_unique_genes": len(gene_celltype_map),
        "genes_in_single_celltype": len(single_celltype_genes),
        "genes_in_multiple_celltypes": len(multi_celltype_genes),
        "overlap_distribution": dict(overlap_counts),
        "gene_celltype_map": gene_celltype_map,
        "multi_celltype_genes": multi_celltype_genes,
        "celltype_genes": celltype_genes,
    }
    
    return summary

def analyze_gene_overlap_across_groups(gene_overlap_summary: dict, celltype_to_group: dict) -> dict:
    """Analyze gene overlap at the neuron group level"""
    group_genes = defaultdict(set)
    
    for gene, celltypes in gene_overlap_summary["gene_celltype_map"].items():
        groups = {celltype_to_group.get(ct, "Other") for ct in celltypes}
        for g in groups:
            group_genes[g].add(gene)
    
    # Genes shared across multiple groups
    gene_group_map = defaultdict(set)
    for gene, celltypes in gene_overlap_summary["gene_celltype_map"].items():
        groups = {celltype_to_group.get(ct, "Other") for ct in celltypes}
        for g in groups:
            gene_group_map[gene].add(g)
    
    multi_group_genes = {g: grps for g, grps in gene_group_map.items() if len(grps) > 1}
    single_group_genes = {g: grps for g, grps in gene_group_map.items() if len(grps) == 1}
    
    group_overlap_counts = defaultdict(int)
    for g, grps in multi_group_genes.items():
        group_overlap_counts[len(grps)] += 1
    
    summary = {
        "total_unique_genes": len(gene_group_map),
        "genes_in_single_group": len(single_group_genes),
        "genes_in_multiple_groups": len(multi_group_genes),
        "group_overlap_distribution": dict(group_overlap_counts),
        "gene_group_map": gene_group_map,
        "multi_group_genes": multi_group_genes,
        "group_genes": dict(group_genes),
    }
    
    return summary

def create_gene_overlap_tables(gene_overlap_summary: dict, group_overlap_summary: dict, celltype_to_group: dict, out: Path):
    """Create detailed tables of gene overlaps"""
    
    # Table 1: Genes appearing in multiple cell types
    multi_ct_rows = []
    for gene, celltypes in gene_overlap_summary["multi_celltype_genes"].items():
        groups = [celltype_to_group.get(ct, "Other") for ct in celltypes]
        multi_ct_rows.append({
            "gene": gene,
            "n_celltypes": len(celltypes),
            "celltypes": "|".join(sorted(celltypes)),
            "groups": "|".join(sorted(set(groups))),
            "n_groups": len(set(groups)),
        })
    
    multi_ct_df = pd.DataFrame(multi_ct_rows).sort_values("n_celltypes", ascending=False)
    multi_ct_df.to_csv(out / "table_genes_shared_across_celltypes.csv", index=False)
    
    # Table 2: Genes appearing in multiple groups
    multi_group_rows = []
    for gene, groups in group_overlap_summary["multi_group_genes"].items():
        celltypes = gene_overlap_summary["gene_celltype_map"][gene]
        multi_group_rows.append({
            "gene": gene,
            "n_groups": len(groups),
            "groups": "|".join(sorted(groups)),
            "n_celltypes": len(celltypes),
            "celltypes": "|".join(sorted(celltypes)),
        })
    
    multi_group_df = pd.DataFrame(multi_group_rows).sort_values("n_groups", ascending=False)
    multi_group_df.to_csv(out / "table_genes_shared_across_groups.csv", index=False)
    
    # Table 3: Summary statistics
    summary_stats = pd.DataFrame([
        {
            "metric": "Total unique genes",
            "celltype_level": gene_overlap_summary["total_unique_genes"],
            "group_level": group_overlap_summary["total_unique_genes"],
        },
        {
            "metric": "Genes in single unit",
            "celltype_level": gene_overlap_summary["genes_in_single_celltype"],
            "group_level": group_overlap_summary["genes_in_single_group"],
        },
        {
            "metric": "Genes in multiple units",
            "celltype_level": gene_overlap_summary["genes_in_multiple_celltypes"],
            "group_level": group_overlap_summary["genes_in_multiple_groups"],
        },
        {
            "metric": "% genes in multiple units",
            "celltype_level": f"{100 * gene_overlap_summary['genes_in_multiple_celltypes'] / gene_overlap_summary['total_unique_genes']:.1f}%",
            "group_level": f"{100 * group_overlap_summary['genes_in_multiple_groups'] / group_overlap_summary['total_unique_genes']:.1f}%",
        },
    ])
    summary_stats.to_csv(out / "table_gene_overlap_summary_statistics.csv", index=False)
    
    return multi_ct_df, multi_group_df, summary_stats

def plot_gene_overlap_distribution(gene_overlap_summary: dict, group_overlap_summary: dict, out: Path):
    """Visualize distribution of gene sharing"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cell-type overlap distribution
    ct_dist = gene_overlap_summary["overlap_distribution"]
    axes[0, 0].bar(ct_dist.keys(), ct_dist.values(), color="#1f77b4", edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Number of cell types sharing gene", fontsize=10)
    axes[0, 0].set_ylabel("Count", fontsize=10)
    axes[0, 0].set_title("Gene distribution across cell types", fontsize=11, fontweight="bold")
    axes[0, 0].grid(axis="y", alpha=0.3)
    
    # Plot 2: Group overlap distribution
    grp_dist = group_overlap_summary["group_overlap_distribution"]
    axes[0, 1].bar(grp_dist.keys(), grp_dist.values(), color="#ff7f0e", edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Number of groups sharing gene", fontsize=10)
    axes[0, 1].set_ylabel("Count", fontsize=10)
    axes[0, 1].set_title("Gene distribution across groups", fontsize=11, fontweight="bold")
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    # Plot 3: Pie chart - cell type level
    ct_labels = ["Single cell type", "Multiple cell types"]
    ct_values = [gene_overlap_summary["genes_in_single_celltype"], gene_overlap_summary["genes_in_multiple_celltypes"]]
    colors_pie = ["#d62728", "#2ca02c"]
    axes[1, 0].pie(ct_values, labels=ct_labels, autopct="%1.1f%%", colors=colors_pie, startangle=90)
    axes[1, 0].set_title("Gene categorization (cell type level)", fontsize=11, fontweight="bold")
    
    # Plot 4: Pie chart - group level
    grp_labels = ["Single group", "Multiple groups"]
    grp_values = [group_overlap_summary["genes_in_single_group"], group_overlap_summary["genes_in_multiple_groups"]]
    colors_pie2 = ["#9467bd", "#8c564b"]
    axes[1, 1].pie(grp_values, labels=grp_labels, autopct="%1.1f%%", colors=colors_pie2, startangle=90)
    axes[1, 1].set_title("Gene categorization (group level)", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out / "plot_gene_overlap_distribution.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_gene_sharing_heatmap(gene_overlap_summary: dict, celltype_to_group: dict, out: Path):
    """Create heatmap showing gene sharing patterns between cell types"""
    celltype_genes = gene_overlap_summary["celltype_genes"]
    celltypes = sorted(celltype_genes.keys())
    
    # Create binary matrix
    gene_sharing = pd.DataFrame(0, index=celltypes, columns=celltypes)
    
    for i, ct1 in enumerate(celltypes):
        for j, ct2 in enumerate(celltypes):
            if i != j:
                genes1 = celltype_genes[ct1]
                genes2 = celltype_genes[ct2]
                shared = len(genes1 & genes2)
                gene_sharing.iloc[i, j] = shared
    
    # Group cell types by group
    ct_by_group = defaultdict(list)
    for ct in celltypes:
        ct_by_group[celltype_to_group.get(ct, "Other")].append(ct)
    
    # Create separate heatmaps per group
    for group_name in sorted(ct_by_group.keys()):
        group_cts = ct_by_group[group_name]
        if len(group_cts) <= 1:
            continue
        
        mat = gene_sharing.loc[group_cts, group_cts]
        
        fig, ax = plt.subplots(figsize=(max(6, len(group_cts) * 0.7), max(6, len(group_cts) * 0.7)))
        sns.heatmap(mat, ax=ax, cmap="YlOrRd", annot=True, fmt="d", linewidths=0.5, 
                    cbar_kws={"label": "Shared genes"})
        ax.set_title(f"{group_name} — Shared genes between cell types", fontsize=11, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        plt.tight_layout()
        safe_name = group_name.replace(" ", "_").replace("+", "plus").lower()
        plt.savefig(out / f"heatmap_gene_sharing_{safe_name}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

def plot_group_gene_overlap(group_overlap_summary: dict, out: Path):
    """Create heatmap of gene sharing between groups"""
    group_genes = group_overlap_summary["group_genes"]
    groups = sorted(group_genes.keys())
    
    # Create binary matrix
    group_sharing = pd.DataFrame(0, index=groups, columns=groups)
    
    for g1 in groups:
        for g2 in groups:
            if g1 != g2:
                genes1 = group_genes[g1]
                genes2 = group_genes[g2]
                shared = len(genes1 & genes2)
                group_sharing.loc[g1, g2] = shared
    
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(group_sharing, ax=ax, cmap="Blues", annot=True, fmt="d", linewidths=1, 
                cbar_kws={"label": "Shared genes"})
    ax.set_title("Shared genes between neuron groups", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out / "heatmap_group_gene_sharing.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def print_gene_overlap_report(gene_overlap_summary: dict, group_overlap_summary: dict, out: Path):
    """Print detailed report of gene overlap analysis"""
    _print_block("GENE OVERLAP ANALYSIS REPORT")
    
    print(f"\n📊 CELL-TYPE LEVEL:")
    print(f"  • Total unique genes: {gene_overlap_summary['total_unique_genes']}")
    print(f"  • Genes in single cell type: {gene_overlap_summary['genes_in_single_celltype']}")
    print(f"  • Genes in multiple cell types: {gene_overlap_summary['genes_in_multiple_celltypes']}")
    print(f"  • Percentage in multiple: {100 * gene_overlap_summary['genes_in_multiple_celltypes'] / gene_overlap_summary['total_unique_genes']:.1f}%")
    
    print(f"\n📊 GROUP LEVEL:")
    print(f"  • Total unique genes: {group_overlap_summary['total_unique_genes']}")
    print(f"  • Genes in single group: {group_overlap_summary['genes_in_single_group']}")
    print(f"  • Genes in multiple groups: {group_overlap_summary['genes_in_multiple_groups']}")
    print(f"  • Percentage in multiple: {100 * group_overlap_summary['genes_in_multiple_groups'] / group_overlap_summary['total_unique_genes']:.1f}%")
    
    print(f"\n📈 OVERLAP DISTRIBUTION (cell types):")
    for n, count in sorted(gene_overlap_summary["overlap_distribution"].items()):
        print(f"  • Shared by {n} cell types: {count} genes")
    
    print(f"\n📈 OVERLAP DISTRIBUTION (groups):")
    for n, count in sorted(group_overlap_summary["group_overlap_distribution"].items()):
        print(f"  • Shared by {n} groups: {count} genes")
    
    # Top shared genes
    print(f"\n🔝 TOP 10 GENES SHARED ACROSS CELL TYPES:")
    top_genes = sorted(gene_overlap_summary["multi_celltype_genes"].items(), 
                      key=lambda x: len(x[1]), reverse=True)[:10]
    for gene, celltypes in top_genes:
        print(f"  • {gene}: {len(celltypes)} cell types")
    
    print(f"\n🔝 TOP 10 GENES SHARED ACROSS GROUPS:")
    top_group_genes = sorted(group_overlap_summary["multi_group_genes"].items(), 
                            key=lambda x: len(x[1]), reverse=True)[:10]
    for gene, groups in top_group_genes:
        print(f"  • {gene}: {len(groups)} groups")

# ---------------------------------------------------------------------------
# Main (updated)
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse pathway/LR connectivity with neuron group organization")
    parser.add_argument("--wdir", required=True, help="MultiXrank working directory")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--lr-csv", default=None)
    parser.add_argument("--reactome-csv", default=None)
    parser.add_argument("--uniprot-symbol-csv", default=None)
    parser.add_argument("--pathway-stems", nargs="+", default=["pathway_layer"])
    parser.add_argument("--lr-stems", nargs="+", default=["ligand_receptor_layer"])
    parser.add_argument("--output-dir", default="pathway_connectivity_output")
    parser.add_argument("--top-n-labels", type=int, default=20)
    parser.add_argument("--min-edge-weight", type=int, default=2)
    parser.add_argument("--exclude-ai", nargs="+", default=[])
    parser.add_argument("--lr-pathways-csv", default=None)
    args = parser.parse_args()
    
    wdir = Path(args.wdir).resolve()
    out = Path(args.output_dir)
    multiplex_dir = wdir / "multiplex" / "gene_multiplex"
    out.mkdir(parents=True, exist_ok=True)
    
    csv_path = Path(args.csv) if args.csv else SCRIPT_DIR / "layers_categories.csv"
    lr_csv_path = Path(args.lr_csv) if args.lr_csv else SCRIPT_DIR / "human_lr_pairs.csv"
    reactome_csv = Path(args.reactome_csv) if args.reactome_csv else SCRIPT_DIR / "reactome_interactions_with_pathways_v2.csv"
    uniprot_sym_csv = Path(args.uniprot_symbol_csv) if args.uniprot_symbol_csv else SCRIPT_DIR / "d_uniprot_to_symbol_20251120.csv"
    lr_pathways_csv = Path(args.lr_pathways_csv) if args.lr_pathways_csv else SCRIPT_DIR / "lr_pairs_with_pathways.csv"
    
    _print_block("Loading input files")
    categories_df = load_categories(csv_path, exclude_ai=args.exclude_ai or None)
    receptor_lookup = load_lr_csv(lr_csv_path)
    uniprot_symbol_map = load_uniprot_symbol_map(uniprot_sym_csv)

    reactome_pw_map = load_reactome_pathway_map(reactome_csv, uniprot_symbol_map)
    lr_pw_map = load_lr_pathway_map(lr_pathways_csv)
    combined_pw_map = merge_pathway_maps(reactome_pw_map, lr_pw_map)

    gene_to_celltypes, celltype_to_broad, celltype_to_ai, celltype_gene_count = build_gene_celltype_index(multiplex_dir, categories_df)

    all_edges = load_all_edges(multiplex_dir, args.pathway_stems, args.lr_stems, combined_pw_map, receptor_lookup)
    if not all_edges: 
        return print("\n⚠  No edges loaded.")

    conn_df = map_edges_to_celltypes(all_edges, gene_to_celltypes, celltype_to_broad, celltype_to_ai)
    if conn_df.empty: 
        return print("\n⚠  No connections found.")
    
    ct_pair_df, group_pair_df, pw_group_matrix, top_pw_df = build_summary_tables(conn_df)
    build_pathway_source_table(conn_df, reactome_pw_map, out)
    
    # ===== NEW: GENE OVERLAP ANALYSIS =====
    _print_block("Analyzing gene overlap across layers")
    gene_overlap_summary = analyze_gene_overlap_across_layers(multiplex_dir, categories_df)
    group_overlap_summary = analyze_gene_overlap_across_groups(gene_overlap_summary, CELLTYPE_TO_GROUP)
    print_gene_overlap_report(gene_overlap_summary, group_overlap_summary, out)
    
    _print_block("Creating gene overlap tables")
    multi_ct_df, multi_group_df, summary_stats = create_gene_overlap_tables(
        gene_overlap_summary, group_overlap_summary, CELLTYPE_TO_GROUP, out
    )
    
    _print_block("Generating gene overlap visualizations")
    plot_gene_overlap_distribution(gene_overlap_summary, group_overlap_summary, out)
    plot_gene_sharing_heatmap(gene_overlap_summary, CELLTYPE_TO_GROUP, out)
    plot_group_gene_overlap(group_overlap_summary, out)
    # ===== END: GENE OVERLAP ANALYSIS =====
    
    _print_block("Generating heatmaps (with cell names & groups)")
    plot_heatmap_celltype_pairs_with_names(conn_df, celltype_gene_count, celltype_to_ai, out)
    plot_heatmap_group_pairs_enhanced(conn_df, celltype_gene_count, out)
    plot_heatmap_label_vs_group(pw_group_matrix, out, top_n_labels=args.top_n_labels)
    
    _print_block("Generating bar charts")
    plot_top_labels_overall(conn_df, out, top_n=args.top_n_labels)
    plot_top_labels_per_group_pair(top_pw_df, out)
    plot_connections_per_group(conn_df, out)
    
    _print_block("Generating networks (colored by group)")
    for et in [None, "pathway", "ligand_receptor"]:
        plot_network_celltype_with_groups(conn_df, out, et, max_nodes=80)
        plot_network_group_with_labels(conn_df, out, et)
    
    _print_block("Saving tables")
    save_tables(ct_pair_df, group_pair_df, pw_group_matrix, top_pw_df, conn_df, out)
    save_celltype_group_tables(conn_df, celltype_gene_count, out)
    
    print(f"\n✅ DONE! Outputs saved to: {out.resolve()}")

if __name__ == "__main__":
    main()