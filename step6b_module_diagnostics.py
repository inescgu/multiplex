#!/usr/bin/env python3
"""
Step 6b (replaces step9b_module_diagnostics.py): exploratory diagnostics for
a disease module built by step6_build_disease_module.py.

What changed vs. the old version
-----------------------------------
The old version read layer_info.json purely to turn a layer STEM (e.g.
"A-LTMR.TAC3_ppi_layer") into a human-readable cell-type name via
layer_display_name(). In the new heterogeneous structure, the 'cell_types'
edge attribute on module_graph.pkl already holds the plain cell-type name
directly (that's just what step 3 wrote it as) — there's no stem to
translate, so layer_info.json and the whole indirection are gone. Every
"layer_stem" column from the old CSVs is just "cell_type" now.

TWO BUGS FIXED (previous heterogeneous-structure port) -- see git history
for full detail: (1) a case-normalization mismatch between seeds.txt
loading and module_graph.pkl's node casing, and (2) edge-attribute-based
cell-type extraction undercounting isolated (degree-0) nodes. Both fixed by
using `_norm_node()` consistently and reading cell type/degree/seed status
straight from disease_module.csv rather than by walking edges.

Interaction-type histograms (source/method + NeuronChat ligand_type)
---------------------------------------------------------------------------
Reads scored_lr_edge_provenance.csv (written by step2 when run with
--scored-lr-csv) and reports, for both the full candidate pool of
scored-LR edges and just the ones present in the final module, how many
came from each LR-inference method and (for NeuronChat rows) each
ligand_type category. Skipped with a one-line note if the file isn't found.

NEW: seed connectivity across ALL layers, and expression specificity
---------------------------------------------------------------------------
Two exploratory analyses folded in from a separate draft script
(step9_seed_gene_report.py), merged here rather than kept standalone
because both are fundamentally about the same seeds this script already
loads, and this script is already the natural home for "what happened to
my seeds" questions:

  1. SEED CONNECTIVITY ACROSS ALL LAYERS. Everything above only looks at
     seeds *within the final module* -- present/isolated/duplicated, etc.
     It never asks how connected a seed gene was in each of the ORIGINAL
     cell-type PPI layers, independent of whether that layer's edges ended
     up in the module at all. This section answers that: for every unique
     seed gene, how many layers is it a node in, its summed degree across
     them, and which single layer it's most connected in (its "dominant"
     layer) -- useful for spotting seeds that are well-connected somewhere
     but got dropped/isolated by the module-building step, or seeds that
     were never well-connected anywhere to begin with. Needs
     celltype_gene_sets.json (step 1) plus the raw multiplex/<ct>/
     ppi_layer.tsv files, both already sitting in --input-dir (the same
     MultiXrank wdir you already pass for the seeds.txt / provenance
     lookups above) -- no new required input.

  2. EXPRESSION SPECIFICITY. Cross-references seed genes (and, separately,
     the top-N highest-degree NON-seed genes the module actually recruited)
     against a cell_type_markers.csv (from extract_expression_matrix.R's
     FindMarkers loop: p_val, avg_log2FC, pct.1, pct.2, p_val_adj, gene,
     cell_type). Reports how many cell types call each gene a significant
     marker, its best avg_log2FC and which cell type that's in, and its
     percentile rank within that cell type's marker list (0 = most
     specific). A gene that's a marker in many cell types, or ranks low
     even in its best cell type, is a hint it isn't actually cell-type
     specific -- worth a second look if it's also high-degree in your
     module, since that combination (central but non-specific) is exactly
     the profile of a generic PPI hub riding along rather than a real
     signal. Entirely optional -- only runs if --markers-csv is given.

Reads:
  <output-dir>/module_graph.pkl              (networkx graph -- used for edge counts)
  <output-dir>/disease_module.csv            (per-node table incl. node, gene, cell_type,
                                               is_seed, degree_in_module -- the authoritative
                                               source for membership/annotation)
  <input-dir>/seeds/seeds.txt                (original seed list, for duplicate detection)
  <input-dir>/scored_lr_edge_provenance.csv  (optional; written by step2 when
                                               --scored-lr-csv was used -- source/method +
                                               NeuronChat ligand_type per scored-LR edge)
  <input-dir>/celltype_gene_sets.json        (optional, or pass --celltype-gene-sets; step 1's
                                               output -- needed for the seed-layer-connectivity
                                               section)
  <input-dir>/multiplex/<ct>/ppi_layer.tsv   (optional, read alongside the gene sets above; also
                                               tries the *.tsv.orig backup if step 2.5 already
                                               namespaced the working copy)
  --markers-csv                              (optional; cell_type_markers.csv, enables the
                                               expression-specificity section)

Writes (to --out-prefix, default "module_exploratory"):
  <prefix>_per_celltype.csv
  <prefix>_seed_membership.csv
  <prefix>_seed_layer_connectivity.csv          NEW
  <prefix>_seed_expression_specificity.csv      NEW (only if --markers-csv given)
  <prefix>_module_top_gene_expression_specificity.csv   NEW (only if --markers-csv given)
  <prefix>_specificity_vs_centrality.png        NEW (only if --markers-csv given)
  <prefix>_summary.txt
  <prefix>_seed_pct_by_celltype.png
  <prefix>_module_pct_by_celltype.png
  <prefix>_top_seed_genes.png
  <prefix>_candidate_lr_by_source.csv/.png
  <prefix>_candidate_lr_by_ligand_type.csv/.png
  <prefix>_module_lr_by_source.csv/.png
  <prefix>_module_lr_by_ligand_type.csv/.png
  (--no-plots skips all PNGs; the underlying CSVs are unaffected)

Usage:
  python step6b_module_diagnostics.py \
      --output-dir 06_disease_module/disease_module_output \
      --input-dir  04_multixrank_prep/multixrank_hetero_input \
      --markers-csv ../R/convert_hdrg_human_2025/cell_type_markers.csv
"""
import argparse
import json
import pickle
from pathlib import Path
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Major-category classification (from the DRG neuron hierarchy used
# elsewhere in the pipeline) -- used to color the per-cell-type bar charts.
# ---------------------------------------------------------------------------
DRG_HIERARCHY = {
    "A-fiber": {"color": "#1f77b4", "subcategories": {
        "Proprioceptors": {"marker": "o", "cell_types": [
            "A-Propr.EPHA3", "A-Propr.HAPLN4", "A-Propr.PCDH8",
        ]},
        "A-beta LTMR": {"marker": "s", "cell_types": [
            "Ab-LTMR.CCKAR", "Ab-LTMR.ETV1", "Ab-LTMR.LGI2", "Ab-LTMR.NSG2",
        ]},
        "A-LTMR": {"marker": "^", "cell_types": ["A-LTMR.TAC3"]},
        "A-peptidergic": {"marker": "D", "cell_types": [
            "A-PEP.CHRNA7_SLC18A3", "A-PEP.KIT", "A-PEP.NTRK3_S100A16", "A-PEP.SCGN_ADRA2C",
        ]},
    }},
    "C-fiber": {"color": "#ff7f0e", "subcategories": {
        "C-LTMR": {"marker": "o", "cell_types": ["C-LTMR.CDH9"]},
        "C-peptidergic": {"marker": "D", "cell_types": [
            "C-PEP.ADORA2B", "C-PEP.TAC1_CACNG5", "C-PEP.TAC1_CHRNA3",
        ]},
        "MRGPRX nociceptors": {"marker": "s", "cell_types": [
            "C-NP.MRGPRX1_GFRA2", "C-NP.MRGPRX1_MRGPRX4",
        ]},
        "C-thermosensitive": {"marker": "*", "cell_types": [
            "C-Thermo.RXFP1", "C-Thermo.TRPM8",
        ]},
        "C-SST nociceptor": {"marker": "v", "cell_types": ["C-NP.SST"]},
    }},
    "Non-classified": {"color": "#d62728", "subcategories": {
        "ATF3 (injured)": {"marker": "x", "cell_types": ["ATF3"]},
    }},
    "Non-neuronal": {"color": "#2ca02c", "subcategories": {
        "Immune": {"marker": "o", "cell_types": [
            "B cell", "CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell",
            "activated CD4-positive, alpha-beta T cell", "activated CD8-positive, alpha-beta T cell",
            "basophil", "classical monocyte", "gamma-delta T cell", "granulocyte",
            "innate lymphoid cell", "intermediate monocyte", "leukocyte", "macrophage",
            "mast cell", "mature NK T cell", "myeloid cell", "myeloid dendritic cell",
            "myeloid leukocyte", "naive thymus-derived CD4-positive, alpha-beta T cell",
            "naive thymus-derived CD8-positive, alpha-beta T cell", "natural killer cell",
            "neutrophil", "non-classical monocyte", "plasma cell",
            "plasmacytoid dendritic cell", "regulatory T cell",
        ]},
        "Glial/Nervous": {"marker": "s", "cell_types": [
            "Schwann cell", "enteroglial cell", "microglial cell", "neuron",
        ]},
        "Immune/Hematopoietic": {"marker": "^", "cell_types": [
            "Langerhans cell", "colon macrophage", "common myeloid progenitor",
            "erythroid progenitor cell", "hematopoietic cell", "hematopoietic precursor cell",
            "hematopoietic stem cell", "tissue-resident macrophage",
        ]},
    }},
}

_CELL_TYPE_TO_MAJOR = {}
for _major, _mdata in DRG_HIERARCHY.items():
    for _subcat, _sdata in _mdata["subcategories"].items():
        for _ct in _sdata["cell_types"]:
            _CELL_TYPE_TO_MAJOR[_ct.strip().lower()] = _major


def get_category_from_cell_type(cell_type: str) -> str:
    if not isinstance(cell_type, str):
        return "unknown"
    return _CELL_TYPE_TO_MAJOR.get(cell_type.strip().lower(), "unknown")


def _major_category_palette() -> dict:
    return {"A-fiber": "#1f77b4", "C-fiber": "#ff7f0e", "Non-classified": "#d62728",
            "Non-neuronal": "#2ca02c", "unknown": "#999999"}


def _norm_node(s: str) -> str:
    s = s.strip()
    if "::" in s:
        ct, gene = s.rsplit("::", 1)
        return f"{ct}::{gene.strip().upper()}"
    return s.upper()


def _gene_of(node: str) -> str:
    return node.rsplit("::", 1)[-1] if "::" in node else node


def _celltype_of(node: str) -> str:
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"

def unique_gene_and_edge_counts(G, module_df):
    """Collapse the namespaced module down to bare-gene counts."""
    unique_genes = module_df["gene"].nunique()
    unique_nodes = module_df["node"].nunique()

    bare_pairs = set()
    n_identity_edges = 0
    for u, v in G.edges():
        gu, gv = _gene_of(u).upper(), _gene_of(v).upper()
        if gu == gv:
            n_identity_edges += 1   # same gene linked across layers
            continue
        bare_pairs.add(frozenset((gu, gv)))  # undirected, dedup across cell types

    return {
        "unique_genes": unique_genes,
        "unique_nodes": unique_nodes,
        "namespaced_edges": G.number_of_edges(),
        "unique_gene_gene_edges": len(bare_pairs),
        "identity_edges": n_identity_edges,
    }
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
    module_df = pd.read_csv(module_csv, dtype={"node": str})
    module_df["node"] = module_df["node"].astype(str).str.strip()
    module_df["gene"] = module_df["gene"].astype(str).str.strip().str.upper()
    if "cell_type" not in module_df.columns:
        module_df["cell_type"] = module_df["node"].apply(_celltype_of)
    print(f"Loaded module table   : {module_csv} ({len(module_df):,} rows)")

    raw_seeds = []
    if input_dir:
        seeds_path = input_dir / "seeds" / "seeds.txt"
        if seeds_path.exists():
            raw_seeds = [_norm_node(line) for line in open(seeds_path) if line.strip()]
            print(f"Loaded seed list      : {seeds_path} ({len(raw_seeds)} lines)")
        else:
            print(f"  [warn] seeds.txt not found at {seeds_path}; "
                  f"falling back to is_seed column in disease_module.csv")

    if not raw_seeds and "is_seed" in module_df.columns:
        raw_seeds = module_df.loc[module_df["is_seed"], "node"].tolist()

    return G, module_df, raw_seeds


def load_lr_provenance(input_dir):
    if not input_dir:
        return None
    path = Path(input_dir) / "scored_lr_edge_provenance.csv"
    if not path.exists():
        print(f"\n[info] {path} not found -- skipping interaction-type histograms. "
              f"(Re-run step2_build_bipartite_edges.py with --scored-lr-csv to generate it.)")
        return None

    df = pd.read_csv(path)
    required = {"cell_type_source", "cell_type_target", "ligand", "receptor", "source"}
    missing = required - set(df.columns)
    if missing:
        print(f"\n[warn] {path} is missing columns {missing} -- skipping interaction-type histograms.")
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

    print(f"\nLoaded scored-LR edge provenance: {path} ({len(df):,} rows)")
    print(f"  Sources present     : {sorted(df['source'].dropna().unique())}")
    lt_present = sorted(df['ligand_type'].dropna().unique())
    print(f"  Ligand types present: {lt_present if lt_present else '(none)'}")
    return df


def filter_provenance_to_module(prov_df, G):
    if prov_df is None or prov_df.empty:
        return prov_df
    lig_nodes = prov_df["cell_type_source"] + "::" + prov_df["ligand"]
    rec_nodes = prov_df["cell_type_target"] + "::" + prov_df["receptor"]
    mask = [G.has_edge(a, b) for a, b in zip(lig_nodes, rec_nodes)]
    return prov_df[pd.Series(mask, index=prov_df.index)]


def interaction_type_counts(prov_df, column: str) -> pd.Series:
    if prov_df is None or prov_df.empty:
        return pd.Series(dtype=int)
    sub = prov_df.dropna(subset=[column])
    if sub.empty:
        return pd.Series(dtype=int)
    key_cols = ["cell_type_source", "cell_type_target", "ligand", "receptor", column]
    sub = sub.drop_duplicates(subset=key_cols)
    return sub.groupby(column).size().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# NEW: seed connectivity across ALL layers (not just the final module)
# ---------------------------------------------------------------------------
def load_gene_sets(path: Path) -> dict[str, set]:
    with open(path) as f:
        raw = json.load(f)
    return {ct: {_gene_of(g).strip().upper() for g in genes} for ct, genes in raw.items()}


def compute_layer_degrees(input_dir: Path, gene_sets: dict, genes_of_interest: set) -> dict[str, dict[str, int]]:
    """gene -> {cell_type: degree}, read directly from
    multiplex/<ct>/ppi_layer.tsv (or its *.tsv.orig backup if step 2.5's
    namespacing already ran)."""
    degrees: dict[str, dict[str, int]] = {g: {} for g in genes_of_interest}
    for ct in gene_sets:
        layer_path = input_dir / "multiplex" / ct / "ppi_layer.tsv"
        if not layer_path.exists():
            layer_path = input_dir / "multiplex" / ct / "ppi_layer.tsv.orig"
        if not layer_path.exists():
            continue
        df = pd.read_csv(layer_path, sep="\t", header=None, names=["g1", "g2"], dtype=str)
        df["g1"] = df["g1"].astype(str).map(_gene_of).str.strip().str.upper()
        df["g2"] = df["g2"].astype(str).map(_gene_of).str.strip().str.upper()
        deg_counter: dict[str, int] = {}
        for a, b in zip(df["g1"], df["g2"]):
            if a == b:
                continue
            deg_counter[a] = deg_counter.get(a, 0) + 1
            deg_counter[b] = deg_counter.get(b, 0) + 1
        for g in genes_of_interest:
            if g in deg_counter:
                degrees[g][ct] = deg_counter[g]
    return degrees


def seed_layer_connectivity(seed_genes: set, degrees: dict) -> pd.DataFrame:
    rows = []
    for g in sorted(seed_genes):
        layer_degs = degrees.get(g, {})
        n_layers = len(layer_degs)
        total_degree = sum(layer_degs.values())
        dominant_layer = max(layer_degs, key=layer_degs.get) if layer_degs else None
        dominant_degree = layer_degs.get(dominant_layer, 0) if dominant_layer else 0
        rows.append({"gene": g, "n_layers_present": n_layers,
                      "total_degree_across_layers": total_degree,
                      "dominant_layer": dominant_layer, "dominant_layer_degree": dominant_degree})
    return pd.DataFrame(rows).sort_values("total_degree_across_layers", ascending=False)


# ---------------------------------------------------------------------------
# NEW: expression specificity (seeds + top module genes) vs. marker DE ranking
# ---------------------------------------------------------------------------
def load_markers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    return df


def expression_specificity_table(genes: list, markers_df: pd.DataFrame, marker_alpha: float) -> pd.DataFrame:
    has_padj = "p_val_adj" in markers_df.columns
    sig_df = markers_df[markers_df["p_val_adj"] < marker_alpha] if has_padj else markers_df

    ranked = markers_df.copy()
    ranked["rank_in_celltype"] = ranked.groupby("cell_type")["avg_log2FC"].rank(ascending=False, method="min")
    ranked["celltype_marker_pool_size"] = ranked.groupby("cell_type")["gene"].transform("count")

    rows = []
    for g in genes:
        gene_sig = sig_df[sig_df["gene"] == g] if has_padj else pd.DataFrame()
        gene_all = ranked[ranked["gene"] == g]
        n_celltypes_marker = gene_sig["cell_type"].nunique() if has_padj else None

        if len(gene_all):
            best_row = gene_all.loc[gene_all["avg_log2FC"].idxmax()]
            best_celltype = best_row["cell_type"]
            best_log2fc = float(best_row["avg_log2FC"])
            best_rank = float(best_row["rank_in_celltype"])
            best_pool = int(best_row["celltype_marker_pool_size"])
            best_percentile = best_rank / best_pool if best_pool else None
        else:
            best_celltype = best_log2fc = best_rank = best_percentile = None

        if has_padj:
            specificity_flag = "not_in_markers" if not len(gene_all) else (
                "specific" if n_celltypes_marker <= 2 else
                "moderate" if n_celltypes_marker <= 5 else "broad_non_specific"
            )
        else:
            specificity_flag = "not_in_markers" if not len(gene_all) else "unknown (no p_val_adj column)"

        rows.append({
            "gene": g,
            "n_celltypes_significant_marker": n_celltypes_marker,
            "best_celltype": best_celltype,
            "best_avg_log2FC": best_log2fc,
            "rank_in_best_celltype": best_rank,
            "specificity_percentile_in_best_celltype": best_percentile,
            "specificity_flag": specificity_flag,
        })
    return pd.DataFrame(rows)


def plot_specificity_vs_centrality(spec_df: pd.DataFrame, module_df: pd.DataFrame, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [plot] matplotlib not available -- skipping '{out_path.name}'.")
        return False

    merged = spec_df.merge(module_df.groupby("gene")["degree_in_module"].max().reset_index(), on="gene", how="left")
    merged = merged.dropna(subset=["specificity_percentile_in_best_celltype", "degree_in_module"])
    if merged.empty:
        print("  [plot] Nothing to plot for specificity-vs-centrality (missing values) -- skipping.")
        return False

    plt.figure(figsize=(7, 6))
    plt.scatter(merged["degree_in_module"], merged["specificity_percentile_in_best_celltype"],
                alpha=0.6, s=25, c="#3b6fa0")
    for _, row in merged.sort_values("degree_in_module", ascending=False).head(15).iterrows():
        plt.annotate(row["gene"], (row["degree_in_module"], row["specificity_percentile_in_best_celltype"]),
                     fontsize=7, alpha=0.8)
    plt.xlabel("Degree in final disease module")
    plt.ylabel("Specificity percentile in best cell type\n(0 = most specific, 1 = least)")
    plt.title("Module centrality vs. expression specificity")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {out_path}")
    return True


# ---------------------------------------------------------------------------
# Cell-type extraction -- node-based (authoritative), not edge-based
# ---------------------------------------------------------------------------
def get_cell_types(module_df):
    return sorted(module_df["cell_type"].dropna().unique().tolist())


def per_celltype_stats(G, module_df, seed_set):
    total_nodes = len(module_df)
    rows = []
    for ct in get_cell_types(module_df):
        ct_rows = module_df[module_df["cell_type"] == ct]
        ct_nodes = set(ct_rows["node"])
        n_nodes = len(ct_nodes)

        n_edges = sum(1 for u, v in G.edges() if u in ct_nodes and v in ct_nodes)

        n_seeds = len(ct_nodes & seed_set)
        n_isolated = int((ct_rows.get("degree_in_module", pd.Series(dtype=int)) == 0).sum()) \
            if "degree_in_module" in ct_rows.columns else 0
        n_isolated_seeds = int(((ct_rows["node"].isin(seed_set)) &
                                 (ct_rows.get("degree_in_module", 1) == 0)).sum()) \
            if "degree_in_module" in ct_rows.columns else 0

        rows.append({
            "cell_type": ct,
            "major_category": get_category_from_cell_type(ct),
            "n_nodes": n_nodes,
            "n_intra_celltype_edges": n_edges,
            "pct_of_module_nodes": round(100 * n_nodes / total_nodes, 2) if total_nodes else 0.0,
            "n_seeds": n_seeds,
            "seed_fraction_of_celltype": round(n_seeds / n_nodes, 4) if n_nodes else 0.0,
            "n_isolated_nodes": n_isolated,
            "n_isolated_seeds": n_isolated_seeds,
        })
    df = pd.DataFrame(rows).sort_values("n_nodes", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_bar(labels, values, title, ylabel, out_path, color="#4C72B0", figsize=(10, 6),
             bar_colors=None, legend_title=None):
    if not labels:
        print(f"  [plot] No data for '{title}' -- skipping.")
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print(f"  [plot] matplotlib not available -- skipping '{title}' "
              f"(underlying data is still saved to CSV).")
        return False

    plt.figure(figsize=figsize)
    plt.bar(range(len(labels)), values, color=bar_colors if bar_colors else color)
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    if bar_colors is not None:
        palette = _major_category_palette()
        present_categories = []
        seen = set()
        for lbl, c in zip(labels, bar_colors):
            cat = next((k for k, v in palette.items() if v == c), None)
            if cat and cat not in seen:
                seen.add(cat)
                present_categories.append(cat)
        if present_categories:
            handles = [mpatches.Patch(color=palette[cat], label=cat) for cat in present_categories]
            plt.legend(handles=handles, title=legend_title or "Major category",
                       loc="upper right", fontsize=8, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {out_path}")
    return True


def compute_top_seed_genes(sm, top_n=20):
    present = sm[sm["in_final_module"]]
    if present.empty:
        return pd.Series(dtype=int)
    counts = present.groupby("gene")["cell_type"].nunique().sort_values(ascending=False)
    return counts.head(top_n)


# ---------------------------------------------------------------------------
# Seed-level diagnostics
# ---------------------------------------------------------------------------
def seed_membership(module_df, raw_seeds):
    node_info = module_df.set_index("node")
    module_node_set = set(node_info.index)

    dup_in_file = Counter(raw_seeds)
    duplicated_in_file = {g: c for g, c in dup_in_file.items() if c > 1}

    unique_seeds = sorted(set(raw_seeds))

    gene_to_celltypes = defaultdict(set)
    for s in unique_seeds:
        gene_to_celltypes[_gene_of(s)].add(_celltype_of(s))

    rows = []
    for s in unique_seeds:
        in_module = s in module_node_set
        if in_module:
            info = node_info.loc[s]
            deg = int(info["degree_in_module"]) if "degree_in_module" in info else None
            ct = info["cell_type"] if "cell_type" in info else _celltype_of(s)
        else:
            deg = None
            ct = _celltype_of(s)
        gene = _gene_of(s)
        n_cts_seeded = len(gene_to_celltypes[gene])
        rows.append({
            "seed": s,
            "gene": gene,
            "cell_type": ct,
            "n_times_in_seeds_txt": dup_in_file.get(s, 0),
            "in_final_module": in_module,
            "degree_in_module": deg if deg is not None else 0,
            "isolated_in_module": in_module and deg == 0,
            "n_celltypes_seeded_in": n_cts_seeded,
            "spans_multiple_celltypes": n_cts_seeded > 1,
        })
    df = pd.DataFrame(rows)
    return df, duplicated_in_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True,
                     help="Directory containing module_graph.pkl / disease_module.csv")
    ap.add_argument("--input-dir", default=None,
                     help="MultiXrank wdir containing seeds/seeds.txt, celltype_gene_sets.json, "
                          "multiplex/<ct>/ppi_layer.tsv, and (if step2 was run with --scored-lr-csv) "
                          "scored_lr_edge_provenance.csv -- same --input-dir passed to step6")
    ap.add_argument("--celltype-gene-sets", default=None,
                     help="celltype_gene_sets.json (step 1). Defaults to <input-dir>/celltype_gene_sets.json. "
                          "Needed for the seed-connectivity-across-all-layers section.")
    ap.add_argument("--markers-csv", default=None,
                     help="Optional cell_type_markers.csv (from your R marker-extraction script). If given, "
                          "adds an expression-specificity section for seeds and top module genes.")
    ap.add_argument("--marker-alpha", type=float, default=0.05)
    ap.add_argument("--top-n-module-genes", type=int, default=50,
                     help="How many top (by degree_in_module) non-seed module genes to check for "
                          "expression specificity, if --markers-csv is given (default: 50)")
    ap.add_argument("--out-prefix", default="module_exploratory")
    ap.add_argument("--top-n-seeds", type=int, default=20,
                     help="Number of top individual seed genes to show in the "
                          "top-seed-genes bar chart (default: 20).")
    ap.add_argument("--no-plots", action="store_true",
                     help="Skip generating the bar-chart PNGs (CSVs/summary are still saved).")
    args = ap.parse_args()

    G, module_df, raw_seeds = load(args.output_dir, args.input_dir)
    seed_set = set(raw_seeds)

    cell_types = get_cell_types(module_df)
    counts = unique_gene_and_edge_counts(G, module_df)

    print("\n" + "=" * 70)
    print("MODULE OVERVIEW")
    print("=" * 70)
    print(f"Total nodes (ct::gene)        : {counts['unique_nodes']:,}")
    print(f"Unique genes (bare symbols)   : {counts['unique_genes']:,}")
    print(f"Total edges (namespaced)      : {counts['namespaced_edges']:,}")
    print(f"  of which cross-layer identity edges : {counts['identity_edges']:,}")
    print(f"Unique gene-gene edges (collapsed across cell types) : {counts['unique_gene_gene_edges']:,}")
    for ct in cell_types:
        print(f"  - {ct}")

    if "degree_in_module" in module_df.columns:
        isolated_df = module_df[module_df["degree_in_module"] == 0]
    else:
        deg = dict(G.degree())
        isolated_df = module_df[module_df["node"].map(deg).fillna(0) == 0]
    isolated_all = sorted(isolated_df["node"])
    print(f"\nIsolated nodes (degree 0, whole module): {len(isolated_all):,}")
    isolated_seeds_all = sorted(set(isolated_all) & seed_set)
    print(f"  of which are seeds                    : {len(isolated_seeds_all)}")
    if isolated_seeds_all:
        print(f"    {isolated_seeds_all[:30]}{' ...' if len(isolated_seeds_all) > 30 else ''}")

    # --- per cell type ---
    pl = per_celltype_stats(G, module_df, seed_set)
    print("\n" + "-" * 70)
    print("PER-CELL-TYPE STATS (sorted by node count, largest first)")
    print("-" * 70)
    print(pl.to_string(index=False))
    pl_path = str(Path(args.output_dir) / f"{args.out_prefix}_per_celltype.csv")
    pl.to_csv(pl_path, index=False)

    if not pl.empty:
        node_share = pl["n_nodes"]
        cv = node_share.std() / node_share.mean() if node_share.mean() else 0
        top = pl.iloc[0]
        print(f"\nLargest cell type: {top['cell_type']} "
              f"({top['n_nodes']:,} nodes, {top['pct_of_module_nodes']}% of module)")
        print(f"Coefficient of variation of cell-type sizes: {cv:.2f} "
              f"({'uneven' if cv > 0.5 else 'fairly even'} representation across cell types)")

    # --- seed membership (within final module) ---
    sm, dup_in_file = seed_membership(module_df, raw_seeds)
    print("\n" + "-" * 70)
    print("SEED DIAGNOSTICS (within final module)")
    print("-" * 70)
    unique_seed_nodes = set(raw_seeds)
    unique_seed_gene_set = {_gene_of(s) for s in raw_seeds}
    
    print(f"Seed lines in seeds.txt      : {len(raw_seeds)}")
    print(f"Unique seed nodes (ct::gene) : {len(unique_seed_nodes)}")
    print(f"Unique seed genes (bare)     : {len(unique_seed_gene_set)}")
    if dup_in_file:
        print(f"Duplicate lines in seeds.txt : {len(dup_in_file)} gene(s) -> {dup_in_file}")
    else:
        print("Duplicate lines in seeds.txt : none")

    print(f"\nSeeds present in final module        : {sm['in_final_module'].sum()} / {len(sm)}")
    print(f"Seeds missing from final module      : {(~sm['in_final_module']).sum()}")
    print(f"Seeds isolated in final module       : {sm['isolated_in_module'].sum()}")
    print(f"Seed genes seeded under >1 cell type : {sm['spans_multiple_celltypes'].sum()}")

    print("\nSeed counts by cell type (present & connected within that cell type):")
    for _, row in pl.iterrows():
        print(f"  {row['cell_type']:<30} {row['n_seeds']:>4} seeds "
              f"({row['seed_fraction_of_celltype']*100:.1f}% of cell type's nodes)  "
              f"| isolated seeds: {row['n_isolated_seeds']}")

    sm_path = str(Path(args.output_dir) / f"{args.out_prefix}_seed_membership.csv")
    sm.to_csv(sm_path, index=False)

    # --- NEW: seed connectivity across ALL layers (not just the final module) ---
    gene_sets_path = Path(args.celltype_gene_sets) if args.celltype_gene_sets else (
        Path(args.input_dir) / "celltype_gene_sets.json" if args.input_dir else None)
    seed_layer_df = None
    seed_layer_path = None
    unique_seed_genes = sorted({_gene_of(s).upper() for s in raw_seeds}) if raw_seeds else \
        sorted(module_df.loc[module_df.get("is_seed", False), "gene"].unique())

    if gene_sets_path and gene_sets_path.exists() and args.input_dir:
        print("\n" + "-" * 70)
        print("SEED CONNECTIVITY ACROSS ALL LAYERS (not just final module)")
        print("-" * 70)
        gene_sets = load_gene_sets(gene_sets_path)
        degrees = compute_layer_degrees(Path(args.input_dir), gene_sets, set(unique_seed_genes))
        seed_layer_df = seed_layer_connectivity(set(unique_seed_genes), degrees)
        seed_layer_path = str(Path(args.output_dir) / f"{args.out_prefix}_seed_layer_connectivity.csv")
        seed_layer_df.to_csv(seed_layer_path, index=False)
        print(seed_layer_df.head(20).to_string(index=False))
        print(f"  ({len(seed_layer_df)} unique seed gene(s); full table -> {seed_layer_path})")
        n_absent = int((seed_layer_df["n_layers_present"] == 0).sum())
        print(f"  {n_absent} seed gene(s) absent from EVERY cell-type PPI layer entirely")
    else:
        print("\n[info] celltype_gene_sets.json not found/given (or --input-dir not passed) -- "
              "skipping seed-connectivity-across-all-layers section.")

    # --- NEW: expression specificity (seeds + top module genes) ---
    seed_spec_path = module_spec_path = spec_plot_path = None
    if args.markers_csv:
        print("\n" + "-" * 70)
        print("EXPRESSION SPECIFICITY (vs. cell_type_markers.csv)")
        print("-" * 70)
        markers_df = load_markers(args.markers_csv)
        print(f"  Loaded {len(markers_df):,} marker rows across {markers_df['cell_type'].nunique()} cell types")

        seed_spec_df = expression_specificity_table(unique_seed_genes, markers_df, args.marker_alpha)
        if seed_layer_df is not None:
            seed_spec_df = seed_spec_df.merge(seed_layer_df, on="gene", how="left")
        seed_spec_path = str(Path(args.output_dir) / f"{args.out_prefix}_seed_expression_specificity.csv")
        seed_spec_df.to_csv(seed_spec_path, index=False)
        print("  Seed specificity flags:")
        print(seed_spec_df["specificity_flag"].value_counts().to_string())

        non_seed_module = module_df[~module_df["node"].isin(seed_set)].copy()
        top_genes = (non_seed_module.groupby("gene")["degree_in_module"].max()
                     .sort_values(ascending=False).head(args.top_n_module_genes).index.tolist())
        module_spec_df = expression_specificity_table(top_genes, markers_df, args.marker_alpha)
        module_spec_path = str(Path(args.output_dir) / f"{args.out_prefix}_module_top_gene_expression_specificity.csv")
        module_spec_df.to_csv(module_spec_path, index=False)
        print(f"\n  Top {args.top_n_module_genes} non-seed module gene specificity flags:")
        print(module_spec_df["specificity_flag"].value_counts().to_string())

        all_top_spec = expression_specificity_table(
            (module_df.groupby("gene")["degree_in_module"].max()
             .sort_values(ascending=False).head(max(args.top_n_module_genes, 100)).index.tolist()),
            markers_df, args.marker_alpha,
        )
        spec_plot_path_obj = Path(args.output_dir) / f"{args.out_prefix}_specificity_vs_centrality.png"
        if not args.no_plots and plot_specificity_vs_centrality(all_top_spec, module_df, spec_plot_path_obj):
            spec_plot_path = str(spec_plot_path_obj)
    else:
        print("\n[info] --markers-csv not given -- skipping expression-specificity section.")

    # --- bar charts (cell type) ---
    seed_pct_path = str(Path(args.output_dir) / f"{args.out_prefix}_seed_pct_by_celltype.png")
    module_pct_path = str(Path(args.output_dir) / f"{args.out_prefix}_module_pct_by_celltype.png")
    top_seeds_path = str(Path(args.output_dir) / f"{args.out_prefix}_top_seed_genes.png")
    seed_pct_ok = module_pct_ok = top_seeds_ok = False

    if not args.no_plots:
        print("\n" + "-" * 70)
        print("PLOTS")
        print("-" * 70)

        total_seeds_in_module = int(pl["n_seeds"].sum())
        if total_seeds_in_module > 0:
            seed_pl = pl[pl["n_seeds"] > 0].copy()
            seed_pl["seed_pct"] = (seed_pl["n_seeds"] / total_seeds_in_module * 100).round(2)
            seed_pl = seed_pl.sort_values("seed_pct", ascending=False)
            palette = _major_category_palette()
            seed_colors = [palette.get(c, palette["unknown"]) for c in seed_pl["major_category"]]
            seed_pct_ok = plot_bar(
                seed_pl["cell_type"].tolist(), seed_pl["seed_pct"].tolist(),
                title="Seed distribution across cell types",
                ylabel="% of module seeds", out_path=seed_pct_path,
                bar_colors=seed_colors,
            )
        else:
            print("  [plot] No seeds present in final module -- skipping seed-distribution plot.")

        module_pl = pl.sort_values("pct_of_module_nodes", ascending=False)
        palette = _major_category_palette()
        module_colors = [palette.get(c, palette["unknown"]) for c in module_pl["major_category"]]
        module_pct_ok = plot_bar(
            module_pl["cell_type"].tolist(), module_pl["pct_of_module_nodes"].tolist(),
            title="Disease module composition by cell type",
            ylabel="% of module nodes", out_path=module_pct_path,
            bar_colors=module_colors,
        )

        top_counts = compute_top_seed_genes(sm, top_n=args.top_n_seeds)
        top_seeds_ok = plot_bar(
            top_counts.index.tolist(), top_counts.values.tolist(),
            title=f"Top {len(top_counts)} seed genes by # cell types seeded (present in module)",
            ylabel="Number of cell types (in module)", out_path=top_seeds_path, color="#C44E52",
        )

    # --- interaction-type histograms (method/source + NeuronChat ligand_type) ---
    prov_df = load_lr_provenance(args.input_dir)
    interaction_csv_paths = {}
    interaction_plot_paths = {}
    module_prov_df = None

    if prov_df is not None and not prov_df.empty:
        module_prov_df = filter_provenance_to_module(prov_df, G)

        print("\n" + "-" * 70)
        print("INTERACTION-TYPE BREAKDOWN (LR-inference method / NeuronChat ligand_type)")
        print("-" * 70)
        print(f"Candidate scored-LR edges (survived step2 filtering) : {len(prov_df):,}")
        print(f"  of which present as an edge in the final module    : {len(module_prov_df):,}")

        scopes = {"candidate": prov_df, "module": module_prov_df}
        dims = {"source": ("source", "LR-inference method/source", "#8172B2"),
                "ligand_type": ("ligand_type", "NeuronChat ligand type", "#55A868")}

        for scope_name, scope_df in scopes.items():
            scope_label = "Final module" if scope_name == "module" else "Candidate"
            for dim_key, (dim_col, dim_label, color) in dims.items():
                counts = interaction_type_counts(scope_df, dim_col)

                csv_path = str(Path(args.output_dir) / f"{args.out_prefix}_{scope_name}_lr_by_{dim_key}.csv")
                counts.rename("n_edges").rename_axis(dim_col).reset_index().to_csv(csv_path, index=False)
                interaction_csv_paths[(scope_name, dim_key)] = csv_path

                print(f"\n{scope_label} LR edges by {dim_label}:")
                if counts.empty:
                    print("  (no rows with this field populated)")
                else:
                    print(counts.to_string())

                plot_path = str(Path(args.output_dir) / f"{args.out_prefix}_{scope_name}_lr_by_{dim_key}.png")
                if not args.no_plots and not counts.empty:
                    ok = plot_bar(
                        counts.index.tolist(), counts.values.tolist(),
                        title=f"{scope_label} LR edges by {dim_label}",
                        ylabel="Number of unique LR edges", out_path=plot_path, color=color,
                    )
                    interaction_plot_paths[(scope_name, dim_key)] = plot_path if ok else None
                else:
                    interaction_plot_paths[(scope_name, dim_key)] = None
    else:
        print("\n[info] No scored-LR provenance available -- interaction-type histograms skipped.")

    # --- summary file ---
    summary_lines = [
        "=" * 70,
        "MODULE EXPLORATORY DIAGNOSTICS",
        "=" * 70,
        f"Total nodes: {counts['unique_nodes']:,}   Unique genes: {counts['unique_genes']:,}",
        f"Total edges (namespaced): {counts['namespaced_edges']:,}   "
        f"Unique gene-gene edges: {counts['unique_gene_gene_edges']:,}   "
        f"Identity edges: {counts['identity_edges']:,}",
        f"Cell types represented: {len(cell_types)}",
        "",
        "Per-cell-type breakdown:",
        pl.to_string(index=False),
        "",
        f"Isolated nodes overall: {len(isolated_all):,} (of which seeds: {len(isolated_seeds_all)})",
        f"Seed lines in seeds.txt: {len(raw_seeds)}  | unique nodes: {len(set(raw_seeds))}  "
        f"| unique genes: {len({_gene_of(s) for s in raw_seeds})}  "
        f"| duplicate lines: {dup_in_file if dup_in_file else 'none'}",
        f"Seeds in final module: {sm['in_final_module'].sum()} / {len(sm)}",
        f"Seeds isolated in module: {sm['isolated_in_module'].sum()}",
        f"Seed genes seeded under >1 cell type: {sm['spans_multiple_celltypes'].sum()}",
        "",
        f"Full seed-level table -> {sm_path}",
        f"Full per-cell-type table -> {pl_path}",
        "",
        f"Seed distribution by cell type plot -> {seed_pct_path if seed_pct_ok else '(not generated)'}",
        f"Module composition by cell type plot -> {module_pct_path if module_pct_ok else '(not generated)'}",
        f"Top seed genes plot -> {top_seeds_path if top_seeds_ok else '(not generated)'}",
    ]

    if seed_layer_df is not None:
        summary_lines += [
            "",
            "Seed connectivity across ALL layers (not just final module):",
            f"  Table -> {seed_layer_path}",
        ]

    if args.markers_csv:
        summary_lines += [
            "",
            "Expression specificity (vs. cell_type_markers.csv):",
            f"  Seed table -> {seed_spec_path}",
            f"  Top module-gene table -> {module_spec_path}",
            f"  Plot -> {spec_plot_path if spec_plot_path else '(not generated)'}",
        ]

    if prov_df is not None and not prov_df.empty:
        summary_lines += [
            "",
            "Interaction-type breakdown (LR-inference method / NeuronChat ligand_type):",
            f"  Candidate scored-LR edges (post step2 filtering): {len(prov_df):,}",
            f"  Of those, present in final module                : {len(module_prov_df):,}",
        ]
        for (scope_name, dim_key), csv_path in interaction_csv_paths.items():
            plot_path = interaction_plot_paths.get((scope_name, dim_key))
            summary_lines.append(
                f"  [{scope_name} / {dim_key}] table -> {csv_path}"
                + (f" | plot -> {plot_path}" if plot_path else " | plot -> (not generated)")
            )
    else:
        summary_lines += ["", "Interaction-type breakdown: skipped (no scored_lr_edge_provenance.csv found)."]

    summary_lines.append("=" * 70)
    summary_path = str(Path(args.output_dir) / f"{args.out_prefix}_summary.txt")
    Path(summary_path).write_text("\n".join(summary_lines))

    print(f"\nSaved: {pl_path}")
    print(f"Saved: {sm_path}")
    if seed_layer_path:
        print(f"Saved: {seed_layer_path}")
    if seed_spec_path:
        print(f"Saved: {seed_spec_path}")
    if module_spec_path:
        print(f"Saved: {module_spec_path}")
    if spec_plot_path:
        print(f"Saved: {spec_plot_path}")
    print(f"Saved: {summary_path}")
    if seed_pct_ok:
        print(f"Saved: {seed_pct_path}")
    if module_pct_ok:
        print(f"Saved: {module_pct_path}")
    if top_seeds_ok:
        print(f"Saved: {top_seeds_path}")
    for (scope_name, dim_key), csv_path in interaction_csv_paths.items():
        print(f"Saved: {csv_path}")
        plot_path = interaction_plot_paths.get((scope_name, dim_key))
        if plot_path:
            print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()