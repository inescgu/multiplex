#!/usr/bin/env python3
"""
multiplex_source_audit_v2.py
============================
Audit the origin of every connection type in your MultiXrank multiplex.

Key questions answered
----------------------
  1. Which Reactome pathways are represented, and how many interactions each?
  2. Which KEGG pathways are represented?
  3. How well do LR-pair genes overlap with:
       a) genes in your cell-type layers
       b) genes in the Reactome interaction network
       c) genes covered by KEGG pathways
  4. Complete printed lists of every pathway name used by each database.

Inputs
------
  --reactome-csv   reactome_interactions_with_pathways.csv
  --lr-csv         lr_pairs.csv
  --kegg-csv       kegg_pathways_with_names_symbols.csv   (optional but recommended)
  --layers-dir     layers_uniprot/                         (optional)
  --uniprot-map    d_uniprot_to_symbol_20251120.csv        (optional, improves overlaps)
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

DIVIDER = "=" * 72


# ============================================================================
# HELPERS
# ============================================================================

def _to_upper_set(series: pd.Series) -> set:
    """Drop nulls, strip whitespace, upper-case → Python set."""
    return set(series.dropna().astype(str).str.strip().str.upper().unique())


def _load_uniprot_map(csv_path: str) -> tuple[dict, dict]:
    """
    Returns
    -------
    uniprot_to_symbol : {uniprot_id  -> SYMBOL}
    symbol_to_uniprot : {SYMBOL      -> uniprot_id}
    """
    df = pd.read_csv(csv_path)
    # accept either casing of column names
    df.columns = df.columns.str.lower()
    uid_col = next(c for c in df.columns if "uniprot" in c)
    sym_col = next(c for c in df.columns if "symbol" in c or "gene" in c)
    u2s = dict(zip(df[uid_col].str.strip(), df[sym_col].str.strip().str.upper()))
    s2u = {v: k for k, v in u2s.items()}
    return u2s, s2u


def _print_block(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def _overlap_report(label_a: str, set_a: set,
                    label_b: str, set_b: set,
                    indent: str = "  ") -> dict:
    """Print a standard A∩B overlap block and return metrics."""
    overlap = set_a & set_b
    only_a  = set_a - set_b
    only_b  = set_b - set_a

    pct_a = 100 * len(overlap) / len(set_a) if set_a else 0
    pct_b = 100 * len(overlap) / len(set_b) if set_b else 0

    print(f"{indent}{label_a:<35} : {len(set_a):>7,}")
    print(f"{indent}{label_b:<35} : {len(set_b):>7,}")
    print(f"{indent}Overlap (A ∩ B){'':20} : {len(overlap):>7,}  "
          f"({pct_a:.1f}% of {label_a} / {pct_b:.1f}% of {label_b})")
    print(f"{indent}Only in {label_a:<26} : {len(only_a):>7,}")
    print(f"{indent}Only in {label_b:<26} : {len(only_b):>7,}")

    return dict(overlap=overlap, only_a=only_a, only_b=only_b,
                pct_a=pct_a, pct_b=pct_b)


# ============================================================================
# SECTION 1 — LOAD EVERYTHING
# ============================================================================

def load_all(reactome_csv: str,
             lr_csv: str,
             kegg_csv: str | None,
             layers_dir: str | None,
             uniprot_map_csv: str | None) -> dict:

    data = {}

    # ── Reactome/KEGG merged interactions ────────────────────────────────
    print(f"Loading {reactome_csv} …")
    data["reactome_df"] = pd.read_csv(reactome_csv, low_memory=False)
    print(f"  {len(data['reactome_df']):,} rows")

    # ── LR pairs ─────────────────────────────────────────────────────────
    print(f"Loading {lr_csv} …")
    data["lr_df"] = pd.read_csv(lr_csv, low_memory=False)
    print(f"  {len(data['lr_df']):,} rows")

    # ── KEGG (optional) ───────────────────────────────────────────────────
    if kegg_csv and os.path.exists(kegg_csv):
        print(f"Loading {kegg_csv} …")
        data["kegg_df"] = pd.read_csv(kegg_csv, low_memory=False)
        print(f"  {len(data['kegg_df']):,} rows")
    else:
        data["kegg_df"] = None

    # ── Cell-type layers (optional) ───────────────────────────────────────
    if layers_dir and os.path.exists(layers_dir):
        pkl_files = sorted(Path(layers_dir).glob("*_layer.pkl"))
        layers = {}
        for fp in pkl_files:
            ct = fp.stem.replace("_layer", "")
            with open(fp, "rb") as f:
                layers[ct] = pickle.load(f)
        data["layers"] = layers
        print(f"Loaded {len(layers)} cell-type layers from {layers_dir}")
    else:
        data["layers"] = {}

    # ── UniProt ↔ symbol map (optional) ──────────────────────────────────
    if uniprot_map_csv and os.path.exists(uniprot_map_csv):
        u2s, s2u = _load_uniprot_map(uniprot_map_csv)
        data["uniprot_to_symbol"] = u2s
        data["symbol_to_uniprot"] = s2u
        print(f"Loaded UniProt map: {len(u2s):,} entries")
    else:
        data["uniprot_to_symbol"] = {}
        data["symbol_to_uniprot"] = {}

    return data


# ============================================================================
# SECTION 2 — BUILD GENE UNIVERSES
# ============================================================================

def build_gene_universes(data: dict) -> dict:
    """
    Produce consistent UPPER-CASE gene-symbol sets for each source.
    When a UniProt map is available, Reactome UniProt IDs are translated.
    """
    _print_block("GENE UNIVERSES")

    u2s = data["uniprot_to_symbol"]        # may be empty
    universes = {}

    # ── Reactome proteins (UniProt IDs → symbols if map available) ────────
    rdf = data["reactome_df"]
    reactome_uniprots: set = set()
    for col in ["uniprot1", "uniprot2"]:
        if col in rdf.columns:
            reactome_uniprots.update(_to_upper_set(rdf[col]))

    if u2s:
        reactome_symbols = {u2s[p].upper() for p in reactome_uniprots
                            if p in u2s}
        unmapped_r = reactome_uniprots - set(u2s.keys())
        print(f"  Reactome unique UniProt IDs   : {len(reactome_uniprots):,}")
        print(f"    → mapped to gene symbols    : {len(reactome_symbols):,}")
        print(f"    → unmapped (kept as UniProt): {len(unmapped_r):,}")
        universes["reactome"] = reactome_symbols
    else:
        print(f"  Reactome unique UniProt IDs   : {len(reactome_uniprots):,}  "
              f"(no symbol map → using UniProt IDs)")
        universes["reactome"] = reactome_uniprots   # UniProt IDs as-is

    # ── KEGG gene symbols ─────────────────────────────────────────────────
    kdf = data.get("kegg_df")
    if kdf is not None:
        # prefer 'gene_symbol' if present, else 'gene_name'
        sym_col = next(
            (c for c in kdf.columns
             if "symbol" in c.lower() or "gene_name" in c.lower()),
            None
        )
        if sym_col:
            universes["kegg"] = _to_upper_set(kdf[sym_col])
            print(f"  KEGG unique gene symbols      : {len(universes['kegg']):,}  "
                  f"(column: '{sym_col}')")
        else:
            universes["kegg"] = set()
            print("  KEGG CSV found but no symbol column detected.")
    else:
        universes["kegg"] = set()
        print("  KEGG CSV not provided.")

    # ── Layer genes ───────────────────────────────────────────────────────
    layer_genes: set = set()
    for ct, G in data["layers"].items():
        layer_genes.update(str(n).upper() for n in G.nodes())
    universes["layers"] = layer_genes
    if layer_genes:
        print(f"  Layer genes (union)           : {len(layer_genes):,}")
    else:
        print("  No layer files loaded.")

    # ── LR genes ─────────────────────────────────────────────────────────
    lr_df = data["lr_df"]
    lig_col = next((c for c in lr_df.columns if "ligand" in c.lower()), None)
    rec_col = next((c for c in lr_df.columns if "receptor" in c.lower()), None)

    lr_genes: set = set()
    if lig_col:
        lr_genes.update(_to_upper_set(lr_df[lig_col]))
    if rec_col:
        lr_genes.update(_to_upper_set(lr_df[rec_col]))
    universes["lr"] = lr_genes
    print(f"  LR unique genes               : {len(lr_genes):,}  "
          f"(ligand col: '{lig_col}', receptor col: '{rec_col}')")

    data["universes"] = universes
    return universes


# ============================================================================
# SECTION 3 — REACTOME PATHWAY AUDIT (full list + source breakdown)
# ============================================================================

def audit_reactome_pathways(data: dict) -> pd.DataFrame:
    _print_block("REACTOME + KEGG — ALL PATHWAYS USED")

    rdf = data["reactome_df"]

    if "common_pathway" not in rdf.columns:
        print("  'common_pathway' column not found — skipping.")
        return pd.DataFrame()

    # Explode pipe-separated pathway names
    exploded = (
        rdf[["common_pathway", "pathway_sources"]]
        .dropna(subset=["common_pathway"])
        .copy()
    )
    exploded["common_pathway"] = exploded["common_pathway"].str.split("|")
    exploded = exploded.explode("common_pathway")
    exploded["common_pathway"] = exploded["common_pathway"].str.strip()

    summary = (
        exploded
        .groupby("common_pathway")
        .agg(
            n_interactions=("common_pathway", "count"),
            sources=(
                "pathway_sources",
                lambda x: "|".join(sorted({
                    s.strip()
                    for v in x
                    for s in str(v).split("|")
                    if s.strip()
                }))
            ),
        )
        .reset_index()
        .sort_values("n_interactions", ascending=False)
        .reset_index(drop=True)
    )
    summary["in_reactome"] = summary["sources"].str.contains("Reactome", na=False)
    summary["in_kegg"]     = summary["sources"].str.contains("KEGG",     na=False)

    # ── Print ALL pathway names grouped by source ─────────────────────────
    reactome_pathways = summary.loc[summary["in_reactome"], "common_pathway"].tolist()
    kegg_pathways_in  = summary.loc[summary["in_kegg"],     "common_pathway"].tolist()
    both_pathways     = summary.loc[
        summary["in_reactome"] & summary["in_kegg"], "common_pathway"
    ].tolist()

    print(f"\n  Total unique pathways across both databases : {len(summary):,}")
    print(f"  Reactome-supported pathways                 : {len(reactome_pathways):,}")
    print(f"  KEGG-supported pathways                     : {len(kegg_pathways_in):,}")
    print(f"  Pathways present in BOTH                    : {len(both_pathways):,}")

    # ── ALL REACTOME PATHWAYS ─────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  ALL REACTOME PATHWAY NAMES IN YOUR DATA  "
          "(sorted by interaction count)")
    print(f"{'─'*72}")
    r_only_df = summary[summary["in_reactome"]].sort_values(
        "n_interactions", ascending=False
    )
    for i, row in enumerate(r_only_df.itertuples(), 1):
        tag = "[+KEGG]" if row.in_kegg else "       "
        print(f"  {i:>4}. {tag}  {row.n_interactions:>6,}  {row.common_pathway}")

    # ── ALL KEGG PATHWAYS ─────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  ALL KEGG PATHWAY NAMES IN YOUR DATA  "
          "(sorted by interaction count)")
    print(f"{'─'*72}")
    k_only_df = summary[summary["in_kegg"]].sort_values(
        "n_interactions", ascending=False
    )
    for i, row in enumerate(k_only_df.itertuples(), 1):
        tag = "[+Reactome]" if row.in_reactome else "           "
        print(f"  {i:>4}. {tag}  {row.n_interactions:>6,}  {row.common_pathway}")

    # ── Pathways in KEGG CSV (all, not just those in merged interactions) ─
    kdf = data.get("kegg_df")
    if kdf is not None and "pathway_name" in kdf.columns:
        all_kegg_names = sorted(kdf["pathway_name"].dropna().unique())
        print(f"\n{'─'*72}")
        print(f"  ALL KEGG PATHWAY NAMES IN kegg_pathways_with_names_symbols.csv")
        print(f"  (total: {len(all_kegg_names)})")
        print(f"{'─'*72}")
        for i, name in enumerate(all_kegg_names, 1):
            print(f"  {i:>4}. {name}")

    data["pathway_summary"] = summary
    return summary


# ============================================================================
# SECTION 4 — LR PAIR AUDIT
# ============================================================================

def audit_lr_pairs(data: dict) -> dict:
    _print_block("LIGAND–RECEPTOR PAIRS")

    lr_df = data["lr_df"]
    print(f"  {len(lr_df):,} LR pairs loaded")
    print(f"  Columns: {lr_df.columns.tolist()}\n")

    results = {}

    # ── Basic counts ──────────────────────────────────────────────────────
    lig_col = next((c for c in lr_df.columns if "ligand"   in c.lower()), None)
    rec_col = next((c for c in lr_df.columns if "receptor" in c.lower()), None)

    if lig_col and rec_col:
        ligs = _to_upper_set(lr_df[lig_col])
        recs = _to_upper_set(lr_df[rec_col])
        print(f"  Unique ligands   : {len(ligs):,}")
        print(f"  Unique receptors : {len(recs):,}")
        print(f"  Union (all LR)   : {len(ligs | recs):,}")
        results["ligands"]   = ligs
        results["receptors"] = recs

    # ── Interaction type / pathway / annotation columns ───────────────────
    for pattern in ["type", "category", "annotation", "pathway", "subtype"]:
        col = next(
            (c for c in lr_df.columns if pattern in c.lower()), None
        )
        if col:
            vc = lr_df[col].value_counts(dropna=False)
            print(f"\n  Breakdown by '{col}':")
            for val, cnt in vc.items():
                pct = 100 * cnt / len(lr_df)
                print(f"    {str(val):<50} {cnt:>5,}  ({pct:.1f}%)")
            results[f"col_{pattern}"] = col

    print()
    data["lr_results"] = results
    return results


# ============================================================================
# SECTION 5 — OVERLAP ANALYSIS
# ============================================================================

def overlap_analysis(data: dict) -> dict:
    _print_block("OVERLAP ANALYSIS")

    universes = data["universes"]
    lr        = universes["lr"]
    layers    = universes["layers"]
    reactome  = universes["reactome"]
    kegg      = universes["kegg"]

    all_results = {}

    # ── 5a. LR ↔ Layer genes ──────────────────────────────────────────────
    print("\n  [5a]  LR pair genes  ↔  Cell-type layer genes")
    print(f"{'─'*60}")
    if layers:
        r = _overlap_report("LR genes", lr, "Layer genes", layers)
        all_results["lr_vs_layers"] = r

        # Per-layer breakdown
        print(f"\n  Per-layer overlap with LR genes:")
        layer_rows = []
        for ct, G in data["layers"].items():
            lg = {str(n).upper() for n in G.nodes()}
            ov = lr & lg
            pct = 100 * len(ov) / len(lr) if lr else 0
            layer_rows.append({"cell_type": ct,
                                "layer_nodes": len(lg),
                                "lr_overlap": len(ov),
                                "pct_lr_covered": round(pct, 1)})
        layer_ov_df = (pd.DataFrame(layer_rows)
                       .sort_values("lr_overlap", ascending=False)
                       .reset_index(drop=True))
        print(layer_ov_df.to_string(index=False))
        all_results["per_layer_lr_overlap"] = layer_ov_df
    else:
        print("  No layer data available.")

    # ── 5b. LR ↔ Reactome proteins ───────────────────────────────────────
    print(f"\n  [5b]  LR pair genes  ↔  Reactome/KEGG interaction proteins")
    print(f"{'─'*60}")
    note = ("(symbols)" if data["uniprot_to_symbol"]
            else "(UniProt IDs — provide --uniprot-map for symbol-level match)")
    print(f"  Reactome set is {note}")
    r2 = _overlap_report("LR genes", lr, "Reactome proteins", reactome)
    all_results["lr_vs_reactome"] = r2

    # ── 5c. LR ↔ KEGG genes ──────────────────────────────────────────────
    if kegg:
        print(f"\n  [5c]  LR pair genes  ↔  KEGG pathway genes")
        print(f"{'─'*60}")
        r3 = _overlap_report("LR genes", lr, "KEGG genes", kegg)
        all_results["lr_vs_kegg"] = r3

    # ── 5d. Layer genes ↔ Reactome proteins ──────────────────────────────
    if layers and reactome:
        print(f"\n  [5d]  Layer genes  ↔  Reactome/KEGG interaction proteins")
        print(f"{'─'*60}")
        r4 = _overlap_report("Layer genes", layers, "Reactome proteins", reactome)
        all_results["layers_vs_reactome"] = r4

    # ── 5e. Layer genes ↔ KEGG genes ────────────────────────────────────
    if layers and kegg:
        print(f"\n  [5e]  Layer genes  ↔  KEGG pathway genes")
        print(f"{'─'*60}")
        r5 = _overlap_report("Layer genes", layers, "KEGG genes", kegg)
        all_results["layers_vs_kegg"] = r5

    # ── 5f. Genes in layers but NOT in any pathway (blind spots) ─────────
    if layers:
        all_pathway_genes = reactome | kegg
        if all_pathway_genes:
            blind = layers - all_pathway_genes
            covered = layers & all_pathway_genes
            pct_cov = 100 * len(covered) / len(layers) if layers else 0
            print(f"\n  [5f]  Layer genes with NO pathway annotation")
            print(f"{'─'*60}")
            print(f"  Layer genes covered by ≥1 pathway database : "
                  f"{len(covered):>6,}  ({pct_cov:.1f}%)")
            print(f"  Layer genes with NO pathway annotation     : "
                  f"{len(blind):>6,}  ({100-pct_cov:.1f}%)")
            all_results["blind_genes"] = blind

    # ── 5g. LR genes absent from both layers AND pathways ────────────────
    if layers or reactome or kegg:
        all_known = layers | reactome | kegg
        lr_absent = lr - all_known
        lr_present = lr & all_known
        print(f"\n  [5g]  LR genes coverage across ALL sources")
        print(f"{'─'*60}")
        print(f"  LR genes found in ≥1 source (layers/Reactome/KEGG) : "
              f"{len(lr_present):>5,}  "
              f"({100*len(lr_present)/len(lr):.1f}% of LR genes)")
        print(f"  LR genes absent from ALL sources                   : "
              f"{len(lr_absent):>5,}  "
              f"({100*len(lr_absent)/len(lr):.1f}% of LR genes)")
        if lr_absent:
            print(f"\n  First 30 LR genes absent from every source:")
            for g in sorted(lr_absent)[:30]:
                print(f"    {g}")
        all_results["lr_absent_from_all"] = lr_absent

    data["overlap_results"] = all_results
    return all_results


# ============================================================================
# SECTION 6 — LAYER-LEVEL STATS
# ============================================================================

def audit_layers(data: dict) -> pd.DataFrame:
    _print_block("CELL-TYPE LAYER STATISTICS")

    if not data["layers"]:
        print("  No layer files loaded.")
        return pd.DataFrame()

    rows = []
    for ct, G in data["layers"].items():
        degs = [d for _, d in G.degree()]
        rows.append({
            "cell_type":  ct,
            "nodes":      G.number_of_nodes(),
            "edges":      G.number_of_edges(),
            "avg_degree": round(np.mean(degs), 2) if degs else 0,
            "max_degree": max(degs) if degs else 0,
            "density":    round(nx.density(G), 6),
        })

    layer_df = (pd.DataFrame(rows)
                .sort_values("edges", ascending=False)
                .reset_index(drop=True))
    print(layer_df.to_string(index=False))
    data["layer_df"] = layer_df
    return layer_df


# ============================================================================
# SECTION 7 — VISUALISATIONS
# ============================================================================

def make_plots(data: dict, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pathway_summary = data.get("pathway_summary", pd.DataFrame())
    layer_df        = data.get("layer_df",        pd.DataFrame())
    overlap_results = data.get("overlap_results",  {})
    lr_df           = data.get("lr_df",            pd.DataFrame())

    # ── Fig 1: Top pathways coloured by source ────────────────────────────
    if not pathway_summary.empty:
        top_n = min(30, len(pathway_summary))
        top   = pathway_summary.head(top_n)

        colors = []
        for _, row in top.iterrows():
            if row["in_reactome"] and row["in_kegg"]:
                colors.append("#6a4c93")
            elif row["in_kegg"]:
                colors.append("#f77f00")
            else:
                colors.append("#1982c4")

        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.38)))
        ax.barh(range(top_n), top["n_interactions"].values,
                color=colors, edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(top_n))
        labels = [
            (p[:65] + "…") if len(p) > 65 else p
            for p in top["common_pathway"]
        ]
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("Number of interactions", fontsize=10)
        ax.set_title(f"Top {top_n} shared pathways in merged interaction data",
                     fontsize=11, fontweight="bold")
        legend_patches = [
            mpatches.Patch(color="#1982c4", label="Reactome only"),
            mpatches.Patch(color="#f77f00", label="KEGG only"),
            mpatches.Patch(color="#6a4c93", label="Both"),
        ]
        ax.legend(handles=legend_patches, fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "fig1_top_pathways.pdf", bbox_inches="tight")
        plt.savefig(out / "fig1_top_pathways.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out / 'fig1_top_pathways.pdf'}")

    # ── Fig 2: Overlap bars (LR vs each source) ───────────────────────────
    overlap_items = []
    universes = data.get("universes", {})
    lr_size   = len(universes.get("lr", set()))

    label_map = {
        "lr_vs_layers":   "LR ↔ Layers",
        "lr_vs_reactome": "LR ↔ Reactome",
        "lr_vs_kegg":     "LR ↔ KEGG",
    }
    for key, label in label_map.items():
        if key in overlap_results:
            r = overlap_results[key]
            overlap_items.append({
                "comparison": label,
                "overlap":    len(r["overlap"]),
                "pct":        r["pct_a"],   # % of LR genes covered
            })

    if overlap_items:
        ov_df = pd.DataFrame(overlap_items)
        fig2, axes = plt.subplots(1, 2, figsize=(13, 4))

        axes[0].bar(ov_df["comparison"], ov_df["overlap"],
                    color=["#ff595e", "#1982c4", "#f77f00"][:len(ov_df)],
                    edgecolor="white")
        axes[0].set_ylabel("Genes overlapping")
        axes[0].set_title("LR gene overlap — absolute count", fontweight="bold")
        axes[0].set_xticklabels(ov_df["comparison"], rotation=15, ha="right")

        axes[1].bar(ov_df["comparison"], ov_df["pct"],
                    color=["#ff595e", "#1982c4", "#f77f00"][:len(ov_df)],
                    edgecolor="white")
        axes[1].set_ylabel("% of LR genes covered")
        axes[1].set_ylim(0, 105)
        axes[1].axhline(100, color="grey", lw=0.8, ls="--")
        axes[1].set_title("LR gene overlap — % of LR genes", fontweight="bold")
        axes[1].set_xticklabels(ov_df["comparison"], rotation=15, ha="right")

        fig2.suptitle("How well are LR-pair genes covered by each data source?",
                      fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out / "fig2_lr_overlap.pdf", bbox_inches="tight")
        plt.savefig(out / "fig2_lr_overlap.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved → {out / 'fig2_lr_overlap.pdf'}")

    # ── Fig 3: Layer sizes ────────────────────────────────────────────────
    if not layer_df.empty:
        n = len(layer_df)
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, max(4, n * 0.32 + 2)))

        axes3[0].barh(layer_df["cell_type"], layer_df["edges"],
                      color="#2dc653", edgecolor="white")
        axes3[0].set_xlabel("Edges")
        axes3[0].set_title("Edges per layer", fontweight="bold")
        axes3[0].invert_yaxis()

        axes3[1].barh(layer_df["cell_type"], layer_df["nodes"],
                      color="#ff595e", edgecolor="white")
        axes3[1].set_xlabel("Nodes")
        axes3[1].set_title("Nodes per layer", fontweight="bold")
        axes3[1].invert_yaxis()

        fig3.suptitle("Cell-Type Layer Statistics", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out / "fig3_layer_sizes.pdf", bbox_inches="tight")
        plt.savefig(out / "fig3_layer_sizes.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved → {out / 'fig3_layer_sizes.pdf'}")

    # ── Fig 4: Per-layer LR overlap ───────────────────────────────────────
    per_layer = overlap_results.get("per_layer_lr_overlap")
    if per_layer is not None and not per_layer.empty:
        n = len(per_layer)
        fig4, ax4 = plt.subplots(figsize=(10, max(4, n * 0.32 + 2)))

        ax4.barh(per_layer["cell_type"], per_layer["pct_lr_covered"],
                 color="#8ac926", edgecolor="white")
        ax4.set_xlabel("% of LR genes present in layer")
        ax4.set_title("LR gene coverage per cell-type layer",
                      fontweight="bold")
        ax4.axvline(50, color="grey", lw=0.8, ls="--", label="50%")
        ax4.legend(fontsize=8)
        ax4.invert_yaxis()

        plt.tight_layout()
        plt.savefig(out / "fig4_per_layer_lr_coverage.pdf", bbox_inches="tight")
        plt.savefig(out / "fig4_per_layer_lr_coverage.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig4)
        print(f"Saved → {out / 'fig4_per_layer_lr_coverage.pdf'}")

    # ── Fig 5: LR breakdown by type / pathway column ─────────────────────
    type_col    = data["lr_results"].get("col_type")
    pathway_col = data["lr_results"].get("col_pathway")
    if type_col or pathway_col:
        n_panels = sum([type_col is not None, pathway_col is not None])
        fig5, axes5 = plt.subplots(1, n_panels,
                                   figsize=(8 * n_panels, 6))
        if n_panels == 1:
            axes5 = [axes5]
        panel = 0
        for col, color in [(type_col, "#ffca3a"), (pathway_col, "#8ac926")]:
            if col is None:
                continue
            vc = lr_df[col].value_counts().head(20)
            axes5[panel].barh(vc.index, vc.values, color=color)
            axes5[panel].set_title(f"LR pairs by '{col}'", fontweight="bold")
            axes5[panel].invert_yaxis()
            axes5[panel].set_xlabel("Count")
            panel += 1
        fig5.suptitle("Ligand–Receptor Pair Breakdown", fontsize=11,
                      fontweight="bold")
        plt.tight_layout()
        plt.savefig(out / "fig5_lr_breakdown.pdf", bbox_inches="tight")
        plt.savefig(out / "fig5_lr_breakdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig5)
        print(f"Saved → {out / 'fig5_lr_breakdown.pdf'}")


# ============================================================================
# SECTION 8 — SAVE ALL AUDIT TABLES
# ============================================================================

def save_tables(data: dict, output_dir: str) -> None:
    _print_block("SAVING AUDIT TABLES")
    out = Path(output_dir)

    pathway_summary = data.get("pathway_summary", pd.DataFrame())
    overlap_results = data.get("overlap_results", {})
    layer_df        = data.get("layer_df",        pd.DataFrame())

    if not pathway_summary.empty:
        fp = out / "table_pathway_summary.csv"
        pathway_summary.to_csv(fp, index=False)
        print(f"  → {fp}")

        # Separate Reactome-only and KEGG-only
        for label, mask in [("reactome",
                              pathway_summary["in_reactome"] & ~pathway_summary["in_kegg"]),
                             ("kegg",
                              pathway_summary["in_kegg"] & ~pathway_summary["in_reactome"]),
                             ("both",
                              pathway_summary["in_reactome"] & pathway_summary["in_kegg"])]:
            fp2 = out / f"table_pathways_{label}_only.csv"
            pathway_summary[mask].to_csv(fp2, index=False)
            print(f"  → {fp2}  ({mask.sum()} pathways)")

    if not layer_df.empty:
        fp = out / "table_layer_stats.csv"
        layer_df.to_csv(fp, index=False)
        print(f"  → {fp}")

    per_layer = overlap_results.get("per_layer_lr_overlap")
    if per_layer is not None:
        fp = out / "table_per_layer_lr_overlap.csv"
        per_layer.to_csv(fp, index=False)
        print(f"  → {fp}")

    for key in ["lr_vs_layers", "lr_vs_reactome", "lr_vs_kegg"]:
        if key in overlap_results:
            absent = overlap_results[key]["only_a"]   # LR genes not in source
            fp = out / f"table_{key}_missing_lr_genes.txt"
            with open(fp, "w") as f:
                for g in sorted(absent):
                    f.write(g + "\n")
            print(f"  → {fp}  ({len(absent)} genes)")

    absent_all = overlap_results.get("lr_absent_from_all")
    if absent_all:
        fp = out / "table_lr_genes_absent_from_all_sources.txt"
        with open(fp, "w") as f:
            for g in sorted(absent_all):
                f.write(g + "\n")
        print(f"  → {fp}  ({len(absent_all)} genes)")

    blind = overlap_results.get("blind_genes")
    if blind:
        fp = out / "table_layer_genes_no_pathway.txt"
        with open(fp, "w") as f:
            for g in sorted(blind):
                f.write(g + "\n")
        print(f"  → {fp}  ({len(blind)} genes)")

    print()


# ============================================================================
# FINAL SUMMARY
# ============================================================================

def print_final_summary(data: dict) -> None:
    _print_block("FINAL AUDIT SUMMARY")

    universes = data.get("universes", {})
    overlap   = data.get("overlap_results", {})
    pathway_s = data.get("pathway_summary", pd.DataFrame())
    layer_df  = data.get("layer_df", pd.DataFrame())
    lr_df     = data.get("lr_df", pd.DataFrame())

    print(f"\n  LR pairs total            : {len(lr_df):,}")
    print(f"  LR unique genes           : {len(universes.get('lr', set())):,}")
    if not layer_df.empty:
        print(f"  Cell-type layers          : {len(layer_df)}")
        print(f"  Layer genes (union)       : {len(universes.get('layers', set())):,}")

    if not pathway_s.empty:
        print(f"\n  Unique pathways (merged)  : {len(pathway_s):,}")
        r = pathway_s["in_reactome"].sum()
        k = pathway_s["in_kegg"].sum()
        b = (pathway_s["in_reactome"] & pathway_s["in_kegg"]).sum()
        print(f"    Reactome-supported      : {r:,}")
        print(f"    KEGG-supported          : {k:,}")
        print(f"    Both databases          : {b:,}")

    print(f"\n  Overlap summary (% = share of LR genes covered):")
    for key, label in [("lr_vs_layers",   "LR ↔ Layers"),
                       ("lr_vs_reactome", "LR ↔ Reactome"),
                       ("lr_vs_kegg",     "LR ↔ KEGG")]:
        if key in overlap:
            r = overlap[key]
            print(f"    {label:<25}  {len(r['overlap']):>5,} genes  "
                  f"({r['pct_a']:.1f}% of LR)")

    absent_all = overlap.get("lr_absent_from_all", set())
    if absent_all:
        print(f"\n  ⚠  LR genes absent from ALL sources : {len(absent_all):,}")
        print(f"     (see table_lr_genes_absent_from_all_sources.txt)")

    blind = overlap.get("blind_genes", set())
    if blind:
        print(f"\n  ⚠  Layer genes with no pathway annotation : {len(blind):,}")
        print(f"     (see table_layer_genes_no_pathway.txt)")

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Audit pathway/connection sources in your MultiXrank multiplex"
    )
    parser.add_argument("--reactome-csv", required=True)
    parser.add_argument("--lr-csv",       required=True)
    parser.add_argument("--kegg-csv",     default=None)
    parser.add_argument("--layers-dir",   default=None)
    parser.add_argument("--uniprot-map",  default=None,
                        help="CSV with columns uniprotid, symbol — improves overlap matching")
    parser.add_argument("--output-dir",   default="audit_output")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = load_all(args.reactome_csv, args.lr_csv,
                    args.kegg_csv, args.layers_dir, args.uniprot_map)

    build_gene_universes(data)
    audit_layers(data)
    audit_reactome_pathways(data)
    audit_lr_pairs(data)
    overlap_analysis(data)
    save_tables(data, args.output_dir)
    make_plots(data, args.output_dir)
    print_final_summary(data)

    print(f"All outputs in: {Path(args.output_dir).resolve()}\n")


if __name__ == "__main__":
    main()