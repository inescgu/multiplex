#!/usr/bin/env python3
"""
Build Integrated Multiplex Interactome (Memory-Optimised + SPEED-Optimised)
============================================================================
Same science, same outputs, same statistics as the original script.
Three things were changed to make this faster, none of which touch what
counts as an edge or how it's counted:

  1. Pathway/KEGG CSV parsing no longer uses `df.iterrows()`. Iterrows
     rebuilds a Python object per row and is typically 20-100x slower than
     vectorised pandas for CSVs in the millions-of-rows range, which is
     exactly what Reactome interaction tables look like. Replaced with
     vectorised string ops + groupby/explode that produce byte-for-byte
     the same gene_to_pathways / pathway_to_genes indices.

  2. The pairwise pathway-edge generation (the O(pathway_size^2) part,
     which IS inherent to the science — you need every cross-layer pair
     within a shared pathway) is now optionally parallelised across
     worker processes with `--n-workers`. Chunks are processed in the
     same order as the sequential version and merged back in that same
     order, so edge attribution/statistics are identical to a sequential
     run — this is a pure speedup, not an approximation. Default
     --n-workers 1 reproduces the original sequential code path exactly.

  3. Repeated `G.nodes[nid]` dict-view lookups inside the hot loops are
     replaced with a plain-Python tuple lookup table built once up front,
     and edges are added in bulk instead of one `add_edge()` call at a
     time.

Everything else (semantics of cross_source_only, PPI validation, pathway
size filtering, LR edge logic, output files) is unchanged.
"""

import os
import gc
import json
import pickle
import time
import argparse
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import networkx as nx


# Globals used by pathway worker processes. Set right before Pool creation
# so that fork()'d children inherit them via copy-on-write (no pickling of
# the (potentially large) node_meta / gene_to_node_ids structures).
_W_NODE_META = None
_W_GENE_TO_NODE_IDS = None
_W_CROSS_SOURCE_ONLY = None
_W_PPI_REF = None
_W_LIGAND_NODE_IDS = None
_W_RECEPTOR_NODE_IDS = None

# ============================================================================
# SECTION 0 — LAYER LOADING (parallel I/O)
# ============================================================================

def _load_one_pkl(fpath: Path):
    with open(fpath, "rb") as f:
        g = pickle.load(f)
    mapping = {n: str(n).strip().upper() for n in g.nodes()}
    g = nx.relabel_nodes(g, mapping)
    cell_type = fpath.stem.replace("_layer", "")
    return cell_type, g


def load_base_ppi(ppi_file: str) -> nx.Graph:
    print(f"Loading base PPI from {ppi_file} ...")
    with open(ppi_file, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip().upper() for n in G.nodes()})
    print(f"  Base PPI: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")
    return G


def load_layers_from_folder(layers_dir: str, source_tag: str, n_workers: int = 1) -> dict:
    layers_dir = Path(layers_dir)
    pkl_files = sorted(layers_dir.glob("*_layer.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No *_layer.pkl files found in {layers_dir}")

    print(f"[{source_tag}] Found {len(pkl_files)} layer file(s)")

    layers = {}
    if n_workers > 1 and len(pkl_files) > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(min(n_workers, len(pkl_files))) as pool:
            for cell_type, g in pool.map(_load_one_pkl, pkl_files):
                layers[(source_tag, cell_type)] = g
                print(f"  {source_tag} | {cell_type}: "
                      f"{g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")
    else:
        for fpath in pkl_files:
            cell_type, g = _load_one_pkl(fpath)
            layers[(source_tag, cell_type)] = g
            print(f"  {source_tag} | {cell_type}: "
                  f"{g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    print(f"[{source_tag}] Loaded {len(layers)} layers\n")
    return layers


# ============================================================================
# SECTION 2 — NODE REGISTRY
# ============================================================================

class NodeRegistry:
    """Maps (source, cell_type, gene) tuples to integer IDs."""
    def __init__(self):
        self._tuple_to_id: dict[tuple, int] = {}
        self._id_to_tuple: list[tuple] = []

    def get_or_add(self, source: str, cell_type: str, gene: str) -> int:
        key = (source, cell_type, gene)
        idx = self._tuple_to_id.get(key)
        if idx is None:
            idx = len(self._id_to_tuple)
            self._tuple_to_id[key] = idx
            self._id_to_tuple.append(key)
        return idx

    def get_id(self, source: str, cell_type: str, gene: str) -> int | None:
        return self._tuple_to_id.get((source, cell_type, gene))

    def get_tuple(self, idx: int) -> tuple:
        return self._id_to_tuple[idx]

    def __len__(self):
        return len(self._id_to_tuple)

    def save(self, path: str):
        df = pd.DataFrame(self._id_to_tuple, columns=["source", "cell_type", "gene"])
        df.index.name = "node_id"
        df.to_csv(path)
        print(f"Saved node registry → {path}")


# ============================================================================
# SECTION 3 — PATHWAY INDEX (REACTOME) — vectorised
# ============================================================================

def load_uniprot_map(map_csv_path: str) -> dict:
    if not map_csv_path:
        return {}
    print(f"Loading UniProt→symbol map from {map_csv_path} ...")
    mapping_df = pd.read_csv(map_csv_path)
    u2s = dict(zip(mapping_df["uniprotid"], mapping_df["symbol"]))
    print(f"  {len(u2s):,} UniProt→symbol entries loaded\n")
    return u2s


def _vectorised_symbol_columns(df: pd.DataFrame, id_col1: str, id_col2: str,
                                use_symbol_cols: bool, u2s: dict) -> tuple[pd.Series, pd.Series]:
    """Reproduces the original per-row symbol-resolution logic, vectorised.

    Original per-row logic when NOT using symbol columns:
        sym = u2s.get(raw, u2s.get(raw.lower(), raw))   # raw is already upper()
    which is a case-insensitive lookup against u2s's own (mixed-case) keys,
    falling back to the raw (uppercased) id if nothing matches.
    """
    # See note in _explode_db_labels: fillna before astype(str) to avoid
    # pandas' nullable string dtype silently leaving NaNs as floats.
    col1 = df[id_col1].astype("object").fillna("").astype(str).str.strip().str.upper()
    col2 = df[id_col2].astype("object").fillna("").astype(str).str.strip().str.upper()

    if use_symbol_cols:
        return col1, col2

    # Build a case-insensitive lookup keyed on the UPPERCASED original key,
    # equivalent to trying both the raw (already-upper) key and its
    # lowercase form against the original dict.
    u2s_lookup: dict[str, str] = {}
    for k, v in u2s.items():
        ku = str(k).upper()
        u2s_lookup.setdefault(ku, v)

    sym1 = col1.map(u2s_lookup)
    sym2 = col2.map(u2s_lookup)
    sym1 = sym1.where(sym1.notna(), col1)
    sym2 = sym2.where(sym2.notna(), col2)
    sym1 = sym1.astype(str).str.strip().str.upper()
    sym2 = sym2.astype(str).str.strip().str.upper()
    return sym1, sym2


def _explode_db_labels(df: pd.DataFrame, source_col: str | None, db_label: str) -> pd.Series:
    """Vectorised replacement for the per-row pipe-split + fallback logic."""
    if source_col is None:
        return pd.Series([[db_label]] * len(df), index=df.index)

    # .fillna("") before .astype(str): on newer pandas (nullable "str" dtype),
    # astype(str) alone can leave missing values as float NaN instead of the
    # string "nan", which then breaks the downstream .str.split(). Explicit
    # fillna makes this robust across pandas versions.
    src = df[source_col].astype("object").fillna("").astype(str).str.strip()
    valid = (~src.str.lower().eq("nan")) & (src != "")
    split = src.where(valid, "").str.split("|")

    def _clean(lst):
        if not lst:
            return [db_label]
        cleaned = [s.strip() for s in lst if s.strip()]
        return cleaned if cleaned else [db_label]

    return split.map(_clean)


def _build_pathway_indices_from_long(long_df: pd.DataFrame) -> tuple[dict, dict]:
    """long_df has columns: pathway, db_label, gene (one row per gene mention)."""
    long_df = long_df[(long_df["gene"] != "") & (long_df["gene"].str.upper() != "NAN")]

    pathway_to_genes = {
        pkey: set(genes)
        for pkey, genes in long_df.groupby(["pathway", "db_label"])["gene"]
    }

    long_df = long_df.copy()
    long_df["pkey"] = list(zip(long_df["pathway"], long_df["db_label"]))
    gene_to_pathways = {
        gene: set(pkeys)
        for gene, pkeys in long_df.groupby("gene")["pkey"]
    }

    return gene_to_pathways, pathway_to_genes


def build_pathway_index_from_csv(
    csv_path: str,
    max_pathway_size: int = 300,
    db_label: str = "reactome",
    uniprot_to_symbol: dict = None,
) -> tuple[dict, dict, dict]:
    print(f"Loading pathway data from {csv_path} ...")

    df_temp = pd.read_csv(csv_path, nrows=5)
    cols = set(df_temp.columns)
    use_symbol_cols = {"symbol1", "symbol2"}.issubset(cols)
    source_col = "pathway_sources" if "pathway_sources" in cols else (
        "pathway_source" if "pathway_source" in cols else None)
    has_source_col = source_col is not None
    del df_temp

    id_cols = ["symbol1", "symbol2"] if use_symbol_cols else ["uniprot1", "uniprot2"]
    usecols = id_cols + ["common_pathway"]
    if has_source_col:
        usecols.append(source_col)

    df = pd.read_csv(csv_path, low_memory=False, usecols=usecols)

    pathway_df = df.dropna(subset=["common_pathway"]).copy()
    del df
    gc.collect()
    print(f"  {len(pathway_df):,} rows with common_pathway")
    print(f"  Using {'symbol1/symbol2 columns directly' if use_symbol_cols else 'uniprot1/uniprot2 + uniprot_to_symbol map'}")
    if has_source_col:
        print(f"  Pathway source column detected: '{source_col}'")
    else:
        print(f"  ⚠ No pathway source column found — all pathways will be labeled '{db_label}'")
    print()

    id_col1, id_col2 = id_cols
    u2s = uniprot_to_symbol or {}

    sym1, sym2 = _vectorised_symbol_columns(pathway_df, id_col1, id_col2, use_symbol_cols, u2s)
    pathway = pathway_df["common_pathway"].astype(str).str.strip()
    db_label_lists = _explode_db_labels(pathway_df, source_col, db_label)

    valid_mask = (pathway != "") & (pathway.str.lower() != "nan") & (sym1 != sym2)

    work = pd.DataFrame({
        "sym1": sym1[valid_mask].values,
        "sym2": sym2[valid_mask].values,
        "pathway": pathway[valid_mask].values,
        "db_labels": db_label_lists[valid_mask].values,
    })
    del pathway_df, sym1, sym2, pathway, db_label_lists
    gc.collect()

    work = work.explode("db_labels", ignore_index=True).rename(columns={"db_labels": "db_label"})

    long_df = pd.melt(work, id_vars=["pathway", "db_label"], value_vars=["sym1", "sym2"],
                       value_name="gene")[["pathway", "db_label", "gene"]]
    del work
    gc.collect()

    gene_to_pathways, pathway_to_genes = _build_pathway_indices_from_long(long_df)
    del long_df
    gc.collect()

    excluded_pathways = {"oversized": []}
    kept_pathway_to_genes = {}
    for pkey, g in pathway_to_genes.items():
        if len(g) > max_pathway_size:
            excluded_pathways["oversized"].append((pkey, len(g)))
        else:
            kept_pathway_to_genes[pkey] = g
    excluded_pathways["oversized"].sort(key=lambda x: -x[1])

    print(f"  Pathways before size filter : {len(pathway_to_genes):,}")
    print(f"  Pathways excluded (>{max_pathway_size} genes): "
          f"{len(excluded_pathways['oversized']):,}")
    print(f"  Pathways kept               : {len(kept_pathway_to_genes):,}")
    print(f"  Genes with pathway annotations: {len(gene_to_pathways):,}\n")

    return dict(gene_to_pathways), kept_pathway_to_genes, excluded_pathways


def validate_pathway_gene_overlap(
    gene_to_pathways: dict,
    gene_universe: set,
    label: str = "pathway",
    min_overlap_fraction: float = 0.05,
) -> None:
    pathway_genes = set(gene_to_pathways.keys())
    overlap = pathway_genes & gene_universe
    frac = len(overlap) / max(len(pathway_genes), 1)

    print(f"  [{label}] gene universe check: {len(pathway_genes):,} pathway-annotated genes, "
          f"{len(overlap):,} overlap with the {len(gene_universe):,} genes in your layers "
          f"({100*frac:.1f}%)")

    if frac < min_overlap_fraction:
        raise SystemExit(
            f"\n{'='*70}\n"
            f"ABORTING: {label} gene index has only {100*frac:.2f}% overlap with the "
            f"actual gene universe in your layers (threshold: {100*min_overlap_fraction:.0f}%).\n"
            f"This almost always means the ID space doesn't match — e.g. raw UniProt "
            f"accessions being compared against gene symbols with no translation applied. "
            f"Check that your --{label}-csv uses gene symbols (or pass --uniprot-map with a "
            f"correct 'uniprotid,symbol' CSV) before re-running this multi-hour job.\n"
            f"{'='*70}\n"
        )
    print()


# ============================================================================
# SECTION 3b — PATHWAY INDEX (KEGG) — vectorised
# ============================================================================

def build_kegg_pathway_index(
    kegg_csv_path: str,
    max_pathway_size: int = 300,
    uniprot_to_symbol: dict = None,
) -> tuple[dict, dict, dict]:
    print(f"Loading KEGG data from {kegg_csv_path} ...")
    df = pd.read_csv(kegg_csv_path, low_memory=False)

    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ("gene1", "protein1", "uniprot1", "gene_a", "symbol1"):
            col_map["gene1"] = col
        elif cl in ("gene2", "protein2", "uniprot2", "gene_b", "symbol2"):
            col_map["gene2"] = col
        elif cl in ("pathway", "pathway_name", "common_pathway", "kegg_pathway"):
            col_map["pathway"] = col

    missing = [k for k in ("gene1", "gene2", "pathway") if k not in col_map]
    if missing:
        raise ValueError(
            f"KEGG CSV missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.rename(columns={
        col_map["gene1"]: "gene1",
        col_map["gene2"]: "gene2",
        col_map["pathway"]: "pathway",
    })
    pathway_df = df.dropna(subset=["pathway"]).copy()
    del df
    gc.collect()
    print(f"  {len(pathway_df):,} rows with pathway annotation\n")

    u2s = uniprot_to_symbol or {}
    sym1, sym2 = _vectorised_symbol_columns(pathway_df, "gene1", "gene2", False, u2s)
    pathway = pathway_df["pathway"].astype(str).str.strip()

    valid_mask = (pathway != "") & (pathway.str.lower() != "nan") & (sym1 != sym2)

    long_df = pd.DataFrame({
        "pathway": pathway[valid_mask].values,
        "db_label": "kegg",
        "sym1": sym1[valid_mask].values,
        "sym2": sym2[valid_mask].values,
    })
    del pathway_df, sym1, sym2, pathway
    gc.collect()

    long_df = pd.melt(long_df, id_vars=["pathway", "db_label"], value_vars=["sym1", "sym2"],
                       value_name="gene")[["pathway", "db_label", "gene"]]

    gene_to_pathways, pathway_to_genes = _build_pathway_indices_from_long(long_df)
    del long_df
    gc.collect()

    excluded_pathways = {"oversized": []}
    kept_pathway_to_genes = {}
    for pkey, g in pathway_to_genes.items():
        if len(g) > max_pathway_size:
            excluded_pathways["oversized"].append((pkey, len(g)))
        else:
            kept_pathway_to_genes[pkey] = g
    excluded_pathways["oversized"].sort(key=lambda x: -x[1])

    print(f"  KEGG pathways before size filter : {len(pathway_to_genes):,}")
    print(f"  KEGG pathways excluded (>{max_pathway_size} genes): "
          f"{len(excluded_pathways['oversized']):,}")
    print(f"  KEGG pathways kept               : {len(kept_pathway_to_genes):,}")
    print(f"  Genes with KEGG annotations      : {len(gene_to_pathways):,}\n")

    return dict(gene_to_pathways), kept_pathway_to_genes, excluded_pathways


def merge_pathway_indices(
    reactome_g2p: dict, reactome_p2g: dict, reactome_excl: dict,
    kegg_g2p: dict, kegg_p2g: dict, kegg_excl: dict,
) -> tuple[dict, dict, dict]:
    merged_g2p: dict[str, set] = defaultdict(set)
    for gene, pkeys in reactome_g2p.items():
        merged_g2p[gene].update(pkeys)
    for gene, pkeys in kegg_g2p.items():
        merged_g2p[gene].update(pkeys)

    merged_p2g = {**reactome_p2g, **kegg_p2g}

    merged_excl = {
        "oversized": reactome_excl.get("oversized", []) + kegg_excl.get("oversized", [])
    }
    merged_excl["oversized"].sort(key=lambda x: -x[1])

    print(f"Merged pathway index: {len(merged_p2g):,} pathways, "
          f"{len(merged_g2p):,} genes\n")
    return dict(merged_g2p), merged_p2g, merged_excl


# ============================================================================
# SECTION 4 — LIGAND-RECEPTOR INDEX
# ============================================================================

def load_lr_pairs(lr_csv_path: str) -> tuple[set, set, set, dict]:
    print(f"Loading ligand-receptor pairs from {lr_csv_path} ...")
    df = pd.read_csv(lr_csv_path)
    if "ligand_gname" not in df.columns:
        df = pd.read_csv(lr_csv_path, sep="\t")

    df["ligand_gname"] = df["ligand_gname"].str.strip().str.upper()
    df["receptor_gname"] = df["receptor_gname"].str.strip().str.upper()
    df = df.dropna(subset=["ligand_gname", "receptor_gname"])

    lr_set = set(zip(df["ligand_gname"], df["receptor_gname"]))
    ligand_genes = set(df["ligand_gname"])
    receptor_genes = set(df["receptor_gname"])

    has_lig_desc = "ligand_description" in df.columns
    has_rec_desc = "receptor_description" in df.columns
    lr_descriptions = {
        (lig, rec): {
            "ligand_description": ld if has_lig_desc else "",
            "receptor_description": rd if has_rec_desc else "",
        }
        for lig, rec, ld, rd in zip(
            df["ligand_gname"], df["receptor_gname"],
            df["ligand_description"] if has_lig_desc else [""] * len(df),
            df["receptor_description"] if has_rec_desc else [""] * len(df),
        )
    }

    print(f"  {len(lr_set):,} LR pairs loaded")
    print(f"  {len(ligand_genes):,} unique ligands, "
          f"{len(receptor_genes):,} unique receptors\n")

    return lr_set, ligand_genes, receptor_genes, lr_descriptions


# ============================================================================
# SECTION 5 — MULTIPLEX ASSEMBLY
# ============================================================================


def _pathway_chunk_worker(chunk):
    """Processes one chunk of (pathway_key, genes) pairs. Returns new/annotated
    edges in the SAME order the sequential code would have produced them, plus
    stat deltas for the "skipped" counters (which don't depend on order)."""
    node_meta = _W_NODE_META
    gene_to_node_ids = _W_GENE_TO_NODE_IDS
    cross_source_only = _W_CROSS_SOURCE_ONLY
    ppi_ref = _W_PPI_REF

    out_edges = []
    skipped_same_layer = 0
    skipped_same_source = defaultdict(int)
    skipped_no_ppi = 0

    for (pathway_name, db_label), genes_in_path in chunk:
        node_ids_in_path = []
        for gene in genes_in_path:
            node_ids_in_path.extend(gene_to_node_ids.get(gene, ()))

        if len(node_ids_in_path) < 2:
            continue

        for ni, nj in combinations(node_ids_in_path, 2):
            si, ci, gi = node_meta[ni]
            sj, cj, gj = node_meta[nj]

            if si == sj and ci == cj:
                skipped_same_layer += 1
                continue

            if cross_source_only and si == sj:
                pair = tuple(sorted((si, sj)))
                skipped_same_source[pair] += 1
                continue

            if ppi_ref is not None and frozenset((gi, gj)) not in ppi_ref:
                skipped_no_ppi += 1
                continue

            src_pair = tuple(sorted((si, sj)))
            out_edges.append((ni, nj, pathway_name, db_label, src_pair))

    return out_edges, skipped_same_layer, dict(skipped_same_source), skipped_no_ppi

def _lr_chunk_worker(chunk):
    """chunk: list of ((lig_gene, rec_gene), desc) pairs. Returns candidate
    edges in the same order the sequential version would produce them, plus
    stat deltas. Mirrors _pathway_chunk_worker — the has_edge check and stats
    merge happen back in the main process, same as pathway edges."""
    node_meta = _W_NODE_META
    ligand_node_ids = _W_LIGAND_NODE_IDS
    receptor_node_ids = _W_RECEPTOR_NODE_IDS
    ppi_ref = _W_PPI_REF

    out_edges = []
    skipped_same_layer = 0
    skipped_no_ppi = 0

    for (lig_gene, rec_gene), desc in chunk:
        lig_nodes = ligand_node_ids.get(lig_gene, [])
        rec_nodes = receptor_node_ids.get(rec_gene, [])
        if not lig_nodes or not rec_nodes:
            continue

        for lig_nid in lig_nodes:
            sl, cl, _ = node_meta[lig_nid]
            for rec_nid in rec_nodes:
                sr, cr, _ = node_meta[rec_nid]

                if sl == sr and cl == cr:
                    skipped_same_layer += 1
                    continue

                if ppi_ref is not None and frozenset((lig_gene, rec_gene)) not in ppi_ref:
                    skipped_no_ppi += 1
                    continue

                src_pair = tuple(sorted((sl, sr)))
                out_edges.append((lig_nid, rec_nid, lig_gene, rec_gene, desc, src_pair))

    return out_edges, skipped_same_layer, skipped_no_ppi

def build_integrated_multiplex(
    ts_layers: dict,
    hdrg_layers: dict,
    gene_to_pathways: dict,
    pathway_to_genes: dict,
    lr_set: set,
    ligand_genes: set,
    receptor_genes: set,
    lr_descriptions: dict,
    cross_source_only: bool = True,
    max_pathway_size: int = 300,
    pathway_chunk_size: int = 50,
    lr_chunk_size: int = 200,          # <-- new
    ppi_reference_edges: set | None = None,
    n_workers: int = 1,
) -> tuple[nx.Graph, NodeRegistry, dict]:
    global _W_NODE_META, _W_GENE_TO_NODE_IDS, _W_CROSS_SOURCE_ONLY, _W_PPI_REF
    global _W_LIGAND_NODE_IDS, _W_RECEPTOR_NODE_IDS
    
    all_layers = {**ts_layers, **hdrg_layers}
    registry = NodeRegistry()
    G = nx.Graph()
    G.graph["description"] = "Integrated TS+hdrg multiplex (pathway + LR edges)"

    edge_stats = {
        "ppi_edges_by_layer": defaultdict(int),
        "pathway_edges_by_db_and_source": defaultdict(int),
        "pathway_edges_by_source_pair": defaultdict(int),
        "pathway_edges_by_pathway": defaultdict(int),
        "pathway_skipped_same_source": defaultdict(int),
        "pathway_skipped_same_layer": 0,
        "pathway_skipped_no_ppi_support": 0,
        "pathway_annotated_existing": defaultdict(int),
        "lr_edges_by_source_pair": defaultdict(int),
        "lr_pairs_no_ligand_nodes": [],
        "lr_pairs_no_receptor_nodes": [],
        "lr_skipped_same_layer": 0,
        "lr_skipped_no_ppi_support": 0,
        "intra_inter_by_edge_type": {
            "ppi": {"intra": 0, "inter": 0},
            "pathway": {"intra": 0, "inter": 0},
            "ligand_receptor": {"intra": 0, "inter": 0},
        },
        "edges_by_source_pair_and_type": defaultdict(lambda: defaultdict(int)),
    }

    # -----------------------------------------------------------------------
    # 5a. Register nodes + PPI edges (bulk add_edges_from)
    # -----------------------------------------------------------------------
    print("Adding intra-layer PPI nodes and edges ...")
    n_ppi = 0
    gene_to_node_ids: dict[str, list[int]] = defaultdict(list)
    node_meta: dict[int, tuple] = {}   # nid -> (source, cell_type, gene) — avoids G.nodes[] overhead in hot loops

    for (source, cell_type), layer in all_layers.items():
        node_attrs = {}
        for gene in layer.nodes():
            nid = registry.get_or_add(source, cell_type, gene)
            node_attrs[nid] = {"source": source, "cell_type": cell_type, "gene": gene}
            node_meta[nid] = (source, cell_type, gene)
            gene_to_node_ids[gene].append(nid)
        G.add_nodes_from(node_attrs.items())

        edge_bunch = []
        layer_key = (source, cell_type)
        for u, v in layer.edges():
            uid = registry.get_id(source, cell_type, u)
            vid = registry.get_id(source, cell_type, v)
            if uid is not None and vid is not None:
                edge_bunch.append((uid, vid, {"edge_type": "ppi", "source": source, "cell_type": cell_type}))

        G.add_edges_from(edge_bunch)
        n_this = len(edge_bunch)
        n_ppi += n_this
        edge_stats["ppi_edges_by_layer"][layer_key] += n_this
        edge_stats["intra_inter_by_edge_type"]["ppi"]["intra"] += n_this
        edge_stats["edges_by_source_pair_and_type"][(source, source)]["ppi"] += n_this

    print(f"  Nodes     : {G.number_of_nodes():,}")
    print(f"  PPI edges : {n_ppi:,}\n")
    gc.collect()

    # -----------------------------------------------------------------------
    # 5b. Pathway edges — chunked, optionally parallel
    # -----------------------------------------------------------------------
    print("Adding pathway inter-layer edges (chunked) ...")
    n_pathway = 0
    pathways = list(pathway_to_genes.items())
    chunks = [pathways[i:i + pathway_chunk_size] for i in range(0, len(pathways), pathway_chunk_size)]

    def _merge_edges(out_edges, s_same_layer, s_same_source, s_no_ppi):
        nonlocal n_pathway
        edge_stats["pathway_skipped_same_layer"] += s_same_layer
        edge_stats["pathway_skipped_no_ppi_support"] += s_no_ppi
        for pair, cnt in s_same_source.items():
            edge_stats["pathway_skipped_same_source"][pair] += cnt

        for ni, nj, pathway_name, db_label, src_pair in out_edges:
            db_src_key = (db_label,) + src_pair
            if G.has_edge(ni, nj):
                ed = G[ni][nj]
                if "pathways" not in ed:
                    ed["pathways"] = set()
                ed["pathways"].add((pathway_name, db_label))
                edge_stats["pathway_annotated_existing"][src_pair] += 1
            else:
                G.add_edge(ni, nj, edge_type="pathway", pathway=pathway_name,
                           pathway_db=db_label, pathways={(pathway_name, db_label)})
                n_pathway += 1
                edge_stats["pathway_edges_by_source_pair"][src_pair] += 1
                edge_stats["pathway_edges_by_db_and_source"][db_src_key] += 1
                edge_stats["pathway_edges_by_pathway"][(pathway_name, db_label)] += 1

                di_source, dj_source = src_pair
                is_inter = di_source != dj_source
                key = "inter" if is_inter else "intra"
                edge_stats["intra_inter_by_edge_type"]["pathway"][key] += 1
                edge_stats["edges_by_source_pair_and_type"][src_pair]["pathway"] += 1

    if n_workers > 1 and len(chunks) > 1:
        _W_NODE_META = node_meta
        _W_GENE_TO_NODE_IDS = dict(gene_to_node_ids)
        _W_CROSS_SOURCE_ONLY = cross_source_only
        _W_PPI_REF = ppi_reference_edges

        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            # imap preserves submission order, so the merge below reproduces
            # exactly the same edge/attribution order as the sequential path.
            for i, result in enumerate(pool.imap(_pathway_chunk_worker, chunks)):
                _merge_edges(*result)
                if i % 20 == 0:
                    done = min((i + 1) * pathway_chunk_size, len(pathways))
                    print(f"  Processed ~{done:,}/{len(pathways):,} pathways | "
                          f"pathway edges so far: {n_pathway:,}")
    else:
        _W_NODE_META = node_meta
        _W_GENE_TO_NODE_IDS = gene_to_node_ids
        _W_CROSS_SOURCE_ONLY = cross_source_only
        _W_PPI_REF = ppi_reference_edges
        for i, chunk in enumerate(chunks):
            result = _pathway_chunk_worker(chunk)
            _merge_edges(*result)
            if i % 20 == 0:
                done = min((i + 1) * pathway_chunk_size, len(pathways))
                print(f"  Processed {done:,}/{len(pathways):,} pathways | "
                      f"pathway edges so far: {n_pathway:,}")

    print(f"  Pathway edges added : {n_pathway:,}")
    if ppi_reference_edges is not None:
        print(f"  Pathway edges skipped (no PPI support) : "
              f"{edge_stats['pathway_skipped_no_ppi_support']:,}")
    print()
    gc.collect()

    # -----------------------------------------------------------------------
    # 5c. Ligand-Receptor edges — chunked, optionally parallel
    # -----------------------------------------------------------------------
    print("Adding ligand-receptor inter-layer edges (chunked) ...")
    n_lr = 0

    ligand_node_ids = {g: gene_to_node_ids[g] for g in ligand_genes if g in gene_to_node_ids}
    receptor_node_ids = {g: gene_to_node_ids[g] for g in receptor_genes if g in gene_to_node_ids}

    edge_stats["lr_pairs_no_ligand_nodes"] = [
        str(g) for g in ligand_genes if g not in gene_to_node_ids
    ]
    edge_stats["lr_pairs_no_receptor_nodes"] = [
        str(g) for g in receptor_genes if g not in gene_to_node_ids
    ]

    def _merge_lr_edges(out_edges, s_same_layer, s_no_ppi):
        nonlocal n_lr
        edge_stats["lr_skipped_same_layer"] += s_same_layer
        edge_stats["lr_skipped_no_ppi_support"] += s_no_ppi

        for lig_nid, rec_nid, lig_gene, rec_gene, desc, src_pair in out_edges:
            if not G.has_edge(lig_nid, rec_nid):
                G.add_edge(lig_nid, rec_nid, edge_type="ligand_receptor",
                           ligand_gene=lig_gene, receptor_gene=rec_gene, **desc)
                n_lr += 1
                edge_stats["lr_edges_by_source_pair"][src_pair] += 1
                is_inter = src_pair[0] != src_pair[1]
                key = "inter" if is_inter else "intra"
                edge_stats["intra_inter_by_edge_type"]["ligand_receptor"][key] += 1
                edge_stats["edges_by_source_pair_and_type"][src_pair]["ligand_receptor"] += 1
            else:
                G[lig_nid][rec_nid]["lr_pair"] = (lig_gene, rec_gene)

    lr_items = list(lr_descriptions.items())
    lr_chunks = [lr_items[i:i + lr_chunk_size] for i in range(0, len(lr_items), lr_chunk_size)]

    if n_workers > 1 and len(lr_chunks) > 1:
        _W_LIGAND_NODE_IDS = ligand_node_ids
        _W_RECEPTOR_NODE_IDS = receptor_node_ids
        # _W_NODE_META and _W_PPI_REF are already set from the pathway step
        # and haven't changed, so we can reuse them as-is.

        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap(_lr_chunk_worker, lr_chunks)):
                _merge_lr_edges(*result)
                if i % 20 == 0:
                    done = min((i + 1) * lr_chunk_size, len(lr_items))
                    print(f"  Processed ~{done:,}/{len(lr_items):,} LR pairs | "
                          f"LR edges so far: {n_lr:,}")
    else:
        _W_LIGAND_NODE_IDS = ligand_node_ids
        _W_RECEPTOR_NODE_IDS = receptor_node_ids
        for i, chunk in enumerate(lr_chunks):
            result = _lr_chunk_worker(chunk)
            _merge_lr_edges(*result)
            if i % 20 == 0:
                done = min((i + 1) * lr_chunk_size, len(lr_items))
                print(f"  Processed {done:,}/{len(lr_items):,} LR pairs | "
                      f"LR edges so far: {n_lr:,}")

    print(f"  LR edges added : {n_lr:,}")
    if ppi_reference_edges is not None:
        print(f"  LR edges skipped (no PPI support) : {edge_stats['lr_skipped_no_ppi_support']:,}")
    print()
    gc.collect()

    # -----------------------------------------------------------------------
    # 5d. Summary
    # -----------------------------------------------------------------------
    edge_types = defaultdict(int)
    for _, _, d in G.edges(data=True):
        edge_types[d.get("edge_type", "unknown")] += 1

    print("=" * 60)
    print("INTEGRATED MULTIPLEX SUMMARY")
    print("=" * 60)
    print(f"  Total nodes : {G.number_of_nodes():,}")
    print(f"  Total edges : {G.number_of_edges():,}")
    for etype, count in sorted(edge_types.items()):
        print(f"    {etype:<20}: {count:,}")
    print("=" * 60 + "\n")

    return G, registry, edge_stats


# ============================================================================
# SECTION 6 — STATISTICS REPORT (unchanged logic)
# ============================================================================

def write_statistics_report(edge_stats: dict, excluded_pathways: dict, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ppi_rows = [
        {"source": src, "cell_type": ct, "n_ppi_edges": count}
        for (src, ct), count in sorted(edge_stats["ppi_edges_by_layer"].items(), key=lambda x: -x[1])
    ]
    pd.DataFrame(ppi_rows).to_csv(out / "stats_ppi_edges_by_layer.csv", index=False)

    pathway_src_rows = [
        {"source_pair": " <-> ".join(pair), "n_pathway_edges": count}
        for pair, count in sorted(edge_stats["pathway_edges_by_source_pair"].items(), key=lambda x: -x[1])
    ]
    pd.DataFrame(pathway_src_rows).to_csv(out / "stats_pathway_edges_by_source_pair.csv", index=False)

    db_src_rows = [
        {"db": key[0], "source_pair": " <-> ".join(key[1:]), "n_pathway_edges": count}
        for key, count in sorted(edge_stats["pathway_edges_by_db_and_source"].items(), key=lambda x: -x[1])
    ]
    pd.DataFrame(db_src_rows).to_csv(out / "stats_pathway_edges_by_db_and_source.csv", index=False)

    pathway_contrib_rows = [
        {"pathway": p, "db": db, "n_edges_contributed": c}
        for (p, db), c in sorted(edge_stats["pathway_edges_by_pathway"].items(), key=lambda x: -x[1])
    ]
    pd.DataFrame(pathway_contrib_rows).to_csv(out / "stats_pathway_edges_by_pathway.csv", index=False)

    excl_rows = [
        {"pathway": p, "db": db, "n_genes": n, "reason": "exceeds max_pathway_size"}
        for (p, db), n in excluded_pathways.get("oversized", [])
    ]
    pd.DataFrame(excl_rows).to_csv(out / "stats_excluded_pathways.csv", index=False)

    skipped_rows = []
    for pair, count in sorted(edge_stats["pathway_skipped_same_source"].items(), key=lambda x: -x[1]):
        skipped_rows.append({"source_pair": " <-> ".join(pair), "n_skipped": count,
                              "reason": "same source (cross_source_only=True)"})
    skipped_rows.append({"source_pair": "same layer",
                          "n_skipped": edge_stats["pathway_skipped_same_layer"],
                          "reason": "same layer (intra-layer pathway edges disabled)"})
    pd.DataFrame(skipped_rows).to_csv(out / "stats_pathway_edges_skipped.csv", index=False)

    annotated_rows = [
        {"source_pair": " <-> ".join(pair), "n_annotated": count}
        for pair, count in sorted(edge_stats["pathway_annotated_existing"].items(), key=lambda x: -x[1])
    ]
    pd.DataFrame(annotated_rows).to_csv(out / "stats_pathway_annotated_existing_edges.csv", index=False)

    lr_src_rows = [
        {"source_pair": " <-> ".join(pair), "n_lr_edges": count}
        for pair, count in sorted(edge_stats["lr_edges_by_source_pair"].items(), key=lambda x: -x[1])
    ]
    pd.DataFrame(lr_src_rows).to_csv(out / "stats_lr_edges_by_source_pair.csv", index=False)

    absent_rows = (
        [{"gene": g, "role": "ligand", "reason": "not found in any layer"}
         for g in sorted(edge_stats["lr_pairs_no_ligand_nodes"])]
        + [{"gene": g, "role": "receptor", "reason": "not found in any layer"}
           for g in sorted(edge_stats["lr_pairs_no_receptor_nodes"])]
    )
    pd.DataFrame(absent_rows).to_csv(out / "stats_lr_genes_absent_from_graph.csv", index=False)

    intra_inter_rows = []
    for etype, counts in edge_stats["intra_inter_by_edge_type"].items():
        total = counts["intra"] + counts["inter"]
        intra_inter_rows.append({"edge_type": etype, "intra_layer": counts["intra"],
                                  "inter_layer": counts["inter"], "total": total})
    pd.DataFrame(intra_inter_rows).to_csv(out / "stats_intra_vs_inter_layer.csv", index=False)

    src_type_rows = []
    for src_pair, type_counts in sorted(edge_stats["edges_by_source_pair_and_type"].items()):
        row = {"source_pair": " <-> ".join(src_pair)}
        row.update(type_counts)
        src_type_rows.append(row)
    pd.DataFrame(src_type_rows).fillna(0).to_csv(out / "stats_edges_by_source_pair_and_type.csv", index=False)

    lines = []
    lines.append("=" * 70)
    lines.append("EDGE STATISTICS REPORT")
    lines.append("=" * 70)

    lines.append("\n--- PPI EDGES BY LAYER ---")
    for r in ppi_rows:
        lines.append(f"  {r['source']:<25} {r['cell_type']:<30} {r['n_ppi_edges']:>10,} edges")

    lines.append("\n--- INTRA vs INTER LAYER BREAKDOWN ---")
    for r in intra_inter_rows:
        lines.append(f"  {r['edge_type']:<20} intra={r['intra_layer']:>10,}  "
                      f"inter={r['inter_layer']:>10,}  total={r['total']:>10,}")

    lines.append("\n--- EDGES BY SOURCE PAIR AND TYPE ---")
    for r in src_type_rows:
        parts = [f"{k}={v:,}" for k, v in r.items() if k != "source_pair"]
        lines.append(f"  {r['source_pair']:<40}  " + "  ".join(parts))

    lines.append("\n--- PATHWAY EDGES BY SOURCE PAIR (all DBs) ---")
    for r in pathway_src_rows:
        lines.append(f"  {r['source_pair']:<40} {r['n_pathway_edges']:>10,} edges")

    lines.append("\n--- PATHWAY EDGES BY DATABASE AND SOURCE PAIR ---")
    for r in db_src_rows:
        lines.append(f"  [{r['db']:<10}] {r['source_pair']:<35} {r['n_pathway_edges']:>10,} edges")

    lines.append("\n--- PATHWAY EDGES ANNOTATED ONTO EXISTING EDGES ---")
    for r in annotated_rows:
        lines.append(f"  {r['source_pair']:<40} {r['n_annotated']:>10,}")

    lines.append("\n--- PATHWAY EDGES SKIPPED ---")
    for r in skipped_rows:
        lines.append(f"  {r['source_pair']:<30} {r['n_skipped']:>10,}  ({r['reason']})")

    lines.append("\n--- EXCLUDED PATHWAYS (size filter) ---")
    lines.append(f"  Total excluded: {len(excl_rows):,}")
    if excl_rows:
        lines.append("  Top 20 largest excluded pathways:")
        for r in excl_rows[:20]:
            lines.append(f"    [{r['db']:<10}] {r['pathway']:<50} {r['n_genes']:>6} genes")
    lines.append("  → Full list saved to stats_excluded_pathways.csv")

    lines.append("\n--- TOP 20 PATHWAYS BY EDGES CONTRIBUTED ---")
    for r in pathway_contrib_rows[:20]:
        lines.append(f"  [{r['db']:<10}] {r['pathway']:<50} {r['n_edges_contributed']:>8,} edges")
    lines.append("  → Full list saved to stats_pathway_edges_by_pathway.csv")

    lines.append("\n--- LR EDGES BY SOURCE PAIR ---")
    for r in lr_src_rows:
        lines.append(f"  {r['source_pair']:<40} {r['n_lr_edges']:>10,} edges")
    lines.append(f"  LR edges skipped (same layer): {edge_stats['lr_skipped_same_layer']:,}")

    lines.append("\n--- LR GENES ABSENT FROM GRAPH ---")
    lines.append(f"  Ligands  not found in any layer: {len(edge_stats['lr_pairs_no_ligand_nodes']):,}")
    lines.append(f"  Receptors not found in any layer: {len(edge_stats['lr_pairs_no_receptor_nodes']):,}")
    lines.append("  → Full list saved to stats_lr_genes_absent_from_graph.csv")

    lines.append("\n" + "=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    report_path = out / "edge_statistics_report.txt"
    report_path.write_text(report_text)
    print(f"\nStatistics report saved → {report_path}")


# ============================================================================
# SECTION 7 — SAVE & LAYER STATISTICS
# ============================================================================

def save_multiplex(G: nx.Graph, registry: NodeRegistry, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    registry.save(str(out / "node_registry.csv"))

    graph_path = out / "integrated_multiplex.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved multiplex graph → {graph_path}")

    # Full edge-list CSV export is intentionally disabled — for a graph this
    # size it exceeds available memory and isn't needed downstream. We still
    # want per-type edge counts for the summary JSON, so tally them directly
    # off the graph instead of building per-edge buffers.
    edge_type_counts = defaultdict(int)
    for _, _, d in G.edges(data=True):
        et = d.get("edge_type", "") or "unknown"
        edge_type_counts[et] += 1

    summary = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "edge_type_counts": dict(edge_type_counts),
    }
    with open(out / "integrated_multiplex_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON → {out / 'integrated_multiplex_summary.json'}\n")

def layer_statistics(G: nx.Graph, registry: NodeRegistry) -> pd.DataFrame:
    layer_nodes: dict[tuple, set] = defaultdict(set)
    for nid, d in G.nodes(data=True):
        layer_nodes[(d["source"], d["cell_type"])].add(nid)

    rows = []
    for (source, ct), node_set in sorted(layer_nodes.items()):
        sub = G.subgraph(node_set)
        n_ppi = sum(1 for _, _, d in sub.edges(data=True) if d.get("edge_type") == "ppi")
        n_path = sum(1 for _, _, d in G.edges(node_set, data=True) if d.get("edge_type") == "pathway")
        n_lr = sum(1 for _, _, d in G.edges(node_set, data=True) if d.get("edge_type") == "ligand_receptor")
        rows.append({"source": source, "cell_type": ct, "n_nodes": len(node_set),
                      "n_ppi": n_ppi, "n_pathway": n_path, "n_lr": n_lr})
    return pd.DataFrame(rows)


# ============================================================================
# SECTION 8 — MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build integrated TS+hdrg multiplex (pathway + LR, speed-optimised)"
    )
    parser.add_argument("--ts-layers-dir", required=True)
    parser.add_argument("--hdrg-layers-dir", required=True)
    parser.add_argument("--reactome-csv", required=None)
    parser.add_argument("--kegg-csv", default=None)
    parser.add_argument("--uniprot-map", default=None)
    parser.add_argument("--lr-csv", required=True)
    parser.add_argument("--output-dir", default="integrated_multiplex_output")
    parser.add_argument("--max-pathway-size", type=int, default=300)
    parser.add_argument("--pathway-chunk-size", type=int, default=50)
    parser.add_argument("--cross-source-only", action="store_true")
    parser.add_argument("--base-ppi", default=None)
    parser.add_argument("--validate-against-ppi", action="store_true")
    parser.add_argument("--ppi-validation-pkl", default=None)
    parser.add_argument("--n-workers", type=int, default=1,
                         help="Worker processes for parallel layer loading and pathway-edge "
                              "assembly. Set to your --cpus-per-task. Default 1 (sequential, "
                              "identical to the original single-threaded code path).")
    parser.add_argument("--skip-pathways", action="store_true",
                    help="Skip Reactome/KEGG pathway edges, only build LR edges")
    parser.add_argument("--lr-chunk-size", type=int, default=200)

    
    args = parser.parse_args()

    t0 = time.time()

    print("\n" + "=" * 60)
    print("STEP 1: LOADING LAYERS")
    print("=" * 60)
    ts_layers = load_layers_from_folder(args.ts_layers_dir, "tabula_sapiens", n_workers=args.n_workers)
    hdrg_layers = load_layers_from_folder(args.hdrg_layers_dir, "hdrg", n_workers=args.n_workers)

    base_ppi_graph = None
    if args.base_ppi:
        print("Loading full base PPI layer ...")
        base_ppi_graph = load_base_ppi(args.base_ppi)
        hdrg_layers[("global_ppi", "full_network")] = base_ppi_graph
        print("  Added base PPI as layer: source='global_ppi', cell_type='full_network'\n")

    ppi_reference_edges = None
    if args.validate_against_ppi:
        validation_path = args.ppi_validation_pkl or args.base_ppi
        if not validation_path:
            parser.error("--validate-against-ppi requires --base-ppi or --ppi-validation-pkl")
        if args.ppi_validation_pkl and args.ppi_validation_pkl != args.base_ppi:
            ppi_for_validation = load_base_ppi(args.ppi_validation_pkl)
        elif base_ppi_graph is not None:
            ppi_for_validation = base_ppi_graph
        else:
            ppi_for_validation = load_base_ppi(validation_path)
        ppi_reference_edges = {frozenset((u, v)) for u, v in ppi_for_validation.edges()}
        print(f"PPI validation ENABLED: {len(ppi_reference_edges):,} reference PPI edges "
              f"loaded from {validation_path}")
        print("  Pathway/Reactome and ligand-receptor edges will only be added if the same "
              "gene pair is also a PPI edge.\n")

    print("=" * 60)
    print("STEP 2: PATHWAY INDEX")
    print("=" * 60)
    uniprot_to_symbol = load_uniprot_map(args.uniprot_map)

    if args.skip_pathways:
        gene_to_pathways = {}
        pathway_to_genes = {}
        excluded_pathways = {}
        print("⚠ SKIPPING PATHWAY ASSEMBLY — LR edges only\n")
    else:
        reactome_g2p, reactome_p2g, reactome_excl = build_pathway_index_from_csv(
            args.reactome_csv,
            max_pathway_size=args.max_pathway_size,
            db_label="reactome",
            uniprot_to_symbol=uniprot_to_symbol,
        )

        if args.kegg_csv:
            kegg_g2p, kegg_p2g, kegg_excl = build_kegg_pathway_index(
                args.kegg_csv,
                max_pathway_size=args.max_pathway_size,
                uniprot_to_symbol=uniprot_to_symbol,
            )
            gene_to_pathways, pathway_to_genes, excluded_pathways = merge_pathway_indices(
                reactome_g2p, reactome_p2g, reactome_excl,
                kegg_g2p, kegg_p2g, kegg_excl,
            )
        else:
            print("No KEGG CSV provided — using Reactome only.\n")
            gene_to_pathways = reactome_g2p
            pathway_to_genes = reactome_p2g
            excluded_pathways = reactome_excl

        gene_universe_check = set()
        for g in ts_layers.values():
            gene_universe_check.update(g.nodes())
        for g in hdrg_layers.values():
            gene_universe_check.update(g.nodes())
        print("Sanity-checking pathway gene overlap against layer gene universe ...")
        validate_pathway_gene_overlap(gene_to_pathways, gene_universe_check, label="pathway")

    print("=" * 60)
    print("STEP 3: LIGAND-RECEPTOR PAIRS")
    print("=" * 60)
    lr_set, ligand_genes, receptor_genes, lr_descriptions = load_lr_pairs(args.lr_csv)

    print("=" * 60)
    print("STEP 4: ASSEMBLING MULTIPLEX")
    print("=" * 60)
    G, registry, edge_stats = build_integrated_multiplex(
        ts_layers, hdrg_layers,
        gene_to_pathways, pathway_to_genes,
        lr_set, ligand_genes, receptor_genes, lr_descriptions,
        cross_source_only=args.cross_source_only,
        max_pathway_size=args.max_pathway_size,
        pathway_chunk_size=args.pathway_chunk_size,
        ppi_reference_edges=ppi_reference_edges,
        n_workers=args.n_workers,
    )

    print("=" * 60)
    print("STEP 5: SAVING OUTPUTS")
    print("=" * 60)
    save_multiplex(G, registry, args.output_dir)

    stats_df = layer_statistics(G, registry)
    stats_path = os.path.join(args.output_dir, "layer_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved layer stats → {stats_path}")
    print(stats_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 6: STATISTICS REPORT")
    print("=" * 60)
    write_statistics_report(edge_stats, excluded_pathways, args.output_dir)

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()