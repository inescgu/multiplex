#!/usr/bin/env python3
"""
Build Integrated Multiplex Interactome (Memory-Optimised)
==========================================================
- Intra-layer PPI edges
- Pathway inter-layer edges (TS ↔ hdrg only, chunked, pathway size capped)
  Sources: Reactome + KEGG
- Ligand-receptor inter-layer edges

Node keys: integer IDs (lookup table saved separately)
"""

import os
import gc
import json
import pickle
import time
import argparse
from collections import defaultdict
from pathlib import Path
from itertools import combinations

import pandas as pd
import networkx as nx

def load_base_ppi(ppi_file: str) -> nx.Graph:
    """Load full base PPI network and normalise node names to uppercase."""
    print(f"Loading base PPI from {ppi_file} ...")
    with open(ppi_file, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip().upper() for n in G.nodes()})
    print(f"  Base PPI: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")
    return G

# ============================================================================
# SECTION 1 — LAYER LOADING
# ============================================================================

def load_layers_from_folder(layers_dir: str, source_tag: str) -> dict:
    layers_dir = Path(layers_dir)
    pkl_files  = sorted(layers_dir.glob("*_layer.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No *_layer.pkl files found in {layers_dir}")

    print(f"[{source_tag}] Found {len(pkl_files)} layer file(s)")
    layers = {}
    for fpath in pkl_files:
        cell_type = fpath.stem.replace("_layer", "")
        with open(fpath, "rb") as f:
            g = pickle.load(f)
        mapping = {n: str(n).strip().upper() for n in g.nodes()}
        g = nx.relabel_nodes(g, mapping)
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
        self._id_to_tuple: list[tuple]      = []

    def get_or_add(self, source: str, cell_type: str, gene: str) -> int:
        key = (source, cell_type, gene)
        if key not in self._tuple_to_id:
            idx = len(self._id_to_tuple)
            self._tuple_to_id[key] = idx
            self._id_to_tuple.append(key)
        return self._tuple_to_id[key]

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
# SECTION 3 — PATHWAY INDEX (REACTOME)
# ============================================================================

def load_uniprot_map(map_csv_path: str) -> dict:
    """Load UniProt→gene-symbol mapping. Returns empty dict if path is None."""
    if not map_csv_path:
        return {}
    print(f"Loading UniProt→symbol map from {map_csv_path} ...")
    mapping_df = pd.read_csv(map_csv_path)
    u2s = dict(zip(mapping_df["uniprotid"], mapping_df["symbol"]))
    print(f"  {len(u2s):,} UniProt→symbol entries loaded\n")
    return u2s


def build_pathway_index_from_csv(
    csv_path: str,
    max_pathway_size: int = 300,
    db_label: str = "reactome", # Default fallback if column missing
    uniprot_to_symbol: dict = None,
) -> tuple[dict, dict, dict]:
    """
    Returns
    -------
    gene_to_pathways   : gene -> set of (pathway_name, db_label)
    pathway_to_genes   : (pathway_name, db_label) -> set of gene symbols
    excluded_pathways  : dict with keys 'oversized' -> list of (pathway, size)

    Column handling
    ---------------
    Prefers 'symbol1'/'symbol2' columns directly if the CSV has them — these
    are already gene symbols and need no translation, so this is the most
    robust path. Only falls back to 'uniprot1'/'uniprot2' + a uniprot_to_symbol
    map if symbol columns aren't present. (Silently trusting raw UniProt IDs
    as if they were gene symbols, with no mapping applied, produces a graph
    with zero real pathway edges — this is what happened before this fix.)
    """
    print(f"Loading pathway data from {csv_path} ...")

    df_temp = pd.read_csv(csv_path, nrows=5)
    cols = set(df_temp.columns)
    use_symbol_cols = {"symbol1", "symbol2"}.issubset(cols)
    # Accept both 'pathway_source' (singular) and 'pathway_sources' (plural)
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

    gene_to_pathways: dict[str, set] = defaultdict(set)
    # Key is now (pathway_name, specific_db_label)
    pathway_to_genes: dict[tuple, set] = defaultdict(set)

    u2s = uniprot_to_symbol or {}
    id_col1, id_col2 = id_cols

    for _, row in pathway_df.iterrows():
        raw1    = str(row[id_col1]).strip().upper()
        raw2    = str(row[id_col2]).strip().upper()
        pathway = str(row["common_pathway"]).strip()

        if not pathway or pathway.lower() == "nan":
            continue

        if use_symbol_cols:
            # Already gene symbols — no translation needed.
            sym1, sym2 = raw1, raw2
        else:
            # Translate UniProt → symbol via mapping (falls back to raw ID
            # — and raw untranslated UniProt IDs) if no mapping provided.
            sym1 = u2s.get(raw1, u2s.get(raw1.lower(), raw1))
            sym2 = u2s.get(raw2, u2s.get(raw2.lower(), raw2))
            sym1 = sym1.strip().upper() if sym1 else raw1
            sym2 = sym2.strip().upper() if sym2 else raw2

        # Skip self-loops
        if sym1 == sym2:
            continue

        # Determine DB labels
        db_labels = []
        if has_source_col:
            sources_str = str(row.get(source_col, "")).strip()
            if sources_str and sources_str.lower() != "nan":
                # Split by pipe if multiple sources e.g. "KEGG|Reactome"
                db_labels = [s.strip() for s in sources_str.split("|") if s.strip()]
        
        # Fallback if no source column or empty
        if not db_labels:
            db_labels = [db_label]

        for db_lbl in db_labels:
            pkey = (pathway, db_lbl)
            for sym in (sym1, sym2):
                if sym and sym != "NAN":
                    gene_to_pathways[sym].add(pkey)
                    pathway_to_genes[pkey].add(sym)

    del pathway_df
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
    """
    Fail fast if a pathway/KEGG gene index barely overlaps the actual node
    gene universe — this is exactly the failure mode that silently produced
    zero pathway edges before (raw UniProt IDs being compared against gene
    symbols). Better to abort in seconds here than discover it after a
    multi-hour run with an empty pathway layer.
    """
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
# SECTION 3b — PATHWAY INDEX (KEGG)
# ============================================================================

def build_kegg_pathway_index(
    kegg_csv_path: str,
    max_pathway_size: int = 300,
    uniprot_to_symbol: dict = None,
) -> tuple[dict, dict, dict]:
    """
    Load KEGG pathway data.
    Expected columns: gene1 (or protein1), gene2 (or protein2), pathway
    (column names are auto-detected).

    Returns same structure as build_pathway_index_from_csv.
    """
    print(f"Loading KEGG data from {kegg_csv_path} ...")
    df = pd.read_csv(kegg_csv_path, low_memory=False)

    # Auto-detect column names
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ("gene1", "protein1","uniprot1", "gene_a", "symbol1"):
            col_map["gene1"] = col
        elif cl in ("gene2", "protein2","uniprot2", "gene_b", "symbol2"):
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
        col_map["gene1"]   : "gene1",
        col_map["gene2"]   : "gene2",
        col_map["pathway"] : "pathway",
    })
    pathway_df = df.dropna(subset=["pathway"]).copy()
    del df
    gc.collect()
    print(f"  {len(pathway_df):,} rows with pathway annotation\n")

    gene_to_pathways: dict[str, set] = defaultdict(set)
    pathway_to_genes: dict[tuple, set] = defaultdict(set)

    u2s = uniprot_to_symbol or {}

    for _, row in pathway_df.iterrows():
        raw1    = str(row["gene1"]).strip().upper()
        raw2    = str(row["gene2"]).strip().upper()
        pathway = str(row["pathway"]).strip()
        if not pathway or pathway.lower() == "nan":
            continue

        # Translate UniProt → symbol if mapping provided
        sym1 = u2s.get(raw1, u2s.get(raw1.lower(), raw1))
        sym2 = u2s.get(raw2, u2s.get(raw2.lower(), raw2))
        sym1 = sym1.strip().upper() if sym1 else raw1
        sym2 = sym2.strip().upper() if sym2 else raw2

        # Skip self-loops
        if sym1 == sym2:
            continue

        pkey = (pathway, "kegg")
        for sym in (sym1, sym2):
            if sym and sym != "NAN":
                gene_to_pathways[sym].add(pkey)
                pathway_to_genes[pkey].add(sym)

    del pathway_df
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
    kegg_g2p:     dict, kegg_p2g:     dict, kegg_excl:     dict,
) -> tuple[dict, dict, dict]:
    """Merge Reactome and KEGG pathway indices."""
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

    df["ligand_gname"]   = df["ligand_gname"].str.strip().str.upper()
    df["receptor_gname"] = df["receptor_gname"].str.strip().str.upper()
    df = df.dropna(subset=["ligand_gname", "receptor_gname"])

    lr_set         = set(zip(df["ligand_gname"], df["receptor_gname"]))
    ligand_genes   = set(df["ligand_gname"])
    receptor_genes = set(df["receptor_gname"])

    lr_descriptions = {}
    for _, row in df.iterrows():
        lr_descriptions[(row["ligand_gname"], row["receptor_gname"])] = {
            "ligand_description"  : row.get("ligand_description", ""),
            "receptor_description": row.get("receptor_description", ""),
        }

    print(f"  {len(lr_set):,} LR pairs loaded")
    print(f"  {len(ligand_genes):,} unique ligands, "
          f"{len(receptor_genes):,} unique receptors\n")

    return lr_set, ligand_genes, receptor_genes, lr_descriptions


# ============================================================================
# SECTION 5 — MULTIPLEX ASSEMBLY
# ============================================================================

def build_integrated_multiplex(
    ts_layers:          dict,
    hdrg_layers:        dict,
    gene_to_pathways:   dict,
    pathway_to_genes:   dict,
    lr_set:             set,
    ligand_genes:       set,
    receptor_genes:     set,
    lr_descriptions:    dict,
    cross_source_only:  bool = True,
    max_pathway_size:   int  = 300,
    pathway_chunk_size: int  = 50,
    ppi_reference_edges: set | None = None,
) -> tuple[nx.Graph, NodeRegistry, dict]:
    """
    Returns G, registry, and an edge_stats dict for reporting.

    ppi_reference_edges : optional set of frozenset({geneA, geneB}) pairs from
        a reference PPI network. When provided, pathway AND ligand-receptor
        edges are only added if the same gene pair is ALSO a physical PPI
        edge — i.e. only add pathway/LR/Reactome connections that are
        corroborated by your highest-confidence PPI source. PPI (intra-layer)
        edges themselves are never filtered by this (they ARE the PPI source).
    """
    all_layers = {**ts_layers, **hdrg_layers}
    registry   = NodeRegistry()
    G          = nx.Graph()
    G.graph["description"] = "Integrated TS+hdrg multiplex (pathway + LR edges)"

    # Tracking dicts for statistics
    edge_stats = {
        # ---- PPI (intra-layer) ----
        "ppi_edges_by_layer"            : defaultdict(int),

        # ---- Pathway edges ----
        # (db_label, source_i, source_j) -> count
        "pathway_edges_by_db_and_source": defaultdict(int),
        # (source_i, source_j) -> count
        "pathway_edges_by_source_pair"  : defaultdict(int),
        # pathway_key (name, db) -> count
        "pathway_edges_by_pathway"      : defaultdict(int),
        # skipped: same source
        "pathway_skipped_same_source"   : defaultdict(int),
        # skipped: same layer
        "pathway_skipped_same_layer"    : 0,
        # skipped: not corroborated by reference PPI network
        "pathway_skipped_no_ppi_support": 0,
        # annotated onto existing edges
        "pathway_annotated_existing"    : defaultdict(int),

        # ---- LR edges ----
        "lr_edges_by_source_pair"       : defaultdict(int),
        "lr_pairs_no_ligand_nodes"      : [],
        "lr_pairs_no_receptor_nodes"    : [],
        "lr_skipped_same_layer"         : 0,
        "lr_skipped_no_ppi_support"     : 0,

        # ---- Intra vs Inter layer breakdown ----
        "intra_inter_by_edge_type"      : {
            "ppi"             : {"intra": 0, "inter": 0},
            "pathway"         : {"intra": 0, "inter": 0},
            "ligand_receptor" : {"intra": 0, "inter": 0},
        },
        # (source_a, source_b) -> {"ppi": int, "pathway": int, "ligand_receptor": int}
        "edges_by_source_pair_and_type" : defaultdict(lambda: defaultdict(int)),
    }

    # -----------------------------------------------------------------------
    # 5a. Register nodes + PPI edges
    # -----------------------------------------------------------------------
    print("Adding intra-layer PPI nodes and edges ...")
    n_ppi = 0
    gene_to_node_ids: dict[str, list[int]] = defaultdict(list)

    for (source, cell_type), layer in all_layers.items():
        for gene in layer.nodes():
            nid = registry.get_or_add(source, cell_type, gene)
            G.add_node(nid, source=source, cell_type=cell_type, gene=gene)
            gene_to_node_ids[gene].append(nid)

        for u, v, edata in layer.edges(data=True):
            uid = registry.get_id(source, cell_type, u)
            vid = registry.get_id(source, cell_type, v)
            if uid is not None and vid is not None:
                G.add_edge(uid, vid, edge_type="ppi",
                           source=source, cell_type=cell_type)
                n_ppi += 1
                layer_key = (source, cell_type)
                edge_stats["ppi_edges_by_layer"][layer_key] += 1
                edge_stats["intra_inter_by_edge_type"]["ppi"]["intra"] += 1
                src_pair = (source, source)
                edge_stats["edges_by_source_pair_and_type"][src_pair]["ppi"] += 1

    print(f"  Nodes     : {G.number_of_nodes():,}")
    print(f"  PPI edges : {n_ppi:,}\n")
    gc.collect()

    # -----------------------------------------------------------------------
    # 5b. Pathway edges — chunked
    # -----------------------------------------------------------------------
    print("Adding pathway inter-layer edges (chunked) ...")
    n_pathway = 0
    pathways  = list(pathway_to_genes.items())

    for chunk_start in range(0, len(pathways), pathway_chunk_size):
        chunk = pathways[chunk_start: chunk_start + pathway_chunk_size]

        for (pathway_name, db_label), genes_in_path in chunk:
            node_ids_in_path = []
            for gene in genes_in_path:
                node_ids_in_path.extend(gene_to_node_ids.get(gene, []))

            if len(node_ids_in_path) < 2:
                continue

            for ni, nj in combinations(node_ids_in_path, 2):
                di = G.nodes[ni]
                dj = G.nodes[nj]

                same_layer = (di["source"] == dj["source"] and
                              di["cell_type"] == dj["cell_type"])
                if same_layer:
                    edge_stats["pathway_skipped_same_layer"] += 1
                    continue

                if cross_source_only and di["source"] == dj["source"]:
                    src_pair = tuple(sorted([di["source"], dj["source"]]))
                    edge_stats["pathway_skipped_same_source"][src_pair] += 1
                    continue

                if ppi_reference_edges is not None:
                    gene_pair = frozenset((di["gene"], dj["gene"]))
                    if gene_pair not in ppi_reference_edges:
                        edge_stats["pathway_skipped_no_ppi_support"] += 1
                        continue

                src_pair = tuple(sorted([di["source"], dj["source"]]))
                db_src_key = (db_label,) + src_pair

                if G.has_edge(ni, nj):
                    ed = G[ni][nj]
                    if "pathways" not in ed:
                        ed["pathways"] = set()
                    ed["pathways"].add((pathway_name, db_label))
                    edge_stats["pathway_annotated_existing"][src_pair] += 1
                else:
                    G.add_edge(ni, nj,
                               edge_type="pathway",
                               pathway=pathway_name,
                               pathway_db=db_label,
                               pathways={(pathway_name, db_label)})
                    n_pathway += 1
                    
                    # Track stats
                    edge_stats["pathway_edges_by_source_pair"][src_pair] += 1
                    edge_stats["pathway_edges_by_db_and_source"][db_src_key] += 1
                    edge_stats["pathway_edges_by_pathway"][(pathway_name, db_label)] += 1

                    is_inter = di["source"] != dj["source"]
                    key = "inter" if is_inter else "intra"
                    edge_stats["intra_inter_by_edge_type"]["pathway"][key] += 1
                    edge_stats["edges_by_source_pair_and_type"][src_pair]["pathway"] += 1

        gc.collect()

        if (chunk_start // pathway_chunk_size) % 10 == 0:
            done = min(chunk_start + pathway_chunk_size, len(pathways))
            print(f"  Processed {done:,}/{len(pathways):,} pathways | "
                  f"pathway edges so far: {n_pathway:,}")

    print(f"  Pathway edges added : {n_pathway:,}")
    if ppi_reference_edges is not None:
        print(f"  Pathway edges skipped (no PPI support) : "
              f"{edge_stats['pathway_skipped_no_ppi_support']:,}")
    print()
    gc.collect()

    # -----------------------------------------------------------------------
    # 5c. Ligand-Receptor edges
    # -----------------------------------------------------------------------
    print("Adding ligand-receptor inter-layer edges ...")
    n_lr = 0

    ligand_node_ids   = {
        gene: gene_to_node_ids[gene]
        for gene in ligand_genes if gene in gene_to_node_ids
    }
    receptor_node_ids = {
        gene: gene_to_node_ids[gene]
        for gene in receptor_genes if gene in gene_to_node_ids
    }

    for lig_gene in ligand_genes:
        if lig_gene not in gene_to_node_ids:
            edge_stats["lr_pairs_no_ligand_nodes"].append(str(lig_gene))
    for rec_gene in receptor_genes:
        if rec_gene not in gene_to_node_ids:
            edge_stats["lr_pairs_no_receptor_nodes"].append(str(rec_gene))

    for (lig_gene, rec_gene), desc in lr_descriptions.items():
        lig_nodes = ligand_node_ids.get(lig_gene, [])
        rec_nodes = receptor_node_ids.get(rec_gene, [])

        if not lig_nodes or not rec_nodes:
            continue

        for lig_nid in lig_nodes:
            for rec_nid in rec_nodes:
                dl = G.nodes[lig_nid]
                dr = G.nodes[rec_nid]

                same_layer = (dl["source"] == dr["source"] and
                              dl["cell_type"] == dr["cell_type"])
                if same_layer:
                    edge_stats["lr_skipped_same_layer"] += 1
                    continue

                if ppi_reference_edges is not None:
                    gene_pair = frozenset((lig_gene, rec_gene))
                    if gene_pair not in ppi_reference_edges:
                        edge_stats["lr_skipped_no_ppi_support"] += 1
                        continue

                src_pair = tuple(sorted([dl["source"], dr["source"]]))

                if not G.has_edge(lig_nid, rec_nid):
                    G.add_edge(lig_nid, rec_nid,
                               edge_type="ligand_receptor",
                               ligand_gene=lig_gene,
                               receptor_gene=rec_gene,
                               **desc)
                    n_lr += 1
                    edge_stats["lr_edges_by_source_pair"][src_pair] += 1

                    is_inter = dl["source"] != dr["source"]
                    key = "inter" if is_inter else "intra"
                    edge_stats["intra_inter_by_edge_type"]["ligand_receptor"][key] += 1
                    edge_stats["edges_by_source_pair_and_type"][src_pair]["ligand_receptor"] += 1
                else:
                    G[lig_nid][rec_nid]["lr_pair"] = (lig_gene, rec_gene)

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
# SECTION 6 — STATISTICS REPORT
# ============================================================================

def write_statistics_report(
    edge_stats:        dict,
    excluded_pathways: dict,
    output_dir:        str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. PPI edges by layer
    # ------------------------------------------------------------------
    ppi_rows = [
        {"source": src, "cell_type": ct, "n_ppi_edges": count}
        for (src, ct), count in sorted(
            edge_stats["ppi_edges_by_layer"].items(),
            key=lambda x: -x[1]
        )
    ]
    pd.DataFrame(ppi_rows).to_csv(
        out / "stats_ppi_edges_by_layer.csv", index=False
    )

    # ------------------------------------------------------------------
    # 2. Pathway edges by source pair (aggregated across DBs)
    # ------------------------------------------------------------------
    pathway_src_rows = [
        {"source_pair": " <-> ".join(pair), "n_pathway_edges": count}
        for pair, count in sorted(
            edge_stats["pathway_edges_by_source_pair"].items(),
            key=lambda x: -x[1]
        )
    ]
    pd.DataFrame(pathway_src_rows).to_csv(
        out / "stats_pathway_edges_by_source_pair.csv", index=False
    )

    # ------------------------------------------------------------------
    # 3. Pathway edges by DB and source pair
    # ------------------------------------------------------------------
    db_src_rows = [
        {"db": key[0], "source_pair": " <-> ".join(key[1:]), "n_pathway_edges": count}
        for key, count in sorted(
            edge_stats["pathway_edges_by_db_and_source"].items(),
            key=lambda x: -x[1]
        )
    ]
    pd.DataFrame(db_src_rows).to_csv(
        out / "stats_pathway_edges_by_db_and_source.csv", index=False
    )

    # ------------------------------------------------------------------
    # 4. Pathway edges per pathway
    # ------------------------------------------------------------------
    pathway_contrib_rows = [
        {"pathway": p, "db": db, "n_edges_contributed": c}
        for (p, db), c in sorted(
            edge_stats["pathway_edges_by_pathway"].items(),
            key=lambda x: -x[1]
        )
    ]
    pd.DataFrame(pathway_contrib_rows).to_csv(
        out / "stats_pathway_edges_by_pathway.csv", index=False
    )

    # ------------------------------------------------------------------
    # 5. Excluded pathways (oversized)
    # ------------------------------------------------------------------
    excl_rows = [
        {"pathway": p, "db": db, "n_genes": n, "reason": "exceeds max_pathway_size"}
        for (p, db), n in excluded_pathways.get("oversized", [])
    ]
    pd.DataFrame(excl_rows).to_csv(
        out / "stats_excluded_pathways.csv", index=False
    )

    # ------------------------------------------------------------------
    # 6. Skipped pathway edges
    # ------------------------------------------------------------------
    skipped_rows = []
    for pair, count in sorted(
        edge_stats["pathway_skipped_same_source"].items(),
        key=lambda x: -x[1]
    ):
        skipped_rows.append({
            "source_pair": " <-> ".join(pair),
            "n_skipped"  : count,
            "reason"     : "same source (cross_source_only=True)",
        })
    skipped_rows.append({
        "source_pair": "same layer",
        "n_skipped"  : edge_stats["pathway_skipped_same_layer"],
        "reason"     : "same layer (intra-layer pathway edges disabled)",
    })
    pd.DataFrame(skipped_rows).to_csv(
        out / "stats_pathway_edges_skipped.csv", index=False
    )

    # ------------------------------------------------------------------
    # 7. Pathway edges annotated onto existing edges
    # ------------------------------------------------------------------
    annotated_rows = [
        {"source_pair": " <-> ".join(pair), "n_annotated": count}
        for pair, count in sorted(
            edge_stats["pathway_annotated_existing"].items(),
            key=lambda x: -x[1]
        )
    ]
    pd.DataFrame(annotated_rows).to_csv(
        out / "stats_pathway_annotated_existing_edges.csv", index=False
    )

    # ------------------------------------------------------------------
    # 8. LR edges by source pair
    # ------------------------------------------------------------------
    lr_src_rows = [
        {"source_pair": " <-> ".join(pair), "n_lr_edges": count}
        for pair, count in sorted(
            edge_stats["lr_edges_by_source_pair"].items(),
            key=lambda x: -x[1]
        )
    ]
    pd.DataFrame(lr_src_rows).to_csv(
        out / "stats_lr_edges_by_source_pair.csv", index=False
    )

    # ------------------------------------------------------------------
    # 9. LR genes absent from graph
    # ------------------------------------------------------------------
    absent_rows = (
        [{"gene": g, "role": "ligand",   "reason": "not found in any layer"}
         for g in sorted(edge_stats["lr_pairs_no_ligand_nodes"])]
      + [{"gene": g, "role": "receptor", "reason": "not found in any layer"}
         for g in sorted(edge_stats["lr_pairs_no_receptor_nodes"])]
    )
    pd.DataFrame(absent_rows).to_csv(
        out / "stats_lr_genes_absent_from_graph.csv", index=False
    )

    # ------------------------------------------------------------------
    # 10. Intra vs inter-layer breakdown
    # ------------------------------------------------------------------
    intra_inter_rows = []
    for etype, counts in edge_stats["intra_inter_by_edge_type"].items():
        total = counts["intra"] + counts["inter"]
        intra_inter_rows.append({
            "edge_type"  : etype,
            "intra_layer": counts["intra"],
            "inter_layer": counts["inter"],
            "total"      : total,
        })
    pd.DataFrame(intra_inter_rows).to_csv(
        out / "stats_intra_vs_inter_layer.csv", index=False
    )

    # ------------------------------------------------------------------
    # 11. Edges by source pair and type
    # ------------------------------------------------------------------
    src_type_rows = []
    for src_pair, type_counts in sorted(
        edge_stats["edges_by_source_pair_and_type"].items()
    ):
        row = {"source_pair": " <-> ".join(src_pair)}
        row.update(type_counts)
        src_type_rows.append(row)
    pd.DataFrame(src_type_rows).fillna(0).to_csv(
        out / "stats_edges_by_source_pair_and_type.csv", index=False
    )

    # ------------------------------------------------------------------
    # 12. Human-readable summary
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("EDGE STATISTICS REPORT")
    lines.append("=" * 70)

    lines.append("\n--- PPI EDGES BY LAYER ---")
    for r in ppi_rows:
        lines.append(f"  {r['source']:<25} {r['cell_type']:<30} "
                     f"{r['n_ppi_edges']:>10,} edges")

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
        lines.append(f"  [{r['db']:<10}] {r['source_pair']:<35} "
                     f"{r['n_pathway_edges']:>10,} edges")

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
        lines.append(f"  [{r['db']:<10}] {r['pathway']:<50} "
                     f"{r['n_edges_contributed']:>8,} edges")
    lines.append("  → Full list saved to stats_pathway_edges_by_pathway.csv")

    lines.append("\n--- LR EDGES BY SOURCE PAIR ---")
    for r in lr_src_rows:
        lines.append(f"  {r['source_pair']:<40} {r['n_lr_edges']:>10,} edges")
    lines.append(f"  LR edges skipped (same layer): "
                 f"{edge_stats['lr_skipped_same_layer']:,}")

    lines.append("\n--- LR GENES ABSENT FROM GRAPH ---")
    lines.append(f"  Ligands  not found in any layer: "
                 f"{len(edge_stats['lr_pairs_no_ligand_nodes']):,}")
    lines.append(f"  Receptors not found in any layer: "
                 f"{len(edge_stats['lr_pairs_no_receptor_nodes']):,}")
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
        pickle.dump(G, f, protocol=4)
    print(f"Saved multiplex graph → {graph_path}")

    edges_path = out / "integrated_multiplex_edges.csv"
    chunk, chunk_size = [], 100_000
    header_written = False

    for u, v, d in G.edges(data=True):
        tu = registry.get_tuple(u)
        tv = registry.get_tuple(v)
        chunk.append({
            "node_id_u"     : u,
            "node_id_v"     : v,
            "source_layer_u": tu[0],
            "cell_type_u"   : tu[1],
            "gene_u"        : tu[2],
            "source_layer_v": tv[0],
            "cell_type_v"   : tv[1],
            "gene_v"        : tv[2],
            "edge_type"     : d.get("edge_type", ""),
            "pathway_db"    : d.get("pathway_db", ""),
            "pathway"       : str(d.get("pathway", "")),
            "ligand_gene"   : d.get("ligand_gene", ""),
            "receptor_gene" : d.get("receptor_gene", ""),
        })
        if len(chunk) >= chunk_size:
            df = pd.DataFrame(chunk)
            # df.to_csv(edges_path, mode="a",
            #           header=not header_written, index=False)
            header_written = True
            chunk = []
            gc.collect()

    if chunk:
        # pd.DataFrame(chunk).to_csv(edges_path, mode="a",
                                   # header=not header_written, index=False)

    # print(f"Saved edge list → {edges_path}")

    edge_type_counts = defaultdict(int)
    for _, _, d in G.edges(data=True):
        edge_type_counts[d.get("edge_type", "unknown")] += 1

    summary = {
        "n_nodes"         : G.number_of_nodes(),
        "n_edges"         : G.number_of_edges(),
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
        sub   = G.subgraph(node_set)
        n_ppi = sum(1 for _, _, d in sub.edges(data=True)
                    if d.get("edge_type") == "ppi")
        n_path = sum(1 for _, _, d in G.edges(node_set, data=True)
                     if d.get("edge_type") == "pathway")
        n_lr   = sum(1 for _, _, d in G.edges(node_set, data=True)
                     if d.get("edge_type") == "ligand_receptor")
        rows.append({
            "source"    : source,
            "cell_type" : ct,
            "n_nodes"   : len(node_set),
            "n_ppi"     : n_ppi,
            "n_pathway" : n_path,
            "n_lr"      : n_lr,
        })
    return pd.DataFrame(rows)


# ============================================================================
# SECTION 8 — MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build integrated TS+hdrg multiplex (pathway + LR, memory-optimised)"
    )
    parser.add_argument("--ts-layers-dir",      required=True)
    parser.add_argument("--hdrg-layers-dir",    required=True)
    parser.add_argument("--reactome-csv",        required=True)
    parser.add_argument("--kegg-csv",            default=None,
                        help="Optional KEGG pathway CSV (gene1, gene2, pathway columns)")
    parser.add_argument("--uniprot-map",         default=None,
                        help="CSV with 'uniprotid' and 'symbol' columns for UniProt→gene-symbol translation")
    parser.add_argument("--lr-csv",              required=True)
    parser.add_argument("--output-dir",          default="integrated_multiplex_output")
    parser.add_argument("--max-pathway-size",    type=int, default=300)
    parser.add_argument("--pathway-chunk-size",  type=int, default=50)
    parser.add_argument("--cross-source-only",   action="store_true")
    parser.add_argument("--base-ppi", default=None,
                    help="Path to full PPI network .pkl to add as a dedicated layer")
    parser.add_argument("--validate-against-ppi", action="store_true",
                    help="Only add pathway/Reactome and ligand-receptor inter-layer edges "
                         "if the same gene pair is ALSO an edge in a reference PPI network "
                         "(your highest-confidence source of real protein-protein interaction). "
                         "Requires --base-ppi or --ppi-validation-pkl.")
    parser.add_argument("--ppi-validation-pkl", default=None,
                    help="PPI network .pkl to validate pathway/LR edges against. Defaults to "
                         "--base-ppi if not given, so you don't have to load the same file twice.")
    args = parser.parse_args()

    t0 = time.time()

    print("\n" + "=" * 60)
    print("STEP 1: LOADING LAYERS")
    print("=" * 60)
    ts_layers   = load_layers_from_folder(args.ts_layers_dir,   "tabula_sapiens")
    hdrg_layers = load_layers_from_folder(args.hdrg_layers_dir, "hdrg")

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
            ppi_for_validation = base_ppi_graph  # reuse, avoid loading twice
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
    # AFTER (fixed):
    uniprot_to_symbol = load_uniprot_map(args.uniprot_map)  # move this BEFORE STEP 2

    reactome_g2p, reactome_p2g, reactome_excl = build_pathway_index_from_csv(
        args.reactome_csv,
        max_pathway_size=args.max_pathway_size,
        db_label="reactome",
        uniprot_to_symbol=uniprot_to_symbol,   # ← add this
    )

    if args.kegg_csv:
        # and if kegg_csv:
        kegg_g2p, kegg_p2g, kegg_excl = build_kegg_pathway_index(
            args.kegg_csv,
            max_pathway_size=args.max_pathway_size,
            uniprot_to_symbol=uniprot_to_symbol,   # ← add this too
        )
        gene_to_pathways, pathway_to_genes, excluded_pathways = merge_pathway_indices(
            reactome_g2p, reactome_p2g, reactome_excl,
            kegg_g2p,     kegg_p2g,     kegg_excl,
        )
    else:
        print("No KEGG CSV provided — using Reactome only.\n")
        gene_to_pathways  = reactome_g2p
        pathway_to_genes  = reactome_p2g
        excluded_pathways = reactome_excl

    # ------------------------------------------------------------------
    # GUARDRAIL: fail fast if the pathway gene ID space doesn't match the
    # actual layer gene universe (this is exactly the failure mode that
    # previously produced a graph with zero real pathway edges after a
    # multi-hour run). Cheap to check now, before STEP 4's expensive
    # combinatorial edge assembly.
    # ------------------------------------------------------------------
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
    )

    print("=" * 60)
    print("STEP 5: SAVING OUTPUTS")
    print("=" * 60)
    save_multiplex(G, registry, args.output_dir)

    stats_df   = layer_statistics(G, registry)
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