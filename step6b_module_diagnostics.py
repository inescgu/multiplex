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

NEW: interaction-type histograms (source/method + NeuronChat ligand_type)
---------------------------------------------------------------------------
Previously this script only ever broke the module down by CELL TYPE. It had
no notion of *interaction type* -- i.e. which LR-inference method
(cellphonedb, natmi, connectome, sca, logfc, liana_consensus, or
neuronchat) or which NeuronChat ligand category (Neurotransmitter,
Neuropeptide, Gas, Gap junction protein) actually produced each
ligand-receptor edge in the module. That's not because it wasn't
interesting -- it's because the information didn't survive the pipeline:
step2_build_bipartite_edges.py used to write scored-LR edges into the
bipartite .tsv files as bare (gene1, gene2) pairs with no provenance at
all, so by the time a node reached step6/step6b there was nothing left to
group by.

step2 now also writes scored_lr_edge_provenance.csv (one row per accepted
scored-LR edge: cell_type_source, cell_type_target, ligand, receptor,
source, ligand_type) -- see that script's docstring. This script reads that
file (from the SAME --input-dir you already pass, i.e. the MultiXrank wdir
that was step2's --wdir) and, for every row, reconstructs the exact
"<celltype>::GENE" node pair step 2.5 would have namespaced it as, to
determine whether that specific edge:
  (a) survived step2's own filtering at all ("candidate" scope -- this is
      every row in the provenance file, i.e. every scored-LR edge that made
      it into a bipartite file, independent of whether it ended up
      connected in the final module), and/or
  (b) is actually present as an edge between two nodes of the FINAL
      disease module ("module" scope -- checked directly against
      module_graph.pkl via G.has_edge()).

For each scope, two histograms are produced: counts grouped by `source`
(the per-method/tool name) and counts grouped by `ligand_type` (NeuronChat
only -- LIANA rows have no ligand_type and are dropped from that
particular breakdown, but still appear in the by-source one).

If scored_lr_edge_provenance.csv isn't found (e.g. step2 was run without
--scored-lr-csv, or with an older version of the script), these histograms
are skipped with a one-line note -- everything else in the script runs as
before.

Answers:
  - How many cell types are represented in the final module?
  - How many nodes / edges per cell type?
  - How many seeds are in the module, broken down by cell type?
  - How many seeds are isolated (degree 0) in the final module, overall and per cell type?
  - How many seeds are duplicated (literal duplicate lines in seeds.txt, and/or
    the same bare gene seeded under more than one cell type)?
  - Are some cell types over/under-represented relative to their size in the module?
  - How many candidate / final-module LR edges came from each inference
    method (cellphonedb, natmi, connectome, sca, logfc, liana_consensus,
    neuronchat), and, among NeuronChat's, from each ligand_type category?

Reads:
  <output-dir>/module_graph.pkl              (networkx graph -- used for edge counts)
  <output-dir>/disease_module.csv            (per-node table incl. node, gene, cell_type,
                                               is_seed, degree_in_module -- the authoritative
                                               source for membership/annotation)
  <input-dir>/seeds/seeds.txt                (original seed list, for duplicate detection)
  <input-dir>/scored_lr_edge_provenance.csv  (optional; written by step2 when
                                               --scored-lr-csv was used -- source/method +
                                               NeuronChat ligand_type per scored-LR edge)

Writes (to --out-prefix, default "module_exploratory"):
  <prefix>_per_celltype.csv
  <prefix>_seed_membership.csv
  <prefix>_summary.txt
  <prefix>_seed_pct_by_celltype.png     -- bar chart: % of the module's seeds
                                            contributed by each cell type
  <prefix>_module_pct_by_celltype.png   -- bar chart: % of the module's total
                                            nodes contributed by each cell type
  <prefix>_top_seed_genes.png           -- bar chart: top individual seed
                                            genes (bare gene symbol, e.g.
                                            "TRPV1", NOT the namespaced node
                                            id like "Ab-LTMR.CCKAR::TRPV1"),
                                            ranked by how many distinct cell
                                            types they were seeded under AND
                                            which are actually present in the
                                            final module (--top-n-seeds caps
                                            how many genes are shown)
  <prefix>_candidate_lr_by_source.csv/.png       -- ALL scored-LR edges that
                                                     survived step2 filtering,
                                                     grouped by method/source
  <prefix>_candidate_lr_by_ligand_type.csv/.png  -- same, grouped by
                                                     NeuronChat's ligand_type
  <prefix>_module_lr_by_source.csv/.png          -- same two, but restricted
  <prefix>_module_lr_by_ligand_type.csv/.png        to edges present in the
                                                     final disease module
  (--no-plots skips all PNGs; the underlying CSVs are unaffected)

Usage:
  python step6b_module_diagnostics.py \
      --output-dir 06_disease_module/disease_module_output \
      --input-dir  04_multixrank_prep/multixrank_hetero_input
"""
import argparse
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

# Case-insensitive lookup: cell-type name (as it appears in the "<celltype>::GENE"
# namespace) -> major category. Built case-insensitively because the namespace
# prefix's case comes straight from the hdrg layer filenames (e.g.
# "Ab-LTMR.CCKAR_layer.pkl") and isn't guaranteed to match DRG_HIERARCHY's
# casing exactly in every dataset.
_CELL_TYPE_TO_MAJOR = {}
for _major, _mdata in DRG_HIERARCHY.items():
    for _subcat, _sdata in _mdata["subcategories"].items():
        for _ct in _sdata["cell_types"]:
            _CELL_TYPE_TO_MAJOR[_ct.strip().lower()] = _major


def get_category_from_cell_type(cell_type: str) -> str:
    """Major category ("A-fiber"/"C-fiber"/"Non-classified"/"Non-neuronal")
    for a cell type, matched case-insensitively; "unknown" if not found
    (e.g. "global_ppi", or any cell type not in the DRG hierarchy)."""
    if not isinstance(cell_type, str):
        return "unknown"
    return _CELL_TYPE_TO_MAJOR.get(cell_type.strip().lower(), "unknown")


def _major_category_palette() -> dict:
    return {"A-fiber": "#1f77b4", "C-fiber": "#ff7f0e", "Non-classified": "#d62728",
            "Non-neuronal": "#2ca02c", "unknown": "#999999"}


def _norm_node(s: str) -> str:
    """Uppercase only the gene portion of a possibly-namespaced node
    string, preserving the cell-type prefix's original case. Must match
    step6's normalization exactly, or seed strings loaded here won't
    match node names in module_graph.pkl / disease_module.csv."""
    s = s.strip()
    if "::" in s:
        ct, gene = s.rsplit("::", 1)
        return f"{ct}::{gene.strip().upper()}"
    return s.upper()


def _gene_of(node: str) -> str:
    return node.rsplit("::", 1)[-1] if "::" in node else node


def _celltype_of(node: str) -> str:
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


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
        # Defensive fallback for older disease_module.csv files that
        # predate the "cell_type" column -- derive it the same way step6
        # does, straight from the node's own namespace.
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
        # Use the full namespaced "node" id, not bare "gene" -- the same
        # bare gene symbol can legitimately be a seed under more than one
        # cell type, so "gene" alone would collapse distinct seeds together.
        raw_seeds = module_df.loc[module_df["is_seed"], "node"].tolist()

    return G, module_df, raw_seeds


# ---------------------------------------------------------------------------
# NEW: interaction-type (source/method + NeuronChat ligand_type) provenance
# ---------------------------------------------------------------------------
def load_lr_provenance(input_dir):
    """Loads scored_lr_edge_provenance.csv written by step2 (only present
    if step2 was run with --scored-lr-csv). Returns None if not found, so
    callers can skip the interaction-type histograms gracefully instead of
    erroring on older runs / runs without scored LR data."""
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
    """Restricts prov_df to rows whose reconstructed "<celltype>::GENE"
    ligand/receptor node pair is actually an edge in the final module graph
    G. This is the "module" scope for the interaction-type histograms --
    the candidate scored-LR edge not only survived step2's filtering, it's
    part of the connected disease module step6 produced."""
    if prov_df is None or prov_df.empty:
        return prov_df
    lig_nodes = prov_df["cell_type_source"] + "::" + prov_df["ligand"]
    rec_nodes = prov_df["cell_type_target"] + "::" + prov_df["receptor"]
    mask = [G.has_edge(a, b) for a, b in zip(lig_nodes, rec_nodes)]
    return prov_df[pd.Series(mask, index=prov_df.index)]


def interaction_type_counts(prov_df, column: str) -> pd.Series:
    """Unique-edge counts of prov_df grouped by `column` ('source' or
    'ligand_type'). Rows with a null value in `column` are dropped (e.g.
    LIANA rows have no ligand_type, so they simply don't appear in that
    particular breakdown). Deduplicates on
    (cell_type_source, cell_type_target, ligand, receptor, <column>) so a
    method/category isn't double-counted if the same edge happened to
    appear more than once in the provenance file."""
    if prov_df is None or prov_df.empty:
        return pd.Series(dtype=int)
    sub = prov_df.dropna(subset=[column])
    if sub.empty:
        return pd.Series(dtype=int)
    key_cols = ["cell_type_source", "cell_type_target", "ligand", "receptor", column]
    sub = sub.drop_duplicates(subset=key_cols)
    return sub.groupby(column).size().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Cell-type extraction -- now node-based (authoritative), not edge-based
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

        # Intra-cell-type edges: both endpoints share this cell type's
        # namespace prefix. Computed straight from node membership rather
        # than trusting a 'cell_types' edge attribute, so it's correct
        # regardless of whether that attribute exists/is populated on
        # every edge type (PPI vs. bipartite/identity edges).
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
    """Generic bar chart, degrading gracefully if matplotlib isn't
    available (mirrors step6's elbow-plot pattern).

    bar_colors: optional list of per-bar colors (same length as labels),
    e.g. one color per major category. When given, a legend mapping
    category name -> color is drawn using _major_category_palette(),
    restricted to whichever categories actually appear in bar_colors.
    """
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
    """Top individual seed genes by BARE gene symbol (not the namespaced
    node id) -- e.g. "TRPV1", never "Ab-LTMR.CCKAR::TRPV1" -- ranked by
    how many distinct cell types they were seeded under that are also
    actually present in the final module (sm['in_final_module'])."""
    present = sm[sm["in_final_module"]]
    if present.empty:
        return pd.Series(dtype=int)
    counts = present.groupby("gene")["cell_type"].nunique().sort_values(ascending=False)
    return counts.head(top_n)


# ---------------------------------------------------------------------------
# Seed-level diagnostics
# ---------------------------------------------------------------------------
def seed_membership(module_df, raw_seeds):
    """
    raw_seeds are already fully-namespaced, case-correct node strings
    (via _norm_node), matching disease_module.csv's "node" column exactly.
    """
    node_info = module_df.set_index("node")
    module_node_set = set(node_info.index)

    dup_in_file = Counter(raw_seeds)
    duplicated_in_file = {g: c for g, c in dup_in_file.items() if c > 1}

    unique_seeds = sorted(set(raw_seeds))

    # Bare-gene -> set of cell types it was seeded under, across ALL
    # unique seeds (this is what "spans multiple cell types" now means --
    # see module docstring for why the old edge-based definition no
    # longer applies in the namespaced structure).
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
                     help="MultiXrank wdir containing seeds/seeds.txt and (if step2 was run with "
                          "--scored-lr-csv) scored_lr_edge_provenance.csv -- same --input-dir passed to step6")
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

    print("\n" + "=" * 70)
    print("MODULE OVERVIEW")
    print("=" * 70)
    print(f"Total nodes           : {G.number_of_nodes():,}")
    print(f"Total edges           : {G.number_of_edges():,}")
    print(f"Cell types represented: {len(cell_types)}")
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

    # --- seed membership ---
    sm, dup_in_file = seed_membership(module_df, raw_seeds)
    print("\n" + "-" * 70)
    print("SEED DIAGNOSTICS")
    print("-" * 70)
    print(f"Seed lines in seeds.txt      : {len(raw_seeds)}")
    print(f"Unique seed genes            : {len(set(raw_seeds))}")
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

    # --- bar charts (cell type) ---
    seed_pct_path = str(Path(args.output_dir) / f"{args.out_prefix}_seed_pct_by_celltype.png")
    module_pct_path = str(Path(args.output_dir) / f"{args.out_prefix}_module_pct_by_celltype.png")
    top_seeds_path = str(Path(args.output_dir) / f"{args.out_prefix}_top_seed_genes.png")
    seed_pct_ok = module_pct_ok = top_seeds_ok = False

    if not args.no_plots:
        print("\n" + "-" * 70)
        print("PLOTS")
        print("-" * 70)

        # % of the module's seeds contributed by each cell type
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

        # % of the module's total nodes contributed by each cell type
        module_pl = pl.sort_values("pct_of_module_nodes", ascending=False)
        palette = _major_category_palette()
        module_colors = [palette.get(c, palette["unknown"]) for c in module_pl["major_category"]]
        module_pct_ok = plot_bar(
            module_pl["cell_type"].tolist(), module_pl["pct_of_module_nodes"].tolist(),
            title="Disease module composition by cell type",
            ylabel="% of module nodes", out_path=module_pct_path,
            bar_colors=module_colors,
        )

        # top individual seed genes (bare gene symbol) by # of module cell
        # types they were seeded under
        top_counts = compute_top_seed_genes(sm, top_n=args.top_n_seeds)
        top_seeds_ok = plot_bar(
            top_counts.index.tolist(), top_counts.values.tolist(),
            title=f"Top {len(top_counts)} seed genes by # cell types seeded (present in module)",
            ylabel="Number of cell types (in module)", out_path=top_seeds_path, color="#C44E52",
        )

    # --- NEW: interaction-type histograms (method/source + NeuronChat ligand_type) ---
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
        f"Total nodes: {G.number_of_nodes():,}   Total edges: {G.number_of_edges():,}",
        f"Cell types represented: {len(cell_types)}",
        "",
        "Per-cell-type breakdown:",
        pl.to_string(index=False),
        "",
        f"Isolated nodes overall: {len(isolated_all):,} (of which seeds: {len(isolated_seeds_all)})",
        f"Seed lines in seeds.txt: {len(raw_seeds)}  | unique: {len(set(raw_seeds))}  "
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