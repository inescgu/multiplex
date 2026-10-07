#!/usr/bin/env python3
"""
Export the disease module's position WITHIN the larger multiplex, for
DataDiVR, at three different scopes: 1-hop neighborhood, 2-hop
neighborhood, and the full network. global_ppi is hidden throughout.

Input
-----
--ref-graph   gene_reference_graph.pkl (step 3 / rebuilt by step 2.5) --
              the FULL namespaced "<celltype>::GENE" multiplex+bipartite
              network, not just the module.
--module-dir  step 6 output dir (disease_module.csv) -- defines which
              nodes in --ref-graph count as "module" (seed or RWR-added).
--scores-csv  OPTIONAL gene_rwr_scores.csv (step 4's aggregated output,
              columns: gene, rwr_score) -- if given, context (non-module)
              nodes also get a real RWR score in their properties instead
              of being blank. The same bare-gene score is applied to that
              gene in every cell type it appears in (this file has no
              per-cell-type breakdown).

Scope / sizing (from your cell_type_metadata.json + gene_reference_graph.pkl):
  full network (global_ppi hidden): ~52,000 nodes, 200,000+ edges.
  This is almost certainly too much for the WebGL preview -- hop1/hop2
  exist so you have something that will actually render.

Output (in --out/<mode>/datadivr_upload/, one folder per --modes entry):
  nodes_stack.csv       plain x,y,z per node (gene-aligned x,y across
                         cell types; z = cell-type layer)
  links.csv              edge id pairs
  nodecolors.csv          R,G,B,A -- module nodes bright (DRG category/
                          subcategory, shaded; seed alpha=255, RWR-added
                          alpha=180); context nodes same hue, alpha=~30
                          (faint backdrop)
  nodeproperties.csv      "gene: X; cell_type: Y; category: Z/W;
                           status: SEED|NEIGHBOR|CONTEXT; score: N.NNN"
  linkcolors.csv          edges with both ends in-module: bright (cell
                          type's color for intra-layer, gold for
                          cross-layer); any edge touching a context-only
                          node: faint uniform gray
  clusterlabels.csv       one cluster per cell type present in this
                          scope, listing every node id in that layer
"""
import argparse
import colorsys
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SEP = "::"


def ct_of(n):   return n.rsplit(SEP, 1)[0] if SEP in n else "unknown"
def gene_of(n): return n.rsplit(SEP, 1)[-1]


def _norm_node(s: str) -> str:
    """Same normalization as step 6: uppercase only the gene portion,
    preserve the cell-type prefix's original case."""
    s = str(s).strip()
    if SEP in s:
        ct, gene = s.rsplit(SEP, 1)
        return f"{ct}{SEP}{gene.strip().upper()}"
    return s.upper()


def sanitize_name(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "-").replace(",", "")


# ---------------------------------------------------------------------------
# DRG hierarchy -> major category / subcategory / base color
# ---------------------------------------------------------------------------
DRG_HIERARCHY = {
    "A-fiber": {"color": "#1f77b4", "subcategories": {
        "Proprioceptors": {"cell_types": [
            "A-Propr.EPHA3", "A-Propr.HAPLN4", "A-Propr.PCDH8",
        ]},
        "A-beta LTMR": {"cell_types": [
            "Ab-LTMR.CCKAR", "Ab-LTMR.ETV1", "Ab-LTMR.LGI2", "Ab-LTMR.NSG2",
        ]},
        "A-LTMR": {"cell_types": ["A-LTMR.TAC3"]},
        "A-peptidergic": {"cell_types": [
            "A-PEP.CHRNA7_SLC18A3", "A-PEP.KIT", "A-PEP.NTRK3_S100A16", "A-PEP.SCGN_ADRA2C",
        ]},
    }},
    "C-fiber": {"color": "#ff7f0e", "subcategories": {
        "C-LTMR": {"cell_types": ["C-LTMR.CDH9"]},
        "C-peptidergic": {"cell_types": [
            "C-PEP.ADORA2B", "C-PEP.TAC1_CACNG5", "C-PEP.TAC1_CHRNA3",
        ]},
        "MRGPRX nociceptors": {"cell_types": [
            "C-NP.MRGPRX1_GFRA2", "C-NP.MRGPRX1_MRGPRX4",
        ]},
        "C-thermosensitive": {"cell_types": [
            "C-Thermo.RXFP1", "C-Thermo.TRPM8",
        ]},
        "C-SST nociceptor": {"cell_types": ["C-NP.SST"]},
    }},
    "Non-classified": {"color": "#d62728", "subcategories": {
        "ATF3 (injured)": {"cell_types": ["ATF3"]},
    }},
    "Non-neuronal": {"color": "#2ca02c", "subcategories": {
        "Immune": {"cell_types": [
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
        "Glial/Nervous": {"cell_types": [
            "Schwann cell", "enteroglial cell", "microglial cell", "neuron",
        ]},
        "Immune/Hematopoietic": {"cell_types": [
            "Langerhans cell", "colon macrophage", "common myeloid progenitor",
            "erythroid progenitor cell", "hematopoietic cell", "hematopoietic precursor cell",
            "hematopoietic stem cell", "tissue-resident macrophage",
        ]},
    }},
}

_CT_TO_MAJOR_SUB = {}
for _major, _mdata in DRG_HIERARCHY.items():
    for _subcat, _sdata in _mdata["subcategories"].items():
        for _ct in _sdata["cell_types"]:
            _CT_TO_MAJOR_SUB[sanitize_name(_ct).lower()] = (_major, _subcat)


def classify(cell_type: str):
    key = sanitize_name(cell_type).lower()
    return _CT_TO_MAJOR_SUB.get(key, ("unknown", "unknown"))


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def shade(hex_color, step, n_steps):
    r, g, b = [v / 255 for v in hex_to_rgb(hex_color)]
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    if n_steps > 1:
        ll = 0.30 + 0.45 * (step / (n_steps - 1))
    r2, g2, b2 = colorsys.hls_to_rgb(hh, ll, min(ss + 0.05, 1.0))
    return tuple(int(round(v * 255)) for v in (r2, g2, b2))


def build_color_table():
    table = {}
    for major, mdata in DRG_HIERARCHY.items():
        subs = list(mdata["subcategories"].keys())
        table[major] = {sub: shade(mdata["color"], i, len(subs)) for i, sub in enumerate(subs)}
    table["unknown"] = {"unknown": (150, 150, 150)}
    return table


def norm01(arr):
    arr = np.asarray(arr, dtype=float)
    span = arr.max(axis=0) - arr.min(axis=0)
    span[span == 0] = 1.0
    out = (arr - arr.min(axis=0)) / span
    return np.clip(out * 0.98 + 0.01, 0, 1)


def clean(s):
    return str(s).replace(",", "").replace(";", "")


def khop_nodeset(H, seeds, k):
    """Multi-source BFS out to depth k from `seeds`, restricted to H."""
    visited = set(seeds) & set(H.nodes())
    frontier = set(visited)
    for _ in range(k):
        nxt = set()
        for n in frontier:
            nxt.update(H.neighbors(n))
        nxt -= visited
        if not nxt:
            break
        visited |= nxt
        frontier = nxt
    return visited


GENE_LAYOUT_THRESHOLD = 4000  # unique genes above which we switch to the faster per-layer+grid layout


def combined_gene_layout(sub, meta, seed):
    """One shared x,y per GENE (not per node), computed from the whole
    subgraph's connectivity, same as the module-only exporter -- this is
    what makes the same gene line up vertically across cell types, and
    gives sparse/small subgraphs real spread since the layout optimizes
    over the whole thing at once instead of one tiny layer at a time."""
    GG = nx.Graph()
    GG.add_nodes_from(meta["gene"].unique())
    GG.add_edges_from((gene_of(u), gene_of(v)) for u, v in sub.edges() if gene_of(u) != gene_of(v))
    pos = nx.spring_layout(GG, seed=seed)
    return [pos[g] for g in meta["gene"]]


def grid_layout(sub, meta, cts_sorted, seed):
    """Fast path for large graphs: lay out each cell type's genes with its
    own small spring layout (cheap), then tile the cell types across a
    grid in x,y so layers can't collapse onto each other the way they did
    when every layer was centered near the same origin. Trades away
    gene-to-gene alignment across layers in exchange for staying fast at
    full-network scale."""
    import math
    n = len(cts_sorted)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    margin = 0.08
    pos_by_ct_gene = {}
    for i, ct in enumerate(cts_sorted):
        col, row = i % cols, i // cols
        ct_nodes = meta.loc[meta["cell_type"] == ct, "node"].tolist()
        ct_sub = sub.subgraph(ct_nodes)
        GG = nx.Graph()
        GG.add_nodes_from(gene_of(n) for n in ct_nodes)
        GG.add_edges_from((gene_of(u), gene_of(v)) for u, v in ct_sub.edges())
        p = nx.spring_layout(GG, seed=seed)
        arr = np.array(list(p.values())) if p else np.zeros((1, 2))
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        cell_w, cell_h = 1.0 / cols, 1.0 / rows
        for gene, (x, y) in p.items():
            nx_, ny_ = (np.array([x, y]) - lo) / span  # 0-1 within this layer's own layout
            gx = col * cell_w + margin * cell_w + nx_ * cell_w * (1 - 2 * margin)
            gy = row * cell_h + margin * cell_h + ny_ * cell_h * (1 - 2 * margin)
            pos_by_ct_gene[(ct, gene)] = (gx, gy)
    return [pos_by_ct_gene[(row.cell_type, row.gene)] for row in meta.itertuples(index=False)]


def build_subset(H, node_set, meta_full, color_table, z_spacing, seed, layout_mode="auto"):
    sub = H.subgraph(node_set).copy()
    nodes = sorted(sub.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = list(sub.edges())
    meta = meta_full.loc[meta_full["node"].isin(nodes)].set_index("node").loc[nodes].reset_index()

    # alpha: module nodes bright, context nodes faint
    def node_rgba(row):
        r, g, b = color_table[row.major_category][row.subcategory]
        if row.status == "SEED":
            a = 255
        elif row.status == "NEIGHBOR":
            a = 180
        else:
            a = 30
        return r, g, b, a

    rgba = meta.apply(node_rgba, axis=1, result_type="expand")
    rgba.columns = ["r", "g", "b", "a"]
    meta = pd.concat([meta, rgba], axis=1)

    cts_sorted = sorted(meta["cell_type"].unique())
    n_unique_genes = meta["gene"].nunique()
    use_grid = (layout_mode == "grid") or (layout_mode == "auto" and n_unique_genes > GENE_LAYOUT_THRESHOLD)

    print(f"    Laying out {len(cts_sorted)} cell-type layer(s), {len(nodes):,} nodes, "
          f"{n_unique_genes:,} unique genes ...")
    if use_grid:
        reason = (f"{n_unique_genes:,} unique genes > {GENE_LAYOUT_THRESHOLD:,}"
                   if layout_mode == "auto" else "--layout grid forced")
        print(f"    -> {reason}: using the fast per-layer grid layout "
              f"(layers tiled in x,y; genes no longer aligned across layers)")
        xy = grid_layout(sub, meta, cts_sorted, seed)
    else:
        print(f"    -> using the combined gene-level layout (same gene aligns across layers)")
        xy = combined_gene_layout(sub, meta, seed)

    stack = np.array([[x, y, cts_sorted.index(row.cell_type) * z_spacing]
                       for (x, y), row in zip(xy, meta.itertuples(index=False))])
    stack = norm01(stack)

    def write_xyz(path, coords):
        pd.DataFrame(coords).to_csv(path, header=False, index=False, float_format="%.6f")

    return sub, nodes, idx, edges, meta, stack, cts_sorted, write_xyz


def write_mode(mode_dir, H, node_set, meta_full, color_table, z_spacing, seed, layout_mode="auto"):
    mode_dir = Path(mode_dir)
    dd = mode_dir / "datadivr_upload"
    dd.mkdir(parents=True, exist_ok=True)

    sub, nodes, idx, edges, meta, stack, cts_sorted, write_xyz = build_subset(
        H, node_set, meta_full, color_table, z_spacing, seed, layout_mode
    )

    write_xyz(dd / "nodes_stack.csv", stack)

    with open(dd / "links.csv", "w") as f:
        for u, v in edges:
            f.write(f"{idx[u]},{idx[v]}\n")

    meta[["r", "g", "b", "a"]].to_csv(dd / "nodecolors.csv", header=False, index=False)

    with open(dd / "nodeproperties.csv", "w") as f:
        for row in meta.itertuples(index=False):
            score_str = f"{row.rwr_score:.3f}" if pd.notna(row.rwr_score) else "n/a"
            f.write(f"gene: {clean(row.gene)}; cell_type: {clean(row.cell_type)}; "
                     f"category: {clean(row.major_category)}/{clean(row.subcategory)}; "
                     f"status: {row.status}; score: {score_str}\n")

    meta_by_node = meta.set_index("node")
    INTER_BRIGHT = (255, 200, 60, 220)
    FADE = (120, 120, 120, 15)
    with open(dd / "linkcolors.csv", "w") as f:
        for u, v in edges:
            su, sv = meta_by_node.loc[u, "status"], meta_by_node.loc[v, "status"]
            both_in_module = su != "CONTEXT" and sv != "CONTEXT"
            if not both_in_module:
                f.write(",".join(str(c) for c in FADE) + "\n")
            elif ct_of(u) == ct_of(v):
                r, g, b, _ = meta_by_node.loc[u, ["r", "g", "b", "a"]]
                f.write(f"{int(r)},{int(g)},{int(b)},180\n")
            else:
                f.write(",".join(str(c) for c in INTER_BRIGHT) + "\n")

    with open(dd / "clusterlabels.csv", "w") as f:
        for ct in cts_sorted:
            major, sub_cat = classify(ct)
            label = clean(f"{ct} ({major}/{sub_cat})" if major != "unknown" else ct)
            ids = [str(idx[n]) for n in meta.loc[meta["cell_type"] == ct, "node"]]
            f.write(label + "," + ",".join(ids) + "\n")

    n_mod = (meta["status"] != "CONTEXT").sum()
    print(f"    -> {len(meta):,} nodes ({n_mod:,} module / {len(meta) - n_mod:,} context), "
          f"{len(edges):,} edges -> {dd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-graph", required=True, help="gene_reference_graph.pkl (full namespaced network)")
    ap.add_argument("--module-dir", required=True, help="step 6 output dir (disease_module.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hide-celltypes", nargs="+", default=["global_ppi"])
    ap.add_argument("--modes", nargs="+", default=["hop1", "hop2", "full"], choices=["hop1", "hop2", "full"])
    ap.add_argument("--scores-csv", default=None, help="optional gene_rwr_scores.csv for context-node scores")
    ap.add_argument("--z-spacing", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", choices=["auto", "combined", "grid"], default="auto",
                     help="auto (default): combined gene-aligned layout below "
                          f"{GENE_LAYOUT_THRESHOLD:,} unique genes, fast per-layer grid above it. "
                          "'combined' / 'grid' forces one or the other regardless of size.")
    args = ap.parse_args()

    out = Path(args.out)

    print(f"Loading full reference graph from {args.ref_graph} ...")
    with open(args.ref_graph, "rb") as f:
        H = pickle.load(f)
    H = nx.relabel_nodes(H, {n: _norm_node(n) for n in H.nodes()})
    print(f"  {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges (raw)")

    hide = {c.upper() for c in args.hide_celltypes}
    drop = [n for n in H if ct_of(n).upper() in hide]
    H.remove_nodes_from(drop)
    print(f"  Hid {len(drop):,} node(s) from {sorted(hide)} -> "
          f"{H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges remain\n")

    mod_df = pd.read_csv(Path(args.module_dir) / "disease_module.csv")
    mod_df["node"] = mod_df["node"].apply(_norm_node)
    module_nodes = set(mod_df["node"]) & set(H.nodes())
    dropped_mod = set(mod_df["node"]) - set(H.nodes())
    if dropped_mod:
        print(f"\u26a0 {len(dropped_mod)} module node(s) were only in a hidden cell type "
              f"and are excluded from every view below: {sorted(dropped_mod)[:10]}")
    is_seed_map = dict(zip(mod_df["node"], mod_df["is_seed"]))
    mod_score_map = dict(zip(mod_df["node"], mod_df["rwr_score"]))

    gene_score_map = {}
    if args.scores_csv:
        sdf = pd.read_csv(args.scores_csv)
        sdf["gene"] = sdf["gene"].astype(str).str.strip().str.upper()
        gene_score_map = dict(zip(sdf["gene"], sdf["rwr_score"]))
        print(f"Loaded {len(gene_score_map):,} bare-gene scores from {args.scores_csv} "
              f"for context-node properties\n")

    # ---- full metadata table, computed once, reused across every mode ----
    color_table = build_color_table()
    unknown_cts = set()
    rows = []
    for n in H.nodes():
        gene, ct = gene_of(n), ct_of(n)
        major, sub = classify(ct)
        if major == "unknown":
            unknown_cts.add(ct)
        if n in module_nodes:
            status = "SEED" if bool(is_seed_map.get(n, False)) else "NEIGHBOR"
            score = float(mod_score_map.get(n, float("nan")))
        else:
            status = "CONTEXT"
            score = float(gene_score_map[gene]) if gene in gene_score_map else float("nan")
        rows.append({"node": n, "gene": gene, "cell_type": ct, "major_category": major,
                     "subcategory": sub, "status": status, "rwr_score": score})
    meta_full = pd.DataFrame(rows)
    if unknown_cts:
        print(f"\u26a0 {len(unknown_cts)} cell type(s) not in the DRG hierarchy table "
              f"(colored gray): {sorted(unknown_cts)}\n")

    for mode in args.modes:
        print(f"[{mode}]")
        if mode == "full":
            node_set = set(H.nodes())
        elif mode == "hop1":
            node_set = khop_nodeset(H, module_nodes, 1)
        elif mode == "hop2":
            node_set = khop_nodeset(H, module_nodes, 2)
        write_mode(out / mode, H, node_set, meta_full, color_table, args.z_spacing, args.seed, args.layout)
        print()

    print("Done. Try hop1 first, then hop2, then full -- stop at whichever the preview still handles.")


if __name__ == "__main__":
    main()