#!/usr/bin/env python3
"""
Export the disease module (module_graph.pkl from step 6) into the actual
DataDiVR_WebApp upload fields, with the global_ppi backbone hidden.

Format confirmed from the ZacharysKarateClub example files on the /upload
page (all headerless, all row-order-aligned, no quoting):

  nodes_<layout>.csv     one row per node:  x,y,z                (floats 0-1)
  links.csv              one row per edge:  node1_id,node2_id    (0-indexed
                          row numbers into the nodes file)
  <name>_nodecolors.csv  one row per node, SAME ORDER as nodes file:
                              R,G,B,A   (ints 0-255)
  <name>_linkcolors.csv  one row per edge, SAME ORDER as links.csv:
                              R,G,B,A
  <name>_nodeproperties.csv  one row per node, SAME ORDER as nodes file:
                              free-text "key: value; key: value; ..."
                              (shown when you click/hover a node)
  <name>_clusterlabels.csv   one row per cluster:
                              label_name,node_id,node_id,...
                              (a partition of node ids into named,
                              presumably selectable/highlightable groups)

Produces (in --out/datadivr_upload/):
  nodes_stack.csv / nodes_cluster.csv   two alternative layouts, same node
                                         order (upload whichever you want
                                         to view; nodecolors/nodeproperties/
                                         links/linkcolors/clusterlabels are
                                         shared by both)
  links.csv
  nodecolors.csv          colored by major DRG category, shaded by
                           subcategory; alpha 255 for seeds, 110 for
                           RWR-added neighbors (translucent)
  nodeproperties.csv      "gene: X; cell_type: Y; category: A-fiber/Sub;
                           status: SEED|NEIGHBOR; score: 0.123"
  linkcolors.csv          intra-cell-type PPI edges get that cell type's
                           color (dim); cross-cell-type LR/pathway edges
                           get a bright neutral gold so they stand out
  clusterlabels.csv       one cluster per cell type, labeled with its
                           name + major category/subcategory, listing
                           every node id in that layer
  legend.json / legend.csv  NOT a confirmed upload format (no example file
                           was available on the upload page) -- written as
                           a best-guess label,R,G,B table plus JSON, for
                           you to try uploading or just read directly
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


def sanitize_name(name: str) -> str:
    """Kept in sync with step1/step2's sanitize_name() so hierarchy keys
    match the actual multiplex/bipartite folder names."""
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
    """node-properties fields use ';' to separate key:value pairs and ','
    to separate rows elsewhere in the format, so strip both from free text."""
    return str(s).replace(",", "").replace(";", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", required=True, help="step 6 output dir (module_graph.pkl, disease_module.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hide-celltypes", nargs="+", default=["global_ppi"])
    ap.add_argument("--z-spacing", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mdir, out = Path(args.module_dir), Path(args.out)
    dd = out / "datadivr_upload"
    dd.mkdir(parents=True, exist_ok=True)

    with open(mdir / "module_graph.pkl", "rb") as f:
        G = pickle.load(f)
    mod = pd.read_csv(mdir / "disease_module.csv").set_index("node")

    hide = {c.upper() for c in args.hide_celltypes}
    drop = [n for n in G if ct_of(n).upper() in hide]
    G.remove_nodes_from(drop)
    print(f"Hid {len(drop)} node(s) from {sorted(hide)}; {G.number_of_nodes()} nodes / {G.number_of_edges()} edges left")
    n_iso = sum(1 for n in G if G.degree(n) == 0)
    if n_iso:
        print(f"  note: {n_iso} node(s) are now isolated (only attached via the hidden layer)")
    ncc = nx.number_connected_components(G)
    if ncc > 1:
        print(f"  note: visible graph has {ncc} components")
    if G.number_of_edges() > 20000:
        print(f"  note: {G.number_of_edges():,} edges -- the WebGL preview may warn "
              f"'too many links for preview'; this is a preview-only rendering limit, "
              f"the uploaded data itself is unaffected.")

    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = list(G.edges())  # fixed order, reused for links.csv AND linkcolors.csv
    color_table = build_color_table()

    unknown_cts = set()
    rows_meta = []
    for n in nodes:
        gene, ct = gene_of(n), ct_of(n)
        major, sub = classify(ct)
        if major == "unknown":
            unknown_cts.add(ct)
        is_seed = bool(mod["is_seed"].get(n, False))
        score = float(mod["rwr_score"].get(n, 0.0))
        r, g, b = color_table[major][sub]
        a = 255 if is_seed else 110
        rows_meta.append({
            "node": n, "gene": gene, "cell_type": ct, "major_category": major,
            "subcategory": sub, "is_seed": is_seed, "rwr_score": score,
            "r": r, "g": g, "b": b, "a": a,
        })
    if unknown_cts:
        print(f"  \u26a0 {len(unknown_cts)} cell type(s) not found in the DRG hierarchy table "
              f"(colored gray): {sorted(unknown_cts)}")

    meta = pd.DataFrame(rows_meta)

    # ---- layout 1: stacked multiplex (gene-aligned x,y; z per cell type) ----
    cts_sorted = sorted(meta["cell_type"].unique())
    GG = nx.Graph()
    GG.add_nodes_from(meta["gene"].unique())
    GG.add_edges_from((gene_of(u), gene_of(v)) for u, v in edges if gene_of(u) != gene_of(v))
    pos2d = nx.spring_layout(GG, seed=args.seed)
    stack = norm01(np.array([[*pos2d[g], cts_sorted.index(ct) * args.z_spacing]
                              for g, ct in zip(meta["gene"], meta["cell_type"])]))

    # ---- layout 2: 3D force layout of the namespaced graph ----
    p3 = nx.spring_layout(G, dim=3, seed=args.seed)
    cluster = norm01(np.array([p3[n] for n in nodes]))

    def write_xyz(path, coords):
        pd.DataFrame(coords).to_csv(path, header=False, index=False, float_format="%.6f")

    write_xyz(dd / "nodes_stack.csv", stack)
    write_xyz(dd / "nodes_cluster.csv", cluster)

    # ---- links.csv ----
    with open(dd / "links.csv", "w") as f:
        for u, v in edges:
            f.write(f"{idx[u]},{idx[v]}\n")

    # ---- nodecolors.csv (row order == nodes file order) ----
    meta[["r", "g", "b", "a"]].to_csv(dd / "nodecolors.csv", header=False, index=False)

    # ---- nodeproperties.csv (row order == nodes file order) ----
    with open(dd / "nodeproperties.csv", "w") as f:
        for row in meta.itertuples(index=False):
            status = "SEED" if row.is_seed else "NEIGHBOR"
            f.write(f"gene: {clean(row.gene)}; cell_type: {clean(row.cell_type)}; "
                     f"category: {clean(row.major_category)}/{clean(row.subcategory)}; "
                     f"status: {status}; score: {row.rwr_score:.3f}\n")

    # ---- linkcolors.csv (row order == links.csv order) ----
    # intra-cell-type PPI edges: dim version of that cell type's color.
    # cross-cell-type LR/pathway edges: bright neutral gold, to stand out.
    meta_by_node = meta.set_index("node")
    INTER_COLOR = (255, 200, 60, 180)
    with open(dd / "linkcolors.csv", "w") as f:
        for u, v in edges:
            if ct_of(u) == ct_of(v):
                r, g, b, _ = meta_by_node.loc[u, ["r", "g", "b", "a"]]
                f.write(f"{int(r)},{int(g)},{int(b)},70\n")
            else:
                f.write(",".join(str(c) for c in INTER_COLOR) + "\n")

    # ---- clusterlabels.csv: one cluster per cell type (layer) ----
    with open(dd / "clusterlabels.csv", "w") as f:
        for ct in cts_sorted:
            major, sub = classify(ct)
            label = clean(f"{ct} ({major}/{sub})" if major != "unknown" else ct)
            ids = [str(idx[n]) for n in meta.loc[meta["cell_type"] == ct, "node"]]
            f.write(label + "," + ",".join(ids) + "\n")

    # ---- legend: NOT confirmed as an upload format (no example file existed) ----
    legend_rows = []
    for major, subs in color_table.items():
        for sub, (r, g, b) in subs.items():
            legend_rows.append({"label": f"{major}/{sub}", "r": r, "g": g, "b": b})
    pd.DataFrame(legend_rows).to_csv(dd / "legend_guess.csv", index=False)
    legend_json = {major: {sub: f"#{r:02x}{g:02x}{b:02x}" for sub, (r, g, b) in subs.items()}
                   for major, subs in color_table.items()}
    (out / "legend.json").write_text(json.dumps(legend_json, indent=2))

    # ---- reference table, not for upload ----
    meta.to_csv(out / "node_annotations.csv", index=False)

    n_seeds = int(meta["is_seed"].sum())
    print(f"\nWrote {len(meta)} nodes ({n_seeds} seeds, {len(meta) - n_seeds} RWR-added), "
          f"{len(edges)} links, {len(cts_sorted)} cluster labels -> {dd}")
    print(f"Reference table -> {out / 'node_annotations.csv'}")
    print(f"legend_guess.csv is UNVERIFIED (no example file was on the upload page) -- "
          f"try uploading it, and if the page rejects it, use legend.json just to read colors off yourself.")


if __name__ == "__main__":
    main()