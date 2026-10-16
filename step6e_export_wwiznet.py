#!/usr/bin/env python3
"""
Step 6e: Export the disease module (step 6 output) as a wwiznet_ppic dataset
(nodes.tsv + edges.tsv) for multilayer visualisation.

Why this is straightforward for your pipeline
---------------------------------------------
wwiznet has no native multilayer concept; a "layer" is just a categorical
column (`layer_c`) on the nodes. Your module nodes are ALREADY namespaced
"<celltype>::GENE", so every node belongs to exactly one layer and the
export is mostly bookkeeping:

  nodes.tsv   id, gene, layer_c, layer_index_n, is_seed_c, rwr_score_n,
              rwr_in_layer_n, degree_n, intra_degree_n, inter_degree_n,
              n_layers_with_gene_n, x_/y_/z_<layout> ...
  edges.tsv   source, target, edge_class_c, layer_source_c, layer_target_c,
              layer_pair_c, [scored_methods_c, ligand_type_c]

Column suffix conventions follow the wwiznet README:
  *_c categorical, *_n numeric, *_a array, x_<name>/y_<name>/z_<name> layouts.
`layer_c` is the same column name Philipp used for the Julia/Marlene data,
and Philipp's multilayer_layout_experiments repo (run_layouts.py) reads
`layer_c` (+ optionally `edge_type_c`) -- so this output can be fed straight
into it for fancier layouts.

edge_class_c values
-------------------
  intra_ppi          PPI edge inside one cell-type layer
  identity           same gene, different layers (e.g. global_ppi <-> neuron)
  inter_lr_pathway   gene-gene edge between layers (flat LR / Reactome /
                     scored LR; see scored_methods_c if provenance given)

Built-in layouts (networkx only, no extra dependencies)
-------------------------------------------------------
  stack    each layer laid out independently (its own intra-layer edges),
           layers stacked along z. Cleanest "layers as planes" picture.
  aligned  ONE xy layout computed on the gene-collapsed graph; every
           (layer, gene) node sits at that gene's xy, z = layer. Same gene
           stacks vertically, so identity/inter-layer edges are short
           vertical lines.
Select with --layouts stack aligned (default: both).
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SEP = "::"


def _norm_node(s: str) -> str:
    s = str(s).strip()
    if SEP in s:
        ct, gene = s.rsplit(SEP, 1)
        return f"{ct}{SEP}{gene.strip().upper()}"
    return s.upper()


def gene_of(n: str) -> str:
    return n.rsplit(SEP, 1)[-1]


def layer_of(n: str) -> str:
    return n.rsplit(SEP, 1)[0] if SEP in n else "unknown"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_module_graph(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: _norm_node(n) for n in G.nodes()})
    print(f"Module graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def load_provenance(path: Path | None):
    """(node_a, node_b) frozenset -> {'methods': set, 'ligand_types': set}"""
    if not path:
        return {}
    df = pd.read_csv(path)
    prov = defaultdict(lambda: {"methods": set(), "ligand_types": set()})
    for r in df.itertuples(index=False):
        a = f"{r.cell_type_source}{SEP}{str(r.ligand).upper()}"
        b = f"{r.cell_type_target}{SEP}{str(r.receptor).upper()}"
        key = frozenset((a, b))
        prov[key]["methods"].add(str(r.source))
        lt = getattr(r, "ligand_type", None)
        if isinstance(lt, str) and lt:
            prov[key]["ligand_types"].add(lt)
    print(f"Provenance: {len(prov):,} scored-LR node pairs loaded")
    return prov


def order_layers(G: nx.Graph) -> list[str]:
    counts = defaultdict(int)
    for n in G.nodes():
        counts[layer_of(n)] += 1
    # backbone (global_ppi) at the bottom, then biggest layers first
    return sorted(counts, key=lambda l: (l != "global_ppi", -counts[l], l))


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
def _rescale(pos: dict, radius: float) -> dict:
    if not pos:
        return pos
    xy = np.array(list(pos.values()), dtype=float)
    xy -= xy.mean(axis=0)
    m = np.abs(xy).max()
    if m > 0:
        xy = xy / m * radius
    return dict(zip(pos.keys(), map(tuple, xy)))


def layout_stack(G, layers, radius, dz, seed):
    out = {}
    for i, layer in enumerate(layers):
        nodes = [n for n in G.nodes() if layer_of(n) == layer]
        sub = nx.Graph()
        sub.add_nodes_from(nodes)
        sub.add_edges_from((u, v) for u, v in G.subgraph(nodes).edges())
        pos = nx.spring_layout(sub, dim=2, seed=seed, iterations=100) if len(nodes) > 1 else {nodes[0]: (0, 0)}
        for n, (x, y) in _rescale(pos, radius).items():
            out[n] = (x, y, i * dz)
        print(f"  [stack] {layer}: {len(nodes):,} nodes")
    return out


def layout_aligned(G, layers, radius, dz, seed):
    idx = {l: i for i, l in enumerate(layers)}
    GG = nx.Graph()
    GG.add_nodes_from({gene_of(n) for n in G.nodes()})
    for u, v in G.edges():
        gu, gv = gene_of(u), gene_of(v)
        if gu != gv:
            GG.add_edge(gu, gv)
    print(f"  [aligned] gene-collapsed graph: {GG.number_of_nodes():,} genes, {GG.number_of_edges():,} edges")
    pos = _rescale(nx.spring_layout(GG, dim=2, seed=seed, iterations=100), radius)
    return {n: (*pos[gene_of(n)], idx[layer_of(n)] * dz) for n in G.nodes()}


LAYOUTS = {"stack": layout_stack, "aligned": layout_aligned}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module-dir", required=True, help="step 6 output dir (module_graph.pkl, disease_module.csv)")
    ap.add_argument("--per-celltype-scores", default=None, help="step 4 per_celltype_scores.csv (adds rwr_in_layer_n)")
    ap.add_argument("--provenance", default=None, help="step 2 scored_lr_edge_provenance.csv (adds method/ligand_type to edges)")
    ap.add_argument("--out", required=True, help="output folder (gets nodes.tsv, edges.tsv, layer_order.txt)")
    ap.add_argument("--layouts", nargs="+", default=["stack", "aligned"], choices=list(LAYOUTS))
    ap.add_argument("--xy-radius", type=float, default=10.0)
    ap.add_argument("--z-spacing", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-identity-edges", action="store_true",
                    help="omit global_ppi<->cell-type identity edges from edges.tsv (nodes stay)")
    args = ap.parse_args()

    mdir, out = Path(args.module_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = load_module_graph(mdir / "module_graph.pkl")
    layers = order_layers(G)
    idx = {l: i for i, l in enumerate(layers)}
    print(f"{len(layers)} layers (bottom -> top): {layers}\n")
    (out / "layer_order.txt").write_text(",".join(layers) + "\n")

    prov = load_provenance(Path(args.provenance) if args.provenance else None)

    # ---- node table -------------------------------------------------------
    mod_csv = mdir / "disease_module.csv"
    meta = pd.read_csv(mod_csv).set_index("node") if mod_csv.exists() else pd.DataFrame()
    per_ct = None
    if args.per_celltype_scores:
        per_ct = pd.read_csv(args.per_celltype_scores)
        per_ct = per_ct.set_index(per_ct.columns[0])
        per_ct.index = per_ct.index.astype(str).str.upper()

    gene_layers = defaultdict(set)
    for n in G.nodes():
        gene_layers[gene_of(n)].add(layer_of(n))

    # edge classification first (needed for intra/inter degree)
    edge_rows = []
    intra_deg, inter_deg = defaultdict(int), defaultdict(int)
    for u, v in G.edges():
        lu, lv = layer_of(u), layer_of(v)
        if idx[lu] > idx[lv]:
            u, v, lu, lv = v, u, lv, lu
        if lu == lv:
            cls = "intra_ppi"
            intra_deg[u] += 1; intra_deg[v] += 1
        else:
            cls = "identity" if gene_of(u) == gene_of(v) else "inter_lr_pathway"
            inter_deg[u] += 1; inter_deg[v] += 1
        row = {"source": u, "target": v, "edge_class_c": cls,
               "layer_source_c": lu, "layer_target_c": lv,
               "layer_pair_c": lu if lu == lv else f"{lu}--{lv}"}
        if prov:
            p = prov.get(frozenset((u, v)))
            row["scored_methods_c"] = ",".join(sorted(p["methods"])) if p else ""
            row["ligand_type_c"] = ",".join(sorted(p["ligand_types"])) if p else ""
        edge_rows.append(row)
    edges = pd.DataFrame(edge_rows)
    if args.drop_identity_edges:
        n0 = len(edges)
        edges = edges[edges["edge_class_c"] != "identity"]
        print(f"Dropped {n0 - len(edges):,} identity edges from edges.tsv")

    # layouts
    layout_cols = {}
    for name in args.layouts:
        print(f"Computing layout '{name}' ...")
        layout_cols[name] = LAYOUTS[name](G, layers, args.xy_radius, args.z_spacing, args.seed)

    rows = []
    for n in G.nodes():
        g, l = gene_of(n), layer_of(n)
        r = {
            "id": n, "gene": g, "layer_c": l, "layer_index_n": idx[l],
            "is_seed_c": "seed" if (n in meta.index and bool(meta.at[n, "is_seed"])) else "non_seed",
            "rwr_score_n": float(meta.at[n, "rwr_score"]) if n in meta.index else 0.0,
            "degree_n": G.degree(n), "intra_degree_n": intra_deg[n], "inter_degree_n": inter_deg[n],
            "n_layers_with_gene_n": len(gene_layers[g]),
        }
        if per_ct is not None:
            r["rwr_in_layer_n"] = float(per_ct.at[g, l]) if (g in per_ct.index and l in per_ct.columns) else 0.0
        for name, pos in layout_cols.items():
            r[f"x_{name}"], r[f"y_{name}"], r[f"z_{name}"] = (round(float(c), 4) for c in pos[n])
        rows.append(r)
    nodes = pd.DataFrame(rows)

    nodes.to_csv(out / "nodes.tsv", sep="\t", index=False)
    edges.to_csv(out / "edges.tsv", sep="\t", index=False)

    print("\n" + "=" * 60)
    print(f"nodes.tsv : {len(nodes):,} rows  ({(nodes.is_seed_c == 'seed').sum()} seeds)")
    print(f"edges.tsv : {len(edges):,} rows")
    print(edges["edge_class_c"].value_counts().to_string())
    print(f"layouts   : {list(layout_cols)}")
    print(f"-> {out.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()