#!/usr/bin/env python3
"""
step6f_export_sl_decomposition_wwiznet.py

Export ONE single-layer HiDeF module (default SL 6) together with the
multiplex neuron-type sub-modules it decomposes into, as a wwiznet_ppic
dataset (nodes.tsv + edges.tsv), and optionally render an animation of the
decomposition.

Needs plot_sl_module_decomposition.py in the same folder (selection logic,
cluster loaders and disk_spread are imported from it, so the sub-modules
chosen here are EXACTLY the ones in your two-worlds figure).

LAYERS (layer_c)
----------------
  SL<id>          the single-layer module: its genes + its PPI edges
  <neuron layer>  one layer per selected multiplex sub-module (dominant
                  cell-type layer of that cluster; "<layer>__<cluster>" if
                  --allow-multiple-per-layer gives two in one layer).
                  Members keep their real cell type in cell_type_c.

NODES
-----
  SL layer      id = "SL6::GENE"
  sub-module    id = "<celltype>::GENE" (as in the supra-graph). A node
                listed by several selected sub-modules is drawn once, in
                the first (lowest q) one -- same rule as the figure.

edge_class_c
------------
  intra_ppi            PPI edge inside one layer (SL network / multiplex intra)
  inter_celltype       multiplex edge between two different layers
  sl_to_submodule      SL gene  ->  its copy inside a sub-module
                       (the "this gene got recruited into this neuron type" edge)

LAYOUTS (x_/y_/z_<name>, all in one coordinate frame so they can be morphed)
-------
  sl_collapsed   everything at z = 0. Every copy of a gene sits exactly on
                 its SL position (== the single-layer picture); sub-module
                 members that are not SL genes sit at the mean of their SL
                 neighbours (or on an outer halo by sub-module).
  sl_aligned     layers lifted along z. SL genes keep the SL xy in every
                 layer (identity edges become vertical lines), the other
                 members relax around them.
  sl_stack       each layer laid out independently from its own edges,
                 stacked in z ("layers as planes").
Animation order: sl_collapsed -> sl_aligned -> sl_stack.

Then feed the folder to run_layouts.py for more layouts, e.g.
  python run_layouts.py -i OUT -o OUT_layouts \
      --layer-column layer_c --type-column cell_type_c \
      --pathway-column none --edge-type-column edge_class_c \
      --layer-order "$(cat OUT/layer_order.txt)"
"""
import argparse
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from plot_sl_module_decomposition import (disk_spread, load_nodes, load_parents,
                                           norm_gene, select_submodules)

LAYOUT_ORDER = ["sl_collapsed", "sl_aligned", "sl_stack"]


# ============================================================================
# layouts
# ============================================================================
def sl_unit_layout(H):
    """Unit-radius 2D layout of the SL module: giant component by
    Kamada-Kawai (blended with a radial-rank spread so it isn't clumpy),
    small components / isolates on an outer ring."""
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    giant = H.subgraph(comps[0]).copy()
    if len(giant) > 2:
        lp = {n: np.asarray(p, float) for n, p in nx.kamada_kawai_layout(giant).items()}
    else:
        lp = {n: np.array([0.3 * i, 0.0]) for i, n in enumerate(giant)}
    if len(giant) > 8:
        d = disk_spread(list(giant), lp)
        lp = {n: 0.6 * lp[n] + 0.4 * d[n] for n in giant}
    arr = np.array(list(lp.values()))
    c = arr.mean(axis=0)
    m = max(np.linalg.norm(arr - c, axis=1).max(), 1e-9)
    pos = {n: (p - c) / m for n, p in lp.items()}
    rest = [n for comp in comps[1:] for n in sorted(comp)]
    for i, n in enumerate(rest):
        th = np.pi / 2 + 2 * np.pi * i / max(len(rest), 1)
        pos[n] = 1.25 * np.array([np.cos(th), np.sin(th)])
    return pos


def build_layouts(nodes, intra_edges, nbr, sl_layer, layer_idx, sl_pos, sub_order,
                  radius, dz, seed):
    rng = np.random.default_rng(seed)
    gene_pos = sl_pos  # gene -> unit xy
    ids = list(nodes)

    # ---- collapsed (unit scale)
    col = {}
    for n in ids:
        d = nodes[n]
        if d["gene"] in gene_pos and d["in_sl"]:
            col[n] = np.array(gene_pos[d["gene"]], float)
            continue
        near = [gene_pos[nodes[k]["gene"]] for k in nbr.get(n, ()) if nodes[k]["in_sl"]
                and nodes[k]["gene"] in gene_pos]
        if near:
            col[n] = np.mean(near, axis=0) + rng.normal(0, 0.03, 2)
        else:  # halo, one sector per sub-module
            k = sub_order.index(d["sub"]) if d["sub"] in sub_order else 0
            th = np.pi / 2 - 2 * np.pi * k / max(len(sub_order), 1)
            col[n] = 1.5 * np.array([np.cos(th), np.sin(th)]) + rng.normal(0, 0.08, 2)

    by_layer = defaultdict(list)
    for n in ids:
        by_layer[nodes[n]["layer"]].append(n)

    # ---- aligned: SL genes fixed at SL xy, other members relax around them
    ali = {}
    for layer, lids in by_layer.items():
        if layer == sl_layer:
            ali.update({n: col[n] for n in lids})
            continue
        L = nx.Graph()
        L.add_nodes_from(lids)
        L.add_edges_from(e for e in intra_edges if e[0] in L and e[1] in L)
        fixed = [n for n in lids if nodes[n]["in_sl"]]
        if len(lids) == 1 or len(fixed) == len(lids):
            ali.update({n: col[n] for n in lids})
            continue
        p = nx.spring_layout(L, pos={n: col[n] for n in lids}, fixed=fixed or None,
                             k=1.2 / np.sqrt(len(lids)), iterations=150, seed=seed)
        ali.update({n: np.asarray(p[n], float) for n in lids})

    # ---- stack: every layer independent
    stk = {}
    for layer, lids in by_layer.items():
        if layer == sl_layer:
            stk.update({n: col[n] for n in lids})
            continue
        L = nx.Graph()
        L.add_nodes_from(lids)
        L.add_edges_from(e for e in intra_edges if e[0] in L and e[1] in L)
        p = nx.spring_layout(L, seed=seed, k=1.5 / np.sqrt(max(len(lids), 1)), iterations=200) \
            if len(lids) > 1 else {lids[0]: np.zeros(2)}
        stk.update(disk_spread(lids, p))

    out = {}
    for name, xy, flat in [("sl_collapsed", col, True), ("sl_aligned", ali, False),
                           ("sl_stack", stk, False)]:
        out[name] = {n: (*(xy[n] * radius), 0.0 if flat else layer_idx[nodes[n]["layer"]] * dz)
                     for n in ids}
    return out


# ============================================================================
# animation
# ============================================================================
def render_animation(nodes_df, edges_df, layouts, layer_order, path, fps, hold, move, spin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401

    names = [n for n in LAYOUT_ORDER if n in layouts]
    ids = nodes_df["id"].tolist()
    idx = {n: i for i, n in enumerate(ids)}
    P = {nm: nodes_df[[f"x_{nm}", f"y_{nm}", f"z_{nm}"]].to_numpy(float) for nm in names}
    ei = np.array([[idx[s], idx[t]] for s, t in zip(edges_df["source"], edges_df["target"])])
    cls = edges_df["edge_class_c"].to_numpy()
    layers = nodes_df["layer_c"].to_numpy()
    genes = nodes_df["gene"].to_numpy()
    layer_rows = {l: np.where(layers == l)[0] for l in layer_order}

    pal = plt.get_cmap("tab10")
    lcol = {l: to_rgba("#333333" if i == 0 else pal((i - 1) % 10)) for i, l in enumerate(layer_order)}
    colors = np.array([lcol[l] for l in layers])
    is_sl = (nodes_df["in_sl_module_c"] == "sl_gene").to_numpy()
    hl = (nodes_df["is_highlight_c"] == "highlight").to_numpy()
    hl_pal = ["#C0392B", "#6C3483", "#D35400", "#0E6655", "#1A5276", "#9A7D0A"]
    gcol = {g: hl_pal[i % len(hl_pal)] for i, g in enumerate(sorted(set(genes[hl])))}
    ecol = np.array([to_rgba(gcol[genes[i]]) if hl[i] else (0, 0, 0, 0) for i in range(len(ids))])
    sizes = np.where(hl, 150, np.where(is_sl, 38, 7))

    allp = np.vstack(list(P.values()))
    lo, hi = allp.min(0), allp.max(0)
    ctr, half = (lo + hi) / 2, (hi - lo).max() / 2 * 0.62
    rad = float(np.linalg.norm(allp[:, :2] - ctr[:2], axis=1).max()) * 1.08
    label_x, label_y = ctr[0] + rad * 1.2, ctr[1]
    z_sl_final = P[names[-1]][layer_rows[layer_order[0]], 2].mean()
    z_final = {l: P[names[-1]][layer_rows[l], 2].mean() for l in layer_order}

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_box_aspect((1, 1, 1))

    style = {"intra_ppi": ("#9a9a9a", 0.25, 0.5), "inter_celltype": ("#7f7f7f", 0.15, 0.4),
             "sl_to_submodule": ("#C0392B", 0.45, 0.8)}
    colls, masks = {}, {}
    p0 = P[names[0]]
    for k, (c, a, lw) in style.items():
        m = cls == k
        if not m.any():  # empty collections break add_collection3d on some matplotlib versions
            continue
        masks[k] = m
        colls[k] = Line3DCollection(np.stack([p0[ei[m, 0]], p0[ei[m, 1]]], axis=1),
                                    colors=c, alpha=a, linewidths=lw)
        ax.add_collection3d(colls[k])

    # one translucent ring per layer: the "plane" that floats up with its nodes
    th = np.linspace(0, 2 * np.pi, 120)
    rings = {}
    for l in layer_order:
        z0 = p0[layer_rows[l], 2].mean()
        rings[l], = ax.plot(ctr[0] + rad * np.cos(th), ctr[1] + rad * np.sin(th),
                            np.full_like(th, z0), color=lcol[l], lw=1.2, alpha=0.0)

    sc = ax.scatter(*p0.T, s=sizes, c=colors, edgecolors=ecol, linewidths=2.0, depthshade=False)
    title = fig.text(0.5, 0.94, "", ha="center", fontsize=16, fontweight="bold")
    dyn = []  # per-frame text artists (layer + gene labels)

    ease = lambda t: t * t * (3 - 2 * t)
    frames = [(names[0], names[0], 0.0)] * hold
    for a, b in zip(names[:-1], names[1:]):
        frames += [(a, b, ease(t)) for t in np.linspace(0, 1, move)]
        frames += [(b, b, 0.0)] * hold
    labels = {"sl_collapsed": "Single-layer module", "sl_aligned": "Decomposing by neuron type",
              "sl_stack": "Neuron-type sub-modules"}

    def update(i):
        a, b, t = frames[i]
        pos = (1 - t) * P[a] + t * P[b]
        sc._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        for k, coll in colls.items():
            m = masks[k]
            coll.set_segments(np.stack([pos[ei[m, 0]], pos[ei[m, 1]]], axis=1))
        ax.view_init(elev=16 + 6 * np.sin(i / len(frames) * 2 * np.pi),
                     azim=-60 + spin * i / len(frames))

        for tx in dyn:
            tx.remove()
        dyn.clear()
        z_sl = pos[layer_rows[layer_order[0]], 2].mean()
        for j, l in enumerate(layer_order):
            zl = pos[layer_rows[l], 2].mean()
            if j == 0:
                al = 1.0
            else:  # fade in as the layer separates from the SL plane
                gap = abs(z_final[l] - z_sl_final) or 1.0
                al = float(np.clip(abs(zl - z_sl) / (0.7 * gap), 0, 1))
            rings[l].set_data_3d(ctr[0] + rad * np.cos(th), ctr[1] + rad * np.sin(th),
                                 np.full_like(th, zl))
            rings[l].set_alpha(0.45 * al)
            if al > 0.03:
                dyn.append(ax.text(label_x, label_y, zl, l, color=lcol[l], alpha=al, fontsize=10,
                                   fontweight="bold", ha="left", va="center"))
        for r in np.where(hl)[0]:
            dyn.append(ax.text(pos[r, 0], pos[r, 1], pos[r, 2] + 0.05 * half, genes[r],
                               color=gcol[genes[r]], fontsize=10 if layers[r] == layer_order[0] else 8,
                               fontweight="bold", ha="center", va="bottom", zorder=10,
                               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)))
        title.set_text(labels.get(b if t > 0.5 else a, ""))
        return sc,

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    path = Path(path)
    if path.suffix == ".mp4" and animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=6000)
    else:
        if path.suffix == ".mp4":
            print("ffmpeg not found -> writing GIF instead")
            path = path.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
    ani.save(path, writer=writer, dpi=110)
    plt.close(fig)
    print(f"animation: {path}  ({len(frames)} frames)")


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = "11_hidef_module/results_cpm"
    ap.add_argument("--sl-cluster", default="6")
    ap.add_argument("--matched-pairs", default=f"{base}/compare_singlelayer/matched_pairs.csv")
    ap.add_argument("--ml-nodes", default=f"{base}/heterogeneous_hidef_faithful_cpm.nodes")
    ap.add_argument("--ml-edges", default=None)
    ap.add_argument("--supra-graph", default="11_hidef_module/supra_graph.pkl")
    ap.add_argument("--sl-communities", default="../og_gene_set/output/combined26/hidef/hidef_communities.pkl")
    ap.add_argument("--sl-network", default="../og_gene_set/output/combined26/rwr/module_graph.pkl")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--allow-multiple-per-layer", action="store_true")
    ap.add_argument("--include", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--highlight", nargs="*", default=["OPRM1", "COMT", "KCNJ6"])
    ap.add_argument("--members", choices=["all", "shared"], default="all",
                    help="all sub-module members (default) or only SL-module genes")
    ap.add_argument("--xy-radius", type=float, default=10.0)
    ap.add_argument("--z-spacing", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, help="output folder (nodes.tsv, edges.tsv, ...)")
    ap.add_argument("--morph-steps", type=int, default=0, metavar="N",
                    help="also write sl_morph_00..sl_morph_NN layouts (N steps collapsed->aligned and "
                         "N steps aligned->stack, eased) so the viewer's layout dropdown can be stepped "
                         "through like an animation; 0 = off")
    ap.add_argument("--animate", default=None, metavar="FILE",
                    help="also render animation (.gif, or .mp4 if ffmpeg is available)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--hold-frames", type=int, default=30)
    ap.add_argument("--move-frames", type=int, default=75)
    ap.add_argument("--spin-degrees", type=float, default=120)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    edges_path = args.ml_edges or re.sub(r"\.nodes$", ".edges", args.ml_nodes)

    # ---------------- load (same as the figure script) ----------------
    with open(args.supra_graph, "rb") as f:
        g = pickle.load(f)
    ml = load_nodes(args.ml_nodes, g.vs["name"])
    parents = load_parents(edges_path)
    with open(args.sl_communities, "rb") as f:
        slc = pickle.load(f)
    slc = slc.get("communities", slc)
    key = next((k for k in slc if str(k) == str(args.sl_cluster)), None)
    if key is None:
        raise SystemExit(f"SL cluster {args.sl_cluster} not found")
    sl_genes = {norm_gene(m) for m in slc[key]["members"]}
    G_sl = None
    if args.sl_network and Path(args.sl_network).exists():
        with open(args.sl_network, "rb") as f:
            G_sl = pickle.load(f)

    mp = pd.read_csv(args.matched_pairs)
    sel, dropped = select_submodules(mp, args.sl_cluster, ml, parents, args.fdr,
                                     not args.allow_multiple_per_layer, args.include, args.exclude)
    print(f"SL {args.sl_cluster}: {len(sl_genes)} genes -> {len(sel)} sub-modules")
    for c, why in dropped.items():
        print(f"  dropped {c}: {why}")

    # ---------------- layers ----------------
    sl_layer = f"SL{args.sl_cluster}"
    used = Counter(sel["layer"])
    layer_of_sub = {m: (l if used[l] == 1 else f"{l}__{m}") for m, l in zip(sel["ml_cluster"], sel["layer"])}
    sub_order = list(sel["ml_cluster"])
    layer_order = [sl_layer] + [layer_of_sub[m] for m in sub_order]
    layer_idx = {l: i for i, l in enumerate(layer_order)}
    (out / "layer_order.txt").write_text(",".join(layer_order) + "\n")
    row_of = {r["ml_cluster"]: r for _, r in sel.iterrows()}
    hl = {h.upper() for h in args.highlight}

    # ---------------- nodes ----------------
    nodes = {}
    for gname in sorted(sl_genes):
        nodes[f"{sl_layer}::{gname}"] = dict(gene=gname, layer=sl_layer, sub=sl_layer,
                                             cell_type="single_layer", in_sl=True)
    assigned = {}
    for m in sub_order:
        for n in ml[m]["members"]:
            if args.members == "shared" and norm_gene(n) not in sl_genes:
                continue
            assigned.setdefault(n, m)
    for n, m in assigned.items():
        gname = norm_gene(n)
        nodes[n] = dict(gene=gname, layer=layer_of_sub[m], sub=m,
                        cell_type=n.rsplit("::", 1)[0] if "::" in n else "", in_sl=gname in sl_genes)

    # ---------------- edges ----------------
    edge_list, seen = [], set()

    def add(u, v, cls, etype=""):
        k = frozenset((u, v))
        if u == v or k in seen:
            return
        seen.add(k)
        if layer_idx[nodes[u]["layer"]] > layer_idx[nodes[v]["layer"]]:
            u, v = v, u
        lu, lv = nodes[u]["layer"], nodes[v]["layer"]
        edge_list.append({"source": u, "target": v, "edge_class_c": cls, "layer_source_c": lu,
                          "layer_target_c": lv, "layer_pair_c": lu if lu == lv else f"{lu}--{lv}",
                          "multiplex_etype_c": etype})

    if G_sl is not None:  # SL PPI edges
        rel = {n: norm_gene(n) for n in G_sl.nodes() if norm_gene(n) in sl_genes}
        for u, v in G_sl.subgraph(rel).edges():
            add(f"{sl_layer}::{rel[u]}", f"{sl_layer}::{rel[v]}", "intra_ppi")
    name_idx = {n: i for i, n in enumerate(g.vs["name"])}
    sub_g = g.induced_subgraph([name_idx[n] for n in assigned if n in name_idx])
    has_et = "etype" in sub_g.es.attributes()
    for e in sub_g.es:
        u, v = sub_g.vs[e.source]["name"], sub_g.vs[e.target]["name"]
        same = nodes[u]["layer"] == nodes[v]["layer"]
        add(u, v, "intra_ppi" if same else "inter_celltype", str(e["etype"]) if has_et else "")
    for n, d in nodes.items():  # SL gene -> its copies
        if d["layer"] != sl_layer and d["in_sl"]:
            add(f"{sl_layer}::{d['gene']}", n, "sl_to_submodule")
    edges = pd.DataFrame(edge_list)
    print(edges["edge_class_c"].value_counts().to_string())

    intra = [(r.source, r.target) for r in edges.itertuples() if r.layer_source_c == r.layer_target_c]
    nbr = defaultdict(set)
    for r in edges.itertuples():
        nbr[r.source].add(r.target)
        nbr[r.target].add(r.source)
    deg = Counter()
    for r in edges.itertuples():
        deg[r.source] += 1
        deg[r.target] += 1

    # ---------------- layouts ----------------
    H = nx.Graph()
    H.add_nodes_from(sl_genes)
    for r in edges[(edges["layer_source_c"] == sl_layer) & (edges["layer_target_c"] == sl_layer)].itertuples():
        H.add_edge(nodes[r.source]["gene"], nodes[r.target]["gene"])
    sl_pos = sl_unit_layout(H)
    layouts = build_layouts(nodes, intra, nbr, sl_layer, layer_idx, sl_pos, sub_order,
                            args.xy_radius, args.z_spacing, args.seed)

    if args.morph_steps > 0:
        ease = lambda t: t * t * (3 - 2 * t)
        seq = []
        for a_, b_ in [("sl_collapsed", "sl_aligned"), ("sl_aligned", "sl_stack")]:
            for t in np.linspace(0, 1, args.morph_steps + 1)[(1 if seq else 0):]:
                seq.append((a_, b_, ease(t)))
        for k, (a_, b_, t) in enumerate(seq):
            layouts[f"sl_morph_{k:02d}"] = {
                n: tuple((1 - t) * np.array(layouts[a_][n]) + t * np.array(layouts[b_][n])) for n in nodes}
        print(f"wrote {len(seq)} sl_morph_* layouts")

    # ---------------- node table ----------------
    gene_layers = defaultdict(set)
    for n, d in nodes.items():
        gene_layers[d["gene"]].add(d["layer"])
    rows = []
    for n, d in nodes.items():
        m = d["sub"]
        r = {"id": n, "gene": d["gene"], "layer_c": d["layer"], "layer_index_n": layer_idx[d["layer"]],
             "cell_type_c": d["cell_type"], "submodule_c": m,
             "in_sl_module_c": "sl_gene" if d["in_sl"] else "other_member",
             "is_highlight_c": "highlight" if d["gene"] in hl else "other",
             "gene_label_c": d["gene"] if d["gene"] in hl else "",
             "recovered_c": ("recovered" if len(gene_layers[d["gene"]]) > 1 else "not_recovered")
             if d["in_sl"] else "not_in_sl",
             "n_layers_with_gene_n": len(gene_layers[d["gene"]]),
             "degree_n": deg[n],
             "spatial_term_c": row_of[m]["spatial_term"] if m in row_of else ""}
        for name, pos in layouts.items():
            r[f"x_{name}"], r[f"y_{name}"], r[f"z_{name}"] = (round(float(c), 4) for c in pos[n])
        rows.append(r)
    nodes_df = pd.DataFrame(rows)
    nodes_df.to_csv(out / "nodes.tsv", sep="\t", index=False)
    edges.to_csv(out / "edges.tsv", sep="\t", index=False)
    sel.drop(columns=[c for c in sel.columns if c.startswith("_")]).to_csv(out / "submodules.csv", index=False)
    print(f"nodes.tsv {len(nodes_df):,} rows | edges.tsv {len(edges):,} rows | layers: {layer_order}")

    if args.animate:
        render_animation(nodes_df, edges, layouts, layer_order, args.animate, args.fps,
                         args.hold_frames, args.move_frames, args.spin_degrees)
    print(f"-> {out.resolve()}")


if __name__ == "__main__":
    main()