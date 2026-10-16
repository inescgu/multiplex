#!/usr/bin/env python3
"""
plot_sl_module_decomposition.py

Network figures showing how ONE single-layer painome HiDeF module (default
SL 6) is split into neuron-type-specific sub-modules in the multiplex
hierarchy.

SELECTION
---------
Start from matched_pairs.csv (compare_multiplex_vs_singlelayer.py) and keep
the multiplex clusters whose gene overlap with the SL module is significant
(hypergeom_q < --fdr). Then:
  1. Collapse nested clusters using the multiplex .edges hierarchy. If one
     candidate is an ancestor/descendant of another, only the one with the
     lowest q is kept, so no sub-module is counted twice.
  2. Keep one sub-module per dominant cell-type layer (the lowest q). Turn
     this off with --allow-multiple-per-layer.
--include / --exclude override this by hand.

OUTPUTS (in --out-dir, PNG + PDF)
-------
  <tag>_two_worlds        LEFT : the SL module in the single-layer painome
                                 network (module_graph.pkl PPI edges); each
                                 gene is a pie coloured by the multiplex
                                 sub-module(s) it ends up in.
                          RIGHT: the multiplex sub-modules, one bubble per
                                 neuron-type layer, drawn from the real
                                 supra-graph edges ('intra' PPI inside a
                                 layer, cross-cell-type edges between
                                 layers). SL-module genes are big coloured
                                 nodes, the other sub-module members small
                                 grey dots, and dashed arcs join copies of
                                 the same gene in different neuron types.
  <tag>_gene_submodule    bipartite gene <-> sub-module network: sub-modules
                          sit on a ring, genes are placed by force layout,
                          so genes recruited into several neuron types end
                          up between them.
  <tag>_submodules.csv    the selected sub-modules (layer, overlap, q, spatial
                          term, shared genes) plus what was dropped and why.
  <tag>_multiplex.graphml / <tag>_gene_submodule.graphml
                          the same networks with layout coordinates and
                          attributes, for polishing in Cytoscape/Gephi.

CAVEAT TO STATE IN THE FIGURE LEGEND
------------------------------------
Each multiplex sub-module's spatial term largely reflects which neuron-type
layer it lives in (layers and spatial terms are both DRG neuron classes), so
those labels show the layers kept their identity. They are not independent
evidence.
"""
import argparse
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Wedge

HIGHLIGHT_COLORS = ["#C0392B", "#6C3483", "#D35400", "#0E6655", "#1A5276", "#9A7D0A"]
LAYER_PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948",
                 "#B07AA1", "#FF9DA7", "#9C755F", "#86BCB6", "#D37295", "#8CD17D",
                 "#499894", "#F1CE63", "#A0CBE8", "#FFBE7D"]
SPATIAL_RE = re.compile(r"\s*(.+?) \(Spatial\) \[q=([0-9.eE+\-]+)\]")


# ============================================================================
# loading
# ============================================================================

def norm_gene(x) -> str:
    s = str(x).strip()
    return (s.rsplit("::", 1)[1] if "::" in s else s).upper()


def load_nodes(nodes_path, node_names):
    df = pd.read_csv(nodes_path, sep="\t", header=None,
                     names=["cluster_id", "size", "members", "persistence"],
                     dtype={"cluster_id": str, "members": str})
    out = {}
    for cid, mem, pers in zip(df["cluster_id"], df["members"], df["persistence"]):
        if pd.isna(mem) or not str(mem).strip():
            continue
        toks = [t for t in re.split(r"[,\s]+", str(mem).strip()) if t]
        if all(t.lstrip("-").isdigit() for t in toks[:5]):
            try:
                toks = [node_names[int(t)] for t in toks]
            except (IndexError, ValueError):
                pass
        cts = Counter(t.rsplit("::", 1)[0] for t in toks if "::" in t)
        layer, n = cts.most_common(1)[0] if cts else ("?", 0)
        out[cid] = {"members": toks, "persistence": float(pers),
                    "layer": layer, "layer_frac": n / max(len(toks), 1)}
    return out


def load_parents(edges_path):
    parents = defaultdict(set)
    if edges_path and Path(edges_path).exists():
        df = pd.read_csv(edges_path, sep="\t", header=None, usecols=[0, 1],
                         names=["parent", "child"], dtype=str)
        for p, c in zip(df["parent"], df["child"]):
            parents[c].add(p)
    else:
        print("⚠ no .edges file -- nested clusters will NOT be collapsed")
    return parents


def ancestors(c, parents):
    seen, stack = set(), [c]
    while stack:
        for p in parents.get(stack.pop(), ()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def spatial_term(s):
    best = None
    for part in str(s).split(";"):
        m = SPATIAL_RE.match(part)
        if m:
            q = float(m.group(2))
            if best is None or q < best[1]:
                best = (m.group(1).strip(), q)
    return best or ("no spatial term", np.nan)


# ============================================================================
# selection
# ============================================================================

def select_submodules(mp, sl_id, ml, parents, fdr, one_per_layer, include, exclude):
    mp = mp[mp["sl_cluster"].astype(str) == str(sl_id)].copy()
    if include:
        rows = mp[mp["ml_cluster"].isin(include)].copy()
        missing = set(include) - set(rows["ml_cluster"])
        if missing:
            print(f"⚠ --include clusters not in matched_pairs for SL {sl_id}: {sorted(missing)}")
    else:
        rows = mp[mp["hypergeom_q"] < fdr].copy()
    rows = rows[~rows["ml_cluster"].isin(exclude or [])]
    rows = rows[rows["ml_cluster"].isin(ml)]
    if rows.empty:
        raise SystemExit(f"No multiplex sub-modules selected for SL {sl_id}.")
    rows["layer"] = rows["ml_cluster"].map(lambda c: ml[c]["layer"])
    rows["layer_frac"] = rows["ml_cluster"].map(lambda c: ml[c]["layer_frac"])
    rows = rows.sort_values(["hypergeom_q", "gene_jaccard"], ascending=[True, False])

    dropped = {}
    if not include:
        anc = {c: ancestors(c, parents) for c in rows["ml_cluster"]}
        kept = []
        for c in rows["ml_cluster"]:
            clash = next((k for k in kept if k in anc[c] or c in anc[k]), None)
            if clash:
                dropped[c] = f"nested with {clash} (higher q)"
            else:
                kept.append(c)
        rows = rows[rows["ml_cluster"].isin(kept)]
        if one_per_layer:
            first = rows.drop_duplicates("layer", keep="first")
            for c, lay in zip(rows["ml_cluster"], rows["layer"]):
                if c not in set(first["ml_cluster"]):
                    dropped[c] = f"second sub-module in layer {lay} (higher q)"
            rows = first

    rows["spatial_term"], rows["spatial_q"] = zip(*rows["ml_lollipop_terms"].map(spatial_term))
    # order around the ring: group by fibre class prefix (A-/Ab-/C-/other), then layer
    rows["_cls"] = rows["layer"].str.extract(r"^([A-Za-z]+)")[0].fillna("")
    rows = rows.sort_values(["_cls", "layer"]).drop(columns="_cls")
    return rows.reset_index(drop=True), dropped


# ============================================================================
# drawing helpers
# ============================================================================

def disk_spread(nodes, lp):
    """Re-space a force layout evenly inside a unit disk: keep each node's
    angle, replace its radius by its radial rank. Stops dense cores and
    flung-out isolates from collapsing everything into a clump."""
    if len(nodes) == 1:
        return {nodes[0]: np.zeros(2)}
    arr = np.array([lp[n] for n in nodes], dtype=float)
    arr -= arr.mean(axis=0)
    ang = np.arctan2(arr[:, 1], arr[:, 0])
    rank = np.linalg.norm(arr, axis=1).argsort().argsort()
    rad = np.sqrt((rank + 0.5) / len(nodes))
    return {n: np.array([r * np.cos(a), r * np.sin(a)]) for n, r, a in zip(nodes, rad, ang)}


def set_limits(ax, xy, pad):
    xy = np.asarray(list(xy))
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def short_label(row):
    lay = row["layer"] + ("" if row["layer_frac"] >= 0.9 else f" ({row['layer_frac']:.0%})")
    return lay, f"{row['ml_cluster']} · {row['spatial_term']}"


def panel_singlelayer(ax, G_sl, sl_genes, gene_to_mods, mod_color, hcol, seed, label_all):
    H = nx.Graph()
    H.add_nodes_from(sl_genes)
    if G_sl is not None:
        rel = {n: norm_gene(n) for n in G_sl.nodes() if norm_gene(n) in sl_genes}
        H.add_edges_from((rel[u], rel[v]) for u, v in G_sl.subgraph(rel).edges() if rel[u] != rel[v])
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    giant = H.subgraph(comps[0]).copy()
    if len(giant) > 2:
        lp = nx.kamada_kawai_layout(giant)
        lp = disk_spread(list(giant), {n: 0.6 * lp[n] + 0.4 * disk_spread(list(giant), lp)[n]
                                       for n in giant}) if len(giant) > 8 else lp
    else:
        lp = {n: np.array([0.3 * i, 0.0]) for i, n in enumerate(giant)}
    pos = {n: np.asarray(v, float) for n, v in lp.items()}
    rest = [n for c in comps[1:] for n in sorted(c)]
    for i, n in enumerate(rest):  # small components / isolates on an outer ring
        th = np.pi / 2 + 2 * np.pi * i / max(len(rest), 1)
        pos[n] = 1.22 * np.array([np.cos(th), np.sin(th)])
    xy = np.array(list(pos.values()))
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) + np.eye(len(xy)) * 9
    r = float(np.clip(0.38 * np.median(d.min(axis=1)), 0.015, 0.06))
    segs = [(pos[u], pos[v]) for u, v in H.edges()]
    ax.add_collection(LineCollection(segs, colors="#9A9A9A", linewidths=0.7, alpha=0.6, zorder=1))
    for g, (x, y) in pos.items():
        mods = gene_to_mods.get(g, [])
        cols = [mod_color[m] for m in mods] or ["#E6E6E6"]
        n = len(cols)
        for i, c in enumerate(cols):
            ax.add_patch(Wedge((x, y), r, 90 + 360 * i / n, 90 + 360 * (i + 1) / n,
                               facecolor=c, edgecolor="white", lw=0.4, zorder=3))
        hc = hcol.get(g)
        ax.add_patch(Circle((x, y), r, fill=False, zorder=4,
                            edgecolor=hc or ("#333333" if mods else "#B5B5B5"),
                            lw=2.4 if hc else (0.8 if mods else 0.5)))
        if mods or label_all or hc:
            ax.text(x, y - r * 1.5, g, ha="center", va="top", zorder=5,
                    fontsize=9 if hc else 6.5, fontweight="bold" if hc else "normal",
                    color=hc or ("#222222" if mods else "#8A8A8A"))
    if rest:
        ax.text(0, -1.42, f"{len(rest)} gene(s) outside the module's main connected component",
                ha="center", fontsize=8, color="#777777", style="italic")
    set_limits(ax, list(pos.values()) + [(0, -1.45)], 0.15)
    return H, pos


def layout_multiplex(g, sel, ml, sl_genes, members_mode, seed):
    """Compute (once) everything that must stay identical between the combined
    figure and every animation frame: which nodes sit in which bubble, bubble
    geometry, and every node position."""
    assigned, groups = {}, []
    for m in sel["ml_cluster"]:
        nodes = [n for n in ml[m]["members"] if n not in assigned]
        if members_mode == "shared":
            nodes = [n for n in nodes if norm_gene(n) in sl_genes]
        for n in nodes:
            assigned[n] = m
        groups.append((m, nodes))

    name_idx = {n: i for i, n in enumerate(g.vs["name"])}
    idxs = [name_idx[n] for n in assigned if n in name_idx]
    sub = g.induced_subgraph(idxs)
    has_et = "etype" in sub.es.attributes()
    edges = [(sub.vs[e.source]["name"], sub.vs[e.target]["name"], e["etype"] if has_et else "intra")
             for e in sub.es]

    radii = {m: float(np.clip(0.11 * np.sqrt(len(nodes)), 0.16, 0.75)) for m, nodes in groups}
    gap = 0.12
    circumference = sum(2 * radii[m] + gap for m, _ in groups)
    R = max(circumference / (2 * np.pi), 0.9)
    centers, theta, cum = {}, {}, 0.0
    for m, _ in groups:
        cum += radii[m] + gap / 2
        th = np.pi / 2 - cum / R
        centers[m] = np.array([R * np.cos(th), R * np.sin(th)])
        theta[m] = th
        cum += radii[m] + gap / 2

    pos = {}
    by_group = defaultdict(list)
    for u, v, et in edges:
        if assigned.get(u) == assigned.get(v):
            by_group[assigned[u]].append((u, v))
    for k, (m, nodes) in enumerate(groups):
        if not nodes:
            continue
        L = nx.Graph()
        L.add_nodes_from(nodes)
        L.add_edges_from(by_group[m])
        lp = nx.spring_layout(L, seed=seed + k, k=1.5 / np.sqrt(max(len(L), 1)), iterations=200)
        for n, p in disk_spread(nodes, lp).items():
            pos[n] = centers[m] + p * radii[m] * 0.92
    return {"assigned": assigned, "groups": groups, "edges": edges, "radii": radii,
            "centers": centers, "theta": theta, "pos": pos}


def draw_multiplex(ax, S, sel, sl_genes, mod_color, hcol, link_all_shared=False, *,
                   fs=1.0, ns=1.0, hl_fs=8.5, bubbles_only=False, gene_labels=True,
                   dim=False, detail=False, partner_labels=False):
    """Draw the multiplex panel from a layout_multiplex() result.
    Defaults reproduce the right-hand panel of the combined figure exactly.
    Frame options: dim (fade everything except highlighted genes), detail
    (bold the highlighted gene's within-neuron-type partners and name them),
    gene_labels (small labels on all SL-module genes), bubbles_only."""
    import matplotlib.patheffects as pe
    assigned, groups, edges = S["assigned"], S["groups"], S["edges"]
    radii, centers, theta, pos = S["radii"], S["centers"], S["theta"], S["pos"]
    faded = dim and bool(hcol)
    halo = [pe.withStroke(linewidth=3, foreground="white")] if detail else None

    row_of = {r["ml_cluster"]: r for _, r in sel.iterrows()}
    for m, _ in groups:
        ax.add_patch(Circle(centers[m], radii[m] * 1.12, facecolor=mod_color[m], alpha=0.08,
                            edgecolor=mod_color[m], lw=1.2, zorder=0))
        lay, sub_lab = short_label(row_of[m])
        lp = centers[m] + (radii[m] * 1.12 + 0.12) * np.array([np.cos(theta[m]), np.sin(theta[m])])
        ha = "left" if np.cos(theta[m]) > 0.25 else ("right" if np.cos(theta[m]) < -0.25 else "center")
        va = "bottom" if np.sin(theta[m]) > 0.25 else ("top" if np.sin(theta[m]) < -0.25 else "center")
        ax.text(lp[0], lp[1], f"{lay}\n{sub_lab}", ha=ha, va=va, fontsize=8.5 * fs,
                color=mod_color[m], fontweight="bold", zorder=6)
    pad = max(radii.values()) + 0.9
    set_limits(ax, list(centers.values()), pad)
    if bubbles_only:
        return

    intra, intra_c, cross = [], [], []
    for u, v, et in edges:
        if u in pos and v in pos:
            if assigned[u] == assigned[v]:
                intra.append((pos[u], pos[v]))
                intra_c.append(mod_color[assigned[u]])
            else:
                cross.append((pos[u], pos[v]))
    ax.add_collection(LineCollection(cross, colors="#7F7F7F", linewidths=0.4,
                                     alpha=0.06 if faded else 0.18, zorder=1))
    ax.add_collection(LineCollection(intra, colors=intra_c, linewidths=0.7,
                                     alpha=0.22 if faded else 0.55, zorder=2))

    a_node = 0.28 if faded else 1.0
    other = [n for n in pos if norm_gene(n) not in sl_genes]
    if other:
        o = np.array([pos[n] for n in other])
        ax.scatter(o[:, 0], o[:, 1], s=7 * ns, color="#C4C4C4", edgecolors="none", zorder=3,
                   alpha=a_node)
    for n, (x, y) in pos.items():
        gname = norm_gene(n)
        if gname not in sl_genes:
            continue
        hc = hcol.get(gname)
        ax.scatter([x], [y], s=(95 if hc else 42) * ns * (1.5 if (hc and detail) else 1.0),
                   color=mod_color[assigned[n]], zorder=4.5 if hc else 4,
                   edgecolors=hc or "#222222", linewidths=2.6 if hc else 0.5,
                   alpha=1.0 if hc else a_node)
        if hc or gene_labels:
            ax.text(x, y - 0.045, gname, ha="center", va="top", zorder=6 if hc else 5,
                    fontsize=hl_fs if hc else 5.8 * fs, fontweight="bold" if hc else "normal",
                    color=hc or "#222222", alpha=1.0 if hc else a_node,
                    path_effects=halo if hc else None)

    if detail:  # the highlighted gene's partners inside each neuron-type sub-module
        adj = defaultdict(list)
        for u, v, et in edges:
            if u in pos and v in pos and assigned[u] == assigned[v]:
                adj[u].append(v)
                adj[v].append(u)
        for n in pos:
            gname = norm_gene(n)
            if gname not in hcol:
                continue
            hc = hcol[gname]
            nbrs = sorted(set(adj[n]), key=lambda q: (norm_gene(q) not in sl_genes, norm_gene(q)))
            for q in nbrs:
                ax.plot(*zip(pos[n], pos[q]), color=hc, lw=1.7, alpha=0.85, zorder=3.2)
            for q in nbrs:
                is_sl = norm_gene(q) in sl_genes
                ax.scatter([pos[q][0]], [pos[q][1]], s=(46 if is_sl else 24) * ns, zorder=4.2,
                           color=mod_color[assigned[q]] if is_sl else "#8A8A8A",
                           edgecolors=hc, linewidths=1.2)
            if partner_labels:
                for q in nbrs:
                    ax.text(pos[q][0], pos[q][1] - 0.04, norm_gene(q), ha="center", va="top", zorder=5.5,
                            fontsize=7.2 * fs, color="#1A1A1A", path_effects=halo)

    copies = defaultdict(list)
    for n in pos:
        if norm_gene(n) in sl_genes:
            copies[norm_gene(n)].append(n)
    for gname, ns_ in copies.items():
        if len(ns_) < 2 or not (gname in hcol or link_all_shared):
            continue
        for i in range(len(ns_)):
            for j in range(i + 1, len(ns_)):
                ax.add_patch(FancyArrowPatch(pos[ns_[i]], pos[ns_[j]], arrowstyle="-",
                                             connectionstyle="arc3,rad=0.25", linestyle=(0, (4, 3)),
                                             color=hcol.get(gname, "#555555"),
                                             lw=(2.4 if detail else 1.8) if gname in hcol else 0.8,
                                             alpha=0.9 if gname in hcol else 0.45, zorder=3.5))


def draw_cards(fig, S, sel, sl_genes, mod_color, hcol, fs, ns, max_partners):
    """Strip of zoom cards under the main plot: one per copy of each highlighted
    gene, showing the gene and its partners inside that neuron-type sub-module."""
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    pos, assigned, edges = S["pos"], S["assigned"], S["edges"]
    adj = defaultdict(set)
    for u, v, et in edges:
        if u in pos and v in pos and assigned[u] == assigned[v]:
            adj[u].add(v)
            adj[v].add(u)
    layer_of = dict(zip(sel["ml_cluster"], sel["layer"]))
    cards = [(gname, n, m) for gname in hcol for m, nodes in S["groups"] for n in nodes
             if norm_gene(n) == gname and n in pos]
    k = len(cards)
    if not k:
        return
    w = min(0.19, 0.94 / k)
    x0 = 0.5 - w * k / 2
    for i, (gname, n, m) in enumerate(cards):
        cax = fig.add_axes([x0 + i * w + 0.004, 0.075, w - 0.008, 0.205])
        nbrs = sorted(adj[n], key=lambda q: (norm_gene(q) not in sl_genes, norm_gene(q)))
        shown = nbrs[:max_partners]
        for sp in cax.spines.values():
            sp.set_edgecolor(hcol[gname])
            sp.set_linewidth(2.2)
        cax.set_xticks([])
        cax.set_yticks([])
        cax.set_xlim(-1.95, 1.95)
        cax.set_ylim(-1.6, 1.6)
        cax.set_aspect("equal")
        cax.set_facecolor(matplotlib.colors.to_rgba(mod_color[m], 0.07))
        cax.set_title(f"{gname} in {layer_of[m]}", fontsize=11.5 * fs * 0.85, fontweight="bold",
                      color=tuple(0.72 * c for c in matplotlib.colors.to_rgb(mod_color[m])), pad=5)
        ang = {q: np.pi / 2 - 2 * np.pi * j / max(len(shown), 1) for j, q in enumerate(shown)}
        pp = {q: np.array([np.cos(a), np.sin(a)]) for q, a in ang.items()}
        for q in shown:
            for r in adj[q] & set(shown):
                if norm_gene(q) < norm_gene(r):
                    cax.plot(*zip(pp[q], pp[r]), color="#9A9A9A", lw=0.7, alpha=0.5, zorder=1)
        for q in shown:
            cax.plot([0, pp[q][0]], [0, pp[q][1]], color=hcol[gname], lw=1.8, alpha=0.85, zorder=2)
        for q in shown:
            is_sl = norm_gene(q) in sl_genes
            cax.scatter([pp[q][0]], [pp[q][1]], s=(190 if is_sl else 120) * ns * 0.8, zorder=3,
                        color=mod_color[m] if is_sl else "#9A9A9A", edgecolors=hcol[gname], linewidths=1.2)
            a = ang[q]
            cax.text(1.3 * np.cos(a), 1.3 * np.sin(a), norm_gene(q), fontsize=8.5 * fs * 0.85,
                     fontweight="bold" if is_sl else "normal", color="#1A1A1A", zorder=5,
                     ha="left" if np.cos(a) > 0.3 else ("right" if np.cos(a) < -0.3 else "center"),
                     va="bottom" if np.sin(a) > 0.3 else ("top" if np.sin(a) < -0.3 else "center"),
                     path_effects=halo)
        cax.scatter([0], [0], s=1500 * ns * 0.8, color=mod_color[m], edgecolors=hcol[gname],
                    linewidths=3, zorder=4)
        cax.text(0, 0, gname, fontsize=8.5 * fs * 0.85, fontweight="bold", ha="center", va="center",
                 color="white", zorder=6, path_effects=[pe.withStroke(linewidth=2, foreground="#222222")])
        more = len(nbrs) - len(shown)
        cax.text(0.04, 0.96, f"{len(nbrs)} partner(s)" + (f"\n({more} not shown)" if more > 0 else ""),
                 transform=cax.transAxes, fontsize=8 * fs * 0.85, ha="left", va="top",
                 color="#555555", style="italic")


def export_graph(S, sl_genes, hcol):
    X = nx.Graph()
    for n, (x, y) in S["pos"].items():
        X.add_node(n, gene=norm_gene(n), cell_type=n.rsplit("::", 1)[0] if "::" in n else "",
                   submodule=S["assigned"][n], in_sl_module=norm_gene(n) in sl_genes,
                   highlight=norm_gene(n) in hcol, x=float(x * 500), y=float(-y * 500))
    for u, v, et in S["edges"]:
        if u in S["pos"] and v in S["pos"]:
            X.add_edge(u, v, etype=str(et), within_submodule=S["assigned"][u] == S["assigned"][v])
    return X


def make_frames(S, sel, sl_genes, sl_id, gene_to_mods, mod_color, base_hcol, steps, out, tag,
                fs, ns, dpi, write_svg, max_partners):
    """One image per step, ALL on an identical canvas (same size, same axes
    limits, same node positions, no tight-bbox cropping) so frames line up
    exactly when stacked or morphed in PowerPoint."""
    import textwrap
    hc_all = dict(base_hcol)
    for spec in steps:
        for gname in [x.strip().upper() for x in spec.split(",") if x.strip()]:
            if gname not in ("NONE", "BUBBLES") and gname not in hc_all:
                hc_all[gname] = HIGHLIGHT_COLORS[len(hc_all) % len(HIGHLIGHT_COLORS)]
    layer_of = dict(zip(sel["ml_cluster"], sel["layer"]))
    written = []
    for i, spec in enumerate(steps, 1):
        key = spec.strip().lower()
        bubbles_only = key == "bubbles"
        genes = [] if key in ("none", "", "bubbles") else [x.strip().upper() for x in spec.split(",") if x.strip()]
        hcol = {gname: hc_all[gname] for gname in genes}
        for gname in genes:
            if gname not in gene_to_mods:
                print(f"  ⚠ frame {i}: {gname} is in none of the selected sub-modules -- nothing to highlight")

        fig = plt.figure(figsize=(16, 13.2))
        ax = fig.add_axes([0.02, 0.30, 0.96, 0.64])   # identical in every frame
        draw_multiplex(ax, S, sel, sl_genes, mod_color, hcol, False, fs=fs, ns=ns,
                       hl_fs=8.5 * fs * 1.45, bubbles_only=bubbles_only, gene_labels=False,
                       dim=True, detail=bool(genes))
        if genes:
            draw_cards(fig, S, sel, sl_genes, mod_color, hcol, fs, ns, max_partners)
        fig.text(0.5, 0.972, f"Multiplex · SL {sl_id} split into {len(sel)} neuron-type-specific sub-modules",
                 ha="center", va="center", fontsize=17 * fs * 0.85, fontweight="bold")
        if not genes:
            cap = (f"{len(sl_genes)}-gene single-layer module → {len(sel)} sub-modules, one per neuron type "
                   f"(coloured nodes = SL {sl_id} genes, grey dots = other cluster members)"
                   if not bubbles_only else
                   f"Each circle is one multiplex HiDeF cluster, labelled by neuron-type layer")
        else:
            parts = []
            for gname in genes:
                mods = gene_to_mods.get(gname, [])
                parts.append(f"{gname} is recruited into {len(mods)} neuron-type sub-module"
                             + ("s" if len(mods) != 1 else "")
                             + (": " + ", ".join(layer_of[m] for m in mods) if mods else ""))
            cap = "   |   ".join(parts) + "   (cards = its partners in each neuron type)"
        fig.text(0.5, 0.03, "\n".join(textwrap.wrap(cap, 118)), ha="center", va="center",
                 fontsize=12.5 * fs * 0.85, color="#333333")
        name = f"{tag}_frame{i}_" + ("bubbles" if bubbles_only else ("_".join(genes) if genes else "all_nodes"))
        fig.savefig(out / f"{name}.png", dpi=dpi, facecolor="white")
        if write_svg:
            with matplotlib.rc_context({"svg.fonttype": "none"}):
                fig.savefig(out / f"{name}.svg", facecolor="white")
        plt.close(fig)
        written.append(name)
        print(f"  frame {i}: {name}")
    return written


def figure_bipartite(sel, gene_to_mods, mod_color, hcol, seed, title):
    B = nx.Graph()
    mods = list(sel["ml_cluster"])
    init = {}
    for k, m in enumerate(mods):
        th = np.pi / 2 - 2 * np.pi * k / len(mods)
        init[("M", m)] = np.array([np.cos(th), np.sin(th)])
        B.add_node(("M", m))
    rng = np.random.default_rng(seed)
    singles = defaultdict(list)
    for gname, ms in gene_to_mods.items():
        for m in ms:
            B.add_edge(("G", gname), ("M", m))
        if len(ms) == 1:
            singles[ms[0]].append(gname)
        else:
            init[("G", gname)] = np.mean([init[("M", m)] for m in ms], axis=0) * 0.6 + rng.normal(0, 0.05, 2)
    fixed = [("M", m) for m in mods]
    step = 2 * np.pi / max(len(mods), 1)
    for m, gl in singles.items():  # fan of single-module genes just inside their module
        th0 = np.arctan2(*init[("M", m)][::-1])
        for i, gname in enumerate(sorted(gl)):
            off = (i - (len(gl) - 1) / 2) * min(step * 0.7 / max(len(gl), 1), 0.12)
            rr = 0.80 - 0.07 * (i % 2)
            init[("G", gname)] = rr * np.array([np.cos(th0 + off), np.sin(th0 + off)])
            fixed.append(("G", gname))
    free = [n for n in B if n not in fixed]
    pos = nx.spring_layout(B, pos=init, fixed=fixed, seed=seed,
                           k=1.4 / np.sqrt(max(len(free), 1) + 4), iterations=400) if free else init
    for n in free:  # shared genes belong inside the ring, between their sub-modules
        rr = np.linalg.norm(pos[n])
        if rr > 0.68:
            pos[n] = pos[n] * 0.68 / rr

    fig, ax = plt.subplots(figsize=(12, 12))
    for (a, b) in B.edges():
        gnode, mnode = (a, b) if a[0] == "G" else (b, a)
        ax.plot(*zip(pos[gnode], pos[mnode]), color=mod_color[mnode[1]], lw=1.0, alpha=0.5, zorder=1)
    row_of = {r["ml_cluster"]: r for _, r in sel.iterrows()}
    for m in mods:
        x, y = pos[("M", m)]
        ax.scatter([x], [y], s=1100, marker="s", color=mod_color[m], edgecolors="#222222",
                   linewidths=0.8, zorder=3)
        ax.text(x, y, str(len([g for g, ms in gene_to_mods.items() if m in ms])), ha="center",
                va="center", fontsize=10, fontweight="bold", color="white", zorder=4)
        lay, sub_lab = short_label(row_of[m])
        th = np.arctan2(y, x)
        ax.text(x * 1.2, y * 1.2, f"{lay}\n{sub_lab}", fontsize=9, fontweight="bold",
                color=mod_color[m], zorder=5,
                ha="left" if np.cos(th) > 0.25 else ("right" if np.cos(th) < -0.25 else "center"),
                va="bottom" if np.sin(th) > 0.25 else ("top" if np.sin(th) < -0.25 else "center"))
    for gname, ms in gene_to_mods.items():
        x, y = pos[("G", gname)]
        hc = hcol.get(gname)
        k = len(ms)
        ax.scatter([x], [y], s=(160 if hc else 25 + 30 * (k - 1)), zorder=4,
                   color=hc or ("#333333" if k > 1 else "#9A9A9A"),
                   edgecolors="white", linewidths=0.8)
        ax.text(x, y + (0.03 if k > 1 or hc else -0.025), gname, ha="center",
                va="bottom" if (k > 1 or hc) else "top", zorder=5,
                fontsize=11 if hc else (8 if k > 1 else 6.5),
                fontweight="bold" if (hc or k > 1) else "normal",
                color=hc or ("#222222" if k > 1 else "#6E6E6E"))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", color="#888888", markersize=12,
               label="multiplex sub-module (number = SL-module genes it contains)"),
        Line2D([], [], marker="o", ls="", color="#333333", markersize=8,
               label="gene in >1 neuron-type sub-module"),
        Line2D([], [], marker="o", ls="", color="#9A9A9A", markersize=5, label="gene in 1 sub-module"),
        *[Line2D([], [], marker="o", ls="", color=c, markersize=9, label=g) for g, c in hcol.items()
          if g in gene_to_mods]],
        loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=9, frameon=False)
    xy = np.array(list(pos.values()))
    lim = np.abs(xy).max() + 0.55
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    G = nx.Graph()
    for n, (x, y) in pos.items():
        G.add_node(f"{n[0]}:{n[1]}", kind="submodule" if n[0] == "M" else "gene",
                   label=n[1], x=float(x * 500), y=float(-y * 500),
                   layer=row_of[n[1]]["layer"] if n[0] == "M" else "",
                   n_submodules=len(gene_to_mods.get(n[1], [])) if n[0] == "G" else 0,
                   highlight=n[0] == "G" and n[1] in hcol)
    G.add_edges_from((f"{a[0]}:{a[1]}", f"{b[0]}:{b[1]}") for a, b in B.edges())
    return fig, G


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = "11_hidef_module/results_cpm"
    ap.add_argument("--sl-cluster", default="6")
    ap.add_argument("--matched-pairs", default=f"{base}/compare_singlelayer/matched_pairs.csv")
    ap.add_argument("--ml-nodes", default=f"{base}/heterogeneous_hidef_faithful_cpm.nodes")
    ap.add_argument("--ml-edges", default=None, help="default: --ml-nodes with .edges")
    ap.add_argument("--supra-graph", default="11_hidef_module/supra_graph.pkl")
    ap.add_argument("--sl-communities", default="../og_gene_set/output/combined26/hidef/hidef_communities.pkl")
    ap.add_argument("--sl-network", default="../og_gene_set/output/combined26/rwr/module_graph.pkl")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--allow-multiple-per-layer", action="store_true")
    ap.add_argument("--include", nargs="*", default=None, help="use exactly these multiplex clusters")
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--highlight", nargs="*", default=["OPRM1", "KCNJ6"])
    ap.add_argument("--members", choices=["all", "shared"], default="all",
                    help="right panel: draw every sub-module member (default) or only SL-module genes")
    ap.add_argument("--link-all-shared", action="store_true",
                    help="dashed same-gene arcs for every recurring gene, not just --highlight")
    ap.add_argument("--label-all", action="store_true", help="left panel: label genes in no sub-module too")
    ap.add_argument("--frames", action="store_true",
                    help="also write step-through frames of the multiplex panel (for PowerPoint)")
    ap.add_argument("--steps", nargs="*", default=None,
                    help='frame definitions, one per frame; comma-separate genes to highlight together. '
                         'Keywords: none (all nodes, no highlight), bubbles (circles only). '
                         'Default: none OPRM1 OPRM1,KCNJ6   e.g. --steps bubbles none OPRM1 OPRM1,KCNJ6')
    ap.add_argument("--frame-font-scale", type=float, default=1.3)
    ap.add_argument("--frame-node-scale", type=float, default=1.3)
    ap.add_argument("--frame-dpi", type=int, default=200)
    ap.add_argument("--max-partner-labels", type=int, default=12,
                    help="per highlighted gene copy, how many partner genes to name")
    ap.add_argument("--no-svg", action="store_true", help="frames: skip the editable SVG copies")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=f"{base}/compare_singlelayer/figures")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"SL{args.sl_cluster}"
    edges_path = args.ml_edges or re.sub(r"\.nodes$", ".edges", args.ml_nodes)

    with open(args.supra_graph, "rb") as f:
        g = pickle.load(f)
    ml = load_nodes(args.ml_nodes, g.vs["name"])
    parents = load_parents(edges_path)
    with open(args.sl_communities, "rb") as f:
        slc = pickle.load(f)
    slc = slc.get("communities", slc)
    key = next((k for k in slc if str(k) == str(args.sl_cluster)), None)
    if key is None:
        raise SystemExit(f"SL cluster {args.sl_cluster} not in {args.sl_communities}")
    sl_genes = {norm_gene(m) for m in slc[key]["members"]}
    sl_pers = slc[key].get("persistence", np.nan)
    G_sl = None
    if args.sl_network and Path(args.sl_network).exists():
        with open(args.sl_network, "rb") as f:
            G_sl = pickle.load(f)

    mp = pd.read_csv(args.matched_pairs)
    sel, dropped = select_submodules(mp, args.sl_cluster, ml, parents, args.fdr,
                                     not args.allow_multiple_per_layer, args.include, args.exclude)
    print(f"SL {args.sl_cluster}: {len(sl_genes)} genes -> {len(sel)} multiplex sub-modules selected")
    for c, why in dropped.items():
        print(f"  dropped {c}: {why}")

    pal = (LAYER_PALETTE * 4)[:len(sel)]
    mod_color = dict(zip(sel["ml_cluster"], pal))
    hcol = {h.upper(): HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)] for i, h in enumerate(args.highlight)}
    gene_to_mods = defaultdict(list)
    for m in sel["ml_cluster"]:
        for gname in sorted({norm_gene(n) for n in ml[m]["members"]} & sl_genes):
            gene_to_mods[gname].append(m)
    missing_h = [h for h in hcol if h not in gene_to_mods]
    if missing_h:
        print(f"  note: highlight gene(s) {missing_h} are in no selected sub-module")

    # ---------------- table ----------------
    tab = sel[["ml_cluster", "layer", "layer_frac", "gene_jaccard", "gene_overlap_coef",
               "hypergeom_q", "spatial_term", "spatial_q", "ml_lollipop_terms"]].copy()
    tab["n_sl_genes_in_submodule"] = tab["ml_cluster"].map(
        lambda m: sum(m in ms for ms in gene_to_mods.values()))
    tab["sl_genes_in_submodule"] = tab["ml_cluster"].map(
        lambda m: ",".join(gn for gn, ms in sorted(gene_to_mods.items()) if m in ms))
    tab["n_members"] = tab["ml_cluster"].map(lambda m: len(ml[m]["members"]))
    tab["persistence"] = tab["ml_cluster"].map(lambda m: ml[m]["persistence"])
    drop_df = pd.DataFrame([{"ml_cluster": c, "dropped_reason": w} for c, w in dropped.items()])
    pd.concat([tab, drop_df], ignore_index=True).to_csv(out / f"{tag}_submodules.csv", index=False)

    # ---------------- figure 1: two worlds ----------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(24, 12.5), gridspec_kw={"width_ratios": [1, 1.45]})
    panel_singlelayer(axL, G_sl, sl_genes, gene_to_mods, mod_color, hcol, args.seed, args.label_all)
    n_cov = len(gene_to_mods)
    axL.set_title(f"Single-layer painome · SL {args.sl_cluster}\n{len(sl_genes)} genes, persistence "
                  f"{sl_pers:g}; {n_cov} genes recovered in multiplex sub-modules\n"
                  f"(pie = neuron-type sub-module(s) each gene falls into; grey = none)",
                  fontsize=12.5, fontweight="bold")
    S = layout_multiplex(g, sel, ml, sl_genes, args.members, args.seed)
    draw_multiplex(axR, S, sel, sl_genes, mod_color, hcol, args.link_all_shared)
    X = export_graph(S, sl_genes, hcol)
    axR.set_title(f"Multiplex · the same genes split into {len(sel)} neuron-type-specific sub-modules\n"
                  f"(bubble = one HiDeF cluster, coloured nodes = SL {args.sl_cluster} genes, "
                  f"grey dots = other members)", fontsize=12.5, fontweight="bold")
    handles = [Patch(facecolor=mod_color[r["ml_cluster"]],
                     label=f"{r['layer']}  —  {r['ml_cluster']}  —  {r['spatial_term']}"
                           + (f" (q={r['spatial_q']:.1e})" if pd.notna(r["spatial_q"]) else "")
                           + f"  —  overlap {r['gene_overlap_coef']:.2f}, q={r['hypergeom_q']:.1e}")
               for _, r in sel.iterrows()]
    handles += [Line2D([], [], color="#9A9A9A", lw=1.5, label="PPI edge (single-layer / within neuron type)"),
                Line2D([], [], color="#7F7F7F", lw=1, alpha=0.4, label="cross-cell-type edge (multiplex)"),
                *[Line2D([], [], color=c, lw=2, ls=(0, (4, 3)), marker="o", markersize=8,
                         markerfacecolor="white", markeredgecolor=c, markeredgewidth=2,
                         label=f"{gname}: same gene, different neuron types")
                  for gname, c in hcol.items() if gname in gene_to_mods]]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02 - 0.012 * len(handles)))
    fig.text(0.5, 0.005, "Note: a multiplex sub-module's spatial term largely reflects its neuron-type "
             "layer (both are DRG neuron classes) -- read it as layer identity, not independent evidence.",
             ha="center", fontsize=9, style="italic", color="#555555")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{tag}_two_worlds.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    nx.write_graphml(X, out / f"{tag}_multiplex.graphml")

    # ---------------- figure 2: gene <-> sub-module ----------------
    fig2, B = figure_bipartite(sel, gene_to_mods, mod_color, hcol, args.seed,
                               f"SL {args.sl_cluster} genes across multiplex neuron-type sub-modules")
    for ext in ("png", "pdf"):
        fig2.savefig(out / f"{tag}_gene_submodule.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    nx.write_graphml(B, out / f"{tag}_gene_submodule.graphml")

    if args.frames or args.steps is not None:
        steps = args.steps if args.steps else ["none", "OPRM1", "OPRM1,KCNJ6"]
        print(f"Writing {len(steps)} animation frame(s):")
        make_frames(S, sel, sl_genes, args.sl_cluster, gene_to_mods, mod_color, hcol, steps, out, tag,
                    args.frame_font_scale, args.frame_node_scale, args.frame_dpi,
                    not args.no_svg, args.max_partner_labels)

    multi = sorted((g_ for g_, ms in gene_to_mods.items() if len(ms) > 1),
                   key=lambda g_: -len(gene_to_mods[g_]))
    print(f"Genes in >1 neuron-type sub-module: "
          + ", ".join(f"{g_}({len(gene_to_mods[g_])})" for g_ in multi))
    print(f"Saved figures, table and GraphML to {out}/")


if __name__ == "__main__":
    main()