#!/usr/bin/env python3
"""
fig_step7d_neighbors.py -- deck figures from step7d_seed_layer_profile.py,
run in NEIGHBOR mode (--disease-module-csv), which is the run the deck uses.

Figures written:
    fig17a_tier_breakdown.png        tier composition + zoom on the A / A* / B shortlist
    fig17b_candidate_heatmap.png     candidate x layer heatmap
    fig17c_layer_percentiles.png     per-layer percentile distributions
    fig18_seeds_vs_neighbors.png     ONE continuous figure: seeds vs neighbours across
                                     degree AUC, avg_log2FC AUC and degree-matched z,
                                     sharing a single cell-type axis (replaces fig18a + fig18b)

Reads (from --step7d-dir, i.e. step 7d's --output-dir, e.g. 07d_module_neighbors):
    layer_seed_tests.csv             per-layer tests for the NEIGHBOURS
    reference_seed_layer_tests.csv   the same tests for the real SEEDS
    seed_layer_presence.csv          long table: gene x layer, degree / expression stats
    seed_layer_summary.csv           per gene: specificity_tier, best layers, scores

Writes everything to --out-dir.

Design rules shared by every figure
-----------------------------------
  * One layer order (fiber type: A, C, ATF3, rest; alphabetical within), with
    thin rules between fiber groups and tick labels coloured by fiber type.
  * Genes are in a layer because they are significantly more expressed there
    than other genes; the figures describe them that way and nowhere refer to
    them as "marker" genes.
  * Larger type than before (see TICK / LAB / PANEL below) and figure heights
    that scale with the number of rows, so labels no longer collide.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from painfig import (C, CMAP_WARM, FS, SLOT, caption, fiber_color, figsize, save,
                     shorten_layer, sig_colors, use_theme)

# ---------------------------------------------------------------------------
# Type sizes (pt). Floors on top of painfig's FS so everything is a bit larger.
# ---------------------------------------------------------------------------
TICK = max(FS["tick"], 8.5)
LAB = max(FS["tick"], 9.5)
PANEL = max(FS["panel"], 10.5)
ANNOT = max(FS["annot"], 9.0)
ROW = 8.0            # dense row labels (layers, genes)

# ---------------------------------------------------------------------------
# One layer order for every figure in the deck
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

#: hierarchy key -> the fiber name painfig's fiber_color() already knows
FIBER_ALIAS = {"Non-classified": "ATF3"}

FIBER_OF, SUBTYPE_OF, MARKER_OF, ORDER_OF = {}, {}, {}, {}
for _f, _spec in DRG_HIERARCHY.items():
    for _sub, _s in _spec["subcategories"].items():
        for _ct in _s["cell_types"]:
            FIBER_OF[_ct] = FIBER_ALIAS.get(_f, _f)
            SUBTYPE_OF[_ct] = _sub
            MARKER_OF[_ct] = _s["marker"]
            ORDER_OF[_ct] = len(ORDER_OF)

FIBER_SEQ = ["A-fiber", "C-fiber", "ATF3", "Non-neuronal", "Unclassified"]


#: Short tier labels, matching the wording on the deck's definitions slide.
TIER_SHORT = {
    "A: significantly and specifically expressed gene + well-connected in an enriched layer": "A  specific + well-connected",
    "A*: significant + specific + connected, but also a generic global hub": "A*  specific + connected, global hub",
    "B: significantly and specifically expressed gene, not well-connected there": "B  specific, not well-connected",
    "C: hub in an enriched layer, not a distinct gene": "C  hub, not a distinct gene",
    "C: hub in an enriched layer (no expression data to assess specificity)": "C  hub (no expression data)",
    "D: generic hub, no enriched layer": "D  generic hub, no enriched layer",
    "F: present, unremarkable": "F  present, unremarkable",
    "E: absent from all cell-type layers (global_ppi only)": "E  absent from every layer",
}
SHORTLIST_TIERS = ("A:", "A*:", "B:")

def fiber_of(layer: str) -> str:
    return FIBER_OF.get(str(layer), "Unclassified")


def subtype_of(layer: str) -> str:
    return SUBTYPE_OF.get(str(layer), "Unclassified")

def layer_order(layers) -> list[str]:
    """Hierarchy order (fiber > subtype > cell type); unknown layers go last, alphabetical."""
    present = [l for l in set(layers) if str(l) != "global_ppi"]
    return sorted(present, key=lambda l: (ORDER_OF.get(str(l), len(ORDER_OF)), str(l)))


def fiber_bounds(layers) -> list[int]:
    """Indices where layer i starts a new fiber group."""
    out, prev = [], None
    for i, l in enumerate(layers):
        f = fiber_of(l)
        if prev is not None and f != prev:
            out.append(i)
        prev = f
    return out


def subtype_spans(layers) -> list[tuple[int, int, str, str]]:
    """(start, end, subtype, fiber) for each run of layers sharing a subtype."""
    spans, start = [], 0
    for i in range(1, len(layers) + 1):
        if i == len(layers) or subtype_of(layers[i]) != subtype_of(layers[start]):
            spans.append((start, i - 1, subtype_of(layers[start]), fiber_of(layers[start])))
            start = i
    return spans


def style_layer_ticks(ax, layers, axis="y"):
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for lab, l in zip(labels, layers):
        lab.set_color(fiber_color(fiber_of(l)))


def fiber_legend_handles(layers):
    seen = [f for f in FIBER_SEQ if any(fiber_of(l) == f for l in layers)]
    return [Patch(color=fiber_color(f), label=f) for f in seen]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(src: Path):
    def _r(name, required=True):
        p = src / name
        if not p.exists():
            if required:
                raise SystemExit(f"ERROR: {p} not found -- point --step7d-dir at step 7d's --output-dir")
            print(f"  [info] {name} not found -- the figures that need it are skipped")
            return None
        return pd.read_csv(p)

    neigh = _r("layer_seed_tests.csv")
    ref = _r("reference_seed_layer_tests.csv", required=False)
    pres = _r("seed_layer_presence.csv")
    summ = _r("seed_layer_summary.csv")
    for df in (neigh, ref):
        if df is not None:
            df["is_global"] = df["is_global"].astype(bool)
    return neigh, ref, pres, summ


# ---------------------------------------------------------------------------
# fig17a -- tier breakdown
# ---------------------------------------------------------------------------
def fig_tier_breakdown(summ: pd.DataFrame, out: Path, size=None) -> Path:
    counts = summ["specificity_tier"].value_counts()
    order = [t for t in TIER_SHORT if t in counts.index]
    order += [t for t in counts.index if t not in order]
    vals = [int(counts[t]) for t in order]
    total = sum(vals)
    is_short = [t.startswith(SHORTLIST_TIERS) for t in order]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=size or (5.6, 4.2), gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.85})

    # --- composition bar: honest about how small the shortlist is ---
    left = 0.0
    for t, v, s in zip(order, vals, is_short):
        light = (not s) and (t.startswith("C") is False)      # pale grey segments need dark text
        pale_blue = (not s) and t.startswith("C")
        col = C.ORANGE if s else (C.BLUE_PALE if pale_blue else C.GREY_LT)
        ax1.barh([0], [v], left=left, height=0.6, color=col, edgecolor="white", linewidth=0.8)
        if v / total > 0.08:
            ax1.text(left + v / 2, 0, f"{TIER_SHORT.get(t, t).split()[0]}  {v:,}",
                     ha="center", va="center", fontsize=ANNOT,
                     color=C.INK if (light or pale_blue) else "white")
        left += v
    ax1.set_xlim(0, total)
    ax1.set_ylim(-0.5, 0.5)
    ax1.axis("off")
    ax1.set_title(f"All {total:,} random-walk neighbours", fontsize=PANEL,
                  color=C.BLUE, fontweight="bold", loc="left", pad=8)
    n_short = sum(v for v, s in zip(vals, is_short) if s)
    ax1.annotate(f"{n_short} candidates (tiers A, A*, B)",
                 xy=(n_short / 2, 0.32), xytext=(total * 0.10, 1.5),
                 fontsize=TICK, color=C.ORANGE, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=C.ORANGE, lw=1.0,
                                 connectionstyle="angle,angleA=0,angleB=90,rad=3"))

    # --- zoom panel: the tiers that matter, at a readable scale ---
    zoom = [(t, v) for t, v, s in zip(order, vals, is_short) if s]
    zlab = [TIER_SHORT.get(t, t) for t, _ in zoom]
    zval = [v for _, v in zoom]
    y = np.arange(len(zoom))[::-1]
    bars = ax2.barh(y, zval, height=0.6, color=C.ORANGE, edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, zval):
        ax2.text(b.get_width() + max(zval) * 0.03, b.get_y() + b.get_height() / 2,
                 f"{v}", va="center", ha="left", fontsize=LAB, color=C.INK)
    ax2.set_yticks(y)
    ax2.set_yticklabels(zlab, fontsize=TICK + 0.5, color=C.INK)
    ax2.tick_params(axis="x", labelsize=TICK)
    ax2.set_xlim(0, max(zval) * 1.25)
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("Neighbour genes", fontsize=LAB)
    ax2.set_title("The shortlist, at scale", fontsize=PANEL, color=C.BLUE,
                  fontweight="bold", loc="left", pad=8)

    caption(fig, "Tiers are defined on the previous slide. C and F are kept in the module but are not "
                 "followed up: C genes are hubs without subtype specificity, F genes are neither.", y=-0.04)
    return save(fig, out / "fig17a_tier_breakdown.png")


# ---------------------------------------------------------------------------
# fig17b -- candidate x layer heatmap
# ---------------------------------------------------------------------------
def fig_candidate_heatmap(pres, summ, out, top_n, genes_file,
                          value_col: str = "fc_pctile", max_cols: int = 30) -> Path:
    p = pres[~pres["is_global"].astype(bool)].copy() if "is_global" in pres else pres.copy()
    ev_map = None          # gene -> literature evidence tier, if the genes file supplies it

    if genes_file:
        gf = Path(genes_file)
        if gf.suffix.lower() in (".csv", ".tsv"):
            tbl = pd.read_csv(gf, sep="\t" if gf.suffix.lower() == ".tsv" else ",")
            if "gene" not in tbl.columns:
                raise SystemExit(f"ERROR: {gf} needs a 'gene' column (found: {list(tbl.columns)})")
            want = [str(g).strip().upper() for g in tbl["gene"].dropna()]
            if "evidence_tier" in tbl.columns:
                ev_map = dict(zip(tbl["gene"].astype(str).str.strip().str.upper(),
                                  pd.to_numeric(tbl["evidence_tier"], errors="coerce")))
        else:                                   # plain whitespace-separated symbols
            want = [g.strip().upper() for g in gf.read_text().split() if g.strip()]
        sel = summ[summ["gene"].str.upper().isin(want)].copy()
        missing = sorted(set(want) - set(sel["gene"].str.upper()))
        if missing:
            print(f"  [warn] {len(missing)} gene(s) in {gf.name} not in seed_layer_summary.csv: "
                  f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
        subtitle = f"{len(sel)} genes from {gf.name}"
    elif top_n:
        sel = summ.head(top_n).copy()
        subtitle = f"top {len(sel)} by cross-layer consistency"
    else:
        sel = summ[summ["specificity_tier"].astype(str).str.startswith(SHORTLIST_TIERS)].copy()
        subtitle = f"All {len(sel)} tier A / A* / B candidates"
    if not len(sel):
        raise SystemExit("ERROR: no genes selected for the heatmap")

    # tier first, then strongest expression evidence
    tier_rank = {t: i for i, t in enumerate(TIER_SHORT)}
    sel["_t"] = sel["specificity_tier"].astype(str).map(lambda t: tier_rank.get(t, 99))
    sort_col = "best_fc_pctile" if "best_fc_pctile" in sel else "consistency_score"
    ev_vals = None
    if ev_map:
        # literature evidence tier first (5 = strongest at top), then network tier, then expression
        sel["_ev"] = sel["gene"].str.upper().map(ev_map).fillna(-1)
        sel = sel.sort_values(["_ev", "_t", sort_col], ascending=[False, True, False])
        ev_vals = sel["_ev"].astype(int).tolist()
    else:
        sel = sel.sort_values(["_t", sort_col], ascending=[True, False])
    genes = sel["gene"].tolist()

    layers = layer_order(p["cell_type"])
    M_all = (p[p["gene"].isin(genes)]
             .pivot_table(index="cell_type", columns="gene", values=value_col, aggfunc="first")
             .reindex(index=layers, columns=genes))

    # --- sizes: big enough to read when the PNG is dropped onto a slide ---
    HM_TICK = 12.0          # gene / layer labels
    HM_SUB = 12.0           # subtype + evidence labels
    CELL_W, CELL_H = 0.34, 0.36          # inches per gene column / per layer row

    n_panels = int(np.ceil(len(genes) / max_cols))
    per = int(np.ceil(len(genes) / n_panels))          # balanced chunks
    chunks = [(i * per, min((i + 1) * per, len(genes))) for i in range(n_panels)]

    panel_h = CELL_H * len(layers) + 1.5               # rows + room for gene labels
    w = CELL_W * per + 6.0                             # + layer labels + subtype labels
    fig, axes = plt.subplots(n_panels, 1, figsize=(w, panel_h * n_panels + 1.4),
                             squeeze=False, gridspec_kw={"hspace": 0.55})
    cmap = CMAP_WARM.copy()
    cmap.set_bad("#F4F5F7")
    fb = set(fiber_bounds(layers))
    spans = subtype_spans(layers)

    for ax, (c0, c1) in zip(axes[:, 0], chunks):
        g = genes[c0:c1]
        M = M_all.iloc[:, c0:c1]
        im = ax.imshow(np.ma.masked_invalid(M.values), cmap=cmap, vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")

        # thick rules between fiber types, thin between subtypes
        for s, e, sub, fib in spans:
            if s == 0:
                continue
            if s in fb:
                ax.axhline(s - 0.5, color="white", lw=3.0)
                ax.axhline(s - 0.5, color=C.GREY, lw=1.0)
            else:
                ax.axhline(s - 0.5, color="white", lw=1.6)
                ax.axhline(s - 0.5, color=C.GREY_LT, lw=0.7)

        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([shorten_layer(l, 24) for l in layers], fontsize=HM_TICK)
        style_layer_ticks(ax, layers, axis="y")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g, rotation=90, fontsize=HM_TICK, color=C.INK)

        # subtype brackets + names on the right
        for s, e, sub, fib in spans:
            col = fiber_color(fib)
            ax.plot([1.006, 1.006], [s - 0.4, e + 0.4], color=col, lw=2.2,
                    transform=ax.get_yaxis_transform(), clip_on=False, solid_capstyle="butt")
            ax.text(1.014, (s + e) / 2, sub, transform=ax.get_yaxis_transform(),
                    fontsize=HM_SUB, color=col, va="center", ha="left", fontweight="bold")

        # literature-evidence blocks (local to this panel)
        if ev_vals:
            ev = ev_vals[c0:c1]
            start = 0
            for i in range(1, len(ev) + 1):
                if i == len(ev) or ev[i] != ev[start]:
                    if i < len(ev):
                        ax.axvline(i - 0.5, color="white", lw=2.6)
                        ax.axvline(i - 0.5, color=C.GREY, lw=0.9)
                    lab = "no evidence" if ev[start] == 0 else f"evidence {ev[start]}"
                    ax.text((start + i - 1) / 2, 1.015, f"{lab} (n={i - start})",
                            transform=ax.get_xaxis_transform(), fontsize=HM_SUB,
                            color=C.MUTED, va="bottom", ha="center")
                    start = i

        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # one shared horizontal colorbar under everything
    cb = fig.colorbar(im, ax=axes[:, 0].tolist(), orientation="horizontal",
                      shrink=0.35, pad=0.06 / n_panels + 0.03, aspect=30)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=HM_TICK, length=2, color=C.GREY_LT, labelcolor=C.MUTED)
    cb.set_label("Fold-change percentile (1 = strongest)", fontsize=HM_TICK + 1, color=C.MUTED)

    caption(fig, "Grey = not among the layer's significantly expressed genes. Columns: "
                 + ("literature evidence tier, then " if ev_vals else "")
                 + "specificity tier, then expression strength. Rows: fiber type (thick rules), "
                 "subtype (thin rules).", y=-0.02)
    return save(fig, out / "fig17b_candidate_heatmap.png", dpi=450)


# ---------------------------------------------------------------------------
# fig17c -- per-layer percentile distributions
# ---------------------------------------------------------------------------
def fig_layer_percentiles(pres: pd.DataFrame, out: Path, have_expr: bool) -> Path:
    p = pres[(~pres["is_global"].astype(bool)) & pres["present"].astype(bool)].copy()
    layers = layer_order(p["cell_type"])
    cols = [("degree_pctile", "Degree percentile in layer")]
    if have_expr and "fc_pctile" in p:
        cols.append(("fc_pctile", "Fold-change percentile among the layer's\nsignificantly expressed genes"))

    n = len(layers)
    # sharey: the cell-type labels are drawn once, on the left-most panel only,
    # so nothing sits in the gap between panels.
    fig, axes = plt.subplots(1, len(cols), figsize=(4.4 * len(cols) + 1.6, max(4.6, 0.24 * n + 1.6)),
                             squeeze=False, sharey=True)
    last = len(cols) - 1
    bounds = fiber_bounds(layers)
    for pi, (ax, (col, label)) in enumerate(zip(axes[0], cols)):
        data = [p.loc[p["cell_type"] == l, col].dropna().values for l in layers]
        bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False,
                        widths=0.64,
                        medianprops=dict(color="white", lw=1.4),
                        whiskerprops=dict(color=C.GREY, lw=0.8),
                        capprops=dict(color=C.GREY, lw=0.0))
        for patch, l in zip(bp["boxes"], layers):
            patch.set_facecolor(fiber_color(fiber_of(l)))
            patch.set_alpha(0.9)
            patch.set_edgecolor("white")
            patch.set_linewidth(0.6)
        for b in bounds:                                   # rule between fiber groups
            ax.axhline(b + 0.5, color=C.GREY_LT, lw=0.8, zorder=0)
        ax.axvline(0.5, color=C.MUTED, lw=0.9, ls=(0, (4, 3)), zorder=0)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="x", labelsize=TICK)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(label, fontsize=LAB)
        if pi > 0:
            ax.tick_params(axis="y", labelleft=False)
        if pi == last:
            for i, d in enumerate(data, start=1):
                ax.text(1.015, i, f"n={len(d)}", transform=ax.get_yaxis_transform(),
                        fontsize=6.5, color=C.MUTED, va="center", ha="left")
    # Set the cell-type labels AFTER all boxplots are drawn. With a shared y axis, every
    # ax.boxplot() call resets the shared tick labels to 1..n, so labels set earlier
    # (or only on the first panel before the second boxplot runs) come out as numbers.
    ax_l = axes[0][0]
    ax_l.set_yticks(range(1, n + 1))
    ax_l.set_yticklabels([shorten_layer(l, 24) for l in layers], fontsize=ROW)
    style_layer_ticks(ax_l, layers, axis="y")
    ax_l.tick_params(axis="y", labelleft=True, length=0)
    ax_l.invert_yaxis()                  # shared axis: invert once only
    axes[0][0].set_title("Neighbours are well connected\u2026", fontsize=PANEL,
                         color=C.BLUE, fontweight="bold", loc="left")
    if len(cols) > 1:
        axes[0][1].set_title("\u2026but not significantly expressed", fontsize=PANEL,
                             color=C.BLUE, fontweight="bold", loc="left")
    caption(fig, "Each box is one cell-type layer. Dashed line = 0.5, what a random gene set would give. "
                 "Boxes shifted right of it mean the neighbours rank above the layer median.", y=-0.03)
    fig.subplots_adjust(wspace=0.10)
    return save(fig, out / "fig17c_layer_percentiles.png")


# ---------------------------------------------------------------------------
# fig18 -- seeds vs neighbours, one continuous figure
# ---------------------------------------------------------------------------
def fig_seeds_vs_neighbors(neigh: pd.DataFrame, ref: pd.DataFrame, out: Path,
                           have_expr: bool) -> Path:
    """Degree AUC | avg_log2FC AUC | degree-matched z, seeds and neighbours side by
    side per layer. All panels share one y axis (cell types labelled once, at the
    far left) and sit flush against each other so it reads as a single plot."""
    a = neigh[~neigh["is_global"]].set_index("layer")
    b = ref[~ref["is_global"]].set_index("layer")
    layers = layer_order(a.index)
    n = len(layers)

    panels = [("degree_auc", "degree_q", "Connectivity\n(degree AUC)", 0.5)]
    if have_expr and "log2fc_auc" in a:
        panels.append(("log2fc_auc", "log2fc_q", "Expression strength\n(avg_log2FC AUC)", 0.5))
    panels.append(("matched_z", "matched_q", "Connectivity vs degree-matched\n(z)", 0.0))

    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels) + 2.2, max(5.2, 0.27 * n + 1.8)),
                             sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.24, right=0.985, top=0.88, bottom=0.17, wspace=0.0)

    y = np.arange(n)
    h = 0.38
    bounds = fiber_bounds(layers)

    for pi, (ax, (col, qcol, title, base)) in enumerate(zip(axes[0], panels)):
        for off, df, col_sig, col_ns in [(-h / 2, b, C.BLUE, C.BLUE_PALE),
                                         (+h / 2, a, C.ORANGE, "#F3C9B4")]:
            vals = df[col].reindex(layers).astype(float).values
            qs = df[qcol].reindex(layers).values.astype(float)
            colors = [col_sig if (not np.isnan(q) and q < 0.05) else col_ns for q in qs]
            v = np.where(np.isnan(vals), base, vals)
            ax.barh(y + off, v - base, left=base, height=h, color=colors,
                    edgecolor="white", linewidth=0.4, zorder=2)
        for bd in bounds:                                    # fiber-group rules, same in every panel
            ax.axhline(bd - 0.5, color=C.GREY_LT, lw=0.8, zorder=1)
        ax.axvline(base, color=C.GREY, lw=0.9, zorder=3)
        ax.set_title(title, fontsize=PANEL, color=C.BLUE, fontweight="bold", loc="left", pad=6)
        ax.tick_params(axis="x", labelsize=TICK)
        ax.tick_params(axis="y", length=0, labelleft=(pi == 0))
        ax.spines["left"].set_visible(pi == 0)
        if pi > 0:                                           # thin divider instead of a gap
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_color(C.GREY_LT)
            ax.spines["left"].set_linewidth(0.8)
        ax.margins(x=0.06)

    # flag layers where the real seeds score higher than the neighbours on degree-matched z
    flagged = []
    sz = b["matched_z"].reindex(layers).astype(float).values
    nz = a["matched_z"].reindex(layers).astype(float).values
    for i, (s, nb) in enumerate(zip(sz, nz)):
        if not np.isnan(s) and not np.isnan(nb) and s > nb:
            flagged.append(i)
    if flagged:
        axes[0][-1].scatter(np.full(len(flagged), 0.965), flagged, marker="*", s=46, color=C.RED,
                            transform=axes[0][-1].get_yaxis_transform(), clip_on=False, zorder=6)

    ax0 = axes[0][0]
    ax0.set_yticks(y)
    ax0.set_yticklabels([shorten_layer(l, 24) for l in layers], fontsize=ROW + 0.5)
    style_layer_ticks(ax0, layers, axis="y")
    ax0.invert_yaxis()                   # shared axis: invert once only

    handles = [Patch(color=C.BLUE, label="seeds, q < 0.05"),
               Patch(color=C.BLUE_PALE, label="seeds, q \u2265 0.05"),
               Patch(color=C.ORANGE, label="neighbours, q < 0.05"),
               Patch(color="#F3C9B4", label="neighbours, q \u2265 0.05")]
    if flagged:
        handles.append(Line2D([], [], marker="*", ls="none", color=C.RED, markersize=8,
                              label="seeds score higher than neighbours"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.6, 0.045),
               ncol=3, frameon=False, fontsize=TICK, columnspacing=1.2)
    extra = ""
    if flagged:
        shown = ", ".join(shorten_layer(layers[i], 18) for i in flagged[:4])
        more = " \u2026" if len(flagged) > 4 else ""
        extra = f" Seeds beat neighbours in {len(flagged)} layer(s): {shown}{more}."
    caption(fig, "Each set is tested against a background that excludes the other set. 0.5 on an AUC and "
                 "0 on a z-score mean no difference from the rest of the layer. Right panel: above zero = "
                 "the layer's own wiring, not literature bias, explains the connectivity." + extra,
            y=-0.01)
    return save(fig, out / "fig18_seeds_vs_neighbors.png")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step7d-dir", required=True,
                    help="step 7d --output-dir from the NEIGHBOR-mode run (e.g. 07d_module_neighbors)")
    ap.add_argument("--out-dir", default="deck_figures")
    ap.add_argument("--figsize", nargs=2, metavar=("W", "H"), default=None,
                    help="canvas size in inches for the tier figure. Set this to the size the PNG "
                         "will be placed at on the slide (default: built-in size).")
    ap.add_argument("--top-n", type=int, default=None,
                    help="heatmap: use the top N neighbours by consistency_score "
                         "instead of the A/A*/B shortlist")
    ap.add_argument("--genes-file", default=None,
                    help="heatmap: whitespace-separated gene symbols to show instead")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["tiers", "heatmap", "percentiles", "svn"],
                    help="draw only these figures (svn = the combined slide-18 figure)")
    ap.add_argument("--max-cols", type=int, default=30,
                    help="heatmap: max genes per panel; more genes wrap into stacked panels")
    args = ap.parse_args()

    use_theme()
    src, out = Path(args.step7d_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    neigh, ref, pres, summ = load(src)
    have_expr = "fc_pctile" in pres.columns and pres["fc_pctile"].notna().any()
    want = set(args.only) if args.only else {"tiers", "heatmap", "percentiles", "svn"}

    if "tiers" in want:
        fig_tier_breakdown(summ, out, size=figsize(args.figsize, None))
    if "heatmap" in want:
        fig_candidate_heatmap(pres, summ, out, args.top_n,
                              Path(args.genes_file) if args.genes_file else None,
                              max_cols=args.max_cols)
    if "percentiles" in want:
        fig_layer_percentiles(pres, out, have_expr)
    if "svn" in want:
        if ref is None:
            print("  [skip] reference_seed_layer_tests.csv missing -- re-run step 7d with --seed-csv")
        else:
            fig_seeds_vs_neighbors(neigh, ref, out, have_expr)


if __name__ == "__main__":
    main()