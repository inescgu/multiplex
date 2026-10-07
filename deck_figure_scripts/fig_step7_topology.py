#!/usr/bin/env python3
"""
fig_step7_topology.py -- deck figures from step7_layer_topology.py's tables.

Replaces, for slide purposes:  04_bar_major_category_summary.png
                               (and optionally 01_bar_per_layer_metrics.png)

Reads (from --step7-dir, i.e. step 7's --output-dir):
    01_table_layer_metrics.csv            one row per cell-type layer
    02_table_major_category_summary.csv   one row per major DRG category

Writes (to --out-dir):
    fig12a_layer_scale.png        category-level: how many layers, how big,
                                  how densely wired
    fig12c_size_vs_density.png    per-layer scatter: density vs layer size,
                                  which is what actually explains the
                                  category bars  (--scatter)

What changed from the original plot, and why
--------------------------------------------
1. The original `04_bar_major_category_summary` is a 2x3 grid of six bar
   charts with a different sort order in each panel. On a slide that is six
   rankings the audience has to reconcile, and the one fact they take away --
   "Unclassified is the biggest" -- is an artefact of Unclassified being the
   global-PPI bridging layer rather than a cell type. The replacement keeps
   three metrics, puts them in ONE shared category order, and labels the
   bridging layer as such on the figure instead of leaving it to be explained
   out loud.

2. Mean node count and mean edge count told the same story twice (edges scale
   with nodes). Edge count is dropped; density and clustering carry the
   "how wired" question on their own.

3. Density is plotted on a log x-axis. Linear density axes make everything
   below the biggest bar look like zero, which is the single most misleading
   thing about the original panel.

4. Colours are the fiber-type colours used on every other figure in the deck,
   so A-fiber is the same blue on slide 12 as on slide 18.

Caveat the figure itself now states: these are means over layers of very
different sizes, and density falls with size by construction. Use
--scatter to show that directly rather than asserting it in the notes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from painfig import C, SLOT, FS, caption, fiber_color, figsize, save, use_theme

#: Order every panel the same way, biggest biological group first, with the
#: bridging layer pinned last so it never wins a ranking by accident.
CAT_ORDER = ["C-fiber", "A-fiber", "ATF3", "ATF3 (injured)", "Non-classified",
             "Non-neuronal", "Unclassified", "unknown", "Global PPI (bridge)"]

BRIDGE_CATS = {"Unclassified", "unknown"}


def order_cats(cats) -> list[str]:
    known = [c for c in CAT_ORDER if c in set(cats)]
    rest = sorted(set(cats) - set(known))
    return known + rest


def apply_rename(cat, lay, pairs):
    """Rename categories for the slide without touching the pipeline.
    step 7 labels the global-PPI layer 'unknown' and the injury layer
    'Non-classified' -- correct inside the code, meaningless on a slide."""
    if not pairs:
        return cat, lay
    m = dict(p.split("=", 1) for p in pairs)
    cat = cat.copy()
    cat["major_category"] = cat["major_category"].replace(m)
    if lay is not None:
        lay = lay.copy()
        lay["major_category"] = lay["major_category"].replace(m)
    global BRIDGE_CATS
    BRIDGE_CATS = {m.get(c, c) for c in BRIDGE_CATS} | BRIDGE_CATS
    return cat, lay


def load(step7_dir: Path):
    cat_path = step7_dir / "02_table_major_category_summary.csv"
    lay_path = step7_dir / "01_table_layer_metrics.csv"
    if not cat_path.exists():
        raise SystemExit(f"ERROR: {cat_path} not found -- point --step7-dir at step 7's --output-dir")
    cat = pd.read_csv(cat_path)
    lay = pd.read_csv(lay_path) if lay_path.exists() else None
    return cat, lay


def fig_layer_scale(cat: pd.DataFrame, out: Path, n_layers_note: str | None,
                    size=None) -> Path:
    cats = order_cats(cat["major_category"])
    d = cat.set_index("major_category").reindex(cats)
    colors = [fiber_color(c) for c in cats]
    y = np.arange(len(cats))[::-1]  # first category at the top

    panels = [
        ("n_layers", "Cell-type\nlayers", "{:,.0f}", False),
        ("mean_nodes", "Mean genes\nper layer", "{:,.0f}", True),
        ("mean_density", "Mean\ndensity", "{:.4f}", True),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=size or SLOT["band"], sharey=True)

    for ax, (col, title, fmt, logx) in zip(axes, panels):
        vals = d[col].astype(float).values
        bars = ax.barh(y, vals, height=0.62, color=colors,
                       edgecolor="white", linewidth=0.6)
        # Hatch the bridging layer: same colour family, visibly "not a cell type".
        for b, c in zip(bars, cats):
            if c in BRIDGE_CATS:
                b.set_facecolor(C.GREY_LT)
                b.set_hatch("///")
                b.set_edgecolor(C.GREY)
                b.set_linewidth(0.6)
        if logx:
            ax.set_xscale("log")
            lo = float(np.nanmin(vals[vals > 0]))
            hi = float(np.nanmax(vals))
            ax.set_xlim(lo * 0.5, hi * 4.5)   # headroom for the value labels
        ax.set_title(title, fontsize=FS["panel"], color=C.BLUE,
                     fontweight="bold", loc="left")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        span = ax.get_xlim()
        for b, v in zip(bars, vals):
            x = b.get_width()
            x = x * 1.15 if logx else x + 0.02 * (span[1] - span[0])
            ax.text(x, b.get_y() + b.get_height() / 2, fmt.format(v),
                    va="center", ha="left", fontsize=FS["annot"], color=C.INK)
        if not logx:
            ax.set_xlim(0, float(np.nanmax(vals)) * 1.28)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(cats, fontsize=FS["label"], color=C.INK)

    note = ("Hatched = the global PPI bridging layer, not a DRG cell type: it carries every gene in the "
            "interactome, so its size is set by the resource, not by biology. "
            "Mean genes and mean density are on log axes; density falls with layer size by "
            "construction, so the three panels have to be read together.")
    if n_layers_note:
        note = n_layers_note + " " + note
    caption(fig, note, y=-0.04)
    fig.subplots_adjust(wspace=0.12)
    return save(fig, out / "fig12a_layer_scale.png")


def fig_size_vs_density(lay: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=SLOT["half"])
    for cat in order_cats(lay["major_category"]):
        sub = lay[lay["major_category"] == cat]
        is_bridge = cat in BRIDGE_CATS
        ax.scatter(sub["n_nodes"], sub["density"],
                   s=34, alpha=0.9, label=cat,
                   facecolor=C.GREY_LT if is_bridge else fiber_color(cat),
                   edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Genes in the layer")
    ax.set_ylabel("Layer density")
    ax.set_title("Density is a function of layer size", fontsize=FS["panel"],
                 color=C.BLUE, fontweight="bold", loc="left")
    ax.grid(True, which="major", color=C.GREY_LT, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=1)
    caption(fig, "Each dot is one cell-type layer. Both axes log. A layer that sits above the trend is "
                 "wired more tightly than its size alone predicts.", y=-0.02)
    return save(fig, out / "fig12c_size_vs_density.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step7-dir", required=True,
                    help="step7_layer_topology.py --output-dir (holds 01_/02_table_*.csv)")
    ap.add_argument("--out-dir", default="deck_figures")
    ap.add_argument("--figsize", nargs=2, metavar=("W", "H"), default=None,
                    help="canvas size in inches. Set this to the size the PNG will be "
                         "placed at on the slide, so the figure's text lands at the point "
                         "size it was asked for (default: the painfig SLOT for this figure).")
    ap.add_argument("--scatter", action="store_true",
                    help="also write fig12c_size_vs_density.png (needs 01_table_layer_metrics.csv)")
    ap.add_argument("--note", default=None, help="extra sentence prepended to the figure caption")
    ap.add_argument("--rename", nargs="*", default=None, metavar="OLD=NEW",
                    help="relabel categories for the slide, e.g. "
                         "--rename unknown='Global PPI (bridge)' Non-classified='ATF3 (injured)'")
    args = ap.parse_args()

    use_theme()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cat, lay = load(Path(args.step7_dir))
    cat, lay = apply_rename(cat, lay, args.rename)

    fig_layer_scale(cat, out, args.note, size=figsize(args.figsize, SLOT["band"]))
    if args.scatter:
        if lay is None:
            print("  [skip] 01_table_layer_metrics.csv not found -- no scatter")
        else:
            fig_size_vs_density(lay, out)


if __name__ == "__main__":
    main()