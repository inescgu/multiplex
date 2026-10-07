#!/usr/bin/env python3
"""
fig_step7c_density.py -- deck figure from step7c_interlayer_connectivity.py.

Replaces, for slide purposes:  heatmap_fibertype_density.png
                               (and, with --level subcategory, its subcategory twin)

Reads (from --step7c-dir, i.e. step 7c's --output-dir):
    edge_density_by_fibertype.csv       group_1, group_2, n_genes_1, n_genes_2,
                                        n_possible_edges, n_actual_edges, density_percent
    edge_counts_by_fibertype.csv        raw counts, used for the "n edges" annotation
    (--level subcategory uses the *_by_subcategory.csv pair instead)

Writes (to --out-dir):
    fig12b_fibertype_density.png

What changed from the original plot, and why
--------------------------------------------
1. The original draws the full square matrix. It is symmetric, so half the
   cells are the same number printed twice -- on a 4x4 that is eight redundant
   annotations competing with the eight real ones. This draws the lower
   triangle only.

2. STRUCTURAL ZEROS ARE NOW MASKED, NOT COLOURED. A group with a single member
   cell type (ATF3) cannot have a within-group *between-cell-type* bipartite
   edge, so its diagonal is necessarily 0.0000. In the original that 0 is
   drawn as the palest colour, which invites the reading "ATF3 is poorly
   connected to itself" when the correct reading is "that cell does not
   exist". Masked cells are drawn as a hatched blank and labelled n/a.
   Any zero that is NOT structural (a real group pair with no edges) is still
   coloured, so a genuine absence of connection is still visible.

3. Values are shown in units of 1e-3 percent-of-possible-pairs. The raw numbers
   are 0.0039 ... 0.0178, which at slide size reads as a column of leading
   zeros. The axis label carries the unit.

4. The colour scale is the deck's blue ramp and is clipped to the range of the
   *drawn* cells, so the contrast is spent on the comparisons the slide is
   making rather than on the masked diagonal.

5. The bridging layer ("Unclassified") is kept but tagged in the tick label,
   because the density normalisation only partly fixes it: its gene pool is
   the whole PPI, so its "possible pairs" denominator is set by the resource.
   Use --drop-unclassified to produce the biology-only version for a slide
   where that caveat would derail the point.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from painfig import C, CMAP_BLUE, FS, SLOT, caption, figsize, save, use_theme

GROUP_ORDER = {
    "fibertype": ["A-fiber", "C-fiber", "ATF3", "Non-neuronal", "Unclassified"],
    "subcategory": [
        "Proprioceptors", "A-beta LTMR", "A-LTMR", "A-peptidergic",
        "C-LTMR", "C-peptidergic", "MRGPRX nociceptors", "C-thermosensitive",
        "C-SST nociceptor", "ATF3 (injured)", "Unclassified",
    ],
}
BRIDGE = {"Unclassified", "unknown"}


def order_groups(groups, level: str) -> list[str]:
    want = GROUP_ORDER.get(level, [])
    present = set(groups)
    out = [g for g in want if g in present]
    return out + sorted(present - set(out))


def to_matrix(df, order, value_col, a="group_1", b="group_2"):
    M = pd.DataFrame(np.nan, index=order, columns=order, dtype=float)
    for r in df.itertuples(index=False):
        g1, g2 = getattr(r, a), getattr(r, b)
        if g1 in M.index and g2 in M.columns:
            v = float(getattr(r, value_col))
            M.loc[g1, g2] = v
            M.loc[g2, g1] = v
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step7c-dir", required=True,
                    help="step7c_interlayer_connectivity.py --output-dir")
    ap.add_argument("--out-dir", default="deck_figures")
    ap.add_argument("--figsize", nargs=2, metavar=("W", "H"), default=None,
                    help="canvas size in inches. Set this to the size the PNG will be "
                         "placed at on the slide, so the figure's text lands at the point "
                         "size it was asked for (default: the painfig SLOT for this figure).")
    ap.add_argument("--level", choices=["fibertype", "subcategory"], default="fibertype")
    ap.add_argument("--drop-unclassified", action="store_true",
                    help="omit the global-PPI bridging group entirely")
    ap.add_argument("--singleton-groups", nargs="*", default=None,
                    help="groups with exactly one member cell type, whose diagonal is a "
                         "STRUCTURAL zero and is masked. Default for fibertype level: ATF3.")
    args = ap.parse_args()

    use_theme()
    src = Path(args.step7c_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dens_path = src / f"edge_density_by_{args.level}.csv"
    cnt_path = src / f"edge_counts_by_{args.level}.csv"
    if not dens_path.exists():
        raise SystemExit(f"ERROR: {dens_path} not found -- point --step7c-dir at step 7c's --output-dir")
    dens = pd.read_csv(dens_path)
    cnt = pd.read_csv(cnt_path) if cnt_path.exists() else None

    groups = order_groups(set(dens["group_1"]) | set(dens["group_2"]), args.level)
    if args.drop_unclassified:
        groups = [g for g in groups if g not in BRIDGE]

    singles = set(args.singleton_groups) if args.singleton_groups is not None else (
        {"ATF3", "ATF3 (injured)", "A-LTMR", "C-LTMR", "C-SST nociceptor"}
    )

    D = to_matrix(dens, groups, "density_percent") * 1000.0  # -> 1e-3 %
    N = to_matrix(dens, groups, "n_actual_edges") if "n_actual_edges" in dens else (
        to_matrix(cnt, groups, "total") if cnt is not None else None)

    n = len(groups)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if j > i:                      # upper triangle: redundant
                mask[i, j] = True
    structural = np.zeros((n, n), dtype=bool)
    for i, g in enumerate(groups):
        if g in singles:
            structural[i, i] = True        # single-member group: no within-group pair exists
            mask[i, i] = True

    vals = np.ma.masked_array(D.values, mask=mask)
    vmin = float(np.nanmin(vals.compressed())) if vals.count() else 0.0
    vmax = float(np.nanmax(vals.compressed())) if vals.count() else 1.0

    fig, ax = plt.subplots(figsize=figsize(args.figsize, (4.5, 3.4) if n <= 6 else (6.4, 5.2)))
    im = ax.imshow(vals, cmap=CMAP_BLUE, vmin=vmin, vmax=vmax, aspect="equal")

    # grid lines between cells, drawn as thin white gaps
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", lw=1.6)
        ax.axvline(k - 0.5, color="white", lw=1.6)

    for i in range(n):
        for j in range(n):
            if mask[i, j] and not structural[i, j]:
                continue
            if structural[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white",
                                       edgecolor=C.GREY_LT, hatch="///", lw=0.6, zorder=2))
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=FS["annot"], color=C.MUTED, zorder=3)
                continue
            v = D.values[i, j]
            if np.isnan(v):
                continue
            frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            txt = f"{v:.1f}"
            if N is not None and not np.isnan(N.values[i, j]):
                txt += f"\n{int(N.values[i, j]):,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=FS["annot"],
                    color="white" if frac > 0.55 else C.INK, linespacing=1.25, zorder=3)

    labels = [g + ("\n(PPI bridge)" if g in BRIDGE else "") for g in groups]
    rot = 0 if n <= 5 else 40
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=rot, ha="center" if rot == 0 else "right",
                       fontsize=FS["tick"], color=C.INK)
    ax.set_yticklabels(labels, fontsize=FS["tick"], color=C.INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Colour bar goes INSIDE the empty upper-right triangle rather than beside
    # the axes -- a lower-triangle matrix leaves that space blank anyway, and a
    # bar on the right pushes the whole matrix left and shrinks it on the slide.
    cax = ax.inset_axes([0.42, 0.80, 0.55, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS["tick"], color=C.GREY_LT, labelcolor=C.MUTED,
                      length=2, pad=1)
    cb.set_label("Density: realised pairs per 100,000",
                 fontsize=FS["tick"], color=C.MUTED, labelpad=3)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("bottom")

    fig.text(0.0, 1.0, "Cross-fiber wiring is denser than within-fiber wiring"
             if args.level == "fibertype" else "Inter-cell-type edge density",
             fontsize=FS["panel"], color=C.BLUE, fontweight="bold", ha="left", va="bottom")

    caption(fig, "Top number = density (edges \u00f7 all possible gene\u2013gene pairs between the two groups); "
                 "bottom = raw edges. n/a = the group has one cell type, so a within-group "
                 "between-cell-type edge cannot exist. Upper triangle omitted (symmetric).",
            y=-0.14)
    save(fig, out / f"fig12b_{args.level}_density.png")


if __name__ == "__main__":
    main()