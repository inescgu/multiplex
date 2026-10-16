#!/usr/bin/env python3
"""
fig_step7b_lcc.py -- deck figure from step7b (degree-normalised LCC test per cell type).

Replaces, for slide purposes:  lcc_significance_by_celltype.png

Reads (from --step7b-dir, i.e. step 7b's --output-dir, e.g. layer_lcc_analysis):
    lcc_summary.csv    one row per cell type with seeds, columns used:
                       cell_type, major_category, n_seeds, lcc_size, lcc_p_value,
                       lcc_z_score, lcc_random_mean, lcc_random_std, significant_lcc

Writes (to --out-dir):
    fig_lcc_by_celltype.png     (rename to the slide number it lands on, e.g. fig13a_...)

Design notes
------------
1. Same layer order as every other deck figure: fiber type (A, C, ATF3,
   non-neuronal, unknown), alphabetical within.
2. Deck colour rule: ORANGE = significant, pale blue = not. Fiber type is carried
   by tick-label colour and by rules between fiber blocks.
3. Horizontal bars so 40+ cell-type names are readable at slide size.
4. Two panels: raw LCC size (tick = random-set mean +/- 2 SD) and the z-score.
   LCC size alone is not comparable between layers; z is.
5. z panel: grey band = +/-2 SD of the null; black error bars = Monte Carlo
   standard error of z (delta method, from the number of random sets N).
   Use --no-errbars to hide the bars, --mc-n to override N for the bars.

Significance is the permutation p from step 7b (p < 0.05, one permutation test per
cell type, NOT multiple-testing corrected). Pass --bh to colour by BH q instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from painfig import C, FS, caption, fiber_color, figsize, save, shorten_layer, sig_colors, use_theme

FIBER_SEQ = ["A-fiber", "C-fiber", "ATF3", "Non-neuronal", "unknown"]
ALIAS = {"Non-classified": "ATF3", "ATF3 (injured)": "ATF3", "Unclassified": "unknown"}


def norm_fiber(f) -> str:
    f = str(f)
    return ALIAS.get(f, f)


def fiber_rank(f: str) -> int:
    return FIBER_SEQ.index(f) if f in FIBER_SEQ else len(FIBER_SEQ)


def label_color(f: str) -> str:
    """Tick-label colour. 'unknown' is pale in the palette, too pale for text."""
    return C.MUTED if f in ("unknown", "Unclassified") else fiber_color(f)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def z_mc_se(z: np.ndarray, n_sims: int) -> np.ndarray:
    """Monte Carlo standard error of z = (L - mean)/sd when mean and sd are
    estimated from n_sims random sets (observed L treated as fixed):
        Var(z) ~= 1/N + z^2 / (2 (N - 1))
    """
    N = max(int(n_sims), 2)
    return np.sqrt(1.0 / N + np.square(z) / (2.0 * (N - 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step7b-dir", required=True,
                    help="step 7b --output-dir (holds lcc_summary.csv)")
    ap.add_argument("--out-dir", default="deck_figures")
    ap.add_argument("--figsize", nargs=2, metavar=("W", "H"), default=None,
                    help="canvas size in inches; set to the size the PNG is placed at on the slide "
                         "(default: 6.2 wide, height scales with the number of cell types)")
    ap.add_argument("--n-sims", type=int, default=1000,
                    help="--num-simulations that step 7b was run with (default 1000); "
                         "printed in the caption. Not stored in lcc_summary.csv, so check your run.")
    ap.add_argument("--mc-n", type=int, default=None,
                    help="number of random sets used for the z error bars (default: --n-sims). "
                         "Cell types that stopped at the 200-set interim stage have larger true error.")
    ap.add_argument("--no-errbars", action="store_true",
                    help="hide the Monte Carlo error bars on the z panel")
    ap.add_argument("--bh", action="store_true",
                    help="colour by Benjamini-Hochberg q across cell types instead of raw p")
    ap.add_argument("--drop-nonneuronal", action="store_true",
                    help="omit non-neuronal / unknown cell types")
    ap.add_argument("--name", default="fig_lcc_by_celltype.png", help="output file name")
    args = ap.parse_args()

    use_theme()
    src, out = Path(args.step7b_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = src / "lcc_summary.csv"
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found -- point --step7b-dir at step 7b's --output-dir")
    d = pd.read_csv(path)
    d["fiber"] = d["major_category"].map(norm_fiber)
    if args.drop_nonneuronal:
        d = d[d["fiber"].isin(["A-fiber", "C-fiber", "ATF3"])]
    if not len(d):
        raise SystemExit("ERROR: no rows left to plot")

    d["_r"] = d["fiber"].map(fiber_rank)
    d = d.sort_values(["_r", "cell_type"]).reset_index(drop=True)

    n = len(d)
    y = np.arange(n)
    pcol = d["lcc_p_value"].astype(float).values
    if args.bh:
        stat = bh_qvalues(pcol)
        sig_word = "BH q < 0.05"
    else:
        stat = pcol
        sig_word = "permutation p < 0.05, uncorrected"
    colors = sig_colors(stat.tolist())

    h = max(2.8, 0.17 * n + 1.5)
    size = figsize(args.figsize, (6.2, h))
    h = size[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size, sharey=True)
    # more bottom room for two-line x labels; wider gap between panels
    fig.subplots_adjust(left=0.27, right=0.93, top=1 - 0.30 / h, bottom=1.05 / h, wspace=0.16)

    # ---- panel 1: LCC size, with the random-set mean as a tick ----
    lcc = d["lcc_size"].astype(float).values
    rmean = d["lcc_random_mean"].astype(float).values
    rstd = d["lcc_random_std"].astype(float).values
    ax1.barh(y, lcc, height=0.66, color=colors, edgecolor="white", linewidth=0.5, zorder=2)
    ax1.hlines(y, np.maximum(rmean - 2 * rstd, 0), rmean + 2 * rstd,
               color=C.INK, lw=0.9, zorder=4)
    ax1.scatter(rmean, y, marker="|", s=90, linewidths=1.5, color=C.INK, zorder=5)
    ax1.set_xlim(0, float(np.nanmax(np.maximum(lcc, rmean + 2 * rstd))) * 1.08 + 1)
    ax1.set_xlabel("Size of largest\nconnected component", linespacing=1.2)
    ax1.set_title("Seed genes in the layer")

    # ---- panel 2: z-score, null band, Monte Carlo error bars ----
    z = d["lcc_z_score"].astype(float).values
    mc_n = args.mc_n if args.mc_n is not None else args.n_sims
    se = z_mc_se(np.nan_to_num(z), mc_n)

    ax2.axvspan(-2, 2, color=C.GREY_LT, alpha=0.45, lw=0, zorder=0)   # null +/- 2 SD
    ax2.barh(y, z, height=0.66, color=colors, edgecolor="white", linewidth=0.5, zorder=2)
    if not args.no_errbars:
        ax2.errorbar(z, y, xerr=se, fmt="none", ecolor=C.INK, elinewidth=0.9,
                     capsize=1.8, capthick=0.9, zorder=4)
    ax2.axvline(0, color=C.GREY, lw=0.9, zorder=3)
    lo = float(np.nanmin(np.minimum(z, z - se)))
    hi = float(np.nanmax(np.maximum(z, z + se)))
    ax2.set_xlim(min(-2.5, lo * 1.15), max(2.5, hi * 1.10))
    ax2.set_xlabel("z vs degree-matched\nrandom genes", linespacing=1.2)
    ax2.set_title("Distance from random")
    for i, ns in enumerate(d["n_seeds"].values):
        ax2.text(1.01, i, f"n={int(ns)}", transform=ax2.get_yaxis_transform(),
                 fontsize=6, color=C.MUTED, va="center", ha="left")

    # ---- shared y: labels coloured by fiber, rules between fiber blocks ----
    ax1.set_yticks(y)
    ax1.set_yticklabels([shorten_layer(c, 24) for c in d["cell_type"]], fontsize=6.5)
    for lab, f in zip(ax1.get_yticklabels(), d["fiber"]):
        lab.set_color(label_color(f))
    ax1.set_ylim(n - 0.5, -0.5)          # first row at the top
    for ax in (ax1, ax2):
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    prev = None
    for i, f in enumerate(d["fiber"]):
        if prev is not None and f != prev:
            for ax in (ax1, ax2):
                ax.axhline(i - 0.5, color=C.GREY_LT, lw=0.8, zorder=1)
        prev = f

    # ---- legend ----
    present = [f for f in FIBER_SEQ if f in set(d["fiber"])]
    present += sorted(set(d["fiber"]) - set(present))
    handles = [Patch(color=C.ORANGE, label=sig_word.split(",")[0]),
               Patch(color=C.BLUE_PALE, label="not significant"),
               Line2D([], [], marker="|", ls="none", color=C.INK, markersize=9,
                      markeredgewidth=1.5, label="random sets, mean \u00b1 2 SD")]
    handles += [Line2D([], [], marker="s", ls="none", markersize=0, label=f)
                for f in present]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0),
                     ncol=len(handles), columnspacing=0.9, handletextpad=0.4, fontsize=FS["tick"])
    for t in leg.get_texts()[3:]:
        t.set_color(label_color(t.get_text()))

    err_txt = ("" if args.no_errbars else
               f" Error bars on z = Monte Carlo standard error from {mc_n:,} random sets; "
               "grey band = random-set \u00b1 2 SD.")
    caption(fig, f"Each cell type: observed LCC of the seed genes vs {args.n_sims:,} degree-matched "
                 f"random gene sets (adaptive: 200 first, extended to {args.n_sims:,} only if interim "
                 f"p \u2264 0.15, so p resolves down to 1/{args.n_sims:,}). Orange = {sig_word}; "
                 f"{n} cell types tested. Tick and line = random-set mean \u00b1 2 SD."
                 f"{err_txt} "
                 "n = seed genes present in the layer. LCC size depends on how many seeds a layer "
                 "has, so compare layers on z. Label colour = fiber type.",
            y=-0.30 / h)
    save(fig, out / args.name)


if __name__ == "__main__":
    main()