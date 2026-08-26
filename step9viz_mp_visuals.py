#!/usr/bin/env python3
"""
step9_mp_visuals.py
------------------------------------------------------------------------------
Generates the 4 slide visuals for the multiplex / Marchenko-Pastur / omega-
sweep results:

  1. multiplex_cartoon.png     -- schematic 3-sheet multiplex diagram with an
                                   omega "slider" icon (illustrative, no data
                                   needed).
  2. supra_spectrum_combined.png -- smallest supra-Laplacian eigenvalues for
                                   each omega, overlaid, with the spectral gap
                                   (eigenvalue[1] - eigenvalue[0]) annotated.
                                   Reads omega_<v>/supra_eigvals.npy, which
                                   step9_multiplex_marchenko_pastur_v3.py
                                   already saves for you.
  3. top30_overlap.png         -- hand-drawn 3-way Venn (no matplotlib_venn
                                   dependency, so it runs on LISC without
                                   extra installs) of the top-30 gene sets
                                   across omega=0.1/1.0/5.0, plus a small
                                   Jaccard bar chart underneath. Reads
                                   omega_sweep_comparison.csv.
  4. adcyap1_qvalue.png        -- ADCYAP1's perm_qvalue_bh across the three
                                   omega values, FDR=0.05 line marked. Reads
                                   omega_<v>/multiplex_bridges.csv.

USAGE
------
    python make_mp_visuals.py \
        --mp-dir 09_marchenko_pastur \
        --omegas 0.1 1.0 5.0 \
        --gene ADCYAP1 \
        --top-n 30 \
        --fdr 0.05 \
        --out-dir 09_marchenko_pastur/slide_figs

All four PNGs are saved (300 dpi, transparent-friendly white background) into
--out-dir. Each function can also be called independently if you only need
one figure.
"""

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Ellipse
from matplotlib.lines import Line2D


PALETTE = {
    "0.1": "#4C72B0",
    "1.0": "#DD8452",
    "5.0": "#55A868",
    "ppi": "#888888",
    "liana": "#C44E52",
    "gap": "#937860",
}


# ==============================================================================
# 1. Multiplex cartoon with omega slider
# ==============================================================================
def plot_multiplex_cartoon(out_path, n_layers=3, nodes_per_layer=6, seed=7):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9, 4.6))

    layer_labels = ["A-Fiber subtype", "C-Fiber subtype", "A$\\delta$ subtype"]
    layer_y = [0.78, 0.48, 0.18]
    x_span = (0.08, 0.62)

    node_xy = []  # (layer_idx, x, y)
    for li, y in enumerate(layer_y):
        xs = np.linspace(x_span[0] + 0.03, x_span[1] - 0.03, nodes_per_layer)
        jitter = rng.uniform(-0.01, 0.01, size=nodes_per_layer)
        pts = [(li, x, y + j) for x, j in zip(xs, jitter)]
        node_xy.append(pts)

    # sheet backgrounds
    for li, y in enumerate(layer_y):
        rect = Rectangle((x_span[0] - 0.03, y - 0.075), (x_span[1] - x_span[0]) + 0.06, 0.15,
                          facecolor="white", edgecolor="0.6", lw=1.2, zorder=1,
                          alpha=0.9)
        ax.add_patch(rect)
        ax.text(x_span[0] - 0.05, y, layer_labels[li], ha="right", va="center",
                 fontsize=10, fontweight="bold", color="0.25")

    # intra-layer PPI edges (thin, gray)
    for li, pts in enumerate(node_xy):
        n = len(pts)
        n_edges = int(n * 1.3)
        pairs = set()
        while len(pairs) < n_edges:
            a, b = rng.choice(n, 2, replace=False)
            pairs.add(tuple(sorted((a, b))))
        for a, b in pairs:
            _, xa, ya = pts[a]
            _, xb, yb = pts[b]
            ax.plot([xa, xb], [ya, yb], color=PALETTE["ppi"], lw=0.9,
                     alpha=0.6, zorder=2)

    # nodes
    for li, pts in enumerate(node_xy):
        for _, x, y in pts:
            ax.add_patch(Circle((x, y), 0.014, facecolor="white",
                                  edgecolor="0.2", lw=1.2, zorder=4))

    # inter-layer LIANA edges (thick, colored) between adjacent layers only
    inter_edges = [(0, 1, 3), (1, 2, 3)]  # (layer_a, layer_b, n_edges)
    for la, lb, n_e in inter_edges:
        pts_a, pts_b = node_xy[la], node_xy[lb]
        idx_a = rng.choice(len(pts_a), n_e, replace=False)
        idx_b = rng.choice(len(pts_b), n_e, replace=False)
        for ia, ib in zip(idx_a, idx_b):
            _, xa, ya = pts_a[ia]
            _, xb, yb = pts_b[ib]
            ax.plot([xa, xb], [ya, yb], color=PALETTE["liana"], lw=2.6,
                     alpha=0.85, zorder=3, solid_capstyle="round")

    # legend
    legend_elems = [
        Line2D([0], [0], color=PALETTE["ppi"], lw=1.5, label="Intra-layer PPI edge"),
        Line2D([0], [0], color=PALETTE["liana"], lw=3, label="Inter-layer LIANA\nligand-receptor edge"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", bbox_to_anchor=(0.0, -0.02),
               fontsize=9, frameon=False)

    # omega slider icon on the right
    sx0, sx1, sy = 0.78, 0.95, 0.5
    ax.plot([sx0, sx1], [sy, sy], color="0.3", lw=3, solid_capstyle="round", zorder=2)
    ax.text((sx0 + sx1) / 2, sy + 0.22, r"$\omega$", ha="center", va="center",
             fontsize=20, fontweight="bold")
    ax.text(sx0, sy - 0.10, "intra-layer\ndominates", ha="center", va="top", fontsize=8, color="0.35")
    ax.text(sx1, sy - 0.10, "inter-layer\ndominates", ha="center", va="top", fontsize=8, color="0.35")
    ax.text((sx0 + sx1) / 2, sy + 0.34, "coupling strength", ha="center", va="center",
             fontsize=9, color="0.3", style="italic")

    for pos, lab, col in [(0.15, "0.1", PALETTE["0.1"]),
                          (0.5, "1.0", PALETTE["1.0"]),
                          (0.88, "5.0", PALETTE["5.0"])]:
        px = sx0 + pos * (sx1 - sx0)
        ax.add_patch(Circle((px, sy), 0.018, facecolor=col, edgecolor="0.2",
                              lw=1, zorder=5))
        ax.text(px, sy - 0.045, lab, ha="center", va="top", fontsize=8, color=col,
                 fontweight="bold")

    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(-0.02, 0.98)
    ax.axis("off")
    ax.set_title("Cell-type-resolved multiplex network: layers, edges, and coupling strength ($\\omega$)",
                  fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ==============================================================================
# 2. Supra-Laplacian spectrum, combined across omega
# ==============================================================================
def plot_supra_spectrum(mp_dir, omegas, out_path, k_show=6):
    fig, axes = plt.subplots(1, len(omegas), figsize=(4 * len(omegas), 4), sharey=False)
    if len(omegas) == 1:
        axes = [axes]

    for ax, om in zip(axes, omegas):
        eig_path = Path(mp_dir) / f"omega_{om}" / "supra_eigvals.npy"
        col = PALETTE.get(str(om), "#333333")
        if not eig_path.exists():
            ax.text(0.5, 0.5, f"missing:\n{eig_path}", ha="center", va="center",
                     transform=ax.transAxes, fontsize=9, color="crimson")
            ax.set_title(f"$\\omega$={om}")
            continue
        vals = np.load(eig_path)[:k_show]
        idx = np.arange(len(vals))
        ax.plot(idx, vals, "o-", color=col, lw=2, ms=7, zorder=3)
        ax.axhline(0, color="0.8", lw=0.8, zorder=1)

        gap = vals[1] - vals[0] if len(vals) > 1 else np.nan
        ax.annotate("", xy=(1, vals[1]), xytext=(1, vals[0]),
                     arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.2))
        ax.text(1.12, (vals[0] + vals[1]) / 2, f"gap={gap:.4f}",
                 fontsize=8.5, color="0.25", va="center")

        ax.scatter([0], [vals[0]], s=60, facecolor="white", edgecolor=col, lw=2, zorder=4)
        ax.scatter([1], [vals[1]], s=90, marker="*", facecolor=col, edgecolor="0.2",
                    lw=0.8, zorder=5, label="Fiedler value")

        ax.set_title(f"$\\omega$ = {om}", fontsize=11, color=col, fontweight="bold")
        ax.set_xlabel("eigenvalue index")
        ax.set_xticks(idx)

    axes[0].set_ylabel("supra-Laplacian eigenvalue")
    fig.suptitle("Smallest supra-Laplacian eigenvalues across $\\omega$ (spectral gap = Fiedler value)",
                  fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ==============================================================================
# 3. Top-N gene overlap: hand-drawn 3-way Venn + Jaccard bars
# ==============================================================================
def _load_top_sets(mp_dir, omegas, top_n):
    """Rebuild top-N gene sets per omega directly from each omega's
    multiplex_bridges.csv, ranked by perm_pvalue (falls back to
    frac_minority_layers if perm columns are absent)."""
    sets = {}
    for om in omegas:
        csv_path = Path(mp_dir) / f"omega_{om}" / "multiplex_bridges.csv"
        df = pd.read_csv(csv_path)
        testable = df[~df["insufficient_data"]].copy()
        if "perm_pvalue" in testable.columns:
            testable = testable.sort_values(
                ["perm_pvalue", "perm_qvalue_bh", "frac_minority_layers"],
                ascending=[True, True, False], na_position="last")
        else:
            testable = testable.sort_values(["frac_minority_layers", "fiedler_std"],
                                              ascending=[False, False])
        sets[om] = set(testable.head(top_n)["gene"])
    return sets


def plot_top_overlap(mp_dir, omegas, out_path, top_n=30, comparison_csv=None):
    if comparison_csv is not None and Path(comparison_csv).exists():
        # use precomputed ranks from omega_sweep_comparison.csv to reconstruct sets
        comp = pd.read_csv(comparison_csv)
        sets = {}
        for om in omegas:
            col = f"rank_omega_{om}"
            if col in comp.columns:
                sets[om] = set(comp.loc[comp[col].notna(), "gene"])
        if not sets or any(len(s) == 0 for s in sets.values()):
            sets = _load_top_sets(mp_dir, omegas, top_n)
    else:
        sets = _load_top_sets(mp_dir, omegas, top_n)

    if len(omegas) != 3:
        raise ValueError("hand-drawn Venn only supports exactly 3 omega values")

    om_a, om_b, om_c = omegas
    A, B, C = sets[om_a], sets[om_b], sets[om_c]
    labels = [f"$\\omega$={om_a}", f"$\\omega$={om_b}", f"$\\omega$={om_c}"]
    colors = [PALETTE.get(str(om), "#333333") for om in omegas]

    only_a = A - B - C
    only_b = B - A - C
    only_c = C - A - B
    ab = (A & B) - C
    ac = (A & C) - B
    bc = (B & C) - A
    abc = A & B & C

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5),
                                     gridspec_kw={"width_ratios": [1.3, 1]})

    # --- hand-drawn 3-circle venn ---
    r = 1.2
    centers = [(-0.65, 0.4), (0.65, 0.4), (0.0, -0.65)]
    for (cx, cy), col, lab in zip(centers, colors, labels):
        ax1.add_patch(Circle((cx, cy), r, facecolor=col, alpha=0.35,
                               edgecolor=col, lw=2))
    region_counts = {
        "only_a": (len(only_a), (-1.05, 0.75)),
        "only_b": (len(only_b), (1.05, 0.75)),
        "only_c": (len(only_c), (0.0, -1.35)),
        "ab": (len(ab), (0.0, 0.75)),
        "ac": (len(ac), (-0.55, -0.35)),
        "bc": (len(bc), (0.55, -0.35)),
        "abc": (len(abc), (0.0, 0.05)),
    }
    for key, (n, (tx, ty)) in region_counts.items():
        weight = "bold" if key == "abc" else "normal"
        fontsize = 13 if key == "abc" else 11
        ax1.text(tx, ty, str(n), ha="center", va="center", fontsize=fontsize,
                  fontweight=weight, color="0.1")

    for (cx, cy), col, lab in zip(centers, colors, labels):
        ly = cy + (r + 0.28) * (1 if cy >= -0.3 else -1)
        ax1.text(cx, ly, lab, ha="center", va="center", fontsize=11,
                  fontweight="bold", color=col)

    ax1.set_xlim(-2.2, 2.2)
    ax1.set_ylim(-2.1, 2.1)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title(f"Top-{top_n} candidate genes:\noverlap across $\\omega$", fontsize=11)

    # --- Jaccard bar chart ---
    pair_labels, jaccards = [], []
    for (om1, s1), (om2, s2) in combinations(zip(omegas, [A, B, C]), 2):
        union = s1 | s2
        jac = len(s1 & s2) / len(union) if union else float("nan")
        pair_labels.append(f"$\\omega${om1} vs {om2}")
        jaccards.append(jac)

    bar_colors = ["#8172B2", "#8172B2", "#CCB974"]
    bars = ax2.bar(pair_labels, jaccards, color=bar_colors[:len(jaccards)],
                    edgecolor="0.2", width=0.55)
    for b, j in zip(bars, jaccards):
        ax2.text(b.get_x() + b.get_width() / 2, j + 0.02, f"{j:.3f}",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Jaccard overlap")
    ax2.set_title("Pairwise top-N Jaccard\n(ranking collapses at high $\\omega$)", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ==============================================================================
# 4. Single gene's q-value across omega
# ==============================================================================
def plot_gene_qvalue(mp_dir, omegas, gene, out_path, fdr=0.05):
    xs, qs, ns = [], [], []
    for om in omegas:
        csv_path = Path(mp_dir) / f"omega_{om}" / "multiplex_bridges.csv"
        df = pd.read_csv(csv_path)
        row = df.loc[df["gene"] == gene]
        if row.empty or "perm_qvalue_bh" not in df.columns:
            xs.append(om); qs.append(np.nan); ns.append(np.nan)
            continue
        xs.append(om)
        qs.append(float(row["perm_qvalue_bh"].iloc[0]))
        ns.append(int(row["n_celltypes_present"].iloc[0]) if "n_celltypes_present" in row else np.nan)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = [PALETTE.get(str(om), "#333333") for om in xs]

    ax.plot(xs, qs, "-", color="0.6", lw=1.5, zorder=1)
    for x, q, c in zip(xs, qs, colors):
        sig = (not np.isnan(q)) and q <= fdr
        ax.scatter([x], [q], s=180, color=c, edgecolor="0.2",
                    lw=1.5, zorder=3,
                    marker="o" if sig else "X")
        ax.annotate(f"q={q:.4f}" if not np.isnan(q) else "n/a",
                     (x, q), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, fontweight="bold", color=c)

    ax.axhline(fdr, color="crimson", ls="--", lw=1.5, zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    xlim = (min(xs) / 1.6, max(xs) * 2.6)
    ax.set_xlim(*xlim)
    ax.text(xlim[1], fdr, f"  FDR = {fdr}", color="crimson", va="center", ha="left", fontsize=9)
    ax.set_xlabel("$\\omega$ (inter-layer coupling strength, log scale)")
    ax.set_ylabel("BH-adjusted permutation q-value")
    ax.set_title(f"{gene}: significance across $\\omega$", fontsize=12)

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5",
                markeredgecolor="0.2", markersize=10, label=f"significant (q $\\leq$ {fdr})"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="0.5",
                markeredgecolor="0.2", markersize=10, label=f"not significant (q > {fdr})"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=-0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mp-dir", default="09_marchenko_pastur",
                     help="Path to the step9 output dir (contains omega_<v>/ subfolders).")
    ap.add_argument("--omegas", type=float, nargs="+", default=[0.1, 1.0, 5.0])
    ap.add_argument("--gene", default="ADCYAP1")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--out-dir", default=None,
                     help="Defaults to <mp-dir>/slide_figs")
    ap.add_argument("--skip", nargs="*", default=[],
                     choices=["cartoon", "spectrum", "overlap", "gene"],
                     help="Skip one or more figures, e.g. --skip cartoon")
    args = ap.parse_args()

    mp_dir = Path(args.mp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else mp_dir / "slide_figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # format omegas the same way the pipeline does when naming folders
    # (float(0.1) -> "0.1", float(1.0) -> "1.0", etc.)
    omegas = args.omegas

    if "cartoon" not in args.skip:
        plot_multiplex_cartoon(out_dir / "1_multiplex_cartoon.png")

    if "spectrum" not in args.skip:
        plot_supra_spectrum(mp_dir, omegas, out_dir / "2_supra_spectrum_combined.png")

    if "overlap" not in args.skip:
        comparison_csv = mp_dir / "omega_sweep_comparison.csv"
        plot_top_overlap(mp_dir, omegas, out_dir / "3_top_overlap.png",
                          top_n=args.top_n, comparison_csv=comparison_csv)

    if "gene" not in args.skip:
        plot_gene_qvalue(mp_dir, omegas, args.gene, out_dir / "4_gene_qvalue.png",
                          fdr=args.fdr)

    print(f"\nAll figures saved -> {out_dir}")


if __name__ == "__main__":
    main()