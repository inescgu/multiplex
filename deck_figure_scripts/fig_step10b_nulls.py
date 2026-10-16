#!/usr/bin/env python3
"""
fig_step10_nulls.py -- deck figures for slide 17, from step10_validate_module.py.

Replaces fig_step10b_nulls.py (which read step 10b's null1_/null2_/null3_ CSVs).
step 10 writes one summary per null, so this reads those instead:

    <validation-dir>/null_seeds_summary.csv       Null 1  random seeds, real network
    <validation-dir>/null_rewired_summary.csv     Null 2  true seeds, rewired network
    <validation-dir>/null_neighbors_summary.csv   Null 3  true seeds, random neighbours, no walk
      columns: metric, true_value, null_mean, null_sd, z_score, p_value, n_null
    <validation-dir>/null_seeds_size_adjusted.csv, null_rewired_size_adjusted.csv
      (from size_adjust_nulls.py; used by --z-basis window, the default)
    (optional, for the appendix figure)
    <validation-dir>/null_{seeds,rewired,neighbors}_permutations.csv

Writes to --out-dir:
    fig15a_null_ladder.png        slide figure. One panel per metric, one row per null.
    fig15a_values.csv             exactly what is drawn (oriented z, p, true/null values)
    fig15b_null_distributions.png appendix histograms incl. module size (needs *_permutations.csv)

Sizing (painfig rules)
----------------------
fig15a is drawn on painfig's SLOT["band"] canvas (9.2 x 2.0 in) with painfig's FS
type sizes, so place the PNG at 1:1 and do NOT resize it in PowerPoint.
savefig uses bbox="tight", so the PNG is a hair under 9.2 in wide; read its pixel
size at 300 dpi to get the physical size to place it at. --figsize W H overrides.

What changed from the old script, and why
-----------------------------------------
1. NEW INPUTS. The old script expected step 10b's *_combined.csv / *_split.csv files.

2. NEIGHBOURS-ONLY IS THE DEFAULT ROW. Seeds are shared by the module and by
   Nulls 2 and 3, and (in the seeds null) the seed score is circular: seeds are
   drawn from pain gene databases and scored against pain drug targets and a
   pain-gene CSV. Whole-module z therefore mixes a built-in signal with the
   thing the project added. Use --scopes combined neighbors to draw both.

3. z IS ORIENTED SO RIGHT = MODULE BETTER THAN THE NULL. Conductance is
   lower-is-better, so its z is sign-flipped. Without this the topology panels
   would show the best result as a negative number.

4. THREE STATES, NOT TWO. orange = better than null (p < 0.05, the deck's
   significance colour, via painfig.sig_colors); pale blue = not distinguishable;
   mid blue = significantly WORSE than the null (oriented z <= -1.96). In the
   real run the rewired null makes the module look worse on drug targets, and a
   two-colour scheme would file that under "not significant". Mid blue rather
   than deep blue because painfig reserves deep blue for seeds.
   p_value in the step 10 CSVs is one-sided in the "better" direction, so a
   worse-than-null result has p = 1; the z-score is what identifies it.
   (sig_colors tests q < 0.05; here it is fed the raw permutation p, because
   these tests have no BH family.)

5. TOPOLOGY PANELS HAVE THEIR OWN AXIS (z reaches 40+ there, 5-15 elsewhere) and
   Null 3 is drawn as n/a: there is no network to score. Edge density is left
   out by default because the true module and the Null 1/2 modules differ in
   size, and density falls with size (--topology edge_density to add it).

7. SIZE-MATCHED BY DEFAULT (--z-basis window). The Null 1/2 modules differ in size from the
   true module, and size explains most of the variation in the fractions (R2 0.5-0.9) and
   nearly all of the topology (R2 > 0.96). A raw z against nulls of a different size mixes the
   effect with a size effect, in opposite directions for the two nulls (Null 1 modules are
   larger, Null 2 smaller). So Nulls 1-2 are drawn from size_adjust_nulls.py's window
   comparison: only null modules within +-25% of the true size. Null 3 is size-matched by
   construction (every null module is exactly the true size). --z-basis raw draws the
   unadjusted z. The regression z (z_size_adj) is NOT used: where R2 ~ 0.99 the residual SD
   is tiny and z explodes (conductance z = -31), which is model misfit, not signal.
   With only 38-70 window nulls the p floor is 0.014-0.026, not 0.01.

6. p-VALUES ARE NOT PRINTED NEXT TO POINTS. They are floored at 1/(n+1)
   (0.001 for 1,000 runs, 0.0099 for 100), so "p = 0.000999" reads as precision
   it does not have. The floors are stated in the caption instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from painfig import (C, FS, SLOT, caption, figsize, panel_title, reference_line,
                     save, sig_colors, use_theme)

#: (file tag, short label, descriptor)
NULLS = [
    ("seeds", "Null 1", "random seeds"),
    ("rewired", "Null 2", "rewired network"),
    ("neighbors", "Null 3", "random neighbours"),
]
FUNCTIONAL = [
    ("drug_target_any", "Drug targets,\nany indication"),
    ("drug_target_disease", "Pain-indication\ndrug targets"),
    ("pain_phenotype", "Pain-phenotype\ngenes"),
]
TOPOLOGY = {
    "mean_internal_degree": "Mean internal\ndegree",
    "conductance": "Conductance\n(lower = tighter)",
    "edge_density": "Edge\ndensity",
}
LOWER_IS_BETTER = {"conductance"}
SCOPE_LABEL = {"neighbors": "neighbours only", "combined": "whole module", "seeds": "seeds only"}
MARKER = {"neighbors": "D", "combined": "o", "seeds": "s"}
MSIZE = {"neighbors": 34, "combined": 44, "seeds": 34}
MINUS = "\u2212"
WORSE = C.BLUE_MID


def load_summaries(val_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for tag, _, _ in NULLS:
        p = val_dir / f"null_{tag}_summary.csv"
        if p.exists():
            out[tag] = pd.read_csv(p).set_index("metric")
        else:
            print(f"  [info] {p.name} not found -- that null is drawn as 'not run'")
    if not out:
        raise SystemExit(f"ERROR: no null_*_summary.csv under {val_dir}")
    return out


def lookup(summ: dict, tag: str, base: str, scope: str, topology: bool, win: dict | None = None):
    """-> dict(z, p, true, null, n) with z oriented (right = better), or None.
    If `win` has the size-adjusted table for this null, use its window z/p (already oriented)."""
    key = base if (topology or scope == "combined") else f"{base}__{scope}"
    if win is not None and tag in win:
        w = win[tag]
        if key not in w.index or pd.isna(w.loc[key, "z_window"]):
            return None
        r = w.loc[key]
        return dict(z=float(r["z_window"]), p=float(r["p_window"]), true=float(r["true_value"]),
                    null=float(r["null_mean"]), n=int(r["n_window"]), key=key)
    df = summ.get(tag)
    if df is None:
        return None
    if key not in df.index:
        return None
    r = df.loc[key]
    if pd.isna(r["z_score"]):
        return None
    sign = -1.0 if base in LOWER_IS_BETTER else 1.0
    return dict(z=float(r["z_score"]) * sign, p=float(r["p_value"]), true=float(r["true_value"]),
                null=float(r["null_mean"]), n=int(r["n_null"]), key=key)


def load_windows(val_dir: Path, basis: str) -> dict:
    """size_adjust_nulls.py output for the nulls whose modules differ in size from the true one."""
    out = {}
    if basis != "window":
        return out
    for tag in ("seeds", "rewired"):
        p = val_dir / f"null_{tag}_size_adjusted.csv"
        if p.exists():
            out[tag] = pd.read_csv(p).set_index("metric")
        else:
            print(f"  [WARNING] {p.name} missing: Null '{tag}' is drawn on the RAW z (not size-matched). "
                  f"Run: python size_adjust_nulls.py --validation-dir {val_dir} --null {tag}")
    return out


def state(z: float, p: float) -> str:
    if sig_colors([p])[0] == C.ORANGE:
        return "better"
    return "worse" if z <= -1.96 else "ns"


def fmt_z(z: float) -> str:
    if round(abs(z), 1) == 0:
        return "0.0"
    return f"{MINUS if z < 0 else '+'}{abs(z):.1f}"


def fig_ladder(summ: dict, out: Path, scopes, topo_metrics, size, win: dict | None = None) -> Path:
    col = {"better": C.ORANGE, "ns": C.BLUE_PALE, "worse": WORSE}
    rows = [(tag, short, desc, sc) for tag, short, desc in NULLS for sc in scopes]
    ypos = {i: len(rows) - 1 - i for i in range(len(rows))}

    panels = [("f", m, lab) for m, lab in FUNCTIONAL] + [("t", m, TOPOLOGY[m]) for m in topo_metrics]
    nf = len(FUNCTIONAL)
    ratios = [1.0] * nf + [0.16] + [0.88] * len(topo_metrics)
    W, H = size
    left = 0.205 if len(scopes) == 1 else 0.30
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, left=left, right=0.995,
                          top=1 - 0.45 / H, bottom=0.62 / H, wspace=0.14)
    axes, records = [], []

    def zs_of(kind):
        zs = []
        for tag, _, _, sc in rows:
            for k, m, _ in panels:
                if k == kind:
                    r = lookup(summ, tag, m, sc, kind == "t", win)
                    if r:
                        zs.append(r["z"])
        return zs or [0.0]
    lims = {}
    for kind in ("f", "t"):
        zs = zs_of(kind)
        hi = max(max(zs), 3.0) / 0.55            # right ~45% stays free for the value label
        lo = min(-0.10 * hi, min(zs) * 1.2 - 2.0, -3.0)
        lims[kind] = (lo, hi)

    ci = 0
    for kind, metric, label in panels:
        if kind == "t" and ci == nf:
            ci += 1                              # spacer column
        ax = fig.add_subplot(gs[0, ci])
        ci += 1
        axes.append(ax)
        lo, hi = lims[kind]
        ax.axvspan(-1.96, 1.96, color=C.CARD, zorder=0)   # not past |z| 1.96
        reference_line(ax, 0)
        for i, (tag, short, desc, sc) in enumerate(rows):
            y = ypos[i]
            if kind == "t" and sc != scopes[0]:
                continue                         # topology is module-level: draw once per null
            r = lookup(summ, tag, metric, sc, kind == "t", win)
            if r is None:
                if kind == "t" and tag == "neighbors":
                    txt = "n/a (no network)"
                elif sc == "seeds" and tag != "seeds":
                    txt = "n/a: seeds are fixed"
                elif tag not in summ:
                    txt = "not run"
                else:
                    txt = "n/a"
                ax.text(lo + 0.30 * (hi - lo), y, txt, fontsize=FS["annot"], color=C.MUTED,
                        va="center", ha="left", style="italic", zorder=5)
                continue
            s = state(r["z"], r["p"])
            ax.plot([0, r["z"]], [y, y], color=col[s], lw=2.0, solid_capstyle="round", zorder=3)
            ax.scatter([r["z"]], [y], marker=MARKER[sc], s=MSIZE[sc], color=col[s],
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(hi, y, fmt_z(r["z"]), fontsize=FS["annot"], ha="right", va="center",
                    color=C.INK if s != "ns" else C.MUTED,
                    fontweight="bold" if s != "ns" else "normal")
            records.append(dict(null=short, scope=sc, metric=metric, panel=label.replace("\n", " "),
                                state=s, z_oriented=r["z"], p_one_sided=r["p"],
                                true_value=r["true"], null_mean=r["null"], n_null=r["n"],
                                source_key=r["key"]))
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        panel_title(ax, label)

    # row labels (bold name + muted descriptor): figure x, first panel's data y
    tr = blended_transform_factory(fig.transFigure, axes[0].transData)
    for i, (tag, short, desc, sc) in enumerate(rows):
        y = ypos[i]
        d = desc if len(scopes) == 1 else f"{desc} \u00b7 {SCOPE_LABEL[sc]}"
        if sc == scopes[0]:
            axes[0].text(0.004, y, short, transform=tr, fontsize=FS["label"], color=C.INK,
                         fontweight="bold", va="center", ha="left")
        axes[0].text(0.052, y, d, transform=tr, fontsize=FS["tick"], color=C.MUTED,
                     va="center", ha="left")

    handles = [Line2D([], [], marker="o", ls="", color=C.ORANGE, label="better than null (p < 0.05)"),
               Line2D([], [], marker="o", ls="", color=C.BLUE_PALE, label="not distinguishable"),
               Line2D([], [], marker="o", ls="", color=WORSE, label="worse than null (z \u2264 \u22121.96)")]
    fig.legend(handles=handles, loc="lower left", ncol=3, fontsize=FS["tick"],
               handletextpad=0.2, columnspacing=1.0, bbox_to_anchor=(left - 0.01, 0.17 / H),
               labelcolor=C.MUTED)
    # aligned with the legend (not at the figure's left edge): on the slide the logo sits under the left corner
    if win:
        nw = {t: int(w["n_window"].max()) for t, w in win.items()}
        cap = ("z vs each null, right = module better (conductance flipped). Nulls 1\u20132 size-matched: "
               f"nulls within \u00b125% of true size (n = {nw.get('seeds', '?')}, {nw.get('rewired', '?')}).")
    else:
        cap = ("z of the true module vs each null; right = module better (conductance flipped). "
               "p floors: Null 3 \u2264 0.001, Nulls 1\u20132 \u2264 0.01. NOT size-matched.")
    fig.text(left - 0.01, 0.09 / H, cap, fontsize=FS["tick"], color=C.MUTED, ha="left", va="top")
    pd.DataFrame(records).to_csv(out / "fig15a_values.csv", index=False)
    return save(fig, out / "fig15a_null_ladder.png")


def fig_distributions(val_dir: Path, summ: dict, out: Path, true_size: int, topo_metrics, size) -> Path | None:
    cols = [(f"{m}__neighbors", lab) for m, lab in FUNCTIONAL] + \
           [(m, TOPOLOGY[m]) for m in topo_metrics] + [("diag_size", "Module size\n(nodes)")]
    tags = [(t, s) for t, s, _ in NULLS if (val_dir / f"null_{t}_permutations.csv").exists()]
    if not tags:
        print("  [info] no null_*_permutations.csv found -- skipping fig15b")
        return None
    fig, axes = plt.subplots(len(tags), len(cols), figsize=size, squeeze=False)
    for i, (tag, short) in enumerate(tags):
        df = pd.read_csv(val_dir / f"null_{tag}_permutations.csv")
        for j, (key, lab) in enumerate(cols):
            ax = axes[i][j]
            if key not in df.columns or df[key].dropna().empty:
                ax.axis("off")
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center",
                        fontsize=FS["annot"], color=C.MUTED, style="italic")
                continue
            v = df[key].dropna().to_numpy()
            if key == "diag_size":
                tv = true_size
            else:
                tv = summ[tag].loc[key, "true_value"] if tag in summ and key in summ[tag].index else np.nan
            ax.hist(v, bins=30, color=C.BLUE_PALE, edgecolor="white", linewidth=0.4)
            if not np.isnan(tv):
                ax.axvline(tv, color=C.ORANGE, lw=1.8)
            fmt = (lambda x: f"{x:,.0f}") if key == "diag_size" else (lambda x: f"{x:.3g}")
            ax.text(0.99, 0.97, f"null {fmt(v.mean())}" + (f"\ntrue {fmt(tv)}" if not np.isnan(tv) else ""),
                    transform=ax.transAxes, ha="right", va="top", fontsize=FS["annot"], color=C.MUTED)
            if i == 0:
                panel_title(ax, lab)
            if j == 0:
                ax.set_ylabel(short, fontsize=FS["label"], color=C.INK, fontweight="bold")
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
    fig.subplots_adjust(hspace=0.55, wspace=0.18, left=0.04, right=0.995, top=0.88, bottom=0.14)
    caption(fig, "Pale = null permutations, orange line = true module; each panel has its own x-axis. "
                 "Null module sizes differ from the true module (right-hand column), so size-dependent "
                 "metrics (fractions, conductance, edge density) are not like-for-like; see size_adjust_nulls.py.",
            y=0.06)
    return save(fig, out / "fig15b_null_distributions.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation-dir", required=True, help="step 10 output dir (10_validation)")
    ap.add_argument("--out-dir", default="deck_figures")
    ap.add_argument("--scopes", nargs="+", default=["neighbors"], choices=["neighbors", "combined", "seeds"],
                    help="rows per null; default neighbours only. 'seeds' exists for Null 1 only and is circular.")
    ap.add_argument("--topology", nargs="*", default=["mean_internal_degree", "conductance"],
                    choices=list(TOPOLOGY), help="topology panels (edge_density is size-confounded)")
    ap.add_argument("--true-size", type=int, default=9940, help="true module size, for the appendix size panel")
    ap.add_argument("--z-basis", choices=["window", "raw"], default="window",
                    help="window (default): Nulls 1-2 compared only with null modules within +-25%% of the true size "
                         "(needs null_*_size_adjusted.csv from size_adjust_nulls.py); raw: unadjusted z")
    ap.add_argument("--figsize", nargs=2, default=None, metavar=("W", "H"),
                    help="override the canvas (default: painfig SLOT['band'] for fig15a, SLOT['full'] for fig15b)")
    args = ap.parse_args()

    use_theme()
    val = Path(args.validation_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summ = load_summaries(val)
    win = load_windows(val, args.z_basis)
    nrows = len(NULLS) * len(args.scopes)
    # SLOT['band'] is exactly right for one row per null; taller only if more rows are requested
    band_w, band_h = SLOT["band"]
    default_a = (band_w, band_h) if len(args.scopes) == 1 else (band_w, 0.31 * nrows + 1.07)
    fig_ladder(summ, out, tuple(args.scopes), list(args.topology), figsize(args.figsize, default_a), win or None)
    fig_distributions(val, summ, out, args.true_size, list(args.topology), figsize(args.figsize, SLOT["full"]))


if __name__ == "__main__":
    main()