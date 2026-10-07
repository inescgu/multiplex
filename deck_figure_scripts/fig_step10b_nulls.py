#!/usr/bin/env python3
"""
fig_step10b_nulls.py -- deck figure from step10b_functional_permutation_test_multiplex.py.

Replaces, for slide purposes:  null1_combined_distributions.png
                               null1_split_distributions.png
                               null2_combined_distributions.png
                               null2_split_distributions.png
                               null3_combined_distributions.png
                               null3_split_distributions.png
                               (eleven histograms, currently all on one slide)

Reads (from --step10b-dir, i.e. step 10b's output_path):
    null1_degree_matched_seeds_combined.csv / _split.csv
    null2_rewired_network_combined.csv      / _split.csv      (if run)
    null3_fixed_seed_random_neighbors_combined.csv / _split.csv (if run)
  each with columns: metric, true_value, null_mean, null_std, z_score, p_value

Optionally also reads raw per-permutation values, if step10b was patched to
write them (see step10b_raw_nulls.patch shipped alongside this script):
    null1_raw_permutations.csv, null2_raw_permutations.csv, null3_raw_permutations.csv
  each with columns: perm, scope ('combined'|'seeds'|'neighbors'), metric, value

Writes (to --out-dir):
    fig15a_null_ladder.png          ALWAYS. One panel per metric; three rows per
                                    panel, one per null. This is the slide figure.
    fig15b_null_distributions.png   only with --raw-dir / raw CSVs present.
                                    The histograms, redrawn to the deck theme,
                                    for the appendix.

What changed from the original plots, and why
----------------------------------------------
1. ELEVEN HISTOGRAMS ON ONE SLIDE IS NOT A RESULT, IT IS A SCREENSHOT. Nobody
   reads a 2cm-wide histogram. The question the slide asks -- "is the module
   more informative than a random one?" -- is answered by one number per
   (metric, null): how far the true value sits from the null, and whether that
   survives. fig15a is a forest plot of exactly that, with the three nulls
   stacked so the reader can see the thing that actually matters: whether the
   signal survives as you move from Null 1 to Null 3.

2. The nulls are ordered as a LADDER, not as a list. Null 1 holds nothing
   fixed, Null 2 fixes the seeds and scrambles the network, Null 3 fixes the
   seeds and discards the network entirely. Each one strips away one possible
   explanation. The figure draws them in that order with that reading printed
   on the axis, because "significant against Null 1 only" and "significant
   against all three" are completely different claims and the old layout made
   them look alike.

3. Seeds vs neighbors is the decomposition the whole slide turns on (are we
   learning anything from the random-walk expansion, or just re-reading the
   seed list?), so it gets its own marker rather than being buried in a
   separate six-panel PNG with the same axes.

4. z-scores, not raw metric values, on a shared axis. The metrics are a
   fraction, a fraction, and a -log10 p; plotting them on one axis in native
   units is meaningless. z puts them on one scale, and the p-value is printed
   next to each point so nothing is hidden behind the standardisation.

5. Where a null produced too few valid permutations, or a metric was NaN, the
   row is drawn as an explicit hollow marker labelled "n/a" instead of
   silently vanishing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from painfig import C, FS, caption, save, use_theme

#: (file stem, short label, what this null removes as an explanation)
NULLS = [
    ("null1_degree_matched_seeds", "Null 1",
     "random seeds, real network\nrules out: any gene set would score this"),
    ("null2_rewired_network", "Null 2",
     "real seeds, rewired network\nrules out: the network adds nothing"),
    ("null3_fixed_seed_random_neighbors", "Null 3",
     "real seeds, random neighbours\nrules out: the walk adds nothing"),
]

#: Pretty names, and the order metrics should appear in.
METRIC_LABEL = {
    "drug_target_fraction_disease": "Pain-indication drug targets",
    "drug_target_fraction_any": "Drug targets, any indication",
    "pain_phenotype_fraction": "Pain-phenotype genes",
    "go_bp_best_neg_log10p": "GO:BP enrichment (best \u2212log\u2081\u2080p)",
}
METRIC_ORDER = list(METRIC_LABEL)


def pretty(metric: str) -> str:
    return METRIC_LABEL.get(metric, metric.replace("_", " "))


def load_summaries(src: Path) -> pd.DataFrame:
    rows = []
    for stem, short, _ in NULLS:
        for scope, suffix in (("combined", "_combined.csv"), ("split", "_split.csv")):
            p = src / f"{stem}{suffix}"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            df["null"] = short
            if scope == "combined":
                df["scope"] = "combined"
                df["base_metric"] = df["metric"]
            else:
                df["scope"] = np.where(df["metric"].str.endswith("__seeds"), "seeds", "neighbors")
                df["base_metric"] = df["metric"].str.replace("__seeds$|__neighbors$", "", regex=True)
            rows.append(df)
    if not rows:
        raise SystemExit(f"ERROR: no null*_combined.csv under {src} -- point --step10b-dir at step 10b's output dir")
    return pd.concat(rows, ignore_index=True)


def p_text(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p == 0:
        return "p < 1/n"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def fig_ladder(summ: pd.DataFrame, out: Path, scopes=("combined", "neighbors")) -> Path:
    metrics = [m for m in METRIC_ORDER if m in set(summ["base_metric"])]
    metrics += sorted(set(summ["base_metric"]) - set(metrics))
    metrics = [m for m in metrics if summ.loc[summ["base_metric"] == m, "z_score"].notna().any()]
    if not metrics:
        raise SystemExit("ERROR: every z_score is NaN -- nothing to plot")

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n + 1.6, 3.5), sharex=True, squeeze=False)
    axes = axes[0]

    # One row per (null, scope). Nulls top to bottom in ladder order.
    rows = [(short, sc) for short in [n[1] for n in NULLS] for sc in scopes]
    ypos = {r: i for i, r in enumerate(rows[::-1])}

    marker_for = {"combined": "o", "neighbors": "D", "seeds": "s"}
    size_for = {"combined": 58, "neighbors": 42, "seeds": 42}

    allz = summ.loc[summ["base_metric"].isin(metrics), "z_score"].dropna()
    zmax = float(np.nanmax(np.abs(allz))) if len(allz) else 3.0
    xlim = (min(-1.2, -0.12 * zmax), zmax * 1.42 + 1.0)

    for ax, metric in zip(axes, metrics):
        sub = summ[summ["base_metric"] == metric]
        ax.axvspan(xlim[0], 1.96, color=C.CARD, zorder=0)
        ax.axvline(1.96, color=C.GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axvline(0, color=C.GREY_LT, lw=0.9, zorder=1)

        for (short, sc), y in ypos.items():
            r = sub[(sub["null"] == short) & (sub["scope"] == sc)]
            if not len(r) or pd.isna(r.iloc[0]["z_score"]):
                ax.scatter([xlim[0] + 0.35], [y], marker=marker_for[sc], s=size_for[sc],
                           facecolor="none", edgecolor=C.GREY, linewidth=0.9, zorder=4)
                ax.text(xlim[0] + 0.75, y, "not run" if not len(r) else "n/a",
                        fontsize=FS["tick"], color=C.MUTED, va="center")
                continue
            z = float(r.iloc[0]["z_score"])
            p = float(r.iloc[0]["p_value"])
            sig = (not np.isnan(p)) and p < 0.05
            col = C.ORANGE if sig else C.BLUE_PALE
            ax.plot([0, z], [y, y], color=col, lw=2.0, solid_capstyle="round", zorder=3)
            ax.scatter([z], [y], marker=marker_for[sc], s=size_for[sc], color=col,
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(z + 0.06 * (xlim[1] - xlim[0]), y, p_text(p), fontsize=FS["tick"],
                    color=C.INK if sig else C.MUTED, va="center")

        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(pretty(metric), fontsize=FS["panel"], color=C.BLUE,
                     fontweight="bold", loc="left")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks([])

    # Row labels, written once on the left of the first panel.
    for (short, sc), y in ypos.items():
        label = short if sc == "combined" else f"{short} \u00b7 neighbours only"
        axes[0].text(-0.035, y, label, transform=axes[0].get_yaxis_transform(),
                     ha="right", va="center", fontsize=FS["tick"],
                     color=C.INK if sc == "combined" else C.MUTED,
                     fontweight="bold" if sc == "combined" else "normal")

    fig.supxlabel("z-score of the true module against the null  (dashed = z 1.96; shaded = not past it)",
                  fontsize=FS["tick"], color=C.MUTED, y=0.0)

    caption(fig, "Orange = p < 0.05; circles are the whole module, diamonds the random-walk neighbours "
                 "alone. Read top to bottom: each null removes one more explanation. Null 1 = "
                 "degree-matched random seeds on the real network. Null 2 = true seeds, degree-preserving "
                 "rewired network. Null 3 = true seeds, degree-matched random neighbours and no walk at "
                 "all. A result that survives Null 3 is a result about the expansion, not the seed list.",
            y=-0.10)
    fig.subplots_adjust(wspace=0.18, left=0.17, bottom=0.17)
    return save(fig, out / "fig15a_null_ladder.png")


def fig_distributions(raw: pd.DataFrame, summ: pd.DataFrame, out: Path) -> Path:
    """Appendix figure: the histograms, redrawn once in the deck theme."""
    metrics = [m for m in METRIC_ORDER if m in set(raw["metric"])]
    metrics += sorted(set(raw["metric"]) - set(metrics))
    nulls = [short for _, short, _ in NULLS if short in set(raw["null"])]
    nrow, ncol = len(nulls), len(metrics)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol + 1.0, 1.9 * nrow + 0.8),
                             squeeze=False)

    for i, short in enumerate(nulls):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            vals = raw[(raw["null"] == short) & (raw["metric"] == metric) &
                       (raw["scope"] == "combined")]["value"].dropna().values
            s = summ[(summ["null"] == short) & (summ["base_metric"] == metric) &
                     (summ["scope"] == "combined")]
            if not len(vals) or not len(s):
                ax.axis("off")
                continue
            ax.hist(vals, bins=36, color=C.BLUE_PALE, edgecolor="white", linewidth=0.4)
            tv = float(s.iloc[0]["true_value"])
            ax.axvline(tv, color=C.ORANGE, lw=1.8)
            ax.text(0.98, 0.93, p_text(float(s.iloc[0]["p_value"])), transform=ax.transAxes,
                    ha="right", va="top", fontsize=FS["tick"], color=C.INK)
            if i == 0:
                ax.set_title(pretty(metric), fontsize=FS["panel"], color=C.BLUE,
                             fontweight="bold", loc="left")
            if j == 0:
                ax.set_ylabel(short, fontsize=FS["label"], color=C.INK, fontweight="bold")
            ax.tick_params(labelsize=FS["tick"])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)

    caption(fig, "Grey = null permutations, orange line = the true module. Same bins across a column.", y=-0.02)
    fig.subplots_adjust(hspace=0.55, wspace=0.18)
    return save(fig, out / "fig15b_null_distributions.png")


def load_raw(src: Path) -> pd.DataFrame | None:
    frames = []
    for stem, short, _ in NULLS:
        tag = stem.split("_")[0]  # null1 / null2 / null3
        p = src / f"{tag}_raw_permutations.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["null"] = short
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step10b-dir", required=True,
                    help="step 10b output dir (holds null1_*_combined.csv etc.)")
    ap.add_argument("--out-dir", default="deck_figures")
    ap.add_argument("--raw-dir", default=None,
                    help="where null*_raw_permutations.csv live (default: --step10b-dir)")
    ap.add_argument("--scopes", nargs="+", default=["combined", "neighbors"],
                    choices=["combined", "seeds", "neighbors"],
                    help="which rows to draw per null (default: whole module + neighbours)")
    args = ap.parse_args()

    use_theme()
    src = Path(args.step10b_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summ = load_summaries(src)
    fig_ladder(summ, out, scopes=tuple(args.scopes))

    raw = load_raw(Path(args.raw_dir) if args.raw_dir else src)
    if raw is None:
        print("  [info] no null*_raw_permutations.csv found -- skipping fig15b. "
              "Apply step10b_raw_nulls.patch and re-run step 10b if you want the histograms.")
    else:
        fig_distributions(raw, summ, out)


if __name__ == "__main__":
    main()