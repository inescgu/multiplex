#!/usr/bin/env python3
"""
fig_step12g_removed.py -- deck figures for the three hub-removal slides, from
step12g_removed_genes.py.

Reads <removed-dir> = 12_hub_removal/hub_removal/removed_genes/ :

    removed_vs_kept_stats.csv                                 (profile)
    top_cutting_nodes_<mode>_static.csv                       (splits; top 50 only)
    node_annotation_<mode>_static.csv                         (splits; OPTIONAL, all nodes)
    enrichment_module_background_removed_<attack>_all.csv     (enrichment)
    enrichment_module_background_removed_<attack>_seed.csv    (enrichment)

Subcommands (each writes a PNG plus a *_values.csv with exactly what is drawn):

    profile      fig35a_removed_profile.png      slide 35, right column
    splits       fig36a_split_timing.png         slide 36, left column
    enrichment   fig37a_removed_enrichment.png   slide 37 (appendix), full width
    all          all three

Usage (same pattern as fig_step12_enrichment.py):

    python fig_step12g_removed.py all \\
        --removed-dir $BASE/12_hub_removal/hub_removal/removed_genes \\
        --attack degree_static --out-dir deck_figures

Sizing (painfig rules)
----------------------
Canvases are sized to the slot the PNG lands in on the 10 x 5.625 in slide, so
place each PNG at 1:1 and do not resize it in PowerPoint:

    profile      4.85 x 3.75 in   (right column; PNG is ~5.1 x 3.2 in after tight cropping)
    splits       5.4 x 3.6 in   (left column)
    enrichment   9.2 x 3.3 in   (painfig SLOT["full"] width)

Which attack
------------
Default is degree_static. Degree attacks are deterministic, so the removal set is
exactly the one that produced f_c. Betweenness attacks are only valid if
step12g was run with the SAME --betweenness-k (and seed) as the main run; the
script prints a warning for any attack whose name contains 'betweenness'.

What the figures do and do not show
-----------------------------------
* profile: removed vs kept shares. The removed set is, by construction, the
  top-ranked nodes, so seed and degree contrasts are partly circular; cut-points,
  community roles and layers are the less circular readouts. Cut-points are
  computed once on the INTACT module.
* splits: split_off = (drop in largest connected component caused by removing
  that node, in attack order) - 1. It depends on order, so it measures timing
  near the collapse, not intrinsic gene importance. With node_annotation present,
  every removed node is drawn (grey) and the 50 largest splits are highlighted;
  with only top_cutting_nodes present, only those 50 are drawn and the caption
  says so.
* enrichment: Enrichr against the MODULE as background. Terms with BH q < 0.05, 10-200
  overlapping genes and odds ratio >= 3, ranked by q (max 3 per library, near-duplicate
  gene sets merged); the bar is -log10(q), the label gives odds ratio and overlap.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from painfig import (C, FS, caption, fiber_color, figsize, panel_title,
                     reference_line, save, shorten_layer, sig_colors, use_theme)

WITH_CAPTION = False       # the slide carries its own caption; --with-caption draws one in the PNG too


def cap(fig, text, **kw):
    """painfig.caption, but only with --with-caption (otherwise it duplicates the slide's caption)."""
    if WITH_CAPTION:
        caption(fig, text, **kw)


SLOT_RIGHT = (4.85, 3.75)     # right column of a text-left / figure-right slide
SLOT_LEFT = (5.4, 3.6)      # left column of a figure-left / text-right slide
SLOT_FULL = (9.2, 3.3)
MINUS = "\u2212"


# --------------------------------------------------------------------------- helpers
def fiber_of(cell_type: str) -> str:
    if cell_type == "ATF3":
        return "ATF3"
    return "C-fiber" if cell_type.startswith("C-") else "A-fiber"


def layer_order(names):
    """Deck-wide fixed order: A-fiber, then C-fiber, then ATF3; alphabetical within."""
    rank = {"A-fiber": 0, "C-fiber": 1, "ATF3": 2}
    return sorted(names, key=lambda n: (rank[fiber_of(n)], n))


def warn_if_betweenness(attack: str) -> None:
    if "betweenness" in attack:
        print("  [warning] betweenness-based removal sets are only valid if step12g used the same "
              "--betweenness-k and --seed as the main run (f_c_f90_observed.csv). Check before "
              "putting this on a slide.")


def get_row(stats: pd.DataFrame, test: str, feature: str) -> pd.Series:
    r = stats[(stats.test == test) & (stats.feature == feature)]
    if r.empty:
        raise SystemExit(f"ERROR: no row test={test} feature={feature} in removed_vs_kept_stats.csv")
    return r.iloc[0]


def fmt_q(q: float) -> str:
    return "q < 0.001" if q < 0.001 else f"q = {q:.3f}"


# --------------------------------------------------------------------------- profile
def fig_profile(removed_dir: Path, attack: str, out: Path, size) -> Path:
    st = pd.read_csv(removed_dir / "removed_vs_kept_stats.csv")
    st = st[st.attack == attack]
    if st.empty:
        raise SystemExit(f"ERROR: attack '{attack}' not in removed_vs_kept_stats.csv")

    seed = get_row(st, "seed_status_fisher", "is_seed")
    art = get_row(st, "mannwhitney", "is_articulation_point")
    hub_p = get_row(st, "role_hypergeom", "provincial_hub")
    hub_c = get_row(st, "role_hypergeom", "connector_hub")
    con = get_row(st, "role_hypergeom", "connector_non_hub")

    bars = [  # label, removed %, kept %, q
        ("Seed genes", seed.removed_value, seed.kept_value, seed.q_bh),
        ("Cut-points", art.removed_value, art.kept_value, art.q_bh),
        ("Community hubs", hub_p.removed_value + hub_c.removed_value,
         hub_p.kept_value + hub_c.kept_value, max(hub_p.q_bh, hub_c.q_bh)),
        ("Connectors\n(non-hub)", con.removed_value, con.kept_value, con.q_bh),
    ]

    lay = st[st.test == "cell_type_hypergeom"].copy()
    order = layer_order(lay.feature.tolist())
    lay = lay.set_index("feature").loc[order].reset_index()

    W, H = size
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.0], left=0.150, right=0.995,
                          top=1 - 0.42 / H, bottom=0.98 / H, wspace=1.0)

    # ---- panel a: removed vs kept shares
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(bars))[::-1]
    h = 0.34
    for yi, (lab, rm, kp, q) in zip(y, bars):
        ax.barh(yi + h / 2, rm * 100, height=h, color=C.BLUE_MID, zorder=3)
        ax.barh(yi - h / 2, kp * 100, height=h, color=C.GREY_LT, zorder=3)
        ax.text(rm * 100 + 0.25, yi + h / 2, f"{rm * 100:.1f}%", va="center", ha="left",
                fontsize=FS["annot"], color=C.INK, fontweight="bold")
        ax.text(kp * 100 + 0.25, yi - h / 2, f"{kp * 100:.1f}%", va="center", ha="left",
                fontsize=FS["annot"], color=C.MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=FS["label"], color=C.INK)
    ax.set_xlim(0, max(b[1] for b in bars) * 100 * 1.38)
    ax.set_ylim(-0.6, len(bars) - 0.4)
    ax.set_xlabel("% of nodes in each group", fontsize=FS["tick"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    panel_title(ax, "Who is removed")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C.BLUE_MID, label="removed"), Patch(color=C.GREY_LT, label="kept")],
              loc="lower right", fontsize=FS["tick"], labelcolor=C.MUTED, handlelength=0.9)

    # ---- panel b: fold enrichment per layer
    ax2 = fig.add_subplot(gs[0, 1])
    yy = np.arange(len(lay))[::-1]
    cols = sig_colors(lay.q_bh.tolist())
    reference_line(ax2, 1.0)
    for yi, (_, r), c in zip(yy, lay.iterrows(), cols):
        ax2.plot([1.0, r.effect], [yi, yi], color=c, lw=1.8, solid_capstyle="round", zorder=3)
        ax2.scatter([r.effect], [yi], s=20, color=c, edgecolor="white", linewidth=0.6, zorder=4)
        if c == C.ORANGE:
            ax2.text(r.effect + 0.03, yi, f"{r.effect:.2f}\u00d7", va="center", ha="left",
                     fontsize=FS["annot"], color=C.INK, fontweight="bold")
    ax2.set_yticks(yy)
    ax2.set_yticklabels([shorten_layer(n, 20) for n in lay.feature], fontsize=FS["tick"])
    for tl, n in zip(ax2.get_yticklabels(), lay.feature):
        tl.set_color(fiber_color(fiber_of(n)))
    ax2.set_ylim(-0.8, len(lay) - 0.2)
    ax2.set_xlim(0.45, 1.55)
    ax2.set_xlabel("fold enrichment", fontsize=FS["tick"])
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)
    panel_title(ax2, "Which layers")

    cap(fig, "Degree attack, fixed ranking. Cut-point = a gene whose removal alone disconnects part "
                 "of the module (computed on the intact module). Orange = BH q < 0.05 "
                 "(hypergeometric); pale blue = not significant.", y=0.46 / H, chars_per_inch=17.0)

    rows = [dict(panel="who", group=b[0].replace("\n", " "), removed_pct=b[1] * 100,
                 kept_pct=b[2] * 100, q_bh=b[3]) for b in bars]
    rows += [dict(panel="layer", group=r.feature, removed_pct=r.removed_value * 100,
                  kept_pct=r.kept_value * 100, fold=r.effect, q_bh=r.q_bh) for _, r in lay.iterrows()]
    pd.DataFrame(rows).to_csv(out / "fig35a_values.csv", index=False)
    return save(fig, out / "fig35a_removed_profile.png")


# --------------------------------------------------------------------------- splits
def load_split_data(removed_dir: Path, mode: str):
    full = removed_dir / f"node_annotation_{mode}_static.csv"
    top = removed_dir / f"top_cutting_nodes_{mode}_static.csv"
    if full.exists():
        df = pd.read_csv(full)
        df = df[df.removed == True].copy()                      # noqa: E712
        topdf = df.sort_values("split_off", ascending=False).head(50)
        return df, topdf, True
    if top.exists():
        topdf = pd.read_csv(top)
        print(f"  [info] {full.name} not found -- drawing only the 50 largest splits from {top.name}")
        return None, topdf, False
    raise SystemExit(f"ERROR: neither {full.name} nor {top.name} under {removed_dir}")


def fig_splits(removed_dir: Path, attack: str, out: Path, size) -> Path:
    mode = attack.replace("_static", "")
    if attack.endswith("cascading"):
        raise SystemExit("ERROR: split_off is only defined for static attacks")
    df, top, have_all = load_split_data(removed_dir, mode)
    n_rm = int(df.removal_rank.max()) if have_all else int(top.removal_rank.max())

    W, H = size
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.115, right=0.985, top=1 - 0.40 / H, bottom=1.02 / H)

    if have_all:
        rest = df[~df.node.isin(top.node)]
        ax.scatter(rest.removal_rank, rest.split_off, s=3, color=C.GREY_LT, zorder=2, linewidths=0)

    col = [C.BLUE if s else C.ORANGE for s in top.is_seed]
    ax.scatter(top.removal_rank, top.split_off, s=22, c=col, edgecolor="white", linewidth=0.5, zorder=4)
    ax.axvline(n_rm, color=C.MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(n_rm, 1.0, f"collapse (f$_c$): {n_rm:,} removed", transform=ax.get_xaxis_transform(),
            fontsize=FS["tick"], color=C.MUTED, ha="right", va="bottom")

    # label the five largest cuts; hand-placed offsets keep them clear of each other
    t5 = top.sort_values("split_off", ascending=False).head(5).reset_index(drop=True)
    offs = [(-60, 6, "right"), (-34, 4, "right"), (-52, 4, "right"), (-32, 4, "right"), (-4, 34, "right")]
    # label offsets are in points, anchored at the data point
    for (_, r), (dx, dy, ha) in zip(t5.iterrows(), offs):
        ax.annotate(f"{r.gene} \u00b7 {shorten_layer(r.cell_type, 22)}", (r.removal_rank, r.split_off),
                    xytext=(dx, dy), textcoords="offset points", fontsize=FS["annot"], ha=ha,
                    va="center", color=C.INK,
                    arrowprops=dict(arrowstyle="-", color=C.GREY, lw=0.6, shrinkA=0, shrinkB=2))

    # the contrast case: the highest-degree node in the plotted set
    src = df if have_all else top
    hub = src.sort_values("degree", ascending=False).iloc[0]
    ax.scatter([hub.removal_rank], [hub.split_off], s=40, facecolor="white", edgecolor=C.INK,
               linewidth=1.2, zorder=5)
    ax.annotate(f"{hub.gene}: degree {int(hub.degree)}, removed #{int(hub.removal_rank)},\n"
                f"splits off {int(hub.split_off)}",
                (hub.removal_rank, hub.split_off), xytext=(26, 52), textcoords="offset points",
                fontsize=FS["annot"], color=C.INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=C.INK, lw=0.8, shrinkA=0, shrinkB=3))

    ax.set_xlim(-n_rm * 0.01, n_rm * 1.03)
    ax.set_ylim(-15, max(top.split_off) * 1.12)
    ax.set_xlabel("position in the removal order (1 = removed first)", fontsize=FS["label"])
    ax.set_ylabel("nodes cut off from the largest piece", fontsize=FS["label"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=C.BLUE, label="seed gene", markersize=5),
         Line2D([], [], marker="o", ls="", color=C.ORANGE, label="bridging gene", markersize=5),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=C.INK,
                label="highest-degree gene shown", markersize=5)]
    if have_all:
        h.append(Line2D([], [], marker="o", ls="", color=C.GREY_LT, label="all other removals", markersize=3))
    ax.legend(handles=h, loc="upper left", fontsize=FS["tick"], labelcolor=C.MUTED, handletextpad=0.2)

    sub = ("50 largest splits highlighted; every other removal in grey." if have_all else
           "Only the 50 largest single-node splits are drawn.")
    cap(fig, f"Degree attack, fixed ranking. Each point is one gene-in-cell-type node. {sub} "
                 "Split size depends on removal order, so it reflects timing near the collapse, "
                 "not intrinsic gene importance.", y=0.46 / H, chars_per_inch=17.0)

    out_cols = ["gene", "cell_type", "is_seed", "degree", "removal_rank", "split_off"]
    (df if have_all else top)[out_cols].to_csv(out / "fig36a_values.csv", index=False)
    return save(fig, out / "fig36a_split_timing.png")


# --------------------------------------------------------------------------- enrichment
LIB_SHORT = {"GO_Biological_Process_2023": "GO BP", "GO_Cellular_Component_2023": "GO CC",
             "GO_Molecular_Function_2023": "GO MF", "KEGG_2021_Human": "KEGG",
             "Reactome_2022": "Reactome"}


def clean_term(t: str) -> str:
    t = re.sub(r"\s*\(GO:\d+\)", "", t)
    t = re.sub(r"\s*R-HSA-\d+", "", t)
    return t.strip()


def pick_terms(path: Path, top_n: int, per_lib_cap: int, min_overlap: int, max_overlap: int,
               min_or: float, libs) -> pd.DataFrame:
    """Terms with BH q < 0.05, min_overlap <= genes <= max_overlap and OR >= min_or, ranked by q.
    At most per_lib_cap per library, and a term is skipped if its gene set overlaps an
    already-chosen one (Jaccard >= 0.6), so 'Ribosome' is not shown four times."""
    e = pd.read_csv(path)
    e["gene_set"] = e["Genes"].fillna("").map(lambda s: frozenset(g for g in str(s).split(";") if g))
    e["overlap"] = e.gene_set.map(len)
    e = e[(e["Adjusted P-value"] < 0.05) & e.overlap.between(min_overlap, max_overlap)
          & (e["Odds Ratio"] >= min_or) & e.Gene_set.isin(libs)]
    e = e.sort_values("Adjusted P-value")
    kept, per = [], {}
    for _, r in e.iterrows():
        if per.get(r.Gene_set, 0) >= per_lib_cap:
            continue
        if any(len(r.gene_set & k.gene_set) / len(r.gene_set | k.gene_set) >= 0.6 for k in kept):
            continue
        kept.append(r)
        per[r.Gene_set] = per.get(r.Gene_set, 0) + 1
        if len(kept) >= top_n:
            break
    p = pd.DataFrame(kept)
    p["term"] = p.Term.map(clean_term)
    p["lib"] = p.Gene_set.map(LIB_SHORT)
    p["neglogq"] = -np.log10(p["Adjusted P-value"].clip(lower=1e-300))
    return p.sort_values("neglogq", ascending=True)


def fig_enrichment(removed_dir: Path, attack: str, out: Path, size, top_n: int, min_overlap: int,
                   max_overlap: int) -> Path:
    libs = ["GO_Biological_Process_2023", "GO_Cellular_Component_2023", "GO_Molecular_Function_2023",
            "KEGG_2021_Human", "Reactome_2022"]
    sets = [("all", "All removed genes"), ("seed", "Removed seed genes")]
    data = {}
    for tag, _ in sets:
        f = removed_dir / f"enrichment_module_background_removed_{attack}_{tag}.csv"
        if not f.exists():
            f = removed_dir / "enrichment" / f.name
        if not f.exists():
            raise SystemExit(f"ERROR: {f.name} not found (run step12g with --module-background)")
        data[tag] = pick_terms(f, top_n, 3, min_overlap, max_overlap, 3.0, libs)

    W, H = size
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(left=0.25, right=0.995, top=1 - 0.40 / H, bottom=1.08 / H, wspace=1.12)
    lib_col = {"GO BP": C.BLUE, "GO CC": C.BLUE_MID, "GO MF": C.BLUE_PALE,
               "KEGG": C.ORANGE, "Reactome": C.ORANGE_LT}
    recs = []
    for ax, (tag, title) in zip(axes, sets):
        p = data[tag].reset_index(drop=True)
        y = np.arange(len(p))
        ax.barh(y, p.neglogq, color=[lib_col[l] for l in p.lib], height=0.68, zorder=3)
        lab = [t if len(t) <= 46 else t[:45] + "\u2026" for t in p.term]
        ax.set_yticks(y)
        ax.set_yticklabels(lab, fontsize=7.5, color=C.INK)
        ax.tick_params(axis="y", length=0)
        xmax = p.neglogq.max()
        for yi, (_, r) in zip(y, p.iterrows()):
            orr = "\u221e" if not np.isfinite(r["Odds Ratio"]) else f"{r['Odds Ratio']:.0f}"
            ax.text(r.neglogq + xmax * 0.02, yi, f"OR {orr} \u00b7 {r.overlap} genes",
                    va="center", ha="left", fontsize=FS["annot"] - 0.5, color=C.MUTED)
            recs.append(dict(set=tag, library=r.lib, term=r.term, neglog10_q=r.neglogq,
                             odds_ratio=r["Odds Ratio"], overlap=r.overlap,
                             q_bh=r["Adjusted P-value"]))
        ax.set_xlim(0, xmax * 1.62)
        ax.set_ylim(-0.7, len(p) - 0.3)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("\u2212log$_{10}$ BH q", fontsize=FS["tick"])
        panel_title(ax, title)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for l, c in lib_col.items()], loc="lower left", ncol=5,
               fontsize=FS["tick"], labelcolor=C.MUTED, handlelength=0.9, columnspacing=1.1,
               bbox_to_anchor=(0.0, 0.50 / H))
    cap(fig, f"Enrichr against the module as background (not the genome). Top {top_n} terms by BH q "
                 f"(q < 0.05, OR \u2265 3, {min_overlap}\u2013{max_overlap} genes, \u2264 3 per library, "
                 "overlapping terms merged). OR = odds ratio; \u221e = every module gene in that term was removed.",
            y=0.28 / H, chars_per_inch=17.0)
    pd.DataFrame(recs).to_csv(out / "fig37a_values.csv", index=False)
    return save(fig, out / "fig37a_removed_enrichment.png")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["profile", "splits", "enrichment", "all"])
    ap.add_argument("--removed-dir", required=True,
                    help="12_hub_removal/hub_removal/removed_genes (enrichment may be in its enrichment/ subfolder)")
    ap.add_argument("--attack", default="degree_static",
                    help="e.g. degree_static, degree_cascading (profile/enrichment only), betweenness_static")
    ap.add_argument("--out-dir", default="deck_figures")
    ap.add_argument("--top-n", type=int, default=8, help="enrichment: terms shown per panel")
    ap.add_argument("--min-overlap", type=int, default=10, help="enrichment: minimum genes overlapping a term")
    ap.add_argument("--max-overlap", type=int, default=200, help="enrichment: drop terms broader than this")
    ap.add_argument("--with-caption", action="store_true",
                    help="also draw the method caption inside the PNG (default: off, the slide has its own)")
    ap.add_argument("--figsize", nargs=2, default=None, metavar=("W", "H"),
                    help="override the canvas of the figure(s) you run")
    a = ap.parse_args()

    global WITH_CAPTION
    WITH_CAPTION = a.with_caption
    use_theme()
    rd, out = Path(a.removed_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warn_if_betweenness(a.attack)
    if a.what in ("profile", "all"):
        fig_profile(rd, a.attack, out, figsize(a.figsize, SLOT_RIGHT))
    if a.what in ("splits", "all"):
        fig_splits(rd, a.attack, out, figsize(a.figsize, SLOT_LEFT))
    if a.what in ("enrichment", "all"):
        fig_enrichment(rd, a.attack, out, figsize(a.figsize, SLOT_FULL), a.top_n, a.min_overlap, a.max_overlap)


if __name__ == "__main__":
    main()