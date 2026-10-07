#!/usr/bin/env python3
"""
fig_step12_enrichment.py  --  enrichment figures in the deck's figure style.

Replaces the lollipop/heatmap pair in run_multiplex_visualizations.py for the
"how many clusters?" problem: nothing here scales with the number of clusters.

Two input modes
---------------
GENE-LEVEL (needs only the combined TSV from ResultsCombiner):
    python fig_step12_enrichment.py gene --tsv combined_gene_results_multiplex_significant_only.tsv --out figs
  -> fig31a_source_saturation.png, fig32a_direction.png, fig32b_celltype.png

CLUSTER-LEVEL (needs the per-analyzer *_summarized_results.csv files, and
optionally celltype_facets.csv, which run_multiplex_enrichment_pipeline.py writes):
    python fig_step12_enrichment.py grid --output-dir ./output --ppi-source combined26_multiplex \
        --facets ./output/combined26_multiplex/celltype_facets.csv --out figs --max-clusters 20
  -> fig33a_lollipop_grid_page{i}of{n}.png   (capped: never more than --max-clusters columns)
  Preferred: build_cluster_enrichment_table.py first, then  grid --table cluster_enrichment_long.csv [--tier 1]

Style comes from painfig.py (keep it next to this file): transparent PNGs sized to their slide slot,
place them at 1:1. Orange = BH q < 0.05 in the grid; in the gene-level figures orange = neighbours (painfig convention).
"""
import argparse, collections, logging, math, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from painfig import (C, FS, SLOT, use_theme, figsize, panel_title, sig_colors,
                     bar_value_labels, caption, save)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
use_theme()
NAVY, ORANGE, PALE, GREY = C.BLUE, C.ORANGE, C.BLUE_PALE, C.GREY   # aliases used below

SOURCES = ["proteomics", "neuropathic_pain", "orphanet", "omim", "phenotypes", "spatial"]
LABEL = {"proteomics": "Proteomics (sex)", "neuropathic_pain": "Neuropathic pain", "orphanet": "Orphanet",
         "omim": "OMIM series", "phenotypes": "Phenotypes", "spatial": "Spatial cell type"}


def _list(v):
    return v.split(",") if isinstance(v, str) and v else []


# ------------------------------------------------------------ gene-level figs
def load_tsv(path):
    df = pd.read_csv(path, sep="\t")
    for s in SOURCES:
        df[f"hit_{s}"] = df[f"{s}_enrichments"].notna() & (df[f"{s}_enrichments"].astype(str) != "")
    df["n_sources"] = df[[f"hit_{s}" for s in SOURCES]].sum(axis=1)
    return df


# Slots (inches) are the physical size each PNG occupies on the slide; place at 1:1.
SLOT_SATURATION = (7.0, 2.6)     # slide 31, left of the callouts
SLOT_PANEL = (4.5, 2.7)          # slide 32, two across


def fig_saturation(df, out, fs=None):
    n = len(df)
    fig, (a, b) = plt.subplots(1, 2, figsize=figsize(fs, SLOT_SATURATION),
                               gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.55})
    cnt = sorted(((LABEL[s], int(df[f"hit_{s}"].sum())) for s in SOURCES), key=lambda t: t[1])
    bars = a.barh(range(len(cnt)), [c for _, c in cnt], color=C.BLUE, height=0.62)
    a.set_yticks(range(len(cnt))); a.set_yticklabels([l for l, _ in cnt])
    a.set_xlim(0, n * 1.3); a.set_xticks([0, n // 2, n]); a.set_xticklabels(["0", f"{n // 2:,}", f"{n:,}"])
    a.set_xlabel("genes with a hit")
    for bar, (_, c) in zip(bars, cnt):
        a.text(c + n * 0.015, bar.get_y() + bar.get_height() / 2, f"{c:,} ({c / n:.0%})",
               va="center", fontsize=FS["annot"], color=C.INK)
    panel_title(a, "Genes with a hit, by source")

    seed = df["is_seed_any_analysis"].astype(bool)
    ks = np.arange(1, 7); w = 0.38
    for off, mask, col, lab in [(-w / 2, seed, C.BLUE, f"seeds (n={int(seed.sum()):,})"),
                                (w / 2, ~seed, C.ORANGE, f"neighbours (n={int((~seed).sum()):,})")]:
        b.bar(ks + off, [(df.loc[mask, "n_sources"] == k).mean() * 100 for k in ks], width=w, color=col, label=lab)
    b.set_xticks(ks); b.set_xlabel("sources with a hit (of 6)"); b.set_ylabel("% of genes in group")
    b.legend(loc="upper left")
    panel_title(b, "Sources per gene")
    caption(fig, f"Unadjusted cluster-level p < 0.05; {n:,} genes in the combined table. A gene counts once per source "
                 "however many of its nested clusters are significant.", y=-0.1)
    return save(fig, out)


def _direction_counts(df, col, a_lab, b_lab):
    toks = df[col].dropna().astype(str).str.split(",")
    has_a = np.array([a_lab in t for t in toks]); has_b = np.array([b_lab in t for t in toks])
    return int((has_a & ~has_b).sum()), int((has_b & ~has_a).sum()), int((has_a & has_b).sum())


def fig_direction(df, out, fs=None):
    fig, axs = plt.subplots(1, 2, figsize=figsize(fs, SLOT_PANEL), gridspec_kw={"wspace": 0.5})
    specs = [("proteomics_enrichments", "male", "female", "Sex (proteomics)", ("male only", "female only")),
             ("neuropathic_pain_enrichments", "pain_upregulated", "no_pain_upregulated", "Pain state",
              ("pain only", "no-pain only"))]
    for ax, (col, la, lb, title, names) in zip(axs, specs):
        a, b, both = _direction_counts(df, col, la, lb); tot = a + b + both; vals = [a, b, both]
        bars = ax.bar([0, 1, 2], vals, color=[C.BLUE, C.BLUE_MID, "none"], edgecolor=["none", "none", C.GREY], width=0.7)
        bars[2].set_hatch("////"); bars[2].set_linewidth(0.8)
        for i, v in enumerate(vals):
            ax.text(i, v + tot * 0.02, f"{v:,}", ha="center", fontsize=FS["annot"], color=C.INK)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels([names[0], names[1], "both"], rotation=30, ha="right")
        ax.set_ylim(0, max(vals) * 1.22)
        ax.text(0.02, 0.97, f"both = {both / tot:.0%}", transform=ax.transAxes, ha="left", va="top",
                fontsize=FS["annot"], color=C.MUTED)
        panel_title(ax, title)
    axs[0].set_ylabel("genes")
    caption(fig, "Hatched = gene sits in clusters of both directions.", y=-0.13)
    return save(fig, out)


def fig_celltype(df, out, top=12, fs=None):
    cnt = collections.Counter()
    for v in df["spatial_enrichments"].dropna():
        for c in set(_list(v)): cnt[c] += 1
    items = cnt.most_common(top)[::-1]
    fig, ax = plt.subplots(figsize=figsize(fs, SLOT_PANEL))
    bars = ax.barh(range(len(items)), [v for _, v in items], color=C.BLUE, height=0.62)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([k.replace("_", " ") for k, _ in items])
    mx = max(v for _, v in items); ax.set_xlim(0, mx * 1.18); ax.set_xlabel("genes")
    bar_value_labels(ax, bars, [v for _, v in items])
    panel_title(ax, "Top spatial call per cluster")
    caption(fig, "Genes in >= 1 cluster whose top spatial call is this cell type (unadjusted p < 0.05).", y=-0.11)
    return save(fig, out)


def gene_stats(df):
    """Numbers quoted on the slides, printed so they can be checked."""
    n = len(df); seed = df["is_seed_any_analysis"].astype(bool)
    d = {"genes": n, "seeds": int(seed.sum()),
         ">=5 sources (all)": f"{(df.n_sources >= 5).mean():.1%}",
         ">=5 sources (seeds)": f"{(df.loc[seed, 'n_sources'] >= 5).mean():.1%}",
         ">=5 sources (neighbours)": f"{(df.loc[~seed, 'n_sources'] >= 5).mean():.1%}",
         "median clusters/gene": float(df["total_unique_clusters"].median()),
         "max clusters/gene": int(df["total_unique_clusters"].max())}
    for s in SOURCES: d[f"hit_{s}"] = int(df[f"hit_{s}"].sum())
    d["sex a,b,both"] = _direction_counts(df, "proteomics_enrichments", "male", "female")
    d["pain a,b,both"] = _direction_counts(df, "neuropathic_pain_enrichments", "pain_upregulated", "no_pain_upregulated")
    return d


# ---------------------------------------------------------- cluster-level grid
SUMMARY_FILES = {   # source -> (directory stem, file name, term column)
    "orphanet": ("orphanet_enrichr_analysis", "orphanet_enrichr_summarized_results.csv", "disease"),
    "omim": ("omim_ps_analysis", "omim_ps_summarized_results.csv", "phenotypic_series"),
    "phenotypes": ("phenotypes_analysis", "phenotype_summarized_results.csv", "top_phenotype"),
    "spatial": ("spatial_analysis", "spatial_transcriptomics_hidef_comparison_summarized_results.csv", "top_cell_type"),
    "proteomics": ("proteomics_analysis", "proteomics_summarized_results.csv", "higher_expression_gender"),
    "neuropathic_pain": ("neuropathic_pain_analysis", "neuropathic_pain_summarized_results.csv", "top_expression"),
}
GRID_ORDER = ["orphanet", "omim", "phenotypes", "spatial", "proteomics", "neuropathic_pain"]
TERM_FIX = {"male": "male-enriched", "female": "female-enriched",
            "pain_upregulated": "pain-upregulated", "no_pain_upregulated": "no-pain-upregulated"}


def _term(raw, source):
    t = TERM_FIX.get(str(raw), str(raw)).replace("_", " ")
    if source == "orphanet": t = re.sub(r"\s*ORPHA[:\s]\d+.*$", "", t, flags=re.I)
    return (t[:44] + "…") if len(t) > 45 else t.strip()


def load_summaries(output_dir, ppi_source, cluster_type="hidef_from_comparison"):
    import glob
    rows = []
    for s, (stem, fname, tcol) in SUMMARY_FILES.items():
        hits = glob.glob(os.path.join(output_dir, ppi_source, f"{stem}_{cluster_type}*", fname))
        if not hits:
            print(f"  [skip] {s}: no {fname}"); continue
        d = pd.read_csv(hits[0])
        if tcol not in d.columns or "adjusted_p_value" not in d.columns:
            print(f"  [skip] {s}: missing {tcol}/adjusted_p_value"); continue
        d = d.assign(source=s, term=[_term(x, s) for x in d[tcol]],
                     cluster=d["cluster_id"].astype(str).str.replace("HiDeF_Cluster_", "", regex=False),
                     q=d["adjusted_p_value"].astype(float))
        rows.append(d[["cluster", "source", "term", "q"]])
        print(f"  {s}: {len(d)} cluster results, {(d.q < 0.05).sum()} with q < 0.05")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


AT2SRC = {"phenotype": "phenotypes", "omim_ps": "omim", "orphanet_enrichr": "orphanet"}   # analysis_type -> source key


def load_table(path, tier=None):
    """Long table from build_cluster_enrichment_table.py -> (res, facets). tier = 1..N keeps one persistence tier."""
    t = pd.read_csv(path)
    if tier:
        t = t[t["tier_index"] == tier]
        print(f"  tier {tier}: {t['cluster_id'].nunique()} clusters")
    t["source"] = t["analysis_type"].map(lambda a: AT2SRC.get(a, a))
    t["term"] = [_term(x, s_) for x, s_ in zip(t["term"], t["source"])]
    res = t.rename(columns={"cluster_id": "cluster"})[["cluster", "source", "term", "q"]]
    fac = (t.groupby("cluster_id").agg(persistence=("persistence", "first")).reset_index().assign(n_cell_types=np.nan))
    return res, fac


def select_clusters(res, facets, min_sources=2):
    sig = res[res.q < 0.05]
    info = sig.groupby("cluster").agg(n_sources=("source", "nunique"), min_q=("q", "min")).reset_index()
    info = info[info.n_sources >= min_sources]
    if facets is not None:
        f = facets.assign(cluster=facets["cluster_id"].astype(str).str.replace("HiDeF_Cluster_", "", regex=False))
        info = info.merge(f[["cluster", "persistence", "n_cell_types"]], on="cluster", how="left")
        info = info.sort_values(["persistence", "n_sources", "min_q"], ascending=[False, False, True])
    else:
        info["persistence"] = np.nan; info["n_cell_types"] = np.nan
        info = info.sort_values(["n_sources", "min_q"], ascending=[False, True])
    return info.reset_index(drop=True)


def fig_lollipop_grid(res, info, page_clusters, out, terms_per_source=6):
    cl = list(page_clusters["cluster"])
    sub = res[res.cluster.isin(cl)]
    rows = []   # (source, term)
    for s in GRID_ORDER:
        ss = sub[(sub.source == s) & (sub.q < 0.05)]
        if ss.empty: continue
        ranked = (ss.groupby("term").agg(n=("cluster", "nunique"), mq=("q", "min"))
                    .sort_values(["n", "mq"], ascending=[False, True]).head(terms_per_source))
        rows += [(s, t) for t in ranked.index]
    if not rows:
        print("  no rows for this page"); return False
    ypos = {r: i for i, r in enumerate(rows[::-1])}
    xpos = {c: i for i, c in enumerate(cl)}
    W = max(5.5, 0.34 * len(cl) + 4.6); H = max(3.0, 0.26 * len(rows) + 1.9)
    fig, ax = plt.subplots(figsize=(W, H))
    by = sub.groupby(["source", "term", "cluster"]).q.min().reset_index()
    for _, r in by.iterrows():
        key = (r.source, r.term)
        if key not in ypos: continue
        x, y = xpos[r.cluster], ypos[key]
        if r.q < 0.05:
            ax.plot([-0.6, x], [y, y], color=C.GREY_LT, lw=0.8, zorder=1)
            ax.scatter(x, y, s=18 + 14 * min(-math.log10(max(r.q, 1e-30)), 15) ** 0.9, color=C.ORANGE, zorder=3, edgecolor="white", linewidth=0.5)
        else:
            ax.scatter(x, y, s=14, color=PALE, zorder=2, edgecolor="none")
    ax.set_xticks(range(len(cl)))
    labs = []
    for _, r in page_clusters.iterrows():
        lab = r.cluster.replace("Cluster", "C")
        if not np.isnan(r.persistence): lab += f"\npers {r.persistence:.0f}"
        labs.append(lab)
    ax.set_xticklabels(labs, fontsize=FS["tick"], rotation=90, ha="center")
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels([t for (_, t) in ypos.keys()], fontsize=FS["tick"])
    ax.set_xlim(-0.8, len(cl) - 0.2); ax.set_ylim(-0.7, len(rows) - 0.3)
    # source brackets on the right
    start = {}
    for (s, t), y in ypos.items(): start.setdefault(s, []).append(y)
    for s, ys in start.items():
        ax.annotate("", xy=(len(cl) - 0.1, min(ys) - 0.3), xytext=(len(cl) - 0.1, max(ys) + 0.3),
                    xycoords="data", annotation_clip=False, arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.6))
        ax.text(len(cl) + 0.15, np.mean(ys), LABEL[s], color=C.BLUE, fontsize=FS["label"], fontweight="bold", va="center")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, label="BH q < 0.05 (size = -log10 q)"),
                        Line2D([], [], marker="o", ls="", color=PALE, markersize=4, label="tested, not significant")],
               frameon=False, fontsize=FS["tick"], loc="lower left", ncol=2)
    fig.subplots_adjust(left=0.30 if W < 9 else 0.24, right=1 - 1.7 / W, top=0.9, bottom=1.25 / H)
    save(fig, out)
    return True


def run_grid(args):
    os.makedirs(args.out, exist_ok=True)
    if args.table:
        res, facets = load_table(args.table, args.tier)
    else:
        res = load_summaries(args.output_dir, args.ppi_source, args.cluster_type)
        facets = pd.read_csv(args.facets) if args.facets and os.path.exists(args.facets) else None
    if res.empty: raise SystemExit("no enrichment results found")
    if facets is None: print("  no facets file: clusters ranked by number of significant sources, not persistence")
    info = select_clusters(res, facets, args.min_sources)
    n_pages = max(1, math.ceil(len(info) / args.max_clusters))
    print(f"{len(info)} clusters with q<0.05 in >= {args.min_sources} sources -> {n_pages} page(s) of <= {args.max_clusters}")
    pages = [args.page] if args.page else range(1, n_pages + 1)
    for p in pages:
        chunk = info.iloc[(p - 1) * args.max_clusters: p * args.max_clusters]
        tag = f"_tier{args.tier}" if args.tier else ""
        path = os.path.join(args.out, f"fig33a_lollipop_grid{tag}_page{p}of{n_pages}.png")
        if fig_lollipop_grid(res, info, chunk, path, args.terms_per_source): print(f"  saved {path}")


def run_gene(args):
    os.makedirs(args.out, exist_ok=True)
    df = load_tsv(args.tsv)
    for k, v in gene_stats(df).items(): print(f"  {k}: {v}")
    fs = args.figsize
    fig_saturation(df, os.path.join(args.out, "fig31a_source_saturation.png"), fs)
    fig_direction(df, os.path.join(args.out, "fig32a_direction.png"), fs)
    fig_celltype(df, os.path.join(args.out, "fig32b_celltype.png"), fs)
    print("saved fig31a, fig32a, fig32b ->", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gene"); g.add_argument("--tsv", required=True); g.add_argument("--out", default="figs")
    g.add_argument("--figsize", nargs=2, type=float, default=None, help="override canvas W H (inches) for all gene figures")
    c = sub.add_parser("grid")
    c.add_argument("--output-dir", default="./output"); c.add_argument("--ppi-source", default="combined26_multiplex")
    c.add_argument("--cluster-type", default="hidef_from_comparison"); c.add_argument("--facets", default=None)
    c.add_argument("--out", default="figs"); c.add_argument("--max-clusters", type=int, default=20)
    c.add_argument("--min-sources", type=int, default=2); c.add_argument("--terms-per-source", type=int, default=6)
    c.add_argument("--table", default=None, help="cluster_enrichment_long.csv from build_cluster_enrichment_table.py (preferred input)")
    c.add_argument("--tier", type=int, default=None, help="with --table: only this persistence tier (1 = highest)")
    c.add_argument("--page", type=int, default=None, help="render only this page (1-based)")
    a = ap.parse_args()
    run_gene(a) if a.mode == "gene" else run_grid(a)