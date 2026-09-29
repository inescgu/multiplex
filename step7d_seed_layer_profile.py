#!/usr/bin/env python3
"""
Step 7d: seed genes across the raw per-cell-type PPI layers -- presence,
degree, hubness and marker strength.

Runs as soon as step 1 has written multiplex/<ct>/ppi_layer.tsv and
celltype_gene_sets.json (before MultiXrank / disease module exist). Does NOT
read disease_module.csv; for the module cross-reference use step6b.

What it does
------------
PROFILE (as before; needs no extra inputs)
  For every seed gene x every layer: is it in the layer, and its degree there.

HUBNESS (new; no extra inputs)
  Is the seeds' degree higher than the other genes' in the same layer?
  Mann-Whitney U (one-sided, seeds > others), effect size = AUC
  (P[random seed > random non-seed]; 0.5 = no difference), plus a binomial
  test on how many seeds fall in the top --hub-frac by degree. Also run on the
  full global PPI layer ('global_ppi', from step 1 --base-ppi) if present.

LAYER ENRICHMENT (new; no extra inputs)
  Are seeds over-represented among a cell type's marker genes relative to all
  genes in global_ppi? (Fisher exact, one-sided.)

MARKER STRENGTH (new; only with --markers-csv)
  Among a cell type's markers (those in the PPI), do seeds have higher
  avg_log2FC / higher (pct.1 - pct.2) than non-seed markers? Seed-level
  percentile ranks are reported too. Ranking uses effect sizes, not the
  Wilcoxon p-values: with thousands of cells nearly every marker has
  p_val_adj ~ 0 (underflow), so p-values carry no ranking information.

Changes vs. the previous 7d (read before comparing old and new numbers)
-----------------------------------------------------------------------
* global_ppi is reported separately (global_degree, global_degree_pctile).
  Previously it was counted as one more layer, so it inflated
  n_layers_present / total_degree_across_layers and was almost always the
  "dominant layer".  Those three columns now cover the cell-type layers only.
* n_layers_present now means "gene is in the layer's gene set"; the new
  n_layers_connected means "... and has >= 1 edge there". Before, the summary
  counted only connected layers while the presence table counted gene-set
  membership, so the two files could disagree.
* Degree counts each undirected edge once (duplicates dropped).
* Genes with no in-layer edges are absent from ppi_layer.tsv (step 1 writes
  edges only) but present in celltype_gene_sets.json. They count here as
  degree 0 so the hubness background isn't biased upward; how many there are
  is printed and saved as n_isolated.

Caveats
-------
* Study bias: seeds are typically well-studied genes, and well-studied genes
  have more known PPIs. "Seeds are hubs" may partly be literature bias; the
  global_ppi result shares that bias, so read layer results against it.
* Layers share genes, so pooled numbers are descriptive, not independent
  tests. Per-layer p-values are BH-corrected across layers.
* Layers are induced subgraphs on marker genes, so layer degree = number of
  marker neighbours, not full-PPI degree.
* Layers with fewer than --min-seeds seeds are profiled but not tested.

Outputs (under --output-dir)
----------------------------
  seed_layer_presence.csv    long: gene, cell_type, is_global, present, degree,
                             degree_pctile, is_hub, [avg_log2FC, pct.1, pct.2,
                             pct_diff, fc_pctile, fc_rank]  (marker cols need
                             --markers-csv)
  seed_layer_summary.csv     one row per seed gene (previous columns kept, more added)
  layer_seed_tests.csv       one row per layer: tests, effect sizes, BH q-values
  seed_hubness_summary.txt   human-readable summary
  seed_vs_background.png     per-layer AUC bars              (unless --no-plots)
  seed_percentile_ecdf.png   pooled seed percentiles vs null (unless --no-plots)

Usage (existing call still works):
  python step7d_seed_layer_profile.py \
      --wdir 04_multixrank_prep/multixrank_hetero_input \
      --output-dir 07d_seed_layer_profile \
      [--markers-csv ../R/convert_hdrg_human_2025/cell_type_markers.csv] \
      [--seed-csv 00_inputs/seed_genes.csv]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEP = "::"
MARKER_COLS = ["avg_log2FC", "pct.1", "pct.2", "pct_diff", "fc_pctile", "fc_rank"]

# Optional layer ordering/coloring for the heatmap. Cell types not listed here
# fall into "Unclassified" and are drawn at the end, alphabetically -- this is
# purely cosmetic, nothing downstream depends on a cell type being in here.
DRG_HIERARCHY = {
    "A-fiber": {"color": "#1f77b4", "cell_types": [
        "A-Propr.EPHA3", "A-Propr.HAPLN4", "A-Propr.PCDH8",
        "Ab-LTMR.CCKAR", "Ab-LTMR.ETV1", "Ab-LTMR.LGI2", "Ab-LTMR.NSG2",
        "A-LTMR.TAC3",
        "A-PEP.CHRNA7_SLC18A3", "A-PEP.KIT", "A-PEP.NTRK3_S100A16", "A-PEP.SCGN_ADRA2C",
    ]},
    "C-fiber": {"color": "#ff7f0e", "cell_types": [
        "C-LTMR.CDH9",
        "C-PEP.ADORA2B", "C-PEP.TAC1_CACNG5", "C-PEP.TAC1_CHRNA3",
        "C-NP.MRGPRX1_GFRA2", "C-NP.MRGPRX1_MRGPRX4", "C-NP.SST",
        "C-Thermo.RXFP1", "C-Thermo.TRPM8",
    ]},
    "ATF3": {"color": "#d62728", "cell_types": ["ATF3"]},
    "Non-neuronal": {"color": "#2ca02c", "cell_types": [
        "B cell", "CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell",
        "macrophage", "mast cell", "microglial cell", "natural killer cell",
        "neutrophil", "Schwann cell", "neuron",
    ]},
}



# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    """Same transform as step 1 / step 2."""
    return name.replace(" ", "_").replace("/", "-").replace(",", "")


def marker_ct_to_layer_name(ct: str) -> str:
    """Raw Atlas_annotation string -> multiplex folder name. The layer builder
    wrote pkl stems with '/' -> '_' before step 1's sanitize_name() ran."""
    return sanitize_name(str(ct).replace("/", "_"))


def _gene_of(node: str) -> str:
    """Bare upper-case gene symbol from a plain or '<celltype>::GENE' string."""
    s = str(node).strip()
    return (s.rsplit(SEP, 1)[-1] if SEP in s else s).strip().upper()


_CT_TO_MAJOR = {sanitize_name(ct): major for major, d in DRG_HIERARCHY.items() for ct in d["cell_types"]}
_MAJOR_COLOR = {major: d["color"] for major, d in DRG_HIERARCHY.items()}
_MAJOR_COLOR["Unclassified"] = "#999999"


def order_layers(layer_names: list[str], global_name: str) -> tuple[list[str], dict]:
    """Cell types grouped by DRG major category (order above), unmapped ones
    bucketed 'Unclassified' and sorted alphabetically at the end; global_ppi
    (if present) always last of all. Returns (ordered_names, name->color)."""
    real = [c for c in layer_names if c != global_name]
    majors = {c: _CT_TO_MAJOR.get(c, "Unclassified") for c in real}
    order_key = list(DRG_HIERARCHY.keys()) + ["Unclassified"]
    ordered = sorted(real, key=lambda c: (order_key.index(majors[c]), c))
    colors = {c: _MAJOR_COLOR[majors[c]] for c in ordered}
    if global_name in layer_names:
        ordered.append(global_name)
        colors[global_name] = "#555555"
    return ordered, colors


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_seeds(wdir: Path, seed_csv: str | None) -> list[str]:
    if seed_csv:
        df = pd.read_csv(seed_csv, encoding="utf-8-sig")
        cols = {c.strip().lower(): c for c in df.columns}
        col = df[cols["gene"]] if "gene" in cols else df.iloc[:, 0]
        genes = {_gene_of(g.strip("'\"")) for g in col.dropna().astype(str)}
        src = seed_csv
    else:
        src, genes = None, set()
        for name in ("seeds_original.txt", "seeds_plain_from_step3.txt", "seeds.txt"):
            p = wdir / "seeds" / name
            if p.exists():
                genes = {_gene_of(line) for line in open(p) if line.strip()}
                src = p
                break
        if src is None:
            raise SystemExit(f"ERROR: no seed file under {wdir / 'seeds'} -- pass --seed-csv instead.")
    genes -= {"", "NAN"}
    print(f"Loaded {len(genes)} unique seed genes from {src}")
    return sorted(genes)


def load_module_neighbors(disease_module_csv: str) -> tuple[list[str], dict[str, set]]:
    """Non-seed genes the disease-module builder (step 6) actually recruited
    to bridge/connect the seeds -- i.e. everything in disease_module.csv with
    is_seed == False. Returns (unique gene list, gene -> set of cell types it
    was recruited into, parsed from the module's own 'cell_type' column).
    Excluding is_excluded_seed==True rows if present (those are seeds you
    told step 6 to exclude, not neighbors it chose)."""
    df = pd.read_csv(disease_module_csv)
    if "is_seed" not in df.columns:
        raise SystemExit(f"ERROR: {disease_module_csv} has no 'is_seed' column -- is this really disease_module.csv?")
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    neigh = df[~df["is_seed"].astype(bool)].copy()
    layers_by_gene: dict[str, set] = {}
    for g, ct in zip(neigh["gene"], neigh.get("cell_type", "")):
        if pd.notna(ct) and str(ct).strip() and str(ct).strip().lower() != "unknown":
            layers_by_gene.setdefault(g, set()).add(str(ct).strip())
    genes = sorted(set(neigh["gene"]) - {"", "NAN"})
    print(f"Loaded {len(genes)} unique non-seed genes recruited by the disease module from {disease_module_csv}")
    return genes, layers_by_gene


def load_gene_sets(path) -> dict[str, set]:
    with open(path) as f:
        raw = json.load(f)
    return {ct: {_gene_of(g) for g in genes} for ct, genes in raw.items()}


def find_layer_file(wdir: Path, ct: str) -> Path | None:
    d = wdir / "multiplex" / ct
    return next((p for p in (d / "ppi_layer.tsv", d / "ppi_layer.tsv.orig") if p.exists()), None)


def layer_degrees(edge_path: Path, node_universe: set) -> pd.Series:
    """Degree of every gene in node_universe (0 if it has no edges in the tsv).
    Each undirected edge counted once; self-loops (step 1's placeholder) ignored."""
    df = pd.read_csv(edge_path, sep="\t", header=None, names=["a", "b"],
                     dtype=str, on_bad_lines="skip").dropna()
    a = df["a"].map(_gene_of).to_numpy(dtype=object)
    b = df["b"].map(_gene_of).to_numpy(dtype=object)
    keep = a != b
    a, b = a[keep], b[keep]
    swap = a > b
    pairs = pd.DataFrame({"lo": np.where(swap, b, a), "hi": np.where(swap, a, b)}).drop_duplicates()
    deg = pd.concat([pairs["lo"], pairs["hi"]]).value_counts()
    universe = sorted(node_universe | set(deg.index))
    return deg.reindex(universe).fillna(0).astype(int)


def load_markers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    df["layer"] = df["cell_type"].map(marker_ct_to_layer_name)
    df["pct_diff"] = df["pct.1"] - df["pct.2"] if {"pct.1", "pct.2"} <= set(df.columns) else np.nan
    return (df.sort_values("avg_log2FC", ascending=False)
              .drop_duplicates(subset=["layer", "gene"], keep="first"))


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def mwu_auc(x, y):
    """One-sided (x > y) Mann-Whitney. Returns (AUC, p)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    try:
        res = stats.mannwhitneyu(x, y, alternative="greater")
        return float(res.statistic / (len(x) * len(y))), float(res.pvalue)
    except ValueError:
        return np.nan, np.nan


def binom_greater(k, n, p0):
    if n == 0 or not (0 < p0 < 1):
        return np.nan
    try:
        return float(stats.binomtest(k, n, p0, alternative="greater").pvalue)
    except AttributeError:  # old scipy
        return float(stats.binom_test(k, n, p0, alternative="greater"))


def bh(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = np.minimum.accumulate((pv[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1.0)
    out[ok] = q
    return out


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------
def analyse_layer(ct, deg, seeds, universe, markers_ct, hub_frac, min_seeds, is_global):
    """Returns (layer_row dict, DataFrame of the seed genes in this layer)."""
    frame = pd.DataFrame({"gene": deg.index, "degree": deg.values})
    frame["degree_pctile"] = frame["degree"].rank(pct=True)  # 1.0 = highest degree
    thr = np.quantile(frame["degree"].values, 1 - hub_frac)
    frame["is_hub"] = (frame["degree"] >= thr) & (frame["degree"] > 0)

    if markers_ct is not None and len(markers_ct):
        m = markers_ct[markers_ct["gene"].isin(frame["gene"])].copy()
        m["fc_pctile"] = m["avg_log2FC"].rank(pct=True)                     # 1.0 = strongest
        m["fc_rank"] = m["avg_log2FC"].rank(ascending=False, method="min")  # 1 = strongest
        frame = frame.merge(m[["gene"] + MARKER_COLS], on="gene", how="left")
    else:
        for c in MARKER_COLS:
            frame[c] = np.nan

    frame["is_seed"] = frame["gene"].isin(seeds)
    s, bg = frame[frame["is_seed"]], frame[~frame["is_seed"]]

    row = {"layer": ct, "is_global": is_global, "n_genes": len(frame),
           "n_isolated": int((frame["degree"] == 0).sum()), "n_seeds": len(s)}

    # enrichment of seeds among this layer's genes (meaningless for global)
    seeds_u = seeds & universe
    if not is_global and len(seeds_u):
        L = set(frame["gene"]) & universe
        a = len(seeds_u & L)
        b = len(L) - a
        c = len(seeds_u) - a
        d = len(universe) - len(L) - c
        orat, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        row.update(enrich_odds_ratio=orat, enrich_p=p)
    else:
        row.update(enrich_odds_ratio=np.nan, enrich_p=np.nan)

    testable = len(s) >= min_seeds
    nan2 = (np.nan, np.nan)

    row["seed_median_degree"] = float(s["degree"].median()) if len(s) else np.nan
    row["bg_median_degree"] = float(bg["degree"].median()) if len(bg) else np.nan
    row["degree_auc"], row["degree_p"] = mwu_auc(s["degree"], bg["degree"]) if testable else nan2
    p0 = frame["is_hub"].mean() if len(frame) else np.nan
    k = int(s["is_hub"].sum())
    row.update(n_seed_hubs=k, expected_seed_hubs=(p0 * len(s)) if len(s) else np.nan,
               hub_p=binom_greater(k, len(s), p0) if testable else np.nan)

    row["seed_median_log2fc"] = float(s["avg_log2FC"].median()) if len(s) else np.nan
    row["bg_median_log2fc"] = float(bg["avg_log2FC"].median()) if len(bg) else np.nan
    row["log2fc_auc"], row["log2fc_p"] = mwu_auc(s["avg_log2FC"], bg["avg_log2FC"]) if testable else nan2
    row["pctdiff_auc"], row["pctdiff_p"] = mwu_auc(s["pct_diff"], bg["pct_diff"]) if testable else nan2

    seed_tbl = s.drop(columns=["is_seed"]).copy()
    seed_tbl.insert(0, "layer", ct)
    return row, seed_tbl


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(layer_df, seed_ng, out, have_markers):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping plots")
        return

    d = layer_df[~layer_df["is_global"]].dropna(subset=["degree_auc"]).sort_values("degree_auc")
    if len(d):
        panels = [("degree_auc", "degree_q", "Seeds vs others: degree")]
        if have_markers:
            panels.append(("log2fc_auc", "log2fc_q", "Seeds vs other markers: avg_log2FC"))
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels) + 1, max(4, 0.28 * len(d) + 1.5)),
                                 sharey=True, squeeze=False)
        for ax, (auc, q, title) in zip(axes[0], panels):
            ax.barh(range(len(d)), d[auc].fillna(0.5).values - 0.5, left=0.5,
                    color=["#c44e52" if (not np.isnan(qq) and qq < 0.05) else "#8c8c8c" for qq in d[q]])
            ax.axvline(0.5, color="black", lw=0.8)
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels(d["layer"], fontsize=8)
            ax.set_xlabel("AUC (0.5 = no difference; red = BH q < 0.05)")
            ax.set_title(title, fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out / "seed_vs_background.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out / 'seed_vs_background.png'}")

    if len(seed_ng):
        cols = [("degree_pctile", "Degree percentile in layer")]
        if have_markers:
            cols.append(("fc_pctile", "avg_log2FC percentile among layer markers"))
        fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.5), squeeze=False)
        for ax, (col, title) in zip(axes[0], cols):
            x = np.sort(seed_ng[col].dropna().values)
            if len(x):
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color="#c44e52", label="seeds")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="null (all genes)")
            ax.set_xlabel(title + "  (1 = highest)")
            ax.set_ylabel("Cumulative fraction of seed-layer entries")
            ax.set_title("Below the diagonal = seeds skew high", fontsize=9)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "seed_percentile_ecdf.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out / 'seed_percentile_ecdf.png'}")


def make_heatmap(pres_all: pd.DataFrame, summ: pd.DataFrame, out: Path, global_name: str,
                  value_col: str, top_n: int) -> bool:
    """Seed x layer heatmap. Rows = top-N seeds by consistency_score (so the
    genes that score high across MULTIPLE layers/namespaces float to the
    top, not just single-layer standouts); columns = layers, DRG-ordered
    when recognised. Cell = value_col (fc_pctile if markers were given, else
    degree_pctile); grey = seed not present in that layer at all."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping heatmap")
        return False

    top_genes = summ.head(top_n)["gene"].tolist()
    if not top_genes:
        print("  No seeds to plot -- skipping heatmap")
        return False

    layers, colors = order_layers(sorted(pres_all["cell_type"].unique()), global_name)
    mat = (pres_all[pres_all["gene"].isin(top_genes)]
           .pivot_table(index="gene", columns="cell_type", values=value_col, aggfunc="first")
           .reindex(index=top_genes, columns=layers))

    fig_w = max(6, 0.42 * len(layers) + 3)
    fig_h = max(4, 0.24 * len(top_genes) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(mat.values)
    cmap = plt.get_cmap("rocket_r") if "rocket_r" in plt.colormaps() else plt.get_cmap("YlOrRd")
    cmap = matplotlib.colormaps.get_cmap("YlOrRd").copy()
    cmap.set_bad("#e8e8e8")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=7)
    for tick, c in zip(ax.get_xticklabels(), [colors[l] for l in layers]):
        tick.set_color(c)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=7)
    label = "avg_log2FC percentile (marker strength)" if value_col == "fc_pctile" else "degree percentile"
    fig.colorbar(im, ax=ax, shrink=0.7, label=label + "  (grey = not a marker gene in that layer)")
    ax.set_title(f"Top {len(top_genes)} seeds by cross-layer consistency", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fp = out / "seed_layer_heatmap.png"
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {fp}")
    return True


def write_seed_reports(pres_all: pd.DataFrame, summ: pd.DataFrame, tests: pd.DataFrame,
                        out: Path, global_name: str, hub_frac: float, have_markers: bool, top_n: int,
                        module_layers_by_gene: dict[str, set] | None = None):
    """One structured, human-readable block per top seed: which layers it's
    in, whether it's a hub/marker there, and whether that layer itself
    passed enrichment -- i.e. everything you need to individually judge a
    gene before going to look up its disease associations."""
    enrich_ok = set(tests.loc[(~tests["is_global"]) & (tests["enrich_q"] < 0.05), "layer"])
    lines = ["=" * 78, "PER-SEED REPORTS (top genes by cross-layer consistency)", "=" * 78,
             "Read each block top to bottom: global role, then per-layer detail.",
             "'enriched layer' = seeds as a group are significantly over-represented",
             "among that cell type's markers (enrich_q<0.05, from layer_seed_tests.csv).",
             ""]

    for _, srow in summ.head(top_n).iterrows():
        gene = srow["gene"]
        rows = pres_all[(pres_all["gene"] == gene) & pres_all["present"]].sort_values(
            "fc_pctile" if have_markers else "degree_pctile", ascending=False)
        lines.append(f"[{gene}]  --  {srow.get('specificity_tier', 'n/a')}")
        gdp = srow.get("global_degree_pctile", np.nan)
        if pd.notna(gdp):
            lines.append(f"  Global PPI degree percentile : {gdp:.2f}"
                         + ("  <- generic hub; layer-specific reads below matter less" if gdp >= 0.90 else ""))
        lines.append(f"  Present in {int(srow['n_layers_present'])} cell-type layer(s), "
                     f"connected in {int(srow['n_layers_connected'])}, "
                     f"'strong' (well-connected + top marker) in {int(srow['n_layers_strong'])}")
        if module_layers_by_gene:
            ml = sorted(module_layers_by_gene.get(gene, set()))
            lines.append(f"  Actually RECRUITED into the disease module under: {', '.join(ml) if ml else '(none -- unexpected, check disease_module.csv)'}")
        if not len(rows):
            lines.append("  Not a marker/node in any cell-type layer (seed acts only via global_ppi, if present)")
        for _, r in rows.iterrows():
            ct = r["cell_type"]
            tag = "ENRICHED LAYER" if ct in enrich_ok else "layer"
            bits = [f"deg={int(r['degree'])} (p{r['degree_pctile']:.2f})",
                    "HUB" if r["is_hub"] else ""]
            if have_markers and pd.notna(r.get("fc_pctile")):
                bits.append(f"log2FC={r['avg_log2FC']:.2f} (p{r['fc_pctile']:.2f}, "
                            f"rank {int(r['fc_rank'])}) pct1-2={r['pct_diff']:.2f}")
            lines.append(f"    [{tag:14s}] {ct:<30} " + " | ".join(b for b in bits if b))
        lines.append("")

    text = "\n".join(lines)
    (out / "seed_individual_reports.txt").write_text(text)
    print(f"  Saved -> {out / 'seed_individual_reports.txt'}")


def enriched_layers(tests: pd.DataFrame) -> set:
    return set(tests.loc[(~tests["is_global"]) & (tests["enrich_q"] < 0.05), "layer"])


def categorize_seeds(pres_all: pd.DataFrame, summ: pd.DataFrame, tests: pd.DataFrame,
                      have_markers: bool) -> pd.Series:
    """One tier per seed gene -- a decision tree over the same columns you'd
    read by eye, in the same priority order described in the docs below.
    Orthogonal to consistency_score/n_layers_strong (that's the "recurs
    across layers" axis; this is the "specific vs. generic" axis) -- read
    both together, don't collapse them into one label.
    """
    enrich_ok = enriched_layers(tests)
    p = pres_all.copy()
    p["in_enriched"] = p["cell_type"].isin(enrich_ok)
    if have_markers:
        p["is_specific"] = (p["fc_pctile"] >= 0.90) & (p["pct_diff"].fillna(0) >= 0.30)
    else:
        p["is_specific"] = False  # can't assess specificity without marker stats

    g = p.groupby("gene")
    flags = pd.DataFrame({
        "has_enriched_strong": g.apply(lambda d: bool((d["in_enriched"] & d["is_hub"] & d["is_specific"]).any())),
        "has_enriched_specific": g.apply(lambda d: bool((d["in_enriched"] & d["is_specific"]).any())),
        "has_enriched_hub": g.apply(lambda d: bool((d["in_enriched"] & d["is_hub"]).any())),
        "any_enriched": g["in_enriched"].any(),
    })
    flags = flags.reindex(summ["gene"]).fillna(False)
    is_global_hub = summ.set_index("gene")["global_degree_pctile"].reindex(flags.index) >= 0.90
    n_present = summ.set_index("gene")["n_layers_present"].reindex(flags.index).fillna(0)

    tier = pd.Series("F: present, unremarkable", index=flags.index)
    tier[n_present == 0] = "E: absent from all cell-type layers (global_ppi only)"
    if have_markers:
        m = flags["has_enriched_strong"]
        tier[m & ~is_global_hub.fillna(False)] = "A: specific marker + well-connected in an enriched layer"
        tier[m & is_global_hub.fillna(False)] = "A*: specific + connected, but also a generic global hub"
        tier[~m & flags["has_enriched_specific"]] = "B: specific marker, not well-connected there"
        tier[~m & ~flags["has_enriched_specific"] & flags["has_enriched_hub"]] = "C: hub in an enriched layer, not a distinct marker"
    else:
        tier[flags["has_enriched_hub"]] = "C: hub in an enriched layer (no marker data to assess specificity)"
    tier[(tier == "F: present, unremarkable") & is_global_hub.fillna(False) & ~flags["any_enriched"]] = \
        "D: generic hub, no enriched layer"
    return tier.reindex(summ["gene"]).values


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wdir", required=True, help="MultiXrank input dir (has multiplex/, celltype_gene_sets.json)")
    ap.add_argument("--celltype-gene-sets", default=None, help="Defaults to <wdir>/celltype_gene_sets.json")
    ap.add_argument("--seed-csv", default=None,
                    help="Read seeds from this CSV ('gene' column) instead of <wdir>/seeds/. "
                         "Ignored if --disease-module-csv is given (see --target-set).")
    ap.add_argument("--disease-module-csv", default=None,
                    help="Run this analysis on the disease module's NON-seed (bridging/neighbor) genes "
                         "instead of the seeds -- pass step6's disease_module.csv. Requires step 6 to "
                         "have run. Use a different --output-dir than your seeds run so files don't clash.")
    ap.add_argument("--markers-csv", default=None,
                    help="cell_type_markers.csv from the R script; enables marker-strength tests")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--global-ppi-name", default="global_ppi")
    ap.add_argument("--hub-frac", type=float, default=0.10,
                    help="Top fraction by degree / log2FC counted as 'hub' / 'top marker' (default 0.10)")
    ap.add_argument("--min-seeds", type=int, default=3, help="Min seeds in a layer to run tests (default 3)")
    ap.add_argument("--top-n-report", type=int, default=40,
                    help="How many top seeds (by consistency_score) to draw in the heatmap and write "
                         "individual reports for (default 40)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    wdir, out = Path(args.wdir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gene_sets = load_gene_sets(args.celltype_gene_sets or (wdir / "celltype_gene_sets.json"))
    module_layers_by_gene: dict[str, set] = {}
    if args.disease_module_csv:
        seeds, module_layers_by_gene = load_module_neighbors(args.disease_module_csv)
        print("[mode: module-neighbors] treating the module's non-seed genes as the target set below -- "
              "every 'seed' in this run's output files/text means 'module neighbor gene'.")
    else:
        seeds = load_seeds(wdir, args.seed_csv)
    seed_set = set(seeds)
    gname = args.global_ppi_name
    print(f"{len(gene_sets)} layers in celltype_gene_sets.json")

    if gname in gene_sets:
        universe = gene_sets[gname]
        print(f"Enrichment universe: '{gname}' ({len(universe):,} genes); "
              f"{len(seed_set & universe)}/{len(seeds)} seeds are in it")
    else:
        universe = set().union(*gene_sets.values())
        print(f"⚠ '{gname}' not found: using union of all layers as enrichment universe "
              f"(biased toward marker genes -- enrichment p-values are optimistic)")

    markers = None
    if args.markers_csv:
        markers = load_markers(args.markers_csv)
        real = set(gene_sets) - {gname}
        unmatched = sorted(set(markers["layer"]) - set(gene_sets))
        no_markers = sorted(real - set(markers["layer"]))
        print(f"Markers: {len(markers):,} rows, {markers['layer'].nunique()} cell types")
        if unmatched:
            print(f"  {len(unmatched)} marker cell type(s) without a layer (fine if you filtered layers): {unmatched[:8]}")
        if no_markers:
            print(f"  ⚠ {len(no_markers)} layer(s) with no matching marker rows (name mismatch?): {no_markers}")
    else:
        print("[info] --markers-csv not given: marker-strength tests skipped (hubness + enrichment still run).")

    # ---- per-layer degrees + tests ----
    layer_rows, seed_tables = [], []
    print("\nLayers:")
    for ct, genes in sorted(gene_sets.items()):
        p = find_layer_file(wdir, ct)
        if p is None:
            print(f"  ⚠ no ppi_layer file for '{ct}' -- skipped")
            continue
        is_global = ct == gname
        deg = layer_degrees(p, genes)
        mk = None if (markers is None or is_global) else markers[markers["layer"] == ct]
        row, st = analyse_layer(ct, deg, seed_set, universe, mk, args.hub_frac, args.min_seeds, is_global)
        layer_rows.append(row)
        seed_tables.append(st.assign(is_global=is_global))
        auc = f"degAUC={row['degree_auc']:.2f}" if not np.isnan(row["degree_auc"]) else "(too few seeds to test)"
        print(f"  {ct:<38} genes={row['n_genes']:>6,} (isolated {row['n_isolated']:>5,}) seeds={row['n_seeds']:>3} {auc}")

    layer_df = pd.DataFrame(layer_rows)
    is_ng = ~layer_df["is_global"]
    for p_col, q_col in [("degree_p", "degree_q"), ("hub_p", "hub_q"), ("enrich_p", "enrich_q"),
                         ("log2fc_p", "log2fc_q"), ("pctdiff_p", "pctdiff_q")]:
        layer_df[q_col] = np.nan
        layer_df.loc[is_ng, q_col] = bh(layer_df.loc[is_ng, p_col].values)
    layer_df.to_csv(out / "layer_seed_tests.csv", index=False)

    # ---- long presence table: every seed x every layer ----
    layer_names = [r["layer"] for r in layer_rows]
    long = pd.MultiIndex.from_product([seeds, layer_names], names=["gene", "cell_type"]).to_frame(index=False)
    seen = pd.concat(seed_tables, ignore_index=True).rename(columns={"layer": "cell_type"}).drop(columns=["is_global"])
    long = long.merge(seen, on=["gene", "cell_type"], how="left")
    long["present"] = long["degree"].notna()
    long["is_hub"] = long["is_hub"] == True  # noqa: E712  (NaN -> False)
    long["is_global"] = long["cell_type"] == gname
    if markers is None:
        long = long.drop(columns=MARKER_COLS)
    front = ["gene", "cell_type", "is_global", "present", "degree", "degree_pctile", "is_hub"]
    long = long[front + [c for c in long.columns if c not in front]].sort_values(["gene", "cell_type"])
    long.to_csv(out / "seed_layer_presence.csv", index=False)
    print(f"\nSaved -> {out / 'seed_layer_presence.csv'}")

    # ---- per-seed summary (cell-type layers only; global reported separately) ----
    pres = long[(~long["is_global"]) & long["present"]]
    conn = pres[pres["degree"] > 0]
    g = pres.groupby("gene")
    summ = pd.DataFrame(index=pd.Index(seeds, name="gene"))
    summ["n_layers_present"] = g.size().reindex(summ.index).fillna(0).astype(int)
    summ["n_layers_connected"] = conn.groupby("gene").size().reindex(summ.index).fillna(0).astype(int)
    summ["total_degree_across_layers"] = g["degree"].sum().reindex(summ.index).fillna(0).astype(int)
    dom = conn.loc[conn.groupby("gene")["degree"].idxmax()].set_index("gene") if len(conn) else pd.DataFrame(
        columns=["cell_type", "degree"])
    summ["dominant_layer"] = dom["cell_type"].reindex(summ.index)
    summ["dominant_layer_degree"] = dom["degree"].reindex(summ.index).fillna(0).astype(int)
    if gname in set(layer_names):
        gl = long[long["is_global"] & long["present"]].set_index("gene")
        summ["global_degree"] = gl["degree"].reindex(summ.index)
        summ["global_degree_pctile"] = gl["degree_pctile"].reindex(summ.index)
    summ["n_layers_hub"] = g["is_hub"].sum().reindex(summ.index).fillna(0).astype(int)
    summ["max_degree_pctile"] = g["degree_pctile"].max().reindex(summ.index)
    summ["median_degree_pctile"] = g["degree_pctile"].median().reindex(summ.index)
    if markers is not None:
        summ["n_layers_top_marker"] = (g["fc_pctile"].apply(lambda x: int((x >= 1 - args.hub_frac).sum()))
                                       .reindex(summ.index).fillna(0).astype(int))
        summ["best_fc_pctile"] = g["fc_pctile"].max().reindex(summ.index)
        summ["best_avg_log2FC"] = g["avg_log2FC"].max().reindex(summ.index)
        best = (pres.dropna(subset=["avg_log2FC"]).sort_values("avg_log2FC", ascending=False)
                .drop_duplicates("gene").set_index("gene")["cell_type"])
        summ["best_marker_layer"] = best.reindex(summ.index)

        # "Strong" in a layer = well-connected there AND a strong marker there.
        # This is the "same seed scoring high across several subtypes" signal:
        # a gene present under many cell-type prefixes that clears both bars
        # in more than one of them, rather than being a fluke in a single layer.
        pres_c = pres.copy()
        pres_c["strong"] = (pres_c["degree_pctile"] >= 0.75) & (pres_c["fc_pctile"] >= 1 - args.hub_frac)
        summ["n_layers_strong"] = pres_c.groupby("gene")["strong"].sum().reindex(summ.index).fillna(0).astype(int)
        summ["mean_fc_pctile"] = g["fc_pctile"].mean().reindex(summ.index)
        summ["consistency_score"] = (summ["mean_fc_pctile"].fillna(0) * summ["median_degree_pctile"].fillna(0)
                                     * np.sqrt(summ["n_layers_present"].clip(lower=1)))
    else:
        pres_c = pres.copy()
        pres_c["strong"] = pres_c["degree_pctile"] >= 0.90
        summ["n_layers_strong"] = pres_c.groupby("gene")["strong"].sum().reindex(summ.index).fillna(0).astype(int)
        summ["consistency_score"] = (summ["median_degree_pctile"].fillna(0)
                                     * np.sqrt(summ["n_layers_present"].clip(lower=1)))

    summ = summ.reset_index().sort_values("consistency_score", ascending=False)

    pres_all = long[~long["is_global"]].copy()
    summ["specificity_tier"] = categorize_seeds(pres_all, summ, layer_df, markers is not None)
    tier_order = ["A: specific marker + well-connected in an enriched layer",
                  "A*: specific + connected, but also a generic global hub",
                  "B: specific marker, not well-connected there",
                  "C: hub in an enriched layer, not a distinct marker",
                  "C: hub in an enriched layer (no marker data to assess specificity)",
                  "D: generic hub, no enriched layer",
                  "F: present, unremarkable",
                  "E: absent from all cell-type layers (global_ppi only)"]
    summ["specificity_tier"] = pd.Categorical(summ["specificity_tier"], categories=tier_order, ordered=True)
    if module_layers_by_gene:
        summ["n_module_layers"] = summ["gene"].map(lambda g: len(module_layers_by_gene.get(g, set())))
        summ["module_layers"] = summ["gene"].map(lambda g: "; ".join(sorted(module_layers_by_gene.get(g, set()))))
    summ = summ.sort_values(["specificity_tier", "consistency_score"], ascending=[True, False])
    summ.to_csv(out / "seed_layer_summary.csv", index=False)
    print(f"Saved -> {out / 'seed_layer_summary.csv'}")

    n_absent = int((summ["n_layers_present"] == 0).sum())
    print(f"\n{n_absent}/{len(seeds)} seed gene(s) absent from every cell-type layer entirely")
    if n_absent:
        print("  " + ", ".join(summ.loc[summ["n_layers_present"] == 0, "gene"].tolist()[:20]))

    # ---- text summary ----
    ng = layer_df[is_ng]
    testable = ng.dropna(subset=["degree_auc"])
    seed_ng = pd.concat([t for t in seed_tables if not t["is_global"].iloc[0]], ignore_index=True) \
        if any(not t["is_global"].iloc[0] for t in seed_tables) else pd.DataFrame()
    L = ["=" * 78, "SEED HUBNESS / MARKER-STRENGTH SUMMARY", "=" * 78,
         f"Seeds: {len(seeds)} | in enrichment universe: {len(seed_set & universe)} | "
         f"hub / top-marker cutoff: top {args.hub_frac:.0%}"]
    if layer_df["is_global"].any():
        gr = layer_df[layer_df["is_global"]].iloc[0]
        L += ["", f"GLOBAL PPI ('{gname}'): full-PPI hubness"]
        if np.isnan(gr["degree_auc"]):
            L.append(f"  too few seeds in the global PPI to test ({gr['n_seeds']})")
        else:
            L += [f"  seeds in PPI: {gr['n_seeds']} | median degree seeds {gr['seed_median_degree']:.1f} "
                  f"vs others {gr['bg_median_degree']:.1f}",
                  f"  degree AUC = {gr['degree_auc']:.3f} (p = {gr['degree_p']:.2e}) | seeds in top "
                  f"{args.hub_frac:.0%}: {gr['n_seed_hubs']} (expected {gr['expected_seed_hubs']:.1f}, p = {gr['hub_p']:.2e})"]
    L += ["", f"PER-LAYER (testable layers = {len(testable)} of {len(ng)}; BH across layers)",
          f"  Seeds higher degree than other genes        (q<0.05): {int((testable['degree_q'] < 0.05).sum())}",
          f"  Seed excess in top-degree bin               (q<0.05): {int((testable['hub_q'] < 0.05).sum())}",
          f"  Seeds enriched among the cell type's markers (q<0.05): {int((ng['enrich_q'] < 0.05).sum())}"]
    if markers is not None:
        L += [f"  Seeds higher avg_log2FC than other markers  (q<0.05): {int((testable['log2fc_q'] < 0.05).sum())}",
              f"  Seeds higher pct.1-pct.2 than other markers (q<0.05): {int((testable['pctdiff_q'] < 0.05).sum())}"]
    if len(seed_ng):
        L += ["", "POOLED seed-layer entries (descriptive; entries are not independent)",
              f"  n = {len(seed_ng)} | median degree percentile {seed_ng['degree_pctile'].median():.2f} (null 0.50) | "
              f"fraction in top {args.hub_frac:.0%}: {seed_ng['is_hub'].mean():.2f} (null ~{args.hub_frac:.2f})"]
        if markers is not None:
            L.append(f"  median avg_log2FC percentile among markers: {seed_ng['fc_pctile'].median():.2f} (null 0.50)")
    L += ["", "SEED CATEGORIZATION (specificity_tier, full breakdown in seed_layer_summary.csv)"]
    for t, n in summ["specificity_tier"].value_counts().reindex(tier_order).items():
        if pd.notna(n) and n > 0:
            L.append(f"  {t:<62} {int(n):>4}")
    L += ["", f"Isolated genes (degree 0) across layers: {int(layer_df['n_isolated'].sum()):,} "
              f"(in gene sets json but with no edges in ppi_layer.tsv)", "=" * 78]
    text = "\n".join(L)
    (out / "seed_hubness_summary.txt").write_text(text)
    print("\n" + text)

    # ---- heatmap + per-seed reports (use ALL layer rows, not just present-only) ----
    value_col = "fc_pctile" if markers is not None else "degree_pctile"
    write_seed_reports(pres_all, summ, layer_df, out, gname, args.hub_frac, markers is not None, args.top_n_report,
                       module_layers_by_gene)

    if not args.no_plots:
        print()
        make_plots(layer_df, seed_ng, out, markers is not None)
        make_heatmap(pres_all, summ, out, gname, value_col, args.top_n_report)
    print(f"\nAll outputs -> {out.resolve()}")


if __name__ == "__main__":
    main()