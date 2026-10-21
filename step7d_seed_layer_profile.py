#!/usr/bin/env python3
"""
Step 7d: profile a gene set across the raw per-cell-type PPI layers --
presence, degree, hubness and marker strength.

TWO MODES (same code, different target gene set)
------------------------------------------------
  SEED mode      (default)              target genes = your seed genes
  NEIGHBOR mode  (--disease-module-csv) target genes = the NON-seed genes the
                                        disease-module builder (step 6)
                                        recruited to connect the seeds.
All printouts, reports and plot labels use the right word ("seed" or
"neighbor") automatically. Column names inside the CSVs (seed_median_degree,
n_seeds, ...) are kept as-is; in neighbor mode read "seed" there as "target".

What's new in this version
--------------------------
1. NEIGHBOR mode defines "enriched layer" from your real SEEDS, not from the
   neighbors. Neighbors were recruited from these layers, so testing them for
   enrichment is circular (every layer comes out "enriched"). The specificity
   tiers (A/A*/B/C/D) now use the layers where the seeds themselves are
   enriched among the cell type's markers. If seeds can't be loaded, the
   script warns and falls back to the old (circular) behaviour.
   Seeds come from --seed-csv, or <wdir>/seeds/.
2. NEIGHBOR mode also profiles the real seeds in the same run (as a reference)
   and draws seed_vs_neighbor.png + writes reference_seed_layer_tests.csv.
   Each set is tested against a background that EXCLUDES the other set
   (disable with --keep-other-set-in-background).
3. DEGREE-MATCHED NULL for hubness. For every layer: do the targets have a
   higher in-layer degree than random genes with the SAME global-PPI degree?
   This controls for "popular genes have lots of known interactions"
   (literature/study bias, and for neighbors the fact that bridging genes are
   hubs). Empirical one-sided permutation p-value, BH-corrected across layers.
   Columns: matched_obs_mean_pctile, matched_null_mean_pctile, matched_z,
   matched_p, matched_q.

Existing analyses
-----------------
PROFILE      Every target gene x every layer: present? degree?
HUBNESS      Mann-Whitney U (targets > other genes in layer), AUC effect size
             (0.5 = no difference), binomial test on the top --hub-frac bin.
             Also on the full global PPI ('global_ppi') if present.
ENRICHMENT   Are targets over-represented among a cell type's markers vs. all
             genes in global_ppi? (Fisher exact, one-sided.)
MARKER       (--markers-csv) targets vs other markers: avg_log2FC, pct.1-pct.2.
STRENGTH     Effect sizes, not Wilcoxon p-values (those underflow to ~0).

Caveats
-------
* Study bias: well-studied genes have more known PPIs (see the matched test).
* Layers share genes, so pooled numbers are descriptive. Per-layer p-values
  are BH-corrected across layers.
* Layers are induced subgraphs on marker genes: layer degree = number of
  marker neighbours, not full-PPI degree.
* Layers with fewer than --min-seeds targets are profiled but not tested.
* The matched test matches on GLOBAL degree, so it is not run on global_ppi
  itself, and genes missing from global_ppi are dropped from it.

Outputs (under --output-dir)
----------------------------
  seed_layer_presence.csv, seed_layer_summary.csv, layer_seed_tests.csv,
  seed_hubness_summary.txt, seed_individual_reports.txt,
  reference_seed_layer_tests.csv   (neighbor mode only)
  Plots (unless --no-plots): seed_vs_background, seed_percentile_ecdf,
  seed_layer_heatmap, layer_enrichment, layer_percentile_boxplots,
  tier_breakdown, genes_per_layer_hist, degree_vs_marker_scatter,
  layer_auc_scatter, global_degree_hist, degree_matched_null,
  seed_vs_neighbor (neighbor mode only)   -- all .png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEP = "::"
MARKER_COLS = ["avg_log2FC", "pct.1", "pct.2", "pct_diff", "fc_pctile", "fc_rank", "is_sig_up"]

# Wording for the target gene set; switched in main() by set_terms().
T = {"s": "seed", "p": "seeds", "S": "Seed", "P": "Seeds", "U": "SEED"}


def set_terms(neighbor_mode: bool):
    if neighbor_mode:
        T.update(s="neighbor", p="neighbors", S="Neighbor", P="Neighbors", U="NEIGHBOR")
    else:
        T.update(s="seed", p="seeds", S="Seed", P="Seeds", U="SEED")


# Optional layer ordering/coloring. Cell types not listed here fall into
# "Unclassified" and are drawn at the end, alphabetically -- purely cosmetic.
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
    """Raw Atlas_annotation string -> multiplex folder name."""
    return sanitize_name(str(ct).replace("/", "_"))


def _gene_of(node: str) -> str:
    """Bare upper-case gene symbol from a plain or '<celltype>::GENE' string."""
    s = str(node).strip()
    return (s.rsplit(SEP, 1)[-1] if SEP in s else s).strip().upper()


_CT_TO_MAJOR = {sanitize_name(ct): major for major, d in DRG_HIERARCHY.items() for ct in d["cell_types"]}
_MAJOR_COLOR = {major: d["color"] for major, d in DRG_HIERARCHY.items()}
_MAJOR_COLOR["Unclassified"] = "#999999"


def order_layers(layer_names: list[str], global_name: str) -> tuple[list[str], dict]:
    """Cell types grouped by DRG major category, unmapped ones 'Unclassified'
    (alphabetical, at the end); global_ppi (if present) last of all."""
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
    """Non-seed genes the disease-module builder (step 6) recruited to
    bridge/connect the seeds (is_seed == False). Returns (unique gene list,
    gene -> set of cell types it was recruited into)."""
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
    print(f"Loaded {len(genes)} unique NEIGHBOR genes (non-seed genes recruited by the disease module) "
          f"from {disease_module_csv}")
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
    Each undirected edge counted once; self-loops ignored."""
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


def load_markers(path: str, alpha: float = 0.05, min_lfc: float = 0.25) -> pd.DataFrame:
    df = pd.read_csv(path)
    for src, dst in [("logFC", "avg_log2FC"), ("marker_log2FC", "avg_log2FC"),
                     ("FDR", "p_val_adj"), ("marker_FDR", "p_val_adj")]:
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    missing = {"gene", "cell_type", "avg_log2FC"} - set(df.columns)
    if missing:
        raise SystemExit(f"--markers-csv {path} is missing {missing}; expected marker statistics "
                         f"(e.g. pseudobulk_markers_all.csv). Found: {list(df.columns)}")
    df = df.dropna(subset=["avg_log2FC"]).copy()
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    df["layer"] = df["cell_type"].map(marker_ct_to_layer_name)
    for c in ("pct.1", "pct.2"):
        if c not in df.columns:
            df[c] = np.nan                      # pseudobulk has no per-cell percentages
    df["pct_diff"] = df["pct.1"] - df["pct.2"]
    df["is_sig_up"] = ((df["p_val_adj"] < alpha) & (df["avg_log2FC"] > min_lfc)
                       if "p_val_adj" in df.columns else df["avg_log2FC"] > min_lfc)
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


def matched_degree_test(s: pd.DataFrame, bg: pd.DataFrame, gdeg: pd.Series, edges: np.ndarray,
                        n_perm: int, rng) -> dict:
    """Do the targets have a higher in-layer degree percentile than random
    genes with the same GLOBAL-PPI degree?

    Genes are binned by global degree (quantile bins, shared across layers).
    For each bin containing targets, the same number of background genes is
    drawn from that bin (with replacement) n_perm times; the null statistic is
    the mean in-layer degree percentile of the drawn genes. One-sided
    empirical p = (1 + #null >= observed) / (1 + n_perm). Targets whose bin
    has no background genes (or that are not in the global PPI) are dropped.
    """
    nan = dict(matched_n=0, matched_obs_mean_pctile=np.nan, matched_null_mean_pctile=np.nan,
               matched_z=np.nan, matched_p=np.nan)
    if gdeg is None or not len(s) or not len(bg):
        return nan
    inner = edges[1:-1]
    s = s.assign(g=s["gene"].map(gdeg)).dropna(subset=["g"])
    bg = bg.assign(g=bg["gene"].map(gdeg)).dropna(subset=["g"])
    if not len(s) or not len(bg):
        return nan
    s_bin = np.digitize(s["g"].values, inner)
    bg_bin = np.digitize(bg["g"].values, inner)
    bg_pct = bg["degree_pctile"].values
    s_pct = s["degree_pctile"].values
    null_sum = np.zeros(n_perm)
    obs_sum, n_used = 0.0, 0
    for b in np.unique(s_bin):
        pool = bg_pct[bg_bin == b]
        sel = s_bin == b
        k = int(sel.sum())
        if len(pool) == 0:
            continue
        null_sum += pool[rng.integers(0, len(pool), size=(n_perm, k))].sum(axis=1)
        obs_sum += s_pct[sel].sum()
        n_used += k
    if n_used == 0:
        return nan
    obs = obs_sum / n_used
    null = null_sum / n_used
    sd = null.std()
    return dict(matched_n=n_used, matched_obs_mean_pctile=float(obs), matched_null_mean_pctile=float(null.mean()),
                matched_z=float((obs - null.mean()) / sd) if sd > 0 else np.nan,
                matched_p=float((1 + (null >= obs).sum()) / (1 + n_perm)))


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------
def analyse_layer(ct, deg, target, universe, markers_ct, hub_frac, min_seeds, is_global,
                  exclude, gdeg, edges, n_perm, rng):
    """Returns (layer_row dict, DataFrame of the target genes in this layer).
    'target' = the gene set under test (seeds OR module neighbors). Genes in
    'exclude' (the other set) are kept for percentiles but dropped from the
    background used in the tests."""
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

    frame["is_seed"] = frame["gene"].isin(target)
    s = frame[frame["is_seed"]]
    bg = frame[~frame["is_seed"] & ~frame["gene"].isin(exclude)]

    row = {"layer": ct, "is_global": is_global, "n_genes": len(frame),
           "n_isolated": int((frame["degree"] == 0).sum()), "n_seeds": len(s)}

    targets_u = target & universe
    if not is_global and len(targets_u):
        if markers_ct is not None and len(markers_ct) and "is_sig_up" in markers_ct.columns:
            # expression-based layers: test against the cell type's significant markers,
            # not against every expressed gene (which would make nearly every layer 'enriched')
            L = set(markers_ct.loc[markers_ct["is_sig_up"].astype(bool), "gene"]) & set(frame["gene"]) & universe
        else:
            L = set(frame["gene"]) & universe
        a = len(targets_u & L)
        b = len(L) - a
        c = len(targets_u) - a
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

    if testable and not is_global:
        row.update(matched_degree_test(s, bg, gdeg, edges, n_perm, rng))
    else:
        row.update(matched_n=0, matched_obs_mean_pctile=np.nan, matched_null_mean_pctile=np.nan,
                   matched_z=np.nan, matched_p=np.nan)

    seed_tbl = s.drop(columns=["is_seed"]).copy()
    seed_tbl.insert(0, "layer", ct)
    return row, seed_tbl


def run_layers(target: set, exclude: set, cache: dict, universe: set, gdeg, edges, args, rng, verbose: bool):
    """Run analyse_layer over every cached layer for one target gene set and
    BH-correct across the non-global layers."""
    rows, tables = [], []
    for ct in sorted(cache):
        deg, mk, is_global = cache[ct]
        row, st = analyse_layer(ct, deg, target, universe, mk, args.hub_frac, args.min_seeds, is_global,
                                exclude, gdeg, edges, args.n_perm, rng)
        rows.append(row)
        tables.append(st.assign(is_global=is_global))
        if verbose:
            auc = f"degAUC={row['degree_auc']:.2f}" if not np.isnan(row["degree_auc"]) else f"(too few {T['p']} to test)"
            print(f"  {ct:<38} genes={row['n_genes']:>6,} (isolated {row['n_isolated']:>5,}) "
                  f"{T['p']}={row['n_seeds']:>3} {auc}")
    df = pd.DataFrame(rows)
    ng = ~df["is_global"]
    for p_col, q_col in [("degree_p", "degree_q"), ("hub_p", "hub_q"), ("enrich_p", "enrich_q"),
                         ("log2fc_p", "log2fc_q"), ("pctdiff_p", "pctdiff_q"), ("matched_p", "matched_q")]:
        df[q_col] = np.nan
        df.loc[ng, q_col] = bh(df.loc[ng, p_col].values)
    return df, tables


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError:
        print("  matplotlib not available -- skipping plots")
        return None, None


def _save(fig, plt, path: Path):
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def make_plots(layer_df, seed_ng, out, have_markers):
    _, plt = _plt()
    if plt is None:
        return

    d = layer_df[~layer_df["is_global"]].dropna(subset=["degree_auc"]).sort_values("degree_auc")
    if len(d):
        panels = [("degree_auc", "degree_q", f"{T['P']} vs other genes: degree")]
        if have_markers:
            panels.append(("log2fc_auc", "log2fc_q", f"{T['P']} vs other markers: avg_log2FC"))
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels) + 1, max(4, 0.28 * len(d) + 1.5)),
                                 sharey=True, squeeze=False)
        for ax, (auc, q, title) in zip(axes[0], panels):
            ax.barh(range(len(d)), d[auc].fillna(0.5).values - 0.5, left=0.5,
                    color=["#c44e52" if (not np.isnan(qq) and qq < 0.05) else "#8c8c8c" for qq in d[q]])
            ax.axvline(0.5, color="black", lw=0.8)
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels(d["layer"], fontsize=8)
            ax.set_xlabel("AUC (0.5 = no difference; >0.5 = higher; red = BH q < 0.05)")
            ax.set_title(title, fontsize=10, fontweight="bold")
        _save(fig, plt, out / "seed_vs_background.png")

    if len(seed_ng):
        cols = [("degree_pctile", "Degree percentile in layer")]
        if have_markers:
            cols.append(("fc_pctile", "avg_log2FC percentile among layer markers"))
        fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.5), squeeze=False)
        for ax, (col, title) in zip(axes[0], cols):
            x = np.sort(seed_ng[col].dropna().values)
            if len(x):
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color="#c44e52", label=T["p"])
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="null (all genes)")
            ax.set_xlabel(title + "  (1 = highest)")
            ax.set_ylabel(f"Cumulative fraction of {T['s']}-layer entries")
            ax.set_title(f"Curve below the diagonal = {T['p']} skew high", fontsize=9)
            ax.legend(fontsize=8)
        _save(fig, plt, out / "seed_percentile_ecdf.png")


def plot_enrichment(df, out, noun: str, subtitle: str):
    """-log10(BH q) of 'this gene set is over-represented among the cell type's markers'."""
    _, plt = _plt()
    if plt is None:
        return
    d = df[~df["is_global"]].dropna(subset=["enrich_q"]).copy()
    if not len(d):
        return
    d["nlq"] = -np.log10(d["enrich_q"].clip(lower=1e-300))
    d = d.sort_values("nlq")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(d) + 1.5)))
    ax.barh(range(len(d)), d["nlq"], color=["#c44e52" if q < 0.05 else "#8c8c8c" for q in d["enrich_q"]])
    ax.axvline(-np.log10(0.05), color="black", ls="--", lw=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{l}  (n={n})" for l, n in zip(d["layer"], d["n_seeds"])], fontsize=8)
    ax.set_xlabel("-log10(BH q)   (dashed = q 0.05; red = significant)")
    ax.set_title(f"Are {noun} over-represented among each cell type's markers?\n{subtitle}",
                 fontsize=10, fontweight="bold")
    _save(fig, plt, out / "layer_enrichment.png")


def plot_matched(df, ref, out):
    """z-score of observed mean degree percentile vs the degree-matched null."""
    _, plt = _plt()
    if plt is None:
        return
    d = df[~df["is_global"]].dropna(subset=["matched_z"]).sort_values("matched_z")
    if not len(d):
        return
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(d) + 1.5)))
    ax.barh(range(len(d)), d["matched_z"],
            color=["#c44e52" if (not np.isnan(q) and q < 0.05) else "#8c8c8c" for q in d["matched_q"]],
            label=T["p"])
    if ref is not None:
        r = ref.set_index("layer")["matched_z"].reindex(d["layer"])
        ax.scatter(r.values, range(len(d)), marker="D", s=22, facecolor="white", edgecolor="black", zorder=3,
                   label="real seeds (reference)")
        ax.legend(fontsize=8, loc="lower right")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["layer"], fontsize=8)
    ax.set_xlabel("z-score vs random genes with the same global-PPI degree   (red = BH q<0.05)")
    ax.set_title(f"Do {T['p']} have higher in-layer degree than expected\n"
                 f"given how connected they are globally?  (>0 = yes)", fontsize=10, fontweight="bold")
    _save(fig, plt, out / "degree_matched_null.png")


def plot_seed_vs_neighbor(neigh: pd.DataFrame, seeds_df: pd.DataFrame, global_name: str, out: Path, have_markers: bool):
    """Per-layer side-by-side of seeds and neighbors. Solid bar = BH q<0.05."""
    mpl, plt = _plt()
    if plt is None:
        return
    a = neigh[~neigh["is_global"]].set_index("layer")
    b = seeds_df[~seeds_df["is_global"]].set_index("layer")
    layers, _ = order_layers(list(a.index), global_name)
    if not layers:
        return
    panels = [("degree_auc", "degree_q", "Degree AUC (0.5 = none)", 0.5)]
    if have_markers:
        panels.append(("log2fc_auc", "log2fc_q", "avg_log2FC AUC (0.5 = none)", 0.5))
    panels.append(("matched_z", "matched_q", "Degree vs degree-matched genes (z; 0 = none)", 0.0))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels) + 1, max(4, 0.42 * len(layers) + 1.8)),
                             sharey=True, squeeze=False)
    y = np.arange(len(layers))
    h = 0.38
    for ax, (col, qcol, title, base) in zip(axes[0], panels):
        for off, df, color in [(-h / 2, b, "#4c72b0"), (h / 2, a, "#dd8452")]:
            vals = df[col].reindex(layers).astype(float).fillna(base).values
            qs = df[qcol].reindex(layers).values
            cols = [mpl.colors.to_rgba(color, 1.0 if (not np.isnan(q) and q < 0.05) else 0.35) for q in qs]
            ax.barh(y + off, vals - base, left=base, height=h, color=cols, edgecolor=color, lw=0.6)
        ax.axvline(base, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(layers, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
    axes[0][0].invert_yaxis()
    handles = [mpl.patches.Patch(color="#4c72b0", label="seeds"), mpl.patches.Patch(color="#dd8452", label="neighbors")]
    axes[0][-1].legend(handles=handles + [mpl.patches.Patch(facecolor="grey", alpha=0.35, label="faded = q>=0.05")],
                       fontsize=8, loc="lower right")
    fig.suptitle("Seeds vs neighbors, layer by layer (each tested against a background without the other set)",
                 fontsize=10, fontweight="bold")
    _save(fig, plt, out / "seed_vs_neighbor.png")


def plot_layer_boxplots(seed_ng, global_name, out, have_markers):
    """Spread of target percentiles per layer; the 0.5 line is what 'no skew' looks like."""
    _, plt = _plt()
    if plt is None or not len(seed_ng):
        return
    layers, colors = order_layers(sorted(seed_ng["layer"].unique()), global_name)
    cols = [("degree_pctile", "Degree percentile")]
    if have_markers:
        cols.append(("fc_pctile", "avg_log2FC percentile (among markers)"))
    fig, axes = plt.subplots(len(cols), 1, figsize=(max(7, 0.45 * len(layers) + 2), 4.5 * len(cols)), squeeze=False)
    for ax, (col, title) in zip(axes[:, 0], cols):
        data = [seed_ng.loc[seed_ng["layer"] == l, col].dropna().values for l in layers]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, l in zip(bp["boxes"], layers):
            patch.set_facecolor(colors[l])
            patch.set_alpha(0.6)
        ax.axhline(0.5, color="black", ls="--", lw=0.8)
        ax.set_xticks(range(1, len(layers) + 1))
        ax.set_xticklabels([f"{l} (n={len(x)})" for l, x in zip(layers, data)], rotation=90, fontsize=7)
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.set_title(f"{T['S']} {title.lower()} by layer (dashed = no skew)", fontsize=10, fontweight="bold")
    _save(fig, plt, out / "layer_percentile_boxplots.png")


def plot_tiers(summ, out, tier_order):
    _, plt = _plt()
    if plt is None:
        return
    counts = summ["specificity_tier"].value_counts().reindex(tier_order).dropna()
    counts = counts[counts > 0]
    if not len(counts):
        return
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(counts) + 1.5))
    ax.barh(range(len(counts)), counts.values[::-1], color="#4c72b0")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(list(counts.index)[::-1], fontsize=8)
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=8)
    ax.set_xlabel(f"Number of {T['p']}")
    ax.set_title(f"{T['S']} specificity tiers", fontsize=10, fontweight="bold")
    _save(fig, plt, out / "tier_breakdown.png")


def plot_layer_counts(summ, out):
    """How many cell-type layers each gene shows up in, and in how many it is 'strong'."""
    _, plt = _plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(axes, ["n_layers_present", "n_layers_strong"],
                              ["Layers the gene is in (is a marker/node)", "Layers where it is 'strong'"]):
        v = summ[col].dropna().astype(int)
        bins = np.arange(0, v.max() + 2) - 0.5 if len(v) else [0, 1]
        ax.hist(v, bins=bins, color="#55a868", edgecolor="white")
        ax.set_xlabel(title)
        ax.set_ylabel(f"Number of {T['p']}")
    fig.suptitle(f"Recurrence of {T['p']} across cell-type layers", fontsize=11, fontweight="bold")
    _save(fig, plt, out / "genes_per_layer_hist.png")


def plot_degree_vs_marker(pres_all, enrich_ok, out, hub_frac, enrich_label):
    """Each dot = one (gene, layer) entry. Top-right = well-connected AND a strong marker."""
    _, plt = _plt()
    if plt is None:
        return
    d = pres_all[pres_all["present"]].dropna(subset=["degree_pctile", "fc_pctile"]).copy()
    if not len(d):
        return
    d["enr"] = d["cell_type"].isin(enrich_ok)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(d.loc[~d["enr"], "degree_pctile"], d.loc[~d["enr"], "fc_pctile"], s=10, c="#8c8c8c", alpha=0.5,
               label="other layers")
    ax.scatter(d.loc[d["enr"], "degree_pctile"], d.loc[d["enr"], "fc_pctile"], s=14, c="#c44e52", alpha=0.7,
               label=enrich_label)
    ax.axvline(1 - hub_frac, color="black", ls=":", lw=0.8)
    ax.axhline(1 - hub_frac, color="black", ls=":", lw=0.8)
    ax.set_xlabel("Degree percentile in layer (1 = most connected)")
    ax.set_ylabel("avg_log2FC percentile among markers (1 = strongest)")
    ax.set_title(f"{T['S']}-layer entries: connectivity vs marker strength\n"
                 f"(top-right box = hub AND top marker)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, plt, out / "degree_vs_marker_scatter.png")


def plot_auc_scatter(layer_df, out):
    """Per layer: do targets skew high on degree (x) and on marker strength (y)?"""
    _, plt = _plt()
    if plt is None:
        return
    d = layer_df[~layer_df["is_global"]].dropna(subset=["degree_auc", "log2fc_auc"])
    if not len(d):
        return
    sig = (d["degree_q"] < 0.05) | (d["log2fc_q"] < 0.05)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(d["degree_auc"], d["log2fc_auc"], s=d["n_seeds"].clip(upper=300) * 0.6 + 15,
               c=np.where(sig, "#c44e52", "#8c8c8c"), alpha=0.7)
    for _, r in d[sig].iterrows():
        ax.annotate(r["layer"], (r["degree_auc"], r["log2fc_auc"]), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.axvline(0.5, color="black", lw=0.6)
    ax.axhline(0.5, color="black", lw=0.6)
    ax.set_xlabel("Degree AUC (0.5 = no difference)")
    ax.set_ylabel("avg_log2FC AUC (0.5 = no difference)")
    ax.set_title(f"Per layer: {T['p']} vs background\n(dot size = number of {T['p']}; red = q<0.05 in either)",
                 fontsize=10, fontweight="bold")
    _save(fig, plt, out / "layer_auc_scatter.png")


def plot_global_hist(summ, out, hub_frac):
    _, plt = _plt()
    if plt is None or "global_degree_pctile" not in summ:
        return
    v = summ["global_degree_pctile"].dropna()
    if not len(v):
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(v, bins=np.linspace(0, 1, 21), color="#4c72b0", edgecolor="white")
    ax.axhline(len(v) / 20, color="black", ls="--", lw=0.8, label="null (uniform)")
    ax.axvline(1 - hub_frac, color="#c44e52", ls=":", lw=1, label=f"top {hub_frac:.0%} cutoff")
    ax.set_xlabel("Degree percentile in the full global PPI (1 = biggest hub)")
    ax.set_ylabel(f"Number of {T['p']}")
    ax.set_title(f"Are {T['p']} hubs in the full PPI?  (bars above dashed line on the right = yes)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    _save(fig, plt, out / "global_degree_hist.png")


def make_heatmap(pres_all, summ, out, global_name, value_col, top_n) -> bool:
    """Gene x layer heatmap. Rows = top-N genes by consistency_score; columns =
    layers (DRG-ordered when recognised). grey = gene not present in that layer."""
    matplotlib, plt = _plt()
    if plt is None:
        return False

    top_genes = summ.head(top_n)["gene"].tolist()
    if not top_genes:
        print(f"  No {T['p']} to plot -- skipping heatmap")
        return False

    layers, colors = order_layers(sorted(pres_all["cell_type"].unique()), global_name)
    mat = (pres_all[pres_all["gene"].isin(top_genes)]
           .pivot_table(index="gene", columns="cell_type", values=value_col, aggfunc="first")
           .reindex(index=top_genes, columns=layers))

    fig, ax = plt.subplots(figsize=(max(6, 0.42 * len(layers) + 3), max(4, 0.24 * len(top_genes) + 1.5)))
    cmap = matplotlib.colormaps.get_cmap("YlOrRd").copy()
    cmap.set_bad("#e8e8e8")
    im = ax.imshow(np.ma.masked_invalid(mat.values), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=7)
    for tick, c in zip(ax.get_xticklabels(), [colors[l] for l in layers]):
        tick.set_color(c)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=7)
    label = "avg_log2FC percentile (marker strength)" if value_col == "fc_pctile" else "degree percentile"
    fig.colorbar(im, ax=ax, shrink=0.7, label=label + "  (grey = not a marker gene in that layer)")
    ax.set_title(f"Top {len(top_genes)} {T['p']} by cross-layer consistency", fontsize=11, fontweight="bold")
    _save(fig, plt, out / "seed_layer_heatmap.png")
    return True


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def enriched_layers(tests: pd.DataFrame) -> set:
    """Layers used to define 'enriched' for tiers/reports/plots. In neighbor
    mode this column holds the SEEDS' enrichment q (see main)."""
    return set(tests.loc[(~tests["is_global"]) & (tests["tier_enrich_q"] < 0.05), "layer"])


def write_seed_reports(pres_all, summ, tests, out, hub_frac, have_markers, top_n, enrich_who,
                       module_layers_by_gene=None):
    """One human-readable block per top gene."""
    enrich_ok = enriched_layers(tests)
    lines = ["=" * 78, f"PER-{T['U']} REPORTS (top {T['p']} by cross-layer consistency)", "=" * 78,
             "Read each block top to bottom: global role, then per-layer detail.",
             f"'ENRICHED LAYER' = {enrich_who} significantly over-represented",
             "among that cell type's markers (q<0.05, tier_enrich_q in layer_seed_tests.csv).",
             "deg = number of marker neighbours in that layer; pXX = percentile (1.00 = top).",
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
            lines.append(f"  Recruited into the disease module under: "
                         f"{', '.join(ml) if ml else '(none -- unexpected, check disease_module.csv)'}")
        if not len(rows):
            lines.append(f"  Not a marker/node in any cell-type layer ({T['s']} acts only via global_ppi, if present)")
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

    (out / "seed_individual_reports.txt").write_text("\n".join(lines))
    print(f"  Saved -> {out / 'seed_individual_reports.txt'}")


def how_to_read(neighbor_mode: bool, have_markers: bool, hub_frac: float, seeds_ok: bool) -> list[str]:
    """Plain-language guide appended to the summary text."""
    tp = T["p"]
    enr = "your SEEDS are" if (neighbor_mode and seeds_ok) else f"the {tp} are"
    L = ["", "-" * 78, "HOW TO READ THIS", "-" * 78,
         f"Target set: the {tp} (" + ("non-seed genes the disease module recruited to connect your seeds)"
                                      if neighbor_mode else "your input seed genes)"),
         "Layer: one PPI network per cell type, built only from that cell type's marker genes.",
         "Degree: how many other genes in the layer a gene is connected to.",
         f"Percentile: where a gene ranks within its layer (0.50 = median; 1.00 = best). If {tp} were random",
         "  genes, their percentiles would average 0.50.",
         f"Hub: top {hub_frac:.0%} of its layer by degree.",
         "AUC: probability that a random target beats a random non-target. 0.50 = no difference,",
         "  0.60 = modest skew, 0.70+ = strong. Read it together with q, not instead of it.",
         "q (BH): p-value corrected across layers. q<0.05 = unlikely to be chance for that layer.",
         "Degree-matched test: compares each target with random genes that have the SAME degree in the",
         "  full global PPI. If a layer is 'hubby' by AUC but NOT by this test, the hubness is explained",
         "  by general popularity (literature bias / being a global hub), not by the cell-type layer.",
         f"Enriched layer: a layer where {enr} over-represented among that cell type's markers.",
         "  Used to define the tiers below. Says nothing about WHICH cell type matters on its own.",
         "Strong (in a layer): well-connected (degree pctile >= 0.75) AND a top marker there"
         if have_markers else "Strong (in a layer): degree percentile >= 0.90 there",
         "  (top-marker cutoff = --hub-frac). Genes strong in several layers recur across cell types.",
         "Specificity tier:",
         "  A  = a specific marker AND well-connected in an enriched layer (best lead)",
         "  A* = like A, but also a generic hub in the full PPI (could be literature/study bias)",
         "  B  = specific marker in an enriched layer, but not well-connected there",
         "  C  = hub in an enriched layer but not a distinctive marker",
         "  D  = generic global hub, no enriched layer (likely just a popular gene)",
         "  F  = present but unremarkable;  E = absent from every cell-type layer",
         "Isolated genes: in a layer's gene set but with no edges; counted as degree 0 so the",
         "  background isn't biased upward."]
    if neighbor_mode:
        L += ["", "NEIGHBOR-MODE NOTES",
              "  * Neighbors were chosen BECAUSE they connect seeds in these layers, so their own layer"]
        if seeds_ok:
            L += ["    enrichment is circular and is ignored. 'Enriched layer' above comes from the seeds."]
        else:
            L += ["    enrichment is circular. WARNING: seeds could not be loaded, so 'enriched layer' fell back",
                  "    to it (every layer looks enriched). Pass --seed-csv."]
        L += ["  * Neighbor hubness is partly built in too (bridging genes are hubs); the degree-matched",
              "    test and the seed reference are the fair comparisons."]
    else:
        L += ["", "SEED-MODE NOTE", "  * Seeds are usually well-studied genes -> more known PPIs. See the degree-matched test."]
    return L


def categorize_seeds(pres_all, summ, tests, have_markers) -> pd.Series:
    """One tier per gene -- a decision tree over the same columns you'd read by
    eye. Orthogonal to consistency_score / n_layers_strong ("recurs across
    layers" axis); this is the "specific vs. generic" axis."""
    enrich_ok = enriched_layers(tests)
    p = pres_all.copy()
    p["in_enriched"] = p["cell_type"].isin(enrich_ok)
    if have_markers:
        if p["pct_diff"].notna().any():
            p["is_specific"] = (p["fc_pctile"] >= 0.90) & (p["pct_diff"].fillna(0) >= 0.30)
        else:
            p["is_specific"] = (p["fc_pctile"] >= 0.90) & (p["is_sig_up"] == True)  # noqa: E712

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
                    help="Seeds CSV ('gene' column). In seed mode: the target set. In neighbor mode: the "
                         "reference seeds used to define enriched layers. Default: <wdir>/seeds/.")
    ap.add_argument("--disease-module-csv", default=None,
                    help="NEIGHBOR mode: analyse the disease module's NON-seed genes (step6 disease_module.csv). "
                         "Use a different --output-dir than your seed run.")
    ap.add_argument("--markers-csv", default=None,
                    help="cell_type_markers.csv from the R script; enables marker-strength tests")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--global-ppi-name", default="global_ppi")
    ap.add_argument("--hub-frac", type=float, default=0.10,
                    help="Top fraction by degree / log2FC counted as 'hub' / 'top marker' (default 0.10)")
    ap.add_argument("--min-seeds", type=int, default=3, help="Min target genes in a layer to run tests (default 3)")
    ap.add_argument("--n-perm", type=int, default=2000,
                    help="Permutations for the degree-matched null (default 2000; 0 disables it)")
    ap.add_argument("--rng-seed", type=int, default=0, help="Random seed for the matched null (default 0)")
    ap.add_argument("--keep-other-set-in-background", action="store_true",
                    help="Neighbor mode: do NOT remove the other set (seeds/neighbors) from the test background.")
    ap.add_argument("--top-n-report", type=int, default=40,
                    help="Top genes (by consistency_score) shown in the heatmap and individual reports (default 40)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    wdir, out = Path(args.wdir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    neighbor_mode = bool(args.disease_module_csv)
    set_terms(neighbor_mode)

    gene_sets = load_gene_sets(args.celltype_gene_sets or (wdir / "celltype_gene_sets.json"))
    module_layers_by_gene: dict[str, set] = {}
    ref_seeds: set = set()
    if neighbor_mode:
        targets, module_layers_by_gene = load_module_neighbors(args.disease_module_csv)
        print("[mode: NEIGHBOR] target set = non-seed genes recruited by the disease module. "
              "'neighbors' below means those genes, NOT your original seeds. "
              "(CSV column names still say 'seed'; read that as 'neighbor'.)")
        try:
            ref_seeds = set(load_seeds(wdir, args.seed_csv))
        except SystemExit as e:
            print(f"WARNING: could not load seeds for the reference analysis ({e}). "
                  f"'Enriched layer' will fall back to the neighbors' own (circular) enrichment.")
    else:
        targets = load_seeds(wdir, args.seed_csv)
        print("[mode: SEED] target set = your input seed genes.")
    target_set = set(targets)
    seeds_ok = neighbor_mode and bool(ref_seeds)
    gname = args.global_ppi_name
    print(f"{len(gene_sets)} layers in celltype_gene_sets.json")

    if gname in gene_sets:
        universe = gene_sets[gname]
        print(f"Enrichment universe: '{gname}' ({len(universe):,} genes); "
              f"{len(target_set & universe)}/{len(targets)} {T['p']} are in it")
    else:
        universe = set().union(*gene_sets.values())
        print(f"WARNING: '{gname}' not found: using union of all layers as enrichment universe "
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
            print(f"  WARNING: {len(no_markers)} layer(s) with no matching marker rows (name mismatch?): {no_markers}")
    else:
        print("[info] --markers-csv not given: marker-strength tests skipped (hubness + enrichment still run).")

    # ---- load every layer's degrees once ----
    cache: dict = {}
    for ct, genes in sorted(gene_sets.items()):
        p = find_layer_file(wdir, ct)
        if p is None:
            print(f"  WARNING: no ppi_layer file for '{ct}' -- skipped")
            continue
        is_global = ct == gname
        mk = None if (markers is None or is_global) else markers[markers["layer"] == ct]
        cache[ct] = (layer_degrees(p, genes), mk, is_global)

    gdeg, edges = None, None
    if args.n_perm > 0 and gname in cache:
        gdeg = cache[gname][0]
        edges = np.unique(np.quantile(gdeg.values, np.linspace(0, 1, 21)))
        print(f"Degree-matched null: {len(edges) - 1} global-degree bins, {args.n_perm} permutations")
    elif args.n_perm > 0:
        print("[info] no global_ppi layer -> degree-matched null skipped.")

    exclude_for_target = set() if args.keep_other_set_in_background else ref_seeds
    exclude_for_ref = set() if args.keep_other_set_in_background else target_set
    if neighbor_mode and seeds_ok and not args.keep_other_set_in_background:
        print(f"Background for the neighbor tests excludes the {len(ref_seeds)} real seeds "
              f"(use --keep-other-set-in-background to restore the old behaviour).")

    print(f"\nLayers (genes in layer, isolated genes, number of {T['p']} in layer, degree AUC):")
    layer_df, seed_tables = run_layers(target_set, exclude_for_target, cache, universe, gdeg, edges, args, rng, True)

    ref_df = None
    if seeds_ok:
        saved = dict(T)
        set_terms(False)  # reference run is about seeds
        ref_df, _ = run_layers(ref_seeds, exclude_for_ref, cache, universe, gdeg, edges, args, rng, False)
        T.update(saved)
        ref_df.to_csv(out / "reference_seed_layer_tests.csv", index=False)
        print(f"Saved -> {out / 'reference_seed_layer_tests.csv'}  (real seeds, same layers)")

    # which enrichment defines 'enriched layer' for tiers/reports/plots
    if seeds_ok:
        layer_df["tier_enrich_q"] = ref_df.set_index("layer")["enrich_q"].reindex(layer_df["layer"]).values
        enrich_who = "your SEEDS are"
        enrich_label = "layers where seeds are enriched (q<0.05)"
    else:
        layer_df["tier_enrich_q"] = layer_df["enrich_q"]
        enrich_who = f"the {T['p']} are"
        enrich_label = "enriched layers (q<0.05)"
    layer_df.to_csv(out / "layer_seed_tests.csv", index=False)
    is_ng = ~layer_df["is_global"]

    # ---- long presence table: every target gene x every layer ----
    layer_names = list(layer_df["layer"])
    long = pd.MultiIndex.from_product([targets, layer_names], names=["gene", "cell_type"]).to_frame(index=False)
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

    # ---- per-gene summary (cell-type layers only; global reported separately) ----
    pres = long[(~long["is_global"]) & long["present"]]
    conn = pres[pres["degree"] > 0]
    g = pres.groupby("gene")
    summ = pd.DataFrame(index=pd.Index(targets, name="gene"))
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
    print(f"\n{n_absent}/{len(targets)} {T['s']} gene(s) absent from every cell-type layer entirely")
    if n_absent:
        print("  " + ", ".join(summ.loc[summ["n_layers_present"] == 0, "gene"].tolist()[:20]))

    # ---- text summary ----
    ng = layer_df[is_ng]
    testable = ng.dropna(subset=["degree_auc"])
    seed_ng = pd.concat([t for t in seed_tables if not t["is_global"].iloc[0]], ignore_index=True) \
        if any(not t["is_global"].iloc[0] for t in seed_tables) else pd.DataFrame()
    L = ["=" * 78, f"{T['U']} HUBNESS / MARKER-STRENGTH SUMMARY", "=" * 78,
         ("MODE: NEIGHBOR -- target genes are the non-seed genes the disease module recruited "
          "(not your input seeds)" if neighbor_mode else "MODE: SEED -- target genes are your input seeds"),
         f"{T['P']}: {len(targets)} | in enrichment universe: {len(target_set & universe)} | "
         f"hub / top-marker cutoff: top {args.hub_frac:.0%}"]
    if neighbor_mode and not args.keep_other_set_in_background and seeds_ok:
        L.append(f"Test background excludes the {len(ref_seeds)} real seeds.")
    if layer_df["is_global"].any():
        gr = layer_df[layer_df["is_global"]].iloc[0]
        L += ["", f"GLOBAL PPI ('{gname}'): are {T['p']} hubs in the full network?"]
        if np.isnan(gr["degree_auc"]):
            L.append(f"  too few {T['p']} in the global PPI to test ({gr['n_seeds']})")
        else:
            L += [f"  {T['p']} in PPI: {gr['n_seeds']} | median degree {T['p']} {gr['seed_median_degree']:.1f} "
                  f"vs others {gr['bg_median_degree']:.1f}",
                  f"  degree AUC = {gr['degree_auc']:.3f} (p = {gr['degree_p']:.2e}) | {T['p']} in top "
                  f"{args.hub_frac:.0%}: {gr['n_seed_hubs']} (expected {gr['expected_seed_hubs']:.1f}, p = {gr['hub_p']:.2e})"]
    L += ["", f"PER-LAYER (testable layers = {len(testable)} of {len(ng)}; BH across layers)",
          f"  {T['P']} have higher degree than other genes        (q<0.05): {int((testable['degree_q'] < 0.05).sum())} layers",
          f"  {T['P']} over-represented in the top-degree bin     (q<0.05): {int((testable['hub_q'] < 0.05).sum())} layers"]
    if args.n_perm > 0 and gdeg is not None:
        L.append(f"  {T['P']} higher degree than DEGREE-MATCHED genes    (q<0.05): "
                 f"{int((ng['matched_q'] < 0.05).sum())} layers  <- survives literature/popularity bias")
    if neighbor_mode and seeds_ok:
        L.append(f"  {T['P']} enriched among markers (q<0.05): {int((ng['enrich_q'] < 0.05).sum())} layers  "
                 f"<- circular by construction, ignore")
        sig_layers = sorted(enriched_layers(layer_df))
        L.append(f"  Your SEEDS enriched among markers (q<0.05): {len(sig_layers)} layers "
                 f"<- these define 'enriched layer' for the tiers")
        for l in sig_layers:
            L.append(f"      - {l}")
    else:
        L.append(f"  {T['P']} enriched among the cell type's markers     (q<0.05): {int((ng['enrich_q'] < 0.05).sum())} layers"
                 + ("  <- circular for neighbors (fell back; pass --seed-csv)" if neighbor_mode else ""))
    if markers is not None:
        L += [f"  {T['P']} have higher avg_log2FC than other markers  (q<0.05): {int((testable['log2fc_q'] < 0.05).sum())} layers",
              f"  {T['P']} have higher pct.1-pct.2 than other markers (q<0.05): {int((testable['pctdiff_q'] < 0.05).sum())} layers"]

    if ref_df is not None:
        rng_ = ref_df[~ref_df["is_global"]]
        rt = rng_.dropna(subset=["degree_auc"])
        L += ["", "SEEDS vs NEIGHBORS (same layers; each tested against a background without the other set)",
              f"  {'':<46}{'seeds':>10}{'neighbors':>12}"]
        gref = ref_df[ref_df["is_global"]]
        if len(gref) and layer_df["is_global"].any():
            L.append(f"  {'global PPI degree AUC':<46}{gref.iloc[0]['degree_auc']:>10.3f}"
                     f"{layer_df[layer_df['is_global']].iloc[0]['degree_auc']:>12.3f}")
        rows_cmp = [("layers: higher degree (q<0.05)", "degree_q"),
                    ("layers: higher than degree-matched (q<0.05)", "matched_q")]
        if markers is not None:
            rows_cmp.append(("layers: higher avg_log2FC (q<0.05)", "log2fc_q"))
        for lab, qc in rows_cmp:
            if qc == "matched_q" and not (args.n_perm > 0 and gdeg is not None):
                continue
            L.append(f"  {lab:<46}{int((rng_[qc] < 0.05).sum()):>10}{int((ng[qc] < 0.05).sum()):>12}")
        L.append(f"  {'median degree AUC across layers':<46}{rt['degree_auc'].median():>10.3f}"
                 f"{testable['degree_auc'].median():>12.3f}")

    if len(seed_ng):
        L += ["", f"POOLED {T['s']}-layer entries (descriptive; entries are not independent)",
              f"  n = {len(seed_ng)} | median degree percentile {seed_ng['degree_pctile'].median():.2f} (null 0.50) | "
              f"fraction in top {args.hub_frac:.0%}: {seed_ng['is_hub'].mean():.2f} (null ~{args.hub_frac:.2f})"]
        if markers is not None:
            L.append(f"  median avg_log2FC percentile among markers: {seed_ng['fc_pctile'].median():.2f} (null 0.50)")
    L += ["", f"{T['U']} CATEGORIZATION (specificity_tier, full breakdown in seed_layer_summary.csv)"]
    for t, n in summ["specificity_tier"].value_counts().reindex(tier_order).items():
        if pd.notna(n) and n > 0:
            L.append(f"  {t:<62} {int(n):>4}")
    L += ["", f"Isolated genes (degree 0) across layers: {int(layer_df['n_isolated'].sum()):,} "
              f"(in gene sets json but with no edges in ppi_layer.tsv)"]
    L += how_to_read(neighbor_mode, markers is not None, args.hub_frac, seeds_ok or not neighbor_mode)
    L += ["=" * 78]
    text = "\n".join(L)
    (out / "seed_hubness_summary.txt").write_text(text)
    print("\n" + text)

    # ---- reports + plots ----
    value_col = "fc_pctile" if markers is not None else "degree_pctile"
    write_seed_reports(pres_all, summ, layer_df, out, args.hub_frac, markers is not None, args.top_n_report,
                       enrich_who, module_layers_by_gene)

    if not args.no_plots:
        print()
        have_m = markers is not None
        make_plots(layer_df, seed_ng, out, have_m)
        make_heatmap(pres_all, summ, out, gname, value_col, args.top_n_report)
        if seeds_ok:
            plot_enrichment(ref_df, out, "your seeds",
                            "(defines 'enriched layers' for the neighbor tiers; n = seeds in that layer)")
        else:
            plot_enrichment(layer_df, out, T["p"], f"(n = number of {T['p']} in that layer)")
        plot_layer_boxplots(seed_ng, gname, out, have_m)
        plot_tiers(summ, out, tier_order)
        plot_layer_counts(summ, out)
        plot_global_hist(summ, out, args.hub_frac)
        if args.n_perm > 0 and gdeg is not None:
            plot_matched(layer_df, ref_df, out)
        if have_m:
            plot_degree_vs_marker(pres_all, enriched_layers(layer_df), out, args.hub_frac, enrich_label)
            plot_auc_scatter(layer_df, out)
        if ref_df is not None:
            plot_seed_vs_neighbor(layer_df, ref_df, gname, out, have_m)
    print(f"\nAll outputs -> {out.resolve()}")


if __name__ == "__main__":
    main()