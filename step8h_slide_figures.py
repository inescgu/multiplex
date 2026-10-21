"""
step8h_slide_figures.py
------------------------------------------------------------------------------------
Generates the figures suggested for the 6-slide deck, reading directly from files
already produced by step8 / step8e / step8f / step8g. Nothing here recomputes any
analysis -- it only visualizes results that already exist on disk.

Produces (into --out-dir, default 08_spectral_analysis/slide_figures/):
  slide2_k_selection.png       Slide 2: silhouette-vs-k + eigenvalue scree, both conditions
  slide2_embedding_scatter.png Slide 2 (optional): first 2 embedding dims, colored by cluster
  slide3_enrichment_bars.png   Slide 3: core/periphery seed enrichment, with_ppi vs no_ppi
  slide3_null_histogram.png    Slide 3: no_ppi permutation-null histogram w/ observed line
  slide4_forest_plot.png       Slide 4: observed vs permutation-null range, 6 rows
  slide4_silhouette_null.png   Slide 4 (optional): one clean null-vs-observed example
  slide5_degree_distributions.png  Slide 5: seed vs non-seed degree histograms (no_ppi)
  slide5_illustrative_subgraph.png Slide 5 (optional): small seed-at-periphery cartoon
  slide6_established_table.png Slide 6: established / not-yet-established summary table

Usage:
  python step8h_slide_figures.py \
      --with-ppi-dir 08_spectral_analysis/module_with_ppi \
      --no-ppi-dir 08_spectral_analysis/module_no_ppi \
      --ref-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
      --seed-genes-csv 00_inputs/seed_genes.csv \
      --out-dir 08_spectral_analysis/slide_figures

Each figure function fails gracefully (prints a warning and skips) if its required
input file is missing, so a partial run still produces whatever it can.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
COLOR_CORE = "#c0392b"       # deep red
COLOR_PERIPHERY = "#2980b9"  # blue
COLOR_WITH_PPI = "#7f8c8d"   # grey
COLOR_NO_PPI = "#27ae60"     # green
COLOR_SEED = "#e67e22"       # orange
COLOR_NONSEED = "#8e8e8e"    # neutral grey
REF_LINE_COLOR = "#333333"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _warn_missing(path):
    print(f"  [skip] file not found, skipping this figure: {path}")


# ---------------------------------------------------------------------------
# SLIDE 2 — k selection (silhouette sweep + eigenvalue scree), both conditions
# ---------------------------------------------------------------------------
def make_slide2_k_selection(with_ppi_dir: Path, no_ppi_dir: Path, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for row, (label, d, color) in enumerate([
        ("with_ppi", with_ppi_dir, COLOR_WITH_PPI),
        ("no_ppi", no_ppi_dir, COLOR_NO_PPI),
    ]):
        k_sel_path = d / "k_selection.csv"
        eig_path = d / "eigvals_sos.npy"
        meta_path = d / "meta.json"

        if not k_sel_path.exists():
            _warn_missing(k_sel_path)
            continue
        k_df = pd.read_csv(k_sel_path)

        k_opt = None
        eigengap_k = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            k_opt = meta.get("k_opt")
            eigengap_k = meta.get("eigengap_k")

        ax = axes[row, 0]
        ax.plot(k_df["k"], k_df["silhouette"], "o-", color=color, markersize=4, linewidth=1.2)
        if k_opt is not None:
            ax.axvline(k_opt, color=REF_LINE_COLOR, linestyle="--", linewidth=1,
                       label=f"silhouette k={k_opt}")
        if eigengap_k is not None:
            ax.axvline(eigengap_k, color="orange", linestyle=":", linewidth=1.5,
                       label=f"eigengap k={eigengap_k}")
        ax.set_xlabel("k")
        ax.set_ylabel("silhouette score")
        ax.set_title(f"{label}: silhouette vs. k")
        ax.legend(fontsize=8)

        ax2 = axes[row, 1]
        if eig_path.exists():
            sigma = np.load(eig_path)
            ax2.bar(range(1, len(sigma) + 1), sigma, color=color, alpha=0.85)
            ax2.set_xlabel("component index")
            ax2.set_ylabel("eigenvalue")
            ax2.set_title(f"{label}: eigenvalue spectrum")
        else:
            _warn_missing(eig_path)
            ax2.axis("off")

    fig.tight_layout()
    out_path = out_dir / "slide2_k_selection.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def make_slide2_embedding_scatter(with_ppi_dir: Path, no_ppi_dir: Path, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (label, d) in zip(axes, [("with_ppi", with_ppi_dir), ("no_ppi", no_ppi_dir)]):
        emb_path = d / "embedding.npy"
        clusters_path = d / "spectral_clusters.csv"
        if not emb_path.exists() or not clusters_path.exists():
            _warn_missing(emb_path if not emb_path.exists() else clusters_path)
            ax.axis("off")
            continue
        U = np.load(emb_path)
        df = pd.read_csv(clusters_path)
        labels = df["cluster"].values
        cmap = plt.get_cmap("Set2")
        for cid in sorted(set(labels)):
            mask = labels == cid
            ax.scatter(U[mask, 0], U[mask, 1], s=6, alpha=0.5,
                      color=cmap(cid % 8), label=f"cluster {cid}")
        ax.set_xlabel("embedding dim 1")
        ax.set_ylabel("embedding dim 2")
        ax.set_title(label)
        ax.legend(fontsize=8, markerscale=2)

    fig.tight_layout()
    out_path = out_dir / "slide2_embedding_scatter.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# SLIDE 3 — seed enrichment bar chart + no_ppi null histogram
# ---------------------------------------------------------------------------
def make_slide3_enrichment_bars(with_ppi_dir: Path, no_ppi_dir: Path, out_dir: Path):
    rows = []
    for label, d in [("with_ppi", with_ppi_dir), ("no_ppi", no_ppi_dir)]:
        summary_path = d / "diagnostics" / f"seed_periphery_summary_{label}.json"
        if not summary_path.exists():
            _warn_missing(summary_path)
            continue
        with open(summary_path) as f:
            s = json.load(f)
        obs = s["observed"]
        rows.append({"label": label, "core": obs["core_seed_enrichment"],
                     "periphery": obs["periphery_seed_enrichment"]})

    if not rows:
        print("  [skip] no seed_periphery_summary files found, skipping slide3 bar chart")
        return

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, df["core"], width, label="core", color=COLOR_CORE)
    ax.bar(x + width / 2, df["periphery"], width, label="periphery", color=COLOR_PERIPHERY)
    ax.axhline(1.0, color=REF_LINE_COLOR, linestyle="--", linewidth=1, label="no enrichment (=1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("seed gene enrichment")
    ax.set_title("Seed gene enrichment: core vs. periphery")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "slide3_enrichment_bars.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def make_slide3_null_histogram(no_ppi_dir: Path, out_dir: Path, metric: str = "periphery_seed_enrichment"):
    perm_csv = no_ppi_dir / "diagnostics" / "permutation_null_summary_no_ppi.csv"
    summary_path = no_ppi_dir / "diagnostics" / "seed_periphery_summary_no_ppi.json"

    if not perm_csv.exists():
        _warn_missing(perm_csv)
        print("  NOTE: this file is a per-metric SUMMARY (mean/std), not raw null values. "
              "If you want the actual 1000-value null distribution plotted, add a raw-values "
              "dump to step8g (e.g. np.save the null[] list before summarize_null collapses it) "
              "and point this function at that instead. Falling back to a normal approximation "
              "built from the summary's null_mean/null_std.")

    if not summary_path.exists():
        _warn_missing(summary_path)
        return

    with open(summary_path) as f:
        s = json.load(f)
    observed = s["observed"][metric]

    null_mean, null_std = None, None
    if perm_csv.exists():
        perm_df = pd.read_csv(perm_csv)
        row = perm_df[perm_df["metric"] == metric]
        if not row.empty:
            null_mean = float(row["null_mean"].iloc[0])
            null_std = float(row["null_std"].iloc[0])

    if null_mean is None:
        print(f"  [skip] could not find null_mean/null_std for {metric}, skipping slide3 histogram")
        return

    rng = np.random.default_rng(0)
    approx_null = rng.normal(null_mean, null_std, size=1000)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(approx_null, bins=40, color=COLOR_PERIPHERY, alpha=0.6,
           label="permutation null (approx. from summary stats)")
    ax.axvline(observed, color=COLOR_CORE, linewidth=2, label=f"observed = {observed:.3f}")
    ax.set_xlabel(metric)
    ax.set_ylabel("count")
    ax.set_title("no_ppi: observed vs. permutation null")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "slide3_null_histogram.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")
    print("  NOTE: this histogram is a Gaussian approximation from null_mean/null_std in the "
          "summary CSV, not the raw 1000 permutation values (those aren't saved to disk by "
          "step8g as written). Fine for a slide, but say 'approximate null distribution' if asked.")


# ---------------------------------------------------------------------------
# SLIDE 4 — forest plot of step8f partition-level null results
# ---------------------------------------------------------------------------
def make_slide4_forest_plot(with_ppi_dir: Path, no_ppi_dir: Path, out_dir: Path):
    rows = []
    for label, d in [("with_ppi", with_ppi_dir), ("no_ppi", no_ppi_dir)]:
        summary_path = d / "diagnostics" / f"null_model_summary_{label}.json"
        if not summary_path.exists():
            _warn_missing(summary_path)
            continue
        with open(summary_path) as f:
            s = json.load(f)
        perm = s.get("permutation_vs_observed", {})
        for stat, vals in perm.items():
            rows.append({
                "row_label": f"{label}: {stat}",
                "observed": vals["observed"],
                "null_mean": vals["null_mean"],
                "null_std": vals["null_std"],
                "p": vals["p_two_sided"],
            })

    if not rows:
        print("  [skip] no null_model_summary files found, skipping slide4 forest plot")
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(df) + 1.5))
    y = np.arange(len(df))[::-1]

    ax.errorbar(df["null_mean"], y, xerr=1.96 * df["null_std"], fmt="o",
               color="lightgray", ecolor="lightgray", elinewidth=6, capsize=0,
               markersize=0, label="permutation null (95% range)")
    ax.scatter(df["null_mean"], y, color="gray", s=20, zorder=3)
    ax.scatter(df["observed"], y, color=COLOR_CORE, s=60, marker="D", zorder=4,
              label="observed")

    for yi, p in zip(y, df["p"]):
        flag = "  *" if p < 0.05 else ""
        ax.text(ax.get_xlim()[1] * 1.02 if ax.get_xlim()[1] > 0 else 0.02, yi,
               f"p={p:.2f}{flag}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["row_label"], fontsize=9)
    ax.set_xlabel("value")
    ax.set_title("Observed vs. permutation-null range (degree-preserving rewiring)")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out_path = out_dir / "slide4_forest_plot.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def make_slide4_silhouette_example(with_ppi_dir: Path, out_dir: Path, label: str = "with_ppi"):
    """One clean example: observed silhouette marked against the permutation-null
    CSV of raw replicate values (step8f DOES save these -- null_permutation_results_*.csv)."""
    perm_csv = with_ppi_dir / "diagnostics" / f"null_permutation_results_{label}.csv"
    summary_path = with_ppi_dir / "diagnostics" / f"null_model_summary_{label}.json"

    if not perm_csv.exists():
        _warn_missing(perm_csv)
        return
    if not summary_path.exists():
        _warn_missing(summary_path)
        return

    perm_df = pd.read_csv(perm_csv)
    with open(summary_path) as f:
        s = json.load(f)
    if "observed" not in s or "silhouette" not in s.get("observed", {}):
        print(f"  [skip] {summary_path} has no observed.silhouette field, skipping slide4b")
        return
    observed = s["observed"]["silhouette"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(perm_df["silhouette"], bins=25, color=COLOR_PERIPHERY, alpha=0.7,
           label="permutation null (100 raw replicates)")
    ax.axvline(observed, color=COLOR_CORE, linewidth=2, label=f"observed = {observed:.3f}")
    ax.set_xlabel("silhouette")
    ax.set_ylabel("count")
    ax.set_title(f"{label}: silhouette, observed vs. permutation null")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "slide4_silhouette_null.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# SLIDE 5 — seed vs non-seed degree distributions + illustrative subgraph
# ---------------------------------------------------------------------------
def _celltype_of(node: str) -> str:
    return node.rsplit("::", 1)[0] if "::" in node else "unknown"


def _gene_of(node: str) -> str:
    return node.rsplit("::", 1)[-1] if "::" in node else node


def make_slide5_degree_distributions(ref_graph_path: Path, seed_genes_csv: Path, out_dir: Path,
                                      exclude_ppi_layer: bool = True):
    if not ref_graph_path.exists():
        _warn_missing(ref_graph_path)
        return
    if not seed_genes_csv.exists():
        _warn_missing(seed_genes_csv)
        return

    with open(ref_graph_path, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip() for n in G.nodes()})

    if exclude_ppi_layer:
        to_remove = [n for n in G.nodes() if _celltype_of(n).upper() == "GLOBAL_PPI"]
        G = G.subgraph([n for n in G.nodes() if n not in set(to_remove)]).copy()

    seed_df = pd.read_csv(seed_genes_csv)
    gene_col = "gene" if "gene" in seed_df.columns else seed_df.columns[0]
    seed_set = set(seed_df[gene_col].astype(str).str.strip().str.upper())

    degrees = dict(G.degree())
    seed_degrees = [d for n, d in degrees.items() if _gene_of(n).upper() in seed_set]
    nonseed_degrees = [d for n, d in degrees.items() if _gene_of(n).upper() not in seed_set]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.arange(0, max(max(seed_degrees, default=1), max(nonseed_degrees, default=1)) + 2)
    ax.hist(nonseed_degrees, bins=bins, alpha=0.6, color=COLOR_NONSEED, label="non-seed", density=True)
    ax.hist(seed_degrees, bins=bins, alpha=0.6, color=COLOR_SEED, label="seed", density=True)
    ax.set_yscale("log")
    ax.set_xlabel("degree")
    ax.set_ylabel("density (log scale)")
    ax.set_title("Seed vs. non-seed degree distribution (no_ppi)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "slide5_degree_distributions.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def make_slide5_illustrative_subgraph(no_ppi_dir: Path, ref_graph_path: Path, seed_genes_csv: Path,
                                       out_dir: Path, n_nodes: int = 60):
    """Small, deliberately simplified cartoon: samples a connected patch of the graph
    around a mix of seed and non-seed genes, colors seeds, draws with spring layout.
    NOT a claim about the full module -- explicitly illustrative."""
    clusters_csv = no_ppi_dir / "spectral_clusters.csv"
    if not clusters_csv.exists() or not ref_graph_path.exists() or not seed_genes_csv.exists():
        for p in (clusters_csv, ref_graph_path, seed_genes_csv):
            if not p.exists():
                _warn_missing(p)
        return

    with open(ref_graph_path, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip() for n in G.nodes()})
    to_remove = [n for n in G.nodes() if _celltype_of(n).upper() == "GLOBAL_PPI"]
    G = G.subgraph([n for n in G.nodes() if n not in set(to_remove)]).copy()

    seed_df = pd.read_csv(seed_genes_csv)
    gene_col = "gene" if "gene" in seed_df.columns else seed_df.columns[0]
    seed_set = set(seed_df[gene_col].astype(str).str.strip().str.upper())

    # Take the largest connected component, then sample a small connected patch
    # around a random high-degree "core-ish" node so the picture actually shows structure.
    largest_cc = max(nx.connected_components(G), key=len)
    sub_G_full = G.subgraph(largest_cc)
    degrees = dict(sub_G_full.degree())
    seed_nodes_in_cc = [n for n in largest_cc if _gene_of(n).upper() in seed_set]
    if not seed_nodes_in_cc:
        print("  [skip] no seed genes found in largest connected component, skipping subgraph figure")
        return

    center = max(seed_nodes_in_cc, key=lambda n: degrees.get(n, 0))
    # BFS out from a decently-connected hub near that seed to build a small illustrative patch
    hub_candidates = sorted(largest_cc, key=lambda n: -degrees.get(n, 0))[:5]
    hub = hub_candidates[0]
    patch_nodes = set([hub])
    frontier = [hub]
    while len(patch_nodes) < n_nodes and frontier:
        nxt = []
        for node in frontier:
            for nb in sub_G_full.neighbors(node):
                if nb not in patch_nodes:
                    patch_nodes.add(nb)
                    nxt.append(nb)
                if len(patch_nodes) >= n_nodes:
                    break
            if len(patch_nodes) >= n_nodes:
                break
        frontier = nxt

    sub_G = G.subgraph(patch_nodes)
    node_colors = [COLOR_SEED if _gene_of(n).upper() in seed_set else COLOR_NONSEED
                  for n in sub_G.nodes()]
    node_sizes = [80 if _gene_of(n).upper() in seed_set else 40 for n in sub_G.nodes()]

    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(sub_G, seed=42, k=0.35)
    nx.draw_networkx_edges(sub_G, pos, ax=ax, alpha=0.25, width=0.6)
    nx.draw_networkx_nodes(sub_G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                          alpha=0.9, linewidths=0)
    ax.scatter([], [], color=COLOR_SEED, label="seed gene", s=60)
    ax.scatter([], [], color=COLOR_NONSEED, label="non-seed gene", s=40)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Illustrative patch of the module graph\n(simplified subsample, not the full module)")
    ax.axis("off")

    fig.tight_layout()
    out_path = out_dir / "slide5_illustrative_subgraph.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# SLIDE 6 — established / not-yet-established summary table
# ---------------------------------------------------------------------------
def make_slide6_established_table(out_dir: Path):
    established = [
        "No disconnected-component artifact (ARI \u2248 0 for both conditions)",
        "Embedding degeneracy diagnosed and corrected (row-normalization)",
        "Seed enrichment (no_ppi) robust to core/periphery definition choice",
        "Seed enrichment (no_ppi) robust to k=2 vs. k=3",
    ]
    not_established = [
        "Module-specific vs. genome-wide seed peripherality (not yet disentangled)",
        "Partition-level non-randomness (step8f: not supported vs. permutation null)",
        "Fine-grained, HiDeF-scale core/periphery resolution (currently 3 large clusters)",
    ]

    fig, ax = plt.subplots(figsize=(11, 0.6 * max(len(established), len(not_established)) + 1.8))
    ax.axis("off")

    col_x = [0.02, 0.52]
    headers = ["Established", "Not yet established"]
    header_colors = ["#27ae60", "#c0392b"]

    for x, header, color in zip(col_x, headers, header_colors):
        ax.text(x, 1.0, header, fontsize=14, fontweight="bold", color=color, va="top")

    for col, items in zip(col_x, [established, not_established]):
        for i, item in enumerate(items):
            y = 0.85 - i * (0.85 / max(len(established), len(not_established)) * 0.9)
            ax.text(col, y, f"\u2022 {item}", fontsize=10.5, va="top", wrap=True)

    fig.tight_layout()
    out_path = out_dir / "slide6_established_table.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-ppi-dir", required=True,
                     help="e.g. 08_spectral_analysis/module_with_ppi")
    ap.add_argument("--no-ppi-dir", required=True,
                     help="e.g. 08_spectral_analysis/module_no_ppi")
    ap.add_argument("--ref-graph", default=None,
                     help="module_graph.pkl, needed for slide5 figures. Optional -- those "
                          "two figures are skipped if omitted.")
    ap.add_argument("--seed-genes-csv", default=None,
                     help="seed_genes.csv, needed for slide5 figures. Optional.")
    ap.add_argument("--out-dir", default="08_spectral_analysis/slide_figures")
    ap.add_argument("--skip", nargs="*", default=[],
                     choices=["slide2", "slide2b", "slide3", "slide3b", "slide4", "slide4b",
                              "slide5", "slide5b", "slide6"],
                     help="Skip specific figures by short name, e.g. --skip slide5b "
                          "(the subgraph figure is the slowest/most fragile one).")
    args = ap.parse_args()

    with_ppi_dir = Path(args.with_ppi_dir)
    no_ppi_dir = Path(args.no_ppi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}\n")

    if "slide2" not in args.skip:
        print("Slide 2: k-selection panels")
        make_slide2_k_selection(with_ppi_dir, no_ppi_dir, out_dir)
    if "slide2b" not in args.skip:
        print("Slide 2 (optional): embedding scatter")
        make_slide2_embedding_scatter(with_ppi_dir, no_ppi_dir, out_dir)

    if "slide3" not in args.skip:
        print("\nSlide 3: enrichment bar chart")
        make_slide3_enrichment_bars(with_ppi_dir, no_ppi_dir, out_dir)
    if "slide3b" not in args.skip:
        print("Slide 3: null histogram")
        make_slide3_null_histogram(no_ppi_dir, out_dir)

    if "slide4" not in args.skip:
        print("\nSlide 4: forest plot")
        make_slide4_forest_plot(with_ppi_dir, no_ppi_dir, out_dir)
    if "slide4b" not in args.skip:
        print("Slide 4 (optional): silhouette null example")
        make_slide4_silhouette_example(with_ppi_dir, out_dir, label="with_ppi")

    if args.ref_graph and args.seed_genes_csv:
        if "slide5" not in args.skip:
            print("\nSlide 5: degree distributions")
            make_slide5_degree_distributions(Path(args.ref_graph), Path(args.seed_genes_csv), out_dir)
        if "slide5b" not in args.skip:
            print("Slide 5 (optional): illustrative subgraph")
            make_slide5_illustrative_subgraph(no_ppi_dir, Path(args.ref_graph),
                                              Path(args.seed_genes_csv), out_dir)
    else:
        print("\n--ref-graph / --seed-genes-csv not given -- skipping slide5 figures.")

    if "slide6" not in args.skip:
        print("\nSlide 6: established/not-established table")
        make_slide6_established_table(out_dir)

    print(f"\nDone. Figures saved in {out_dir}")


if __name__ == "__main__":
    main()