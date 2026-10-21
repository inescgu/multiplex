#!/usr/bin/env python3
"""
Step 7c: inter-cell-type (bipartite) connectivity analysis.

Rewritten for the expression-based (pseudobulk) layers. What changed vs the
previous version and why:

1. EDGE CLASSES ARE MUTUALLY EXCLUSIVE NOW. Every non-identity bipartite
   edge gets exactly one class, in priority order:
       scored_lr     - proposed by LIANA consensus / NeuronChat for THIS
                       cell-type pair (scored_lr_edge_provenance.csv)
       flat_lr_only  - only in the flat LR resource (--lr-csv)
       reactome      - shares a Reactome pathway (only meaningful if step2
                       was run with --reactome-csv)
       other         - none of the above (should be ~0)
   The old version incremented both the LR and Reactome counters for edges
   that were both, then summed them, so pair totals were double-counted
   (and fine-level totals disagreed with group-level totals).

2. WHY SPLIT scored_lr FROM flat_lr_only. A flat-LR edge between cell types
   A and B exists whenever the ligand is expressed in A and the receptor in
   B -- with expression-based layers that is mostly a function of layer
   size, not of cell-type-specific signalling. scored_lr edges are the
   cell-type-specific evidence. Heatmaps and the report show both, and the
   directional (sender -> receiver) heatmap uses scored_lr only.

3. global_ppi IS EXCLUDED by default (--exclude-celltypes). It is a
   backbone, not a cell type: identity edges to it carry no biology, and
   step2's flat-LR expansion also attaches LR edges to it. Excluding it keeps
   it from appearing as a giant 'Unclassified' group.

4. IDENTITY EDGES (gene X <-> gene X) are dropped, as before.

5. NAMESPACED FILES ARE HANDLED. If step3b already ran, bipartite files
   contain '<celltype>::GENE'; the prefix is stripped automatically.

6. VECTORISED. No per-edge Python loops (the old iterrows over ~10^5-10^6
   edges was the slow part).

7. NEW OUTPUTS: directional scored-LR sender x receiver heatmap + CSV, and
   per-cell-type layer-specific genes (genes present in only that layer),
   which is where cell-type identity lives once layers are expression-based.

Outputs (--output-dir):
    edges_annotated.csv.gz             one row per non-identity bipartite edge
    edge_counts_by_celltype_pair.csv   per pair: counts per class, total, density
    scored_lr_directional.csv          sender x receiver unique scored-LR edges
    edge_counts_by_subcategory.csv / edge_counts_by_fibertype.csv (+ density)
    lr_edges_by_source.csv / lr_edges_by_ligand_type.csv
    gene_layer_membership.csv          gene -> number of layers, which layers
    layer_specific_genes.csv           genes present in exactly one layer
    top_genes_scored_lr.csv / top_pathways.csv (if --reactome-csv)
    interlayer_analysis_report.txt
    *.png heatmaps / bar charts (unless --no-plots)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# DRG neuron hierarchy (keep in sync with step 7)
# ============================================================================
DRG_HIERARCHY = {
    "A-fiber": {"color": "#1f77b4", "subcategories": {
        "Proprioceptors": ["A-Propr.EPHA3", "A-Propr.HAPLN4", "A-Propr.PCDH8"],
        "A-beta LTMR": ["Ab-LTMR.CCKAR", "Ab-LTMR.ETV1", "Ab-LTMR.LGI2", "Ab-LTMR.NSG2"],
        "A-LTMR": ["A-LTMR.TAC3"],
        "A-peptidergic": ["A-PEP.CHRNA7_SLC18A3", "A-PEP.KIT", "A-PEP.NTRK3_S100A16", "A-PEP.SCGN_ADRA2C"],
    }},
    "C-fiber": {"color": "#ff7f0e", "subcategories": {
        "C-LTMR": ["C-LTMR.CDH9"],
        "C-peptidergic": ["C-PEP.ADORA2B", "C-PEP.TAC1_CACNG5", "C-PEP.TAC1_CHRNA3"],
        "MRGPRX nociceptors": ["C-NP.MRGPRX1_GFRA2", "C-NP.MRGPRX1_MRGPRX4"],
        "C-thermosensitive": ["C-Thermo.RXFP1", "C-Thermo.TRPM8"],
        "C-SST nociceptor": ["C-NP.SST"],
    }},
    "ATF3": {"color": "#d62728", "subcategories": {
        "ATF3 (injured)": ["ATF3"],
    }},
}
UNCLASSIFIED = "Unclassified"
UNCLASSIFIED_COLOR = "#7f7f7f"
CLASSES = ["scored_lr", "flat_lr_only", "reactome", "other"]


def build_hierarchy():
    h = {"ct_to_fiber": {}, "ct_to_subcat": {}, "subcat_to_fiber": {}, "fiber_colors": {},
         "fiber_order": [], "subcat_order": [], "fine_order": [],
         "fine_fiber_bounds": [], "fine_subcat_bounds": [], "subcat_fiber_bounds": []}
    i_fine = i_sub = 0
    for fiber, fdata in DRG_HIERARCHY.items():
        h["fiber_order"].append(fiber)
        h["fiber_colors"][fiber] = fdata["color"]
        for subcat, cts in fdata["subcategories"].items():
            h["subcat_order"].append(subcat)
            h["subcat_to_fiber"][subcat] = fiber
            i_sub += 1
            for ct in cts:
                h["fine_order"].append(ct)
                h["ct_to_fiber"][ct] = fiber
                h["ct_to_subcat"][ct] = subcat
                i_fine += 1
            h["fine_subcat_bounds"].append(i_fine)
        h["fine_fiber_bounds"].append(i_fine)
        h["subcat_fiber_bounds"].append(i_sub)
    for k in ("fine_fiber_bounds", "fine_subcat_bounds", "subcat_fiber_bounds"):
        h[k] = h[k][:-1]
    return h


def sorted_key(a: pd.Series, b: pd.Series, sep: str = "|") -> pd.Series:
    """Order-independent key for two string columns."""
    a = a.astype(str).values
    b = b.astype(str).values
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    return pd.Series(lo, dtype=object) + sep + pd.Series(hi, dtype=object)


def strip_namespace(s: pd.Series) -> pd.Series:
    return s.astype(str).str.split("::").str[-1].str.strip().str.upper()


# ============================================================================
# Loading
# ============================================================================
def load_gene_sets(wdir: Path, exclude: set) -> dict:
    with open(wdir / "celltype_gene_sets.json") as f:
        raw = json.load(f)
    return {ct: set(genes) for ct, genes in raw.items() if ct not in exclude}


def load_bipartite(wdir: Path, exclude: set) -> pd.DataFrame:
    manifest_path = wdir / "bipartite_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"ERROR: {manifest_path} not found - run step 2 first.")
    with open(manifest_path) as f:
        manifest = json.load(f)

    frames, n_skipped_files = [], 0
    for m in manifest:
        if m["source"] in exclude or m["target"] in exclude:
            n_skipped_files += 1
            continue
        df = pd.read_csv(wdir / m["file"], sep="\t", header=None, names=["g1", "g2"], dtype=str)
        df["ct1"], df["ct2"] = m["source"], m["target"]
        frames.append(df)
    if not frames:
        raise SystemExit("ERROR: no bipartite files left after exclusions.")

    e = pd.concat(frames, ignore_index=True)
    e["g1"], e["g2"] = strip_namespace(e["g1"]), strip_namespace(e["g2"])
    n_raw = len(e)
    e = e[(e["g1"] != e["g2"]) & (e["g1"] != "") & (e["g2"] != "")]
    e = e.drop_duplicates(subset=["ct1", "ct2", "g1", "g2"]).reset_index(drop=True)
    print(f"Bipartite files: {len(manifest)} total, {n_skipped_files} skipped (excluded cell types)")
    print(f"Edges: {n_raw:,} raw -> {len(e):,} after dropping identity edges / duplicates\n")
    return e


def load_lr_keys(lr_csv: str | None) -> set:
    if not lr_csv:
        return set()
    df = pd.read_csv(lr_csv)
    if "ligand_gname" not in df.columns:
        df = pd.read_csv(lr_csv, sep="\t")
    df = df.dropna(subset=["ligand_gname", "receptor_gname"])
    keys = sorted_key(df["ligand_gname"].str.strip().str.upper(),
                      df["receptor_gname"].str.strip().str.upper())
    print(f"Flat LR resource: {len(set(keys)):,} unique gene pairs")
    return set(keys)


def load_provenance(wdir: Path, override: str | None, exclude: set) -> pd.DataFrame | None:
    path = Path(override) if override else wdir / "scored_lr_edge_provenance.csv"
    if not path.exists():
        print(f"[info] {path} not found - no scored-LR breakdown.\n")
        return None
    p = pd.read_csv(path)
    for c in ("cell_type_source", "cell_type_target"):
        p[c] = p[c].astype(str).str.strip()
    for c in ("ligand", "receptor"):
        p[c] = p[c].astype(str).str.strip().str.upper()
    if "ligand_type" not in p.columns:
        p["ligand_type"] = pd.NA
    p.loc[p["ligand_type"].astype(str).isin(["", "nan", "None", "<NA>"]), "ligand_type"] = pd.NA
    p = p[~p["cell_type_source"].isin(exclude) & ~p["cell_type_target"].isin(exclude)].copy()
    p["key"] = (sorted_key(p["cell_type_source"], p["cell_type_target"]).values + "||" +
                sorted_key(p["ligand"], p["receptor"]).values)
    print(f"Scored-LR provenance: {len(p):,} rows; sources: {sorted(p['source'].unique())}")
    return p


def load_pathway_map(reactome_csv: str | None, uniprot_map: str | None, universe: set) -> dict:
    if not reactome_csv:
        return {}
    head = pd.read_csv(reactome_csv, nrows=5)
    sym = {"symbol1", "symbol2"}.issubset(head.columns)
    cols = ["symbol1", "symbol2"] if sym else ["uniprot1", "uniprot2"]
    df = pd.read_csv(reactome_csv, usecols=cols + ["common_pathway"], low_memory=False)
    df = df.dropna(subset=["common_pathway"])
    a = df[cols[0]].astype(str).str.strip().str.upper()
    b = df[cols[1]].astype(str).str.strip().str.upper()
    if not sym and uniprot_map:
        u = pd.read_csv(uniprot_map)
        lut = {str(k).upper(): str(v).upper() for k, v in zip(u["uniprotid"], u["symbol"])}
        a, b = a.map(lut).fillna(a), b.map(lut).fillna(b)
    ok = (a != b) & a.isin(universe) & b.isin(universe)
    keys = sorted_key(a[ok].reset_index(drop=True), b[ok].reset_index(drop=True))
    pw = df.loc[ok, "common_pathway"].astype(str).reset_index(drop=True)
    long = pd.DataFrame({"key": keys, "pw": pw})
    long = long.assign(pw=long["pw"].str.split("|")).explode("pw")
    long["pw"] = long["pw"].str.strip()
    long = long[long["pw"] != ""].drop_duplicates()
    pmap = long.groupby("key")["pw"].apply(lambda x: "; ".join(sorted(x))).to_dict()
    print(f"Reactome map: {len(pmap):,} gene pairs with a shared pathway")
    return pmap


# ============================================================================
# Annotation + counts
# ============================================================================
def annotate_edges(e: pd.DataFrame, lr_keys: set, prov: pd.DataFrame | None, pmap: dict) -> pd.DataFrame:
    e = e.copy()
    e["gene_key"] = sorted_key(e["g1"], e["g2"]).values
    e["key"] = sorted_key(e["ct1"], e["ct2"]).values + "||" + e["gene_key"]

    if prov is not None and len(prov):
        agg = prov.groupby("key").agg(
            lr_sources=("source", lambda x: "; ".join(sorted(set(x)))),
            lr_ligand_types=("ligand_type", lambda x: "; ".join(sorted(set(x.dropna())))),
        ).reset_index()
        e = e.merge(agg, on="key", how="left")
    else:
        e["lr_sources"], e["lr_ligand_types"] = np.nan, np.nan

    in_scored = e["lr_sources"].notna()
    in_flat = e["gene_key"].isin(lr_keys) if lr_keys else pd.Series(False, index=e.index)
    e["pathways"] = e["gene_key"].map(pmap).fillna("") if pmap else ""
    shares_pw = e["pathways"] != ""

    e["in_flat_lr"] = in_flat
    e["shares_pathway"] = shares_pw
    e["edge_class"] = np.select([in_scored, in_flat, shares_pw], CLASSES[:3], default="other")
    e[["lr_sources", "lr_ligand_types"]] = e[["lr_sources", "lr_ligand_types"]].fillna("")
    return e.drop(columns=["key"])


def pair_counts(e: pd.DataFrame, gene_sets: dict) -> pd.DataFrame:
    c = (e.groupby(["ct1", "ct2", "edge_class"]).size()
         .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0).reset_index())
    c = c.rename(columns={k: f"n_{k}" for k in CLASSES})
    c["total"] = c[[f"n_{k}" for k in CLASSES]].sum(axis=1)
    n1 = c["ct1"].map(lambda x: len(gene_sets.get(x, ())))
    n2 = c["ct2"].map(lambda x: len(gene_sets.get(x, ())))
    c["n_genes_1"], c["n_genes_2"] = n1, n2
    possible = (n1 * n2).replace(0, np.nan)
    c["density_total_per_1e6"] = 1e6 * c["total"] / possible
    c["density_scored_per_1e6"] = 1e6 * c["n_scored_lr"] / possible
    return c.sort_values("total", ascending=False).reset_index(drop=True)


def group_rollup(pairs: pd.DataFrame, lookup: dict, gene_sets: dict) -> pd.DataFrame:
    d = pairs.copy()
    d["grp1"] = d["ct1"].map(lambda x: lookup.get(x, UNCLASSIFIED))
    d["grp2"] = d["ct2"].map(lambda x: lookup.get(x, UNCLASSIFIED))
    d["group_1"] = np.where(d["grp1"] <= d["grp2"], d["grp1"], d["grp2"])
    d["group_2"] = np.where(d["grp1"] <= d["grp2"], d["grp2"], d["grp1"])
    cols = [f"n_{k}" for k in CLASSES] + ["total"]
    g = d.groupby(["group_1", "group_2"])[cols].sum().reset_index()

    pools = {}
    for ct, genes in gene_sets.items():
        pools.setdefault(lookup.get(ct, UNCLASSIFIED), set()).update(genes)
    sizes = {k: len(v) for k, v in pools.items()}
    possible = (g["group_1"].map(sizes) * g["group_2"].map(sizes)).replace(0, np.nan)
    g["density_total_per_1e6"] = 1e6 * g["total"] / possible
    g["density_scored_per_1e6"] = 1e6 * g["n_scored_lr"] / possible
    return g.sort_values("total", ascending=False).reset_index(drop=True)


def directional_scored(prov: pd.DataFrame | None) -> pd.DataFrame:
    if prov is None or prov.empty:
        return pd.DataFrame()
    u = prov.drop_duplicates(subset=["cell_type_source", "cell_type_target", "ligand", "receptor"])
    return pd.crosstab(u["cell_type_source"], u["cell_type_target"])


def gene_membership(gene_sets: dict) -> pd.DataFrame:
    rows = {}
    for ct, genes in gene_sets.items():
        for g in genes:
            rows.setdefault(g, []).append(ct)
    df = pd.DataFrame({"gene": list(rows), "n_layers": [len(v) for v in rows.values()],
                       "layers": ["|".join(sorted(v)) for v in rows.values()]})
    return df.sort_values(["n_layers", "gene"]).reset_index(drop=True)


# ============================================================================
# Plotting
# ============================================================================
def symmetric_matrix(df, c1, c2, order, value):
    mat = pd.DataFrame(0.0, index=order, columns=order)
    for a, b, v in zip(df[c1], df[c2], df[value]):
        if a in mat.index and b in mat.index and pd.notna(v):
            mat.loc[a, b] += v
            if a != b:
                mat.loc[b, a] += v
    return mat


def plot_heatmap(mat, path, title, tick_colors, major=None, minor=None,
                 cbar="Edges", fmt=".0f", xlabel=None, ylabel=None):
    if mat.empty or np.nansum(mat.values) == 0:
        print(f"  skip {path.name}: no data")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    n = max(mat.shape)
    size = max(6, min(0.55 * n + 3, 22))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(mat, cmap="rocket_r", annot=n <= 18, fmt=fmt, annot_kws={"size": 7},
                linewidths=0.4, linecolor="white", cbar_kws={"label": cbar}, square=True, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_color(tick_colors.get(lab.get_text(), "black"))
        lab.set_fontsize(8)
    for b in minor or []:
        ax.axhline(b, color="grey", lw=0.9, ls="--")
        ax.axvline(b, color="grey", lw=0.9, ls="--")
    for b in major or []:
        ax.axhline(b, color="black", lw=2.2)
        ax.axvline(b, color="black", lw=2.2)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_bar(series, path, title, ylabel, color):
    if series is None or series.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(series) + 2), 5))
    ax.bar(range(len(series)), series.values, color=color)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(series.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_stacked(groups: pd.DataFrame, path, title):
    if groups.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{a}\n<->\n{b}" for a, b in zip(groups["group_1"], groups["group_2"])]
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    colors = {"scored_lr": "#2ca02c", "flat_lr_only": "#98df8a", "reactome": "#ff7f0e", "other": "#d62728"}
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(groups) + 2), 7))
    for k in CLASSES:
        v = groups[f"n_{k}"].values
        if v.sum() == 0:
            continue
        ax.bar(x, v, 0.6, bottom=bottom, label=k, color=colors[k])
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Edges")
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Edge class")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_membership(mem: pd.DataFrame, n_layers: int, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dist = mem["n_layers"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(dist.index, dist.values, color="#1f77b4")
    ax.set_xlabel(f"Number of layers containing the gene (of {n_layers})")
    ax.set_ylabel("Genes")
    ax.set_title("Gene sharing across cell-type layers", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Inter-cell-type connectivity analysis (heterogeneous structure)")
    ap.add_argument("--wdir", required=True)
    ap.add_argument("--lr-csv", default=None, help="Flat LR resource used in step2 (ligand_gname, receptor_gname)")
    ap.add_argument("--reactome-csv", default=None,
                    help="Optional: annotate edges with shared Reactome pathways")
    ap.add_argument("--uniprot-map", default=None)
    ap.add_argument("--scored-lr-provenance-csv", default=None,
                    help="Default: <wdir>/scored_lr_edge_provenance.csv")
    ap.add_argument("--exclude-celltypes", nargs="*", default=["global_ppi"],
                    help="Multiplexes to leave out of the analysis (default: global_ppi). "
                         "Pass the flag with no values to include everything.")
    ap.add_argument("--output-dir", default="interlayer_analysis")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()

    wdir, out = Path(args.wdir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exclude = set(args.exclude_celltypes)
    h = build_hierarchy()

    print("=" * 78 + "\nSTEP 7c: INTER-CELL-TYPE CONNECTIVITY ANALYSIS\n" + "=" * 78)
    print(f"Excluded multiplexes: {sorted(exclude) if exclude else '(none)'}\n")

    gene_sets = load_gene_sets(wdir, exclude)
    edges = load_bipartite(wdir, exclude)
    lr_keys = load_lr_keys(args.lr_csv)
    prov = load_provenance(wdir, args.scored_lr_provenance_csv, exclude)
    universe = set().union(*gene_sets.values()) if gene_sets else set()
    pmap = load_pathway_map(args.reactome_csv, args.uniprot_map, universe)
    print()

    unmapped = sorted(set(gene_sets) - set(h["fine_order"]))
    if unmapped:
        print(f"WARNING: cell types not in DRG_HIERARCHY (grouped as '{UNCLASSIFIED}'): {unmapped}\n")
    missing = sorted(set(h["fine_order"]) - set(gene_sets))
    if missing:
        print(f"NOTE: hierarchy cell types with no layer in this wdir: {missing}\n")

    # ---- annotate + count ----
    e = annotate_edges(edges, lr_keys, prov, pmap)
    e.to_csv(out / "edges_annotated.csv.gz", index=False)
    class_totals = e["edge_class"].value_counts().reindex(CLASSES, fill_value=0)
    print("Edge classes (mutually exclusive):")
    print(class_totals.to_string() + "\n")
    if class_totals["other"] > 0 and not pmap:
        print("NOTE: 'other' edges present and no --reactome-csv given; if step2 used --reactome-csv, "
              "pass the same file here so those edges are labelled 'reactome'.\n")

    pairs = pair_counts(e, gene_sets)
    pairs.to_csv(out / "edge_counts_by_celltype_pair.csv", index=False)

    sub = group_rollup(pairs, h["ct_to_subcat"], gene_sets)
    sub.to_csv(out / "edge_counts_by_subcategory.csv", index=False)
    fib = group_rollup(pairs, h["ct_to_fiber"], gene_sets)
    fib.to_csv(out / "edge_counts_by_fibertype.csv", index=False)

    direc = directional_scored(prov)
    if not direc.empty:
        # full layer order (not just types with edges) so hierarchy boundary lines stay aligned
        order = [c for c in h["fine_order"] if c in gene_sets] + unmapped
        order += sorted(set(direc.index.union(direc.columns)) - set(order))
        direc = direc.reindex(index=order, columns=order, fill_value=0)
        direc.to_csv(out / "scored_lr_directional.csv")

    # ---- LR method / ligand-type breakdown ----
    by_source = by_ltype = pd.Series(dtype=int)
    if prov is not None and len(prov):
        by_source = prov.drop_duplicates(["key", "source"]).groupby("source").size().sort_values(ascending=False)
        lt = prov.dropna(subset=["ligand_type"])
        by_ltype = lt.drop_duplicates(["key", "ligand_type"]).groupby("ligand_type").size().sort_values(ascending=False)
        by_source.rename("n_unique_edges").to_csv(out / "lr_edges_by_source.csv")
        by_ltype.rename("n_unique_edges").to_csv(out / "lr_edges_by_ligand_type.csv")

    # ---- gene membership / layer-specific genes ----
    mem = gene_membership(gene_sets)
    mem.to_csv(out / "gene_layer_membership.csv", index=False)
    specific = mem[mem["n_layers"] == 1].rename(columns={"layers": "cell_type"})[["cell_type", "gene"]]
    specific.sort_values(["cell_type", "gene"]).to_csv(out / "layer_specific_genes.csv", index=False)
    n_layers = len(gene_sets)

    # ---- top genes / pathways ----
    sc = e[e["edge_class"] == "scored_lr"]
    if len(sc):
        long = pd.concat([sc[["g1", "ct1", "ct2"]].rename(columns={"g1": "gene"}),
                          sc[["g2", "ct1", "ct2"]].rename(columns={"g2": "gene"})])
        top_genes = (long.drop_duplicates().groupby("gene").size()
                     .sort_values(ascending=False).rename("n_celltype_pairs"))
    else:
        top_genes = pd.Series(dtype=int, name="n_celltype_pairs")
    top_genes.to_csv(out / "top_genes_scored_lr.csv")

    top_pw = pd.Series(dtype=int)
    if pmap:
        pw = e[e["pathways"] != ""][["pathways", "ct1", "ct2"]]
        pw = pw.assign(pathways=pw["pathways"].str.split("; ")).explode("pathways")
        top_pw = pw.drop_duplicates().groupby("pathways").size().sort_values(ascending=False)
        top_pw.rename("n_celltype_pairs").to_csv(out / "top_pathways.csv")

    # ---- plots ----
    if not args.no_plots:
        print("Plotting ...")
        fine = [c for c in h["fine_order"] if c in gene_sets] + unmapped
        fine_colors = {c: h["fiber_colors"].get(h["ct_to_fiber"].get(c), UNCLASSIFIED_COLOR) for c in fine}
        bounds_ok = not missing and not unmapped
        fb = h["fine_fiber_bounds"] if bounds_ok else None
        sb = h["fine_subcat_bounds"] if bounds_ok else None

        for value, fname, title, cbar, fmt in [
            ("total", "heatmap_celltype_total.png", "All inter-cell-type edges", "Edges", ".0f"),
            ("n_scored_lr", "heatmap_celltype_scored_lr.png", "Scored LR edges (LIANA/NeuronChat)", "Edges", ".0f"),
            ("n_flat_lr_only", "heatmap_celltype_flat_lr.png",
             "Flat-resource-only LR edges\n(driven by layer gene content)", "Edges", ".0f"),
            ("density_scored_per_1e6", "heatmap_celltype_scored_density.png",
             "Scored LR edge density", "Edges per 10^6 possible gene pairs", ".2f"),
        ]:
            mat = symmetric_matrix(pairs, "ct1", "ct2", fine, value)
            plot_heatmap(mat, out / fname, title, fine_colors, fb, sb, cbar, fmt)

        if not direc.empty:
            dcol = {c: h["fiber_colors"].get(h["ct_to_fiber"].get(c), UNCLASSIFIED_COLOR) for c in direc.index}
            plot_heatmap(direc, out / "heatmap_scored_lr_directional.png",
                         "Scored LR edges, sender (rows) -> receiver (columns)", dcol,
                         fb, sb, "Unique LR edges", ".0f", xlabel="Receiver", ylabel="Sender")

        sub_order = [s for s in h["subcat_order"] if s in set(sub["group_1"]) | set(sub["group_2"])]
        if UNCLASSIFIED in set(sub["group_1"]) | set(sub["group_2"]):
            sub_order.append(UNCLASSIFIED)
        sub_colors = {s: h["fiber_colors"].get(h["subcat_to_fiber"].get(s), UNCLASSIFIED_COLOR) for s in sub_order}
        for value, fname, title, cbar, fmt in [
            ("n_scored_lr", "heatmap_subcategory_scored_lr.png", "Scored LR edges - subcategory", "Edges", ".0f"),
            ("density_scored_per_1e6", "heatmap_subcategory_scored_density.png",
             "Scored LR density - subcategory", "Edges per 10^6 possible gene pairs", ".2f"),
        ]:
            plot_heatmap(symmetric_matrix(sub, "group_1", "group_2", sub_order, value), out / fname,
                         title, sub_colors, None, None, cbar, fmt)

        fib_order = [f for f in h["fiber_order"] if f in set(fib["group_1"]) | set(fib["group_2"])]
        fib_colors = {f: h["fiber_colors"].get(f, UNCLASSIFIED_COLOR) for f in fib_order}
        plot_heatmap(symmetric_matrix(fib, "group_1", "group_2", fib_order, "density_scored_per_1e6"),
                     out / "heatmap_fibertype_scored_density.png", "Scored LR density - fiber type",
                     fib_colors, None, None, "Edges per 10^6 possible gene pairs", ".2f")

        plot_stacked(sub, out / "edgeclass_breakdown_subcategory.png", "Edge classes - subcategory pairs")
        plot_stacked(fib, out / "edgeclass_breakdown_fibertype.png", "Edge classes - fiber-type pairs")
        plot_bar(by_source, out / "lr_edges_by_source.png", "Scored LR edges by method", "Unique edges", "#8172B2")
        plot_bar(by_ltype, out / "lr_edges_by_ligand_type.png", "NeuronChat edges by ligand type",
                 "Unique edges", "#55A868")
        plot_membership(mem, n_layers, out / "gene_layer_membership.png")

    # ---- report ----
    L = ["=" * 78, "INTER-CELL-TYPE CONNECTIVITY REPORT", "=" * 78,
         f"wdir: {wdir}", f"excluded multiplexes: {sorted(exclude) if exclude else 'none'}",
         f"cell-type layers analysed: {n_layers}",
         f"non-identity bipartite edges: {len(e):,} across {len(pairs)} cell-type pairs", "",
         "EDGE CLASSES (mutually exclusive, priority scored_lr > flat_lr_only > reactome > other)"]
    for k in CLASSES:
        L.append(f"  {k:<14} {class_totals[k]:>10,}  ({100 * class_totals[k] / max(len(e), 1):5.1f}%)")
    L += ["  note: flat_lr_only edges exist wherever ligand and receptor are both in the two layers;",
          "  with expression-based layers they mostly track layer size. scored_lr is the",
          "  cell-type-specific signal.", ""]

    L.append(f"TOP {min(10, len(pairs))} CELL-TYPE PAIRS BY SCORED-LR EDGES")
    for _, r in pairs.sort_values("n_scored_lr", ascending=False).head(10).iterrows():
        L.append(f"  {r.ct1:<24} <-> {r.ct2:<24} scored {r.n_scored_lr:>6,} | flat {r.n_flat_lr_only:>6,} "
                 f"| density {r.density_scored_per_1e6:7.2f}/1e6")
    L.append("")

    if not direc.empty:
        L.append("SCORED LR, OUTGOING (sender) vs INCOMING (receiver) PER CELL TYPE")
        outgoing, incoming = direc.sum(axis=1), direc.sum(axis=0)
        for ct in direc.index:
            L.append(f"  {ct:<24} out {int(outgoing.get(ct, 0)):>6,} | in {int(incoming.get(ct, 0)):>6,}")
        L.append("")

    L.append("FIBER-TYPE LEVEL")
    for _, r in fib.iterrows():
        L.append(f"  {r.group_1:<10} <-> {r.group_2:<10} total {r.total:>8,} | scored {r.n_scored_lr:>7,} "
                 f"| scored density {r.density_scored_per_1e6:7.2f}/1e6")
    L.append("")

    if len(by_source):
        L.append("SCORED LR EDGES BY METHOD (unique cell-type-pair x gene-pair)")
        L += [f"  {k:<20} {v:>8,}" for k, v in by_source.items()]
        L.append("")
    if len(by_ltype):
        L.append("NEURONCHAT EDGES BY LIGAND TYPE")
        L += [f"  {k:<28} {v:>8,}" for k, v in by_ltype.items()]
        L.append("")

    L.append("GENE SHARING ACROSS LAYERS")
    L.append(f"  unique genes: {len(mem):,} | in all {n_layers} layers: {(mem.n_layers == n_layers).sum():,} "
             f"| in exactly 1 layer: {(mem.n_layers == 1).sum():,}")
    spec_counts = specific["cell_type"].value_counts()
    for ct in [c for c in h["fine_order"] if c in gene_sets] + unmapped:
        L.append(f"    layer-specific genes in {ct:<24} {int(spec_counts.get(ct, 0)):>5,}")
    L.append("")

    if len(top_genes):
        L.append(f"TOP {args.top_n} GENES BY NUMBER OF CELL-TYPE PAIRS WITH A SCORED-LR EDGE")
        L += [f"  {i:>2}. {g:<16} {n:>4}" for i, (g, n) in enumerate(top_genes.head(args.top_n).items(), 1)]
        L.append("")
    if len(top_pw):
        L.append(f"TOP {args.top_n} SHARED PATHWAYS BY CELL-TYPE-PAIR INVOLVEMENT")
        L += [f"  {i:>2}. {p[:60]:<60} {n:>4}" for i, (p, n) in enumerate(top_pw.head(args.top_n).items(), 1)]
        L.append("")

    text = "\n".join(L)
    (out / "interlayer_analysis_report.txt").write_text(text + "\n")
    print(text)
    print(f"\nAll outputs in {out.resolve()}")


if __name__ == "__main__":
    main()