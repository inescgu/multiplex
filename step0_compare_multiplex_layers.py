#!/usr/bin/env python3
"""
Compare two multiplex-layer builds (e.g. old vs new marker sets).

Expects each directory to look like the output of build_multiplex_from_markers:
    <dir>/layers/<cell_type>_layer.pkl        (networkx graphs)
    <dir>/markers_with_layers.csv             (optional, but very informative)

Usage:
    python compare_multiplex_layers.py \
        --old-dir multiplex_markers_OLD \
        --new-dir multiplex_markers_NEW \
        --output-dir layer_comparison

Optional:
    --name-map map.csv     CSV with columns old,new (layer file-stem names) for
                           layers whose names don't match automatically
    --old-markers / --new-markers   override the markers CSV location

Outputs (in --output-dir):
    layer_comparison.csv      per-layer node/edge/seed overlap
    cross_jaccard.csv         old layers x new layers node-set Jaccard matrix
    best_match.csv            best new match for every old layer (catches label swaps)
    gene_diffs.csv            genes present in only one build, per layer
    marker_comparison.csv     upstream (pre-PPI) marker-set comparison + quality diagnostics
    summary.txt               plain-text summary
    cross_jaccard.png         heatmap (if matplotlib is installed)
"""

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def norm(name):
    """Normalise a cell-type name so 'Aβ-LTMR.NSG2', 'Ab-LTMR.NSG2' and
    'C-NP.MRGPRX1/GFRA2' vs 'C-NP.MRGPRX1_GFRA2' all compare equal."""
    s = str(name).lower().replace("β", "b").replace("δ", "d")
    return re.sub(r"[^a-z0-9]", "", s)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def load_layers(directory):
    layers = {}
    for f in sorted(Path(directory, "layers").glob("*_layer.pkl")):
        stem = f.name[: -len("_layer.pkl")]
        with open(f, "rb") as fh:
            layers[stem] = pickle.load(fh)
    if not layers:
        raise SystemExit(f"No *_layer.pkl files found in {directory}/layers")
    return layers


def edge_set(g):
    return {frozenset(e) for e in g.edges()}


def seed_set(g):
    return {n for n, d in g.nodes(data=True) if d.get("is_seed")}


def pair_layers(old, new, name_map=None):
    manual = {}
    if name_map:
        m = pd.read_csv(name_map)
        manual = dict(zip(m["old"], m["new"]))
    new_by_norm = {}
    for n in new:
        new_by_norm.setdefault(norm(n), n)

    pairs, used, unmatched_old = [], set(), []
    for o in old:
        n = manual.get(o)
        if n is not None:
            n = n.replace("/", "_")
        else:
            n = new_by_norm.get(norm(o))
        if n in new and n not in used:
            pairs.append((o, n))
            used.add(n)
        else:
            unmatched_old.append(o)
    unmatched_new = [n for n in new if n not in used]
    return pairs, unmatched_old, unmatched_new


def load_markers(path):
    if path is None or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    gcol = next((c for c in ["gene", "Gene", "gene_id", "symbol", "Symbol"] if c in df.columns), None)
    ccol = next((c for c in ["cell_type", "Cell_Type", "celltype", "CellType", "cluster"] if c in df.columns), None)
    if gcol is None or ccol is None:
        print(f"  WARNING: could not find gene/cell_type columns in {path}")
        return None
    df["_gene"] = df[gcol].astype(str).str.upper().str.strip()
    df["_key"] = df[ccol].astype(str).map(norm)
    return df


def marker_quality(df):
    """Fraction of rows that are downregulated / non-significant."""
    out = {"n_rows": len(df)}
    if "avg_log2FC" in df.columns:
        out["pct_negative_log2FC"] = round(float(100 * (df["avg_log2FC"] < 0).mean()), 1)
    if "p_val_adj" in df.columns:
        out["pct_padj_ge_0.05"] = round(float(100 * (df["p_val_adj"] >= 0.05).mean()), 1)
    return out


def layers_per_gene(layers):
    counts = {}
    for g in layers.values():
        for n in g.nodes():
            counts[n] = counts.get(n, 0) + 1
    return pd.Series(counts, dtype=float)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--output-dir", default="layer_comparison")
    ap.add_argument("--name-map", default=None)
    ap.add_argument("--old-markers", default=None)
    ap.add_argument("--new-markers", default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    old = load_layers(args.old_dir)
    new = load_layers(args.new_dir)
    pairs, unmatched_old, unmatched_new = pair_layers(old, new, args.name_map)

    say("=" * 70)
    say("LAYER SET OVERVIEW")
    say("=" * 70)
    say(f"Old layers: {len(old)}   New layers: {len(new)}   Matched by name: {len(pairs)}")
    if unmatched_old:
        say(f"Old-only layers (no name match): {unmatched_old}")
    if unmatched_new:
        say(f"New-only layers (no name match): {unmatched_new}")
    if unmatched_old or unmatched_new:
        say("  -> check best_match.csv for likely renamed layers, or supply --name-map")

    # ---- per-layer comparison ----
    rows, diff_rows = [], []
    old_mk = load_markers(args.old_markers or Path(args.old_dir, "markers_with_layers.csv"))
    new_mk = load_markers(args.new_markers or Path(args.new_dir, "markers_with_layers.csv"))

    fc_old = fc_new = {}
    if old_mk is not None and "avg_log2FC" in old_mk.columns:
        fc_old = old_mk.groupby(["_key", "_gene"])["avg_log2FC"].max().to_dict()
    if new_mk is not None and "avg_log2FC" in new_mk.columns:
        fc_new = new_mk.groupby(["_key", "_gene"])["avg_log2FC"].max().to_dict()

    for o, n in pairs:
        go, gn = old[o], new[n]
        no, nn = set(go.nodes()), set(gn.nodes())
        shared = no & nn
        eo, en = edge_set(go), edge_set(gn)
        eo_sh = {e for e in eo if e <= shared}
        en_sh = {e for e in en if e <= shared}
        so, sn = seed_set(go), seed_set(gn)
        rows.append({
            "old_layer": o, "new_layer": n,
            "nodes_old": len(no), "nodes_new": len(nn), "nodes_shared": len(shared),
            "nodes_old_only": len(no - nn), "nodes_new_only": len(nn - no),
            "node_jaccard": jaccard(no, nn),
            "edges_old": len(eo), "edges_new": len(en), "edge_jaccard": jaccard(eo, en),
            # if the PPI is unchanged these two should both be 0
            "edges_among_shared_old_only": len(eo_sh - en_sh),
            "edges_among_shared_new_only": len(en_sh - eo_sh),
            "seeds_old": len(so), "seeds_new": len(sn),
            "seeds_lost": ";".join(sorted(so - sn)), "seeds_gained": ";".join(sorted(sn - so)),
        })
        for g in sorted(no - nn):
            diff_rows.append({"layer": n, "gene": g, "status": "old_only",
                              "log2FC_in_old_markers": fc_old.get((norm(o), g))})
        for g in sorted(nn - no):
            diff_rows.append({"layer": n, "gene": g, "status": "new_only",
                              "log2FC_in_new_markers": fc_new.get((norm(n), g))})

    comp = pd.DataFrame(rows)
    comp.to_csv(out / "layer_comparison.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(out / "gene_diffs.csv", index=False)

    say("")
    say("=" * 70)
    say("PER-LAYER OVERLAP (matched layers)")
    say("=" * 70)
    if not comp.empty:
        say(f"Mean node Jaccard: {comp['node_jaccard'].mean():.3f}   "
            f"median: {comp['node_jaccard'].median():.3f}")
        say(f"Mean edge Jaccard: {comp['edge_jaccard'].mean():.3f}")
        say(f"Total nodes  old: {comp['nodes_old'].sum():,}   new: {comp['nodes_new'].sum():,}")
        say(f"Total edges  old: {comp['edges_old'].sum():,}   new: {comp['edges_new'].sum():,}")
        ppi_changed = ((comp["edges_among_shared_old_only"] + comp["edges_among_shared_new_only"]) > 0).sum()
        if ppi_changed:
            say(f"NOTE: {ppi_changed} layers have different edges among *shared* genes -> "
                f"the base PPI probably changed between builds.")
        else:
            say("Edges among shared genes are identical -> same base PPI; differences come from marker sets.")
        say("Least similar layers (node Jaccard):")
        for _, r in comp.nsmallest(5, "node_jaccard").iterrows():
            say(f"   {r['new_layer']:<35} J={r['node_jaccard']:.3f}  "
                f"old={r['nodes_old']}  new={r['nodes_new']}")

    # ---- cross-matrix / best match ----
    cross = pd.DataFrame(
        {n: {o: jaccard(old[o].nodes(), new[n].nodes()) for o in old} for n in new}
    )
    cross.to_csv(out / "cross_jaccard.csv")
    best = []
    matched_lookup = dict(pairs)
    for o in cross.index:
        bn = cross.loc[o].idxmax()
        best.append({"old_layer": o, "best_new_layer": bn, "jaccard": cross.loc[o, bn],
                     "name_matched_new_layer": matched_lookup.get(o),
                     "agrees_with_name_match": matched_lookup.get(o) == bn})
    best_df = pd.DataFrame(best)
    best_df.to_csv(out / "best_match.csv", index=False)
    disagree = best_df[(~best_df["agrees_with_name_match"]) & best_df["name_matched_new_layer"].notna()]
    say("")
    say(f"Layers whose best node-overlap partner is NOT their name match: {len(disagree)}")
    for _, r in disagree.iterrows():
        say(f"   old {r['old_layer']}  -> best new {r['best_new_layer']} (J={r['jaccard']:.3f}) "
            f"vs name match {r['name_matched_new_layer']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(new) + 3), max(5, 0.35 * len(old) + 2)))
        im = ax.imshow(cross.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(cross.columns)))
        ax.set_xticklabels(cross.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(cross.index)))
        ax.set_yticklabels(cross.index, fontsize=7)
        ax.set_xlabel("new layers")
        ax.set_ylabel("old layers")
        fig.colorbar(im, label="node Jaccard")
        fig.tight_layout()
        fig.savefig(out / "cross_jaccard.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # matplotlib optional
        say(f"(heatmap skipped: {e})")

    # ---- global gene-promiscuity ----
    say("")
    say("=" * 70)
    say("GENE PROMISCUITY (how many layers each gene appears in)")
    say("=" * 70)
    for label, layers in [("old", old), ("new", new)]:
        s = layers_per_gene(layers)
        say(f"{label}: {len(s):,} unique genes; mean layers/gene {s.mean():.2f}, "
            f"median {s.median():.0f}; genes in >= half of layers: {(s >= len(layers) / 2).sum():,}")

    # ---- seed genes ----
    if not comp.empty and (comp["seeds_old"].sum() or comp["seeds_new"].sum()):
        say("")
        say("=" * 70)
        say("SEED GENES")
        say("=" * 70)
        say(f"Seed-gene hits summed over layers  old: {comp['seeds_old'].sum()}   new: {comp['seeds_new'].sum()}")
        lost = comp[comp["seeds_lost"] != ""]
        gained = comp[comp["seeds_gained"] != ""]
        say(f"Layers that lost seeds: {len(lost)}   gained seeds: {len(gained)} (see layer_comparison.csv)")

    # ---- upstream marker comparison ----
    if old_mk is not None and new_mk is not None:
        say("")
        say("=" * 70)
        say("UPSTREAM MARKER SETS (before PPI filtering)")
        say("=" * 70)
        qo, qn = marker_quality(old_mk), marker_quality(new_mk)
        say(f"Old markers: {qo}")
        say(f"New markers: {qn}")
        if qo.get("pct_negative_log2FC", 0) > 0:
            say(f"  -> {qo['pct_negative_log2FC']:.1f}% of OLD marker rows are downregulated genes.")
        mrows = []
        for o, n in pairs:
            go = set(old_mk.loc[old_mk["_key"] == norm(o), "_gene"])
            gn = set(new_mk.loc[new_mk["_key"] == norm(n), "_gene"])
            mrows.append({"old_layer": o, "new_layer": n,
                          "markers_old": len(go), "markers_new": len(gn),
                          "shared": len(go & gn), "jaccard": jaccard(go, gn),
                          "frac_of_new_in_old": len(go & gn) / len(gn) if gn else float("nan"),
                          "frac_of_old_in_new": len(go & gn) / len(go) if go else float("nan")})
        mdf = pd.DataFrame(mrows)
        mdf.to_csv(out / "marker_comparison.csv", index=False)
        if not mdf.empty:
            say(f"Mean marker-set Jaccard: {mdf['jaccard'].mean():.3f}; "
                f"mean fraction of NEW markers already in OLD: {mdf['frac_of_new_in_old'].mean():.3f}")
    else:
        say("")
        say("(markers_with_layers.csv not found in one or both dirs; skipped upstream marker comparison)")

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote outputs to {out}/")


if __name__ == "__main__":
    main()