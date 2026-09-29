#!/usr/bin/env python3
"""
Step 2 (replaces old step3's inter-layer half + step6's LR/Reactome
extraction): Build BIPARTITE edge files connecting cell-type multiplexes.

In the old homogeneous design, ligand-receptor and pathway edges got
annotated directly onto one shared graph. In the heterogeneous design they
become bipartite edge files (MultiXrank's `bipartite` config section) —
each file explicitly says "these two multiplexes are connected here", with
a probability of crossing it controlled later by `lamb`/`delta`.

IMPORTANT MultiXrank constraint (confirmed from source, BipartiteAll.py):
a bipartite file can only connect ONE (source, target) multiplex pair per
config entry — if you gave it two files for the same pair, the second
would silently overwrite the first. So LR edges (flat CSV), Reactome
pathway edges, scored/directional LR edges (LIANA + NeuronChat, new),
AND global_ppi identity edges between the same two cell types are all
merged into a single bipartite file here.

Also: a gene name in a bipartite file only "counts" if it already exists as
a node in the corresponding multiplex's own layer file(s). Edges to genes
absent from a cell type's PPI layer are dropped — same behaviour the old
pipeline had via its gene_universe checks, just enforced earlier here.

Requires celltype_gene_sets.json from step 1.

NEW: scored/directional LR edges (LIANA + NeuronChat)
--------------------------------------------------------
step0a_run_liana.R and step0b_run_neuronchat.R each write a long-format
table (ligand, receptor, source, score, cell_type_source, cell_type_target)
to pathway_prep/lr_sources/. Unlike the flat --lr-csv (a resource of
ligand/receptor GENE SYMBOL pairs with no cell-type information, expanded
here to every pair of cell types that happen to express both genes
somewhere), these files already say WHICH cell type is the ligand-sender
and WHICH is the receptor-receiver, per interaction, from actual expression
in the atlas. Pass one or more of these files via --scored-lr-csv.

Two things this requires that the flat-CSV path didn't:
  1. cell_type_source/cell_type_target are raw Atlas_annotation strings
     (can contain "/", spaces, etc.) and must be sanitize_name()'d to match
     the multiplex folder / celltype_gene_sets.json keys step 1 wrote —
     see sanitize_name() below, kept in sync with step1's copy.
  2. LIANA's ligand.complex/receptor.complex and NeuronChat's ligand/
     receptor fields encode multi-subunit complexes joined by "_" (e.g.
     "ITGA4_ITGB1"), which will never match a single-gene PPI node. By
     default (disable with --no-split-complexes) these are split on "_"
     and expanded into every (ligand subunit, receptor subunit) pair
     before the existence/PPI checks below, same convention used
     elsewhere for LR complexes.

Score values across --scored-lr-csv files are NOT on a comparable scale
(LIANA's per-method columns, e.g. logfc_comb vs edge_specificity vs
1-aggregate_rank for consensus, vs NeuronChat's own weight) — bipartite
files here are unweighted presence/absence edge lists anyway (MultiXrank
reads no weight column), so this mostly doesn't matter. --scored-lr-min-
score, if set, is still applied as one global row['score'] >= threshold
filter across every file/method, which is a blunt instrument given the
scale mismatch above — leave it unset (default) unless you've already
picked a threshold that makes sense per-method, or pre-filter the CSVs
yourself before passing them in.

--validate-against-ppi (if passed) is now also applied to scored LR edges,
same as it already was for the flat LR CSV and Reactome edges, using the
same --ppi-pkl reference.

FIX vs previous version: add_cross_celltype_edges() used to bucket edges
under `tuple(sorted((ct_a, ct_b)))` but store the raw (gene_a, gene_b)
tuple without checking that gene_a actually belongs to the FIRST cell type
in that sorted pair — if ct_b happened to sort before ct_a, the edge got
written with the wrong gene in the wrong g1/g2 column relative to the
manifest's source/target. Harmless for LR/Reactome edges as long as
MultiXrank bipartite edges are treated as symmetric (undirected) internally,
but it matters now: scored LR edges are directional (ligand lives in
cell_type_source, receptor in cell_type_target), so the column a gene ends
up in must consistently match the multiplex it was validated against. Both
add_cross_celltype_edges() and the new add_scored_lr_edges() now always
orient (pair_key, edge) together so edge[0] is guaranteed to belong to
pair_key[0]'s gene space and edge[1] to pair_key[1]'s.

NEW: before/after-PPI-check histogram, by source
--------------------------------------------------------
Whenever --ppi-pkl is given (whether or not you also pass
--validate-against-ppi to actually filter edges by it), the script now also
reports, per source — ligand_receptor (your flat --lr-csv), reactome, liana,
neuronchat — how many of the UNIQUE candidate ligand-receptor gene pairs
that source proposes are actually corroborated by an edge in --ppi-pkl.
This is a property of each resource/method itself (is this gene pair a
known PPI edge at all?), computed independently of cell-type multiplex
membership, so liana/neuronchat/the flat table can be compared apples to
apples regardless of how many cell types each one's genes happen to be
expressed in. liana and neuronchat are reported separately (split on the
scored-lr files' 'source' column: rows with source=="neuronchat" vs
everything else) even though both come in via --scored-lr-csv, since that's
the actual comparison you want. Complex ligands/receptors are expanded the
same way (--no-split-complexes) as when building the real edges, since
that's what's actually checked against --ppi-pkl.

Written to (only when --ppi-pkl is given):
    ppi_check_stats.csv
    ppi_check_histogram.png

--validate-against-ppi still independently controls whether edges failing
this check get dropped from the bipartite files themselves — you can pass
--ppi-pkl alone to get the diagnostic without filtering anything.

NEW: scored_lr_edge_provenance.csv (interaction-type tracking)
--------------------------------------------------------------
Previously, once a scored LR edge from --scored-lr-csv made it into a
bipartite/<ct1>__<ct2>.tsv file, it was just a bare (gene1, gene2) pair --
there was no record left anywhere of WHICH method produced it (LIANA's
cellphonedb/natmi/connectome/sca/logfc/liana_consensus, or NeuronChat) or,
for NeuronChat rows, WHICH ligand_type category it belongs to
(Neurotransmitter, Neuropeptide, Gas, Gap junction protein). That
information was silently dropped at exactly this step, so nothing
downstream (step6/step6b included) could ever break a histogram down by
interaction type.

Now, every scored-LR edge that survives cell-type/gene/optional-PPI
filtering (i.e. every edge actually written into a bipartite file) also
gets one row appended to scored_lr_edge_provenance.csv, with columns:
    cell_type_source, cell_type_target, ligand, receptor, source, ligand_type
cell_type_source/cell_type_target are already sanitize_name()'d (matching
the multiplex folder names / node namespace step 2.5 uses), and
ligand/receptor are the individual gene subunits actually checked/added
(post complex-splitting). `source` is the method name from the input file
("neuronchat", "liana_consensus", "cellphonedb", etc). `ligand_type` is
populated only for rows that came from a file with that column (i.e.
NeuronChat's output) and is blank/NaN otherwise.

step6b_module_diagnostics.py reads this file to build interaction-type
histograms, both over ALL candidate scored-LR edges that made it this far
(this file, as-is) and restricted to whichever of those edges actually end
up present in the final disease module (by checking each row's two
"<celltype>::GENE" node names against module_graph.pkl).

Only written when --scored-lr-csv is given.

NEW: unique-gene-pair vs. absolute-edge-count reporting
--------------------------------------------------------
Because genes are namespaced by cell type across bipartite files (the same
gene symbol pair, e.g. (TNF, TNFRSF1A), can be added once per cell-type
pair that happens to express both genes), the total edge count summed
across all bipartite/<ct1>__<ct2>.tsv files ("absolute") is NOT the same
as the number of distinct underlying gene-gene relationships ("unique") —
the former can be inflated by however many cell-type pairs a given gene
pair happens to be eligible for. This was previously invisible: nothing
reported the two side by side.

Now, every edge-adding function also records edges into a per-source
tracker (source_edge_tracker: source_label -> pair_key -> set of edges),
mirroring `buckets` but split out per source/method. Scored-LR edges are
tracked per actual method (row['source'], e.g. "liana_consensus",
"cellphonedb", "neuronchat", ...) rather than lumped as one "scored_lr"
bucket, so methods can be compared directly. After all bipartite files are
written, the script computes, per source and overall:
    absolute_edges     = total (pair_key, edge) entries (i.e. how many
                          bipartite-file lines this source is responsible
                          for, matching what's actually on disk)
    unique_gene_pairs   = distinct gene-symbol pairs, IGNORING which
                          cell-type pair they appear under
    overcounting_ratio  = absolute_edges / unique_gene_pairs
and prints this table, writes it to unique_vs_absolute_stats.csv, and
plots it as two bar charts. The "OVERALL" row uses the final `buckets`
directly (already deduplicated within each cell-type pair across all
sources), so it reflects exactly what was written to the bipartite files.

Written (always, since this needs no extra inputs beyond what's already
built):
    unique_vs_absolute_stats.csv
    chart_unique_vs_absolute_overall.png
    chart_unique_vs_absolute_by_source.png

Outputs (under --wdir):
    bipartite/<ct1>__<ct2>.tsv          (one file per connected cell-type pair)
    bipartite_manifest.json             (file -> source/target/edge counts;
                                          step 3 reads this to build config.yml)
    stats_bipartite_edges.csv
    scored_lr_edge_provenance.csv       (only if --scored-lr-csv given; see above)
    ppi_check_stats.csv                 (only if --ppi-pkl given)
    ppi_check_histogram.png             (only if --ppi-pkl given)
    unique_vs_absolute_stats.csv
    chart_unique_vs_absolute_overall.png
    chart_unique_vs_absolute_by_source.png
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Cell-type name sanitizing (kept in sync with step1_build_celltype_ppi.py's
# sanitize_name() — celltype_gene_sets.json keys were written using THAT
# function, so any raw cell-type string we compare against those keys here
# (i.e. cell_type_source/cell_type_target from the scored-lr files) must be
# put through the identical transform or it will silently fail to match.
# ---------------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").replace(",", "")


# ---------------------------------------------------------------------------
# Ligand-receptor loading (unchanged from old pipeline)
# ---------------------------------------------------------------------------
def load_lr_pairs(lr_csv_path: str) -> set[tuple[str, str]]:
    print(f"Loading ligand-receptor pairs from {lr_csv_path} ...")
    df = pd.read_csv(lr_csv_path)
    if "ligand_gname" not in df.columns:
        df = pd.read_csv(lr_csv_path, sep="\t")
    df["ligand_gname"] = df["ligand_gname"].str.strip().str.upper()
    df["receptor_gname"] = df["receptor_gname"].str.strip().str.upper()
    df = df.dropna(subset=["ligand_gname", "receptor_gname"])
    lr_set = set(zip(df["ligand_gname"], df["receptor_gname"]))
    print(f"  {len(lr_set):,} LR pairs loaded\n")
    return lr_set


# ---------------------------------------------------------------------------
# NEW: scored/directional LR loading (LIANA + NeuronChat long-format output)
# ---------------------------------------------------------------------------
def expand_complex(gene_str: str) -> list[str]:
    """LIANA's ligand.complex/receptor.complex and NeuronChat's ligand/
    receptor fields encode multi-subunit complexes joined by '_'
    (e.g. 'ITGA4_ITGB1', or NeuronChat receptor complexes like
    'NTRK2_SORT1'). Split into individual gene symbols so each subunit can
    be checked against / connected to actual single-gene PPI-layer nodes.
    A plain single-gene ligand/receptor with no '_' just returns itself."""
    return [g for g in gene_str.split("_") if g]


def load_scored_lr_files(paths: list[str], allowed_sources: list[str] | None,
                          min_score: float | None) -> pd.DataFrame:
    required = {"ligand", "receptor", "source", "score", "cell_type_source", "cell_type_target"}
    frames = []
    for p in paths:
        print(f"Loading scored LR edges from {p} ...")
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p} missing columns: {missing}. Found: {list(df.columns)}")
        df["ligand"] = df["ligand"].astype(str).str.strip().str.upper()
        df["receptor"] = df["receptor"].astype(str).str.strip().str.upper()
        df["cell_type_source"] = df["cell_type_source"].astype(str).str.strip()
        df["cell_type_target"] = df["cell_type_target"].astype(str).str.strip()
        print(f"  {len(df):,} rows, sources: {sorted(df['source'].unique())}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined scored LR rows: {len(combined):,}")

    if allowed_sources:
        before = len(combined)
        combined = combined[combined["source"].isin(allowed_sources)]
        print(f"  Filtered to sources {allowed_sources}: {before:,} -> {len(combined):,} rows")

    if min_score is not None:
        before = len(combined)
        combined = combined[combined["score"] >= min_score]
        print(f"  Filtered to score >= {min_score}: {before:,} -> {len(combined):,} rows "
              f"(NOTE: score scales differ across methods/files — see module docstring)")

    combined = combined.dropna(subset=["ligand", "receptor", "cell_type_source", "cell_type_target"])
    combined = combined[combined["cell_type_source"] != combined["cell_type_target"]]
    # NOTE: dedup key includes 'source'. Deduping WITHOUT 'source' (the
    # previous behavior) silently collapsed multiple methods that agree on
    # the same (ligand, receptor, cell_type_source, cell_type_target) down
    # to a single arbitrary row, discarding exactly the per-method
    # provenance step6b's interaction-type histograms need. The actual
    # bipartite edge this expands into is still added at most once either
    # way, since add_scored_lr_edges() stores edges in a set -- keeping
    # every source here only affects scored_lr_edge_provenance.csv, not
    # the edge count in the .tsv files.
    combined = combined.drop_duplicates(subset=["ligand", "receptor", "cell_type_source", "cell_type_target", "source"])
    print(f"  {len(combined):,} unique (ligand, receptor, cell_type_source, cell_type_target, source) "
          f"rows after dedup\n")
    return combined


# ---------------------------------------------------------------------------
# NEW: before/after-PPI-check diagnostic, by source (ligand_receptor,
# reactome, liana, neuronchat) — see module docstring.
# ---------------------------------------------------------------------------
def gene_pairs_from_lr_df(df: pd.DataFrame, split_complexes: bool) -> set[frozenset]:
    """Unique (ligand, receptor) gene pairs a scored-LR dataframe proposes,
    expanded the same way (--no-split-complexes) the real edges are, IGNORING
    cell type — this is purely "what pairs does this source claim", used to
    check each against --ppi-pkl."""
    pairs: set[frozenset] = set()
    for row in df.itertuples(index=False):
        ligs = expand_complex(row.ligand) if split_complexes else [row.ligand]
        recs = expand_complex(row.receptor) if split_complexes else [row.receptor]
        for lig in ligs:
            for rec in recs:
                pairs.add(frozenset((lig, rec)))
    return pairs


def compute_ppi_check_stats(pairs_by_source: dict[str, set[frozenset]], ppi_ref: set[frozenset]) -> pd.DataFrame:
    rows = []
    for label, pairs in pairs_by_source.items():
        total = len(pairs)
        passed = sum(1 for p in pairs if p in ppi_ref)
        rows.append({
            "source": label,
            "before_ppi_check": total,
            "after_ppi_check": passed,
            "pct_ppi_corroborated": round(100 * passed / total, 1) if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)


def plot_ppi_check_histogram(stats_df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"  ⚠ matplotlib not available — skipping {out_path.name} (ppi_check_stats.csv was still written)")
        return

    sources = stats_df["source"].tolist()
    before = stats_df["before_ppi_check"].tolist()
    after = stats_df["after_ppi_check"].tolist()

    x = np.arange(len(sources))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(sources)), 5))
    ax.bar(x - width / 2, before, width, label="candidate LR pairs")
    ax.bar(x + width / 2, after, width, label="PPI-corroborated")
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=15, ha="right")
    ax.set_ylabel("Unique ligand-receptor gene pairs")
    ax.set_title("LR pairs before vs after PPI validation, by source")
    for xi, b, a in zip(x, before, after):
        pct = 100 * a / b if b else 0
        ax.annotate(f"{pct:.0f}%", (xi + width / 2, a), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram -> {out_path}")


# ---------------------------------------------------------------------------
# NEW: unique gene-gene relationships vs. absolute (cell-type-namespaced)
# edge counts — see module docstring section "unique-gene-pair vs.
# absolute-edge-count reporting".
# ---------------------------------------------------------------------------
OVERALL_LABEL = "OVERALL (as written to bipartite files)"
OVERALL_EXCL_LABEL_TMPL = "OVERALL (excluding {sources})"


def compute_unique_vs_absolute_stats(
    buckets: dict[tuple[str, str], set],
    source_edge_tracker: dict[str, dict[tuple[str, str], set]],
    hide_sources: set[str] | None = None,
) -> pd.DataFrame:
    """hide_sources (e.g. {"global_ppi_identity"}) does NOT remove anything
    from the per-source rows below -- every source that contributed edges
    always gets its own row, full transparency. It only controls whether an
    ADDITIONAL 'OVERALL (excluding ...)' row is computed alongside the
    unfiltered OVERALL row, for sources whose edges aren't really "gene-gene
    relationships" in the same sense as LR/pathway/scored-LR edges (identity
    bridges to a backbone are structural, not biological claims) and would
    otherwise dominate/skew the combined ratio. See plot_* functions below
    for how this feeds into which bars actually get drawn."""
    hide_sources = hide_sources or set()
    rows = []
    for source_label in sorted(source_edge_tracker.keys()):
        per_pair = source_edge_tracker[source_label]
        absolute_edges = sum(len(edges) for edges in per_pair.values())
        unique_pairs = {frozenset(e) for edges in per_pair.values() for e in edges}
        n_unique = len(unique_pairs)
        rows.append({
            "source": source_label,
            "absolute_edges": absolute_edges,
            "unique_gene_pairs": n_unique,
            "overcounting_ratio": round(absolute_edges / n_unique, 2) if n_unique else 0.0,
        })

    # OVERALL row: computed straight from the final, cross-source-deduped
    # `buckets` — i.e. exactly what ends up written to the bipartite .tsv
    # files, not a sum of the per-source rows above (which can double-count
    # an edge that two sources both proposed for the same cell-type pair,
    # since buckets collapses that to one line but the per-source trackers
    # don't share a set).
    overall_absolute = sum(len(edges) for edges in buckets.values())
    overall_unique = len({frozenset(e) for edges in buckets.values() for e in edges})
    rows.append({
        "source": OVERALL_LABEL,
        "absolute_edges": overall_absolute,
        "unique_gene_pairs": overall_unique,
        "overcounting_ratio": round(overall_absolute / overall_unique, 2) if overall_unique else 0.0,
    })

    # OPTIONAL second OVERALL row with one or more sources' edges left out
    # entirely (not just hidden from a chart) -- e.g. identity bridges to a
    # --base-ppi backbone, which are (gene, gene) self-pairs by construction
    # and inflate the combined "unique gene-gene relationship" count with
    # entries that were never really a claimed biological relationship.
    # Derived from source_edge_tracker (the union of every NON-hidden
    # source's edges per pair_key), not from `buckets` minus something,
    # since buckets doesn't retain per-source provenance.
    present_hidden = sorted(s for s in hide_sources if s in source_edge_tracker)
    if present_hidden:
        kept_by_pair: dict[tuple[str, str], set] = defaultdict(set)
        for source_label, per_pair in source_edge_tracker.items():
            if source_label in hide_sources:
                continue
            for pair_key, edges in per_pair.items():
                kept_by_pair[pair_key] |= edges
        excl_absolute = sum(len(edges) for edges in kept_by_pair.values())
        excl_unique = len({frozenset(e) for edges in kept_by_pair.values() for e in edges})
        rows.append({
            "source": OVERALL_EXCL_LABEL_TMPL.format(sources=", ".join(present_hidden)),
            "absolute_edges": excl_absolute,
            "unique_gene_pairs": excl_unique,
            "overcounting_ratio": round(excl_absolute / excl_unique, 2) if excl_unique else 0.0,
        })

    return pd.DataFrame(rows)


def plot_unique_vs_absolute_overall(stats_df: pd.DataFrame, out_path: Path,
                                     hide_sources: set[str] | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  ⚠ matplotlib not available — skipping {out_path.name} (unique_vs_absolute_stats.csv was still written)")
        return

    hide_sources = hide_sources or set()
    # Prefer the "excluding {hidden sources}" row when one was computed
    # (i.e. hide_sources was non-empty and at least one of those sources
    # actually contributed edges) -- this is the row that answers "how
    # skewed is my real LR/pathway overcounting picture by the identity
    # bridges", which is the whole point of --exclude-identity-from-charts.
    excl_rows = stats_df[stats_df["source"].str.startswith("OVERALL (excluding")]
    if hide_sources and not excl_rows.empty:
        overall = excl_rows.iloc[0]
        subtitle = f"\n(excludes: {', '.join(sorted(hide_sources))})"
    else:
        overall = stats_df[stats_df["source"] == OVERALL_LABEL].iloc[0]
        subtitle = ""

    absolute = overall["absolute_edges"]
    unique = overall["unique_gene_pairs"]
    ratio = overall["overcounting_ratio"]

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(["Absolute edges\n(bipartite file lines)", "Unique gene-gene\nrelationships"],
                   [absolute, unique], color=["#4C72B0", "#55A868"])
    for bar, val in zip(bars, [absolute, unique]):
        ax.annotate(f"{val:,}", (bar.get_x() + bar.get_width() / 2, val),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title(f"Overall: absolute vs. unique edges\n(overcounting ratio ≈ {ratio:.2f}x){subtitle}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Chart -> {out_path}")


def plot_unique_vs_absolute_by_source(stats_df: pd.DataFrame, out_path: Path,
                                       hide_sources: set[str] | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"  ⚠ matplotlib not available — skipping {out_path.name} (unique_vs_absolute_stats.csv was still written)")
        return

    hide_sources = hide_sources or set()
    per_source = stats_df[~stats_df["source"].str.startswith("OVERALL")]
    per_source = per_source[~per_source["source"].isin(hide_sources)]
    if per_source.empty:
        return

    sources = per_source["source"].tolist()
    absolute = per_source["absolute_edges"].tolist()
    unique = per_source["unique_gene_pairs"].tolist()

    x = np.arange(len(sources))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(sources)), 5.5))
    ax.bar(x - width / 2, absolute, width, label="absolute edges (file lines)", color="#4C72B0")
    ax.bar(x + width / 2, unique, width, label="unique gene-gene pairs", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Absolute vs. unique edges, by source/method")
    for xi, a, u in zip(x, absolute, unique):
        ratio = a / u if u else 0
        ax.annotate(f"{ratio:.1f}x", (xi, max(a, u)), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Chart -> {out_path}")


# ---------------------------------------------------------------------------
# Reactome pathway pairs (vectorised, reused from old step3_fastertest.py —
# same logic, trimmed to only what's needed: pathway -> gene set)
# ---------------------------------------------------------------------------
def load_uniprot_map(map_csv_path: str) -> dict:
    if not map_csv_path:
        return {}
    df = pd.read_csv(map_csv_path)
    return dict(zip(df["uniprotid"], df["symbol"]))


def build_pathway_to_genes(csv_path: str, max_pathway_size: int, uniprot_to_symbol: dict,
                            restrict_to: set[str] | None = None) -> dict[str, set]:
    print(f"Loading pathway data from {csv_path} ...")
    df_temp = pd.read_csv(csv_path, nrows=5)
    cols = set(df_temp.columns)
    use_symbol_cols = {"symbol1", "symbol2"}.issubset(cols)
    id_cols = ["symbol1", "symbol2"] if use_symbol_cols else ["uniprot1", "uniprot2"]
    usecols = id_cols + ["common_pathway"]
    df = pd.read_csv(csv_path, low_memory=False, usecols=usecols)
    df = df.dropna(subset=["common_pathway"])
    print(f"  {len(df):,} rows with common_pathway")

    c1 = df[id_cols[0]].astype("object").fillna("").astype(str).str.strip().str.upper()
    c2 = df[id_cols[1]].astype("object").fillna("").astype(str).str.strip().str.upper()

    if not use_symbol_cols and uniprot_to_symbol:
        lookup = {str(k).upper(): v for k, v in uniprot_to_symbol.items()}
        s1 = c1.map(lookup); s1 = s1.where(s1.notna(), c1).astype(str).str.upper()
        s2 = c2.map(lookup); s2 = s2.where(s2.notna(), c2).astype(str).str.upper()
    else:
        s1, s2 = c1, c2

    pathway = df["common_pathway"].astype(str).str.strip()
    valid = (pathway != "") & (pathway.str.lower() != "nan") & (s1 != s2) & (s1 != "") & (s2 != "")
    long_df = pd.DataFrame({"pathway": pathway[valid].values, "g1": s1[valid].values, "g2": s2[valid].values})

    pathway_to_genes: dict[str, set] = defaultdict(set)
    for pway, g1, g2 in zip(long_df["pathway"], long_df["g1"], long_df["g2"]):
        pathway_to_genes[pway].add(g1)
        pathway_to_genes[pway].add(g2)

    kept = {}
    n_excluded = 0
    for pway, genes in pathway_to_genes.items():
        if restrict_to is not None:
            genes = genes & restrict_to
            if len(genes) < 2:
                continue
        if len(genes) > max_pathway_size:
            n_excluded += 1
            continue
        kept[pway] = genes

    print(f"  Pathways kept: {len(kept):,} (excluded {n_excluded:,} oversized)\n")
    return kept


# ---------------------------------------------------------------------------
# Core: expand gene-pair edges into cell-type-pair bipartite buckets
# ---------------------------------------------------------------------------
def add_cross_celltype_edges(
    gene_pairs: set[tuple[str, str]],
    gene_to_celltypes: dict[str, set[str]],
    buckets: dict[tuple[str, str], set],
    ppi_ref: set | None,
    source_label: str,
    stats: dict,
    source_edge_tracker: dict[str, dict[tuple[str, str], set]] | None = None,
) -> None:
    for ga, gb in gene_pairs:
        cts_a = gene_to_celltypes.get(ga, ())
        cts_b = gene_to_celltypes.get(gb, ())
        if not cts_a or not cts_b:
            continue
        if ppi_ref is not None and frozenset((ga, gb)) not in ppi_ref:
            stats["skipped_no_ppi"] += 1
            continue
        for ct_a in cts_a:
            for ct_b in cts_b:
                if ct_a == ct_b:
                    continue
                # Orient (pair_key, edge) together — see FIX note in module
                # docstring. Previously pair_key was sorted independently of
                # (ga, gb)'s order, so ga could end up written to the g1
                # column of a file whose manifest "source" was actually the
                # cell type gb belongs to.
                if ct_a < ct_b:
                    pair_key = (ct_a, ct_b)
                    edge = (ga, gb)
                else:
                    pair_key = (ct_b, ct_a)
                    edge = (gb, ga)
                buckets[pair_key].add(edge)
                stats["edges_by_pair_and_source"][(pair_key, source_label)] += 1
                if source_edge_tracker is not None:
                    source_edge_tracker[source_label][pair_key].add(edge)


def add_global_ppi_identity_edges(
    celltype_gene_sets: dict[str, list],
    global_ppi_name: str,
    buckets: dict[tuple[str, str], set],
    stats: dict,
    source_edge_tracker: dict[str, dict[tuple[str, str], set]] | None = None,
) -> int:
    """Connects the base/comprehensive PPI pseudo-cell-type multiplex (added
    in step 1 via --base-ppi) to every other cell-type multiplex through
    identity bipartite edges: gene X in <cell_type> <-> gene X in
    global_ppi, for every gene shared between the two. This is deliberately
    NOT gated by ppi_ref validation — it isn't asserting a PPI/LR/pathway
    relationship, just that the same gene exists in both node spaces, so
    the random walk can move between them.

    Returns the number of cell types successfully connected.
    """
    if global_ppi_name not in celltype_gene_sets:
        return 0

    global_genes = set(celltype_gene_sets[global_ppi_name])
    n_connected = 0
    print(f"Connecting '{global_ppi_name}' to other cell types via identity bipartite edges ...")
    for ct, genes in celltype_gene_sets.items():
        if ct == global_ppi_name:
            continue
        shared = global_genes & set(genes)
        if not shared:
            continue
        pair_key = tuple(sorted((ct, global_ppi_name)))
        for g in shared:
            edge = (g, g)
            buckets[pair_key].add(edge)
            if source_edge_tracker is not None:
                source_edge_tracker["global_ppi_identity"][pair_key].add(edge)
        stats["edges_by_pair_and_source"][(pair_key, "global_ppi_identity")] += len(shared)
        n_connected += 1
        print(f"  {global_ppi_name} <-> {ct}: {len(shared):,} shared genes (identity edges)")
    print()
    return n_connected


def add_scored_lr_edges(
    df: pd.DataFrame,
    celltype_gene_sets_as_sets: dict[str, set],
    buckets: dict[tuple[str, str], set],
    ppi_ref: set | None,
    split_complexes: bool,
    stats: dict,
    provenance_rows: list,
    source_edge_tracker: dict[str, dict[tuple[str, str], set]] | None = None,
) -> None:
    n_added = 0
    n_dropped_ct = 0
    n_dropped_gene = 0
    n_dropped_ppi = 0

    has_ligand_type = "ligand_type" in df.columns

    for row in df.itertuples(index=False):
        ct_src = sanitize_name(row.cell_type_source)
        ct_tgt = sanitize_name(row.cell_type_target)
        if ct_src not in celltype_gene_sets_as_sets or ct_tgt not in celltype_gene_sets_as_sets:
            n_dropped_ct += 1
            continue
        if ct_src == ct_tgt:
            continue

        src_genes = celltype_gene_sets_as_sets[ct_src]
        tgt_genes = celltype_gene_sets_as_sets[ct_tgt]

        ligand_subunits = expand_complex(row.ligand) if split_complexes else [row.ligand]
        receptor_subunits = expand_complex(row.receptor) if split_complexes else [row.receptor]

        row_source = row.source
        row_ligand_type = getattr(row, "ligand_type", None) if has_ligand_type else None
        if isinstance(row_ligand_type, float) and pd.isna(row_ligand_type):
            row_ligand_type = None

        for lig in ligand_subunits:
            for rec in receptor_subunits:
                if lig not in src_genes or rec not in tgt_genes:
                    n_dropped_gene += 1
                    continue
                if ppi_ref is not None and frozenset((lig, rec)) not in ppi_ref:
                    n_dropped_ppi += 1
                    continue
                # Record provenance BEFORE the reorientation below -- ct_src/
                # lig is always the ligand-sender side and ct_tgt/rec the
                # receptor-receiver side here, regardless of which cell type
                # ends up sorting first for bucket storage. step6b rebuilds
                # the exact same "<celltype>::GENE" node pair from these
                # columns to look the edge up in the final module graph.
                provenance_rows.append({
                    "cell_type_source": ct_src,
                    "cell_type_target": ct_tgt,
                    "ligand": lig,
                    "receptor": rec,
                    "source": row_source,
                    "ligand_type": row_ligand_type,
                })
                # Orient (pair_key, edge) together so edge[0] always belongs
                # to pair_key[0]'s gene space and edge[1] to pair_key[1]'s —
                # see FIX note in module docstring.
                if ct_src < ct_tgt:
                    pair_key = (ct_src, ct_tgt)
                    edge = (lig, rec)
                else:
                    pair_key = (ct_tgt, ct_src)
                    edge = (rec, lig)
                buckets[pair_key].add(edge)
                stats["edges_by_pair_and_source"][(pair_key, "scored_lr")] += 1
                if source_edge_tracker is not None:
                    # Tracked per actual method (row_source, e.g.
                    # "liana_consensus", "cellphonedb", "neuronchat", ...)
                    # rather than lumped as one "scored_lr" bucket, so
                    # methods can be compared directly in the unique-vs-
                    # absolute report.
                    source_edge_tracker[row_source][pair_key].add(edge)
                n_added += 1

    print(f"Scored LR edges: {n_added:,} added, {n_dropped_ct:,} dropped (cell type not in multiplex), "
          f"{n_dropped_gene:,} dropped (subunit not in that cell type's PPI layer)"
          + (f", {n_dropped_ppi:,} dropped (no PPI support)" if ppi_ref is not None else "") + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build LR (+ optional Reactome, + optional scored LIANA/NeuronChat, "
                                               "+ global_ppi identity) bipartite edges between cell-type multiplexes")
    ap.add_argument("--wdir", required=True, help="MultiXrank working dir (output of step 1)")
    ap.add_argument("--lr-csv", required=True)
    ap.add_argument("--reactome-csv", default=None, help="Optional. If omitted, only LR bipartite edges are built.")
    ap.add_argument("--uniprot-map", default=None)
    ap.add_argument("--max-pathway-size", type=int, default=300)
    ap.add_argument("--ppi-pkl", default=None, help="Optional comprehensive PPI reference for validation")
    ap.add_argument("--validate-against-ppi", action="store_true",
                     help="Only keep LR/pathway/scored-LR bipartite edges whose gene pair is ALSO an edge in --ppi-pkl")
    ap.add_argument("--scored-lr-csv", nargs="+", default=[],
                     help="One or more long-format scored LR files from step0a_run_liana.R / "
                          "step0b_run_neuronchat.R (columns: ligand, receptor, source, score, "
                          "cell_type_source, cell_type_target). Adds directional ligand(source)->"
                          "receptor(target) bipartite edges using the actual cell types each row names, "
                          "instead of the flat --lr-csv's all-cell-types-that-express-both-genes expansion.")
    ap.add_argument("--scored-lr-sources", nargs="+", default=None,
                     help="Optional filter on the scored-LR files' 'source' column (e.g. "
                          "liana_consensus neuronchat cellphonedb). Default: keep every source/method present.")
    ap.add_argument("--scored-lr-min-score", type=float, default=None,
                     help="Optional row['score'] >= threshold filter across ALL --scored-lr-csv files. "
                          "Score scales differ across methods (see module docstring) — leave unset unless "
                          "you've already picked a threshold that makes sense for every method/file you're passing.")
    ap.add_argument("--no-split-complexes", action="store_true",
                     help="By default, multi-subunit ligand/receptor complexes in --scored-lr-csv files "
                          "(joined by '_', e.g. 'ITGA4_ITGB1') are split into individual gene subunits before "
                          "matching against PPI-layer nodes. Pass this flag to instead require the complex "
                          "STRING ITSELF to exist as a node (will drop almost all complex-involving edges).")
    ap.add_argument("--global-ppi-name", default="global_ppi",
                     help="Name of the base/comprehensive PPI pseudo-cell-type multiplex added in step 1 via "
                          "--base-ppi (must match its multiplex/ folder name). Used to auto-connect it to every "
                          "other cell type via identity bipartite edges.")
    ap.add_argument("--no-global-ppi-bipartite", action="store_true",
                     help="Skip auto-connecting --global-ppi-name to every other cell type via identity bipartite edges")
    ap.add_argument("--exclude-identity-from-charts", action="store_true",
                     help="Keep building global_ppi identity edges as usual (network, config.yml, "
                          "everything downstream is unaffected) but leave the 'global_ppi_identity' "
                          "source out of chart_unique_vs_absolute_by_source.png, and use an "
                          "identity-excluded total for chart_unique_vs_absolute_overall.png instead "
                          "of the raw one. Identity edges are (gene, gene) self-pairs bridging to the "
                          "backbone, not real LR/pathway/scored-LR relationships, so they otherwise "
                          "dominate the 'unique vs. absolute' picture with something that isn't really "
                          "a gene-gene relationship being overcounted. Both the raw OVERALL row and "
                          "global_ppi_identity's own row are still written to unique_vs_absolute_stats.csv "
                          "either way -- this flag only changes what gets drawn.")
    args = ap.parse_args()

    wdir = Path(args.wdir)
    gene_sets_path = wdir / "celltype_gene_sets.json"
    if not gene_sets_path.exists():
        raise SystemExit(f"ERROR: {gene_sets_path} not found — run step 1 first.")

    with open(gene_sets_path) as f:
        celltype_gene_sets = json.load(f)
    celltype_gene_sets_as_sets = {ct: set(genes) for ct, genes in celltype_gene_sets.items()}

    gene_to_celltypes: dict[str, set[str]] = defaultdict(set)
    for ct, genes in celltype_gene_sets.items():
        for g in genes:
            gene_to_celltypes[g].add(ct)
    print(f"Loaded gene universe: {len(gene_to_celltypes):,} genes across {len(celltype_gene_sets)} cell types\n")

    # ppi_ref (the set of PPI edges from --ppi-pkl) is built whenever
    # --ppi-pkl is given, and always used for the before/after-PPI-check
    # diagnostic below. Whether it also FILTERS the real bipartite edges is
    # controlled separately by --validate-against-ppi (filter_ppi_ref):
    # pass --ppi-pkl alone to get the diagnostic without dropping anything.
    ppi_ref = None
    if args.validate_against_ppi and not args.ppi_pkl:
        ap.error("--validate-against-ppi requires --ppi-pkl")
    if args.ppi_pkl:
        import pickle
        import networkx as nx
        print(f"Loading PPI reference: {args.ppi_pkl} ...")
        with open(args.ppi_pkl, "rb") as f:
            G_ppi = pickle.load(f)
        G_ppi = nx.relabel_nodes(G_ppi, {n: str(n).strip().upper() for n in G_ppi.nodes()})
        ppi_ref = {frozenset((u, v)) for u, v in G_ppi.edges()}
        print(f"  {len(ppi_ref):,} reference PPI edges loaded\n")
    filter_ppi_ref = ppi_ref if args.validate_against_ppi else None

    buckets: dict[tuple[str, str], set] = defaultdict(set)
    stats = {"skipped_no_ppi": 0, "edges_by_pair_and_source": defaultdict(int)}
    pairs_by_source: dict[str, set[frozenset]] = {}
    # NEW: mirrors `buckets` but split per source/method, used only for the
    # unique-vs-absolute report at the end (see module docstring).
    source_edge_tracker: dict[str, dict[tuple[str, str], set]] = defaultdict(lambda: defaultdict(set))

    lr_set = load_lr_pairs(args.lr_csv)
    add_cross_celltype_edges(lr_set, gene_to_celltypes, buckets, filter_ppi_ref, "ligand_receptor", stats,
                              source_edge_tracker=source_edge_tracker)
    if ppi_ref is not None:
        pairs_by_source["ligand_receptor"] = {frozenset(p) for p in lr_set}

    if args.reactome_csv:
        u2s = load_uniprot_map(args.uniprot_map)
        pathway_to_genes = build_pathway_to_genes(
            args.reactome_csv, args.max_pathway_size, u2s,
            restrict_to=set(gene_to_celltypes.keys()),
        )
        print("Expanding pathway-sharing gene pairs into cross-cell-type edges ...")
        n_pway_pairs = 0
        reactome_pairs: set[frozenset] = set()
        for pway, genes in pathway_to_genes.items():
            gene_list = sorted(genes)
            pairs = {tuple(sorted(p)) for p in combinations(gene_list, 2)}
            add_cross_celltype_edges(pairs, gene_to_celltypes, buckets, filter_ppi_ref, "reactome", stats,
                                      source_edge_tracker=source_edge_tracker)
            if ppi_ref is not None:
                reactome_pairs.update(frozenset(p) for p in pairs)
            n_pway_pairs += len(pairs)
        if ppi_ref is not None:
            pairs_by_source["reactome"] = reactome_pairs
        print(f"  Considered {n_pway_pairs:,} within-pathway gene pairs\n")

    if args.scored_lr_csv:
        split_complexes = not args.no_split_complexes
        scored_df = load_scored_lr_files(args.scored_lr_csv, args.scored_lr_sources, args.scored_lr_min_score)
        provenance_rows: list[dict] = []
        add_scored_lr_edges(
            scored_df, celltype_gene_sets_as_sets, buckets, filter_ppi_ref,
            split_complexes=split_complexes, stats=stats, provenance_rows=provenance_rows,
            source_edge_tracker=source_edge_tracker,
        )
        if provenance_rows:
            prov_df = pd.DataFrame(provenance_rows)
            prov_path = wdir / "scored_lr_edge_provenance.csv"
            prov_df.to_csv(prov_path, index=False)
            print(f"Wrote {len(prov_df):,} scored-LR edge provenance row(s) "
                  f"(source/method + NeuronChat ligand_type where available) -> {prov_path}")
            print(f"  Sources present: {sorted(prov_df['source'].dropna().unique())}")
            lt_present = sorted(prov_df['ligand_type'].dropna().unique())
            print(f"  Ligand types present: {lt_present if lt_present else '(none -- no NeuronChat rows survived filtering)'}\n")
        if ppi_ref is not None:
            is_neuronchat = scored_df["source"] == "neuronchat"
            liana_df = scored_df[~is_neuronchat]
            neuronchat_df = scored_df[is_neuronchat]
            if len(liana_df):
                pairs_by_source["liana"] = gene_pairs_from_lr_df(liana_df, split_complexes)
            if len(neuronchat_df):
                pairs_by_source["neuronchat"] = gene_pairs_from_lr_df(neuronchat_df, split_complexes)

    if ppi_ref is not None and pairs_by_source:
        print("Checking each source's candidate LR pairs against the PPI reference ...")
        ppi_check_df = compute_ppi_check_stats(pairs_by_source, ppi_ref)
        ppi_check_df.to_csv(wdir / "ppi_check_stats.csv", index=False)
        print(ppi_check_df.to_string(index=False))
        print()
        plot_ppi_check_histogram(ppi_check_df, wdir / "ppi_check_histogram.png")
        print()

    if not args.no_global_ppi_bipartite:
        n_connected = add_global_ppi_identity_edges(celltype_gene_sets, args.global_ppi_name, buckets, stats,
                                                      source_edge_tracker=source_edge_tracker)
        if n_connected == 0 and args.global_ppi_name in celltype_gene_sets:
            print(f"⚠ '{args.global_ppi_name}' found in celltype_gene_sets.json but shares no genes with any "
                  f"other cell type — no identity edges added\n")

    if not buckets:
        raise SystemExit("ERROR: no bipartite edges were generated at all — check your LR/Reactome/scored-LR "
                          "CSVs, gene-symbol overlap with the loaded PPI layers, and (if using --base-ppi) "
                          "whether --global-ppi-name matches step 1's output.")

    bip_dir = wdir / "bipartite"
    bip_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    print("Writing bipartite edge files ...")
    for (ct_a, ct_b), pairs in sorted(buckets.items()):
        fname = f"{ct_a}__{ct_b}.tsv"
        rel_path = Path("bipartite") / fname
        pd.DataFrame(sorted(pairs), columns=["g1", "g2"]).to_csv(wdir / rel_path, sep="\t", index=False, header=False)
        manifest.append({"file": str(rel_path), "source": ct_a, "target": ct_b, "n_edges": len(pairs)})
        print(f"  {ct_a} <-> {ct_b}: {len(pairs):,} edges -> {rel_path}")
    print()

    with open(wdir / "bipartite_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    stats_rows = [
        {"cell_type_pair": " <-> ".join(pair), "source": src, "n_edges": n}
        for (pair, src), n in sorted(stats["edges_by_pair_and_source"].items())
    ]
    pd.DataFrame(stats_rows).to_csv(wdir / "stats_bipartite_edges.csv", index=False)

    # NEW: unique gene-gene relationships vs. absolute (cell-type-namespaced)
    # edge counts, per source/method and overall. See module docstring.
    print("Computing unique-gene-pair vs. absolute-edge-count stats (namespacing overcounting check) ...")
    hide_sources = {"global_ppi_identity"} if args.exclude_identity_from_charts else set()
    uva_df = compute_unique_vs_absolute_stats(buckets, source_edge_tracker, hide_sources=hide_sources)
    uva_path = wdir / "unique_vs_absolute_stats.csv"
    uva_df.to_csv(uva_path, index=False)
    print(uva_df.to_string(index=False))
    print(f"  -> {uva_path}")
    plot_unique_vs_absolute_overall(uva_df, wdir / "chart_unique_vs_absolute_overall.png",
                                     hide_sources=hide_sources)
    plot_unique_vs_absolute_by_source(uva_df, wdir / "chart_unique_vs_absolute_by_source.png",
                                       hide_sources=hide_sources)
    if hide_sources:
        print(f"  [exclude-identity-from-charts] '{', '.join(sorted(hide_sources))}' left out of both "
              f"PNGs (still logged in unique_vs_absolute_stats.csv for reference)")
    overall_row = uva_df[uva_df["source"] == OVERALL_LABEL].iloc[0]
    print(f"  Overall (raw, all sources): {overall_row['absolute_edges']:,} absolute edges written across "
          f"all bipartite files represent only {overall_row['unique_gene_pairs']:,} unique gene-gene pairs "
          f"(≈{overall_row['overcounting_ratio']:.2f}x namespacing overcounting)")
    if hide_sources:
        excl_rows = uva_df[uva_df["source"].str.startswith("OVERALL (excluding")]
        if not excl_rows.empty:
            er = excl_rows.iloc[0]
            print(f"  Overall (excluding {', '.join(sorted(hide_sources))}): {er['absolute_edges']:,} absolute "
                  f"edges represent {er['unique_gene_pairs']:,} unique gene-gene pairs "
                  f"(≈{er['overcounting_ratio']:.2f}x) -- this is the picture drawn in the PNGs")
    print()

    print("=" * 60)
    print(f"Step 2 complete: {len(manifest)} bipartite file(s), "
          f"{sum(m['n_edges'] for m in manifest):,} total edges")
    if filter_ppi_ref is not None:
        print(f"  Edges skipped (no PPI support): {stats['skipped_no_ppi']:,}")
    elif ppi_ref is not None:
        print("  --ppi-pkl was used for the ppi_check_stats.csv/.png diagnostic only "
              "(pass --validate-against-ppi to also filter edges by it)")
    print(f"  Manifest -> {wdir / 'bipartite_manifest.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()