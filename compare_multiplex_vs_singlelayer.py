#!/usr/bin/env python3
"""
compare_multiplex_vs_singlelayer.py

Match HIGH-PERSISTENCE clusters from the multiplex (cell-type-namespaced)
HiDeF run (steps 11a-11c) against the single-layer painome HiDeF run
(hidef_only.py), on two independent axes:

  1. GENE MEMBERSHIP
     Multiplex "ct::GENE" members are collapsed to plain gene symbols (so
     Fibroblast::EGFR and Enteric_Neuron::EGFR both count as EGFR). Every
     (multiplex, single-layer) pair is scored by Jaccard, overlap
     coefficient and a hypergeometric p-value. The p-value is computed
     against a universe of genes present in BOTH networks and BH-corrected
     over ALL pairs compared.

  2. ENRICHMENT PATTERN -- read from the SAME tables your lollipop plots use
     (<analysis>_summarized_results.csv, filtered on adjusted_p_value <
     --adj-p and cluster_size >= --enrich-min-cluster-size, with the term
     column picked the same way as get_enrichment_term_column()). Terms are
     labelled exactly like clean_enrichment_term() labels them, e.g.
     "Mantle cell lymphoma (Orphanet)", so every shared term in the output
     is a y-axis label on your lollipop plot. Each cluster's signature is
     {analysis|term: -log10 adjusted p}. It is IDF-weighted, so terms that
     nearly every cluster hits count for less. Pairs are scored by cosine
     similarity and by shared-term count. Shared terms from
     --generic-analyses (default proteomics, neuropathic_pain: binary,
     near-universal calls) are reported but don't count as an enrichment
     match on their own.

Each candidate pair is put in one category:
  same_module_same_enrichment     genes overlap significantly AND share >= 1
                                  specific lollipop term
  same_genes_generic_only         genes overlap, only share generic terms
                                  (e.g. "female", "pain upregulated")
  same_genes_enrichment_differs   genes overlap, both tested, no shared term
  same_genes_enrichment_untested  genes overlap, >=1 side not in the summarized results
  convergent_enrichment           enrichment matches strongly, genes do NOT overlap
                                  significantly (same biology, different genes --
                                  often the interesting multiplex-specific case)

PERSISTENCE
-----------
The two runs' persistence values are NOT on the same scale. The multiplex
values come from a windowed consensus over 40 CPM gammas x 5 restarts. The
single-layer values come from hidef_finder.run's own RB sweep. So "high
persistence" is therefore defined per side. The default is the top quartile
(--persistence-quantile 0.75), and you can set absolute cutoffs per side
with --ml-min-persistence / --sl-min-persistence. Each output row also
carries a within-side persistence percentile.

CLUSTER-ID MAPPING
------------------
The gene-level result files (which list each cluster's genes) are used
ONLY to map enrichment cluster IDs onto source clusters. The summarized
tables are then joined through that mapping. Each gene-level ID is mapped
to a source cluster in this order:
  1. exact ID match
  2. normalized ID match (e.g. "cluster_12" == "12")
Each ID match is then verified: >= 90% of the genes listed for that cluster
in the gene-level files must be members of the matched source cluster. If
the check fails, or no ID matched, the script falls back to gene content.
It picks the source cluster that contains >= 90% of those genes with the
highest Jaccard. The mapping counts per method are printed, so ID
mismatches can't slip by silently.

OUTPUTS (in --output-dir)
-------
  matched_pairs.csv               every candidate pair, categorized + annotated
  reciprocal_best_hits.csv        pairs that are each other's best gene-overlap match
  best_match_per_singlelayer.csv  for each single-layer cluster: best multiplex match
                                  by genes and by enrichment
  best_match_per_multiplex.csv    the same, from the multiplex side
  shared_terms_summary.csv        which enrichment terms recur across both hierarchies
  similarity_scatter.png          gene Jaccard vs enrichment cosine, by category
  selected_clusters_{ml,sl}.csv   the high-persistence clusters that were compared
"""
import argparse
import glob
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

# Gene-level files: used only to map enrichment cluster IDs -> source clusters.
GENE_LEVEL = {
    "proteomics": ("proteomics_analysis", "proteomics_gene_level_results.csv"),
    "neuropathic_pain": ("neuropathic_pain_analysis", "neuropathic_pain_gene_level_results.csv"),
    "orphanet": ("orphanet_enrichr_analysis", "orphanet_enrichr_gene_level_results.csv"),
    "omim": ("omim_ps_analysis", "omim_ps_gene_level_results.csv"),
    "phenotypes": ("phenotypes_analysis", "phenotype_gene_level_results.csv"),
    "spatial": ("spatial_analysis", "spatial_transcriptomics_hidef_comparison_gene_level_results.csv"),
}

# Summarized files: what the lollipop plot reads. Mirrors results_file +
# get_enrichment_term_column() in your plotting script.
SUMMARIZED = {
    "neuropathic_pain": ("neuropathic_pain_analysis", "neuropathic_pain_summarized_results.csv", "top_expression"),
    "omim_ps": ("omim_ps_analysis", "omim_ps_summarized_results.csv", "phenotypic_series"),
    "orphanet_enrichr": ("orphanet_enrichr_analysis", "orphanet_enrichr_summarized_results.csv", "cluster_top_disease"),
    "phenotype": ("phenotypes_analysis", "phenotype_summarized_results.csv", "top_phenotype"),
    "proteomics": ("proteomics_analysis", "proteomics_summarized_results.csv", "higher_expression_gender"),
    "spatial": ("spatial_analysis", "spatial_transcriptomics_hidef_comparison_summarized_results.csv", "top_cell_type"),
}
FALLBACK_TERM_COLS = ["top_disease", "top_phenotype", "top_cell_type", "term", "disease", "phenotype", "cell_type"]
GROUP_LABEL = {"spatial": "Spatial", "phenotype": "Phenotype", "omim_disease": "OMIM Disease",
               "omim_ps": "OMIM PS", "orphanet_enrichr": "Orphanet", "proteomics": "Proteomics",
               "neuropathic_pain": "Neuropathic Pain"}

CATEGORY_ORDER = ["same_module_same_enrichment", "same_genes_generic_only",
                  "same_genes_enrichment_differs", "same_genes_enrichment_untested",
                  "convergent_enrichment"]


# ============================================================================
# helpers
# ============================================================================

def norm_gene(x) -> str:
    s = str(x).strip()
    if "::" in s:
        s = s.rsplit("::", 1)[1]
    return s.upper()


def norm_id(x) -> str:
    s = str(x).strip().lower()
    for _ in range(2):
        s = re.sub(r"^(hidef|cluster|community|comm)[_\-\s:]*", "", s)
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def lollipop_label(term, analysis) -> str:
    """Same as clean_enrichment_term() in the lollipop script."""
    t = str(term).replace("_", " ")
    if analysis == "orphanet_enrichr":
        t = re.sub(r"\s*ORPHA[:\s]\d+.*$", "", t, flags=re.IGNORECASE).strip()
    return f"{t} ({GROUP_LABEL.get(analysis, analysis.replace('_', ' ').title())})"


def build_incidence(sets_list, gene_index):
    """genes x clusters binary CSC matrix."""
    rows, cols = [], []
    for j, s in enumerate(sets_list):
        for gname in s:
            i = gene_index.get(gname)
            if i is not None:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.int32)
    return sp.csc_matrix((data, (rows, cols)), shape=(len(gene_index), len(sets_list)))


def bh(p, m):
    """Benjamini-Hochberg over m total tests, of which only len(p) are
    observed (the rest have zero overlap, p = 1, and can't lower any q)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


# ============================================================================
# loading clusters
# ============================================================================

def load_ml_clusters(nodes_path, supra_graph_path):
    node_names, universe = None, None
    if supra_graph_path:
        with open(supra_graph_path, "rb") as f:
            g = pickle.load(f)
        node_names = g.vs["name"]
        universe = {norm_gene(n) for n in node_names}
        print(f"[multiplex] supra-graph: {len(node_names):,} nodes -> {len(universe):,} distinct genes")

    df = pd.read_csv(nodes_path, sep="\t", header=None,
                     names=["cluster_id", "size", "members", "persistence"],
                     dtype={"members": str, "cluster_id": str})
    clusters = {}
    for cid, members, pers in zip(df["cluster_id"], df["members"], df["persistence"]):
        if pd.isna(members) or not str(members).strip():
            continue
        toks = [t for t in re.split(r"[,\s]+", str(members).strip()) if t]
        if node_names is not None and all(t.lstrip("-").isdigit() for t in toks[:5]):
            try:
                toks = [node_names[int(t)] for t in toks]
            except (IndexError, ValueError):
                pass
        gene_cts = defaultdict(set)
        for t in toks:
            if "::" in t:
                gene_cts[norm_gene(t)].add(t.rsplit("::", 1)[0])
        clusters[str(cid)] = {
            "genes": {norm_gene(t) for t in toks},
            "n_nodes": len(toks),
            "persistence": float(pers),
            "gene_cts": dict(gene_cts),
            "ct_counts": Counter(t.rsplit("::", 1)[0] for t in toks if "::" in t),
        }
    print(f"[multiplex] {len(clusters):,} clusters from {nodes_path}")
    return clusters, universe


def load_sl_clusters(pkl_path, network_path):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict) and "communities" in raw:  # hidef_results.pkl
        raw = raw["communities"]
    clusters = {}
    for cid, c in raw.items():
        members = list(c["members"])
        clusters[str(cid)] = {
            "genes": {norm_gene(m) for m in members},
            "n_nodes": len(members),
            "persistence": float(c.get("persistence", np.nan)),
            "gene_cts": {},
            "ct_counts": Counter(),
        }
    print(f"[single-layer] {len(clusters):,} clusters from {pkl_path}")
    universe = None
    if network_path:
        with open(network_path, "rb") as f:
            G = pickle.load(f)
        universe = {norm_gene(n) for n in G.nodes()}
        print(f"[single-layer] network: {len(universe):,} distinct genes")
    return clusters, universe


def select_high_persistence(clusters, min_pers, quantile, min_size, max_size, label):
    eligible = {k: v for k, v in clusters.items()
                if v["persistence"] > 0 and not np.isnan(v["persistence"])
                and len(v["genes"]) >= min_size
                and (max_size is None or len(v["genes"]) <= max_size)}
    if not eligible:
        raise SystemExit(f"[{label}] no clusters pass the size filter")
    pers = np.array([v["persistence"] for v in eligible.values()])
    thr = min_pers if min_pers is not None else float(np.quantile(pers, quantile))
    selected = [k for k, v in eligible.items() if v["persistence"] >= thr]
    how = "absolute" if min_pers is not None else f"q{quantile:.2f}"
    print(f"[{label}] persistence range {pers.min():g}-{pers.max():g}; "
          f"threshold {thr:g} ({how}) -> {len(selected):,}/{len(eligible):,} clusters selected")
    # within-side percentile over all positive-persistence clusters
    allp = pd.Series({k: v["persistence"] for k, v in clusters.items() if v["persistence"] > 0})
    pct = allp.rank(pct=True).to_dict()
    return selected, pct


# ============================================================================
# enrichment signatures
# ============================================================================

def find_analysis_file(root, prefix, fname, cluster_type, dir_filter, label, name):
    pattern = os.path.join(root, f"{prefix}_{cluster_type}*")
    matches = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if dir_filter:
        filt = [d for d in matches if dir_filter in os.path.basename(d)]
        matches = filt or matches
    with_file = [d for d in matches if os.path.exists(os.path.join(d, fname))]
    if not with_file:
        print(f"  [{label}] {name}: no directory with {fname} (pattern {pattern})")
        return None
    if len(with_file) > 1:
        print(f"  [{label}] {name}: ⚠ {len(with_file)} candidate dirs, using "
              f"{os.path.basename(with_file[0])} (others: "
              f"{[os.path.basename(d) for d in with_file[1:]]}) -- set --*-dir-filter to choose")
    return os.path.join(with_file[0], fname)


def load_gene_level_ids(root, cluster_type, dir_filter, label):
    """{enrichment cluster_id: set(genes)} from the gene-level files -- used
    only to map enrichment IDs onto source clusters."""
    tested_genes = defaultdict(set)
    for name, (prefix, fname) in GENE_LEVEL.items():
        path = find_analysis_file(root, prefix, fname, cluster_type, dir_filter, label, name)
        if path is None:
            continue
        df = pd.read_csv(path, usecols=lambda c: c in ("cluster_id", "gene"))
        if {"cluster_id", "gene"} - set(df.columns):
            continue
        for cid, gene in zip(df["cluster_id"].astype(str), df["gene"]):
            tested_genes[cid].add(norm_gene(gene))
    return dict(tested_genes)


def load_summarized(root, cluster_type, dir_filter, adj_thr, min_csize, label):
    """Exactly what the lollipop plot shows: rows with adjusted_p_value <
    adj_thr and cluster_size >= min_csize. Returns sig {cid: {key: -log10 q}},
    padj {cid: {key: q}}, tested (cids present after the size filter) and
    labels {key: lollipop y-axis label}."""
    sig, padj, tested, labels = defaultdict(dict), defaultdict(dict), set(), {}
    print(f"[{label}] loading summarized enrichment (lollipop tables) from {root}")
    for name, (prefix, fname, tcol) in SUMMARIZED.items():
        path = find_analysis_file(root, prefix, fname, cluster_type, dir_filter, label, name)
        if path is None:
            continue
        df = pd.read_csv(path)
        if "cluster_id" not in df.columns or "adjusted_p_value" not in df.columns:
            print(f"  [{label}] {name}: no cluster_id/adjusted_p_value column, skipped")
            continue
        if "cluster_size" in df.columns:
            df = df[df["cluster_size"] >= min_csize]
        col = tcol if tcol in df.columns else next((c for c in FALLBACK_TERM_COLS if c in df.columns), None)
        if col is None:
            print(f"  [{label}] {name}: no term column found, skipped")
            continue
        df = df.assign(_cid=df["cluster_id"].astype(str).str.replace("HiDeF_Cluster_", "", regex=False))
        tested.update(df["_cid"])
        s = df[df["adjusted_p_value"] < adj_thr]
        for cid, term, q in zip(s["_cid"], s[col], s["adjusted_p_value"]):
            if pd.isna(term) or str(term).strip().lower() in ("", "nan", "none"):
                continue
            disp = lollipop_label(term, name)
            key = f"{name}|{disp.lower()}"
            labels.setdefault(key, disp)
            score = -math.log10(max(float(q), 1e-300))
            if score > sig[cid].get(key, -1.0):
                sig[cid][key] = score
                padj[cid][key] = float(q)
        print(f"  [{label}] {name}: {df['_cid'].nunique():,} clusters, {s['_cid'].nunique():,} with "
              f"adjusted p < {adj_thr} (term column '{col}')")
    return dict(sig), dict(padj), tested, labels


def map_ids_to_source(tested_genes, clusters, label, min_containment=0.9):
    """enrichment cluster_id -> source cluster id (exact / normalized ID,
    verified by gene containment, else best gene-content match)."""
    src_ids = list(clusters)
    exact = set(src_ids)
    by_norm = defaultdict(list)
    for c in src_ids:
        by_norm[norm_id(c)].append(c)

    def containment(eid, sid):
        e = tested_genes.get(eid, set())
        return 1.0 if not e else len(e & clusters[sid]["genes"]) / len(e)

    mapping, how, todo = {}, Counter(), []
    for eid in tested_genes:
        cand, method = (eid, "exact") if eid in exact else (None, None)
        if cand is None:
            hits = by_norm.get(norm_id(eid), [])
            if len(hits) == 1:
                cand, method = hits[0], "normalized_id"
        if cand is not None and containment(eid, cand) >= min_containment:
            mapping[eid] = cand
            how[method] += 1
        else:
            if cand is not None:
                how["id_match_failed_gene_check"] += 1
            todo.append(eid)

    if todo:
        genes_all = set().union(*(clusters[c]["genes"] for c in src_ids),
                                *(tested_genes[e] for e in todo))
        gi = {g: i for i, g in enumerate(sorted(genes_all))}
        E = build_incidence([tested_genes[e] for e in todo], gi)
        S = build_incidence([clusters[c]["genes"] for c in src_ids], gi)
        ne = np.asarray(E.sum(axis=0)).ravel().astype(float)
        ns = np.asarray(S.sum(axis=0)).ravel().astype(float)
        ov = (E.T @ S).tocsr()
        for r, eid in enumerate(todo):
            row = ov.getrow(r)
            if row.nnz == 0 or ne[r] == 0:
                how["unmatched"] += 1
                continue
            k = row.data.astype(float)
            cols = row.indices
            cont = k / ne[r]
            jac = k / (ne[r] + ns[cols] - k)
            ok = cont >= min_containment
            if not ok.any():
                how["unmatched"] += 1
                continue
            mapping[eid] = src_ids[cols[ok][np.argmax(jac[ok])]]
            how["gene_content"] += 1

    print(f"[{label}] enrichment-ID -> source-cluster mapping (via gene-level files): {dict(how)}")
    if how.get("unmatched", 0) > 0.2 * max(len(tested_genes), 1):
        print(f"  ⚠ [{label}] >20% of enrichment clusters couldn't be mapped -- check that the "
              f"cluster file and enrichment dir come from the SAME run")
    return mapping


def rekey_to_source(sig, padj, tested, mapping, clusters, label):
    """Join summarized-table cluster IDs onto source clusters through the
    gene-verified mapping (falling back to a direct ID match)."""
    lookup = {}
    for eid, src in mapping.items():
        k = norm_id(eid)
        lookup[k] = src if lookup.get(k, src) == src else None  # None = ambiguous
    by_norm = defaultdict(list)
    for c in clusters:
        by_norm[norm_id(c)].append(c)

    def resolve(cid):
        k = norm_id(cid)
        if lookup.get(k):
            return lookup[k]
        if cid in clusters:
            return cid
        hits = by_norm.get(k, [])
        return hits[0] if len(hits) == 1 else None

    sig_s, padj_s, tested_s, unmapped = defaultdict(dict), defaultdict(dict), set(), []
    for cid in tested:
        src = resolve(cid)
        if src is None:
            unmapped.append(cid)
            continue
        tested_s.add(src)
        for k, v in sig.get(cid, {}).items():
            if v > sig_s[src].get(k, -1.0):
                sig_s[src][k] = v
                padj_s[src][k] = padj[cid][k]
    print(f"[{label}] summarized-table clusters mapped to source clusters: "
          f"{len(tested) - len(unmapped):,}/{len(tested):,}")
    if unmapped:
        print(f"  ⚠ [{label}] unmapped summarized cluster IDs (e.g. {unmapped[:5]}) -- their terms are ignored")
    return dict(sig_s), dict(padj_s), tested_s


def lollipop_tiers(clusters):
    """Persistence panel as the lollipop functions define it: percentiles over
    ALL clusters, lower-inclusive (75-90 panel is 75th <= p < 90th)."""
    vals = [v["persistence"] for v in clusters.values() if not np.isnan(v["persistence"])]
    p75, p90 = np.percentile(vals, 75), np.percentile(vals, 90)
    return {k: ("90+" if v["persistence"] >= p90 else "75-90" if v["persistence"] >= p75 else "<75")
            for k, v in clusters.items()}


# ============================================================================
# similarity
# ============================================================================

def compare_membership(ml_sets, ml_ids, sl_sets, sl_ids, universe):
    gi = {g: i for i, g in enumerate(sorted(universe))}
    A = build_incidence([ml_sets[c] for c in ml_ids], gi)
    B = build_incidence([sl_sets[c] for c in sl_ids], gi)
    na = np.asarray(A.sum(axis=0)).ravel().astype(float)
    nb = np.asarray(B.sum(axis=0)).ravel().astype(float)
    ov = (A.T @ B).tocoo()
    k, i, j = ov.data.astype(float), ov.row, ov.col
    N = len(universe)
    a, b = na[i], nb[j]
    p = hypergeom.sf(k - 1, N, a, b)
    q = bh(p, len(ml_ids) * len(sl_ids))
    print(f"Membership: {len(ml_ids):,} x {len(sl_ids):,} pairs, {len(k):,} with any shared gene, "
          f"{int((q < 0.05).sum()):,} at q < 0.05 (universe {N:,} genes)")
    return pd.DataFrame({
        "ml_cluster": np.asarray(ml_ids, dtype=object)[i],
        "sl_cluster": np.asarray(sl_ids, dtype=object)[j],
        "n_shared_genes": k.astype(int),
        "gene_jaccard": k / (a + b - k),
        "gene_overlap_coef": k / np.minimum(a, b),
        "gene_fold_enrichment": k * N / (a * b),
        "hypergeom_p": p,
        "hypergeom_q": q,
    })


def compare_enrichment(ml_sig, ml_ids, sl_sig, sl_ids):
    ml_e = [c for c in ml_ids if ml_sig.get(c)]
    sl_e = [c for c in sl_ids if sl_sig.get(c)]
    cols = ["ml_cluster", "sl_cluster", "enrich_cosine", "n_shared_terms", "term_jaccard"]
    if not ml_e or not sl_e:
        print("Enrichment: no selected clusters with significant terms on one side -- skipped")
        return pd.DataFrame(columns=cols)

    df_count = Counter()
    for c in ml_e:
        df_count.update(ml_sig[c].keys())
    for c in sl_e:
        df_count.update(sl_sig[c].keys())
    terms = sorted(df_count)
    tidx = {t: n for n, t in enumerate(terms)}
    n_docs = len(ml_e) + len(sl_e)
    idf = np.array([math.log((1 + n_docs) / (1 + df_count[t])) + 1 for t in terms])

    def mats(sig, ids):
        rows, cs, vals = [], [], []
        for r, c in enumerate(ids):
            for t, s in sig[c].items():
                rows.append(r)
                cs.append(tidx[t])
                vals.append(s * idf[tidx[t]])
        X = sp.csr_matrix((vals, (rows, cs)), shape=(len(ids), len(terms)))
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        X = sp.diags(1.0 / np.where(norms > 0, norms, 1.0)) @ X
        Bm = (X > 0).astype(np.int32)
        return X.tocsr(), Bm.tocsr()

    Xa, Ba = mats(ml_sig, ml_e)
    Xb, Bb = mats(sl_sig, sl_e)
    cos = (Xa @ Xb.T).tocoo()
    shared = (Ba @ Bb.T).tocsr()
    s = np.asarray(shared[cos.row, cos.col]).ravel().astype(float)
    na = np.asarray(Ba.sum(axis=1)).ravel()[cos.row]
    nb = np.asarray(Bb.sum(axis=1)).ravel()[cos.col]
    print(f"Enrichment: {len(ml_e):,} x {len(sl_e):,} clusters with significant terms, "
          f"{len(terms):,} distinct terms, {cos.nnz:,} pairs share >= 1 term")
    return pd.DataFrame({
        "ml_cluster": np.asarray(ml_e, dtype=object)[cos.row],
        "sl_cluster": np.asarray(sl_e, dtype=object)[cos.col],
        "enrich_cosine": cos.data,
        "n_shared_terms": s.astype(int),
        "term_jaccard": s / (na + nb - s),
    })


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ml-nodes", required=True, help="multiplex .nodes from step 11c Phase 2")
    ap.add_argument("--ml-supra-graph", default=None,
                    help="step 11a supra_graph.pkl (gene universe + index->name resolution)")
    ap.add_argument("--ml-enrich-dir", default=None,
                    help="multiplex phen output/<PPI_SOURCE> dir holding *_analysis_* folders")
    ap.add_argument("--ml-cluster-type", default="*", help="CLUSTER_TYPE from multiplex pipeline_config")
    ap.add_argument("--ml-dir-filter", default=None, help="substring to pick among several analysis dirs")
    ap.add_argument("--ml-min-persistence", type=float, default=None)

    ap.add_argument("--sl-communities", required=True,
                    help="single-layer hidef_communities.pkl (or hidef_results.pkl)")
    ap.add_argument("--sl-network", default=None, help="single-layer module_graph.pkl (gene universe)")
    ap.add_argument("--sl-enrich-dir", default=None,
                    help="single-layer phen output/<ppi_source> dir holding *_analysis_* folders")
    ap.add_argument("--sl-cluster-type", default="hidef_from_comparison")
    ap.add_argument("--sl-dir-filter", default=None)
    ap.add_argument("--sl-min-persistence", type=float, default=None)

    ap.add_argument("--persistence-quantile", type=float, default=0.75,
                    help="per-side quantile used when no absolute cutoff is given (default 0.75)")
    ap.add_argument("--min-size", type=int, default=4, help="min distinct genes per cluster")
    ap.add_argument("--max-size", type=int, default=None, help="max distinct genes (drops near-root clusters)")
    ap.add_argument("--adj-p", type=float, default=0.05,
                    help="adjusted_p_value cutoff, as the lollipop's significance_threshold (default 0.05)")
    ap.add_argument("--enrich-min-cluster-size", type=int, default=5,
                    help="cluster_size filter on summarized tables, as the lollipop's min_cluster_size (default 5)")
    ap.add_argument("--generic-analyses", nargs="*", default=["proteomics", "neuropathic_pain"],
                    help="analyses whose shared terms alone don't count as an enrichment match")
    ap.add_argument("--fdr", type=float, default=0.05, help="BH q cutoff for gene-overlap significance")
    ap.add_argument("--min-enrich-cosine", type=float, default=0.3)
    ap.add_argument("--min-shared-terms", type=int, default=1,
                    help="specific (non-generic) shared terms required for a convergent pair (default 1)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    generic = set(args.generic_analyses)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from provenance import write_manifest
        write_manifest(out, args, inputs=[p for p in [args.ml_nodes, args.ml_supra_graph,
                                                       args.sl_communities, args.sl_network] if p])
    except Exception as e:  # provenance is optional here
        print(f"(provenance manifest not written: {e})")

    # ---------------- clusters ----------------
    ml, ml_univ = load_ml_clusters(args.ml_nodes, args.ml_supra_graph)
    sl, sl_univ = load_sl_clusters(args.sl_communities, args.sl_network)
    ml_univ = ml_univ or set().union(*(c["genes"] for c in ml.values()))
    sl_univ = sl_univ or set().union(*(c["genes"] for c in sl.values()))
    universe = ml_univ & sl_univ
    print(f"\nShared gene universe: {len(universe):,} "
          f"(multiplex {len(ml_univ):,}, single-layer {len(sl_univ):,})\n")
    if not universe:
        raise SystemExit("No genes shared between the two networks -- check symbol format/case.")

    ml_ids, ml_pct = select_high_persistence(ml, args.ml_min_persistence, args.persistence_quantile,
                                             args.min_size, args.max_size, "multiplex")
    sl_ids, sl_pct = select_high_persistence(sl, args.sl_min_persistence, args.persistence_quantile,
                                             args.min_size, args.max_size, "single-layer")
    ml_tier, sl_tier = lollipop_tiers(ml), lollipop_tiers(sl)
    ml_sets = {c: ml[c]["genes"] & universe for c in ml_ids}
    sl_sets = {c: sl[c]["genes"] & universe for c in sl_ids}
    print()

    # ---------------- enrichment (lollipop tables) ----------------
    labels = {}
    sides = {}
    for side, root, ctype, dfilt, clusters in [
            ("multiplex", args.ml_enrich_dir, args.ml_cluster_type, args.ml_dir_filter, ml),
            ("single-layer", args.sl_enrich_dir, args.sl_cluster_type, args.sl_dir_filter, sl)]:
        if not root:
            print(f"[{side}] no enrichment dir given -- enrichment axis skipped")
            sides[side] = ({}, {}, set())
            continue
        sig_raw, padj_raw, tested_raw, lab = load_summarized(root, ctype, dfilt, args.adj_p,
                                                             args.enrich_min_cluster_size, side)
        labels.update({k: v for k, v in lab.items() if k not in labels})
        tg = load_gene_level_ids(root, ctype, dfilt, side)
        mapping = map_ids_to_source(tg, clusters, side) if tg else {}
        sides[side] = rekey_to_source(sig_raw, padj_raw, tested_raw, mapping, clusters, side)
        print()
    ml_sig, ml_padj, ml_tested = sides["multiplex"]
    sl_sig, sl_padj, sl_tested = sides["single-layer"]
    print(f"Selected clusters in the lollipop tables: multiplex "
          f"{sum(c in ml_tested for c in ml_ids):,}/{len(ml_ids):,} "
          f"({sum(bool(ml_sig.get(c)) for c in ml_ids):,} with >= 1 significant term), single-layer "
          f"{sum(c in sl_tested for c in sl_ids):,}/{len(sl_ids):,} "
          f"({sum(bool(sl_sig.get(c)) for c in sl_ids):,} with >= 1 significant term)\n")

    # ---------------- similarity ----------------
    mem = compare_membership(ml_sets, ml_ids, sl_sets, sl_ids, universe)
    enr = compare_enrichment(ml_sig, ml_ids, sl_sig, sl_ids)

    mem_keep = mem.loc[mem["hypergeom_q"] < args.fdr, ["ml_cluster", "sl_cluster"]]
    enr_keep = enr.loc[enr["enrich_cosine"] >= args.min_enrich_cosine, ["ml_cluster", "sl_cluster"]]
    keys = pd.concat([mem_keep, enr_keep]).drop_duplicates()
    comb = keys.merge(mem, how="left", on=["ml_cluster", "sl_cluster"]) \
               .merge(enr, how="left", on=["ml_cluster", "sl_cluster"])
    comb["n_shared_genes"] = comb["n_shared_genes"].fillna(0).astype(int)
    for c in ["gene_jaccard", "gene_overlap_coef", "gene_fold_enrichment"]:
        comb[c] = comb[c].fillna(0.0)
    for c in ["hypergeom_p", "hypergeom_q"]:
        comb[c] = comb[c].fillna(1.0)
    comb["ml_tested"] = comb["ml_cluster"].isin(ml_tested)
    comb["sl_tested"] = comb["sl_cluster"].isin(sl_tested)
    both_tested = comb["ml_tested"] & comb["sl_tested"]
    comb["n_shared_terms"] = comb["n_shared_terms"].fillna(0).astype(int)
    comb["enrich_cosine"] = comb["enrich_cosine"].where(comb["enrich_cosine"].notna(),
                                                        np.where(both_tested, 0.0, np.nan))
    comb["term_jaccard"] = comb["term_jaccard"].where(comb["term_jaccard"].notna(),
                                                      np.where(both_tested, 0.0, np.nan))

    # reciprocal best hits on gene Jaccard (over ALL overlapping pairs)
    best_ml = mem.loc[mem.groupby("ml_cluster")["gene_jaccard"].idxmax(), ["ml_cluster", "sl_cluster"]]
    best_sl = mem.loc[mem.groupby("sl_cluster")["gene_jaccard"].idxmax(), ["ml_cluster", "sl_cluster"]]
    rbh = set(map(tuple, best_ml.merge(best_sl).values))
    comb["reciprocal_best_hit"] = [(a, b) in rbh for a, b in zip(comb["ml_cluster"], comb["sl_cluster"])]

    # ---------------- annotation ----------------
    def lab(k):
        return labels.get(k, k.split("|", 1)[1])

    ann = defaultdict(list)
    for mc, sc in zip(comb["ml_cluster"], comb["sl_cluster"]):
        m, s = ml[mc], sl[sc]
        shared = sorted(ml_sets[mc] & sl_sets[sc])
        ct_shared = Counter(ct for gname in shared for ct in m["gene_cts"].get(gname, ()))
        ms, ss = ml_sig.get(mc, {}), sl_sig.get(sc, {})
        sh = sorted(set(ms) & set(ss))
        sh_specific = [k for k in sh if k.split("|", 1)[0] not in generic]
        a_ml = {k.split("|", 1)[0] for k in ms}
        a_sl = {k.split("|", 1)[0] for k in ss}
        both_a = a_ml & a_sl
        agree_a = {k.split("|", 1)[0] for k in sh}
        tot = sum(m["ct_counts"].values()) or 1
        ann["n_shared_specific_terms"].append(len(sh_specific))
        ann["shared_terms"].append("; ".join(
            f"{lab(k)} [SL q={sl_padj[sc][k]:.1e}, ML q={ml_padj[mc][k]:.1e}]" for k in sh))
        ann["sl_lollipop_terms"].append("; ".join(f"{lab(k)} [q={sl_padj[sc][k]:.1e}]" for k in sorted(ss)))
        ann["ml_lollipop_terms"].append("; ".join(f"{lab(k)} [q={ml_padj[mc][k]:.1e}]" for k in sorted(ms)))
        ann["analyses_both_significant"].append(",".join(sorted(both_a)))
        ann["analyses_agreeing"].append(",".join(sorted(agree_a)))
        ann["analyses_disagreeing"].append(",".join(sorted(both_a - agree_a)))
        ann["ml_persistence"].append(m["persistence"])
        ann["sl_persistence"].append(s["persistence"])
        ann["ml_persistence_pct"].append(round(ml_pct.get(mc, np.nan), 3))
        ann["sl_persistence_pct"].append(round(sl_pct.get(sc, np.nan), 3))
        ann["ml_lollipop_panel"].append(ml_tier.get(mc))
        ann["sl_lollipop_panel"].append(sl_tier.get(sc))
        ann["ml_n_nodes"].append(m["n_nodes"])
        ann["ml_n_genes"].append(len(m["genes"]))
        ann["sl_n_genes"].append(len(s["genes"]))
        ann["ml_n_cell_types"].append(len(m["ct_counts"]))
        ann["ml_top_cell_types"].append("; ".join(f"{ct}:{n / tot:.2f}" for ct, n in m["ct_counts"].most_common(3)))
        ann["shared_gene_cell_types"].append("; ".join(f"{ct}:{n}" for ct, n in ct_shared.most_common()))
        ann["shared_genes_first50"].append(",".join(shared[:50]))
    for k, v in ann.items():
        comb[k] = v

    mem_sig = comb["hypergeom_q"] < args.fdr
    enr_match = (comb["n_shared_specific_terms"] >= 1) & (comb["enrich_cosine"].fillna(0) >= args.min_enrich_cosine)
    comb["category"] = np.select(
        [mem_sig & enr_match,
         mem_sig & (comb["n_shared_terms"] > 0),
         mem_sig & both_tested,
         mem_sig,
         enr_match & (comb["n_shared_specific_terms"] >= args.min_shared_terms)],
        CATEGORY_ORDER, default="other")
    comb = comb[comb["category"] != "other"].copy()
    comb["combined_score"] = np.sqrt(comb["gene_jaccard"] * comb["enrich_cosine"].fillna(0))

    comb["_cat"] = comb["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)})
    comb = comb.sort_values(["_cat", "combined_score", "gene_jaccard", "enrich_cosine"],
                            ascending=[True, False, False, False]).drop(columns="_cat")
    lead = ["category", "ml_cluster", "sl_cluster", "reciprocal_best_hit", "combined_score",
            "gene_jaccard", "gene_overlap_coef", "n_shared_genes", "hypergeom_q",
            "enrich_cosine", "n_shared_terms", "n_shared_specific_terms", "shared_terms",
            "sl_lollipop_terms", "ml_lollipop_terms", "sl_lollipop_panel", "ml_lollipop_panel",
            "ml_persistence", "ml_persistence_pct", "sl_persistence", "sl_persistence_pct",
            "ml_n_genes", "sl_n_genes", "ml_n_cell_types", "ml_top_cell_types",
            "shared_gene_cell_types", "shared_genes_first50"]
    comb = comb[lead + [c for c in comb.columns if c not in lead]]
    comb.to_csv(out / "matched_pairs.csv", index=False)
    comb[comb["reciprocal_best_hit"] & (comb["hypergeom_q"] < args.fdr)] \
        .to_csv(out / "reciprocal_best_hits.csv", index=False)

    # ---------------- best match per cluster ----------------
    def best_tables(side, other):
        rows = []
        for cid, grp in comb.groupby(f"{side}_cluster"):
            r = {f"{side}_cluster": cid,
                 f"{side}_persistence": grp[f"{side}_persistence"].iloc[0],
                 f"{side}_lollipop_panel": grp[f"{side}_lollipop_panel"].iloc[0],
                 f"{side}_n_genes": grp[f"{side}_n_genes"].iloc[0],
                 f"{side}_lollipop_terms": grp[f"{side}_lollipop_terms"].iloc[0]}
            g_sig = grp[grp["hypergeom_q"] < args.fdr]
            if len(g_sig):
                b = g_sig.loc[g_sig["gene_jaccard"].idxmax()]
                r.update({f"best_{other}_by_genes": b[f"{other}_cluster"],
                          "genes_jaccard": b["gene_jaccard"], "genes_q": b["hypergeom_q"],
                          "genes_pair_shared_terms": b["shared_terms"],
                          "genes_pair_shared_genes": b["shared_genes_first50"]})
            g_enr = grp[grp["n_shared_specific_terms"] > 0]
            if len(g_enr):
                b = g_enr.loc[g_enr["enrich_cosine"].idxmax()]
                r.update({f"best_{other}_by_enrichment": b[f"{other}_cluster"],
                          "enrich_cosine": b["enrich_cosine"], "enrich_shared_terms": b["shared_terms"],
                          "enrich_pair_gene_jaccard": b["gene_jaccard"]})
            r["n_matches"] = len(grp)
            r["categories"] = ",".join(f"{k}:{v}" for k, v in grp["category"].value_counts().items())
            rows.append(r)
        return pd.DataFrame(rows).sort_values(f"{side}_persistence", ascending=False) if rows else pd.DataFrame()

    best_tables("sl", "ml").to_csv(out / "best_match_per_singlelayer.csv", index=False)
    best_tables("ml", "sl").to_csv(out / "best_match_per_multiplex.csv", index=False)

    # ---------------- term-level summary ----------------
    ml_term_n = Counter(t for c in ml_ids for t in ml_sig.get(c, {}))
    sl_term_n = Counter(t for c in sl_ids for t in sl_sig.get(c, {}))
    term_rows = [{"analysis": t.split("|", 1)[0], "lollipop_label": lab(t),
                  "n_multiplex_clusters": ml_term_n.get(t, 0),
                  "n_singlelayer_clusters": sl_term_n.get(t, 0),
                  "in_both": t in ml_term_n and t in sl_term_n}
                 for t in sorted(set(ml_term_n) | set(sl_term_n))]
    if term_rows:
        pd.DataFrame(term_rows).sort_values(["in_both", "n_singlelayer_clusters", "n_multiplex_clusters"],
                                            ascending=False).to_csv(out / "shared_terms_summary.csv", index=False)

    for side, ids, cl, pct, tested, sig, tier in [("ml", ml_ids, ml, ml_pct, ml_tested, ml_sig, ml_tier),
                                                  ("sl", sl_ids, sl, sl_pct, sl_tested, sl_sig, sl_tier)]:
        pd.DataFrame([{"cluster_id": c, "persistence": cl[c]["persistence"],
                       "persistence_pct": pct.get(c), "lollipop_panel": tier.get(c),
                       "n_genes": len(cl[c]["genes"]), "in_lollipop_tables": c in tested,
                       "lollipop_terms": "; ".join(lab(k) for k in sorted(sig.get(c, {})))} for c in ids]) \
            .to_csv(out / f"selected_clusters_{side}.csv", index=False)

    # ---------------- plot ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 6))
        colors = dict(zip(CATEGORY_ORDER, ["#C44E52", "#DD8452", "#4C72B0", "#8C8C8C", "#55A868"]))
        plot_df = comb.dropna(subset=["enrich_cosine"])
        for cat in CATEGORY_ORDER:
            d = plot_df[plot_df["category"] == cat]
            if len(d):
                ax.scatter(d["gene_jaccard"], d["enrich_cosine"], s=12, alpha=0.6,
                           color=colors[cat], label=f"{cat} (n={len(d):,})", edgecolors="none")
        r = plot_df[plot_df["reciprocal_best_hit"]]
        ax.scatter(r["gene_jaccard"], r["enrich_cosine"], s=40, facecolors="none",
                   edgecolors="black", linewidths=0.8, label=f"reciprocal best hit (n={len(r):,})")
        ax.set_xlabel("Gene-membership Jaccard (multiplex collapsed to genes)")
        ax.set_ylabel("Enrichment cosine (lollipop terms, adjusted p, IDF-weighted)")
        ax.set_title("High-persistence multiplex vs single-layer cluster pairs")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(out / "similarity_scatter.png", dpi=170)
        plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")

    # ---------------- report ----------------
    print("\n" + "=" * 70)
    print("Pair categories:")
    for cat, n in comb["category"].value_counts().reindex(CATEGORY_ORDER).dropna().items():
        print(f"  {cat:34s} {int(n):,}")
    n_rbh = int((comb["reciprocal_best_hit"] & (comb["hypergeom_q"] < args.fdr)).sum())
    print(f"  reciprocal best hits (q < {args.fdr})   {n_rbh:,}")
    for cat, title in [("same_module_same_enrichment", "Same module, same lollipop term"),
                       ("convergent_enrichment", "Same lollipop term, different genes")]:
        top = comb[comb["category"] == cat].head(15)
        if len(top):
            print(f"\n{title}:")
            for r in top.itertuples(index=False):
                print(f"  ML {r.ml_cluster} <-> SL {r.sl_cluster} (SL panel {r.sl_lollipop_panel}): "
                      f"Jaccard={r.gene_jaccard:.2f} cos={r.enrich_cosine:.2f} | {r.shared_terms[:110]}")
    print(f"\nOutputs in {out}/")
    print("=" * 70)


if __name__ == "__main__":
    main()