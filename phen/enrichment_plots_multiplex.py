#!/usr/bin/env python3
"""
enrichment_plots_multiplex.py

Same lollipop + heatmap plotting functions as your original script, with
persistence pulled from EITHER source:
  - hidef_source='pickle'    -> old monoplex hidef_communities.pkl (unchanged behavior)
  - hidef_source='multiplex' -> new heterogeneous .nodes file, via
                                 hidef_multiplex_adapter.load_multiplex_clusters()

CHANGES IN THIS VERSION (vs. the one that produced the 50th-percentile-turned-
90th-percentile bug):

1. Percentile-vs-floor decoupling (THE BUG FIX):
   Previously, `persistence_threshold` was passed straight into the multiplex
   loader, which filtered clusters out of the persistence map *before*
   `filter_persistence_by_percentile()` ever saw them. That meant the "50th
   percentile" cutoff was computed against whatever was left after that
   floor -- so raising PERSISTENCE_THRESHOLD elsewhere (e.g. 5 -> 8) silently
   shrank the reference population and pushed the real cutoff much higher
   than 50%, with no visible warning.

   Now, both `create_persistence_lollipop_plot()` and
   `create_enrichment_heatmap()` always load the FULL persistence population
   (persistence_threshold=0 at the loader level) and then apply your
   `persistence_threshold` as an explicit, separate floor via
   `apply_persistence_floor()`, applied BEFORE any percentile math. Percentile
   cutoffs are therefore always computed against a population whose size you
   can see and control directly, rather than one that quietly depends on a
   threshold set somewhere else in your pipeline.

2. No more PDF output -- only PNG (dpi=300) is saved, since that's all you
   actually use.

3. Heatmap filenames no longer hardcode "_phenotype_spatial" (a leftover
   from when the heatmap only ever covered those two analysis types --
   misleading once you started passing all six). Filenames are now just
   `enrichment_heatmap_{cluster_type}_{value_col}{filename_suffix}.png`.

4. The lollipop plot auto-sizes its figure (when figsize=None) based on how
   many clusters/terms actually end up on the axes, and uses slightly wider
   spacing between clusters and between persistence-tier groups, so dense
   plots don't overlap.

Everything else (tier thresholds, term cleaning, bracket annotations, the
CEMM color palette, etc.) is untouched.
"""
import os
import re
import math
import pickle
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hidef_multiplex_adapter import load_multiplex_clusters


# ============================================================================
# PERSISTENCE LOADING
# ============================================================================

def load_hidef_results(hidef_results_file):
    """Load HiDeF results from pickle file (monoplex pipeline)."""
    if not os.path.exists(hidef_results_file):
        print(f"Error: HiDeF results file not found: {hidef_results_file}")
        return None

    with open(hidef_results_file, 'rb') as f:
        hidef_results = pickle.load(f)

    return hidef_results


def extract_persistent_communities(hidef_results, min_persistence=5):
    """Extract the most persistent communities from HiDeF results (monoplex pipeline)."""
    communities = hidef_results.get('communities', hidef_results)

    persistent_communities = {}
    for comm_id, comm_data in communities.items():
        persistence = comm_data.get('persistence', 0)
        if persistence >= min_persistence:
            persistent_communities[comm_id] = comm_data

    return persistent_communities


def load_persistence_map(hidef_source='pickle',
                          hidef_pickle_file="../output/combined26/hidef/hidef_communities.pkl",
                          nodes_file=None, supra_graph_pkl=None, seed_genes_file=None,
                          persistence_threshold=0):
    """
    Returns {clean_cluster_id: persistence_value} (keys already have the
    "HiDeF_Cluster_" prefix stripped, matching what both plot functions
    expect) regardless of which pipeline produced the clustering.

    hidef_source='pickle'    : reads hidef_pickle_file (old monoplex path)
    hidef_source='multiplex' : reads nodes_file via load_multiplex_clusters()
                                (new heterogeneous pipeline). supra_graph_pkl
                                is only needed if your .nodes members are
                                integer indices rather than "ct::gene"
                                strings; seed_genes_file is optional and only
                                affects the (unused here) seed mapping.

    NOTE: `persistence_threshold` here is forwarded straight to the
    underlying loader (which, for the multiplex source, may drop clusters
    below it before you ever see them). The plot functions below no longer
    call this with a nonzero threshold directly -- they always load the full
    population (threshold=0) and apply your requested floor themselves via
    `apply_persistence_floor()`, so that percentile calculations stay
    consistent regardless of what floor you ask for. Call this function
    directly with a nonzero threshold only if you deliberately want the
    loader-level filtering (e.g. to mirror exactly what an analyzer step saw).
    """
    if hidef_source == 'pickle':
        if not hidef_pickle_file or not os.path.exists(hidef_pickle_file):
            print(f"Error: HiDeF results file not found: {hidef_pickle_file}")
            return None
        hidef_results = load_hidef_results(hidef_pickle_file)
        if hidef_results is None:
            return None
        persistent_communities = extract_persistent_communities(hidef_results, min_persistence=0)
        persistence_map = {}
        for k, v in persistent_communities.items():
            cluster_id = f"HiDeF_Cluster_{k}" if isinstance(k, int) else str(k)
            persistence_map[cluster_id.replace('HiDeF_Cluster_', '')] = v['persistence']
        return persistence_map

    elif hidef_source == 'multiplex':
        if not nodes_file or not os.path.exists(nodes_file):
            print(f"Error: multiplex .nodes file not found: {nodes_file}")
            return None

        node_names = None
        if supra_graph_pkl:
            with open(supra_graph_pkl, 'rb') as f:
                g = pickle.load(f)
            node_names = g.vs['name']

        seed_genes = None
        if seed_genes_file:
            seed_genes = set(pd.read_csv(seed_genes_file)['gene'].astype(str))

        _, _, extra = load_multiplex_clusters(
            nodes_file, node_names=node_names, seed_genes=seed_genes,
            persistence_threshold=persistence_threshold,
        )
        persistence_map = {
            cluster_id.replace('HiDeF_Cluster_', ''): persistence
            for cluster_id, persistence in extra['persistence'].items()
        }
        return persistence_map

    else:
        raise ValueError(f"Unknown hidef_source: {hidef_source!r} (expected 'pickle' or 'multiplex')")


def apply_persistence_floor(persistence_map, min_persistence_floor=0):
    """
    Explicit, self-contained floor filter applied AFTER loading the full
    persistence population (persistence_threshold=0 at the loader level).

    This exists so that percentile-based tier cutoffs are always computed
    against a population size you can see printed below -- rather than one
    that silently depends on a threshold baked into some other pipeline step.
    A floor of 0 (the default) is a no-op.
    """
    if not persistence_map or not min_persistence_floor:
        return persistence_map
    filtered = {k: v for k, v in persistence_map.items() if v >= min_persistence_floor}
    print(f"Persistence floor: keeping >= {min_persistence_floor} "
          f"({len(filtered)}/{len(persistence_map)} clusters)")
    return filtered


def select_top_persistence_clusters(persistence_map, top_n_values=None, top_n_clusters=None):
    """
    Pick out the top-persistence clusters, two ways -- set exactly ONE:

      top_n_values   : every cluster whose persistence is among the N highest
                        DISTINCT persistence values. top_n_values=1 gets you
                        every cluster tied at the single max value.
      top_n_clusters : exactly this many clusters, ranked by persistence
                        (ties broken by cluster_id).

    Returns a list of cluster ids, sorted by persistence descending.
    """
    assert (top_n_values is None) != (top_n_clusters is None), \
        "Set exactly one of top_n_values or top_n_clusters, leave the other as None"

    if not persistence_map:
        return []

    if top_n_values is not None:
        distinct_values_desc = sorted(set(persistence_map.values()), reverse=True)
        cutoff_values = set(distinct_values_desc[:top_n_values])
        top_ids = [cid for cid, v in persistence_map.items() if v in cutoff_values]
    else:
        top_ids = sorted(persistence_map, key=lambda cid: (-persistence_map[cid], cid))[:top_n_clusters]

    return sorted(top_ids, key=lambda cid: -persistence_map[cid])


def filter_persistence_by_percentile(persistence_map, min_percentile=50):
    """
    Keep only clusters at or above the Nth percentile of persistence,
    computed against the population passed in (call `apply_persistence_floor`
    first if you want a floor applied before the percentile math).

    Returns (filtered_map, cutoff_value).
    """
    if not persistence_map:
        return persistence_map, None
    all_values = list(persistence_map.values())
    cutoff = np.percentile(all_values, min_percentile)
    filtered = {k: v for k, v in persistence_map.items() if v >= cutoff}
    print(f"Persistence filter: keeping >= {min_percentile}th percentile "
          f"(persistence >= {cutoff:.2f}): {len(filtered)}/{len(all_values)} clusters")
    return filtered, cutoff


def filter_persistence_by_tier(persistence_map, thresholds, tier_labels, tier):
    """
    Restrict persistence_map to exactly ONE persistence tier, using
    thresholds/tier_labels from `compute_persistence_tier_thresholds()` --
    i.e. tier boundaries fixed against a reference population, not
    recomputed from whatever's already been filtered.

    tier : 'tier1', 'tier2', ... -- 1-indexed position in tier_labels,
        highest persistence first (so 'tier1' is always the top tier,
        regardless of how many tiers there are) -- or the literal label
        string itself (e.g. 'Tier 3 (50th-75th)').
    """
    if not persistence_map:
        return persistence_map

    tier_label = tier
    m = re.match(r'^tier(\d+)$', str(tier).strip(), flags=re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if not (0 <= idx < len(tier_labels)):
            raise ValueError(f"tier_filter={tier!r} is out of range -- "
                              f"this population only has {len(tier_labels)} tiers: {tier_labels}")
        tier_label = tier_labels[idx]
    elif tier not in tier_labels:
        raise ValueError(f"tier_filter={tier!r} doesn't match any tier label: {tier_labels}")

    value_to_tier = assign_persistence_tiers(sorted(set(persistence_map.values())), thresholds, tier_labels)
    filtered = {cid: v for cid, v in persistence_map.items() if value_to_tier.get(v) == tier_label}
    print(f"Persistence tier filter: keeping {tier_label}: {len(filtered)}/{len(persistence_map)} clusters")
    return filtered


def filter_persistence_below_percentile(persistence_map, max_percentile=50):
    """
    Keep only clusters STRICTLY BELOW the Nth percentile of persistence,
    computed against the population passed in -- the complement of
    `filter_persistence_by_percentile`. Uses the same np.percentile cutoff
    convention (and the same >= / < boundary split) so a cluster is never
    double-counted or dropped between an "above" cut and a "below" cut of
    the same percentile.

    Returns (filtered_map, cutoff_value).
    """
    if not persistence_map:
        return persistence_map, None
    all_values = list(persistence_map.values())
    cutoff = np.percentile(all_values, max_percentile)
    filtered = {k: v for k, v in persistence_map.items() if v < cutoff}
    print(f"Persistence filter: keeping < {max_percentile}th percentile "
          f"(persistence < {cutoff:.2f}): {len(filtered)}/{len(all_values)} clusters")
    return filtered, cutoff


def compute_persistence_tier_thresholds(persistence_values, percentiles=[90, 75, 50, 25]):
    """
    Compute the persistence value at each requested percentile (sorted
    descending), plus the tier label each boundary produces. Works for any
    number of boundaries -- N percentiles produce N+1 tiers, e.g.
    [90, 75, 50, 25] -> 'Tier 1 (>=90th)', 'Tier 2 (75th-90th)',
    'Tier 3 (50th-75th)', 'Tier 4 (25th-50th)', 'Tier 5 (<25th)'.

    Call this ONCE against your reference population -- e.g. the population
    BEFORE any percentile-based inclusion filter narrows it down -- then
    reuse the same (thresholds, tier_labels) via `assign_persistence_tiers()`
    for whatever subset actually ends up on the plot.

    This split exists because computing thresholds AFTER filtering (the
    original behavior) means "Tier 3" secretly means "that percentile range
    of whatever subset survived filtering" rather than of the real
    population -- which is how a "50th percentile and above" plot could
    still show clusters labeled "<50th": the label was relative to an
    already-filtered, higher-floor population, not the true distribution.

    Returns (thresholds, tier_labels).
    """
    if not persistence_values:
        return [], []

    percentiles_sorted = sorted(set(percentiles), reverse=True)
    thresholds = [np.percentile(persistence_values, p) for p in percentiles_sorted]

    print(f"\nPersistence Tier Thresholds (Percentiles {percentiles_sorted}):")
    for p, t in zip(percentiles_sorted, thresholds):
        print(f"  {p}th percentile: {t:.2f}")

    tier_labels = []
    for i, p in enumerate(percentiles_sorted):
        if i == 0:
            tier_labels.append(f'Tier {i + 1} (>={p}th)')
        else:
            tier_labels.append(f'Tier {i + 1} ({p}th-{percentiles_sorted[i - 1]}th)')
    tier_labels.append(f'Tier {len(percentiles_sorted) + 1} (<{percentiles_sorted[-1]}th)')

    return thresholds, tier_labels


def assign_persistence_tiers(persistence_values, thresholds, tier_labels):
    """
    Label each value with a tier, using (thresholds, tier_labels) computed
    elsewhere (see `compute_persistence_tier_thresholds`) rather than
    recomputed from `persistence_values` itself. This is what keeps tier
    labels honest when `persistence_values` is already a filtered subset of
    a larger reference population -- see the two plot functions below for
    the pattern.

    thresholds : sorted-descending list of persistence cutoffs.
    tier_labels : one longer than thresholds -- tier_labels[i] applies to
        values >= thresholds[i] (and < thresholds[i-1] if i > 0);
        tier_labels[-1] applies to anything below the last threshold.
    """
    tier_mapping = {}
    for value in persistence_values:
        assigned = tier_labels[-1]
        for t, label in zip(thresholds, tier_labels):
            if value >= t:
                assigned = label
                break
        tier_mapping[value] = assigned
    return tier_mapping


def get_persistence_tiers(persistence_values, percentiles=[90, 75, 50, 25]):
    """
    Back-compat convenience wrapper: computes thresholds AND assigns tiers
    from the SAME population passed in. The two plot functions below no
    longer call this directly -- they call `compute_persistence_tier_thresholds`
    once on the reference population and `assign_persistence_tiers` separately
    on whatever filtered subset is actually being plotted, so tier labels stay
    correct regardless of which slice of the data you're looking at.
    """
    if not persistence_values:
        return {}, []
    thresholds, tier_labels = compute_persistence_tier_thresholds(persistence_values, percentiles)
    tier_mapping = assign_persistence_tiers(persistence_values, thresholds, tier_labels)
    return tier_mapping, thresholds

def wrap_label(text, width=40):
    """
    Wrap a long label onto multiple lines instead of leaving it as one very
    long line. Long labels -- particularly Orphanet disease names, which
    routinely run 60-100+ characters -- otherwise blow out the horizontal
    extent of the axis label area far beyond the plot itself. Because
    plt.savefig(..., bbox_inches='tight') expands the saved PNG to fit every
    artist including tick labels, a handful of long Orphanet names can make
    the final image many times wider than intended -- which is what makes
    the *whole plot* look squished and tiny when viewed at a normal zoom
    level (the pixel dimensions ballooned, the content didn't).
    Only breaks on whitespace (never mid-word), so gene/disease names stay
    intact. A label with no whitespace longer than `width` is left as-is.
    """
    if not isinstance(text, str) or len(text) <= width:
        return text
    wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return '\n'.join(wrapped) if wrapped else text

def cap_terms_by_cluster_breadth(df, term_col, cluster_id_col='cluster_id',
                                  max_terms=10, p_value_col='adjusted_p_value'):
    """
    Restrict df to at most `max_terms` distinct values of term_col, chosen by
    how many distinct clusters each term co-occurs with (breadth) rather than
    by p_value/effect size. This is what stops a single cluster with many
    highly-significant Orphanet terms from eating all the plot's term slots
    at the expense of coverage across other clusters -- a term that shows up
    across 6 clusters outranks a term that's only in 1 cluster, no matter how
    significant that 1-cluster hit is. Ties on breadth are broken by the
    term's best (lowest) p_value_col, only as a last resort.
    """
    if term_col not in df.columns or df.empty:
        return df

    breadth = df.groupby(term_col)[cluster_id_col].nunique()
    if p_value_col in df.columns:
        best_p = df.groupby(term_col)[p_value_col].min()
    else:
        best_p = pd.Series(0, index=breadth.index)

    ranking = pd.DataFrame({'breadth': breadth, 'best_p': best_p})
    ranking = ranking.sort_values(['breadth', 'best_p'], ascending=[False, True])
    top_terms = ranking.head(max_terms).index

    kept = df[df[term_col].isin(top_terms)]
    print(f"  Orphanet term cap: kept {len(top_terms)}/{df[term_col].nunique()} terms "
          f"(ranked by # distinct clusters, not significance) "
          f"-> covering {kept[cluster_id_col].nunique()} clusters")
    return kept

def clean_enrichment_term(term, analysis_type):
    """
    Clean an enrichment term:
    - Remove underscores
    - Strip ORPHA suffix for orphanet terms
    - Add group label in parentheses
    """
    group_label_map = {
        'spatial': 'Spatial',
        'phenotype': 'Phenotype',
        'omim_disease': 'OMIM Disease',
        'omim_ps': 'OMIM PS',
        'orphanet_enrichr': 'Orphanet',
        'proteomics': 'Proteomics',
        'neuropathic_pain': 'Neuropathic Pain',
    }

    term = term.replace('_', ' ')

    if analysis_type == 'orphanet_enrichr':
        term = re.sub(r'\s*ORPHA[:\s]\d+.*$', '', term, flags=re.IGNORECASE).strip()

    group_label = group_label_map.get(analysis_type, analysis_type.replace('_', ' ').title())
    return f"{term} ({group_label})"


def get_enrichment_term_column(df, analysis_type):
    """Get the appropriate enrichment term column based on analysis type."""

    term_mapping = {
        'spatial': 'top_cell_type',
        'omim_disease': 'top_disease',
        'omim_ps': 'phenotypic_series',
        'orphanet_enrichr': 'cluster_top_disease',
        'phenotype': 'top_phenotype',
        'proteomics': 'higher_expression_gender',
        'neuropathic_pain': 'top_expression'
    }

    expected_col = term_mapping.get(analysis_type, 'term')
    available_cols = df.columns.tolist()

    if expected_col in available_cols:
        return expected_col

    fallback_cols = ['top_disease', 'top_phenotype', 'top_cell_type', 'term', 'disease', 'phenotype', 'cell_type']

    for col in fallback_cols:
        if col in available_cols:
            return col

    return None


# ============================================================================
# LOLLIPOP PLOT
# ============================================================================

def create_persistence_lollipop_plot(output_dir, ppi_source, cluster_type='hidef_from_comparison',
                                     analysis_types=['phenotype', 'spatial'],
                                     significance_threshold=0.05,
                                     min_cluster_size=10,
                                     persistence_percentiles=[90, 75, 50],
                                     figsize=None,
                                     min_persistence_percentile=50,
                                     max_persistence_percentile=None,
                                     tier_filter=None,
                                     cluster_id_whitelist=None,
                                     max_clusters_per_plot=None,
                                     orphanet_term_cap=10,
                                     filename_suffix='',
                                     hidef_source='pickle',
                                     hidef_pickle_file="../output/combined26/hidef/hidef_communities.pkl",
                                     nodes_file=None, supra_graph_pkl=None, seed_genes_file=None,
                                     persistence_threshold=0,
                                     label_wrap_width=40,
                                     legend_bottom_threshold=10):
    """
    Create a lollipop chart showing enrichment terms on Y-axis and clusters on X-axis,
    grouped by analysis type with bracket annotations.

    figsize : tuple or None
        If None (default), the figure is auto-sized from the number of
        clusters and terms that end up on the axes, so dense plots don't
        overlap. Pass an explicit tuple to override.

    persistence_threshold : float
        Explicit floor applied to the FULL persistence population (loaded
        with threshold=0 internally) BEFORE the percentile cut below. Use 0
        to compute the percentile against every cluster ever produced,
        regardless of any floor used elsewhere in your pipeline.

    tier_filter : str or None
        If given, restrict to exactly one persistence tier -- 'tier1'
        (90th+), 'tier2' (75-90th), 'tier3' (50-75th), or 'tier4' (<50th) --
        computed against the reference population. Takes precedence over
        `max_persistence_percentile`/`min_persistence_percentile`. Ignored
        if `cluster_id_whitelist` is given.
    max_persistence_percentile : float or None
        If given (and `tier_filter` is not), plot only clusters STRICTLY
        BELOW this percentile of the reference persistence distribution,
        instead of at/above `min_persistence_percentile`.
    cluster_id_whitelist : iterable of cluster ids, optional
        If given, skip the percentile-based inclusion filter entirely and
        plot ONLY these clusters (accepts either bare ids like 'Cluster1-0'
        or prefixed 'HiDeF_Cluster_Cluster1-0' -- both are normalized).
    max_clusters_per_plot : int or None
        If given, and the number of clusters that actually end up with
        matching enrichment results exceeds this, split them (sorted by
        persistence descending) into count-balanced chunks and render one
        PNG per chunk instead of one overcrowded plot. This is the
        recommended way to deal with a dense tier -- capping figsize alone
        can't fix crowding caused by many clusters sharing the same (or
        very similar) persistence value, since there's no percentile cut
        that can separate values that are literally tied. Filenames get
        `_part{N}of{total}` appended. None (default) never paginates.
    filename_suffix : str, optional
        Appended to the output filename before the extension, so different
        tiers of the same underlying data don't overwrite each other.
    label_wrap_width : 
        Max characters per line before a term label (e.g. a long Orphanet name) 
        wraps onto a new line. Lower this if labels still look cramped.
    """

    print(f"\n{'='*80}")
    print(f"Creating Persistence-based Lollipop Plot with Phenotype and Spatial Integration")
    print(f"{'='*80}")

    persistence_map = load_persistence_map(
        hidef_source=hidef_source, hidef_pickle_file=hidef_pickle_file,
        nodes_file=nodes_file, supra_graph_pkl=supra_graph_pkl,
        seed_genes_file=seed_genes_file, persistence_threshold=0,
    )
    if persistence_map is None:
        return False

    print(f"Loaded {len(persistence_map)} clusters with persistence data (full population)")
    persistence_map = apply_persistence_floor(persistence_map, persistence_threshold)
    print(f"Sample cluster IDs: {list(persistence_map.keys())[:5]}")

    # Tier thresholds computed ONCE against this reference population -- i.e.
    # before any percentile-based inclusion filter below narrows it further.
    # This is what keeps "Tier 3 (50-75th)" meaning the same thing regardless
    # of which subset of clusters ends up on the plot.
    reference_values = list(persistence_map.values())
    tier_thresholds, tier_labels_list = compute_persistence_tier_thresholds(reference_values, persistence_percentiles)

    if cluster_id_whitelist is not None:
        whitelist_clean = {str(c).replace('HiDeF_Cluster_', '') for c in cluster_id_whitelist}
        persistence_map = {k: v for k, v in persistence_map.items() if k in whitelist_clean}
        print(f"Restricted to explicit cluster whitelist: {len(persistence_map)}/{len(whitelist_clean)} matched")
    elif tier_filter is not None:
        persistence_map = filter_persistence_by_tier(persistence_map, tier_thresholds, tier_labels_list, tier_filter)
    elif max_persistence_percentile is not None:
        persistence_map, percentile_cutoff = filter_persistence_below_percentile(
            persistence_map, max_percentile=max_persistence_percentile
        )
    else:
        persistence_map, percentile_cutoff = filter_persistence_by_percentile(
            persistence_map, min_percentile=min_persistence_percentile
        )
    persistence_values = list(persistence_map.values())
    tier_mapping = assign_persistence_tiers(persistence_values, tier_thresholds, tier_labels_list)

    # Load enrichment results
    all_enrichments = {}

    for analysis_type in analysis_types:
        if analysis_type == 'spatial':
            results_file = os.path.join(output_dir, ppi_source,
                                       'spatial_analysis_hidef_from_comparison',
                                       'spatial_transcriptomics_hidef_comparison_summarized_results.csv')
        elif analysis_type == 'phenotype':
            results_file = os.path.join(output_dir, ppi_source,
                                       'phenotypes_analysis_hidef_from_comparison',
                                       'phenotype_summarized_results.csv')
        else:
            analysis_dir = os.path.join(output_dir, ppi_source, f'{analysis_type}_analysis_{cluster_type}')
            results_file = os.path.join(analysis_dir, f'{analysis_type}_summarized_results.csv')

        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df = df[df['adjusted_p_value'] < significance_threshold]
            if 'cluster_size' in df.columns:
                df = df[df['cluster_size'] >= min_cluster_size]

            term_col = get_enrichment_term_column(df, analysis_type)
            
            if term_col and analysis_type == 'orphanet_enrichr' and orphanet_term_cap:
                df = cap_terms_by_cluster_breadth(df, term_col, max_terms=orphanet_term_cap)

            if term_col:
                resolved_value_col = get_enrichment_value_column(df, analysis_type, 'effect_size')
                for _, row in df.iterrows():
                    cluster_id = str(row['cluster_id']).replace('HiDeF_Cluster_', '')
                    raw_term = str(row[term_col])
                    term = clean_enrichment_term(raw_term, analysis_type)

                    if cluster_id not in all_enrichments:
                        all_enrichments[cluster_id] = []

                    all_enrichments[cluster_id].append({
                        'analysis_type': analysis_type,
                        'term': term,
                        'p_value': row['adjusted_p_value'],
                        'fold_enrichment': row.get(resolved_value_col, np.nan) if resolved_value_col else np.nan
                    })

            print(f"Loaded enrichments from {analysis_type}: {len(df)} significant results")
            if len(df) > 0:
                print(f"  Sample cluster IDs from {analysis_type}: {df['cluster_id'].unique()[:5]}")
        else:
            print(f"Warning: Results file not found for {analysis_type}: {results_file}")

    def get_persistence_category(persistence, cluster_id, all_enrichments, tier_mapping):
        base_category = tier_mapping.get(persistence, 'Unknown')
        has_phenotype = any(e['analysis_type'] == 'phenotype' for e in all_enrichments.get(cluster_id, []))
        has_spatial = any(e['analysis_type'] == 'spatial' for e in all_enrichments.get(cluster_id, []))
        indicators = ("P" if has_phenotype else "") + ("S" if has_spatial else "")
        return f"{base_category}*({indicators})" if indicators else base_category

    plot_data = []

    print(f"\nDebug Info:")
    print(f"  Persistence map keys (sample): {list(persistence_map.keys())[:5]}")
    print(f"  Enrichments keys (sample): {list(all_enrichments.keys())[:5]}")
    print(f"  Total clusters in persistence_map: {len(persistence_map)}")
    print(f"  Total clusters in all_enrichments: {len(all_enrichments)}")

    for cluster_id, persistence in persistence_map.items():
        if cluster_id in all_enrichments:
            persistence_cat = get_persistence_category(persistence, cluster_id, all_enrichments, tier_mapping)
            for enrichment in all_enrichments[cluster_id]:
                plot_data.append({
                    'cluster_id': cluster_id,
                    'persistence': persistence,
                    'persistence_category': persistence_cat,
                    'analysis_type': enrichment['analysis_type'],
                    'term': enrichment['term'],
                    'p_value': enrichment['p_value'],
                    'fold_enrichment': enrichment.get('fold_enrichment', np.nan)
                })

    if not plot_data:
        print("No enriched clusters found!")
        print(f"\nTroubleshooting:")
        print(f"  - Persistence map has {len(persistence_map)} clusters")
        print(f"  - Enrichments loaded for {len(all_enrichments)} clusters")
        print(f"  - Matching clusters: {len(set(persistence_map.keys()) & set(all_enrichments.keys()))}")
        print(f"  - Persistence map sample keys: {list(persistence_map.keys())[:10]}")
        print(f"  - Enrichments sample keys: {list(all_enrichments.keys())[:10]}")
        return False

    plot_df = pd.DataFrame(plot_data)

    # Order lowest-tier-first for the x-axis, matching whatever tier scheme
    # was actually computed (works for any number of tiers, not just 4).
    persistence_order = list(reversed(tier_labels_list))

    colors = {
        'phenotype': '#FF6B6B',
        'spatial': '#1f77b4',
        'omim_disease': '#ff7f0e',
        'omim_ps': '#2ca02c',
        'orphanet_enrichr': '#d62728',
        'proteomics': '#8c564b',
        'neuropathic_pain': '#e377c2'
    }

    group_label_map = {
        'spatial': 'Spatial',
        'phenotype': 'Phenotype',
        'omim_disease': 'OMIM Disease',
        'omim_ps': 'OMIM PS',
        'orphanet_enrichr': 'Orphanet',
        'proteomics': 'Proteomics',
        'neuropathic_pain': 'Neuropathic Pain',
    }

    def _render_chunk(chunk_plot_df, chunk_suffix):
        """Render and save one lollipop PNG for a subset of clusters. Split
        out so a dense tier can be paginated into several calls instead of
        one overcrowded plot -- see max_clusters_per_plot above."""

        clusters_by_persistence = chunk_plot_df.groupby('cluster_id').agg({
            'persistence': 'first',
            'persistence_category': 'first'
        }).sort_values('persistence', ascending=False)

        # --- x positions: clusters, grouped by persistence tier -------------
        CLUSTER_STEP = 2.2    # horizontal space between adjacent clusters
        TIER_GAP = 3.0          # extra horizontal space between persistence tiers

        x_pos = 0
        x_labels = []
        x_ticks = []
        cluster_positions = {}
        category_x_positions = {}

        for persistence_cat_base in persistence_order:
            matching_cats = [cat for cat in clusters_by_persistence['persistence_category'].unique()
                             if cat.startswith(persistence_cat_base.split('(')[0])]
            clusters_in_cat = []
            for cat in matching_cats:
                clusters_in_cat.extend(
                    clusters_by_persistence[clusters_by_persistence['persistence_category'] == cat].index.tolist()
                )

            if clusters_in_cat:
                cat_x_start = x_pos
                for cluster_id in clusters_in_cat:
                    cluster_positions[cluster_id] = x_pos
                    x_labels.append(cluster_id)
                    x_ticks.append(x_pos)
                    x_pos += CLUSTER_STEP
                category_x_positions[persistence_cat_base] = (cat_x_start, x_pos - CLUSTER_STEP)
                x_pos += TIER_GAP

        # --- y positions: enrichment terms, grouped by analysis type --------
        # Terms are wrapped first, and TERM_STEP is sized to the longest
        # wrap found, so multi-line labels get enough vertical room and
        # don't visually collide with the tick above/below them.
        terms_by_type = {}
        max_wrap_lines = 1
        
        for analysis_type in analysis_types:
            terms_for_type = sorted(set(
                row['term'] for _, row in chunk_plot_df[chunk_plot_df['analysis_type'] == analysis_type].iterrows()
            ))
            terms_by_type[analysis_type] = terms_for_type
            for term in terms_for_type:
                max_wrap_lines = max(max_wrap_lines, wrap_label(term, width=label_wrap_width).count('\n') + 1)
        
        TERM_STEP = 0.75 + 0.35 * (max_wrap_lines - 1)
        GROUP_GAP = 1.1
        
        y_pos_map = {}
        y_pos = 0
        y_labels = []
        y_ticks = []
        group_y_positions = {}
        
        for analysis_type in analysis_types:
            terms_for_type = terms_by_type.get(analysis_type, [])
            if not terms_for_type:
                continue
            group_y_start = y_pos
            for term in terms_for_type:
                term_key = f"{analysis_type}|{term}"
                y_pos_map[term_key] = y_pos
                y_labels.append(wrap_label(term, width=label_wrap_width))
                y_ticks.append(y_pos)
                y_pos += TERM_STEP
            group_y_positions[analysis_type] = (group_y_start, y_pos - TERM_STEP)
            y_pos += GROUP_GAP

        # --- auto figure sizing ----------------------------------------------
        # Width is capped -- a very wide image gets shrunk to fit typical
        # viewers/previews, which is what made text unreadable before cluster
        # pagination was added. Height is NOT capped the same way: a tall
        # image just needs scrolling, it doesn't get uniformly shrunk the way
        # an overly-wide one does, so capping it only squeezed term rows --
        # this was the actual cause of illegible text even after cluster
        # pagination, since a chunk can have very few clusters and still have
        # dozens of enrichment terms (term count isn't paginated, only
        # clusters are). If a chunk has an extreme number of terms and the
        # resulting image is unwieldy, narrow `analysis_types` for that call
        # rather than relying on figsize alone.
        MAX_WIDTH = 40.0
        chunk_figsize = figsize
        if chunk_figsize is None:
            width = max(14.0, x_pos * 0.55 + 6.0)
            height = max(9.0, y_pos * 0.7 + 6.0)
            width_capped = width > MAX_WIDTH
            width = min(width, MAX_WIDTH)
            chunk_figsize = (width, height)
            print(f"Auto figsize: {chunk_figsize[0]:.1f} x {chunk_figsize[1]:.1f} "
                  f"({len(x_labels)} clusters, {len(y_labels)} terms)"
                  f"{' [width capped -- lower max_clusters_per_plot if labels overlap]' if width_capped else ''}")

        fig, ax = plt.subplots(figsize=chunk_figsize)

        for cluster_id, x in cluster_positions.items():
            cluster_data = chunk_plot_df[chunk_plot_df['cluster_id'] == cluster_id]
            for _, row in cluster_data.iterrows():
                term_key = f"{row['analysis_type']}|{row['term']}"
                if term_key not in y_pos_map:
                    continue
                y = y_pos_map[term_key]
                analysis_type = row['analysis_type']
                color = colors.get(analysis_type, 'gray')

                ax.plot([0, x], [y, y], color=color, linewidth=1.5, alpha=0.5, zorder=1)
                ax.scatter(x, y, s=150, color=color, zorder=3, edgecolors='black', linewidth=0.8)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=20)
        ax.set_xlabel('Clusters (grouped by persistence tier)', fontsize=20, fontweight='bold', labelpad=12)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=20)
        ax.set_ylabel('Enrichment Terms (grouped by analysis type)', fontsize=20, fontweight='bold', labelpad=12)

        for persistence_cat_base, (x_start, x_end) in category_x_positions.items():
            mid_x = (x_start + x_end) / 2
            ax.annotate('', xy=(x_start - 0.3, -3.0), xytext=(x_end + 0.3, -3.0),
                        arrowprops=dict(arrowstyle='-', color='black', lw=1.5))
            ax.text(mid_x, -3.9, persistence_cat_base, ha='center', va='top',
                    fontsize=20, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.6))

        bracket_x = x_pos + 0.5
        for analysis_type, (y_start, y_end) in group_y_positions.items():
            mid_y = (y_start + y_end) / 2
            group_color = colors.get(analysis_type, 'gray')
            group_label = group_label_map.get(analysis_type, analysis_type.replace('_', ' ').title())
            ax.annotate('', xy=(bracket_x, y_start - 0.3), xytext=(bracket_x, y_end + 0.3),
                        arrowprops=dict(arrowstyle='-', color=group_color, lw=2.5))
            ax.text(bracket_x + 0.3, mid_y, group_label, ha='left', va='center',
                    fontsize=20, fontweight='bold', color=group_color)

        ax.set_xlim(-1, x_pos + 1)
        ax.set_ylim(-5.0, y_pos + 1)

        title_suffix = f" -- {chunk_suffix.lstrip('_')}" if chunk_suffix != filename_suffix else ""
        ax.set_title(
            f'Cluster Enrichment by Persistence Tier (Percentiles {persistence_percentiles})\n'
            f'({ppi_source} - {cluster_type}{title_suffix})',
            fontsize=20, fontweight='bold', pad=22
        )

        from matplotlib.patches import Patch
        present_types = [at for at in analysis_types if terms_by_type.get(at)]
        legend_elements = [Patch(facecolor=colors[at], label=group_label_map.get(at, at))
                           for at in present_types if at in colors]
        # The bracket labels already live just right of the axes (see bracket_x
        # below) -- with a single analysis type, the one bracket color already IS
        # the legend, so drawing a legend too is pure clutter and it's what
        # collides. With few terms overall, the axes are short and there isn't
        # room beside the brackets for a legend, so move it below the plot instead
        # of fighting them for the same space.
        show_legend = len(present_types) > 2
        if show_legend:
            if len(y_labels) <= legend_bottom_threshold:
                ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                          ncol=min(len(legend_elements), 4), fontsize=20, framealpha=0.9,
                          title='Analysis Type', title_fontsize=20)
            else:
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                          fontsize=20, framealpha=0.9,
                          title='Analysis Type', title_fontsize=20)

        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plot_dir = os.path.join('plots', ppi_source, 'persistence_lollipop')
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        plot_file = os.path.join(plot_dir, f'persistence_lollipop_{cluster_type}_percentile_tiers{chunk_suffix}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close(fig)

        print(f"\nPlot saved to: {plot_file}")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for persistence_cat in persistence_order:
            matching_cats = [cat for cat in clusters_by_persistence['persistence_category'].unique()
                             if cat.startswith(persistence_cat.split('(')[0])]
            count = sum(len(clusters_by_persistence[clusters_by_persistence['persistence_category'] == cat])
                        for cat in matching_cats)
            if count > 0:
                print(f"Clusters in {persistence_cat}: {count}")

        return plot_file

    # --- decide whether to paginate, based on the ACTUAL clusters that
    # ended up with matching enrichment results (not the raw tier
    # population, which is usually much larger before that match) ----------
    ordered_cluster_ids = (
        plot_df.groupby('cluster_id')['persistence'].first()
        .sort_values(ascending=False)
        .index.tolist()
    )

    if max_clusters_per_plot and len(ordered_cluster_ids) > max_clusters_per_plot:
        n_parts = math.ceil(len(ordered_cluster_ids) / max_clusters_per_plot)
        chunks = [ordered_cluster_ids[i * max_clusters_per_plot:(i + 1) * max_clusters_per_plot]
                  for i in range(n_parts)]
        print(f"\n{len(ordered_cluster_ids)} clusters with matching results exceed "
              f"max_clusters_per_plot={max_clusters_per_plot} -- splitting into {n_parts} parts")
    else:
        chunks = [ordered_cluster_ids]

    saved_files = []
    for part_idx, chunk_ids in enumerate(chunks, start=1):
        chunk_suffix = filename_suffix if len(chunks) == 1 else f'{filename_suffix}_part{part_idx}of{len(chunks)}'
        chunk_plot_df = plot_df[plot_df['cluster_id'].isin(chunk_ids)]
        saved_files.append(_render_chunk(chunk_plot_df, chunk_suffix))

    return bool(saved_files)


# ============================================================================
# HEATMAP
# ============================================================================

def load_all_enrichment_results(output_dir, ppi_source, cluster_type, analysis_types):
    """Load all enrichment results from different analysis types."""

    all_results = []

    for analysis_type in analysis_types:
        if analysis_type == 'spatial':
            results_file = os.path.join(output_dir, ppi_source,
                                       'spatial_analysis_hidef_from_comparison',
                                       'spatial_transcriptomics_hidef_comparison_summarized_results.csv')
        elif analysis_type == 'phenotype':
            results_file = os.path.join(output_dir, ppi_source,
                                       'phenotypes_analysis_hidef_from_comparison',
                                       'phenotype_summarized_results.csv')
        else:
            analysis_dir = os.path.join(output_dir, ppi_source, f'{analysis_type}_analysis_{cluster_type}')
            results_file = os.path.join(analysis_dir, f'{analysis_type}_summarized_results.csv')

        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df['analysis_type'] = analysis_type
            all_results.append(df)
            print(f"Loaded {len(df)} results from {analysis_type}")
        else:
            print(f"Warning: Results file not found for {analysis_type}: {results_file}")

    if not all_results:
        print("No results files found!")
        return None

    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal combined results: {len(combined_df)} rows across {len(analysis_types)} analysis types")

    return combined_df


# Different analyzers report what is conceptually the same "effect size"
# under different literal column names -- there's no single name that
# exists across all of them the way 'percentage'/'p_value'/
# 'adjusted_p_value' do.
VALUE_COLUMN_MAP = {
    'proteomics': 'mean_fold_enrichment',
    'neuropathic_pain': 'mean_fold_enrichment',
    'phenotype': 'mean_fold_enrichment',
    'omim_ps': 'mean_combined_score',
    'orphanet_enrichr': 'mean_combined_score',
    'spatial': 'mean_odds_ratio',
}


def get_enrichment_value_column(df, analysis_type, requested_value_col):
    """
    Resolve which column actually holds the numeric value for this
    analysis_type. If requested_value_col exists literally in df, use it
    as-is (this is what lets you ask for a universal column like
    'percentage' or 'adjusted_p_value' and have it just work everywhere).
    Otherwise, fall back to this analysis type's own effect-size column
    via VALUE_COLUMN_MAP -- e.g. asking for 'effect_size' (or any name
    that isn't literally present) resolves to 'mean_fold_enrichment' for
    proteomics/neuropathic_pain/phenotype, 'mean_combined_score' for
    omim_ps/orphanet_enrichr, 'mean_odds_ratio' for spatial.

    Returns None if nothing usable is found, so the caller can skip that
    analysis type for this heatmap rather than silently filling NaN.
    """
    if requested_value_col in df.columns:
        return requested_value_col
    mapped = VALUE_COLUMN_MAP.get(analysis_type)
    if mapped and mapped in df.columns:
        return mapped
    return None


def get_enrichment_term_column_heatmap(df, analysis_type):
    """Get the appropriate enrichment term column based on analysis type for heatmap."""

    term_mapping = {
        'spatial': 'top_cell_type',
        'omim_disease': 'top_disease',
        'omim_ps': 'phenotypic_series',
        'orphanet_enrichr': 'cluster_top_disease',
        'phenotype': 'top_phenotype',
        'proteomics': 'higher_expression_gender',
        'neuropathic_pain': 'top_expression'
    }

    if analysis_type in ['proteomics', 'neuropathic_pain']:
        actual_col = term_mapping.get(analysis_type)
        available_cols = df.columns.tolist()
        if actual_col and actual_col in available_cols:
            return actual_col
        return None

    expected_col = term_mapping.get(analysis_type, 'term')
    available_cols = df.columns.tolist()

    if expected_col in available_cols:
        return expected_col

    fallback_cols = ['top_disease', 'top_phenotype', 'top_cell_type', 'term', 'disease', 'phenotype', 'cell_type']

    for col in fallback_cols:
        if col in available_cols:
            print(f"Using fallback column '{col}' for {analysis_type}")
            return col

    print(f"Warning: No suitable term column found for {analysis_type}")
    return None


def create_enrichment_heatmap(output_dir, ppi_source, cluster_type='hidef_from_comparison',
                            analysis_types=['phenotype', 'spatial'],
                            value_col='mean_log_fold_change',
                            significance_col='adjusted_p_value',
                            significance_threshold=0.05,
                            min_cluster_size=10,
                            figsize=(16, 12),
                            min_persistence_percentile=50,
                            max_persistence_percentile=None,
                            tier_filter=None,
                            cluster_id_whitelist=None,
                            max_clusters_per_plot=None,
                            orphanet_term_cap=10,
                            filename_suffix='',
                            hidef_source='pickle',
                            hidef_pickle_file="../output/combined26/hidef/hidef_communities.pkl",
                            nodes_file=None, supra_graph_pkl=None, seed_genes_file=None,
                            persistence_threshold=0,
                            label_wrap_width=40):
    """
    Create a comprehensive heatmap of enrichment results across analysis types,
    including phenotype and spatial enrichments.

    persistence_threshold : float
        Explicit floor applied to the FULL persistence population (loaded
        with threshold=0 internally) BEFORE the percentile cut below. Use 0
        to compute the percentile against every cluster ever produced,
        regardless of any floor used elsewhere in your pipeline.

    tier_filter : str or None
        If given, restrict to exactly one persistence tier -- 'tier1'
        (90th+), 'tier2' (75-90th), 'tier3' (50-75th), or 'tier4' (<50th) --
        computed against the reference population. Takes precedence over
        `max_persistence_percentile`/`min_persistence_percentile`. Ignored
        if `cluster_id_whitelist` is given.
    max_persistence_percentile : float or None
        If given (and `tier_filter` is not), restrict to clusters STRICTLY
        BELOW this percentile of the reference persistence distribution,
        instead of at/above `min_persistence_percentile`.
    cluster_id_whitelist : iterable of cluster ids, optional
        If given, skip the percentile-based inclusion filter entirely and
        plot ONLY these clusters (accepts either bare ids like 'Cluster1-0'
        or prefixed 'HiDeF_Cluster_Cluster1-0' -- both are normalized).
    max_clusters_per_plot : int or None
        If given, and the number of clusters that end up as heatmap columns
        exceeds this, split them (sorted alphabetically for a stable,
        reproducible split) into count-balanced chunks and render one PNG
        per chunk instead of one overcrowded heatmap. Filenames get
        `_part{N}of{total}` appended. None (default) never paginates.
    filename_suffix : str, optional
        Appended to the output filename before the extension, so different
        tiers of the same underlying data don't overwrite each other.
    """

    print(f"\n{'='*80}")
    print(f"Creating Enrichment Heatmap with Phenotype and Spatial Integration")
    print(f"{'='*80}")

    persistence_map = load_persistence_map(
        hidef_source=hidef_source, hidef_pickle_file=hidef_pickle_file,
        nodes_file=nodes_file, supra_graph_pkl=supra_graph_pkl,
        seed_genes_file=seed_genes_file, persistence_threshold=0,
    )
    if persistence_map is None:
        return False

    print(f"Loaded {len(persistence_map)} clusters with persistence data (full population)")
    persistence_map = apply_persistence_floor(persistence_map, persistence_threshold)
    print(f"Sample cluster IDs: {list(persistence_map.keys())[:5]}")

    persistence_percentiles = [90, 75, 50]

    # Tier thresholds computed ONCE against this reference population -- see
    # compute_persistence_tier_thresholds() for why this has to happen before
    # the inclusion filter below, not after.
    reference_values = list(persistence_map.values())
    tier_thresholds, tier_labels_list = compute_persistence_tier_thresholds(reference_values, persistence_percentiles)

    if cluster_id_whitelist is not None:
        whitelist_clean = {str(c).replace('HiDeF_Cluster_', '') for c in cluster_id_whitelist}
        persistence_map = {k: v for k, v in persistence_map.items() if k in whitelist_clean}
        print(f"Restricted to explicit cluster whitelist: {len(persistence_map)}/{len(whitelist_clean)} matched")
    elif tier_filter is not None:
        persistence_map = filter_persistence_by_tier(persistence_map, tier_thresholds, tier_labels_list, tier_filter)
    elif max_persistence_percentile is not None:
        persistence_map, percentile_cutoff = filter_persistence_below_percentile(
            persistence_map, max_percentile=max_persistence_percentile
        )
    else:
        persistence_map, percentile_cutoff = filter_persistence_by_percentile(
            persistence_map, min_percentile=min_persistence_percentile
        )
    persistence_values = list(persistence_map.values())
    tier_mapping = assign_persistence_tiers(persistence_values, tier_thresholds, tier_labels_list)
    valid_clusters = set(persistence_map.keys())

    combined_df = load_all_enrichment_results(output_dir, ppi_source, cluster_type, analysis_types)

    if combined_df is None:
        return False

    print(f"\nFiltering results...")
    print(f"Before filtering: {len(combined_df)} rows")

    if significance_col in combined_df.columns:
        significant_df = combined_df[combined_df[significance_col] < significance_threshold].copy()
        print(f"After significance filter ({significance_col} < {significance_threshold}): {len(significant_df)} rows")
    else:
        significant_df = combined_df.copy()
        print(f"Warning: {significance_col} column not found, skipping significance filter")

    if 'cluster_size' in significant_df.columns:
        significant_df = significant_df[significant_df['cluster_size'] >= min_cluster_size].copy()
        print(f"After cluster size filter (>= {min_cluster_size}): {len(significant_df)} rows")

    significant_df = significant_df[significant_df['cluster_id'].astype(str)
                                    .str.replace('HiDeF_Cluster_', '').isin(valid_clusters)]

    if len(significant_df) == 0:
        print("No significant results found after filtering!")
        return False

    group_label_map = {
        'spatial': 'Spatial',
        'phenotype': 'Phenotype',
        'omim_disease': 'OMIM Disease',
        'omim_ps': 'OMIM PS',
        'orphanet_enrichr': 'Orphanet',
        'proteomics': 'Proteomics',
        'neuropathic_pain': 'Neuropathic Pain',
    }

    heatmap_data = []

    for analysis_type in analysis_types:
        analysis_df = significant_df[significant_df['analysis_type'] == analysis_type].copy()

        if len(analysis_df) == 0:
            continue

        term_col = get_enrichment_term_column_heatmap(analysis_df, analysis_type)

        if term_col is None:
            continue
            
        if analysis_type == 'orphanet_enrichr' and orphanet_term_cap:
            analysis_df = cap_terms_by_cluster_breadth(analysis_df, term_col, max_terms=orphanet_term_cap)

        resolved_value_col = get_enrichment_value_column(analysis_df, analysis_type, value_col)
        if resolved_value_col is None:
            print(f"  ⚠ {analysis_type}: no usable column for value_col={value_col!r} "
                  f"(tried literal, then mapped fallback {VALUE_COLUMN_MAP.get(analysis_type)!r}) "
                  f"-- skipping this analysis type for this heatmap")
            continue

        group_label = group_label_map.get(analysis_type, analysis_type.replace('_', ' ').title())

        def build_term(raw, atype):
            t = str(raw).replace('_', ' ').replace('nan', '').strip()
            if atype == 'orphanet_enrichr':
                t = re.sub(r'\s*ORPHA[:\s]\d+.*$', '', t, flags=re.IGNORECASE).strip()
            lbl = group_label_map.get(atype, atype.replace('_', ' ').title())
            return f"{t} ({lbl})" if t else ''

        analysis_df['enrichment_term'] = analysis_df[term_col].apply(lambda x: build_term(x, analysis_type))

        if 'cluster_id' not in analysis_df.columns:
            print(f"Warning: cluster_id column not found for {analysis_type}")
            continue

        for _, row in analysis_df.iterrows():
            if not row['enrichment_term'] or row['enrichment_term'] in ['', 'nan', 'NaN', f' ({group_label})']:
                continue

            cluster_name = str(row['cluster_id']).replace('HiDeF_Cluster_', '').replace('Community_', '')

            heatmap_data.append({
                'cluster_name': cluster_name,
                'enrichment_term': row['enrichment_term'],
                'analysis_type': analysis_type,
                'value': row[resolved_value_col] if resolved_value_col in row else np.nan,
                'p_value': row.get(significance_col, np.nan),
                'cluster_size': row.get('cluster_size', np.nan),
                'seed_gene_count': row.get('seed_gene_count_in_cluster', np.nan)
            })

    if not heatmap_data:
        print("No data available for heatmap!")
        return False

    heatmap_df = pd.DataFrame(heatmap_data)
    print(f"\nHeatmap data shape: {heatmap_df.shape}")
    print(f"Unique clusters: {heatmap_df['cluster_name'].nunique()}")
    print(f"Unique terms: {heatmap_df['enrichment_term'].nunique()}")

    if value_col == 'mean_log_fold_change':
        max_finite = heatmap_df[heatmap_df['value'] != np.inf]['value'].max()
        if pd.isna(max_finite):
            max_finite = 10
        heatmap_df['value'] = heatmap_df['value'].replace([np.inf, -np.inf], [max_finite * 2, -max_finite * 2])
        heatmap_df['value'] = heatmap_df['value'] * 10
    elif 'p_value' in value_col:
        heatmap_df['value'] = -np.log10(heatmap_df['value'].clip(lower=1e-10))
    else:
        if heatmap_df['value'].min() >= 0:
            heatmap_df['value'] = np.sqrt(heatmap_df['value'])
        else:
            heatmap_df['value'] = np.sign(heatmap_df['value']) * np.power(np.abs(heatmap_df['value']), 1/3)

    heatmap_df_agg = heatmap_df.groupby(['enrichment_term', 'cluster_name'])['value'].mean().reset_index()
    pivot_df = heatmap_df_agg.pivot(index='enrichment_term', columns='cluster_name', values='value')
    print(f"Pivot table shape: {pivot_df.shape}")

    pivot_df = pivot_df.dropna(axis=0, how='all')
    pivot_df = pivot_df.dropna(axis=1, how='all')

    if pivot_df.empty:
        print("No data remaining after removing empty rows and columns!")
        return False

    print(f"Pivot table shape after removing empty rows/columns: {pivot_df.shape}")

    cemm_colors = {
        'blue': '#00AECC',
        'medium_blue': '#1B88A7',
        'dark_blue': '#0F5569',
        'light_blue': '#BEE3EF',
        'pink': '#C73188',
        'orange': '#E95E30',
        'yellow': '#F1DD00',
        'green': '#96C03A',
        'red': '#FF6B6B'
    }

    if value_col in ['mean_log_fold_change', 'log_fold_change']:
        colors_list = [cemm_colors['dark_blue'], cemm_colors['medium_blue'], cemm_colors['light_blue'],
                 'white', cemm_colors['yellow'], cemm_colors['orange'], cemm_colors['pink']]
        cmap = sns.blend_palette(colors_list, n_colors=256, as_cmap=True)
        center = 0
        cbar_label = f'{value_col.replace("_", " ").title()} (×10)'
    elif 'p_value' in value_col:
        colors_list = [cemm_colors['light_blue'], cemm_colors['blue'], cemm_colors['dark_blue']]
        cmap = sns.blend_palette(colors_list, n_colors=256, as_cmap=True)
        center = None
        cbar_label = f'-log10({value_col.replace("_", " ").title()})'
    else:
        colors_list = [cemm_colors['light_blue'], cemm_colors['blue'], cemm_colors['medium_blue'],
                 cemm_colors['dark_blue'], cemm_colors['pink']]
        cmap = sns.blend_palette(colors_list, n_colors=256, as_cmap=True)
        center = None
        cbar_label = f'Transformed {value_col.replace("_", " ").title()}'

    if value_col in ['mean_log_fold_change', 'log_fold_change']:
        vmin = 3
        vmax = 3.5
        center = 4
    elif value_col in ['mean_odds_ratio']:
        vmin = 0.10
        vmax = 0.12
        center = 0.13
    else:
        vmin = None
        vmax = None

    def _render_heatmap_chunk(chunk_pivot_df, chunk_suffix):
        """Render and save one heatmap PNG for a subset of cluster columns."""
        # Wrap long row labels (e.g. Orphanet disease names) and grow the
        # figure height to match, so the saved PNG's dimensions aren't
        # dictated by a handful of very long single-line labels.
        row_labels = [wrap_label(str(t), width=label_wrap_width) for t in chunk_pivot_df.index]
        max_wrap_lines = max((lbl.count('\n') + 1 for lbl in row_labels), default=1)
        base_width, base_height = figsize
        n_rows = len(chunk_pivot_df.index)
        height = max(base_height, n_rows * 0.42 * max_wrap_lines + 3.0)
        plt.figure(figsize=(base_width, height))
        ax = sns.heatmap(chunk_pivot_df,
                         cmap=cmap,
                         center=center,
                         vmin=vmin,
                         vmax=vmax,
                         cbar_kws={'label': cbar_label},
                         linewidths=0.5,
                         linecolor='white',
                         square=False,
                         xticklabels=True,
                         yticklabels=True)
        ax.set_yticklabels(row_labels, rotation=0, fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.figure.axes[-1].tick_params(labelsize=20)

        title_suffix = f" -- {chunk_suffix.lstrip('_')}" if chunk_suffix != filename_suffix else ""
        plt.title(f'Cluster Enrichment Heatmap\n({ppi_source} - {cluster_type}{title_suffix})',
                  fontsize=20, fontweight='bold', pad=20)

        plt.xlabel('Clusters', fontsize=20, fontweight='bold')
        plt.ylabel('Enrichment Terms', fontsize=20, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        plot_dir = os.path.join('plots', ppi_source, 'enrichment_heatmaps')
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        plot_file = os.path.join(plot_dir, f'enrichment_heatmap_{cluster_type}_{value_col}{chunk_suffix}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close()

        print(f"\n{'='*60}")
        print("HEATMAP COMPLETE")
        print(f"{'='*60}")
        print(f"Saved heatmap to: {plot_file}")
        print(f"Clusters visualized: {chunk_pivot_df.shape[1]}")
        print(f"Enrichment terms: {chunk_pivot_df.shape[0]}")

        return plot_file

    all_columns = sorted(pivot_df.columns.tolist())
    if max_clusters_per_plot and len(all_columns) > max_clusters_per_plot:
        n_parts = math.ceil(len(all_columns) / max_clusters_per_plot)
        col_chunks = [all_columns[i * max_clusters_per_plot:(i + 1) * max_clusters_per_plot]
                      for i in range(n_parts)]
        print(f"\n{len(all_columns)} clusters exceed max_clusters_per_plot={max_clusters_per_plot} "
              f"-- splitting into {n_parts} parts")
    else:
        col_chunks = [all_columns]

    saved_files = []
    for part_idx, cols in enumerate(col_chunks, start=1):
        chunk_suffix = filename_suffix if len(col_chunks) == 1 else f'{filename_suffix}_part{part_idx}of{len(col_chunks)}'
        chunk_pivot_df = pivot_df[cols].dropna(axis=0, how='all')
        saved_files.append(_render_heatmap_chunk(chunk_pivot_df, chunk_suffix))

    print(f"\nSummary by analysis type:")
    for analysis_type in analysis_types:
        count = heatmap_df[heatmap_df['analysis_type'] == analysis_type].shape[0]
        if count > 0:
            print(f"  {analysis_type}: {count} clusters")

    return bool(saved_files)