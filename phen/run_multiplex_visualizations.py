#!/usr/bin/env python3
"""
run_multiplex_visualizations.py

Regenerates the enrichment heatmaps and persistence lollipop plot from
results your analyzers already wrote to disk -- no analyzer reruns, no
ResultsCombiner. Use this whenever you're only iterating on plot
parameters (which analysis types to include, persistence tier, figure
size, etc.) so you're not burning a multi-hour SLURM allocation just to
tweak a figure.

Produces FIVE lollipop plots (and, for each, a matching set of heatmaps):

    tier1  : 90th percentile and above
    tier2  : 75th-90th percentile
    tier3  : 50th-75th percentile
    tier4  : below the 50th percentile
    top    : only the cluster(s) tied at the single highest persistence
             value (see TOP_N_VALUES below)

tier1-tier4 are mutually exclusive and together cover every cluster with
enrichment results.

CROWDING: an earlier version of this script tried to fix a crowded tier by
adding a 5th percentile boundary (splitting "below 50th" into two). That
didn't work -- with real persistence data, hundreds of clusters can share
the exact same (or a very similar) persistence value, so there's no
percentile cut that can separate them; the split just moved the crowding
into a different tier instead of fixing it. The actual fix is
MAX_CLUSTERS_PER_PLOT below: any tier (or the top section) whose matched
cluster count exceeds this gets automatically split into count-balanced
`_part{N}of{total}` files, regardless of why it's big. This means the
number of files per tier isn't fixed -- a quiet tier might produce one
file, a dense one several.

All four tiers use `tier_filter` in enrichment_plots_multiplex.py, which
computes the 90th/75th/50th percentile boundaries ONCE against the full
persistence distribution (persistence_threshold=0) and reuses those same
fixed boundaries for every tier -- so a cluster's tier assignment doesn't
depend on which tier you happen to be plotting, and can't drift the way
plain percentile filtering did before.

`top` is separate from the four tiers (a cluster in `top` will also show
up in whichever of tier1-tier4 it belongs to) since "the single highest
persistence value" isn't a percentile cut, and follows PERSISTENCE_THRESHOLD
as a floor to mirror exactly what the analyzers tested.
"""
import os
import sys
from enrichment_plots_multiplex import (
    load_persistence_map, load_all_enrichment_results,
    create_persistence_lollipop_plot, create_enrichment_heatmap,
)

# ============================================================================
# CONFIG -- keep in sync with run_multiplex_enrichment_pipeline.py
# ============================================================================
import pipeline_config as cfg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from provenance import write_manifest

PPI_SOURCE = cfg.PPI_SOURCE
CLUSTER_TYPE = cfg.CLUSTER_TYPE
OUTPUT_DIR = cfg.OUTPUT_DIR
PERSISTENCE_THRESHOLD = cfg.PERSISTENCE_THRESHOLD
NODES_FILE = cfg.NODES_FILE
SUPRA_GRAPH_PKL = cfg.SUPRA_GRAPH_PKL
SEED_GENES_FILE = cfg.SEED_GENES_FILE

ORPHANET_TERM_CAP = 10
LEGEND_BOTTOM_THRESHOLD = 10



# 'spatial' dropped -- spatial transcriptomics results are no longer part of
# this analysis.
ANALYSIS_TYPES = ['neuropathic_pain', 'phenotype', 'proteomics', 'omim_ps', 'orphanet_enrichr', 'spatial']

# Which effect-size columns to render a heatmap for, per tier. With
# pagination (below), the number of files per tier varies with how crowded
# it turns out to be -- this no longer multiplies out to a fixed count.
HEATMAP_VALUE_COLS = ['effect_size', 'percentage', 'adjusted_p_value']

# Any tier (or the top-persistence section) whose matched cluster count
# exceeds this gets split into count-balanced parts instead of one
# overcrowded plot. ~25 keeps a single-tier lollipop plot comfortably under
# the 40x30in auto-figsize cap in enrichment_plots_multiplex.py; lower it
# further if parts still look crowded, raise it if you'd rather have fewer,
# denser files.
MAX_CLUSTERS_PER_PLOT = 25

# Top-persistence section: pull the same results/plots as the four tiers,
# but restricted to your highest-persistence clusters instead of a
# percentile cut. Set exactly ONE of the two below, leave the other as None.
RUN_TOP_PERSISTENCE_SECTION = True
TOP_N_VALUES = 1        # e.g. 1 = only clusters tied at the single max persistence value
TOP_N_CLUSTERS = None   # e.g. 15 = exactly the top 15 clusters by persistence


# ============================================================================
# TIER 1-4: four mutually-exclusive plots covering every cluster with
# enrichment results
# ============================================================================

TIERS = [
    ('tier1', '90th percentile and above'),
    ('tier2', '75th-90th percentile'),
    ('tier3', '50th-75th percentile'),
    ('tier4', 'below the 50th percentile'),
]

for tier_key, tier_desc in TIERS:
    print(f"\n{'#' * 80}")
    print(f"# {tier_key.upper()}: {tier_desc}")
    print(f"{'#' * 80}")

    for value_col in HEATMAP_VALUE_COLS:
        create_enrichment_heatmap(
            output_dir=OUTPUT_DIR, ppi_source=PPI_SOURCE, cluster_type=CLUSTER_TYPE, orphanet_term_cap=ORPHANET_TERM_CAP,
            analysis_types=ANALYSIS_TYPES,
            value_col=value_col, significance_threshold=0.05, min_cluster_size=10,
            figsize=(20, 14),
            tier_filter=tier_key,
            max_clusters_per_plot=MAX_CLUSTERS_PER_PLOT,
            hidef_source='multiplex', nodes_file=NODES_FILE, supra_graph_pkl=SUPRA_GRAPH_PKL,
            seed_genes_file=SEED_GENES_FILE, persistence_threshold=0,
            filename_suffix=f'_{tier_key}',
        )

    create_persistence_lollipop_plot(
        output_dir=OUTPUT_DIR, ppi_source=PPI_SOURCE, cluster_type=CLUSTER_TYPE, orphanet_term_cap=ORPHANET_TERM_CAP,
        legend_bottom_threshold=LEGEND_BOTTOM_THRESHOLD,
        analysis_types=ANALYSIS_TYPES,
        tier_filter=tier_key,
        max_clusters_per_plot=MAX_CLUSTERS_PER_PLOT,
        figsize=None,  # auto-sized to whatever ends up in this tier/chunk
        hidef_source='multiplex', nodes_file=NODES_FILE, supra_graph_pkl=SUPRA_GRAPH_PKL,
        seed_genes_file=SEED_GENES_FILE, persistence_threshold=0,
        filename_suffix=f'_{tier_key}',
    )

print("\nDone with tier1-tier4 plots.")


# ============================================================================
# TOP-PERSISTENCE SECTION: same results/plots, restricted to the clusters
# with the highest persistence values instead of a percentile cut. Uses
# PERSISTENCE_THRESHOLD as the floor here (not 0) since this section is
# meant to mirror exactly what the analyzers tested, and top-cluster
# selection is unaffected by the floor anyway (a floor only removes clusters
# from the bottom, never changes which one is the max).
# ============================================================================

if RUN_TOP_PERSISTENCE_SECTION:
    assert (TOP_N_VALUES is None) != (TOP_N_CLUSTERS is None), \
        "Set exactly one of TOP_N_VALUES or TOP_N_CLUSTERS above, leave the other as None"

    print(f"\n{'=' * 80}")
    print("TOP-PERSISTENCE SECTION")
    print(f"{'=' * 80}")

    persistence_map = load_persistence_map(
        hidef_source='multiplex', nodes_file=NODES_FILE, supra_graph_pkl=SUPRA_GRAPH_PKL,
        seed_genes_file=SEED_GENES_FILE, persistence_threshold=PERSISTENCE_THRESHOLD,
    )
    if persistence_map is None:
        print("Could not load persistence map -- skipping top-persistence section")
    else:
        # --- pick the top clusters ---
        if TOP_N_VALUES is not None:
            distinct_values_desc = sorted(set(persistence_map.values()), reverse=True)
            cutoff_values = set(distinct_values_desc[:TOP_N_VALUES])
            top_cluster_ids = [cid for cid, v in persistence_map.items() if v in cutoff_values]
        else:
            top_cluster_ids = sorted(
                persistence_map, key=lambda cid: (-persistence_map[cid], cid)
            )[:TOP_N_CLUSTERS]
        top_cluster_ids = sorted(top_cluster_ids, key=lambda cid: -persistence_map[cid])

        print(f"{len(top_cluster_ids)} top-persistence clusters selected:")
        for cid in top_cluster_ids:
            print(f"  {cid}  (persistence = {persistence_map[cid]:.0f})")

        if not top_cluster_ids:
            print("No clusters selected -- check TOP_N_VALUES/TOP_N_CLUSTERS and PERSISTENCE_THRESHOLD")
        else:
            # --- export all enrichment results for just these clusters ---
            combined_df = load_all_enrichment_results(OUTPUT_DIR, PPI_SOURCE, CLUSTER_TYPE, ANALYSIS_TYPES)
            if combined_df is not None:
                top_cluster_names = {f"HiDeF_Cluster_{cid}" for cid in top_cluster_ids}
                top_df = combined_df[combined_df['cluster_id'].astype(str).isin(top_cluster_names)].copy()
                top_df['persistence'] = (
                    top_df['cluster_id'].astype(str).str.replace('HiDeF_Cluster_', '').map(persistence_map)
                )
                top_df = top_df.sort_values(
                    ['persistence', 'cluster_id', 'analysis_type'], ascending=[False, True, True]
                )

                out_dir = os.path.join(OUTPUT_DIR, PPI_SOURCE)
                os.makedirs(out_dir, exist_ok=True)

                out_csv = os.path.join(out_dir, 'top_persistence_clusters_all_results.csv')
                top_df.to_csv(out_csv, index=False)
                print(f"\nAll enrichment results (no significance filter) for top clusters -> {out_csv}")
                print(f"  {len(top_df)} rows across {top_df['cluster_id'].nunique()} clusters, "
                      f"{top_df['analysis_type'].nunique()} analysis types")

                if 'adjusted_p_value' in top_df.columns:
                    sig_df = top_df[top_df['adjusted_p_value'] < 0.05]
                else:
                    sig_df = top_df
                out_sig_csv = os.path.join(out_dir, 'top_persistence_clusters_significant_only.csv')
                sig_df.to_csv(out_sig_csv, index=False)
                print(f"Significant-only (adjusted p < 0.05) -> {out_sig_csv}")
                print(f"  {len(sig_df)} rows")
            else:
                print("\nNo enrichment result files found -- skipping the table export")

            # --- plots restricted to exactly these clusters ---
            for value_col in HEATMAP_VALUE_COLS:
                create_enrichment_heatmap(
                    output_dir=OUTPUT_DIR, ppi_source=PPI_SOURCE, cluster_type=CLUSTER_TYPE, orphanet_term_cap=ORPHANET_TERM_CAP,
                    analysis_types=ANALYSIS_TYPES,
                    value_col=value_col, significance_threshold=0.05, min_cluster_size=1,
                    figsize=(12, 10),
                    max_clusters_per_plot=MAX_CLUSTERS_PER_PLOT,
                    hidef_source='multiplex', nodes_file=NODES_FILE, supra_graph_pkl=SUPRA_GRAPH_PKL,
                    seed_genes_file=SEED_GENES_FILE, persistence_threshold=PERSISTENCE_THRESHOLD,
                    cluster_id_whitelist=top_cluster_ids,
                    filename_suffix='_top_persistence',
                )

            create_persistence_lollipop_plot(
                output_dir=OUTPUT_DIR, ppi_source=PPI_SOURCE, cluster_type=CLUSTER_TYPE, orphanet_term_cap=ORPHANET_TERM_CAP,
                analysis_types=ANALYSIS_TYPES,
                legend_bottom_threshold=LEGEND_BOTTOM_THRESHOLD,
                significance_threshold=0.05, min_cluster_size=1,
                figsize=(14, 10),
                max_clusters_per_plot=MAX_CLUSTERS_PER_PLOT,
                hidef_source='multiplex', nodes_file=NODES_FILE, supra_graph_pkl=SUPRA_GRAPH_PKL,
                seed_genes_file=SEED_GENES_FILE, persistence_threshold=PERSISTENCE_THRESHOLD,
                cluster_id_whitelist=top_cluster_ids,
                filename_suffix='_top_persistence',
            )

print("\nDone.")