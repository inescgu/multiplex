#!/usr/bin/env python3
"""
plot_tier_source_variation.py

Illustrates how the mix of evidence sources (proteomics, neuropathic_pain,
orphanet, omim, phenotypes, spatial) behind each cluster's significant
enrichment(s) varies with the cluster's persistence tier.

This is a companion to run_multiplex_visualizations.py, not a replacement --
it reuses the SAME persistence loading and tier-threshold machinery from
enrichment_plots_multiplex.py (load_persistence_map,
compute_persistence_tier_thresholds, assign_persistence_tiers), so tier
boundaries here are identical to the tier1-4 boundaries in your lollipop/
heatmap plots. It reads combined_gene_results_multiplex.tsv (the output of
results_combiner_multiplex.py) rather than any individual analyzer's
_summarized_results.csv.

Operates at the CLUSTER level, not the gene level -- taking a gene's max
persistence across its cluster memberships is degenerate, since almost
every gene sits inside at least one giant, near-universal, high-persistence
root cluster in HiDeF's hierarchical output. Clusters, unlike genes, are
sharply differentiated by persistence tier.

OUTPUTS (PNG, dpi=300) under plots/{PPI_SOURCE}/tier_source_variation/:
  - tier_specificity_stacked_bar.png : % of clusters per tier that are
    single-source-specific (disease/sex/cell-type/phenotype/pain-state) vs
    broad/multi-source vs mixed.
  - tier_source_prevalence_heatmap.png : tier x source grid, % of clusters
    in each tier containing >=1 significant hit from that source.

Also writes cluster_level_categorization.csv under output/{PPI_SOURCE}/ so
you can slice the underlying table yourself.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from enrichment_plots_multiplex import (
    load_persistence_map, compute_persistence_tier_thresholds, assign_persistence_tiers,
)

# ============================================================================
# CONFIG -- keep in sync with run_multiplex_enrichment_pipeline.py /
# run_multiplex_visualizations.py
# ============================================================================
OUTPUT_DIR = './output'
PPI_SOURCE = 'combined26_multiplex'
CLUSTER_TYPE = 'hidef_from_comparison'

NODES_FILE = '../11_hidef_results_faithful_cpm/heterogeneous_hidef_faithful_cpm.nodes'
SUPRA_GRAPH_PKL = '../11_hidef_hetero/supra_graph.pkl'
SEED_GENES_FILE = '../00_inputs/seed_genes.csv'

# Wherever run_multiplex_enrichment_pipeline.py's ResultsCombiner call wrote
# this -- by default that script calls
# combiner.save_combined_results('combined_gene_results_multiplex.tsv', ...)
# with a bare filename, i.e. relative to wherever you run the driver script
# from. Point this at that file directly.
COMBINED_TSV = 'combined_gene_results_multiplex.tsv'

# Percentiles used to cut the reference persistence population into tiers --
# same convention/order as run_multiplex_visualizations.py's TIERS list.
TIER_PERCENTILES = [90, 75, 50]

SOURCES = ['proteomics', 'neuropathic_pain', 'orphanet', 'omim', 'phenotypes', 'spatial']
SOURCE_LABELS = {
    'proteomics': 'Proteomics', 'neuropathic_pain': 'Neuropathic Pain',
    'orphanet': 'Orphanet', 'omim': 'OMIM',
    'phenotypes': 'Phenotypes', 'spatial': 'Spatial',
}

CEMM_COLORS = {
    'blue': '#00AECC', 'medium_blue': '#1B88A7', 'dark_blue': '#0F5569',
    'light_blue': '#BEE3EF', 'pink': '#C73188', 'orange': '#E95E30',
    'yellow': '#F1DD00', 'green': '#96C03A', 'red': '#FF6B6B',
}
SPECIFICITY_COLORS = {
    'broad / multi-source': CEMM_COLORS['dark_blue'],
    'mixed (2-3 sources)': CEMM_COLORS['medium_blue'],
    'disease-specific': CEMM_COLORS['pink'],
    'sex-specific': CEMM_COLORS['orange'],
    'cell-type-specific': CEMM_COLORS['green'],
    'phenotype-specific': CEMM_COLORS['yellow'],
    'pain-state-specific': CEMM_COLORS['red'],
}


# ============================================================================
# STEP 1 -- build the cluster-level table (cluster_id, persistence, tier,
# which sources have a significant hit in it, specificity label)
# ============================================================================

def specificity_of(sources_present):
    s = set(sources_present)
    if s <= {'orphanet', 'omim'} and s:
        return 'disease-specific'
    if s == {'proteomics'}:
        return 'sex-specific'
    if s == {'spatial'}:
        return 'cell-type-specific'
    if s == {'phenotypes'}:
        return 'phenotype-specific'
    if s == {'neuropathic_pain'}:
        return 'pain-state-specific'
    if len(s) >= 4:
        return 'broad / multi-source'
    return 'mixed (2-3 sources)'


def build_cluster_table(combined_tsv, persistence_map, tier_thresholds, tier_labels):
    if not os.path.exists(combined_tsv):
        print(f"Error: combined results file not found: {combined_tsv}")
        print("  -> check COMBINED_TSV points at the output of "
              "results_combiner_multiplex.py's save_combined_results()")
        return None

    df = pd.read_csv(combined_tsv, sep='\t')

    # cluster_id (stripped of HiDeF_Cluster_ prefix, matching persistence_map
    # keys) -> source -> {genes with a significant hit in that cluster for
    # that source}
    cluster_genes = defaultdict(lambda: defaultdict(set))
    for _, row in df.iterrows():
        gene = row['gene']
        for src in SOURCES:
            val = row.get(f'{src}_clusters', '')
            if isinstance(val, str) and val.strip():
                for cid in val.split(','):
                    cid = cid.strip().replace('HiDeF_Cluster_', '')
                    if cid:
                        cluster_genes[cid][src].add(gene)

    tier_value_to_label = assign_persistence_tiers(
        sorted(set(persistence_map.values())), tier_thresholds, tier_labels
    )

    rows = []
    for cid, src_map in cluster_genes.items():
        persistence = persistence_map.get(cid)
        if persistence is None:
            continue
        sources_present = sorted(src_map.keys())
        rows.append({
            'cluster_id': cid,
            'persistence': persistence,
            'tier': tier_value_to_label.get(persistence, 'Unknown'),
            'n_sources': len(sources_present),
            'sources': ','.join(sources_present),
            'total_genes_with_any_hit': len(set().union(*src_map.values())),
            'specificity': specificity_of(sources_present),
            **{f'has_{src}': (src in src_map) for src in SOURCES},
        })

    cluster_df = pd.DataFrame(rows)
    print(f"Built cluster-level table: {len(cluster_df)} clusters with >=1 significant hit "
          f"(matched against {len(persistence_map)} clusters with persistence data)")
    return cluster_df


# ============================================================================
# STEP 2a -- stacked bar: specificity composition per tier
# ============================================================================

def plot_specificity_stacked_bar(cluster_df, output_dir, tier_order):
    comp = pd.crosstab(cluster_df['tier'], cluster_df['specificity'], normalize='index') * 100
    comp = comp.reindex([t for t in tier_order if t in comp.index])
    col_order = [c for c in SPECIFICITY_COLORS if c in comp.columns]
    comp = comp[col_order]

    n_clusters_per_tier = cluster_df['tier'].value_counts().reindex(comp.index)

    fig, ax = plt.subplots(figsize=(10, 7))
    bottom = np.zeros(len(comp))
    x = np.arange(len(comp))
    for col in comp.columns:
        vals = comp[col].values
        ax.bar(x, vals, bottom=bottom, label=col, color=SPECIFICITY_COLORS[col],
               edgecolor='white', linewidth=0.8, width=0.65)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 6:
                ax.text(xi, b + v / 2, f'{v:.0f}%', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n(n={n_clusters_per_tier[t]} clusters)' for t in comp.index], fontsize=12)
    ax.set_ylabel('% of clusters in tier', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title(f'Cluster specificity composition by persistence tier\n({PPI_SOURCE} - {CLUSTER_TYPE})',
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11,
              title='Specificity', title_fontsize=11, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'tier_specificity_stacked_bar.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return out_path


# ============================================================================
# STEP 2b -- heatmap: per-source prevalence by tier
# ============================================================================

def plot_source_prevalence_heatmap(cluster_df, output_dir, tier_order):
    prevalence = cluster_df.groupby('tier')[[f'has_{s}' for s in SOURCES]].mean() * 100
    prevalence = prevalence.reindex([t for t in tier_order if t in prevalence.index])
    prevalence.columns = [SOURCE_LABELS[s] for s in SOURCES]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors_list = [CEMM_COLORS['light_blue'], CEMM_COLORS['blue'], CEMM_COLORS['dark_blue']]
    cmap = sns.blend_palette(colors_list, n_colors=256, as_cmap=True)

    sns.heatmap(prevalence, annot=True, fmt='.0f', cmap=cmap, vmin=0, vmax=100,
                linewidths=0.5, linecolor='white', cbar_kws={'label': '% of clusters in tier'},
                ax=ax, annot_kws={'fontsize': 13, 'fontweight': 'bold'})

    ax.set_xlabel('Evidence source', fontsize=13, fontweight='bold')
    ax.set_ylabel('Persistence tier', fontsize=13, fontweight='bold')
    ax.set_title(f'Source prevalence by persistence tier\n({PPI_SOURCE} - {CLUSTER_TYPE})',
                 fontsize=15, fontweight='bold', pad=15)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'tier_source_prevalence_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    plot_dir = os.path.join('plots', PPI_SOURCE, 'tier_source_variation')
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    table_out_dir = os.path.join(OUTPUT_DIR, PPI_SOURCE)
    Path(table_out_dir).mkdir(parents=True, exist_ok=True)
    table_out_path = os.path.join(table_out_dir, 'cluster_level_categorization.csv')

    # Load the FULL persistence population (threshold=0), same pattern as
    # both plot functions in enrichment_plots_multiplex.py, so tier
    # boundaries here match the ones your lollipop/heatmap plots use.
    persistence_map = load_persistence_map(
        hidef_source='multiplex', nodes_file=NODES_FILE, supra_graph_pkl=SUPRA_GRAPH_PKL,
        seed_genes_file=SEED_GENES_FILE, persistence_threshold=0,
    )
    if persistence_map is None:
        sys.exit(1)
    print(f"Loaded {len(persistence_map)} clusters with persistence data")

    reference_values = list(persistence_map.values())
    tier_thresholds, tier_labels = compute_persistence_tier_thresholds(reference_values, TIER_PERCENTILES)
    tier_order = tier_labels  # highest tier first, however many tiers TIER_PERCENTILES produces

    cluster_df = build_cluster_table(COMBINED_TSV, persistence_map, tier_thresholds, tier_labels)
    if cluster_df is None or cluster_df.empty:
        sys.exit(1)

    cluster_df.to_csv(table_out_path, index=False)
    print(f"Saved underlying table -> {table_out_path}")

    print("\n=== Cluster counts per tier ===")
    print(cluster_df['tier'].value_counts().reindex(tier_order))

    plot_specificity_stacked_bar(cluster_df, plot_dir, tier_order)
    plot_source_prevalence_heatmap(cluster_df, plot_dir, tier_order)

    print("\nDone.")