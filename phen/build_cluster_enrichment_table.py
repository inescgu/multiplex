#!/usr/bin/env python3
"""
build_cluster_enrichment_table.py -- the cluster <-> enrichment map.

Reads every analyzer's *_summarized_results.csv (one row per cluster, BH-adjusted
where the analyzer applied it) with the SAME loaders enrichment_plots_multiplex.py
uses, attaches persistence + tier from the .nodes file, and writes:

  cluster_enrichment_long.csv     one row per (cluster, analysis): term, effect, q, size, persistence, tier
  cluster_enrichment_summary.csv  one row per cluster: persistence, tier, how many sources are BH q < 0.05, which

Tiers are fixed against the FULL persistence population (same rule as
run_multiplex_visualizations.py), so tier numbers here match the lollipop/heatmap files.
Feed the long table to:  fig_step12_enrichment.py grid --table cluster_enrichment_long.csv

    python build_cluster_enrichment_table.py                      # paths from pipeline_config.py
    python build_cluster_enrichment_table.py --floor 8 --out tables
"""
import argparse, os
import numpy as np
import pandas as pd

from enrichment_plots_multiplex import (
    load_persistence_map, apply_persistence_floor, compute_persistence_tier_thresholds,
    assign_persistence_tiers, load_all_enrichment_results, get_enrichment_value_column,
    get_enrichment_term_column_heatmap,
)

ANALYSIS_TYPES = ['neuropathic_pain', 'phenotype', 'proteomics', 'omim_ps', 'orphanet_enrichr', 'spatial']
# term columns, taken from what each analyzer writes into result_dict
TERM_COL = {'orphanet_enrichr': 'disease', 'omim_ps': 'phenotypic_series', 'phenotype': 'top_phenotype',
            'spatial': 'top_cell_type', 'proteomics': 'higher_expression_gender', 'neuropathic_pain': 'top_expression'}


def main():
    import pipeline_config as cfg
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=cfg.OUTPUT_DIR)
    ap.add_argument('--ppi-source', default=cfg.PPI_SOURCE)
    ap.add_argument('--cluster-type', default=cfg.CLUSTER_TYPE)
    ap.add_argument('--nodes-file', default=cfg.NODES_FILE)
    ap.add_argument('--supra-graph', default=cfg.SUPRA_GRAPH_PKL)
    ap.add_argument('--seed-genes', default=cfg.SEED_GENES_FILE)
    ap.add_argument('--floor', type=float, default=0, help="persistence floor applied before tiering (0 = full population)")
    ap.add_argument('--percentiles', type=float, nargs='+', default=[90, 75, 50])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.output_dir, a.ppi_source)
    os.makedirs(out, exist_ok=True)

    pmap = load_persistence_map(hidef_source='multiplex', nodes_file=a.nodes_file, supra_graph_pkl=a.supra_graph,
                                seed_genes_file=a.seed_genes, persistence_threshold=0)
    if pmap is None: raise SystemExit("no persistence map")
    pmap = apply_persistence_floor(pmap, a.floor)
    thr, labels = compute_persistence_tier_thresholds(list(pmap.values()), a.percentiles)
    v2t = assign_persistence_tiers(sorted(set(pmap.values())), thr, labels)

    df = load_all_enrichment_results(a.output_dir, a.ppi_source, a.cluster_type, ANALYSIS_TYPES)
    if df is None: raise SystemExit("no summarized_results files found")

    rows = []
    for at in ANALYSIS_TYPES:
        d = df[df['analysis_type'] == at]
        if d.empty: continue
        tc = TERM_COL[at] if TERM_COL[at] in d.columns else get_enrichment_term_column_heatmap(d, at)
        vc = get_enrichment_value_column(d, at, 'effect_size')
        print(f"  {at}: term column = {tc!r}, effect column = {vc!r}")
        if tc is None: continue
        r = pd.DataFrame({
            'cluster_id': d['cluster_id'].astype(str).str.replace('HiDeF_Cluster_', '', regex=False),
            'analysis_type': at, 'term': d[tc].astype(str),
            'effect': d[vc] if vc else np.nan,
            'q': d['adjusted_p_value'], 'p': d['p_value'] if 'p_value' in d.columns else np.nan,
            'cluster_size': d['cluster_size'] if 'cluster_size' in d.columns else np.nan,
            'n_genes_hit': d['total_genes_analyzed'] if 'total_genes_analyzed' in d.columns else np.nan})
        rows.append(r)
    long = pd.concat(rows, ignore_index=True)
    long['persistence'] = long['cluster_id'].map(pmap)
    n_unmapped = long['persistence'].isna().sum()
    long = long.dropna(subset=['persistence']).copy()
    long['tier'] = long['persistence'].map(v2t)
    long['tier_index'] = long['tier'].map({l: i + 1 for i, l in enumerate(labels)})
    long['significant'] = long['q'] < 0.05
    long = long.sort_values(['persistence', 'cluster_id', 'analysis_type'], ascending=[False, True, True])
    long.to_csv(os.path.join(out, 'cluster_enrichment_long.csv'), index=False)

    sig = long[long.significant]
    agg = (sig.groupby('cluster_id').agg(n_sources_sig=('analysis_type', 'nunique'),
                                         sources_sig=('analysis_type', lambda x: ','.join(sorted(set(x)))))
           .reset_index())
    summ = pd.DataFrame({'cluster_id': sorted(pmap)}).merge(agg, on='cluster_id', how='left')
    summ['n_sources_sig'] = summ['n_sources_sig'].fillna(0).astype(int)
    summ['sources_sig'] = summ['sources_sig'].fillna('')
    summ['persistence'] = summ['cluster_id'].map(pmap); summ['tier'] = summ['persistence'].map(v2t)
    summ.to_csv(os.path.join(out, 'cluster_enrichment_summary.csv'), index=False)

    print(f"\n{len(long)} enrichment rows over {long.cluster_id.nunique()} clusters "
          f"({n_unmapped} rows dropped: cluster not in persistence map/floor)")
    print(f"{int(long.significant.sum())} rows with q < 0.05; {int((summ.n_sources_sig >= 2).sum())} clusters significant in >= 2 sources")
    print(pd.crosstab(summ['tier'], summ['n_sources_sig']).reindex(labels))
    print(f"-> {out}")


if __name__ == '__main__':
    main()