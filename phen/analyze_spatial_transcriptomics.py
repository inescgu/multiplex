#!/usr/bin/env python3
"""
analyze_spatial_transcriptomics_hidef_v2.py - Analyze spatial transcriptomics data across HiDeF clusters

Loads HiDeF clusters from the output of the HiDeF vs Louvain comparison script.

Multiple testing: every (cluster, cell type) Fisher test is computed ONCE in
`_celltype_test_table()` and corrected ONCE (family = all cluster x cell type
tests) with pipeline_config.MULTIPLE_TESTING_METHOD. The cluster summary,
the cell-type over-representation table and the detailed table all read their
adjusted p-values from that single table, so they can never disagree.
The "top cell type" per cluster is chosen after correction.

Usage:
    python analyze_spatial_transcriptomics_hidef_v2.py --ppi_source STRING --expression_data spatial_data.csv --input_dir ./input --output_dir ./output
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
import pickle
import os
from common_utils import *
from hidef_multiplex_adapter import MultiplexClusterLoaderMixin
from multiple_testing import adjust_pvalues, get_alpha, method_label

class SpatialTranscriptomicsAnalyzer(MultiplexClusterLoaderMixin):
    def __init__(self, ppi_source, expression_file, input_dir, output_dir, persistence_threshold=None):
        import pipeline_config as cfg
        if persistence_threshold is None:
            persistence_threshold = cfg.PERSISTENCE_THRESHOLD
        print(f"\n{'='*80}")
        print(f"INITIALIZING SpatialTranscriptomicsAnalyzer for HiDeF Clusters")
        print(f"{'='*80}")
        print(f"PPI Source: {ppi_source}")
        print(f"Input Dir: {input_dir}")
        print(f"Output Dir: {output_dir}")
        print(f"Persistence Threshold: {persistence_threshold}")
        print(f"Expression File: {expression_file}")
        print(f"Multiple testing: {method_label()} (alpha={get_alpha()})")

        self.ppi_source = ppi_source
        self.expression_file = expression_file
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.persistence_threshold = persistence_threshold
        self.annotation_prefix = None
        self.selected_prefix = None

        # Setup output directories
        self.cluster_type_str = 'hidef_from_comparison'
        self.analysis_dir = os.path.join(output_dir, ppi_source, f'spatial_analysis_{self.cluster_type_str}')

        print(f"\nOutput Configuration:")
        print(f"  Analysis directory: {self.analysis_dir}")

        ensure_dir(self.analysis_dir)
        print(f"  Directory created/verified: {os.path.exists(self.analysis_dir)}")

        # Data containers
        self.clusters = None
        self.seed_genes = None
        self.cluster_seed_mapping = None
        self.expression_data = None
        self.cell_type_genes = {}
        self.results = []
        self.gene_results = []
        self._celltype_tests_df = None   # cache for _celltype_test_table()

        print(f"{'='*80}\n")

    @property
    def safe_prefix(self):
        """Return a safe prefix for filename generation"""
        return self.cluster_type_str

    def safe_log2(self, value, min_value=1e-10, max_value=1e10):
        """Safely calculate log2 with handling for zero/negative values and infinite results."""
        bounded_value = max(min_value, min(value, max_value))
        return np.log2(bounded_value)
    
    def safe_fold_change(self, numerator, denominator, min_value=1e-10):
        """Safely calculate fold change with handling for zero denominators."""
        return numerator / max(denominator, min_value)
    
    def load_expression_data(self, threshold='top_quartile'):
        """Load spatial transcriptomics expression data."""
        print(f"\n{'='*80}")
        print(f"LOADING EXPRESSION DATA")
        print(f"{'='*80}")
        print(f"Expression file: {self.expression_file}")
        print(f"Threshold method: {threshold}")

        try:
            df = pd.read_csv(self.expression_file, index_col=0)
            df.columns = [col.replace(' ', '_') for col in df.columns]

            print(f"  Initial shape: {df.shape[0]} genes, {df.shape[1]} cell types")

            # Remove zero rows
            df = df.loc[(df != 0).any(axis=1)]

            print(f"  After removing zero rows: {df.shape[0]} genes")
            print(f"  Cell types: {', '.join(df.columns)}")

            self.expression_data = df
            self._celltype_tests_df = None   # data changed -> invalidate cache

            print(f"\nIdentifying highly expressed genes for each cell type...")
            # Identify highly expressed genes for each cell type
            for cell_type in df.columns:
                expression_values = df[cell_type]

                # Define threshold
                if threshold == 'top_quartile':
                    thresh = expression_values.quantile(0.75)
                elif threshold == 'top_third':
                    thresh = expression_values.quantile(0.67)
                elif threshold == 'above_mean':
                    thresh = expression_values.mean()
                else:
                    thresh = expression_values.quantile(0.75)

                # Get highly expressed genes
                highly_expressed = expression_values[expression_values > thresh].index.tolist()
                self.cell_type_genes[cell_type] = set(highly_expressed)

                print(f"  {cell_type}: {len(highly_expressed)} highly expressed genes (threshold: {thresh:.2f})")

            print(f"\n✓ Successfully loaded expression data")
            print(f"{'='*80}\n")
            return True

        except Exception as e:
            print(f"\n✗ Error loading expression data: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            return False
    
    # ------------------------------------------------------------------
    # Single source of truth for every cluster x cell type test
    # ------------------------------------------------------------------
    def _celltype_test_table(self, force=False):
        """One row per (cluster, cell type) Fisher test, with raw AND adjusted p.

        The family is every row of this table (clusters with expression-data
        overlap x all cell types). Zero-overlap cell types are kept as
        p = 1 rows so they count toward the family size.
        """
        if self._celltype_tests_df is not None and not force:
            return self._celltype_tests_df

        expr_genes = set(self.expression_data.index)
        rows = []

        for cluster_name, cluster_genes in self.clusters.items():
            cluster_genes_set = set(str(g) for g in cluster_genes)
            overlap_genes = cluster_genes_set.intersection(expr_genes)

            if not overlap_genes:
                continue

            cluster_seed_genes = self.cluster_seed_mapping[cluster_name]

            for cell_type, cell_type_gene_set in self.cell_type_genes.items():
                overlap_with_cell_type = overlap_genes.intersection(cell_type_gene_set)
                count = len(overlap_with_cell_type)

                a = count                                   # in cluster AND cell type
                b = len(overlap_genes) - count              # in cluster, not cell type
                c = len(cell_type_gene_set) - count         # in cell type, not cluster
                d = len(expr_genes) - a - b - c             # in neither

                if a > 0 and d >= 0:
                    try:
                        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
                        odds_ratio = min(odds_ratio, 1e6)
                    except Exception:
                        odds_ratio, p_value = 1.0, 1.0
                else:
                    odds_ratio, p_value = 1.0, 1.0

                rows.append({
                    'cluster_id': cluster_name,
                    'cell_type': cell_type,
                    'genes_in_overlap': count,
                    'seed_genes_in_overlap': len([g for g in overlap_with_cell_type if g in cluster_seed_genes]),
                    'cluster_size': len(overlap_genes),
                    'cell_type_size': len(cell_type_gene_set),
                    'odds_ratio': odds_ratio,
                    'p_value': p_value,
                    'percentage_of_cluster': (count / len(overlap_genes)) * 100,
                    'percentage_of_cell_type': (count / len(cell_type_gene_set)) * 100 if cell_type_gene_set else 0.0,
                    'genes_in_cell_type': sorted(overlap_with_cell_type),
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df['adjusted_p_value'] = adjust_pvalues(df['p_value'].values)
            print(f"\nMultiple testing: {method_label()} over {len(df)} cluster x cell type tests "
                  f"({df['cluster_id'].nunique()} clusters x {len(self.cell_type_genes)} cell types)")

        self._celltype_tests_df = df
        return df
    
    def analyze_enrichment(self):
        """Analyze cell type enrichment in each cluster using cluster-level Fisher's exact test."""
        alpha = get_alpha()
        print(f"\n{'='*80}")
        print(f"STARTING ENRICHMENT ANALYSIS")
        print(f"{'='*80}")

        if not self.clusters:
            print("ERROR: No clusters loaded!")
            return pd.DataFrame(), pd.DataFrame()

        print(f"Total clusters to process: {len(self.clusters)}")
        print(f"Total cell types: {len(self.cell_type_genes)}")

        tests_df = self._celltype_test_table(force=True)

        summarized_results = []
        gene_level_results = []
        expr_genes = set(self.expression_data.index)

        for cluster_idx, (cluster_name, cluster_genes) in enumerate(self.clusters.items()):
            print(f"\n{'-'*80}")
            print(f"Processing cluster {cluster_idx + 1}/{len(self.clusters)}: {cluster_name}")
            print(f"  Cluster size: {len(cluster_genes)} genes")

            cluster_genes_set = set(str(g) for g in cluster_genes)
            overlap_genes = cluster_genes_set.intersection(expr_genes)

            if not overlap_genes:
                print(f"  ✗ No overlap with expression data")
                continue

            print(f"  Genes in expression data: {len(overlap_genes)}")
            cluster_seed_genes = self.cluster_seed_mapping[cluster_name]
            print(f"  Seed genes: {len(cluster_seed_genes)}")

            cluster_rows = tests_df[tests_df['cluster_id'] == cluster_name]
            if cluster_rows.empty:
                continue

            # Best cell type = highest odds ratio (selection rule unchanged);
            # p / adjusted p are those of that specific test.
            best = cluster_rows.loc[cluster_rows['odds_ratio'].idxmax()]
            best_cell_type = best['cell_type']
            best_or = float(best['odds_ratio'])
            best_p = float(best['p_value'])
            best_adj_p = float(best['adjusted_p_value'])

            print(f"  Top cell type: {best_cell_type}")
            print(f"    Odds ratio: {best_or:.2f}")
            print(f"    P-value: {best_p:.2e}  (adj p: {best_adj_p:.2e})")
            print(f"    Genes in overlap: {best['genes_in_overlap']}")

            percentage = (len(overlap_genes) / len(cluster_genes)) * 100

            result = create_standardized_results(
                cluster_name=cluster_name,
                cluster_genes=cluster_genes,
                cluster_seed_genes=cluster_seed_genes,
                cluster_type='hidef_from_comparison',
                total_genes_analyzed=len(overlap_genes),
                percentage=percentage,
                mean_statistic=best_or,
                p_value=best_p,
                adjusted_p_value=best_adj_p,
                statistic_name='odds_ratio',
                genes_in_expression_data=len(overlap_genes)
            )
            result['top_cell_type'] = best_cell_type
            result['top_odds_ratio'] = best_or
            result['cell_type_count'] = len(self.cell_type_genes)
            summarized_results.append(result)

            # Gene-level results: just annotate with cluster-level stats
            for gene in cluster_genes_set:
                is_seed = gene in cluster_seed_genes
                is_in_expr = gene in overlap_genes

                if is_in_expr:
                    gene_expr = self.expression_data.loc[gene]
                    top_gene_cell_type = gene_expr.idxmax()
                    top_cell_type_expression = gene_expr[top_gene_cell_type]
                    mean_expression_all_cell_types = gene_expr.mean()
                    expression_fold_change = self.safe_fold_change(
                        top_cell_type_expression, 
                        mean_expression_all_cell_types
                    )
                    log2_fold_change = self.safe_log2(expression_fold_change)
                    top_cell_type_all_genes = self.expression_data[top_gene_cell_type]
                    expression_percentile = (top_cell_type_all_genes < top_cell_type_expression).sum() / len(top_cell_type_all_genes) * 100
                else:
                    top_gene_cell_type = None
                    top_cell_type_expression = np.nan
                    mean_expression_all_cell_types = np.nan
                    expression_fold_change = np.nan
                    log2_fold_change = np.nan
                    expression_percentile = np.nan

                gene_level_results.append({
                    'cluster_id': cluster_name,
                    'gene': gene,
                    'is_seed_gene': is_seed,
                    'is_in_expression_data': is_in_expr,
                    'expression_in_top_cell_type': top_cell_type_expression,
                    'mean_expression_all_cell_types': mean_expression_all_cell_types,
                    'expression_fold_change': expression_fold_change,
                    'log2_fold_change': log2_fold_change,
                    'expression_percentile_in_top_cell_type': expression_percentile,
                    # Cluster-level stats
                    'cluster_odds_ratio': best_or,
                    'cluster_p_value': best_p,
                    'cluster_adjusted_p_value': best_adj_p,
                    'cluster_top_cell_type': best_cell_type
                })

            print(f"  ✓ Added enrichment results for this cluster")

        print(f"\n{'='*80}")
        print(f"ENRICHMENT ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total cluster enrichments: {len(summarized_results)}")
        print(f"Total gene-level results: {len(gene_level_results)}")

        results_df = pd.DataFrame(summarized_results)
        gene_results_df = pd.DataFrame(gene_level_results)

        print(f"\nResults DataFrame shape: {results_df.shape}")
        print(f"Gene Results DataFrame shape: {gene_results_df.shape}")

        if not results_df.empty:
            sig_count = (results_df['adjusted_p_value'] < alpha).sum()
            print(f"  Significant clusters (adj p < {alpha}): {sig_count}/{len(results_df)}")

        if not gene_results_df.empty:
            gene_results_df['is_cluster_significant'] = gene_results_df['cluster_adjusted_p_value'] < alpha

        self.results = results_df
        self.gene_results = gene_results_df

        print(f"\n{'='*80}\n")
        return results_df, gene_results_df
    
    def analyze_cell_type_overrepresentation(self):
        """Specialized analysis: Are specific cell types overrepresented in clusters?

        Reads the shared, already-corrected test table."""
        alpha = get_alpha()
        tests_df = self._celltype_test_table()
        if tests_df.empty:
            return pd.DataFrame()

        # Only cell types with at least one overlapping gene are reported (as before),
        # but they were all counted in the correction family.
        reported = tests_df[tests_df['genes_in_overlap'] > 0].copy()
        reported['overrepresented'] = (reported['adjusted_p_value'] < alpha) & (reported['odds_ratio'] > 1.5)

        cell_type_overrep_results = []

        for cluster_name, rows in reported.groupby('cluster_id', sort=False):
            rows = rows.sort_values('odds_ratio', ascending=False, kind='stable')
            top = rows.iloc[0]
            overrep_cell_types = rows.loc[rows['overrepresented'], 'cell_type'].tolist()

            cell_type_overrep_results.append({
                'cluster_id': cluster_name,
                'cluster_type': 'hidef_from_comparison',
                'genes_in_cluster': int(top['cluster_size']),
                'cell_types_tested': len(self.cell_type_genes),
                'significantly_overrepresented_cell_types': len(overrep_cell_types),
                'overrepresented_cell_types': overrep_cell_types,
                'top_overrepresented_cell_type': top['cell_type'],
                'top_odds_ratio': top['odds_ratio'],
                'top_p_value': top['p_value'],
                'adjusted_top_p_value': top['adjusted_p_value'],
                'top_genes_in_cell_type': int(top['genes_in_overlap']),
                'top_percentage_of_cluster': top['percentage_of_cluster'],
                'has_cell_type_bias': len(overrep_cell_types) > 0,
                'significant_cell_type_overrepresentation': bool(top['adjusted_p_value'] < alpha)
            })
        
        return pd.DataFrame(cell_type_overrep_results)
    
    def create_detailed_cell_type_enrichment_table(self):
        """Detailed table: one row per cluster x cell type, using the shared corrected table."""
        alpha = get_alpha()
        tests_df = self._celltype_test_table()
        if tests_df.empty:
            return pd.DataFrame()

        detailed_df = tests_df.copy()
        detailed_df['enrichment_status'] = np.where(
            (detailed_df['adjusted_p_value'] < alpha) & (detailed_df['odds_ratio'] > 1.5),
            'Significant', 'Not Significant')
        detailed_df['significant_enrichment'] = detailed_df['adjusted_p_value'] < alpha
        return detailed_df
    
    def create_visualizations(self):
        """Create standardized visualizations."""
        alpha = get_alpha()
        print(f"\n{'='*80}")
        print(f"CREATING VISUALIZATIONS")
        print(f"{'='*80}")

        plots_dir = os.path.join('plots', self.ppi_source, f'spatial_analysis_{self.cluster_type_str}')
        ensure_dir(plots_dir)
        print(f"Plots directory: {plots_dir}")

        if self.results.empty:
            print("✗ No results to visualize")
            return

        print(f"Creating visualizations for {len(self.results)} results...")

        # 1. Summary heatmap by cell type
        print("\n1. Creating summary heatmap...")
        if 'top_cell_type' in self.results.columns:
            # Create a more detailed heatmap showing enrichment by cell type
            detailed_results = []
            for cluster_name, cluster_genes in self.clusters.items():
                cluster_genes_set = set(str(g) for g in cluster_genes)
                expr_genes = set(self.expression_data.index)
                overlap_genes = cluster_genes_set.intersection(expr_genes)

                if overlap_genes:
                    for cell_type, cell_type_gene_set in self.cell_type_genes.items():
                        overlap_with_cell_type = overlap_genes.intersection(cell_type_gene_set)
                        count = len(overlap_with_cell_type)
                        percentage = (count / len(overlap_genes)) * 100 if overlap_genes else 0

                        detailed_results.append({
                            'cluster_id': cluster_name,
                            'category': cell_type,
                            'percentage': percentage
                        })

            if detailed_results:
                detailed_df = pd.DataFrame(detailed_results)
                fig = create_summary_heatmap(
                    detailed_df, 
                    value_col='percentage',
                    title=f'Cell Type Expression Enrichment - {self.ppi_source} (HiDeF from Comparison)'
                )
                if fig:
                    heatmap_path = os.path.join(plots_dir, 'cell_type_enrichment_heatmap.png')
                    fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"   ✓ Saved: {heatmap_path}")

        # 2. Significance plot
        print("\n2. Creating significance plot...")
        plt.figure(figsize=(12, 8))

        # Prepare data for plotting
        sig_data = self.results[self.results['adjusted_p_value'] < alpha].copy()
        if not sig_data.empty:
            sig_data['label'] = sig_data['cluster_id'] + ' - ' + sig_data['top_cell_type']
            sig_data = sig_data.sort_values('mean_odds_ratio', ascending=True).tail(20)

            plt.barh(range(len(sig_data)), sig_data['mean_odds_ratio'])
            plt.yticks(range(len(sig_data)), sig_data['label'])
            plt.xlabel('Mean Odds Ratio')
            plt.title(f'Top Significant Enrichments ({method_label()} adj p < {alpha}) - {self.ppi_source} (HiDeF from Comparison)')
            plt.tight_layout()
            sig_path = os.path.join(plots_dir, 'significant_enrichments.png')
            plt.savefig(sig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: {sig_path}")
        else:
            print(f"   No significant results to plot")
            plt.close()

        # 3. Volcano plot
        print("\n3. Creating volcano plot...")
        plt.figure(figsize=(10, 8))

        self.results['neg_log_pvalue'] = -np.log10(self.results['p_value'] + 1e-300)
        self.results['log2_odds_ratio'] = self.results['mean_odds_ratio'].apply(self.safe_log2)

        # Color by significance
        colors = ['red' if pval < alpha else 'gray' for pval in self.results['adjusted_p_value']]
        sig_count = sum(1 for c in colors if c == 'red')
        print(f"   Significant results (red, adj p < {alpha}): {sig_count}/{len(colors)}")

        plt.scatter(self.results['log2_odds_ratio'], self.results['neg_log_pvalue'], 
                   c=colors, alpha=0.6, s=100)

        plt.axhline(y=-np.log10(alpha), color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        plt.xlabel('log2(Mean Odds Ratio)')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Spatial Enrichment Volcano Plot - {self.ppi_source}\n(colored by {method_label()} adj p < {alpha})')
        plt.tight_layout()
        volcano_path = os.path.join(plots_dir, 'spatial_volcano_plot.png')
        plt.savefig(volcano_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {volcano_path}")

        # 4. P-value distribution
        print("\n4. Creating p-value distribution...")
        plt.figure(figsize=(10, 6))
        plt.hist(self.results['p_value'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=alpha, color='red', linestyle='--', label=f'p={alpha}')
        plt.xlabel('P-value')
        plt.ylabel('Frequency')
        plt.title('P-value Distribution')
        plt.legend()
        plt.tight_layout()
        pval_path = os.path.join(plots_dir, 'pvalue_distribution.png')
        plt.savefig(pval_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {pval_path}")

        print(f"\n{'='*80}\n")
    
    def generate_report(self):
        """Generate a simple text report."""
        alpha = get_alpha()
        print(f"\n{'='*80}")
        print(f"GENERATING REPORT")
        print(f"{'='*80}")

        report_file = os.path.join(self.analysis_dir, 'analysis_report.txt')
        print(f"Report file: {report_file}")

        with open(report_file, 'w') as f:
            f.write(f"Spatial Transcriptomics Analysis Report - HiDeF Clusters from Comparison\n")
            f.write(f"======================================================================\n\n")
            f.write(f"PPI Source: {self.ppi_source}\n")
            f.write(f"Cluster Source: HiDeF vs Louvain Comparison Results\n")
            f.write(f"Persistence Threshold: {self.persistence_threshold}\n")
            f.write(f"Expression file: {self.expression_file}\n")
            f.write(f"Multiple testing: {method_label()}, alpha={alpha} "
                    f"(family = all cluster x cell type tests)\n")
            f.write(f"Total clusters: {len(self.clusters)}\n")
            f.write(f"Total cell types: {len(self.cell_type_genes)}\n\n")

            if not self.results.empty:
                f.write(f"Analysis Results:\n")
                f.write(f"Total clusters analyzed: {len(self.results)}\n")
                f.write(f"Significant enrichments (adj p < {alpha}): {(self.results['adjusted_p_value'] < alpha).sum()}\n\n")

                # Top enrichments
                f.write("Top 10 Enrichments:\n")
                top_results = self.results.nlargest(10, 'mean_odds_ratio')
                for _, row in top_results.iterrows():
                    f.write(f"  {row['cluster_id']} - {row.get('top_cell_type', 'N/A')}: ")
                    f.write(f"OR={row['mean_odds_ratio']:.2f}, p={row['p_value']:.2e}, "
                            f"adj_p={row['adjusted_p_value']:.2e}\n")

                # Cell type distribution
                if 'top_cell_type' in self.results.columns:
                    f.write(f"\nTop Cell Type Associations:\n")
                    cell_type_counts = self.results['top_cell_type'].value_counts()
                    for cell_type, count in cell_type_counts.head(5).items():
                        f.write(f"  {cell_type}: {count} clusters\n")

        print(f"✓ Report saved: {report_file}")
        print(f"\n{'='*80}\n")    
        
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print(f"\n{'#'*80}")
        print(f"# STARTING SPATIAL TRANSCRIPTOMICS ANALYSIS - HIDEF CLUSTERS FROM COMPARISON")
        print(f"# PPI Source: {self.ppi_source}")
        print(f"{'#'*80}\n")

        # Load clusters
        print(f"STEP 1: Loading HiDeF clusters from comparison results...")
        if not self.load_hidef_clusters_multiplex():
            print("✗ Failed to load HiDeF clusters")
            return False

        print(f"✓ Loaded {len(self.clusters)} clusters with {len(self.seed_genes)} seed genes")
        print(f"  Cluster names: {list(self.clusters.keys())[:5]}{'...' if len(self.clusters) > 5 else ''}")

        # Load expression data
        print(f"\nSTEP 2: Loading expression data...")
        if not self.load_expression_data():
            return False

        # Run enrichment analysis
        print(f"\nSTEP 3: Running enrichment analysis...")
        results_df, gene_results_df = self.analyze_enrichment()

        # Run specialized cell type overrepresentation analysis
        print(f"\nSTEP 4: Analyzing cell type overrepresentation in clusters...")
        cell_type_overrep_df = self.analyze_cell_type_overrepresentation()

        # Create detailed cell type enrichment table
        print(f"\nSTEP 5: Creating detailed cell type enrichment table...")
        detailed_enrichment_df = self.create_detailed_cell_type_enrichment_table()

        # Save results
        if not results_df.empty:
            print(f"\nSTEP 6: Saving results...")
            save_results(results_df, gene_results_df, self.analysis_dir, 'spatial_transcriptomics_hidef_comparison')
            print(f"✓ Main results saved")

            # Save specialized cell type overrepresentation results
            if not cell_type_overrep_df.empty:
                cell_type_overrep_df.to_csv(os.path.join(self.analysis_dir, 'cell_type_overrepresentation_analysis.csv'), index=False)
                print(f"✓ Saved cell type overrepresentation analysis")

                # Save clusters with significant cell type bias
                sig_cell_type_bias = cell_type_overrep_df[cell_type_overrep_df['significant_cell_type_overrepresentation']]
                if not sig_cell_type_bias.empty:
                    sig_cell_type_bias.to_csv(os.path.join(self.analysis_dir, 'clusters_with_cell_type_bias.csv'), index=False)
                    print(f"✓ Saved {len(sig_cell_type_bias)} clusters with significant cell type bias")

            # Save detailed cell type enrichment table
            if not detailed_enrichment_df.empty:
                detailed_enrichment_df.to_csv(os.path.join(self.analysis_dir, 'detailed_cell_type_enrichment_table.csv'), index=False)
                print(f"✓ Saved detailed cell type enrichment table")

                # Save significant enrichments
                sig_enrichments = detailed_enrichment_df[detailed_enrichment_df['significant_enrichment']]
                if not sig_enrichments.empty:
                    sig_enrichments.to_csv(os.path.join(self.analysis_dir, 'significant_cell_type_enrichments.csv'), index=False)
                    print(f"✓ Saved {len(sig_enrichments)} significant cell type enrichments")

            # Create visualizations
            print(f"\nSTEP 7: Creating visualizations...")
            self.create_visualizations()

            # Generate report
            print(f"\nSTEP 8: Generating report...")
            self.generate_report()

            print(f"\n{'#'*80}")
            print(f"# ANALYSIS COMPLETE!")
            print(f"# Results saved to: {self.analysis_dir}")
            print(f"{'#'*80}\n")

            return True
        else:
            print("\n✗ No results generated")
            print(f"  - Clusters loaded: {len(self.clusters) if self.clusters else 0}")
            print(f"  - Results DataFrame empty: {results_df.empty}")
            print(f"  - Gene Results DataFrame empty: {gene_results_df.empty}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Analyze spatial transcriptomics data across HiDeF clusters from comparison results')
    parser.add_argument('--ppi_source', type=str, required=True, help='PPI source name')
    parser.add_argument('--expression_data', type=str, required=True, help='Expression data CSV file')
    parser.add_argument('--input_dir', type=str, default='./input', help='Input directory with comparison results')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--persistence_threshold', type=int, default=None, 
                    help='Minimum persistence threshold for HiDeF clusters (default: pipeline_config.PERSISTENCE_THRESHOLD)')
    
    args = parser.parse_args()
    
    analyzer = SpatialTranscriptomicsAnalyzer(
        ppi_source=args.ppi_source,
        expression_file=args.expression_data,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        persistence_threshold=args.persistence_threshold
    )
    
    success = analyzer.run_analysis()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())