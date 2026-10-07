#!/usr/bin/env python3
"""
analyze_spatial_transcriptomics_hidef_v2.py - Analyze spatial transcriptomics data across HiDeF clusters

Loads HiDeF clusters from the output of the HiDeF vs Louvain comparison script.

Usage:
    python analyze_spatial_transcriptomics_hidef_v2.py --ppi_source STRING --expression_data spatial_data.csv --input_dir ./input --output_dir ./output
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import pickle
import os
from common_utils import *
from hidef_multiplex_adapter import MultiplexClusterLoaderMixin

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
    
    def analyze_enrichment(self):
        """Analyze cell type enrichment in each cluster using cluster-level Fisher's exact test."""
        print(f"\n{'='*80}")
        print(f"STARTING ENRICHMENT ANALYSIS")
        print(f"{'='*80}")

        if not self.clusters:
            print("ERROR: No clusters loaded!")
            return pd.DataFrame(), pd.DataFrame()

        print(f"Total clusters to process: {len(self.clusters)}")
        print(f"Total cell types: {len(self.cell_type_genes)}")

        summarized_results = []
        gene_level_results = []
        cluster_info = {}

        for cluster_idx, (cluster_name, cluster_genes) in enumerate(self.clusters.items()):
            print(f"\n{'-'*80}")
            print(f"Processing cluster {cluster_idx + 1}/{len(self.clusters)}: {cluster_name}")
            print(f"  Cluster size: {len(cluster_genes)} genes")

            cluster_genes_set = set(str(g) for g in cluster_genes)
            expr_genes = set(self.expression_data.index)
            overlap_genes = cluster_genes_set.intersection(expr_genes)

            if not overlap_genes:
                print(f"  ✗ No overlap with expression data")
                continue

            print(f"  Genes in expression data: {len(overlap_genes)}")
            cluster_seed_genes = self.cluster_seed_mapping[cluster_name]
            print(f"  Seed genes: {len(cluster_seed_genes)}")

            cell_type_results = []

            print(f"  Testing enrichment for {len(self.cell_type_genes)} cell types...")
            for cell_type, cell_type_gene_set in self.cell_type_genes.items():
                overlap_with_cell_type = overlap_genes.intersection(cell_type_gene_set)
                count = len(overlap_with_cell_type)

                # Cluster-level Fisher's exact test
                a = count  # genes in cluster AND in cell type
                b = len(overlap_genes) - count  # genes in cluster NOT in cell type
                c = len(cell_type_gene_set) - count  # genes in cell type NOT in cluster
                d = len(expr_genes) - a - b - c  # genes in neither

                if a > 0 and d >= 0:
                    try:
                        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
                        odds_ratio = min(odds_ratio, 1e6)
                    except:
                        odds_ratio, p_value = 1.0, 1.0
                else:
                    odds_ratio, p_value = 1.0, 1.0

                cell_type_results.append({
                    'cell_type': cell_type,
                    'odds_ratio': odds_ratio,
                    'p_value': p_value,
                    'genes_in_overlap': count
                })

            if cell_type_results:
                # Find best cell type at cluster level
                best_result = max(cell_type_results, key=lambda x: x['odds_ratio'])

                print(f"  Top cell type: {best_result['cell_type']}")
                print(f"    Odds ratio: {best_result['odds_ratio']:.2f}")
                print(f"    P-value: {best_result['p_value']:.2e}")
                print(f"    Genes in overlap: {best_result['genes_in_overlap']}")

                percentage = (len(overlap_genes) / len(cluster_genes)) * 100

                cluster_info[cluster_name] = {
                    'odds_ratio': best_result['odds_ratio'],
                    'p_value': best_result['p_value'],
                    'top_cell_type': best_result['cell_type']
                }

                result = create_standardized_results(
                    cluster_name=cluster_name,
                    cluster_genes=cluster_genes,
                    cluster_seed_genes=cluster_seed_genes,
                    cluster_type='hidef_from_comparison',
                    total_genes_analyzed=len(overlap_genes),
                    percentage=percentage,
                    mean_statistic=best_result['odds_ratio'],
                    p_value=best_result['p_value'],
                    adjusted_p_value=best_result['p_value'],
                    statistic_name='odds_ratio',
                    genes_in_expression_data=len(overlap_genes)
                )
                result['top_cell_type'] = best_result['cell_type']
                result['top_odds_ratio'] = best_result['odds_ratio']
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
                        'cluster_odds_ratio': cluster_info[cluster_name]['odds_ratio'],
                        'cluster_p_value': cluster_info[cluster_name]['p_value'],
                        'cluster_top_cell_type': cluster_info[cluster_name]['top_cell_type']
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

        # Multiple testing correction at cluster level only
        if not results_df.empty and len(results_df) > 1:
            print(f"\nApplying multiple testing correction...")
            _, adjusted_pvalues, _, _ = multipletests(results_df['p_value'].values, method='fdr_bh')
            results_df['adjusted_p_value'] = adjusted_pvalues
            sig_count = (adjusted_pvalues < 0.05).sum()
            print(f"  Significant clusters (FDR < 0.05): {sig_count}/{len(results_df)}")

        self.results = results_df
        self.gene_results = gene_results_df

        print(f"\n{'='*80}\n")
        return results_df, gene_results_df
    
    def analyze_cell_type_overrepresentation(self):
        """Specialized analysis: Are specific cell types overrepresented in clusters?"""
        cell_type_overrep_results = []
        
        for cluster_name, cluster_genes in self.clusters.items():
            cluster_genes_set = set(str(g) for g in cluster_genes)
            expr_genes = set(self.expression_data.index)
            overlap_genes = cluster_genes_set.intersection(expr_genes)
            
            if not overlap_genes:
                continue
            
            # Analyze each cell type for overrepresentation
            cell_type_enrichments = []
            
            for cell_type, cell_type_gene_set in self.cell_type_genes.items():
                # Count overlapping genes
                overlap_with_cell_type = overlap_genes.intersection(cell_type_gene_set)
                count = len(overlap_with_cell_type)
                
                # Fisher's exact test for overrepresentation
                a = count
                b = len(overlap_genes) - count
                c = len(cell_type_gene_set) - count
                d = len(expr_genes) - a - b - c
                
                if a > 0 and d >= 0:
                    try:
                        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
                        odds_ratio = min(odds_ratio, 1e6)
                    except:
                        odds_ratio, p_value = 1.0, 1.0
                    
                    # Calculate enrichment metrics
                    percentage_of_cluster = (count / len(overlap_genes)) * 100
                    percentage_of_cell_type = (count / len(cell_type_gene_set)) * 100
                    
                    cell_type_enrichments.append({
                        'cell_type': cell_type,
                        'genes_in_overlap': count,
                        'odds_ratio': odds_ratio,
                        'p_value': p_value,
                        'percentage_of_cluster': percentage_of_cluster,
                        'percentage_of_cell_type': percentage_of_cell_type,
                        'overrepresented': p_value < 0.05 and odds_ratio > 1.5,
                        'genes_list': list(overlap_with_cell_type)
                    })
            
            if cell_type_enrichments:
                # Find most overrepresented cell type
                cell_type_enrichments.sort(key=lambda x: x['odds_ratio'], reverse=True)
                top_enrichment = cell_type_enrichments[0]
                
                # Count significantly overrepresented cell types
                sig_overrep_count = sum(1 for ct in cell_type_enrichments if ct['overrepresented'])
                
                # Get all overrepresented cell types
                overrep_cell_types = [ct['cell_type'] for ct in cell_type_enrichments if ct['overrepresented']]
                
                cell_type_overrep_results.append({
                    'cluster_id': cluster_name,
                    'cluster_type': 'hidef_from_comparison',
                    'genes_in_cluster': len(overlap_genes),
                    'cell_types_tested': len(self.cell_type_genes),
                    'significantly_overrepresented_cell_types': sig_overrep_count,
                    'overrepresented_cell_types': overrep_cell_types,
                    'top_overrepresented_cell_type': top_enrichment['cell_type'],
                    'top_odds_ratio': top_enrichment['odds_ratio'],
                    'top_p_value': top_enrichment['p_value'],
                    'top_genes_in_cell_type': top_enrichment['genes_in_overlap'],
                    'top_percentage_of_cluster': top_enrichment['percentage_of_cluster'],
                    'has_cell_type_bias': sig_overrep_count > 0
                })
        
        cell_type_overrep_df = pd.DataFrame(cell_type_overrep_results)
        
        # Apply multiple testing correction
        if not cell_type_overrep_df.empty and len(cell_type_overrep_df) > 1:
            _, adjusted_pvalues, _, _ = multipletests(cell_type_overrep_df['top_p_value'].values, method='fdr_bh')
            cell_type_overrep_df['adjusted_top_p_value'] = adjusted_pvalues
            cell_type_overrep_df['significant_cell_type_overrepresentation'] = cell_type_overrep_df['adjusted_top_p_value'] < 0.05
        
        return cell_type_overrep_df
    
    def create_detailed_cell_type_enrichment_table(self):
        """Create detailed table showing enrichment for each cluster-cell type combination."""
        detailed_enrichment_results = []
        
        for cluster_name, cluster_genes in self.clusters.items():
            cluster_genes_set = set(str(g) for g in cluster_genes)
            expr_genes = set(self.expression_data.index)
            overlap_genes = cluster_genes_set.intersection(expr_genes)
            
            if not overlap_genes:
                continue
            
            # Get seed genes for this cluster
            cluster_seed_genes = self.cluster_seed_mapping[cluster_name]
            
            for cell_type, cell_type_gene_set in self.cell_type_genes.items():
                overlap_with_cell_type = overlap_genes.intersection(cell_type_gene_set)
                count = len(overlap_with_cell_type)
                
                # Fisher's exact test
                a = count
                b = len(overlap_genes) - count
                c = len(cell_type_gene_set) - count
                d = len(expr_genes) - a - b - c
                
                if d >= 0:
                    try:
                        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
                        odds_ratio = min(odds_ratio, 1e6)
                    except:
                        odds_ratio, p_value = 1.0, 1.0
                    
                    # Calculate metrics
                    percentage_of_cluster = (count / len(overlap_genes)) * 100 if overlap_genes else 0
                    percentage_of_cell_type = (count / len(cell_type_gene_set)) * 100
                    
                    # Count seed genes in this cell type
                    seed_genes_in_cell_type = len([g for g in overlap_with_cell_type if g in cluster_seed_genes])
                    
                    detailed_enrichment_results.append({
                        'cluster_id': cluster_name,
                        'cell_type': cell_type,
                        'genes_in_overlap': count,
                        'seed_genes_in_overlap': seed_genes_in_cell_type,
                        'cluster_size': len(overlap_genes),
                        'cell_type_size': len(cell_type_gene_set),
                        'odds_ratio': odds_ratio,
                        'p_value': p_value,
                        'percentage_of_cluster': percentage_of_cluster,
                        'percentage_of_cell_type': percentage_of_cell_type,
                        'enrichment_status': 'Significant' if p_value < 0.05 and odds_ratio > 1.5 else 'Not Significant',
                        'genes_in_cell_type': list(overlap_with_cell_type)
                    })
        
        detailed_df = pd.DataFrame(detailed_enrichment_results)
        
        # Apply multiple testing correction
        if not detailed_df.empty and len(detailed_df) > 1:
            _, adjusted_pvalues, _, _ = multipletests(detailed_df['p_value'].values, method='fdr_bh')
            detailed_df['adjusted_p_value'] = adjusted_pvalues
            detailed_df['significant_enrichment'] = detailed_df['adjusted_p_value'] < 0.05
        
        return detailed_df
    
    def create_visualizations(self):
        """Create standardized visualizations."""
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
        sig_data = self.results[self.results['adjusted_p_value'] < 0.05].copy()
        if not sig_data.empty:
            sig_data['label'] = sig_data['cluster_id'] + ' - ' + sig_data['top_cell_type']
            sig_data = sig_data.sort_values('mean_odds_ratio', ascending=True).tail(20)

            plt.barh(range(len(sig_data)), sig_data['mean_odds_ratio'])
            plt.yticks(range(len(sig_data)), sig_data['label'])
            plt.xlabel('Mean Odds Ratio')
            plt.title(f'Top Significant Enrichments - {self.ppi_source} (HiDeF from Comparison)')
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
        colors = ['red' if pval < 0.05 else 'gray' for pval in self.results['adjusted_p_value']]
        sig_count = sum(1 for c in colors if c == 'red')
        print(f"   Significant results (red): {sig_count}/{len(colors)}")

        plt.scatter(self.results['log2_odds_ratio'], self.results['neg_log_pvalue'], 
                   c=colors, alpha=0.6, s=100)

        plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        plt.xlabel('log2(Mean Odds Ratio)')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Spatial Enrichment Volcano Plot - {self.ppi_source}')
        plt.tight_layout()
        volcano_path = os.path.join(plots_dir, 'spatial_volcano_plot.png')
        plt.savefig(volcano_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {volcano_path}")

        # 4. P-value distribution
        print("\n4. Creating p-value distribution...")
        plt.figure(figsize=(10, 6))
        plt.hist(self.results['p_value'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
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
            f.write(f"Total clusters: {len(self.clusters)}\n")
            f.write(f"Total cell types: {len(self.cell_type_genes)}\n\n")

            if not self.results.empty:
                f.write(f"Analysis Results:\n")
                f.write(f"Total clusters analyzed: {len(self.results)}\n")
                f.write(f"Significant enrichments: {(self.results['adjusted_p_value'] < 0.05).sum()}\n\n")

                # Top enrichments
                f.write("Top 10 Enrichments:\n")
                top_results = self.results.nlargest(10, 'mean_odds_ratio')
                for _, row in top_results.iterrows():
                    f.write(f"  {row['cluster_id']} - {row.get('top_cell_type', 'N/A')}: ")
                    f.write(f"OR={row['mean_odds_ratio']:.2f}, p={row['p_value']:.2e}\n")

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
                sig_cell_type_bias = cell_type_overrep_df[cell_type_overrep_df.get('significant_cell_type_overrepresentation', False)]
                if not sig_cell_type_bias.empty:
                    sig_cell_type_bias.to_csv(os.path.join(self.analysis_dir, 'clusters_with_cell_type_bias.csv'), index=False)
                    print(f"✓ Saved {len(sig_cell_type_bias)} clusters with significant cell type bias")

            # Save detailed cell type enrichment table
            if not detailed_enrichment_df.empty:
                detailed_enrichment_df.to_csv(os.path.join(self.analysis_dir, 'detailed_cell_type_enrichment_table.csv'), index=False)
                print(f"✓ Saved detailed cell type enrichment table")

                # Save significant enrichments
                sig_enrichments = detailed_enrichment_df[detailed_enrichment_df.get('significant_enrichment', False)]
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
                    help='Minimum persistence threshold for HiDeF clusters (default: 5)')
    
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