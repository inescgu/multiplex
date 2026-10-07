#!/usr/bin/env python3
"""
analyze_proteomics.py - Analyze proteomics data (e.g., DRG by gender) across HiDeF clusters

Multiple testing: the family is every test performed -- 2 per cluster
(male-high and female-high). It is corrected once with
pipeline_config.MULTIPLE_TESTING_METHOD BEFORE any significance filtering,
and a cluster is kept only if a direction passes the ADJUSTED threshold.
(Previously clusters were pre-filtered on raw p < 0.05 and FDR was then run
on the survivors only.)

Usage:
    python analyze_proteomics.py --ppi_source STRING --expression_data proteomics.csv --input_dir ./input --output_dir ./output
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact, hypergeom
import pickle
import os
from common_utils import *
from hidef_multiplex_adapter import MultiplexClusterLoaderMixin
from multiple_testing import adjust_pvalues, get_alpha, method_label

class ProteomicsAnalyzer(MultiplexClusterLoaderMixin):
    def __init__(self, ppi_source, proteomics_file, input_dir, output_dir, cluster_type='seed_lcc_hidef', annotation_prefix=None, metadata_file=None, persistence_threshold=None):
        import pipeline_config as cfg
        if persistence_threshold is None:
            persistence_threshold = cfg.PERSISTENCE_THRESHOLD
        print(f"\n{'='*80}")
        print(f"INITIALIZING ProteomicsAnalyzer")
        print(f"{'='*80}")
        print(f"PPI Source: {ppi_source}")
        print(f"Input Dir: {input_dir}")
        print(f"Output Dir: {output_dir}")
        print(f"Cluster Type: {cluster_type}")
        print(f"Persistence Threshold: {persistence_threshold}")
        print(f"Proteomics File: {proteomics_file}")
        print(f"Metadata File: {metadata_file}")
        print(f"Multiple testing: {method_label()} (alpha={get_alpha()})")

        self.ppi_source = ppi_source
        self.proteomics_file = proteomics_file
        self.metadata_file = metadata_file
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cluster_type = cluster_type
        self.persistence_threshold = persistence_threshold
        self.annotation_prefix = None
        self.selected_prefix = None

        # Setup output directories
        self.cluster_type_str = cluster_type
        self.analysis_dir = os.path.join(output_dir, ppi_source, f'proteomics_analysis_{self.cluster_type_str}')

        print(f"\nOutput Configuration:")
        print(f"  Safe prefix: {self.safe_prefix}")
        print(f"  Analysis directory: {self.analysis_dir}")

        ensure_dir(self.analysis_dir)
        print(f"  Directory created/verified: {os.path.exists(self.analysis_dir)}")

        # Data containers
        self.clusters = None
        self.seed_genes = None
        self.cluster_seed_mapping = None
        self.expression_data = None
        self.sample_info = None
        self.results = []
        self.gene_results = []

        print(f"{'='*80}\n")

    @property
    def safe_prefix(self):
        """Return a safe prefix for filename generation"""
        return self.cluster_type

    def safe_log2(self, value, min_value=1e-10, max_value=1e10):
        """Safely calculate log2 with handling for zero/negative values and infinite results."""
        bounded_value = max(min_value, min(value, max_value))
        return np.log2(bounded_value)
    
    def safe_fold_change(self, numerator, denominator, min_value=1e-10):
        """Safely calculate fold change with handling for zero denominators."""
        # Convert to scalar if Series/array
        if hasattr(numerator, 'iloc'):
            numerator = float(numerator.iloc[0]) if len(numerator) > 0 else numerator
        if hasattr(denominator, 'iloc'):
            denominator = float(denominator.iloc[0]) if len(denominator) > 0 else denominator

        # Handle numpy arrays
        numerator = float(numerator) if hasattr(numerator, '__iter__') and not isinstance(numerator, str) else numerator
        denominator = float(denominator) if hasattr(denominator, '__iter__') and not isinstance(denominator, str) else denominator

        return numerator / max(denominator, min_value)
       
    def load_expression_data(self):
        """Load proteomics data and metadata."""
        print(f"\n{'='*80}")
        print(f"LOADING PROTEOMICS DATA")
        print(f"{'='*80}")
        print(f"Proteomics file: {self.proteomics_file}")

        try:
            df = pd.read_csv(self.proteomics_file)

            print(f"  Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")

            # Detect format - could be wide (proteins as rows) or long format
            if 'Protein' in df.columns or 'Gene' in df.columns:
                # Long format or has protein/gene column
                protein_col = 'Protein' if 'Protein' in df.columns else 'Gene'
                print(f"  Using protein/gene column: {protein_col}")
                df = df.set_index(protein_col)

                # Extract sample columns (numeric columns)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                self.expression_data = df[numeric_cols]
                print(f"  Found {len(numeric_cols)} numeric sample columns")
            else:
                # Assume first column is protein/gene names
                print(f"  Assuming first column contains protein/gene names")
                df = df.set_index(df.columns[0])
                self.expression_data = df

            print(f"\n✓ Loaded proteomics data: {self.expression_data.shape[0]} proteins, {self.expression_data.shape[1]} samples")

            # Load metadata if provided
            if self.metadata_file:
                print(f"\nLoading metadata from {self.metadata_file}...")
                metadata_df = pd.read_csv(self.metadata_file)
                print(f"  Metadata shape: {metadata_df.shape}")

                # Create sample info mapping mergedID to metadata
                self.sample_info = pd.DataFrame(index=self.expression_data.columns)

                # Map metadata to sample columns
                matched_samples = 0
                for col in self.expression_data.columns:
                    # Find matching row in metadata
                    matching_row = metadata_df[metadata_df['mergedID'] == col]
                    if not matching_row.empty:
                        self.sample_info.loc[col, 'gender'] = matching_row['Sex'].iloc[0]
                        self.sample_info.loc[col, 'age'] = matching_row['Age'].iloc[0]
                        self.sample_info.loc[col, 'tissue'] = matching_row['Tissue'].iloc[0]
                        self.sample_info.loc[col, 'donorID'] = matching_row['donorID'].iloc[0]
                        self.sample_info.loc[col, 'DRG'] = matching_row['DRG'].iloc[0]
                        matched_samples += 1
                    else:
                        print(f"  Warning: No metadata found for sample {col}")
                        self.sample_info.loc[col, 'gender'] = 'Unknown'

                print(f"  Matched {matched_samples}/{len(self.expression_data.columns)} samples to metadata")

                # Print summary
                if 'gender' in self.sample_info.columns:
                    gender_counts = self.sample_info['gender'].value_counts()
                    print(f"\nGender distribution:")
                    for gender, count in gender_counts.items():
                        print(f"  {gender}: {count} samples")

                    tissue_counts = self.sample_info['tissue'].value_counts()
                    print(f"\nTissue distribution:")
                    for tissue, count in tissue_counts.items():
                        print(f"  {tissue}: {count} samples")
            else:
                # No metadata file provided - create basic sample info
                print("\n⚠ No metadata file provided - cannot determine gender")
                self.sample_info = pd.DataFrame(index=self.expression_data.columns)

            print(f"\n{'='*80}\n")
            return True

        except Exception as e:
            print(f"\n✗ Error loading proteomics data: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            return False
    
    @staticmethod
    def _hypergeom_enrichment(overlap_genes, highly_expressed, total):
        """Overlap count, fold enrichment and hypergeometric p vs a highly-expressed set."""
        overlap = len(overlap_genes.intersection(highly_expressed))
        if overlap == 0:
            return overlap, 1.0, 1.0
        fold_enrichment = (overlap / len(overlap_genes)) / (len(highly_expressed) / total)
        try:
            p_value = hypergeom.sf(overlap - 1, total, len(highly_expressed), len(overlap_genes))
        except Exception:
            p_value = 1.0
        return overlap, fold_enrichment, p_value
    
    def analyze_cluster_expression(self):
        """Analyze protein expression patterns in clusters using hypergeometric test."""
        alpha = get_alpha()
        print(f"\n{'='*80}")
        print(f"STARTING CLUSTER EXPRESSION ANALYSIS")
        print(f"{'='*80}")

        summarized_results = []
        gene_level_results = []

        if 'gender' not in self.sample_info.columns:
            print("✗ No gender information - cannot perform analysis")
            print(f"{'='*80}\n")
            return pd.DataFrame(), pd.DataFrame()

        male_samples = [s for s in self.sample_info[self.sample_info['gender'] == 'M'].index 
                        if s in self.expression_data.columns]
        female_samples = [s for s in self.sample_info[self.sample_info['gender'] == 'F'].index 
                          if s in self.expression_data.columns]

        print(f"Sample groups:")
        print(f"  Male samples: {len(male_samples)}")
        print(f"  Female samples: {len(female_samples)}")

        if not male_samples or not female_samples:
            print("\n✗ Need both male and female samples")
            print(f"{'='*80}\n")
            return pd.DataFrame(), pd.DataFrame()

        male_expr_means = self.expression_data[male_samples].mean(axis=1)
        female_expr_means = self.expression_data[female_samples].mean(axis=1)

        male_threshold = male_expr_means.quantile(0.90)
        female_threshold = female_expr_means.quantile(0.90)

        male_highly_expressed = set(male_expr_means[male_expr_means > male_threshold].index)
        female_highly_expressed = set(female_expr_means[female_expr_means > female_threshold].index)

        print(f"\nExpression thresholds:")
        print(f"  Male threshold (90th percentile): {male_threshold:.2f}")
        print(f"  Female threshold (90th percentile): {female_threshold:.2f}")
        print(f"  Highly expressed in males: {len(male_highly_expressed)} proteins")
        print(f"  Highly expressed in females: {len(female_highly_expressed)} proteins")

        total_proteins = len(self.expression_data.index)
        print(f"  Total proteins in dataset: {total_proteins}")

        expr_genes = set(self.expression_data.index)

        print(f"\nProcessing {len(self.clusters)} clusters...")

        # ---------------- PASS 1: run every test, keep everything ----------------
        cluster_tests = {}
        for cluster_idx, (cluster_name, cluster_genes) in enumerate(self.clusters.items()):
            print(f"\n{'-'*80}")
            print(f"Processing cluster {cluster_idx + 1}/{len(self.clusters)}: {cluster_name}")
            print(f"  Cluster size: {len(cluster_genes)} genes")

            cluster_genes_set = set(str(g) for g in cluster_genes)
            overlap_genes = cluster_genes_set.intersection(expr_genes)

            if len(overlap_genes) < 2:
                print(f"  ✗ Insufficient overlap ({len(overlap_genes)} genes)")
                continue

            print(f"  Genes in expression data: {len(overlap_genes)}")

            male_overlap, male_fe, male_p = self._hypergeom_enrichment(
                overlap_genes, male_highly_expressed, total_proteins)
            print(f"  Male enrichment:")
            print(f"    Overlap: {male_overlap} proteins")
            print(f"    Fold enrichment: {male_fe:.2f}")
            print(f"    P-value: {male_p:.2e}")

            female_overlap, female_fe, female_p = self._hypergeom_enrichment(
                overlap_genes, female_highly_expressed, total_proteins)
            print(f"  Female enrichment:")
            print(f"    Overlap: {female_overlap} proteins")
            print(f"    Fold enrichment: {female_fe:.2f}")
            print(f"    P-value: {female_p:.2e}")

            cluster_tests[cluster_name] = {
                'cluster_genes': cluster_genes,
                'cluster_genes_set': cluster_genes_set,
                'overlap_genes': overlap_genes,
                'male': {'overlap': male_overlap, 'fe': male_fe, 'p': male_p},
                'female': {'overlap': female_overlap, 'fe': female_fe, 'p': female_p},
            }

        # ---------------- PASS 2: correct over the full family ----------------
        names = list(cluster_tests)
        raw_p = []
        for name in names:
            raw_p.append(cluster_tests[name]['male']['p'])
            raw_p.append(cluster_tests[name]['female']['p'])
        adj_p = adjust_pvalues(raw_p)
        for i, name in enumerate(names):
            cluster_tests[name]['male']['adj'] = float(adj_p[2 * i])
            cluster_tests[name]['female']['adj'] = float(adj_p[2 * i + 1])

        print(f"\n{'='*80}")
        print(f"MULTIPLE TESTING CORRECTION")
        print(f"{'='*80}")
        print(f"Method: {method_label()}; family = {len(raw_p)} tests "
              f"({len(names)} clusters x 2 directions); alpha = {alpha}")

        # ---------------- PASS 3: select direction on ADJUSTED significance ----------------
        for cluster_name, t in cluster_tests.items():
            male, female = t['male'], t['female']
            overlap_genes = t['overlap_genes']
            male_significant = male['adj'] < alpha
            female_significant = female['adj'] < alpha

            if male_significant and female_significant:
                # Both significant: choose by fold enrichment
                chosen = 'male' if male['fe'] > female['fe'] else 'female'
            elif male_significant:
                chosen = 'male'
            elif female_significant:
                chosen = 'female'
            else:
                print(f"  ✗ {cluster_name}: no significant gender difference "
                      f"(male adj p={male['adj']:.2e}, female adj p={female['adj']:.2e})")
                continue

            best = male if chosen == 'male' else female
            higher_expr_gender = chosen
            percentage = (best['overlap'] / len(overlap_genes)) * 100 if len(overlap_genes) > 0 else 0
            print(f"  → {cluster_name}: {higher_expr_gender} (FE={best['fe']:.2f}, "
                  f"p={best['p']:.2e}, adj p={best['adj']:.2e})")

            cluster_seed_genes = self.cluster_seed_mapping[cluster_name]

            result = create_standardized_results(
                cluster_name=cluster_name,
                cluster_genes=t['cluster_genes'],
                cluster_seed_genes=cluster_seed_genes,
                cluster_type=self.cluster_type,
                total_genes_analyzed=len(overlap_genes),
                percentage=percentage,
                mean_statistic=best['fe'],
                p_value=best['p'],
                adjusted_p_value=best['adj'],
                statistic_name='fold_enrichment',
                genes_in_expression_data=len(overlap_genes)
            )
            result['higher_expression_gender'] = higher_expr_gender
            summarized_results.append(result)

            # Gene-level results
            for gene in t['cluster_genes_set']:
                gene_level_results.append({
                    'cluster_id': cluster_name,
                    'gene': gene,
                    'is_seed_gene': gene in cluster_seed_genes,
                    'is_in_expression_data': gene in overlap_genes,
                    'cluster_fold_enrichment': best['fe'],
                    'cluster_p_value': best['p'],
                    'cluster_adjusted_p_value': best['adj'],
                    'higher_expression_gender_cluster': higher_expr_gender
                })

        print(f"\n{'='*80}")
        print(f"CLUSTER EXPRESSION ANALYSIS COMPLETE")
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
            print(f"\nAnnotating gene results with cluster significance...")
            gene_results_df['is_cluster_significant'] = gene_results_df['cluster_adjusted_p_value'] < alpha
            sig_count = gene_results_df['is_cluster_significant'].sum()
            print(f"  {sig_count} genes in clusters with significant gender differences")

        self.results = results_df
        self.gene_results = gene_results_df

        print(f"\n{'='*80}\n")
        return results_df, gene_results_df
    
    def create_visualizations(self):
        """Create standardized visualizations."""
        alpha = get_alpha()
        plots_dir = os.path.join('plots', self.ppi_source, f'proteomics_analysis_{self.cluster_type_str}')
        ensure_dir(plots_dir)
        
        if self.results.empty:
            print("No results to visualize")
            return
        
        # 1. Protein coverage by cluster
        plt.figure(figsize=(10, 6))
        self.results_sorted = self.results.sort_values('total_genes_analyzed', ascending=True)
        plt.barh(range(len(self.results_sorted)), self.results_sorted['total_genes_analyzed'])
        plt.yticks(range(len(self.results_sorted)), self.results_sorted['cluster_id'])
        plt.xlabel('Number of Proteins')
        plt.title(f'Protein Coverage by Cluster - {self.ppi_source}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'protein_coverage.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gender comparison if available
        if 'mean_fold_enrichment' in self.results.columns:
            # Filter out rows without meaningful comparisons
            gender_results = self.results[self.results['p_value'] < 1.0].copy()
            
            if not gender_results.empty:
                # Volcano plot
                plt.figure(figsize=(10, 8))
                
                gender_results['neg_log_pvalue'] = -np.log10(gender_results['p_value'] + 1e-300)
                
                # Color by significance
                colors = ['red' if pval < alpha else 'gray' for pval in gender_results['adjusted_p_value']]
                
                plt.scatter(gender_results['mean_fold_enrichment'], gender_results['neg_log_pvalue'], 
                           c=colors, alpha=0.6, s=100)
                
                plt.axhline(y=-np.log10(alpha), color='black', linestyle='--', alpha=0.5)
                plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                
                plt.xlabel('Mean log2(Fold Change) Male/Female')
                plt.ylabel('-log10(p-value)')
                plt.title(f'Gender Differences in Protein Expression - {self.ppi_source}\n(colored by {method_label()} adj p < {alpha})')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'gender_volcano_plot.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Bar plot of significant changes
                sig_results = gender_results[gender_results['adjusted_p_value'] < alpha].copy()
                if not sig_results.empty:
                    sig_results = sig_results.sort_values('mean_fold_enrichment')
                    
                    plt.figure(figsize=(10, max(6, len(sig_results) * 0.3)))
                    
                    colors = ['blue' if x > 0 else 'pink' for x in sig_results['mean_fold_enrichment']]
                    plt.barh(range(len(sig_results)), sig_results['mean_fold_enrichment'], color=colors)
                    plt.yticks(range(len(sig_results)), sig_results['cluster_id'])
                    plt.xlabel('Mean log2(Fold Change) Male/Female')
                    plt.title(f'Significant Gender Differences by Cluster - {self.ppi_source}')
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, 'significant_gender_differences.png'), dpi=300, bbox_inches='tight')
                    plt.close()
        
        # 3. Expression level distribution
        if 'mean_expression' in self.results.columns:
            plt.figure(figsize=(10, 6))
            mean_expressions = self.results['mean_expression'].values
            plt.hist(mean_expressions, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Mean Expression Level')
            plt.ylabel('Number of Clusters')
            plt.title(f'Distribution of Mean Protein Expression - {self.ppi_source}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'expression_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self):
        """Generate a simple text report."""
        alpha = get_alpha()
        report_file = os.path.join(self.analysis_dir, 'analysis_report.txt')
        
        with open(report_file, 'w') as f:
            f.write(f"Proteomics Analysis Report\n")
            f.write(f"=========================\n\n")
            f.write(f"PPI Source: {self.ppi_source}\n")
            f.write(f"Cluster Type: {self.cluster_type}\n")
            f.write(f"Persistence Threshold: {self.persistence_threshold}\n")
            f.write(f"Proteomics file: {self.proteomics_file}\n")
            f.write(f"Multiple testing: {method_label()}, alpha={alpha}\n")
            f.write(f"Total clusters: {len(self.clusters)}\n")
            
            if self.expression_data is not None:
                f.write(f"Total proteins: {self.expression_data.shape[0]}\n")
                f.write(f"Total samples: {self.expression_data.shape[1]}\n")
            
            if self.sample_info is not None and 'gender' in self.sample_info.columns:
                gender_counts = self.sample_info['gender'].value_counts()
                f.write(f"\nSample distribution:\n")
                for gender, count in gender_counts.items():
                    f.write(f"  {gender}: {count} samples\n")
            
            if not self.results.empty:
                f.write(f"\nAnalysis Results:\n")
                f.write(f"Clusters reported (significant after correction): {len(self.results)}\n")
                f.write(f"Average proteins per cluster: {self.results['total_genes_analyzed'].mean():.1f}\n")
                
                gender_comparisons = self.results[self.results['p_value'] < 1.0]
                if not gender_comparisons.empty:
                    sig_gender = gender_comparisons[gender_comparisons['adjusted_p_value'] < alpha]
                    f.write(f"Clusters with significant gender differences (adj p < {alpha}): {len(sig_gender)}\n")
                    
                    # Top differences
                    if len(sig_gender) > 0:
                        f.write("\nTop Gender Differences:\n")
                        top_results = sig_gender.nlargest(5, 'mean_fold_enrichment')
                        for _, row in top_results.iterrows():
                            f.write(f"  {row['cluster_id']}: FE={row['mean_fold_enrichment']:.2f}, ")
                            f.write(f"p={row['p_value']:.2e}, adj_p={row['adjusted_p_value']:.2e}\n")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print(f"\n{'#'*80}")
        print(f"# STARTING PROTEOMICS ANALYSIS")
        print(f"# PPI Source: {self.ppi_source}")
        print(f"# Cluster Type: {self.cluster_type}")
        print(f"{'#'*80}\n")

        # Load clusters
        print(f"STEP 1: Loading HiDeF clusters from comparison results...")
        if not self.load_hidef_clusters_multiplex():
            print("✗ Failed to load HiDeF clusters")
            return False

        print(f"✓ Loaded {len(self.clusters)} clusters with {len(self.seed_genes)} seed genes")
        print(f"  Cluster names: {list(self.clusters.keys())[:5]}{'...' if len(self.clusters) > 5 else ''}")

        # Load proteomics data
        print(f"\nSTEP 2: Loading proteomics data...")
        if not self.load_expression_data():
            return False

        # Run analysis
        print(f"\nSTEP 3: Analyzing protein expression in clusters...")
        results_df, gene_results_df = self.analyze_cluster_expression()

        # Save results
        if not results_df.empty:
            print(f"\nSTEP 4: Saving results...")
            save_results(results_df, gene_results_df, self.analysis_dir, 'proteomics')
            print(f"✓ Main results saved")

            # Create visualizations
            print(f"\nSTEP 5: Creating visualizations...")
            self.create_visualizations()

            # Generate report
            print(f"\nSTEP 6: Generating report...")
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
    parser = argparse.ArgumentParser(description='Analyze proteomics data across HiDeF clusters')
    parser.add_argument('--ppi_source', type=str, required=True, help='PPI source name')
    parser.add_argument('--expression_data', type=str, required=True, help='Proteomics data CSV file')        
    parser.add_argument('--metadata_file', type=str, help='Metadata CSV file with gender information')
    parser.add_argument('--input_dir', type=str, default='./input', help='Input directory with HiDeF comparison results')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--persistence_threshold', type=int, default=None, 
                    help='Minimum persistence threshold for HiDeF clusters (default: pipeline_config.PERSISTENCE_THRESHOLD)')
   
    args = parser.parse_args()
    
    analyzer = ProteomicsAnalyzer(
        ppi_source=args.ppi_source,
        proteomics_file=args.expression_data,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cluster_type='hidef_from_comparison',
        metadata_file=args.metadata_file,
        persistence_threshold=args.persistence_threshold
    )
    
    success = analyzer.run_analysis()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())