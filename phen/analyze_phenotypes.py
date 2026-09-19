#!/usr/bin/env python3
"""
analyze_phenotypes.py - Analyze pain phenotypes across HiDeF clusters

Usage:
    python analyze_phenotypes.py --ppi_source STRING --expression_data phenotypes.csv --input_dir ./input --output_dir ./output
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

class PhenotypesAnalyzer:
    def __init__(self, ppi_source, expression_file, input_dir, output_dir, cluster_type='seed_lcc_hidef', persistence_threshold=5):
        print(f"\n{'='*80}")
        print(f"INITIALIZING PhenotypesAnalyzer")
        print(f"{'='*80}")
        print(f"PPI Source: {ppi_source}")
        print(f"Input Dir: {input_dir}")
        print(f"Output Dir: {output_dir}")
        print(f"Cluster Type: {cluster_type}")
        print(f"Persistence Threshold: {persistence_threshold}")
        print(f"Expression File: {expression_file}")

        self.ppi_source = ppi_source
        self.expression_file = expression_file
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cluster_type = cluster_type
        self.persistence_threshold = persistence_threshold
        self.annotation_prefix = None
        self.selected_prefix = None

        # Setup output directories
        self.cluster_type_str = cluster_type
        self.analysis_dir = os.path.join(output_dir, ppi_source, f'phenotypes_analysis_{self.cluster_type_str}')

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
        self.phenotype_genes = {}
        self.results = []
        self.gene_results = []
        self.gene_phenotypes={}
        self.phenotype_genes={}        

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
        return numerator / max(denominator, min_value)
    
    def load_hidef_clusters_multiplex(self, nodes_file, supra_graph_pkl=None,
                                   seed_genes_file=None):
        '''Drop-in replacement for load_hidef_clusters_from_comparison() that
        reads the heterogeneous pipeline's .nodes file instead of the old
        monoplex hidef_communities.pkl. Populates the exact same attributes
        (self.clusters, self.cluster_seed_mapping, self.seed_genes) so nothing
        else in this class needs to change.'''
        import pickle
        from hidef_multiplex_adapter import load_multiplex_clusters

        node_names = None
        if supra_graph_pkl:
            with open(supra_graph_pkl, "rb") as f:
                g = pickle.load(f)
            node_names = g.vs["name"]

        seed_genes = None
        if seed_genes_file:
            seed_genes = set(pd.read_csv(seed_genes_file)["gene"].astype(str))

        self.clusters, self.cluster_seed_mapping, self._multiplex_extra = load_multiplex_clusters(
            nodes_file, node_names=node_names, seed_genes=seed_genes,
            persistence_threshold=self.persistence_threshold,
        )
        self.seed_genes = (
            set().union(*self.cluster_seed_mapping.values())
            if self.cluster_seed_mapping else set()
        )

        if self.clusters:
            sizes = [len(g) for g in self.clusters.values()]
            print(f"Loaded {len(self.clusters)} multiplex clusters "
                  f"(deduped gene-set sizes {min(sizes)}-{max(sizes)}, "
                  f"avg {sum(sizes)/len(sizes):.1f})")
        return bool(self.clusters)

        
    def load_phenotype_data(self):
        """Load gene-phenotype associations."""
        print(f"\n{'='*80}")
        print(f"LOADING PHENOTYPE DATA")
        print(f"{'='*80}")
        print(f"Phenotype file: {self.expression_file}")

        try:
            df = pd.read_csv(self.expression_file, low_memory=False)
            df.columns = [col.strip() for col in df.columns]

            print(f"  Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"  Available columns: {df.columns.tolist()}")

            # Find gene and phenotype columns
            gene_col = None
            phenotype_col = None

            for col in ['Loci', 'Gene', 'GeneSymbol']:
                if col in df.columns:
                    gene_col = col
                    break

            for col in ['Phenotype', 'Trait', 'Condition']:
                if col in df.columns:
                    phenotype_col = col
                    break

            if not gene_col or not phenotype_col:
                print(f"✗ Could not find gene/phenotype columns")
                print(f"  Available columns: {df.columns.tolist()}")
                print(f"{'='*80}\n")
                return False

            print(f"  Using gene column: {gene_col}")
            print(f"  Using phenotype column: {phenotype_col}")

            # Clean data
            df = df.dropna(subset=[phenotype_col])
            df = df[df[gene_col].notna()]  # Keep rows where gene column is not null
            df[phenotype_col] = df[phenotype_col].astype(str).str.lower().str.strip()
            df[gene_col] = df[gene_col].astype(str).str.strip()

            # Remove rows with empty gene names
            df = df[df[gene_col] != '']
            df = df[df[gene_col] != 'nan']

            print(f"  After cleaning: {df.shape[0]} rows")

            # Initialize phenotype_genes if not already done
            if not hasattr(self, 'phenotype_genes'):
                self.phenotype_genes = {}

            # Create mappings
            print(f"\nCreating gene-phenotype mappings...")
            genes_added = 0
            phenotypes_added = 0

            for _, row in df.iterrows():
                gene = row[gene_col]
                phenotype = row[phenotype_col]

                if gene and phenotype and gene != 'nan' and phenotype != 'nan':
                    # Add to gene_phenotypes
                    if gene not in self.gene_phenotypes:
                        self.gene_phenotypes[gene] = set()
                        genes_added += 1
                    self.gene_phenotypes[gene].add(phenotype)

                    # Add to phenotype_genes
                    if phenotype not in self.phenotype_genes:
                        self.phenotype_genes[phenotype] = set()
                        phenotypes_added += 1
                    self.phenotype_genes[phenotype].add(gene)

            print(f"✓ Loaded {len(self.gene_phenotypes)} genes with {len(self.phenotype_genes)} phenotypes")

            # Show top phenotypes
            phenotype_counts = [(p, len(genes)) for p, genes in self.phenotype_genes.items()]
            phenotype_counts.sort(key=lambda x: x[1], reverse=True)

            print(f"\nTop 10 phenotypes:")
            for phenotype, count in phenotype_counts[:10]:
                print(f"  {phenotype}: {count} genes")

            print(f"\n{'='*80}\n")
            return True

        except Exception as e:
            print(f"\n✗ Error loading phenotype data: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            return False
    
    def analyze_enrichment(self):
        """Analyze phenotype enrichment in each cluster using hypergeometric test."""
        print(f"\n{'='*80}")
        print(f"STARTING ENRICHMENT ANALYSIS")
        print(f"{'='*80}")

        if not self.clusters:
            print("ERROR: No clusters loaded!")
            return pd.DataFrame(), pd.DataFrame()

        print(f"Total clusters to process: {len(self.clusters)}")
        print(f"Total phenotypes: {len(self.phenotype_genes)}")

        summarized_results = []
        gene_level_results = []
        cluster_info = {}

        for cluster_idx, (cluster_name, cluster_genes) in enumerate(self.clusters.items()):
            print(f"\n{'-'*80}")
            print(f"Processing cluster {cluster_idx + 1}/{len(self.clusters)}: {cluster_name}")
            print(f"  Cluster size: {len(cluster_genes)} genes")

            cluster_genes_set = set(str(g) for g in cluster_genes)
            phenotype_gene_set = set(self.gene_phenotypes.keys())
            overlap_genes = cluster_genes_set.intersection(phenotype_gene_set)

            if not overlap_genes:
                print(f"  ✗ No overlap with phenotype data")
                continue

            print(f"  Genes in phenotype data: {len(overlap_genes)}")

            phenotype_results = []
            total_genes = len(self.gene_phenotypes)

            print(f"  Testing enrichment for {len(self.phenotype_genes)} phenotypes...")
            for phenotype, phenotype_genes in self.phenotype_genes.items():
                genes_in_both = len(overlap_genes.intersection(phenotype_genes))

                if genes_in_both == 0:
                    continue

                fold_enrichment = (genes_in_both / len(overlap_genes)) / (len(phenotype_genes) / total_genes)

                try:
                    from scipy.stats import hypergeom
                    p_value = hypergeom.sf(
                        genes_in_both - 1,
                        total_genes,
                        len(phenotype_genes),
                        len(overlap_genes)
                    )
                except:
                    p_value = 1.0

                phenotype_results.append({
                    'phenotype': phenotype,
                    'fold_enrichment': fold_enrichment,
                    'p_value': p_value,
                    'genes_in_phenotype': genes_in_both
                })

            if phenotype_results:
                best_result = max(phenotype_results, key=lambda x: x['fold_enrichment'])
                top_phenotype = best_result['phenotype']
                best_fold_enrichment = best_result['fold_enrichment']
                best_p_value = best_result['p_value']

                print(f"  Top phenotype: {top_phenotype}")
                print(f"    Fold enrichment: {best_fold_enrichment:.2f}")
                print(f"    P-value: {best_p_value:.2e}")
                print(f"    Genes in phenotype: {best_result['genes_in_phenotype']}")

                percentage = (len(overlap_genes) / len(cluster_genes)) * 100
                cluster_seed_genes = self.cluster_seed_mapping[cluster_name]

                cluster_info[cluster_name] = {
                    'fold_enrichment': best_fold_enrichment,
                    'p_value': best_p_value,
                    'top_phenotype': top_phenotype,
                    'genes_in_phenotype': best_result['genes_in_phenotype']
                }

                result = create_standardized_results(
                    cluster_name=cluster_name,
                    cluster_genes=cluster_genes,
                    cluster_seed_genes=cluster_seed_genes,
                    cluster_type=self.cluster_type,
                    total_genes_analyzed=len(overlap_genes),
                    percentage=percentage,
                    mean_statistic=best_fold_enrichment,
                    p_value=best_p_value,
                    adjusted_p_value=best_p_value,
                    statistic_name='fold_enrichment',
                    genes_in_expression_data=len(overlap_genes)
                )
                result['top_phenotype'] = top_phenotype
                result['phenotype_count'] = len(phenotype_results)
                result['genes_in_phenotype'] = best_result['genes_in_phenotype']
                summarized_results.append(result)

                for gene in cluster_genes_set:
                    is_seed = gene in cluster_seed_genes
                    is_in_expr = gene in overlap_genes

                    gene_level_results.append({
                        'cluster_id': cluster_name,
                        'gene': gene,
                        'is_seed_gene': is_seed,
                        'is_in_expression_data': is_in_expr,
                        'cluster_fold_enrichment': cluster_info[cluster_name]['fold_enrichment'],
                        'cluster_p_value': cluster_info[cluster_name]['p_value'],
                        'cluster_top_phenotype': cluster_info[cluster_name]['top_phenotype']
                    })

                print(f"  ✓ Added enrichment results for this cluster")

        print(f"\n{'='*80}")
        print(f"ENRICHMENT ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total phenotype enrichments: {len(summarized_results)}")
        print(f"Total gene-level results: {len(gene_level_results)}")

        results_df = pd.DataFrame(summarized_results)
        gene_results_df = pd.DataFrame(gene_level_results)

        print(f"\nResults DataFrame shape: {results_df.shape}")
        print(f"Gene Results DataFrame shape: {gene_results_df.shape}")

        if not results_df.empty and len(results_df) > 1:
            print(f"\nApplying multiple testing correction...")
            _, adjusted_pvalues, _, _ = multipletests(results_df['p_value'].values, method='fdr_bh')
            results_df['adjusted_p_value'] = adjusted_pvalues
            sig_count = (adjusted_pvalues < 0.05).sum()
            print(f"  Significant clusters (FDR < 0.05): {sig_count}/{len(results_df)}")

        if not gene_results_df.empty:
            print(f"\nAnnotating gene results with cluster significance...")
            gene_results_df['is_cluster_significant'] = gene_results_df['cluster_id'].map(
                lambda x: results_df[results_df['cluster_id'] == x]['adjusted_p_value'].values[0] < 0.05
                if len(results_df[results_df['cluster_id'] == x]) > 0 else False
            )
            sig_count = gene_results_df['is_cluster_significant'].sum()
            print(f"  {sig_count} genes in clusters with significant phenotype enrichments")

        self.results = results_df
        self.gene_results = gene_results_df

        print(f"\n{'='*80}\n")
        return results_df, gene_results_df
    
    def analyze_pain_phenotype_overrepresentation(self):
        """Specialized analysis: Are certain pain phenotypes overrepresented in clusters?"""
        pain_keywords = ['pain', 'nociception', 'analgesia', 'hyperalgesia', 'allodynia', 'neuropathy', 'neuralgia']
        
        # Get all pain-related phenotypes
        pain_phenotypes = set()
        for phenotype in self.phenotype_genes.keys():
            if any(keyword in phenotype.lower() for keyword in pain_keywords):
                pain_phenotypes.add(phenotype)
        
        if not pain_phenotypes:
            print("No pain-related phenotypes found")
            return pd.DataFrame()
        
        pain_phenotype_results = []
        
        for cluster_name, cluster_genes in self.clusters.items():
            cluster_genes_set = set(str(g) for g in cluster_genes)
            phenotype_gene_set = set(self.gene_phenotypes.keys())
            overlap_genes = cluster_genes_set.intersection(phenotype_gene_set)
            
            if not overlap_genes:
                continue
            
            # Analyze each pain phenotype
            cluster_pain_phenotypes = []
            total_pain_genes_in_cluster = 0
            pain_phenotype_odds_ratios = []
            pain_phenotype_p_values = []
            
            for pain_phenotype in pain_phenotypes:
                # Count genes in cluster with this pain phenotype
                genes_with_pain_phenotype = []
                for gene in overlap_genes:
                    if gene in self.gene_phenotypes and pain_phenotype in self.gene_phenotypes[gene]:
                        genes_with_pain_phenotype.append(gene)
                
                if genes_with_pain_phenotype:
                    cluster_pain_phenotypes.append(pain_phenotype)
                    total_pain_genes_in_cluster += len(genes_with_pain_phenotype)
                    
                    # Fisher's exact test for this pain phenotype
                    phenotype_total = len(self.phenotype_genes[pain_phenotype])
                    total_genes = len(self.gene_phenotypes)
                    cluster_size = len(overlap_genes)
                    
                    a = len(genes_with_pain_phenotype)
                    b = cluster_size - a
                    c = phenotype_total - a
                    d = total_genes - a - b - c
                    
                    if a > 0 and d >= 0:
                        try:
                            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
                            odds_ratio = min(odds_ratio, 1e6)
                            pain_phenotype_odds_ratios.append(odds_ratio)
                            pain_phenotype_p_values.append(p_value)
                        except:
                            pain_phenotype_odds_ratios.append(1.0)
                            pain_phenotype_p_values.append(1.0)
            
            if cluster_pain_phenotypes:
                # Summary statistics for this cluster
                mean_pain_odds_ratio = np.mean(pain_phenotype_odds_ratios) if pain_phenotype_odds_ratios else 1.0
                min_pain_p_value = min(pain_phenotype_p_values) if pain_phenotype_p_values else 1.0
                pain_percentage = (total_pain_genes_in_cluster / len(overlap_genes)) * 100
                
                # Most significant pain phenotype
                if pain_phenotype_p_values:
                    most_sig_idx = np.argmin(pain_phenotype_p_values)
                    top_pain_phenotype = list(pain_phenotypes)[most_sig_idx]
                    top_pain_odds_ratio = pain_phenotype_odds_ratios[most_sig_idx]
                else:
                    top_pain_phenotype = cluster_pain_phenotypes[0] if cluster_pain_phenotypes else 'None'
                    top_pain_odds_ratio = 1.0
                
                pain_phenotype_results.append({
                    'cluster_id': cluster_name,
                    'cluster_type': self.cluster_type,
                    'pain_phenotypes_in_cluster': cluster_pain_phenotypes,
                    'pain_phenotype_count': len(cluster_pain_phenotypes),
                    'total_pain_genes_in_cluster': total_pain_genes_in_cluster,
                    'cluster_size': len(overlap_genes),
                    'pain_gene_percentage': pain_percentage,
                    'mean_pain_odds_ratio': mean_pain_odds_ratio,
                    'top_pain_phenotype': top_pain_phenotype,
                    'top_pain_odds_ratio': top_pain_odds_ratio,
                    'min_pain_p_value': min_pain_p_value,
                    'overrepresented_for_pain': min_pain_p_value < 0.05 and mean_pain_odds_ratio > 1.5
                })
        
        pain_phenotype_df = pd.DataFrame(pain_phenotype_results)
        
        # Apply multiple testing correction
        if not pain_phenotype_df.empty and len(pain_phenotype_df) > 1:
            _, adjusted_pvalues, _, _ = multipletests(pain_phenotype_df['min_pain_p_value'].values, method='fdr_bh')
            pain_phenotype_df['adjusted_pain_p_value'] = adjusted_pvalues
            pain_phenotype_df['significant_pain_overrepresentation'] = pain_phenotype_df['adjusted_pain_p_value'] < 0.05
        
        return pain_phenotype_df
    
    def analyze_pain_phenotypes(self):
        """Special analysis for pain-related phenotypes."""
        pain_keywords = ['pain', 'nociception', 'analgesia', 'hyperalgesia', 'allodynia']
        
        pain_results = []
        
        for cluster_name, cluster_genes in self.clusters.items():
            cluster_genes_set = set(str(g) for g in cluster_genes)
            
            # Find pain genes in cluster
            pain_genes_in_cluster = []
            
            for gene in cluster_genes_set:
                if gene in self.gene_phenotypes:
                    phenotypes = self.gene_phenotypes[gene]
                    if any(any(keyword in phenotype for keyword in pain_keywords) for phenotype in phenotypes):
                        pain_genes_in_cluster.append(gene)
            
            if pain_genes_in_cluster:
                pain_results.append({
                    'cluster_id': cluster_name,
                    'pain_genes': pain_genes_in_cluster,
                    'pain_gene_count': len(pain_genes_in_cluster),
                    'cluster_size': len(cluster_genes),
                    'pain_percentage': (len(pain_genes_in_cluster) / len(cluster_genes)) * 100
                })
        
        if pain_results:
            pain_df = pd.DataFrame(pain_results)
            pain_df.to_csv(os.path.join(self.analysis_dir, 'pain_phenotype_analysis.csv'), index=False)
            
            # Create pain-specific visualization
            plt.figure(figsize=(10, 6))
            pain_df = pain_df.sort_values('pain_percentage', ascending=True)
            plt.barh(range(len(pain_df)), pain_df['pain_percentage'])
            plt.yticks(range(len(pain_df)), pain_df['cluster_id'])
            plt.xlabel('Percentage of Pain-Related Genes')
            plt.title('Pain Gene Distribution Across Clusters')
            plt.tight_layout()
            plots_dir = os.path.join('plots', self.ppi_source, f'phenotype_analysis_{self.cluster_type_str}_{self.safe_prefix}')
            ensure_dir(plots_dir)

            plt.savefig(os.path.join(plots_dir, 'pain_gene_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_visualizations(self):
        """Create standardized visualizations."""
        print(f"\n{'='*80}")
        print(f"CREATING VISUALIZATIONS")
        print(f"{'='*80}")

        plots_dir = os.path.join('plots', self.ppi_source, f'phenotype_analysis_{self.cluster_type_str}_{self.safe_prefix}')
        ensure_dir(plots_dir)
        print(f"Plots directory: {plots_dir}")

        if self.results.empty:
            print("✗ No results to visualize")
            return

        print(f"Creating visualizations for {len(self.results)} results...")

        # 1. Volcano plot
        print("\n1. Creating volcano plot...")
        plt.figure(figsize=(10, 8))

        # Calculate -log10(p-value) and log2(odds_ratio) with safe handling
        self.results['neg_log_pvalue'] = -np.log10(self.results['p_value'] + 1e-300)
        self.results['log2_fold_enrichment'] = self.results['mean_fold_enrichment'].apply(self.safe_log2)

        # Color by significance
        colors = ['red' if pval < 0.05 else 'gray' for pval in self.results['adjusted_p_value']]
        sig_count = sum(1 for c in colors if c == 'red')
        print(f"   Significant results (red): {sig_count}/{len(colors)}")

        plt.scatter(self.results['log2_fold_enrichment'], self.results['neg_log_pvalue'], 
                   c=colors, alpha=0.6, s=50)

        plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        plt.xlabel('log2(Fold Enrichment)')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Phenotype Enrichment Volcano Plot - {self.ppi_source}')
        plt.tight_layout()
        volcano_path = os.path.join(plots_dir, 'phenotype_volcano_plot.png')
        plt.savefig(volcano_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {volcano_path}")

        # 2. Top enrichments bar plot
        print("\n2. Creating top enrichments bar plot...")
        if not self.results.empty:
            top_results = self.results.nlargest(10, 'mean_fold_enrichment')
            print(f"   Plotting top {len(top_results)} enrichments")

            plt.figure(figsize=(12, 6))
            plt.barh(range(len(top_results)), top_results['mean_fold_enrichment'])
            plt.yticks(range(len(top_results)), top_results['cluster_id'])
            plt.xlabel('Mean Fold Enrichment')
            plt.title(f'Top Phenotype Enrichments - {self.ppi_source}')
            plt.tight_layout()
            bar_path = os.path.join(plots_dir, 'top_enrichments.png')
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: {bar_path}")

        print(f"\n{'='*80}\n")
    
    def generate_report(self):
        """Generate a simple text report."""
        print(f"\n{'='*80}")
        print(f"GENERATING REPORT")
        print(f"{'='*80}")

        report_file = os.path.join(self.analysis_dir, 'analysis_report.txt')
        print(f"Report file: {report_file}")

        with open(report_file, 'w') as f:
            f.write(f"Phenotype Analysis Report\n")
            f.write(f"========================\n\n")
            f.write(f"PPI Source: {self.ppi_source}\n")
            f.write(f"Cluster Type: {self.cluster_type}\n")
            f.write(f"Annotation: {self.safe_prefix}\n")
            f.write(f"Gene data file: {self.expression_file}\n")
            f.write(f"Total clusters: {len(self.clusters)}\n")
            f.write(f"Total genes with phenotypes: {len(self.gene_phenotypes)}\n")
            f.write(f"Total phenotypes: {len(self.phenotype_genes)}\n\n")

            if not self.results.empty:
                f.write(f"Analysis Results:\n")
                f.write(f"Total clusters analyzed: {len(self.results)}\n")
                f.write(f"Significant enrichments: {(self.results['adjusted_p_value'] < 0.05).sum()}\n\n")

                # Top enrichments
                f.write("Top 10 Enrichments:\n")
                top_results = self.results.nlargest(10, 'mean_fold_enrichment')
                for _, row in top_results.iterrows():
                    f.write(f"  {row['cluster_id']}: ")
                    f.write(f"OR={row['mean_fold_enrichment']:.2f}, p={row['p_value']:.2e}\n")

        print(f"✓ Report saved: {report_file}")
        print(f"\n{'='*80}\n")  
        
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print(f"\n{'#'*80}")
        print(f"# STARTING PHENOTYPES ANALYSIS - HIDEF CLUSTERS FROM COMPARISON")
        print(f"# PPI Source: {self.ppi_source}")
        print(f"{'#'*80}\n")

        # Load clusters
        print(f"STEP 1: Loading HiDeF clusters from comparison results...")
        if not self.load_hidef_clusters_multiplex(
            nodes_file='../11_hidef_results_faithful_cpm/heterogeneous_hidef_faithful_cpm.nodes',
            supra_graph_pkl='../11_hidef_hetero/supra_graph.pkl',
            seed_genes_file='../00_inputs/seed_genes.csv'):
            print("✗ Failed to load HiDeF clusters")
            return False

        print(f"✓ Loaded {len(self.clusters)} clusters with {len(self.seed_genes)} seed genes")
        print(f"  Cluster names: {list(self.clusters.keys())[:5]}{'...' if len(self.clusters) > 5 else ''}")

        # Load expression data
        print(f"\nSTEP 2: Loading phenotype data...")
        if not self.load_phenotype_data():
            return False

        # Run enrichment analysis
        print(f"\nSTEP 3: Running enrichment analysis...")
        results_df, gene_results_df = self.analyze_enrichment()

        # Run pain-specific analysis
        print(f"\nSTEP 4: Analyzing pain phenotypes...")
        self.analyze_pain_phenotypes()

        # Save results
        if not results_df.empty:
            print(f"\nSTEP 5: Saving results...")
            save_results(results_df, gene_results_df, self.analysis_dir, 'phenotype')
            print(f"✓ Results saved to: {self.analysis_dir}")

            # Create visualizations
            print(f"\nSTEP 6: Creating visualizations...")
            self.create_visualizations()

            # Generate report
            print(f"\nSTEP 7: Generating report...")
            self.generate_report()

            print(f"\n{'#'*80}")
            print(f"# ANALYSIS COMPLETE!")
            print(f"# Results saved to: {self.analysis_dir}")
            print(f"{'#'*80}\n")

            return True

def main():
    parser = argparse.ArgumentParser(description='Analyze pain phenotypes across HiDeF clusters from comparison results')
    parser.add_argument('--ppi_source', type=str, required=True, help='PPI source name')
    parser.add_argument('--expression_data', type=str, required=True, help='Phenotype data CSV file')
    parser.add_argument('--input_dir', type=str, default='./input', help='Input directory with HiDeF comparison results')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--persistence_threshold', type=int, default=5, 
                    help='Minimum persistence threshold for HiDeF clusters (default: 5)')
    
    args = parser.parse_args()
    
    analyzer = PhenotypesAnalyzer(
        ppi_source=args.ppi_source,
        expression_file=args.expression_data,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cluster_type='hidef_from_comparison',
        persistence_threshold=args.persistence_threshold
    )
    
    success = analyzer.run_analysis()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())