#!/usr/bin/env python3
"""
analyze_neuropathic_pain.py - Analyze neuropathic pain gene expression across HiDeF clusters

Usage:
    python analyze_neuropathic_pain.py --ppi_source STRING --expression_data pain_expr.csv --input_dir ./input --output_dir ./output
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

class NeuropathicPainAnalyzer:
    def __init__(self, ppi_source, expression_file, input_dir, output_dir, cluster_type='seed_lcc_hidef', persistence_threshold=5):
        print(f"\n{'='*80}")
        print(f"INITIALIZING NeuropathicPainAnalyzer")
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
        self.analysis_dir = os.path.join(output_dir, ppi_source, f'neuropathic_pain_analysis_{self.cluster_type_str}')

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
        self.metadata = None
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

    
    def load_expression_data(self):
        """Load pain expression data with metadata."""
        print(f"\n{'='*80}")
        print(f"LOADING NEUROPATHIC PAIN EXPRESSION DATA")
        print(f"{'='*80}")
        print(f"Expression file: {self.expression_file}")

        try:
            df = pd.read_csv(self.expression_file)

            print(f"  Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")

            # Check for metadata in first rows
            if 'pain state' in df.iloc[0].values or 'pain_state' in df.iloc[0].values:
                print(f"  Detected metadata rows in file")
                # Extract metadata
                meta_rows = []
                data_start_idx = 0

                for i in range(min(10, df.shape[0])):
                    row = df.iloc[i]
                    if any(meta_key in str(row.iloc[0]).lower() for meta_key in ['pain state', 'age', 'sex', 'gender']):
                        meta_rows.append(row)
                        data_start_idx = i + 1
                    else:
                        if data_start_idx > 0:
                            break

                print(f"  Found {len(meta_rows)} metadata rows")
                print(f"  Expression data starts at row {data_start_idx}")

                # Extract expression data
                expr_df = df.iloc[data_start_idx:].copy()
                expr_df = expr_df.set_index(expr_df.columns[0])

                # Create metadata DataFrame
                meta_dict = {}
                for row in meta_rows:
                    meta_key = row.iloc[0]
                    for j, col in enumerate(df.columns[1:]):
                        if col not in meta_dict:
                            meta_dict[col] = {}
                        meta_dict[col][meta_key] = row.iloc[j+1]

                self.metadata = pd.DataFrame.from_dict(meta_dict, orient='index')
                self.expression_data = expr_df.apply(pd.to_numeric, errors='coerce')

            else:
                print(f"  Standard format detected (no metadata rows)")
                # Standard format
                expr_df = df.set_index(df.columns[0])
                self.expression_data = expr_df.apply(pd.to_numeric, errors='coerce')

                # Try to extract metadata from column names or separate file
                self.metadata = pd.DataFrame(index=expr_df.columns)

            print(f"\n✓ Loaded expression data: {self.expression_data.shape[0]} genes, {self.expression_data.shape[1]} samples")

            if self.metadata is not None and 'pain state' in self.metadata.columns:
                pain_counts = self.metadata['pain state'].value_counts()
                print(f"\nPain state distribution:")
                for state, count in pain_counts.items():
                    print(f"  {state}: {count} samples")

                # Also check for gender information
                if any(col in self.metadata.columns for col in ['sex', 'gender', 'Sex', 'Gender']):
                    gender_col = next((col for col in ['sex', 'gender', 'Sex', 'Gender'] if col in self.metadata.columns), None)
                    gender_counts = self.metadata[gender_col].value_counts()
                    print(f"\nGender distribution:")
                    for gender, count in gender_counts.items():
                        print(f"  {gender}: {count} samples")

                    # Standardize gender column name
                    if gender_col != 'gender':
                        self.metadata['gender'] = self.metadata[gender_col]

            print(f"\n{'='*80}\n")
            return True

        except Exception as e:
            print(f"\n✗ Error loading expression data: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            return False
    
    def analyze_differential_expression_overall(self):
        """Analyze differential expression between pain states for each cluster (overall)."""
        print(f"\n{'='*80}")
        print(f"ANALYZING OVERALL DIFFERENTIAL EXPRESSION")
        print(f"{'='*80}")

        if self.metadata is None or 'pain state' not in self.metadata.columns:
            print("✗ No pain state information available")
            print(f"{'='*80}\n")
            return pd.DataFrame(), pd.DataFrame()

        # Get pain and no-pain samples
        pain_samples = self.metadata[self.metadata['pain state'] == 'P'].index
        no_pain_samples = self.metadata[self.metadata['pain state'] == 'N'].index

        print(f"Sample groups:")
        print(f"  Pain samples: {len(pain_samples)}")
        print(f"  No-pain samples: {len(no_pain_samples)}")

        if len(pain_samples) == 0 or len(no_pain_samples) == 0:
            print("\n✗ Need both pain and no-pain samples")
            print(f"{'='*80}\n")
            return pd.DataFrame(), pd.DataFrame()

        print(f"\n✓ Proceeding with differential analysis")
        print(f"{'='*80}\n")

        return self._perform_differential_analysis(pain_samples, no_pain_samples, 'overall')
    
    def analyze_differential_expression_by_gender(self):
        """Analyze differential expression separately for males and females."""
        print(f"\n{'='*80}")
        print(f"ANALYZING GENDER-SPECIFIC DIFFERENTIAL EXPRESSION")
        print(f"{'='*80}")

        if self.metadata is None or 'pain state' not in self.metadata.columns or 'gender' not in self.metadata.columns:
            print("✗ No pain state or gender information available for gender-specific analysis")
            print(f"{'='*80}\n")
            return {}, {}

        results_by_gender = {}
        gene_results_by_gender = {}

        # Analyze males
        print(f"\nChecking male samples...")
        male_samples = self.metadata[self.metadata['gender'].isin(['M', 'Male', 'male', 'm'])].index
        print(f"  Total male samples: {len(male_samples)}")

        if len(male_samples) > 0:
            male_pain = self.metadata.loc[male_samples][self.metadata.loc[male_samples]['pain state'] == 'P'].index
            male_no_pain = self.metadata.loc[male_samples][self.metadata.loc[male_samples]['pain state'] == 'N'].index

            print(f"  Male pain samples: {len(male_pain)}")
            print(f"  Male no-pain samples: {len(male_no_pain)}")

            if len(male_pain) > 0 and len(male_no_pain) > 0:
                print(f"  ✓ Analyzing males: {len(male_pain)} pain, {len(male_no_pain)} no pain")
                male_results, male_gene_results = self._perform_differential_analysis(
                    male_pain, male_no_pain, 'male'
                )
                results_by_gender['male'] = male_results
                gene_results_by_gender['male'] = male_gene_results
            else:
                print(f"  ✗ Insufficient male samples for analysis")

        # Analyze females
        print(f"\nChecking female samples...")
        female_samples = self.metadata[self.metadata['gender'].isin(['F', 'Female', 'female', 'f'])].index
        print(f"  Total female samples: {len(female_samples)}")

        if len(female_samples) > 0:
            female_pain = self.metadata.loc[female_samples][self.metadata.loc[female_samples]['pain state'] == 'P'].index
            female_no_pain = self.metadata.loc[female_samples][self.metadata.loc[female_samples]['pain state'] == 'N'].index

            print(f"  Female pain samples: {len(female_pain)}")
            print(f"  Female no-pain samples: {len(female_no_pain)}")

            if len(female_pain) > 0 and len(female_no_pain) > 0:
                print(f"  ✓ Analyzing females: {len(female_pain)} pain, {len(female_no_pain)} no pain")
                female_results, female_gene_results = self._perform_differential_analysis(
                    female_pain, female_no_pain, 'female'
                )
                results_by_gender['female'] = female_results
                gene_results_by_gender['female'] = female_gene_results
            else:
                print(f"  ✗ Insufficient female samples for analysis")

        print(f"\n{'='*80}")
        print(f"GENDER-SPECIFIC ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Analyses completed: {list(results_by_gender.keys())}")
        print(f"{'='*80}\n")

        return results_by_gender, gene_results_by_gender
    
    def _perform_differential_analysis(self, pain_samples, no_pain_samples, analysis_type):
        """Perform differential expression analysis using hypergeometric test."""
        print(f"\n{'='*80}")
        print(f"STARTING DIFFERENTIAL ANALYSIS ({analysis_type.upper()})")
        print(f"{'='*80}")
        print(f"Pain samples: {len(pain_samples)}")
        print(f"No-pain samples: {len(no_pain_samples)}")

        summarized_results = []
        gene_level_results = []
        cluster_info = {}

        pain_expr_means = self.expression_data[pain_samples].mean(axis=1)
        no_pain_expr_means = self.expression_data[no_pain_samples].mean(axis=1)

        pain_threshold = pain_expr_means.quantile(0.90)
        no_pain_threshold = no_pain_expr_means.quantile(0.90)

        pain_highly_expressed = set(pain_expr_means[pain_expr_means > pain_threshold].index)
        no_pain_highly_expressed = set(no_pain_expr_means[no_pain_expr_means > no_pain_threshold].index)

        print(f"\nExpression thresholds:")
        print(f"  Pain threshold (90th percentile): {pain_threshold:.2f}")
        print(f"  No-pain threshold (90th percentile): {no_pain_threshold:.2f}")
        print(f"  Highly expressed in pain: {len(pain_highly_expressed)} genes")
        print(f"  Highly expressed in no-pain: {len(no_pain_highly_expressed)} genes")

        total_genes = len(self.expression_data.index)
        print(f"  Total genes in dataset: {total_genes}")

        for cluster_idx, (cluster_name, cluster_genes) in enumerate(self.clusters.items()):
            print(f"\n{'-'*80}")
            print(f"Processing cluster {cluster_idx + 1}/{len(self.clusters)}: {cluster_name}")
            print(f"  Cluster size: {len(cluster_genes)} genes")

            cluster_genes_set = set(str(g) for g in cluster_genes)
            expr_genes = set(self.expression_data.index)
            overlap_genes = cluster_genes_set.intersection(expr_genes)

            if len(overlap_genes) < 2:
                print(f"  ✗ Insufficient overlap ({len(overlap_genes)} genes)")
                continue

            print(f"  Genes in expression data: {len(overlap_genes)}")

            # Test pain overexpression
            pain_overlap = len(overlap_genes.intersection(pain_highly_expressed))

            if pain_overlap == 0:
                pain_fold_enrichment = 1.0
                pain_p_value = 1.0
            else:
                pain_fold_enrichment = (pain_overlap / len(overlap_genes)) / (len(pain_highly_expressed) / total_genes)
                try:
                    from scipy.stats import hypergeom
                    pain_p_value = hypergeom.sf(
                        pain_overlap - 1,
                        total_genes,
                        len(pain_highly_expressed),
                        len(overlap_genes)
                    )
                except:
                    pain_p_value = 1.0

            print(f"  Pain enrichment:")
            print(f"    Overlap: {pain_overlap} genes")
            print(f"    Fold enrichment: {pain_fold_enrichment:.2f}")
            print(f"    P-value: {pain_p_value:.2e}")

            # Test no-pain overexpression
            no_pain_overlap = len(overlap_genes.intersection(no_pain_highly_expressed))

            if no_pain_overlap == 0:
                no_pain_fold_enrichment = 1.0
                no_pain_p_value = 1.0
            else:
                no_pain_fold_enrichment = (no_pain_overlap / len(overlap_genes)) / (len(no_pain_highly_expressed) / total_genes)
                try:
                    from scipy.stats import hypergeom
                    no_pain_p_value = hypergeom.sf(
                        no_pain_overlap - 1,
                        total_genes,
                        len(no_pain_highly_expressed),
                        len(overlap_genes)
                    )
                except:
                    no_pain_p_value = 1.0

            print(f"  No-pain enrichment:")
            print(f"    Overlap: {no_pain_overlap} genes")
            print(f"    Fold enrichment: {no_pain_fold_enrichment:.2f}")
            print(f"    P-value: {no_pain_p_value:.2e}")

            # Use biggest effect size (fold enrichment), but require significance (p < 0.05)
            pain_significant = pain_p_value < 0.05
            no_pain_significant = no_pain_p_value < 0.05

            if pain_significant and no_pain_significant:
                # Both significant: choose by fold enrichment
                if pain_fold_enrichment > no_pain_fold_enrichment:
                    best_fold_enrichment = pain_fold_enrichment
                    best_p_value = pain_p_value
                    top_expression = 'pain_upregulated'
                    percentage = (pain_overlap / len(overlap_genes)) * 100 if len(overlap_genes) > 0 else 0
                    print(f"  → Best result: Pain upregulated (FE={pain_fold_enrichment:.2f})")
                else:
                    best_fold_enrichment = no_pain_fold_enrichment
                    best_p_value = no_pain_p_value
                    top_expression = 'no_pain_upregulated'
                    percentage = (no_pain_overlap / len(overlap_genes)) * 100 if len(overlap_genes) > 0 else 0
                    print(f"  → Best result: No-pain upregulated (FE={no_pain_fold_enrichment:.2f})")
            elif pain_significant:
                # Only pain significant
                best_fold_enrichment = pain_fold_enrichment
                best_p_value = pain_p_value
                top_expression = 'pain_upregulated'
                percentage = (pain_overlap / len(overlap_genes)) * 100 if len(overlap_genes) > 0 else 0
                print(f"  → Best result: Pain upregulated (FE={pain_fold_enrichment:.2f})")
            elif no_pain_significant:
                # Only no-pain significant
                best_fold_enrichment = no_pain_fold_enrichment
                best_p_value = no_pain_p_value
                top_expression = 'no_pain_upregulated'
                percentage = (no_pain_overlap / len(overlap_genes)) * 100 if len(overlap_genes) > 0 else 0
                print(f"  → Best result: No-pain upregulated (FE={no_pain_fold_enrichment:.2f})")
            else:
                # Neither significant: skip this cluster
                print(f"  ✗ No significant differential expression (pain p={pain_p_value:.2e}, no-pain p={no_pain_p_value:.2e})")
                continue

            cluster_seed_genes = self.cluster_seed_mapping[cluster_name]
            print(f"  Seed genes: {len(cluster_seed_genes)}")

            cluster_info[cluster_name] = {
                'fold_enrichment': best_fold_enrichment,
                'p_value': best_p_value,
                'top_expression': top_expression
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
            result['analysis_type'] = analysis_type
            result['top_expression'] = top_expression
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
                    'cluster_top_expression': cluster_info[cluster_name]['top_expression']
                })

            print(f"  ✓ Added enrichment results for this cluster")

        print(f"\n{'='*80}")
        print(f"DIFFERENTIAL ANALYSIS COMPLETE ({analysis_type.upper()})")
        print(f"{'='*80}")
        print(f"Total cluster enrichments: {len(summarized_results)}")
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
            print(f"  {sig_count} genes in clusters with significant differential expression")

        print(f"\n{'='*80}\n")
        return results_df, gene_results_df
    
    def analyze_cluster_overexpression_in_pain(self):
        """Specialized analysis: Are clusters overexpressed in pain/neuropathic pain patients?"""
        print(f"\n{'='*80}")
        print(f"ANALYZING CLUSTER OVEREXPRESSION IN PAIN")
        print(f"{'='*80}")

        if self.metadata is None or 'pain state' not in self.metadata.columns:
            print("✗ No pain state information available")
            print(f"{'='*80}\n")
            return pd.DataFrame()

        pain_samples = self.metadata[self.metadata['pain state'] == 'P'].index
        no_pain_samples = self.metadata[self.metadata['pain state'] == 'N'].index

        print(f"Sample groups:")
        print(f"  Pain samples: {len(pain_samples)}")
        print(f"  No-pain samples: {len(no_pain_samples)}")

        if len(pain_samples) == 0 or len(no_pain_samples) == 0:
            print("\n✗ Need both pain and no-pain samples")
            print(f"{'='*80}\n")
            return pd.DataFrame()

        pain_overexpression_results = []

        print(f"\nProcessing {len(self.clusters)} clusters...")

        for cluster_idx, (cluster_name, cluster_genes) in enumerate(self.clusters.items()):
            print(f"\n{'-'*80}")
            print(f"Cluster {cluster_idx + 1}/{len(self.clusters)}: {cluster_name}")

            cluster_genes_set = set(str(g) for g in cluster_genes)
            expr_genes = set(self.expression_data.index)
            common_genes = list(cluster_genes_set.intersection(expr_genes))

            if len(common_genes) < 2:
                print(f"  ✗ Insufficient genes ({len(common_genes)})")
                continue

            print(f"  Genes in expression data: {len(common_genes)}")

            # Get cluster expression
            cluster_expr = self.expression_data.loc[common_genes]

            # Calculate mean cluster expression for each sample
            cluster_mean_pain = cluster_expr[pain_samples].mean(axis=0)
            cluster_mean_no_pain = cluster_expr[no_pain_samples].mean(axis=0)

            # Use Fisher's exact test approach instead of Mann-Whitney
            # Define high expression threshold
            all_cluster_means = pd.concat([cluster_mean_pain, cluster_mean_no_pain])
            high_expr_threshold = all_cluster_means.quantile(0.75)

            pain_high_expr = (cluster_mean_pain > high_expr_threshold).sum()
            pain_low_expr = len(cluster_mean_pain) - pain_high_expr
            no_pain_high_expr = (cluster_mean_no_pain > high_expr_threshold).sum()
            no_pain_low_expr = len(cluster_mean_no_pain) - no_pain_high_expr

            print(f"  High expression threshold: {high_expr_threshold:.2f}")
            print(f"  Pain samples with high expression: {pain_high_expr}/{len(cluster_mean_pain)}")
            print(f"  No-pain samples with high expression: {no_pain_high_expr}/{len(cluster_mean_no_pain)}")

            # Fisher's exact test for overexpression in pain
            try:
                odds_ratio, p_value_overexp = fisher_exact([[pain_high_expr, pain_low_expr], 
                                                          [no_pain_high_expr, no_pain_low_expr]], 
                                                         alternative='greater')
                odds_ratio = min(odds_ratio, 1e6)
                log2_fold_change = self.safe_log2(odds_ratio)
                print(f"  Odds ratio: {odds_ratio:.2f}")
                print(f"  P-value: {p_value_overexp:.2e}")
            except:
                odds_ratio, p_value_overexp = 1.0, 1.0
                log2_fold_change = 0.0
                print(f"  ✗ Fisher's exact test failed")

            # Effect size calculations
            pain_mean = cluster_mean_pain.mean()
            no_pain_mean = cluster_mean_no_pain.mean()
            fold_change = self.safe_fold_change(pain_mean, no_pain_mean)

            print(f"  Mean expression (pain): {pain_mean:.2f}")
            print(f"  Mean expression (no-pain): {no_pain_mean:.2f}")
            print(f"  Fold change: {fold_change:.2f}")

            # Percentage of genes overexpressed in pain
            gene_overexp_count = 0
            for gene in common_genes:
                gene_pain_mean = cluster_expr.loc[gene, pain_samples].mean()
                gene_no_pain_mean = cluster_expr.loc[gene, no_pain_samples].mean()
                if gene_pain_mean > gene_no_pain_mean:
                    gene_overexp_count += 1

            overexp_percentage = (gene_overexp_count / len(common_genes)) * 100
            print(f"  Genes overexpressed in pain: {gene_overexp_count}/{len(common_genes)} ({overexp_percentage:.1f}%)")

            # Classify overexpression status
            is_overexpressed = p_value_overexp < 0.05 and log2_fold_change > 0.5
            overexpression_status = 'Overexpressed in Pain' if is_overexpressed else 'Not Overexpressed'

            print(f"  Status: {overexpression_status}")

            pain_overexpression_results.append({
                'cluster_id': cluster_name,
                'cluster_type': self.cluster_type,
                'genes_in_cluster': len(common_genes),
                'mean_expression_pain': pain_mean,
                'mean_expression_no_pain': no_pain_mean,
                'odds_ratio': odds_ratio,
                'p_value_overexpression': p_value_overexp,
                'percentage_genes_overexpressed': overexp_percentage,
                'overexpression_status': overexpression_status,
                'effect_size': 'Large' if odds_ratio > 2 else 'Medium' if odds_ratio > 1.5 else 'Small'
            })

            print(f"  ✓ Added overexpression results")

        pain_overexp_df = pd.DataFrame(pain_overexpression_results)

        print(f"\n{'='*80}")
        print(f"CLUSTER OVEREXPRESSION ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total clusters analyzed: {len(pain_overexp_df)}")

        # Apply multiple testing correction
        if not pain_overexp_df.empty and len(pain_overexp_df) > 1:
            print(f"\nApplying multiple testing correction...")
            _, adjusted_pvalues, _, _ = multipletests(pain_overexp_df['p_value_overexpression'].values, method='fdr_bh')
            pain_overexp_df['adjusted_p_value_overexpression'] = adjusted_pvalues
            pain_overexp_df['significant_overexpression'] = pain_overexp_df['adjusted_p_value_overexpression'] < 0.05
            sig_count = pain_overexp_df['significant_overexpression'].sum()
            print(f"  Significant overexpressions (FDR < 0.05): {sig_count}/{len(pain_overexp_df)}")

        print(f"\n{'='*80}\n")
        return pain_overexp_df
    
    def create_visualizations(self):
        """Create standardized visualizations."""
        plots_dir = os.path.join('plots', self.ppi_source, f'neuropathic_pain_analysis_{self.cluster_type_str}')
        ensure_dir(plots_dir)

        if self.results.empty:
            print("No results to visualize")
            return

        # 1. Volcano plot
        plt.figure(figsize=(10, 8))

        # Calculate -log10(p-value) and log2(odds_ratio) with safe handling
        self.results['neg_log_pvalue'] = -np.log10(self.results['p_value'] + 1e-300)
        self.results['log2_fold_enrichment'] = self.results['mean_fold_enrichment'].apply(self.safe_log2)

        # Color by significance
        colors = ['red' if pval < 0.05 else 'gray' for pval in self.results['adjusted_p_value']]

        plt.scatter(self.results['log2_fold_enrichment'], self.results['neg_log_pvalue'], 
                   c=colors, alpha=0.6, s=100)

        plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p=0.05')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='OR=1')

        plt.xlabel('log2(Fold enrichment) Pain vs. No Pain')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Neuropathic Pain Enrichment Volcano Plot - {self.ppi_source}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'pain_volcano_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Top enrichments bar plot
        if not self.results.empty:
            # Sort by Fold Enrichment and take top 10
            top_results = self.results.nlargest(10, 'mean_fold_enrichment')

            plt.figure(figsize=(12, 6))

            # Color bars by direction (pain upregulated vs no-pain upregulated)
            colors = ['#d62728' if expr == 'pain_upregulated' else '#1f77b4' 
                     for expr in top_results.get('top_expression', ['pain_upregulated'] * len(top_results))]

            plt.barh(range(len(top_results)), top_results['mean_fold_enrichment'], color=colors)
            plt.yticks(range(len(top_results)), top_results['cluster_id'])
            plt.xlabel('Mean Fold Enrichment')
            plt.title(f'Top Pain-Associated Clusters - {self.ppi_source}')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d62728', label='Pain Upregulated'),
                Patch(facecolor='#1f77b4', label='No Pain Upregulated')
            ]
            plt.legend(handles=legend_elements, loc='lower right')

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'top_pain_enrichments.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Fold Enrichment distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.results['mean_fold_enrichment'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=1, color='red', linestyle='--', label='OR=1 (no effect)')
        plt.axvline(x=1.5, color='orange', linestyle='--', label='OR=1.5 (threshold)')
        plt.xlabel('Mean Fold Enrichment')
        plt.ylabel('Number of Clusters')
        plt.title(f'Distribution of Fold Enrichments - {self.ppi_source}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'odds_ratio_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. P-value distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.results['p_value'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        plt.xlabel('P-value')
        plt.ylabel('Frequency')
        plt.title(f'P-value Distribution - {self.ppi_source}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'pvalue_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Significant clusters bar plot (if any)
        sig_results = self.results[self.results['adjusted_p_value'] < 0.05].copy()
        if not sig_results.empty:
            sig_results = sig_results.sort_values('mean_fold_enrichment', ascending=True)

            plt.figure(figsize=(10, max(6, len(sig_results) * 0.3)))

            # Color by direction
            colors = ['#d62728' if expr == 'pain_upregulated' else '#1f77b4' 
                     for expr in sig_results.get('top_expression', ['pain_upregulated'] * len(sig_results))]

            plt.barh(range(len(sig_results)), sig_results['mean_fold_enrichment'], color=colors)
            plt.yticks(range(len(sig_results)), sig_results['cluster_id'])
            plt.xlabel('Mean Fold Enrichment')
            plt.title(f'Significant Pain-Associated Clusters (FDR < 0.05) - {self.ppi_source}')
            plt.axvline(x=1, color='black', linestyle='-', alpha=0.3)

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d62728', label='Pain Upregulated'),
                Patch(facecolor='#1f77b4', label='No Pain Upregulated')
            ]
            plt.legend(handles=legend_elements, loc='lower right')

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'significant_pain_clusters.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self):
        """Generate a simple text report."""
        report_file = os.path.join(self.analysis_dir, 'analysis_report.txt')
        
        with open(report_file, 'w') as f:
            f.write(f"Neuropathic Pain Expression Analysis Report\n")
            f.write(f"==========================================\n\n")
            f.write(f"PPI Source: {self.ppi_source}\n")
            f.write(f"Cluster Type: {self.cluster_type}\n")
            f.write(f"Persistence Threshold: {self.persistence_threshold}\n")
            f.write(f"Expression file: {self.expression_file}\n")
            f.write(f"Total clusters: {len(self.clusters)}\n")

            if self.expression_data is not None:
                f.write(f"Total genes: {self.expression_data.shape[0]}\n")
                f.write(f"Total samples: {self.expression_data.shape[1]}\n")
            
            if self.metadata is not None and 'pain state' in self.metadata.columns:
                pain_counts = self.metadata['pain state'].value_counts()
                f.write(f"\nSample distribution:\n")
                for state, count in pain_counts.items():
                    f.write(f"  {state}: {count} samples\n")
                
                if 'gender' in self.metadata.columns:
                    gender_counts = self.metadata['gender'].value_counts()
                    f.write(f"\nGender distribution:\n")
                    for gender, count in gender_counts.items():
                        f.write(f"  {gender}: {count} samples\n")
            
            if not self.results.empty:
                f.write(f"\nDifferential Expression Results:\n")
                f.write(f"Total clusters analyzed: {len(self.results)}\n")
                f.write(f"Significant changes: {(self.results['adjusted_p_value'] < 0.05).sum()}\n")
                
                sig_up = len(self.results[(self.results['adjusted_p_value'] < 0.05) & (self.results['mean_fold_enrichment'] > 0)])
                sig_down = len(self.results[(self.results['adjusted_p_value'] < 0.05) & (self.results['mean_fold_enrichment'] < 0)])
                f.write(f"  Upregulated: {sig_up}\n")
                f.write(f"  Downregulated: {sig_down}\n")
                
                # Top changes
                f.write("\nTop Expression Changes:\n")
                top_results = self.results.nlargest(5, 'mean_fold_enrichment', keep='all')
                for _, row in top_results.iterrows():
                    f.write(f"  {row['cluster_id']}: OR={row['mean_fold_enrichment']:.2f}, ")
                    f.write(f"p={row['p_value']:.2e}\n")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print(f"\n{'#'*80}")
        print(f"# STARTING NEUROPATHIC PAIN ANALYSIS")
        print(f"# PPI Source: {self.ppi_source}")
        print(f"# Cluster Type: {self.cluster_type}")
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
        print(f"\nSTEP 2: Loading expression data...")
        if not self.load_expression_data():
            return False

        # Run overall differential expression analysis
        print(f"\nSTEP 3: Running overall differential expression analysis...")
        results_df, gene_results_df = self.analyze_differential_expression_overall()

        # Store overall results
        self.results = results_df
        self.gene_results = gene_results_df

        # Run gender-specific differential expression analysis
        print(f"\nSTEP 4: Running gender-specific differential expression analysis...")
        gender_results, gender_gene_results = self.analyze_differential_expression_by_gender()

        # Run cluster overexpression analysis
        print(f"\nSTEP 5: Analyzing cluster overexpression in pain...")
        pain_overexp_df = self.analyze_cluster_overexpression_in_pain()

        # Save results
        if not results_df.empty:
            print(f"\nSTEP 6: Saving results...")
            save_results(results_df, gene_results_df, self.analysis_dir, 'neuropathic_pain')
            print(f"✓ Main results saved")

            # Save gender-specific results
            for gender, gender_df in gender_results.items():
                if not gender_df.empty:
                    gender_df.to_csv(os.path.join(self.analysis_dir, f'neuropathic_pain_differential_expression_{gender}.csv'), index=False)
                    print(f"✓ Saved {gender} differential expression results")

            for gender, gender_gene_df in gender_gene_results.items():
                if not gender_gene_df.empty:
                    gender_gene_df.to_csv(os.path.join(self.analysis_dir, f'gene_level_results_{gender}.csv'), index=False)
                    print(f"✓ Saved {gender} gene-level results")

            # Save pain overexpression results
            if not pain_overexp_df.empty:
                pain_overexp_df.to_csv(os.path.join(self.analysis_dir, 'cluster_overexpression_in_pain.csv'), index=False)
                print(f"✓ Saved cluster overexpression analysis")

                # Save significant overexpressions
                sig_overexp = pain_overexp_df[pain_overexp_df.get('significant_overexpression', False)]
                if not sig_overexp.empty:
                    sig_overexp.to_csv(os.path.join(self.analysis_dir, 'significant_pain_overexpressions.csv'), index=False)
                    print(f"✓ Saved {len(sig_overexp)} significant pain overexpressions")

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
    parser = argparse.ArgumentParser(description='Analyze neuropathic pain expression across HiDeF clusters')
    parser.add_argument('--ppi_source', type=str, required=True, help='PPI source name')
    parser.add_argument('--expression_data', type=str, required=True, help='Expression data CSV file')
    parser.add_argument('--input_dir', type=str, default='./input', help='Input directory with HiDeF comparison results')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--persistence_threshold', type=int, default=5, 
                    help='Minimum persistence threshold for HiDeF clusters (default: 5)')
    
    args = parser.parse_args()
    
    analyzer = NeuropathicPainAnalyzer(
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