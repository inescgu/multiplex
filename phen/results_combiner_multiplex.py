#!/usr/bin/env python3
"""
results_combiner_multiplex.py

Same ResultsCombiner as your original script, with one change: cluster_type
is now a constructor argument instead of hardcoded to
'hidef_from_comparison'. Nothing else needed changing -- by the time this
class runs, your patched analyzers have already written flat, cell-type-
stripped gene lists into the same *_gene_level_results.csv shape the
monoplex pipeline used, so every extract_* method works unmodified.

Pass a DIFFERENT ppi_source (and/or cluster_type) than your monoplex run
used, or this will glob into the same output folders and mix the two
pipelines' results together.
"""
import argparse
import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict

class ResultsCombiner:
    def __init__(self, output_dir, ppi_source, cluster_type='hidef_from_comparison'):
        self.output_dir = output_dir
        self.ppi_source = ppi_source
        self.cluster_type = cluster_type
        self.annotation_prefix = None

        # Data containers
        self.gene_data = defaultdict(lambda: defaultdict(dict))

        # Analysis types to look for
        self.analysis_types = {
            'proteomics': 'proteomics_analysis',
            'neuropathic_pain': 'neuropathic_pain_analysis',
            'orphanet': 'orphanet_enrichr_analysis',
            'omim': 'omim_ps_analysis',
            'phenotypes': 'phenotypes_analysis',
            'spatial': 'spatial_analysis'
        }

    def find_analysis_directories(self):
        """Find all analysis directories for the given PPI source and cluster type."""
        base_path = os.path.join(self.output_dir, self.ppi_source)

        if not os.path.exists(base_path):
            print(f"Base path not found: {base_path}")
            return {}

        found_dirs = {}
        print(base_path)
        for analysis_name, dir_pattern in self.analysis_types.items():
            pattern = f"{dir_pattern}_{self.cluster_type}*"
            matching_dirs = glob.glob(os.path.join(base_path, pattern))

            if matching_dirs:
                if self.annotation_prefix:
                    filtered = [d for d in matching_dirs if self.annotation_prefix in os.path.basename(d)]
                    if filtered:
                        found_dirs[analysis_name] = filtered[0]
                    else:
                        found_dirs[analysis_name] = matching_dirs[0]
                else:
                    found_dirs[analysis_name] = matching_dirs[0]

                print(f"Found {analysis_name}: {found_dirs[analysis_name]}")

        return found_dirs

    def load_gene_level_results(self, analysis_dir, analysis_type):
        """Load gene-level results from an analysis directory."""
        file_name_map = {
            'proteomics': 'proteomics_gene_level_results.csv',
            'neuropathic_pain': 'neuropathic_pain_gene_level_results.csv',
            'orphanet': 'orphanet_enrichr_gene_level_results.csv',
            'omim': 'omim_ps_gene_level_results.csv',
            'phenotypes': 'phenotype_gene_level_results.csv',
            'spatial': 'spatial_transcriptomics_hidef_comparison_gene_level_results.csv'
        }

        file_name = file_name_map.get(analysis_type, f'{analysis_type}_gene_level_results.csv')
        gene_level_file = os.path.join(analysis_dir, file_name)

        if not os.path.exists(gene_level_file):
            print(f"Gene-level results not found: {gene_level_file}")
            return None

        try:
            df = pd.read_csv(gene_level_file)
            print(f"Loaded {len(df)} gene-level results from {analysis_type}")
            return df
        except Exception as e:
            print(f"Error loading {gene_level_file}: {e}")
            return None

    def extract_proteomics_data(self, df):
        for _, row in df.iterrows():
            gene = str(row['gene'])
            cluster = str(row['cluster_id'])

            higher_expr_gender_cluster = row.get('higher_expression_gender_cluster', None)
            enrichment = str(higher_expr_gender_cluster) if pd.notna(higher_expr_gender_cluster) else ''

            if enrichment:
                self.gene_data[gene]['proteomics']['enrichments'] = self.gene_data[gene]['proteomics'].get('enrichments', [])
                self.gene_data[gene]['proteomics']['enrichments'].append(enrichment)

            p_val_cluster = row.get('cluster_p_value', np.nan)
            self.gene_data[gene]['proteomics']['p_values'] = self.gene_data[gene]['proteomics'].get('p_values', [])
            self.gene_data[gene]['proteomics']['p_values'].append(p_val_cluster if pd.notna(p_val_cluster) else np.nan)

            fold_enr_cluster = row.get('cluster_fold_enrichment', np.nan)
            self.gene_data[gene]['proteomics']['fold_enrichments'] = self.gene_data[gene]['proteomics'].get('fold_enrichments', [])
            self.gene_data[gene]['proteomics']['fold_enrichments'].append(fold_enr_cluster if pd.notna(fold_enr_cluster) else np.nan)

            self.gene_data[gene]['proteomics']['is_seed'] = row.get('is_seed_gene', False)
            self.gene_data[gene]['proteomics']['clusters'] = self.gene_data[gene]['proteomics'].get('clusters', set())
            self.gene_data[gene]['proteomics']['clusters'].add(cluster)

    def extract_pain_data(self, df):
        for _, row in df.iterrows():
            gene = str(row['gene'])
            cluster = str(row['cluster_id'])

            top_expr_cluster = row.get('cluster_top_expression', None)
            enrichment = str(top_expr_cluster) if pd.notna(top_expr_cluster) else ''

            if enrichment:
                self.gene_data[gene]['neuropathic_pain']['enrichments'] = self.gene_data[gene]['neuropathic_pain'].get('enrichments', [])
                self.gene_data[gene]['neuropathic_pain']['enrichments'].append(enrichment)

            p_val_cluster = row.get('cluster_p_value', np.nan)
            self.gene_data[gene]['neuropathic_pain']['p_values'] = self.gene_data[gene]['neuropathic_pain'].get('p_values', [])
            self.gene_data[gene]['neuropathic_pain']['p_values'].append(p_val_cluster if pd.notna(p_val_cluster) else np.nan)

            fold_enr_cluster = row.get('cluster_fold_enrichment', np.nan)
            self.gene_data[gene]['neuropathic_pain']['fold_enrichments'] = self.gene_data[gene]['neuropathic_pain'].get('fold_enrichments', [])
            self.gene_data[gene]['neuropathic_pain']['fold_enrichments'].append(fold_enr_cluster if pd.notna(fold_enr_cluster) else np.nan)

            self.gene_data[gene]['neuropathic_pain']['is_seed'] = row.get('is_seed_gene', False)
            self.gene_data[gene]['neuropathic_pain']['clusters'] = self.gene_data[gene]['neuropathic_pain'].get('clusters', set())
            self.gene_data[gene]['neuropathic_pain']['clusters'].add(cluster)

    def extract_orphanet_data(self, df):
        for _, row in df.iterrows():
            gene = str(row['gene'])
            cluster = str(row['cluster_id'])

            top_disease_cluster = row.get('cluster_top_disease', None)
            enrichment = str(top_disease_cluster) if pd.notna(top_disease_cluster) else ''

            if enrichment:
                self.gene_data[gene]['orphanet']['enrichments'] = self.gene_data[gene]['orphanet'].get('enrichments', [])
                self.gene_data[gene]['orphanet']['enrichments'].append(enrichment)

            p_val_cluster = row.get('cluster_p_value', np.nan)
            self.gene_data[gene]['orphanet']['p_values'] = self.gene_data[gene]['orphanet'].get('p_values', [])
            self.gene_data[gene]['orphanet']['p_values'].append(p_val_cluster if pd.notna(p_val_cluster) else np.nan)

            fold_enr_cluster = row.get('cluster_combined_score', np.nan)
            self.gene_data[gene]['orphanet']['fold_enrichments'] = self.gene_data[gene]['orphanet'].get('fold_enrichments', [])
            self.gene_data[gene]['orphanet']['fold_enrichments'].append(fold_enr_cluster if pd.notna(fold_enr_cluster) else np.nan)

            self.gene_data[gene]['orphanet']['is_seed'] = row.get('is_seed_gene', False)
            self.gene_data[gene]['orphanet']['clusters'] = self.gene_data[gene]['orphanet'].get('clusters', set())
            self.gene_data[gene]['orphanet']['clusters'].add(cluster)

    def extract_omim_data(self, df):
        for _, row in df.iterrows():
            gene = str(row['gene'])
            cluster = str(row['cluster_id'])

            top_disease_cluster = row.get('cluster_top_ps', None)
            enrichment = str(top_disease_cluster) if pd.notna(top_disease_cluster) else ''

            if enrichment:
                self.gene_data[gene]['omim']['enrichments'] = self.gene_data[gene]['omim'].get('enrichments', [])
                self.gene_data[gene]['omim']['enrichments'].append(enrichment)

            p_val_cluster = row.get('cluster_p_value', np.nan)
            self.gene_data[gene]['omim']['p_values'] = self.gene_data[gene]['omim'].get('p_values', [])
            self.gene_data[gene]['omim']['p_values'].append(p_val_cluster if pd.notna(p_val_cluster) else np.nan)

            fold_enr_cluster = row.get('cluster_combined_score', np.nan)
            self.gene_data[gene]['omim']['fold_enrichments'] = self.gene_data[gene]['omim'].get('fold_enrichments', [])
            self.gene_data[gene]['omim']['fold_enrichments'].append(fold_enr_cluster if pd.notna(fold_enr_cluster) else np.nan)

            self.gene_data[gene]['omim']['is_seed'] = row.get('is_seed_gene', False)
            self.gene_data[gene]['omim']['clusters'] = self.gene_data[gene]['omim'].get('clusters', set())
            self.gene_data[gene]['omim']['clusters'].add(cluster)

    def extract_phenotype_data(self, df):
        for _, row in df.iterrows():
            gene = str(row['gene'])
            cluster = str(row['cluster_id'])

            top_phenotype_cluster = row.get('cluster_top_phenotype', None)
            enrichment = str(top_phenotype_cluster) if pd.notna(top_phenotype_cluster) else ''

            if enrichment:
                self.gene_data[gene]['phenotypes']['enrichments'] = self.gene_data[gene]['phenotypes'].get('enrichments', [])
                self.gene_data[gene]['phenotypes']['enrichments'].append(enrichment)

            p_val_cluster = row.get('cluster_p_value', np.nan)
            self.gene_data[gene]['phenotypes']['p_values'] = self.gene_data[gene]['phenotypes'].get('p_values', [])
            self.gene_data[gene]['phenotypes']['p_values'].append(p_val_cluster if pd.notna(p_val_cluster) else np.nan)

            fold_enr_cluster = row.get('cluster_fold_enrichment', np.nan)
            self.gene_data[gene]['phenotypes']['fold_enrichments'] = self.gene_data[gene]['phenotypes'].get('fold_enrichments', [])
            self.gene_data[gene]['phenotypes']['fold_enrichments'].append(fold_enr_cluster if pd.notna(fold_enr_cluster) else np.nan)

            self.gene_data[gene]['phenotypes']['is_seed'] = row.get('is_seed_gene', False)
            self.gene_data[gene]['phenotypes']['clusters'] = self.gene_data[gene]['phenotypes'].get('clusters', set())
            self.gene_data[gene]['phenotypes']['clusters'].add(cluster)

    def extract_spatial_data(self, df):
        for _, row in df.iterrows():
            gene = str(row['gene'])
            cluster = str(row['cluster_id'])

            top_cell_type = row.get('cluster_top_cell_type', None)
            enrichment = str(top_cell_type).strip() if pd.notna(top_cell_type) and str(top_cell_type).strip() else ''

            if enrichment:
                self.gene_data[gene]['spatial']['enrichments'] = self.gene_data[gene]['spatial'].get('enrichments', [])
                self.gene_data[gene]['spatial']['enrichments'].append(enrichment)

            p_val = row.get('cluster_p_value', np.nan)
            self.gene_data[gene]['spatial']['p_values'] = self.gene_data[gene]['spatial'].get('p_values', [])
            self.gene_data[gene]['spatial']['p_values'].append(p_val if pd.notna(p_val) else np.nan)

            self.gene_data[gene]['spatial']['adj_p_values'] = self.gene_data[gene]['spatial'].get('adj_p_values', [])
            self.gene_data[gene]['spatial']['adj_p_values'].append(p_val if pd.notna(p_val) else np.nan)

            fold_enr = row.get('cluster_odds_ratio', np.nan)
            self.gene_data[gene]['spatial']['fold_enrichments'] = self.gene_data[gene]['spatial'].get('fold_enrichments', [])
            self.gene_data[gene]['spatial']['fold_enrichments'].append(fold_enr if pd.notna(fold_enr) else np.nan)

            self.gene_data[gene]['spatial']['is_seed'] = row.get('is_seed_gene', False)
            self.gene_data[gene]['spatial']['clusters'] = self.gene_data[gene]['spatial'].get('clusters', set())
            self.gene_data[gene]['spatial']['clusters'].add(cluster)

    def process_all_analyses(self):
        analysis_dirs = self.find_analysis_directories()
        print("analysis dirs available")
        print(analysis_dirs)
        if not analysis_dirs:
            print("No analysis directories found!")
            return False

        extractors = {
            'proteomics': self.extract_proteomics_data,
            'neuropathic_pain': self.extract_pain_data,
            'orphanet': self.extract_orphanet_data,
            'omim': self.extract_omim_data,
            'phenotypes': self.extract_phenotype_data,
            'spatial': self.extract_spatial_data
        }

        for analysis_type, analysis_dir in analysis_dirs.items():
            print(f"\nProcessing {analysis_type}...")
            df = self.load_gene_level_results(analysis_dir, analysis_type)

            if df is not None and analysis_type in extractors:
                extractors[analysis_type](df)

        return True

    def create_combined_dataframe(self):
        rows = []

        for gene, analyses in self.gene_data.items():
            row = {'gene': gene}

            for analysis_type in ['proteomics', 'neuropathic_pain', 'orphanet', 'omim', 'phenotypes', 'spatial']:
                if analysis_type in analyses:
                    data = analyses[analysis_type]

                    enrichments = data.get('enrichments', [])
                    row[f'{analysis_type}_enrichments'] = ','.join(enrichments) if enrichments else ''

                    p_values = data.get('p_values', [])
                    p_values_str = ','.join([f"{p:.2e}" if pd.notna(p) and not np.isnan(p) else 'NA' for p in p_values])
                    row[f'{analysis_type}_p_values'] = p_values_str if p_values else ''

                    if analysis_type == 'spatial':
                        adj_p_values = data.get('adj_p_values', [])
                        adj_p_values_str = ','.join([f"{p:.2e}" if pd.notna(p) and not np.isnan(p) else 'NA' for p in adj_p_values])
                        row[f'{analysis_type}_adj_p_values'] = adj_p_values_str if adj_p_values else ''

                    fold_enrichments = data.get('fold_enrichments', [])
                    fold_enrichments_str = ','.join([f"{or_:.3f}" if pd.notna(or_) and not np.isnan(or_) else 'NA' for or_ in fold_enrichments])
                    row[f'{analysis_type}_fold_enrichments'] = fold_enrichments_str if fold_enrichments else ''

                    valid_p_values = [p for p in p_values if pd.notna(p) and not np.isnan(p)]
                    row[f'{analysis_type}_min_p_value'] = min(valid_p_values) if valid_p_values else np.nan

                    if analysis_type == 'spatial':
                        valid_adj_p_values = [p for p in adj_p_values if pd.notna(p) and not np.isnan(p)]
                        row[f'{analysis_type}_min_adj_p_value'] = min(valid_adj_p_values) if valid_adj_p_values else np.nan

                    row[f'{analysis_type}_is_seed'] = data.get('is_seed', False)

                    clusters = data.get('clusters', set())
                    row[f'{analysis_type}_num_clusters'] = len(clusters)

                    row[f'{analysis_type}_clusters'] = ','.join(sorted(clusters)) if clusters else ''
                else:
                    row[f'{analysis_type}_enrichments'] = ''
                    row[f'{analysis_type}_p_values'] = ''
                    if analysis_type == 'spatial':
                        row[f'{analysis_type}_adj_p_values'] = ''
                    row[f'{analysis_type}_fold_enrichments'] = ''
                    row[f'{analysis_type}_min_p_value'] = np.nan
                    if analysis_type == 'spatial':
                        row[f'{analysis_type}_min_adj_p_value'] = np.nan
                    row[f'{analysis_type}_is_seed'] = False
                    row[f'{analysis_type}_num_clusters'] = 0
                    row[f'{analysis_type}_clusters'] = ''

            all_min_p_values = []
            all_min_adj_p_values = []
            is_seed_any = False
            total_clusters = set()

            for analysis_type in ['proteomics', 'neuropathic_pain', 'orphanet', 'omim', 'phenotypes', 'spatial']:
                min_p = row.get(f'{analysis_type}_min_p_value', np.nan)
                if pd.notna(min_p) and not np.isnan(min_p):
                    all_min_p_values.append(min_p)

                if analysis_type == 'spatial':
                    min_adj_p = row.get(f'{analysis_type}_min_adj_p_value', np.nan)
                    if pd.notna(min_adj_p) and not np.isnan(min_adj_p):
                        all_min_adj_p_values.append(min_adj_p)

                if row.get(f'{analysis_type}_is_seed', False):
                    is_seed_any = True

                clusters_str = row.get(f'{analysis_type}_clusters', '')
                if clusters_str:
                    total_clusters.update(clusters_str.split(','))

            row['overall_min_p_value'] = min(all_min_p_values) if all_min_p_values else np.nan
            row['overall_min_adj_p_value'] = min(all_min_adj_p_values) if all_min_adj_p_values else np.nan
            row['is_seed_any_analysis'] = is_seed_any
            row['total_unique_clusters'] = len(total_clusters)
            row['num_analyses_with_data'] = sum([1 for at in ['proteomics', 'neuropathic_pain', 'orphanet', 'omim', 'phenotypes', 'spatial']
                                                 if row.get(f'{at}_num_clusters', 0) > 0])

            rows.append(row)

        df = pd.DataFrame(rows)

        sort_col = 'overall_min_adj_p_value' if 'overall_min_adj_p_value' in df.columns else 'overall_min_p_value'
        df = df.sort_values(sort_col, na_position='last')

        return df

    def filter_significant_enrichments(self, df, p_value_threshold=0.05):
        filtered_rows = []

        for _, row in df.iterrows():
            filtered_row = {'gene': row['gene']}
            has_significant = False

            for analysis_type in ['proteomics', 'neuropathic_pain', 'orphanet', 'omim', 'phenotypes', 'spatial']:
                enrichments_str = row.get(f'{analysis_type}_enrichments', '')

                if analysis_type == 'spatial':
                    p_values_str = row.get(f'{analysis_type}_adj_p_values', '')
                else:
                    p_values_str = row.get(f'{analysis_type}_p_values', '')

                fold_enrichments_str = row.get(f'{analysis_type}_fold_enrichments', '')

                if enrichments_str and p_values_str:
                    enrichments = enrichments_str.split(',')
                    p_values = p_values_str.split(',')
                    fold_enrichments = fold_enrichments_str.split(',') if fold_enrichments_str else []

                    sig_enrichments = []
                    sig_p_values = []
                    sig_fold_enrichments = []

                    for i, (enr, p) in enumerate(zip(enrichments, p_values)):
                        if p != 'NA':
                            try:
                                p_float = float(p)
                                if p_float < p_value_threshold:
                                    sig_enrichments.append(enr)
                                    sig_p_values.append(p)
                                    if i < len(fold_enrichments):
                                        sig_fold_enrichments.append(fold_enrichments[i])
                                    has_significant = True
                            except ValueError:
                                pass

                    filtered_row[f'{analysis_type}_enrichments'] = ','.join(sig_enrichments) if sig_enrichments else ''

                    if analysis_type == 'spatial':
                        filtered_row[f'{analysis_type}_adj_p_values'] = ','.join(sig_p_values) if sig_p_values else ''
                        filtered_row[f'{analysis_type}_p_values'] = row.get(f'{analysis_type}_p_values', '')
                    else:
                        filtered_row[f'{analysis_type}_p_values'] = ','.join(sig_p_values) if sig_p_values else ''

                    filtered_row[f'{analysis_type}_fold_enrichments'] = ','.join(sig_fold_enrichments) if sig_fold_enrichments else ''

                    if sig_p_values:
                        if analysis_type == 'spatial':
                            filtered_row[f'{analysis_type}_min_adj_p_value'] = min([float(p) for p in sig_p_values])
                        filtered_row[f'{analysis_type}_min_p_value'] = min([float(p) for p in sig_p_values])
                    else:
                        filtered_row[f'{analysis_type}_min_p_value'] = np.nan
                        if analysis_type == 'spatial':
                            filtered_row[f'{analysis_type}_min_adj_p_value'] = np.nan

                    filtered_row[f'{analysis_type}_is_seed'] = row.get(f'{analysis_type}_is_seed', False)
                    filtered_row[f'{analysis_type}_num_clusters'] = row.get(f'{analysis_type}_num_clusters', 0)
                    filtered_row[f'{analysis_type}_clusters'] = row.get(f'{analysis_type}_clusters', '')
                else:
                    filtered_row[f'{analysis_type}_enrichments'] = ''
                    filtered_row[f'{analysis_type}_p_values'] = ''
                    if analysis_type == 'spatial':
                        filtered_row[f'{analysis_type}_adj_p_values'] = ''
                    filtered_row[f'{analysis_type}_fold_enrichments'] = ''
                    filtered_row[f'{analysis_type}_min_p_value'] = np.nan
                    if analysis_type == 'spatial':
                        filtered_row[f'{analysis_type}_min_adj_p_value'] = np.nan
                    filtered_row[f'{analysis_type}_is_seed'] = row.get(f'{analysis_type}_is_seed', False)
                    filtered_row[f'{analysis_type}_num_clusters'] = row.get(f'{analysis_type}_num_clusters', 0)
                    filtered_row[f'{analysis_type}_clusters'] = row.get(f'{analysis_type}_clusters', '')

            filtered_row['overall_min_p_value'] = row.get('overall_min_p_value', np.nan)
            filtered_row['overall_min_adj_p_value'] = row.get('overall_min_adj_p_value', np.nan)
            filtered_row['is_seed_any_analysis'] = row.get('is_seed_any_analysis', False)
            filtered_row['total_unique_clusters'] = row.get('total_unique_clusters', 0)
            filtered_row['num_analyses_with_data'] = row.get('num_analyses_with_data', 0)

            if has_significant:
                filtered_rows.append(filtered_row)

        filtered_df = pd.DataFrame(filtered_rows)

        if not filtered_df.empty:
            sort_col = 'overall_min_adj_p_value' if 'overall_min_adj_p_value' in filtered_df.columns else 'overall_min_p_value'
            filtered_df = filtered_df.sort_values(sort_col, na_position='last')

        return filtered_df

    def create_compact_enrichment_dataframe(self, df, p_value_threshold=0.05):
        compact_rows = []

        source_names = {
            'proteomics': 'proteomics',
            'neuropathic_pain': 'neuropathicpain',
            'orphanet': 'orphanet',
            'omim': 'omim',
            'phenotypes': 'phenotypes',
            'spatial': 'spatial'
        }

        for _, row in df.iterrows():
            gene = row['gene']
            enrichment_dict = {}

            for analysis_type in ['proteomics', 'neuropathic_pain', 'orphanet', 'omim', 'phenotypes', 'spatial']:
                enrichments_str = row.get(f'{analysis_type}_enrichments', '')

                if analysis_type == 'spatial':
                    p_values_str = row.get(f'{analysis_type}_adj_p_values', '')
                else:
                    p_values_str = row.get(f'{analysis_type}_p_values', '')

                if enrichments_str and p_values_str:
                    enrichments = enrichments_str.split(',')
                    p_values = p_values_str.split(',')

                    source_name = source_names[analysis_type]

                    for enr, p in zip(enrichments, p_values):
                        if p != 'NA':
                            try:
                                p_float = float(p)
                                if p_float < p_value_threshold:
                                    if ';' in enr:
                                        sub_terms = enr.split(';')
                                        for sub_term in sub_terms:
                                            clean_term = sub_term.replace('(top)', '').replace('(weak)', '').strip()
                                            clean_term = clean_term.replace(' ', '_').replace('(', '').replace(')', '')
                                            key = f"{clean_term}_{source_name}"
                                            if key not in enrichment_dict or p_float < enrichment_dict[key]:
                                                enrichment_dict[key] = p_float
                                    else:
                                        clean_term = enr.replace('(top)', '').replace('(weak)', '').strip()
                                        clean_term = clean_term.replace(' ', '_').replace('(', '').replace(')', '')
                                        key = f"{clean_term}_{source_name}"
                                        if key not in enrichment_dict or p_float < enrichment_dict[key]:
                                            enrichment_dict[key] = p_float
                            except ValueError:
                                pass

            if enrichment_dict:
                enrichment_list = [f"{term}:{p:.2e}" for term, p in enrichment_dict.items()]
                compact_rows.append({
                    'gene': gene,
                    'enrichments': ', '.join(enrichment_list)
                })

        compact_df = pd.DataFrame(compact_rows)
        return compact_df

    def save_combined_results(self, output_file, p_value_threshold=0.05):
        df = self.create_combined_dataframe()

        if df.empty:
            print("No data to save!")
            return False

        df.to_csv(output_file, sep='\t', index=False)
        print(f"\nSaved combined results to: {output_file}")
        print(f"Total genes: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        sig_df = self.filter_significant_enrichments(df, p_value_threshold=p_value_threshold)

        if not sig_df.empty:
            base_name = output_file.rsplit('.', 1)[0]
            ext = output_file.rsplit('.', 1)[1] if '.' in output_file else 'tsv'
            sig_output_file = f"{base_name}_significant_only.{ext}"

            sig_df.to_csv(sig_output_file, sep='\t', index=False)
            print(f"\nSaved significant enrichments (p < {p_value_threshold}) to: {sig_output_file}")
            print(f"Significant genes: {len(sig_df)}")
        else:
            print(f"\nNo significant enrichments found at threshold p < {p_value_threshold}")

        compact_df = self.create_compact_enrichment_dataframe(df, p_value_threshold=p_value_threshold)

        if not compact_df.empty:
            base_name = output_file.rsplit('.', 1)[0]
            ext = output_file.rsplit('.', 1)[1] if '.' in output_file else 'tsv'
            compact_output_file = f"{base_name}_compact.{ext}"

            compact_df.to_csv(compact_output_file, sep='\t', index=False)
            print(f"\nSaved compact enrichment format to: {compact_output_file}")
            print(f"Genes with significant enrichments: {len(compact_df)}")
        else:
            print(f"\nNo significant enrichments for compact format")

        print("\n=== Summary Statistics (All Results) ===")
        for analysis_type in ['proteomics', 'neuropathic_pain', 'orphanet', 'omim', 'phenotypes', 'spatial']:
            genes_with_data = (df[f'{analysis_type}_num_clusters'] > 0).sum()
            print(f"{analysis_type}: {genes_with_data} genes with data")

        print(f"\nGenes with data in any analysis: {(df['num_analyses_with_data'] > 0).sum()}")
        print(f"Genes with data in multiple analyses: {(df['num_analyses_with_data'] > 1).sum()}")
        print(f"Seed genes: {df['is_seed_any_analysis'].sum()}")

        sig_genes = (df['overall_min_p_value'] < p_value_threshold).sum()
        print(f"\nGenes with significant enrichment (p < {p_value_threshold}): {sig_genes}")

        return True
