#!/usr/bin/env python3
"""
common_utils.py - Common utilities for cluster analysis scripts
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def detect_annotation_prefix(input_dir, ppi_source, cluster_type='seed_lcc', preferred_prefix=None):
    """Detect and select annotation prefix from available files."""
    
    # Handle topology cluster types (NEW)
    if cluster_type in ['top_lcc', 'top_net']:
        if cluster_type == 'top_lcc':
            search_dir = os.path.join(input_dir, ppi_source, 'cluster_test_lcc')
            expected_file = 'top_lcc_clusters.csv'
        else:  # top_net
            search_dir = os.path.join(input_dir, ppi_source, 'cluster_test')
            expected_file = 'top_net_clusters.csv'
        
        if os.path.exists(os.path.join(search_dir, expected_file)):
            return cluster_type  # Return the cluster_type itself as the "prefix"
        else:
            return None
    
    # Original logic for other cluster types
    if cluster_type == 'seed_lcc':
        search_dir = os.path.join(input_dir, ppi_source, 'seed_lcc_endotypes')
        pattern = '*_seed_lcc_clusters.csv'
        suffix = '_seed_lcc_clusters.csv'
    else:  # full_network
        search_dir = os.path.join(input_dir, ppi_source, 'endotypes')
        pattern = '*_clusters.csv'
        suffix = '_clusters.csv'
    
    if not os.path.exists(search_dir):
        return None
    
    # Look for cluster files with different prefixes
    cluster_files = glob.glob(os.path.join(search_dir, pattern))
    prefixes = []
    
    for file_path in cluster_files:
        filename = os.path.basename(file_path)
        # Extract prefix by removing the suffix
        prefix = filename.replace(suffix, '')
        # For seed_lcc files, there might be additional suffixes to handle
        if cluster_type == 'seed_lcc' and prefix.endswith('_seed_lcc'):
            prefix = prefix[:-9]  # Remove the extra _seed_lcc
        prefixes.append(prefix)
    
    if not prefixes:
        return None
    
    # Select prefix
    if preferred_prefix and preferred_prefix in prefixes:
        return preferred_prefix
    
    # Auto-select annotation-related prefixes
    annotation_prefixes = [p for p in prefixes if any(term in p.lower() for term in ['go', 'reactome', 'kegg'])]
    if annotation_prefixes:
        return annotation_prefixes[0]
    
    return prefixes[0]

def load_clusters(ppi_source, input_dir, annotation_prefix, cluster_type='seed_lcc', persistence_threshold=5):
    """
    Load cluster data in a standardized way.
    
    Parameters:
    -----------
    ppi_source : str
        PPI source name
    input_dir : str
        Base input directory
    annotation_prefix : str
        Annotation prefix to use (or 'top_lcc'/'top_net' for topology clusters)
    cluster_type : str
        'seed_lcc', 'full_network', 'disease_module_hidef', 'seed_lcc_hidef', 'top_lcc', or 'top_net'
    persistence_threshold : int
        Minimum persistence threshold for hidef cluster types (default: 5)
    
    Returns:
    --------
    clusters : dict
        Dictionary of cluster_name -> gene_list
    seed_genes : list
        List of seed genes (may be empty for full_network)
    cluster_seed_mapping : dict
        Dictionary of cluster_name -> seed_genes_in_cluster
    """
    clusters = {}
    seed_genes = []
    cluster_seed_mapping = {}
    
    # Handle topology cluster types (NEW)
    if cluster_type in ['top_lcc', 'top_net']:
        if cluster_type == 'top_lcc':
            cluster_dir = os.path.join(input_dir, ppi_source, 'cluster_test_lcc')
            cluster_file = os.path.join(cluster_dir, 'top_lcc_clusters.csv')
        else:  # top_net
            cluster_dir = os.path.join(input_dir, ppi_source, 'cluster_test')
            cluster_file = os.path.join(cluster_dir, 'top_net_clusters.csv')
        
        if not os.path.exists(cluster_file):
            print(f"Topology cluster file not found: {cluster_file}")
            return None, None, None
        
        try:
            cluster_df = pd.read_csv(cluster_file)
            print(f"Loaded topology cluster file: {cluster_file}")
            
            # Ensure correct column names
            if 'gene' in cluster_df.columns and 'cluster_id' in cluster_df.columns:
                # Good, already in correct format
                pass
            elif 'Gene' in cluster_df.columns and 'Cluster' in cluster_df.columns:
                cluster_df = cluster_df.rename(columns={'Gene': 'gene', 'Cluster': 'cluster_id'})
            else:
                print(f"Unexpected column format in {cluster_file}: {cluster_df.columns.tolist()}")
                return None, None, None
            
            # For topology clusters, we need to load seed genes separately
            # Try to find seed genes file
            seed_file_candidates = [
                os.path.join(cluster_dir, 'seed_genes.csv'),
                os.path.join(input_dir, ppi_source, 'seed_genes.csv'),
                os.path.join(input_dir, 'seed_genes.csv'),
                'seed_genes.csv'
            ]
            
            seed_genes_set = set()
            for seed_file in seed_file_candidates:
                if os.path.exists(seed_file):
                    try:
                        seed_df = pd.read_csv(seed_file)
                        gene_col = 'gene' if 'gene' in seed_df.columns else seed_df.columns[0]
                        seed_genes_set = set(seed_df[gene_col].astype(str).tolist())
                        print(f"Loaded seed genes from: {seed_file}")
                        break
                    except Exception as e:
                        print(f"Could not load seed genes from {seed_file}: {e}")
                        continue
            
            if not seed_genes_set:
                print("Warning: Could not find seed genes file, treating all genes as potential seeds")
                seed_genes_set = set(cluster_df['gene'].astype(str).unique().tolist())
            
            # Create cluster mappings, skipping single gene clusters
            for cluster_id in cluster_df['cluster_id'].unique():
                if pd.notna(cluster_id):
                    cluster_genes = cluster_df[cluster_df['cluster_id'] == cluster_id]['gene'].dropna().astype(str).tolist()
                    
                    # Skip single gene clusters
                    if len(cluster_genes) <= 1:
                        print(f"Skipping single gene cluster: {cluster_id}")
                        continue
                    
                    # Use the cluster_id as is (it's already in format like "cluster_1")
                    cluster_name = str(cluster_id)
                    clusters[cluster_name] = cluster_genes
                    
                    # Map seed genes in this cluster
                    cluster_seed_genes = [gene for gene in cluster_genes if gene in seed_genes_set]
                    cluster_seed_mapping[cluster_name] = cluster_seed_genes
            
            print(f"Loaded {len(clusters)} topology clusters (skipped single-gene clusters), {len(seed_genes_set)} seed genes")
            return clusters, list(seed_genes_set), cluster_seed_mapping
            
        except Exception as e:
            print(f"Error loading topology cluster data: {e}")
            return None, None, None
    
    # Handle new hidef cluster types
    elif cluster_type in ['disease_module_hidef', 'seed_lcc_hidef']:
        cluster_dir = os.path.join(input_dir, ppi_source, cluster_type)
        # Use regular if/else instead of np.where
        if cluster_type == 'disease_module_hidef':
            seed_file = 'disease_communities.csv'
        else:  # seed_lcc_hidef
            seed_file = 'seed_communities.csv'
            
        cluster_file = os.path.join(cluster_dir, seed_file)
        
        if not os.path.exists(cluster_file):
            print(f"HiDef cluster file not found: {cluster_file}")
            return None, None, None
        
        try:
            cluster_df = pd.read_csv(cluster_file)
            print(f"Loaded HiDef cluster file: {cluster_file}")
            
            # Standardize column names
            expected_cols = ['Community_ID', 'Gene', 'Gene_Type', 'Community_Size', 'Persistence', 'N_Resolutions']
            if not all(col in cluster_df.columns for col in expected_cols):
                print(f"Expected columns {expected_cols}, got {cluster_df.columns.tolist()}")
                return None, None, None
            
            # Filter by persistence threshold
            print(f"Filtering clusters with persistence >= {persistence_threshold}")
            cluster_df = cluster_df[cluster_df['Persistence'] >= persistence_threshold]
            
            if cluster_df.empty:
                print(f"No clusters found with persistence >= {persistence_threshold}")
                return None, None, None
            
            # Extract seed genes (those with Gene_Type indicating they are seed genes, not RWR_Neighbor)
            seed_genes = set(cluster_df[cluster_df['Gene_Type'] != 'RWR_Neighbor']['Gene'].astype(str).tolist())
            
            # Create cluster mappings, skipping single gene clusters
            for community_id in cluster_df['Community_ID'].unique():
                if pd.notna(community_id):
                    community_genes = cluster_df[cluster_df['Community_ID'] == community_id]['Gene'].dropna().astype(str).tolist()
                    
                    # Skip single gene clusters
                    if len(community_genes) <= 1:
                        print(f"Skipping single gene community: Community_{community_id}")
                        continue
                    
                    cluster_name = f"Community_{community_id}"
                    clusters[cluster_name] = community_genes
                    
                    # Map seed genes in this cluster
                    cluster_seed_genes = [gene for gene in community_genes if gene in seed_genes]
                    cluster_seed_mapping[cluster_name] = cluster_seed_genes
            
            print(f"Loaded {len(clusters)} communities (skipped single-gene communities), {len(seed_genes)} seed genes")
            return clusters, list(seed_genes), cluster_seed_mapping
            
        except Exception as e:
            print(f"Error loading HiDef cluster data: {e}")
            return None, None, None
    
    # Original logic for seed_lcc and full_network types
    else:
        # Determine paths based on cluster type
        if cluster_type == 'seed_lcc':
            cluster_dir = os.path.join(input_dir, ppi_source, 'seed_lcc_endotypes')
            cluster_file = os.path.join(cluster_dir, f'{annotation_prefix}_seed_lcc_clusters.csv')
            seed_file = os.path.join(cluster_dir, f'{annotation_prefix}_seed_lcc_genes.csv')
        else:  # full_network
            cluster_dir = os.path.join(input_dir, ppi_source, 'endotypes')
            cluster_file = os.path.join(cluster_dir, f'{annotation_prefix}_clusters.csv')
            seed_file = None  # Full network may not have separate seed file
        
        if not os.path.exists(cluster_file):
            print(f"Cluster file not found: {cluster_file}")
            # Try fallback paths
            if cluster_type == 'seed_lcc':
                fallback_paths = [
                    os.path.join(cluster_dir, 'seed_lcc_clusters.csv'),
                    os.path.join(input_dir, ppi_source, 'lcc_cluster_analysis', 'seed_lcc_clusters.csv')
                ]
            else:
                fallback_paths = [
                    os.path.join(cluster_dir, 'clusters.csv'),
                    os.path.join(input_dir, ppi_source, 'clusters.csv')
                ]
            
            for path in fallback_paths:
                if os.path.exists(path):
                    cluster_file = path
                    print(f"Using fallback cluster file: {cluster_file}")
                    break
            else:
                return None, None, None
        
        # Load clusters
        try:
            cluster_df = pd.read_csv(cluster_file)
            print(f"Loaded cluster file: {cluster_file}")
            
            # Standardize column names
            if 'Gene' not in cluster_df.columns:
                gene_cols = [col for col in cluster_df.columns if 'gene' in col.lower()]
                if gene_cols:
                    cluster_df = cluster_df.rename(columns={gene_cols[0]: 'Gene'})
            
            if 'Cluster' not in cluster_df.columns:
                cluster_cols = [col for col in cluster_df.columns if 'cluster' in col.lower()]
                if cluster_cols:
                    cluster_df = cluster_df.rename(columns={cluster_cols[0]: 'Cluster'})
            
            # Load seed genes if available and if seed_lcc type
            if cluster_type == 'seed_lcc' and seed_file and os.path.exists(seed_file):
                seed_df = pd.read_csv(seed_file)
                gene_col = 'Gene' if 'Gene' in seed_df.columns else seed_df.columns[0]
                seed_genes = set(seed_df[gene_col].astype(str).tolist())
            else:
                # Use all cluster genes as seed genes for full network
                seed_genes = set(cluster_df['Gene'].astype(str).unique().tolist())
            
            # Create cluster mappings, skipping single gene clusters
            for cluster_id in cluster_df['Cluster'].unique():
                if pd.notna(cluster_id):
                    cluster_genes = cluster_df[cluster_df['Cluster'] == cluster_id]['Gene'].dropna().astype(str).tolist()
                    
                    # Skip single gene clusters
                    if len(cluster_genes) <= 1:
                        print(f"Skipping single gene cluster: Cluster_{cluster_id}")
                        continue
                    
                    cluster_name = f"Cluster_{cluster_id}"
                    clusters[cluster_name] = cluster_genes
                    
                    # Map seed genes in this cluster
                    cluster_seed_genes = [gene for gene in cluster_genes if gene in seed_genes]
                    cluster_seed_mapping[cluster_name] = cluster_seed_genes
            
            print(f"Loaded {len(clusters)} clusters (skipped single-gene clusters), {len(seed_genes)} seed genes")
            return clusters, list(seed_genes), cluster_seed_mapping
            
        except Exception as e:
            print(f"Error loading cluster data: {e}")
            return None, None, None

def create_standardized_results(cluster_name, cluster_genes, cluster_seed_genes, cluster_type, 
                              total_genes_analyzed, percentage, mean_statistic, p_value, 
                              adjusted_p_value, statistic_name='odds_ratio', 
                              genes_in_expression_data=None):  # NEW PARAMETER
    """Create standardized result dictionary."""
    result = {
        'cluster_id': cluster_name,
        'cluster_type': cluster_type,
        'seed_genes_in_cluster': cluster_seed_genes,
        'seed_gene_count_in_cluster': len(cluster_seed_genes),
        'cluster_size': len(cluster_genes),
        'total_genes_analyzed': total_genes_analyzed,
        'genes_in_expression_data': genes_in_expression_data if genes_in_expression_data is not None else total_genes_analyzed,  # NEW
        'genes_not_in_expression_data': len(cluster_genes) - (genes_in_expression_data if genes_in_expression_data is not None else total_genes_analyzed),  # NEW
        'percentage_in_expression_data': (genes_in_expression_data / len(cluster_genes) * 100) if genes_in_expression_data is not None else (total_genes_analyzed / len(cluster_genes) * 100),  # NEW
        'percentage': percentage,
        f'mean_{statistic_name}': mean_statistic,
        'p_value': p_value,
        'adjusted_p_value': adjusted_p_value
    }
    return result

def create_gene_level_results(cluster_name, gene_name, gene_statistic, gene_p_value, 
                            is_seed_gene, statistic_name='odds_ratio', top_category=None):  # NEW PARAMETER
    """Create gene-level result dictionary."""
    result = {
        'cluster_id': cluster_name,
        'gene': gene_name,
        'is_seed_gene': is_seed_gene,
        f'{statistic_name}': gene_statistic,
        'p_value': gene_p_value if gene_p_value is not None else np.nan
    }
    if top_category is not None:  # NEW
        result['top_category'] = top_category
    return result

def create_summary_heatmap(enrichment_df, value_col='percentage', title='Enrichment Summary'):
    """Create a standardized summary heatmap."""
    if enrichment_df.empty:
        return None
    
    # Create pivot table
    pivot_data = enrichment_df.pivot_table(
        index='cluster_id', 
        columns='category' if 'category' in enrichment_df.columns else 'phenotype',
        values=value_col,
        fill_value=0
    )
    
    if pivot_data.empty:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(pivot_data.columns) * 0.5), 
                                   max(8, len(pivot_data) * 0.4)))
    
    # Plot heatmap
    sns.heatmap(pivot_data, 
                annot=True, 
                fmt='.1f',
                cmap='YlOrRd',
                linewidths=0.5,
                ax=ax)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def save_results(results_df, gene_results_df, output_dir, prefix):
    """Save results in standardized format."""
    ensure_dir(output_dir)
    
    # Save summarized results
    results_df.to_csv(os.path.join(output_dir, f'{prefix}_summarized_results.csv'), index=False)
    
    # Save gene-level results
    if gene_results_df is not None and not gene_results_df.empty:
        gene_results_df.to_csv(os.path.join(output_dir, f'{prefix}_gene_level_results.csv'), index=False)
    
    # Save significant results
    if 'adjusted_p_value' in results_df.columns:
        sig_results = results_df[results_df['adjusted_p_value'] < 0.05]
        if not sig_results.empty:
            sig_results.to_csv(os.path.join(output_dir, f'{prefix}_significant_summarized_results.csv'), index=False)
    
    # Save summary statistics
    summary_stats = {
        'total_clusters_analyzed': len(results_df),
        'significant_clusters': len(results_df[results_df['adjusted_p_value'] < 0.05]) if 'adjusted_p_value' in results_df.columns else 0,
        'total_genes_analyzed': gene_results_df['gene'].nunique() if gene_results_df is not None and not gene_results_df.empty else 0
    }
    
    with open(os.path.join(output_dir, f'{prefix}_summary.txt'), 'w') as f:
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")
            
def detect_available_annotation_prefixes(seed_lcc_dir):
    """Detect available annotation prefixes from existing files"""
    
    if not os.path.exists(seed_lcc_dir):
        return []
    
    # Look for cluster files with different prefixes
    cluster_files = glob.glob(f"{seed_lcc_dir}/*_seed_lcc_clusters.csv")
    prefixes = []
    
    for file_path in cluster_files:
        filename = os.path.basename(file_path)
        # Extract prefix (everything before "_seed_lcc_clusters.csv")
        prefix = filename.replace("_seed_lcc_clusters.csv", "")
        prefixes.append(prefix)
    
    return prefixes

def detect_available_annotation_prefixes_full_network(endotypes_dir):
    """Detect available annotation prefixes from endotypes directory for full_network"""
    
    if not os.path.exists(endotypes_dir):
        return []
    
    # Look for cluster files with different prefixes in endotypes directory
    cluster_files = glob.glob(f"{endotypes_dir}/*_clusters.csv")
    prefixes = []
    
    for file_path in cluster_files:
        filename = os.path.basename(file_path)
        # Extract prefix (everything before "_clusters.csv")
        prefix = filename.replace("_clusters.csv", "")
        prefixes.append(prefix)
    
    return prefixes

def select_annotation_prefix(available_prefixes, preferred_prefix=None):
    """Select annotation prefix to use"""
    if not available_prefixes:
        return None
    
    if preferred_prefix:
        # First try exact match
        if preferred_prefix in available_prefixes:
            return preferred_prefix
        
        # Then try partial match (e.g., 'go' matches 'go-bp-2025')
        for prefix in available_prefixes:
            if preferred_prefix.lower() in prefix.lower():
                return prefix
    
    return available_prefixes[0] if available_prefixes else None

def create_enriched_gene_level_results(cluster_name, gene_name, gene_statistic, gene_p_value, 
                                      is_seed_gene, is_in_expression_data,
                                      cluster_significant, cluster_statistic, cluster_p_value, 
                                      cluster_adjusted_p_value, top_category_gene, top_category_cluster,
                                      statistic_name='odds_ratio', gene_adjusted_p_value=None):
    """Create enriched gene-level result dictionary with cluster context."""
    return {
        'cluster_id': cluster_name,
        'gene': gene_name,
        'is_seed_gene': is_seed_gene,
        'is_in_expression_data': is_in_expression_data,
        'is_gene_significant': gene_adjusted_p_value < 0.05 if gene_adjusted_p_value is not None else False,
        'is_cluster_significant': cluster_significant,
        f'{statistic_name}_gene': gene_statistic,
        f'{statistic_name}_cluster': cluster_statistic,
        'p_value_gene': gene_p_value if gene_p_value is not None else np.nan,
        'adjusted_p_value_gene': gene_adjusted_p_value if gene_adjusted_p_value is not None else np.nan,
        'p_value_cluster': cluster_p_value,
        'adjusted_p_value_cluster': cluster_adjusted_p_value,
        f'top_{statistic_name.split("_")[0]}_gene': top_category_gene,  # e.g., top_disease_gene or top_log_gene
        f'top_{statistic_name.split("_")[0]}_cluster': top_category_cluster
    }