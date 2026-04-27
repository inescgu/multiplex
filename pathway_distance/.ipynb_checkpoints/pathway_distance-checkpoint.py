#!/usr/bin/env python

"""
Pathway Distance Analysis: Compute distance from disease module to known signaling pathways
"""

import os
import sys
import argparse
import networkx as nx
import pickle
import pandas as pd
from collections import defaultdict, deque
import numpy as np

def load_disease_module(module_file):
    """Load disease module genes"""
    with open(module_file, 'rb') as f:
        module = pickle.load(f)
    return set(module)

def load_gene_clusters(cluster_file):
    """
    Load gene clusters from CSV file
    
    Expected format: Community_ID, Gene (and other columns)
    
    Args:
        cluster_file: Path to CSV file with gene clusters
    
    Returns:
        clusters dict {cluster_id: set of genes}
    """
    try:
        df = pd.read_csv(cluster_file)
        
        if 'Community_ID' not in df.columns or 'Gene' not in df.columns:
            raise ValueError(f"CSV must have 'Community_ID' and 'Gene' columns. Found: {df.columns.tolist()}")
        
        clusters = {}
        for _, row in df.iterrows():
            cluster_id = str(row['Community_ID'])
            gene = str(row['Gene'])
            
            if cluster_id not in clusters:
                clusters[cluster_id] = set()
            clusters[cluster_id].add(gene)
        
        print(f"Loaded {len(clusters)} clusters:")
        for cluster_id in sorted(clusters.keys()):
            print(f"  Cluster {cluster_id}: {len(clusters[cluster_id])} genes")
        
        return clusters
    
    except Exception as e:
        print(f"ERROR loading clusters: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



    """Load disease module genes"""
    with open(module_file, 'rb') as f:
        module = pickle.load(f)
    return set(module)

def load_pathway_genes(pathway_file, gene_id_column=None):
    """
    Load pathway genes from CSV file
    
    Automatically detects gene ID column (priority order):
    1. gene_name_symbol (preferred - human readable gene symbols like BRCA1)
    2. gene_symbol (alternative symbol column)
    3. gene_name_uniprot (UniProt IDs like P38398)
    4. gene_name (KEGG IDs like hsa:10327)
    
    And uses 'pathway_name' column for pathways.
    
    Args:
        pathway_file: Path to CSV file
        gene_id_column: Optional - explicitly specify which column to use for gene IDs
    
    Returns:
        pathways dict {pathway_name: set of gene_ids}
    """
    try:
        df = pd.read_csv(pathway_file, index_col=0)  # Skip index column
        
        # Available columns
        available_cols = df.columns.tolist()
        
        # Find pathway column
        if 'pathway_name' not in available_cols:
            print(f"ERROR: 'pathway_name' column not found")
            print(f"Available columns: {available_cols}")
            sys.exit(1)
        
        # Find gene ID column
        gene_col = gene_id_column
        
        if gene_col is None:
            # Priority order for which gene ID column to use
            priority_cols = [
                'gene_name_symbol',  # Best - human readable symbols
                'gene_symbol',        # Alternative symbol
                'gene_name_uniprot',  # UniProt IDs
                'gene_name'           # KEGG IDs
            ]
            
            for col in priority_cols:
                if col in available_cols:
                    gene_col = col
                    print(f"Using '{gene_col}' column for gene IDs")
                    break
            
            if gene_col is None:
                print(f"ERROR: Could not find suitable gene ID column")
                print(f"Available columns: {available_cols}")
                print(f"Expected one of: {', '.join(priority_cols)}")
                sys.exit(1)
        else:
            if gene_col not in available_cols:
                print(f"ERROR: Specified column '{gene_col}' not found")
                print(f"Available columns: {available_cols}")
                sys.exit(1)
        
        # Build pathways dict
        pathways = defaultdict(set)
        
        for _, row in df.iterrows():
            gene = row[gene_col]
            pathway = row['pathway_name']
            
            # Skip NaN values
            if pd.notna(gene) and pd.notna(pathway):
                pathways[pathway].add(str(gene))
        
        print(f"Loaded {len(pathways)} pathways with {sum(len(genes) for genes in pathways.values())} gene associations")
        
        return pathways
    
    except Exception as e:
        print(f"ERROR loading pathways: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def bfs_distance_to_pathways(G, disease_module, pathway_genes_dict):
    """
    Compute shortest distance from each disease module gene to each pathway
    
    Returns:
        distances: dict {gene: {pathway: distance}}
        closest_pathway_genes: dict {gene: {pathway: [closest_genes]}}
    """
    distances = {}
    closest_pathway_genes = {}
    
    print(f"Computing distances from {len(disease_module)} disease module genes to pathways...")
    
    for i, gene in enumerate(disease_module):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(disease_module)} genes")
        
        if gene not in G:
            continue
        
        distances[gene] = {}
        closest_pathway_genes[gene] = {}
        
        # BFS from this gene
        queue = deque([(gene, 0)])
        visited = {gene}
        
        while queue:
            current_node, dist = queue.popleft()
            
            # Check if this node belongs to any pathway
            for pathway_name, pathway_genes in pathway_genes_dict.items():
                if current_node in pathway_genes and pathway_name not in distances[gene]:
                    distances[gene][pathway_name] = dist
                    closest_pathway_genes[gene][pathway_name] = [current_node]
                elif current_node in pathway_genes and pathway_name in distances[gene]:
                    if dist == distances[gene][pathway_name]:
                        closest_pathway_genes[gene][pathway_name].append(current_node)
            
            # Continue BFS (limit depth to avoid infinite search)
            if dist < 5:  # Maximum distance of 5
                for neighbor in G.neighbors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
    
    return distances, closest_pathway_genes

def compute_pathway_statistics(distances, pathway_genes_dict, disease_module):
    """
    Compute comprehensive distance statistics for each pathway
    
    Includes:
    - Basic distances: min, max, mean, median, std dev
    - Weighted measures accounting for pathway size
    - Quartile distances
    """
    stats = {}
    
    for pathway_name in pathway_genes_dict.keys():
        pathway_distances = []
        genes_reaching_pathway = 0
        
        for gene in disease_module:
            if gene in distances and pathway_name in distances[gene]:
                pathway_distances.append(distances[gene][pathway_name])
                genes_reaching_pathway += 1
        
        if pathway_distances:
            pathway_distances_array = np.array(pathway_distances)
            
            stats[pathway_name] = {
                # Basic statistics
                'mean_distance': float(np.mean(pathway_distances_array)),
                'median_distance': float(np.median(pathway_distances_array)),
                'min_distance': float(np.min(pathway_distances_array)),
                'max_distance': float(np.max(pathway_distances_array)),
                'std_distance': float(np.std(pathway_distances_array)),
                
                # Quartile distances
                'q25_distance': float(np.percentile(pathway_distances_array, 25)),
                'q75_distance': float(np.percentile(pathway_distances_array, 75)),
                
                # Weighted average accounting for pathway size
                'weighted_distance': float(np.mean(pathway_distances_array) / len(pathway_genes_dict[pathway_name])),
                
                # Coverage metrics
                'genes_reaching_pathway': genes_reaching_pathway,
                'percent_coverage': 100 * genes_reaching_pathway / len(disease_module),
                'pathway_size': len(pathway_genes_dict[pathway_name]),
                
                # Accessibility score (0-1, higher = more accessible)
                'accessibility_score': 1.0 / (1.0 + np.mean(pathway_distances_array))
            }
        else:
            stats[pathway_name] = {
                'mean_distance': None,
                'median_distance': None,
                'min_distance': None,
                'max_distance': None,
                'std_distance': None,
                'q25_distance': None,
                'q75_distance': None,
                'weighted_distance': None,
                'genes_reaching_pathway': 0,
                'percent_coverage': 0,
                'pathway_size': len(pathway_genes_dict[pathway_name]),
                'accessibility_score': 0.0
            }
    
    return stats


def _compute_within_module_distances(G, module, max_distance=5):
    """Compute all pairwise distances within a module"""
    distances = []
    module_list = list(module)
    
    for i, gene_a in enumerate(module_list):
        if gene_a not in G:
            continue
        
        # BFS from gene_a to all other genes in module
        queue = deque([(gene_a, 0)])
        visited = {gene_a}
        
        while queue:
            current, dist = queue.popleft()
            
            if dist > 0 and current in module:
                distances.append(dist)
            
            if dist < max_distance:
                for neighbor in G.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
    
    return distances


def _compute_between_module_distances(G, module_1, module_2, max_distance=5):
    """Compute distances between two modules"""
    distances = []
    
    for gene_a in module_1:
        if gene_a not in G:
            continue
        
        # BFS from gene_a to module_2
        queue = deque([(gene_a, 0)])
        visited = {gene_a}
        found_in_module_2 = False
        
        while queue and not found_in_module_2:
            current, dist = queue.popleft()
            
            if current in module_2 and dist > 0:
                distances.append(dist)
                found_in_module_2 = True
                break
            
            if dist < max_distance:
                for neighbor in G.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
    
    return distances


def compute_network_separation(G, disease_module_1, disease_module_2, max_distance=5):
    """
    Compute network-based separation metric between two disease modules/pathways
    
    Based on: sAB = <dAB> - (<dAA> + <dBB>)/2
    
    Where:
    - <dAA>: mean distance between proteins within disease A
    - <dBB>: mean distance between proteins within disease B  
    - <dAB>: mean distance between proteins in A and B
    
    Interpretation:
    - sAB < 0: Overlapping modules (shared pathways)
    - 0 < sAB < 1: Moderately separated
    - sAB > 1: Topologically separated (distinct modules)
    
    Args:
        G: NetworkX graph
        disease_module_1: Set of gene IDs for disease/pathway 1
        disease_module_2: Set of gene IDs for disease/pathway 2
        max_distance: Maximum search distance in BFS
    
    Returns:
        dict with separation metrics and statistics
    """
    
    # Convert to sets if needed and intersect with graph nodes
    module_1 = set(disease_module_1) & set(G.nodes())
    module_2 = set(disease_module_2) & set(G.nodes())
    
    if len(module_1) == 0 or len(module_2) == 0:
        return {
            'separation_score': None,
            'error': 'One or both disease modules have no genes in network'
        }
    
    # Compute within-disease distances
    within_distances_1 = _compute_within_module_distances(G, module_1, max_distance)
    within_distances_2 = _compute_within_module_distances(G, module_2, max_distance)
    
    # Compute between-disease distances
    between_distances = _compute_between_module_distances(G, module_1, module_2, max_distance)
    
    # Calculate means
    mean_within_1 = float(np.mean(within_distances_1)) if len(within_distances_1) > 0 else None
    mean_within_2 = float(np.mean(within_distances_2)) if len(within_distances_2) > 0 else None
    mean_between = float(np.mean(between_distances)) if len(between_distances) > 0 else None
    
    # Classic clustering linkage metrics for between-module distances
    single_linkage_between = float(np.min(between_distances)) if len(between_distances) > 0 else None
    complete_linkage_between = float(np.max(between_distances)) if len(between_distances) > 0 else None
    average_linkage_between = mean_between
    centroid_linkage_between = float(np.median(between_distances)) if len(between_distances) > 0 else None
    
    # Same metrics for within-module distances
    single_linkage_within_1 = float(np.min(within_distances_1)) if len(within_distances_1) > 0 else None
    complete_linkage_within_1 = float(np.max(within_distances_1)) if len(within_distances_1) > 0 else None
    average_linkage_within_1 = mean_within_1
    centroid_linkage_within_1 = float(np.median(within_distances_1)) if len(within_distances_1) > 0 else None
    
    single_linkage_within_2 = float(np.min(within_distances_2)) if len(within_distances_2) > 0 else None
    complete_linkage_within_2 = float(np.max(within_distances_2)) if len(within_distances_2) > 0 else None
    average_linkage_within_2 = mean_within_2
    centroid_linkage_within_2 = float(np.median(within_distances_2)) if len(within_distances_2) > 0 else None
    
    # Calculate separation score: sAB = <dAB> - (<dAA> + <dBB>)/2
    separation_score = None
    if mean_between is not None and mean_within_1 is not None and mean_within_2 is not None:
        separation_score = mean_between - (mean_within_1 + mean_within_2) / 2.0
    
    # Calculate linkage-based separation scores
    # Single linkage separation: min_between - (min_within_1 + min_within_2)/2
    single_linkage_separation = None
    if single_linkage_between is not None and single_linkage_within_1 is not None and single_linkage_within_2 is not None:
        single_linkage_separation = single_linkage_between - (single_linkage_within_1 + single_linkage_within_2) / 2.0
    
    # Complete linkage separation: max_between - (max_within_1 + max_within_2)/2
    complete_linkage_separation = None
    if complete_linkage_between is not None and complete_linkage_within_1 is not None and complete_linkage_within_2 is not None:
        complete_linkage_separation = complete_linkage_between - (complete_linkage_within_1 + complete_linkage_within_2) / 2.0
    
    # Interpretation
    interpretation = None
    if separation_score is not None:
        if separation_score < 0:
            interpretation = "Overlapping - modules share network neighborhood"
        elif separation_score < 1:
            interpretation = "Moderately separated - some pathway overlap"
        else:
            interpretation = "Topologically separated - distinct network modules"
    
    return {
        # Main separation score (average linkage)
        'separation_score': separation_score,
        
        # Within-module distances (average linkage)
        'mean_within_module_1': mean_within_1,
        'mean_within_module_2': mean_within_2,
        
        # Between-module distances (average linkage)
        'mean_between_modules': mean_between,
        
        # Classic linkage metrics - Single Linkage (closest pair)
        'single_linkage_within_module_1': single_linkage_within_1,
        'single_linkage_within_module_2': single_linkage_within_2,
        'single_linkage_between_modules': single_linkage_between,
        'single_linkage_separation': single_linkage_separation,
        
        # Classic linkage metrics - Complete Linkage (furthest pair)
        'complete_linkage_within_module_1': complete_linkage_within_1,
        'complete_linkage_within_module_2': complete_linkage_within_2,
        'complete_linkage_between_modules': complete_linkage_between,
        'complete_linkage_separation': complete_linkage_separation,
        
        # Classic linkage metrics - Average Linkage (mean)
        'average_linkage_within_module_1': average_linkage_within_1,
        'average_linkage_within_module_2': average_linkage_within_2,
        'average_linkage_between_modules': average_linkage_between,
        'average_linkage_separation': separation_score,  # Same as main score
        
        # Classic linkage metrics - Centroid Linkage (median)
        'centroid_linkage_within_module_1': centroid_linkage_within_1,
        'centroid_linkage_within_module_2': centroid_linkage_within_2,
        'centroid_linkage_between_modules': centroid_linkage_between,
        
        # Counts and sizes
        'within_distances_1_count': len(within_distances_1),
        'within_distances_2_count': len(within_distances_2),
        'between_distances_count': len(between_distances),
        'module_1_size': len(module_1),
        'module_2_size': len(module_2),
        'overlap_size': len(module_1 & module_2),
        'interpretation': interpretation,
        
        # Additional statistics
        'min_between_distance': single_linkage_between,
        'max_between_distance': complete_linkage_between,
        'median_between_distance': centroid_linkage_between,
        'std_between_distance': float(np.std(between_distances)) if len(between_distances) > 0 else None,
    }

def main():
    parser = argparse.ArgumentParser(
        description='Compute distance from disease module to signaling pathways'
    )
    parser.add_argument('--module', type=str, required=True,
                       help='Path to disease_module.pkl file')
    parser.add_argument('--graph', type=str, required=True,
                       help='Path to module_graph.pkl or full network graph')
    parser.add_argument('--pathways', type=str, required=True,
                       help='CSV file with pathway genes')
    parser.add_argument('--gene-id-column', type=str, default=None,
                       help='Which column to use for gene IDs (default: auto-detect from gene_name_symbol, gene_symbol, gene_name_uniprot, or gene_name)')
    parser.add_argument('--clusters', type=str, default=None,
                       help='Optional: CSV file with gene clusters (must have Community_ID and Gene columns)')
    parser.add_argument('--output-dir', type=str, default='pathway_output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Pathway Distance Analysis")
    print("="*80)
    
    # Load disease module
    print(f"\nLoading disease module from {args.module}...")
    disease_module = load_disease_module(args.module)
    print(f"Loaded {len(disease_module)} disease module genes")
    
    # Load graph
    print(f"\nLoading graph from {args.graph}...")
    with open(args.graph, 'rb') as f:
        G = pickle.load(f)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Load pathway genes
    print(f"\nLoading pathway genes from {args.pathways}...")
    pathway_genes_dict = load_pathway_genes(args.pathways, gene_id_column=args.gene_id_column)
    print(f"Loaded {len(pathway_genes_dict)} pathways")
    for pathway_name, genes in pathway_genes_dict.items():
        print(f"  {pathway_name}: {len(genes)} genes")
    
    # Compute distances
    print("\n" + "="*80)
    distances, closest_pathway_genes = bfs_distance_to_pathways(G, disease_module, pathway_genes_dict)
    print("Distance computation complete")
    
    # Compute statistics
    print("\nComputing pathway statistics...")
    stats = compute_pathway_statistics(distances, pathway_genes_dict, disease_module)
    
    # Print summary
    print("\n" + "="*80)
    print("Pathway Distance Summary")
    print("="*80)
    for pathway_name, stat in sorted(stats.items(), key=lambda x: x[1]['mean_distance'] if x[1]['mean_distance'] else float('inf')):
        print(f"\n{pathway_name}:")
        print(f"  Genes reaching pathway: {stat['genes_reaching_pathway']}/{len(disease_module)} ({stat['percent_coverage']:.1f}%)")
        if stat['mean_distance'] is not None:
            print(f"  Mean distance: {stat['mean_distance']:.2f}")
            print(f"  Median distance: {stat['median_distance']:.2f}")
            print(f"  Range: {stat['min_distance']:.0f} - {stat['max_distance']:.0f}")
    
    # Save detailed results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    # Save per-gene distances
    gene_results = []
    for gene in disease_module:
        if gene in distances:
            for pathway_name, dist in distances[gene].items():
                closest_genes = closest_pathway_genes[gene].get(pathway_name, [])
                gene_results.append({
                    'gene': gene,
                    'pathway': pathway_name,
                    'distance': dist,
                    'closest_pathway_gene': closest_genes[0] if closest_genes else None
                })
    
    gene_df = pd.DataFrame(gene_results)
    gene_output = os.path.join(args.output_dir, "gene_pathway_distances.csv")
    gene_df.to_csv(gene_output, index=False)
    print(f"Saved per-gene distances to {gene_output}")
    
    # Save pathway summary statistics
    stats_data = []
    for pathway_name, stat in stats.items():
        stat['pathway'] = pathway_name
        stats_data.append(stat)
    
    stats_df = pd.DataFrame(stats_data)
    stats_output = os.path.join(args.output_dir, "pathway_summary_statistics.csv")
    stats_df.to_csv(stats_output, index=False)
    print(f"Saved pathway statistics to {stats_output}")
    
    # Compute network-based separation between disease module and each pathway
    print("\n" + "="*80)
    print("Computing Network-Based Separation Analysis")
    print("="*80)
    
    separation_results = []
    for pathway_name, pathway_genes in pathway_genes_dict.items():
        if len(pathway_genes) > 0:
            separation = compute_network_separation(
                G, disease_module, pathway_genes, max_distance=5
            )
            separation['pathway_name'] = pathway_name
            separation['disease_module_size'] = len(disease_module)
            separation_results.append(separation)
            
            if separation['separation_score'] is not None:
                print(f"\n{pathway_name}:")
                print(f"  Separation Score: {separation['separation_score']:.3f}")
                print(f"  Interpretation: {separation['interpretation']}")
                print(f"  Mean within disease: {separation['mean_within_module_1']:.2f}")
                print(f"  Mean within pathway: {separation['mean_within_module_2']:.2f}")
                print(f"  Mean between: {separation['mean_between_modules']:.2f}")
                if separation['overlap_size'] > 0:
                    print(f"  Overlap size: {separation['overlap_size']} genes")
    
    separation_df = pd.DataFrame(separation_results)
    separation_output = os.path.join(args.output_dir, "network_separation_analysis.csv")
    separation_df.to_csv(separation_output, index=False)
    print(f"\nSaved network separation analysis to {separation_output}")
    
    # Compute pairwise cluster-to-cluster comparisons
    print("\n" + "="*80)
    print("Computing Pairwise Pathway-to-Pathway Separation Analysis")
    print("="*80)
    print("(This shows how separated different signaling pathways are from each other)")
    
    pathway_list = list(pathway_genes_dict.keys())
    pairwise_results = []
    
    for i, pathway_1 in enumerate(pathway_list):
        for j, pathway_2 in enumerate(pathway_list):
            if i < j:  # Only compute upper triangle to avoid duplicates
                pathway_1_genes = pathway_genes_dict[pathway_1]
                pathway_2_genes = pathway_genes_dict[pathway_2]
                
                if len(pathway_1_genes) > 0 and len(pathway_2_genes) > 0:
                    pairwise_sep = compute_network_separation(
                        G, pathway_1_genes, pathway_2_genes, max_distance=5
                    )
                    pairwise_sep['pathway_1'] = pathway_1
                    pairwise_sep['pathway_2'] = pathway_2
                    pairwise_sep['pathway_1_size'] = len(pathway_1_genes)
                    pairwise_sep['pathway_2_size'] = len(pathway_2_genes)
                    pairwise_results.append(pairwise_sep)
    
    if pairwise_results:
        pairwise_df = pd.DataFrame(pairwise_results)
        pairwise_output = os.path.join(args.output_dir, "pairwise_pathway_separation.csv")
        pairwise_df.to_csv(pairwise_output, index=False)
        print(f"Computed {len(pairwise_results)} pairwise pathway comparisons")
        print(f"Saved to {pairwise_output}")
        
        # Show overlapping pathways
        overlapping_pathways = pairwise_df[pairwise_df['separation_score'] < 0]
        if len(overlapping_pathways) > 0:
            print(f"\nFound {len(overlapping_pathways)} overlapping pathway pairs:")
            for _, row in overlapping_pathways.nsmallest(5, 'separation_score').iterrows():
                print(f"  {row['pathway_1']} <-> {row['pathway_2']}: s={row['separation_score']:.3f}")
    else:
        pairwise_df = pd.DataFrame()
        print("No pairwise comparisons computed (need at least 2 pathways)")
    
    # Cluster-to-pathway analysis (if clusters provided)
    cluster_pathway_results = []
    cluster_stats = {}
    
    if args.clusters:
        print("\n" + "="*80)
        print("Computing Cluster-to-Pathway Separation Analysis")
        print("="*80)
        
        # Load clusters
        clusters = load_gene_clusters(args.clusters)
        
        # Compute cluster-to-pathway separation
        for cluster_id, cluster_genes in clusters.items():
            cluster_genes_in_network = cluster_genes & set(G.nodes())
            
            if len(cluster_genes_in_network) == 0:
                print(f"Warning: Cluster {cluster_id} has no genes in network")
                continue
            
            cluster_stats[cluster_id] = {}
            
            for pathway_name, pathway_genes in pathway_genes_dict.items():
                if len(pathway_genes) == 0:
                    continue
                
                # Compute separation
                sep = compute_network_separation(
                    G, cluster_genes_in_network, pathway_genes, max_distance=5
                )
                
                # Compute distances from cluster to pathway
                cluster_distances = []
                for gene in cluster_genes_in_network:
                    if gene not in G:
                        continue
                    
                    # BFS to find closest pathway gene
                    queue = deque([(gene, 0)])
                    visited = {gene}
                    found = False
                    
                    while queue and not found:
                        current, dist = queue.popleft()
                        if current in pathway_genes and dist > 0:
                            cluster_distances.append(dist)
                            found = True
                            break
                        if dist < 5:
                            for neighbor in G.neighbors(current):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    queue.append((neighbor, dist + 1))
                
                # Store results
                result = {
                    'cluster_id': cluster_id,
                    'pathway_name': pathway_name,
                    'cluster_size': len(cluster_genes_in_network),
                    'pathway_size': len(pathway_genes),
                    
                    # Separation metrics
                    'separation_score': sep.get('separation_score'),
                    'single_linkage_separation': sep.get('single_linkage_separation'),
                    'complete_linkage_separation': sep.get('complete_linkage_separation'),
                    'interpretation': sep.get('interpretation'),
                    
                    # Within-cluster/pathway distances
                    'mean_within_cluster': sep.get('mean_within_module_1'),
                    'mean_within_pathway': sep.get('mean_within_module_2'),
                    
                    # Between distances
                    'mean_between': sep.get('mean_between_modules'),
                    'min_between': sep.get('min_between_distance'),
                    'max_between': sep.get('max_between_distance'),
                    'median_between': sep.get('median_between_distance'),
                    
                    # Linkage metrics
                    'single_linkage_between': sep.get('single_linkage_between_modules'),
                    'complete_linkage_between': sep.get('complete_linkage_between_modules'),
                    'centroid_linkage_between': sep.get('centroid_linkage_between_modules'),
                    
                    # Coverage
                    'overlap_size': sep.get('overlap_size'),
                }
                
                # Add distance statistics
                if cluster_distances:
                    result['mean_cluster_to_pathway'] = float(np.mean(cluster_distances))
                    result['median_cluster_to_pathway'] = float(np.median(cluster_distances))
                    result['min_cluster_to_pathway'] = float(np.min(cluster_distances))
                    result['max_cluster_to_pathway'] = float(np.max(cluster_distances))
                    result['std_cluster_to_pathway'] = float(np.std(cluster_distances))
                    result['genes_in_cluster_reaching_pathway'] = len(cluster_distances)
                    result['percent_cluster_coverage'] = 100 * len(cluster_distances) / len(cluster_genes_in_network)
                else:
                    result['mean_cluster_to_pathway'] = None
                    result['median_cluster_to_pathway'] = None
                    result['min_cluster_to_pathway'] = None
                    result['max_cluster_to_pathway'] = None
                    result['std_cluster_to_pathway'] = None
                    result['genes_in_cluster_reaching_pathway'] = 0
                    result['percent_cluster_coverage'] = 0
                
                cluster_pathway_results.append(result)
        
        # Save cluster-to-pathway results
        if cluster_pathway_results:
            cluster_pathway_df = pd.DataFrame(cluster_pathway_results)
            cluster_pathway_output = os.path.join(args.output_dir, "cluster_pathway_separation.csv")
            cluster_pathway_df.to_csv(cluster_pathway_output, index=False)
            print(f"Computed {len(cluster_pathway_results)} cluster-pathway comparisons")
            print(f"Saved to {cluster_pathway_output}")
            
            # Show summary per cluster
            print("\nCluster-Pathway Summary:")
            for cluster_id in clusters.keys():
                cluster_data = cluster_pathway_df[cluster_pathway_df['cluster_id'] == cluster_id]
                involved_pathways = cluster_data[cluster_data['separation_score'] < 0]
                if len(involved_pathways) > 0:
                    print(f"\n  Cluster {cluster_id}: {len(clusters[cluster_id])} genes")
                    print(f"    Involved pathways ({len(involved_pathways)}):")
                    for _, row in involved_pathways.nsmallest(3, 'separation_score').iterrows():
                        print(f"      - {row['pathway_name']}: s={row['separation_score']:.3f}, overlap={row['overlap_size']}")
    else:
        cluster_pathway_df = pd.DataFrame()
        print("\nNo cluster file provided (--clusters argument)")
        print("Skipping cluster-to-pathway analysis")
    
    # Save pickle for further analysis
    results = {
        'distances': distances,
        'closest_pathway_genes': closest_pathway_genes,
        'statistics': stats,
        'separation_analysis': separation_results,
        'pairwise_pathway_separation': pairwise_results,
        'cluster_pathway_separation': cluster_pathway_results,
        'disease_module_size': len(disease_module),
        'pathways_analyzed': list(pathway_genes_dict.keys()),
        'clusters_analyzed': list(clusters.keys()) if args.clusters else []
    }
    
    pkl_output = os.path.join(args.output_dir, "pathway_distance_results.pkl")
    with open(pkl_output, 'wb') as f:
        pickle.dump(results, f, protocol=4)
    print(f"Saved detailed results to {pkl_output}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())