#!/usr/bin/env python3
"""
Multiplex Disease Module HiDeF Clustering

Apply HiDeF clustering to disease modules from multiplex RWR analysis.
Can analyze either layer-specific modules or aggregated multiplex module.
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import json
import time
import sys
from pathlib import Path

# Community detection
try:
    import community.community_louvain as community_louvain
except ImportError:
    try:
        import community as community_louvain
    except ImportError:
        print("ERROR: python-louvain not installed")
        print("Install with: pip install python-louvain --user")
        sys.exit(1)

def load_multiplex_disease_module(disease_module_file):
    """Load disease module from multiplex RWR"""
    print(f"Loading disease module from {disease_module_file}...")
    
    module_df = pd.read_csv(disease_module_file)
    print(f"  Loaded {len(module_df)} genes")
    print(f"  Seed genes: {sum(module_df['Is_Seed'])}")
    
    return module_df

def load_multiplex_layers(multiplex_dir):
    """Load multiplex layer networks"""
    layers_dir = os.path.join(multiplex_dir, 'layers')
    
    layers = {}
    for layer_file in os.listdir(layers_dir):
        if layer_file.endswith('_layer.pkl'):
            layer_name = layer_file.replace('_layer.pkl', '')
            with open(os.path.join(layers_dir, layer_file), 'rb') as f:
                layers[layer_name] = pickle.load(f)
            print(f"  Loaded {layer_name}")
    
    return layers

def create_aggregated_disease_subgraph(layers, disease_module_genes):
    """
    Create aggregated disease subgraph by combining all layers
    Edge exists if it exists in ANY layer, weighted by number of layers
    """
    print("\nCreating aggregated disease subgraph...")
    
    # Initialize graph with all disease module genes
    G_agg = nx.Graph()
    G_agg.add_nodes_from(disease_module_genes)
    
    # Track edge weights (number of layers supporting each edge)
    edge_support = {}
    
    for layer_name, layer_graph in layers.items():
        # Get edges within disease module for this layer
        layer_module_genes = [g for g in disease_module_genes if g in layer_graph.nodes()]
        layer_subgraph = layer_graph.subgraph(layer_module_genes)
        
        for u, v, data in layer_subgraph.edges(data=True):
            edge = tuple(sorted([u, v]))
            if edge not in edge_support:
                edge_support[edge] = {
                    'count': 0,
                    'weights': [],
                    'layers': []
                }
            edge_support[edge]['count'] += 1
            edge_support[edge]['weights'].append(data.get('weight', 1.0))
            edge_support[edge]['layers'].append(layer_name)
    
    # Add edges to aggregated graph
    for (u, v), support_data in edge_support.items():
        # Weight by layer support and average edge weight
        weight = support_data['count'] * np.mean(support_data['weights'])
        G_agg.add_edge(u, v, 
                      weight=weight,
                      layer_count=support_data['count'],
                      avg_layer_weight=np.mean(support_data['weights']),
                      layers=support_data['layers'])
    
    print(f"  Aggregated graph: {G_agg.number_of_nodes()} nodes, {G_agg.number_of_edges()} edges")
    
    # Take largest connected component
    if not nx.is_connected(G_agg):
        components = list(nx.connected_components(G_agg))
        largest = max(components, key=len)
        G_agg = G_agg.subgraph(largest).copy()
        print(f"  Largest component: {G_agg.number_of_nodes()} nodes, {G_agg.number_of_edges()} edges")
    
    return G_agg

def detect_communities_at_resolution(graph, resolution):
    """Detect communities using Louvain with consensus"""
    if graph.number_of_nodes() == 0:
        return {}
    
    # Run multiple times for stability
    num_runs = 10
    all_partitions = []
    
    for seed in range(num_runs):
        partition = community_louvain.best_partition(
            graph, resolution=resolution, random_state=42 + seed
        )
        all_partitions.append(partition)
    
    # Simple consensus: use most common partition structure
    # (For production, use co-occurrence matrix approach)
    return all_partitions[0]  # Simplified for now

def run_hidef_clustering(disease_subgraph, min_resolution=0.1, max_resolution=2.0,
                        num_steps=20, similarity_threshold=0.75, min_persistence=5):
    """Run HiDeF clustering on disease subgraph"""
    print("\nRunning HiDeF clustering...")
    
    resolution_values = np.linspace(min_resolution, max_resolution, num_steps)
    communities_per_resolution = []
    
    # Multi-resolution community detection
    print("  Detecting communities at multiple resolutions...")
    for resolution in resolution_values:
        communities = detect_communities_at_resolution(disease_subgraph, resolution)
        
        if not communities:
            continue
        
        # Convert to matrix format
        unique_comms = sorted(set(communities.values()))
        node_list = sorted(list(disease_subgraph.nodes()))
        n_nodes = len(node_list)
        
        comm_matrix = np.zeros((len(unique_comms), n_nodes))
        for i, comm_id in enumerate(unique_comms):
            for j, node in enumerate(node_list):
                if communities.get(node, -1) == comm_id:
                    comm_matrix[i, j] = 1
        
        # Filter small communities
        valid_rows = []
        for i in range(comm_matrix.shape[0]):
            if np.sum(comm_matrix[i, :]) >= 2:
                valid_rows.append(comm_matrix[i, :])
        
        if valid_rows:
            valid_matrix = np.vstack(valid_rows)
            communities_per_resolution.append({
                'resolution': resolution,
                'communities': communities,
                'matrix': valid_matrix,
                'node_list': node_list,
                'n_communities': len(valid_rows)
            })
    
    if not communities_per_resolution:
        print("  WARNING: No communities found!")
        return {'persistent_communities': {}, 'communities_per_resolution': []}
    
    # Build cluster graph
    print("  Building cluster graph...")
    cluster_graph = nx.Graph()
    cluster_info = {}
    cluster_id = 0
    
    for res_data in communities_per_resolution:
        matrix = res_data['matrix']
        node_list = res_data['node_list']
        
        for i in range(matrix.shape[0]):
            members = [node_list[j] for j in range(len(node_list)) if matrix[i, j] == 1]
            cluster_info[cluster_id] = {
                'resolution': res_data['resolution'],
                'members': members,
                'binary': matrix[i, :],
                'size': len(members)
            }
            cluster_graph.add_node(cluster_id)
            cluster_id += 1
    
    # Add edges based on similarity
    print("  Computing community similarities...")
    cluster_ids = sorted(list(cluster_graph.nodes()))
    
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            id1, id2 = cluster_ids[i], cluster_ids[j]
            binary1 = cluster_info[id1]['binary']
            binary2 = cluster_info[id2]['binary']
            
            # Jaccard similarity
            intersection = np.sum(binary1 * binary2)
            union = np.sum((binary1 + binary2) > 0)
            
            if union > 0:
                jaccard = intersection / union
                if jaccard >= similarity_threshold:
                    cluster_graph.add_edge(id1, id2, weight=jaccard)
    
    # Find persistent communities
    print("  Finding persistent communities...")
    persistent_communities = {}
    components = list(nx.connected_components(cluster_graph))
    
    valid_idx = 0
    for component in components:
        if len(component) >= min_persistence:
            all_members = set()
            for cluster_id in component:
                all_members.update(cluster_info[cluster_id]['members'])
            
            persistent_communities[valid_idx] = {
                'members': sorted(list(all_members)),
                'persistence': len(component),
                'size': len(all_members),
                'contributing_clusters': list(component)
            }
            valid_idx += 1
    
    print(f"  Found {len(persistent_communities)} persistent communities")
    
    return {
        'persistent_communities': persistent_communities,
        'communities_per_resolution': communities_per_resolution,
        'cluster_graph': cluster_graph,
        'cluster_info': cluster_info
    }

def save_results(hidef_results, disease_module_df, output_dir, plots_dir):
    """Save clustering results"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    persistent_comms = hidef_results['persistent_communities']
    
    # Save communities
    comm_rows = []
    seed_genes = set(disease_module_df[disease_module_df['Is_Seed']]['Gene'])
    
    for comm_id, comm_data in persistent_comms.items():
        for gene in comm_data['members']:
            comm_rows.append({
                'Community_ID': f'MC{comm_id}',  # MC = Multiplex Community
                'Gene': gene,
                'Is_Seed': gene in seed_genes,
                'Community_Size': comm_data['size'],
                'Persistence': comm_data['persistence']
            })
    
    if comm_rows:
        pd.DataFrame(comm_rows).to_csv(
            os.path.join(output_dir, 'multiplex_disease_communities.csv'),
            index=False
        )
    
    # Save statistics
    stats_rows = []
    for comm_id, comm_data in persistent_comms.items():
        members = comm_data['members']
        seed_count = sum(1 for g in members if g in seed_genes)
        
        stats_rows.append({
            'Community_ID': f'MC{comm_id}',
            'Size': comm_data['size'],
            'Persistence': comm_data['persistence'],
            'Seed_Count': seed_count,
            'Neighbor_Count': len(members) - seed_count,
            'Seed_Percentage': seed_count / len(members) * 100 if members else 0
        })
    
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(
            os.path.join(output_dir, 'multiplex_community_statistics.csv'),
            index=False
        )
    
    # Save complete results
    with open(os.path.join(output_dir, 'multiplex_hidef_results.pkl'), 'wb') as f:
        pickle.dump(hidef_results, f, protocol=4)
    
    print(f"\nResults saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Multiplex HiDeF clustering')
    parser.add_argument('--multiplex-dir', type=str, required=True,
                      help='Multiplex interactome directory')
    parser.add_argument('--disease-module', type=str, required=True,
                      help='Disease module CSV file')
    parser.add_argument('--output-dir', type=str, default='output/multiplex/hidef',
                      help='Output directory')
    parser.add_argument('--plots-dir', type=str, default='plots/multiplex/hidef',
                      help='Plots directory')
    parser.add_argument('--min-resolution', type=float, default=0.1)
    parser.add_argument('--max-resolution', type=float, default=2.0)
    parser.add_argument('--resolution-steps', type=int, default=20)
    parser.add_argument('--min-persistence', type=int, default=5)
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTIPLEX DISEASE MODULE HIDEF CLUSTERING")
    print("="*80)
    
    # Load disease module
    disease_module_df = load_multiplex_disease_module(args.disease_module)
    disease_genes = disease_module_df['Gene'].tolist()
    
    # Load multiplex layers
    print("\nLoading multiplex layers...")
    layers = load_multiplex_layers(args.multiplex_dir)
    
    # Create aggregated subgraph
    disease_subgraph = create_aggregated_disease_subgraph(layers, disease_genes)
    
    # Run HiDeF
    hidef_results = run_hidef_clustering(
        disease_subgraph,
        min_resolution=args.min_resolution,
        max_resolution=args.max_resolution,
        num_steps=args.resolution_steps,
        min_persistence=args.min_persistence
    )
    
    # Save results
    save_results(hidef_results, disease_module_df, args.output_dir, args.plots_dir)
    
    print("\n" + "="*80)
    print("CLUSTERING COMPLETE")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())