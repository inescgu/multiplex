#!/usr/bin/env python3
"""
Comprehensive Comparison Analysis of Three Multiplex Networks

Compares:
1. Tabula Sapiens multiplex (full - from 2024 release)
2. 2025 hDRG Atlas (2025)
3. Pain subset of Tabula Sapiens multiplex (at the moment Manuela's 2022 subset, needs to be updated with 2024 subset)

Generates:
- Comparative statistics (CSV)
- Seed gene analysis across all three
- Network visualizations
- Comparative plots
"""

import os
import sys
import argparse
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_multiplex_layers(layers_dir):
    """Load all layer networks from directory"""
    layers = {}
    layer_files = sorted(Path(layers_dir).glob('*_layer.pkl'))
  
    for layer_file in layer_files:
        cell_type = layer_file.stem.replace('_layer', '')
        try:
            with open(layer_file, 'rb') as f:
                G = pickle.load(f)
            if G.number_of_nodes() > 0:
                layers[cell_type] = G
        except Exception as e:
            print(f"Error loading {cell_type}: {str(e)}")
  
    return layers

def load_seed_genes(seed_genes_file):
    """Load seed genes from CSV file"""
    if not os.path.exists(seed_genes_file):
        return set()
  
    try:
        seed_df = pd.read_csv(seed_genes_file)
      
        # Try to find gene column
        gene_col = None
        for col in ['gene', 'Gene', 'gene_id', 'Gene_ID', 'symbol', 'Symbol', 'SYMBOL']:
            if col in seed_df.columns:
                gene_col = col
                break
      
        if gene_col is None:
            gene_col = seed_df.columns[0]
      
        seed_genes = set(seed_df[gene_col].dropna().astype(str))
        seed_genes = set([g.strip().upper() for g in seed_genes])
      
        return seed_genes
    except Exception as e:
        print(f"Error loading seed genes: {str(e)}")
        return set()

def analyze_multiplex(layers, seed_genes=None, name="Multiplex"):
    """
    Analyze a multiplex network
  
    Returns:
    - Dictionary with comprehensive statistics
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {name}")
    print(f"{'='*80}")
  
    analysis = {
        'name': name,
        'n_layers': len(layers),
        'layer_stats': [],
        'multiplex_stats': {},
        'seed_gene_stats': {}
    }
  
    # ============================================================================
    # PER-LAYER STATISTICS
    # ============================================================================
    print(f"\nPer-layer statistics:")
  
    all_genes = set()
    all_edges = set()
  
    for cell_type in sorted(layers.keys()):
        layer = layers[cell_type]
      
        n_nodes = layer.number_of_nodes()
        n_edges = layer.number_of_edges()
        all_genes.update(layer.nodes())
        all_edges.update((min(u, v), max(u, v)) for u, v in layer.edges())
      
        # Network properties
        density = nx.density(layer) if n_nodes > 0 else 0
        avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
      
        # Connected components
        n_components = nx.number_connected_components(layer)
        largest_cc_size = len(max(nx.connected_components(layer), default=set()))
        largest_cc_pct = largest_cc_size / n_nodes * 100 if n_nodes > 0 else 0
      
        # Seed gene analysis
        seeds_in_layer = seed_genes.intersection(layer.nodes()) if seed_genes else set()
        seed_pct_of_layer = len(seeds_in_layer) / n_nodes * 100 if n_nodes > 0 else 0
        seed_pct_of_total = len(seeds_in_layer) / len(seed_genes) * 100 if seed_genes else 0
      
        # Seed gene connectivity
        seed_genes_connected = False
        n_seed_components = 0
        if seeds_in_layer and len(seeds_in_layer) > 1:
            seed_subgraph = layer.subgraph(seeds_in_layer)
            n_seed_components = nx.number_connected_components(seed_subgraph)
            seed_genes_connected = (n_seed_components == 1)
        elif len(seeds_in_layer) == 1:
            seed_genes_connected = True
            n_seed_components = 1
      
        # Average shortest path in largest CC
        try:
            largest_cc = max(nx.connected_components(layer), default=set())
            if len(largest_cc) > 1:
                subgraph = layer.subgraph(largest_cc)
                avg_shortest_path = nx.average_shortest_path_length(subgraph)
            else:
                avg_shortest_path = 0
        except:
            avg_shortest_path = 0
      
        stat = {
            'cell_type': cell_type,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'avg_degree': avg_degree,
            'n_components': n_components,
            'largest_cc_size': largest_cc_size,
            'largest_cc_pct': largest_cc_pct,
            'avg_shortest_path': avg_shortest_path,
            'n_seed_genes': len(seeds_in_layer),
            'seed_pct_of_layer': seed_pct_of_layer,
            'seed_pct_of_total': seed_pct_of_total,
            'seed_genes_connected': seed_genes_connected,
            'n_seed_components': n_seed_components
        }
      
        analysis['layer_stats'].append(stat)
      
        print(f"  {cell_type}:")
        print(f"    Nodes: {n_nodes} | Edges: {n_edges} | Density: {density:.4f} | Avg degree: {avg_degree:.2f}")
        print(f"    Components: {n_components} | Largest CC: {largest_cc_size} ({largest_cc_pct:.1f}%)")
        if seed_genes:
            print(f"    Seed genes: {len(seeds_in_layer)} ({seed_pct_of_layer:.1f}% of layer, {seed_pct_of_total:.1f}% of total)")
            print(f"    Seed genes connected: {seed_genes_connected} ({n_seed_components} component(s))")
  
    # ============================================================================
    # MULTIPLEX-LEVEL STATISTICS
    # ============================================================================
    print(f"\nMultiplex-level statistics:")
  
    analysis['multiplex_stats'] = {
        'n_total_genes': len(all_genes),
        'n_total_edges': len(all_edges),
        'avg_genes_per_layer': np.mean([s['n_nodes'] for s in analysis['layer_stats']]),
        'avg_edges_per_layer': np.mean([s['n_edges'] for s in analysis['layer_stats']]),
        'avg_density': np.mean([s['density'] for s in analysis['layer_stats']]),
        'avg_degree': np.mean([s['avg_degree'] for s in analysis['layer_stats']]),
        'avg_components': np.mean([s['n_components'] for s in analysis['layer_stats']]),
        'avg_largest_cc_pct': np.mean([s['largest_cc_pct'] for s in analysis['layer_stats']]),
        'avg_shortest_path': np.mean([s['avg_shortest_path'] for s in analysis['layer_stats'] if s['avg_shortest_path'] > 0])
    }
  
    print(f"  Total unique genes: {analysis['multiplex_stats']['n_total_genes']}")
    print(f"  Total edges: {analysis['multiplex_stats']['n_total_edges']}")
    print(f"  Avg genes per layer: {analysis['multiplex_stats']['avg_genes_per_layer']:.1f}")
    print(f"  Avg edges per layer: {analysis['multiplex_stats']['avg_edges_per_layer']:.1f}")
    print(f"  Avg density: {analysis['multiplex_stats']['avg_density']:.4f}")
    print(f"  Avg degree: {analysis['multiplex_stats']['avg_degree']:.2f}")
    print(f"  Avg components per layer: {analysis['multiplex_stats']['avg_components']:.2f}")
    print(f"  Avg largest CC: {analysis['multiplex_stats']['avg_largest_cc_pct']:.1f}%")
  
    # ============================================================================
    # SEED GENE STATISTICS
    # ============================================================================
    if seed_genes:
        print(f"\nSeed gene statistics:")
      
        seeds_in_multiplex = seed_genes.intersection(all_genes)
        seeds_pct = len(seeds_in_multiplex) / len(seed_genes) * 100
      
        print(f"  Total seed genes: {len(seed_genes)}")
        print(f"  Seed genes in multiplex: {len(seeds_in_multiplex)} ({seeds_pct:.1f}%)")
      
        # Seed genes in all layers
        seeds_in_all = set(layers[list(layers.keys())[0]].nodes()).intersection(seed_genes)
        for cell_type in list(layers.keys())[1:]:
            seeds_in_all = seeds_in_all.intersection(set(layers[cell_type].nodes()).intersection(seed_genes))
      
        print(f"  Seed genes in ALL layers: {len(seeds_in_all)}")
      
        # Seed gene spread
        seed_layer_counts = defaultdict(int)
        for seed in seeds_in_multiplex:
            for cell_type, layer in layers.items():
                if seed in layer.nodes():
                    seed_layer_counts[seed] += 1
      
        if seed_layer_counts:
            avg_layers_per_seed = np.mean(list(seed_layer_counts.values()))
            max_layers_per_seed = max(seed_layer_counts.values())
            min_layers_per_seed = min(seed_layer_counts.values())
          
            print(f"  Seed gene spread across layers:")
            print(f"    Avg layers per seed: {avg_layers_per_seed:.2f}")
            print(f"    Min layers per seed: {min_layers_per_seed}")
            print(f"    Max layers per seed: {max_layers_per_seed}")
          
            analysis['seed_gene_stats'] = {
                'total_seed_genes': len(seed_genes),
                'seed_genes_in_multiplex': len(seeds_in_multiplex),
                'seed_gene_recovery_pct': seeds_pct,
                'seed_genes_in_all_layers': len(seeds_in_all),
                'avg_layers_per_seed': avg_layers_per_seed,
                'min_layers_per_seed': min_layers_per_seed,
                'max_layers_per_seed': max_layers_per_seed,
                'seed_layer_distribution': dict(seed_layer_counts)
            }
  
    return analysis

def create_comparison_dataframe(analyses):
    """Create a comparison dataframe from multiple analyses"""
    comparison_data = []
  
    for analysis in analyses:
        for layer_stat in analysis['layer_stats']:
            row = {
                'Multiplex': analysis['name'],
                'Cell Type': layer_stat['cell_type'],
                'Nodes': layer_stat['n_nodes'],
                'Edges': layer_stat['n_edges'],
                'Density': layer_stat['density'],
                'Avg Degree': layer_stat['avg_degree'],
                'Components': layer_stat['n_components'],
                'Largest CC %': layer_stat['largest_cc_pct'],
                'Avg Shortest Path': layer_stat['avg_shortest_path'],
                'Seed Genes': layer_stat['n_seed_genes'],
                'Seed % of Layer': layer_stat['seed_pct_of_layer'],
                'Seed % of Total': layer_stat['seed_pct_of_total'],
                'Seed Connected': layer_stat['seed_genes_connected'],
                'Seed Components': layer_stat['n_seed_components']
            }
            comparison_data.append(row)
  
    return pd.DataFrame(comparison_data)

def create_multiplex_summary(analyses):
    """Create a summary of multiplex-level statistics"""
    summary_data = []
  
    for analysis in analyses:
        row = {
            'Multiplex': analysis['name'],
            'N Layers': analysis['n_layers'],
            'Total Genes': analysis['multiplex_stats']['n_total_genes'],
            'Total Edges': analysis['multiplex_stats']['n_total_edges'],
            'Avg Genes/Layer': analysis['multiplex_stats']['avg_genes_per_layer'],
            'Avg Edges/Layer': analysis['multiplex_stats']['avg_edges_per_layer'],
            'Avg Density': analysis['multiplex_stats']['avg_density'],
            'Avg Degree': analysis['multiplex_stats']['avg_degree'],
            'Avg Components': analysis['multiplex_stats']['avg_components'],
            'Avg Largest CC %': analysis['multiplex_stats']['avg_largest_cc_pct'],
            'Avg Shortest Path': analysis['multiplex_stats']['avg_shortest_path']
        }
      
        if analysis['seed_gene_stats']:
            row['Seed Gene Recovery %'] = analysis['seed_gene_stats']['seed_gene_recovery_pct']
            row['Avg Layers/Seed'] = analysis['seed_gene_stats']['avg_layers_per_seed']
      
        summary_data.append(row)
  
    return pd.DataFrame(summary_data)

def plot_density_comparison(comparison_df, output_dir):
    """Plot density comparison across multiplexes"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  
    multiplexes = comparison_df['Multiplex'].unique()
  
    for idx, multiplex in enumerate(multiplexes):
        data = comparison_df[comparison_df['Multiplex'] == multiplex]['Density']
        axes[idx].hist(data, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].set_title(f'{multiplex}\nDensity Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Density')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.4f}')
        axes[idx].legend()
  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'density_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: density_comparison.png")
    plt.close()

def plot_degree_comparison(comparison_df, output_dir):
    """Plot average degree comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  
    multiplexes = comparison_df['Multiplex'].unique()
  
    for idx, multiplex in enumerate(multiplexes):
        data = comparison_df[comparison_df['Multiplex'] == multiplex]['Avg Degree']
        axes[idx].hist(data, bins=15, alpha=0.7, color='coral', edgecolor='black')
        axes[idx].set_title(f'{multiplex}\nAverage Degree Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Average Degree')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        axes[idx].legend()
  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'degree_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: degree_comparison.png")
    plt.close()

def plot_seed_gene_spread(comparison_df, output_dir):
    """Plot seed gene spread across layers"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  
    multiplexes = comparison_df['Multiplex'].unique()
  
    for idx, multiplex in enumerate(multiplexes):
        data = comparison_df[comparison_df['Multiplex'] == multiplex]['Seed % of Layer']
        axes[idx].hist(data, bins=15, alpha=0.7, color='mediumseagreen', edgecolor='black')
        axes[idx].set_title(f'{multiplex}\nSeed Gene % of Layer', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Seed Gene % of Layer')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}%')
        axes[idx].legend()
  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seed_gene_spread.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: seed_gene_spread.png")
    plt.close()

def plot_seed_connectivity(comparison_df, output_dir):
    """Plot seed gene connectivity"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  
    multiplexes = comparison_df['Multiplex'].unique()
  
    for idx, multiplex in enumerate(multiplexes):
        data = comparison_df[comparison_df['Multiplex'] == multiplex]
        connected = data['Seed Connected'].sum()
        disconnected = len(data) - connected
      
        axes[idx].bar(['Connected', 'Disconnected'], [connected, disconnected], 
                     color=['green', 'red'], alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{multiplex}\nSeed Gene Connectivity', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Number of Layers')
      
        # Add percentages
        total = connected + disconnected
        axes[idx].text(0, connected + 1, f'{connected/total*100:.1f}%', ha='center', fontweight='bold')
        axes[idx].text(1, disconnected + 1, f'{disconnected/total*100:.1f}%', ha='center', fontweight='bold')
  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seed_connectivity.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: seed_connectivity.png")
    plt.close()

def plot_components_comparison(comparison_df, output_dir):
    """Plot number of components comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  
    multiplexes = comparison_df['Multiplex'].unique()
  
    for idx, multiplex in enumerate(multiplexes):
        data = comparison_df[comparison_df['Multiplex'] == multiplex]['Components']
        axes[idx].hist(data, bins=range(int(data.min()), int(data.max())+2), 
                      alpha=0.7, color='mediumpurple', edgecolor='black')
        axes[idx].set_title(f'{multiplex}\nNumber of Components', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Number of Components')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        axes[idx].legend()
  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'components_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: components_comparison.png")
    plt.close()

def plot_largest_cc_comparison(comparison_df, output_dir):
    """Plot largest connected component size comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  
    multiplexes = comparison_df['Multiplex'].unique()
  
    for idx, multiplex in enumerate(multiplexes):
        data = comparison_df[comparison_df['Multiplex'] == multiplex]['Largest CC %']
        axes[idx].hist(data, bins=15, alpha=0.7, color='lightsalmon', edgecolor='black')
        axes[idx].set_title(f'{multiplex}\nLargest Connected Component %', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Largest CC %')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}%')
        axes[idx].legend()
  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'largest_cc_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: largest_cc_comparison.png")
    plt.close()

def plot_multiplex_summary(summary_df, output_dir):
    """Create a comprehensive multiplex summary visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
  
    multiplexes = summary_df['Multiplex'].values
  
    # Total Genes
    axes[0, 0].bar(multiplexes, summary_df['Total Genes'], color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Total Unique Genes', fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(summary_df['Total Genes']):
        axes[0, 0].text(i, v + 100, str(int(v)), ha='center', fontweight='bold')
  
    # Total Edges
    axes[0, 1].bar(multiplexes, summary_df['Total Edges'], color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Total Edges', fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    for i, v in enumerate(summary_df['Total Edges']):
        axes[0, 1].text(i, v + 100, str(int(v)), ha='center', fontweight='bold')
  
    # Avg Density
    axes[0, 2].bar(multiplexes, summary_df['Avg Density'], color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Average Density', fontweight='bold')
    axes[0, 2].set_ylabel('Density')
    for i, v in enumerate(summary_df['Avg Density']):
        axes[0, 2].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold', fontsize=9)
  
    # Avg Degree
    axes[1, 0].bar(multiplexes, summary_df['Avg Degree'], color='mediumpurple', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Average Degree', fontweight='bold')
    axes[1, 0].set_ylabel('Degree')
    for i, v in enumerate(summary_df['Avg Degree']):
        axes[1, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
  
    # Avg Largest CC %
    axes[1, 1].bar(multiplexes, summary_df['Avg Largest CC %'], color='lightsalmon', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Average Largest CC %', fontweight='bold')
    axes[1, 1].set_ylabel('Percentage')
    for i, v in enumerate(summary_df['Avg Largest CC %']):
        axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
  
    # Seed Gene Recovery (if available)
    if 'Seed Gene Recovery %' in summary_df.columns:
        axes[1, 2].bar(multiplexes, summary_df['Seed Gene Recovery %'], color='gold', alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Seed Gene Recovery %', fontweight='bold')
        axes[1, 2].set_ylabel('Percentage')
        for i, v in enumerate(summary_df['Seed Gene Recovery %']):
            axes[1, 2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    else:
        axes[1, 2].axis('off')
  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiplex_summary.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: multiplex_summary.png")
    plt.close()

def plot_seed_components_comparison(comparison_df, output_dir):
    """Plot seed gene components comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  
    multiplexes = comparison_df['Multiplex'].unique()
  
    for idx, multiplex in enumerate(multiplexes):
        data = comparison_df[comparison_df['Multiplex'] == multiplex]['Seed Components']
        # Filter out zeros (no seed genes)
        data = data[data > 0]
      
        if len(data) > 0:
            axes[idx].hist(data, bins=range(int(data.min()), int(data.max())+2), 
                          alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'{multiplex}\nSeed Gene Components', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Number of Components')
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
            axes[idx].legend()
        else:
            axes[idx].text(0.5, 0.5, 'No seed genes', ha='center', va='center', transform=axes[idx].transAxes)
  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seed_components_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: seed_components_comparison.png")
    plt.close()
    
def plot_interlayer_jaccard(layers_dict, name, output_dir):
    """Heatmap of Jaccard similarity between all layer gene sets"""
    cell_types = sorted(layers_dict.keys())
    n = len(cell_types)
    matrix = np.zeros((n, n))

    for i, ct1 in enumerate(cell_types):
        for j, ct2 in enumerate(cell_types):
            s1 = set(layers_dict[ct1].nodes())
            s2 = set(layers_dict[ct2].nodes())
            union = s1 | s2
            matrix[i, j] = len(s1 & s2) / len(union) if union else 0

    fig, ax = plt.subplots(figsize=(max(8, n//2), max(6, n//2)))
    sns.heatmap(matrix, xticklabels=cell_types, yticklabels=cell_types,
                cmap='YlOrRd', ax=ax, vmin=0, vmax=1)
    ax.set_title(f'{name} — Inter-Layer Jaccard Similarity', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'jaccard_{name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: jaccard_{name}.png")

def plot_gene_ubiquity(layers_dict, seed_genes, name, output_dir):
    """Distribution of how many layers each gene appears in"""
    gene_counts = defaultdict(int)
    for layer in layers_dict.values():
        for gene in layer.nodes():
            gene_counts[gene] += 1

    counts = list(gene_counts.values())
    n_layers = len(layers_dict)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(counts, bins=range(1, n_layers + 2), alpha=0.7, color='steelblue',
            edgecolor='black', align='left')
    ax.set_xlabel('Number of Layers Gene Appears In')
    ax.set_ylabel('Number of Genes')
    ax.set_title(f'{name} — Gene Ubiquity Distribution', fontweight='bold')
    ax.set_xticks(range(1, n_layers + 1))

    # Overlay seed gene distribution
    if seed_genes:
        seed_counts = [gene_counts[g] for g in seed_genes if g in gene_counts]
        ax.hist(seed_counts, bins=range(1, n_layers + 2), alpha=0.5,
                color='red', edgecolor='black', align='left', label='Seed genes')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gene_ubiquity_{name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
def compute_clustering_stats(layers):
    """Add clustering coefficient to layer stats"""
    clustering_data = []
    for cell_type, G in layers.items():
        avg_clustering = nx.average_clustering(G)
        try:
            transitivity = nx.transitivity(G)
        except:
            transitivity = 0
        clustering_data.append({
            'cell_type': cell_type,
            'avg_clustering': avg_clustering,
            'transitivity': transitivity
        })
    return pd.DataFrame(clustering_data)

def plot_clustering_comparison(analyses_layers, output_dir):
    """Boxplot of clustering coefficients across multiplexes"""
    records = []
    for name, layers in analyses_layers.items():
        for cell_type, G in layers.items():
            records.append({
                'Multiplex': name,
                'Avg Clustering': nx.average_clustering(G)
            })
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Multiplex', y='Avg Clustering', palette='Set2', ax=ax)
    sns.stripplot(data=df, x='Multiplex', y='Avg Clustering', color='black',
                  alpha=0.4, size=3, ax=ax)
    ax.set_title('Clustering Coefficient Distribution Across Multiplexes', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
def analyze_hub_genes(layers, seed_genes, name, top_n=20):
    """Find top hub genes by aggregate degree across all layers"""
    degree_sum = defaultdict(int)
    degree_count = defaultdict(int)

    for G in layers.values():
        for node, deg in G.degree():
            degree_sum[node] += deg
            degree_count[node] += 1

    hub_df = pd.DataFrame([
        {
            'gene': gene,
            'total_degree': degree_sum[gene],
            'layers_present': degree_count[gene],
            'avg_degree': degree_sum[gene] / degree_count[gene],
            'is_seed': gene in seed_genes
        }
        for gene in degree_sum
    ]).sort_values('total_degree', ascending=False).head(top_n)

    print(f"\n[{name}] Top {top_n} Hub Genes:")
    print(hub_df.to_string(index=False))
    return hub_df

def plot_hub_genes(hub_dfs, output_dir):
    """Bar chart of top hub genes colored by seed status"""
    fig, axes = plt.subplots(1, len(hub_dfs), figsize=(18, 6), sharey=False)
    for ax, (name, df) in zip(axes, hub_dfs.items()):
        colors = ['crimson' if s else 'steelblue' for s in df['is_seed']]
        ax.barh(df['gene'], df['total_degree'], color=colors, edgecolor='black', alpha=0.8)
        ax.invert_yaxis()
        ax.set_title(f'{name}\nTop Hub Genes', fontweight='bold')
        ax.set_xlabel('Total Degree (all layers)')
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='crimson', label='Seed gene'),
                            Patch(color='steelblue', label='Non-seed')], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hub_genes.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_edge_weight_distribution(layers_dict, name, output_dir):
    """Distribution of edge weights across all layers"""
    all_weights = []
    for G in layers_dict.values():
        for u, v, d in G.edges(data=True):
            w = d.get('weight', 1.0)
            all_weights.append(w)

    if not all_weights:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_weights, bins=50, alpha=0.7, color='teal', edgecolor='black', log=True)
    ax.set_xlabel('Edge Weight')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_title(f'{name} — Edge Weight Distribution', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'edge_weights_{name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_gene_overlap_venn(analyses_layers, output_dir):
    """Venn-style overlap of gene sets across multiplexes (text summary + bar)"""
    gene_sets = {name: set().union(*[G.nodes() for G in layers.values()])
                 for name, layers in analyses_layers.items()}

    names = list(gene_sets.keys())
    sets = list(gene_sets.values())

    # Pairwise overlaps
    overlap_data = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            key = f"{names[i]} ∩ {names[j]}"
            overlap_data[key] = len(sets[i] & sets[j])

    all_three = sets[0] & sets[1] & sets[2]
    overlap_data[f"All three"] = len(all_three)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(overlap_data.keys()), list(overlap_data.values()),
           color='mediumpurple', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Shared Genes')
    ax.set_title('Gene Set Overlap Across Multiplexes', fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    for i, v in enumerate(overlap_data.values()):
        ax.text(i, v + 50, str(v), ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_overlap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Genes in all three multiplexes: {len(all_three)}")
    

def main():
    parser = argparse.ArgumentParser(
        description='Compare three multiplex networks comprehensively'
    )
    parser.add_argument('--tabula-layers', type=str, required=True,
                      help='Directory with Tabula Sapiens multiplex layers')
    parser.add_argument('--pain-layers', type=str, required=True,
                      help='Directory with 2025 hDRG Atlas layers')
    parser.add_argument('--pain-subset-layers', type=str, required=True,
                      help='Directory with Pain subset of Tabula multiplex layers')
    parser.add_argument('--seed-genes', type=str, required=True,
                      help='Path to seed genes CSV file')
    parser.add_argument('--output-dir', type=str, default='multiplex_comparison',
                      help='Output directory for results')
  
    args = parser.parse_args()
  
    os.makedirs(args.output_dir, exist_ok=True)
  
    # Load seed genes
    print("\n" + "="*80)
    print("LOADING SEED GENES")
    print("="*80)
    seed_genes = load_seed_genes(args.seed_genes)
    print(f"Loaded {len(seed_genes)} seed genes")
  
    # Load all three multiplexes
    print("\n" + "="*80)
    print("LOADING MULTIPLEX NETWORKS")
    print("="*80)
  
    print("\nLoading Tabula Sapiens multiplex...")
    tabula_layers = load_multiplex_layers(args.tabula_layers)
    print(f"Loaded {len(tabula_layers)} layers")
  
    print("\nLoading 2025 hDRG Atlas...")
    pain_layers = load_multiplex_layers(args.pain_layers)
    print(f"Loaded {len(pain_layers)} layers")
  
    print("\nLoading Pain subset of Tabula multiplex...")
    pain_subset_layers = load_multiplex_layers(args.pain_subset_layers)
    print(f"Loaded {len(pain_subset_layers)} layers")
          
    # After loading layers, build this dict for convenience
    analyses_layers = {
        "Tabula Sapiens": tabula_layers,
        "2025 hDRG Atlas": pain_layers,
        "Pain Subset (Tabula)": pain_subset_layers
    }

  
    # Analyze all three
    print("\n" + "="*80)
    print("ANALYZING MULTIPLEX NETWORKS")
    print("="*80)
  
    tabula_analysis = analyze_multiplex(tabula_layers, seed_genes=seed_genes, name="Tabula Sapiens")
    pain_analysis = analyze_multiplex(pain_layers, seed_genes=seed_genes, name="2025 hDRG Atlas")
    pain_subset_analysis = analyze_multiplex(pain_subset_layers, seed_genes=seed_genes, name="Pain Subset (Tabula)")
  
    analyses = [tabula_analysis, pain_analysis, pain_subset_analysis]
  
    # Create comparison dataframe
    print("\n" + "="*80)
    print("CREATING COMPARISON DATAFRAMES")
    print("="*80)
  
    comparison_df = create_comparison_dataframe(analyses)
    summary_df = create_multiplex_summary(analyses)
  
    # Save CSVs
    comparison_file = os.path.join(args.output_dir, 'layer_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved: {comparison_file}")
  
    summary_file = os.path.join(args.output_dir, 'multiplex_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved: {summary_file}")
  
    # Save detailed JSON report
    report = {
        'tabula_sapiens': tabula_analysis,
        'pain_multiplex': pain_analysis,
        'pain_subset': pain_subset_analysis
    }
  
    report_file = os.path.join(args.output_dir, 'detailed_analysis.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {report_file}")
  
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
  
    plot_density_comparison(comparison_df, args.output_dir)
    plot_degree_comparison(comparison_df, args.output_dir)
    plot_seed_gene_spread(comparison_df, args.output_dir)
    plot_seed_connectivity(comparison_df, args.output_dir)
    plot_components_comparison(comparison_df, args.output_dir)
    plot_largest_cc_comparison(comparison_df, args.output_dir)
    plot_seed_components_comparison(comparison_df, args.output_dir)
    plot_multiplex_summary(summary_df, args.output_dir)


    # Add these calls in your visualization block:
    for name, layers in analyses_layers.items():
        plot_interlayer_jaccard(layers, name, args.output_dir)
        plot_gene_ubiquity(layers, seed_genes, name, args.output_dir)
        plot_edge_weight_distribution(layers, name, args.output_dir)

    plot_clustering_comparison(analyses_layers, args.output_dir)
    plot_gene_overlap_venn(analyses_layers, args.output_dir)

    hub_dfs = {name: analyze_hub_genes(layers, seed_genes, name)
               for name, layers in analyses_layers.items()}
    plot_hub_genes(hub_dfs, args.output_dir)
  
    # Print summary tables
    print("\n" + "="*80)
    print("MULTIPLEX SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
  
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Files generated:")
    print(f"  - layer_comparison.csv")
    print(f"  - multiplex_summary.csv")
    print(f"  - detailed_analysis.json")
    print(f"  - density_comparison.png")
    print(f"  - degree_comparison.png")
    print(f"  - seed_gene_spread.png")
    print(f"  - seed_connectivity.png")
    print(f"  - components_comparison.png")
    print(f"  - largest_cc_comparison.png")
    print(f"  - seed_components_comparison.png")
    print(f"  - multiplex_summary.png")
  
    return 0

if __name__ == "__main__":
    sys.exit(main())