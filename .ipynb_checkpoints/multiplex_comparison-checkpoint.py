#!/usr/bin/env python3
"""
Comprehensive Comparison Analysis of Three Multiplex Networks

Compares:
1. Tabula Sapiens multiplex (full)
2. Pain multiplex (2025)
3. Pain subset of Tabula Sapiens multiplex

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
    all_edges = 0
    
    for cell_type in sorted(layers.keys()):
        layer = layers[cell_type]
        
        n_nodes = layer.number_of_nodes()
        n_edges = layer.number_of_edges()
        all_genes.update(layer.nodes())
        all_edges += n_edges
        
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
        'n_total_edges': all_edges,
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

def main():
    parser = argparse.ArgumentParser(
        description='Compare three multiplex networks comprehensively'
    )
    parser.add_argument('--tabula-layers', type=str, required=True,
                      help='Directory with Tabula Sapiens multiplex layers')
    parser.add_argument('--pain-layers', type=str, required=True,
                      help='Directory with Pain multiplex layers')
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
    
    print("\nLoading Pain multiplex...")
    pain_layers = load_multiplex_layers(args.pain_layers)
    print(f"Loaded {len(pain_layers)} layers")
    
    print("\nLoading Pain subset of Tabula multiplex...")
    pain_subset_layers = load_multiplex_layers(args.pain_subset_layers)
    print(f"Loaded {len(pain_subset_layers)} layers")
    
    # Analyze all three
    print("\n" + "="*80)
    print("ANALYZING MULTIPLEX NETWORKS")
    print("="*80)
    
    tabula_analysis = analyze_multiplex(tabula_layers, seed_genes=seed_genes, name="Tabula Sapiens")
    pain_analysis = analyze_multiplex(pain_layers, seed_genes=seed_genes, name="Pain Multiplex")
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