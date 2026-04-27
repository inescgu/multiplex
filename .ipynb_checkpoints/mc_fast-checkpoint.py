#!/usr/bin/env python3
"""
HPC-optimized multiplex network comparison
- Loads data once in main process
- Processes layers sequentially but efficiently
- Minimal memory overhead
- Vectorized computations
"""

import os
import sys
import argparse
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_multiplex_layers(layers_dir):
    """Load all layer networks from directory"""
    layers = {}
    layer_files = sorted(Path(layers_dir).glob('*_layer.pkl'))
    
    for i, layer_file in enumerate(layer_files):
        cell_type = layer_file.stem.replace('_layer', '')
        try:
            with open(layer_file, 'rb') as f:
                G = pickle.load(f)
            if G.number_of_nodes() > 0:
                layers[cell_type] = G
                print(f"  [{i+1}/{len(layer_files)}] Loaded {cell_type}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            print(f"  Error loading {cell_type}: {str(e)}")
    
    return layers

def load_seed_genes(seed_genes_file):
    """Load seed genes from CSV file"""
    if not os.path.exists(seed_genes_file):
        return set()
    
    try:
        seed_df = pd.read_csv(seed_genes_file)
        gene_col = None
        for col in ['gene', 'Gene', 'gene_id', 'Gene_ID', 'symbol', 'Symbol', 'SYMBOL']:
            if col in seed_df.columns:
                gene_col = col
                break
        if gene_col is None:
            gene_col = seed_df.columns[0]
        seed_genes = set(seed_df[gene_col].dropna().astype(str).str.strip().str.upper())
        return seed_genes
    except Exception as e:
        print(f"Error loading seed genes: {str(e)}")
        return set()

def compute_layer_stats(cell_type, layer, seed_genes=None):
    """Compute statistics for a single layer (memory-efficient)"""
    n_nodes = layer.number_of_nodes()
    n_edges = layer.number_of_edges()
    
    if n_nodes == 0:
        return None
    
    # Basic properties
    density = nx.density(layer)
    avg_degree = 2 * n_edges / n_nodes
    
    # Connected components
    components = list(nx.connected_components(layer))
    n_components = len(components)
    largest_cc = max(components, default=set())
    largest_cc_size = len(largest_cc)
    largest_cc_pct = largest_cc_size / n_nodes * 100
    
    # # Average shortest path (only if largest CC is reasonable)
    # avg_shortest_path = 0
    # if len(largest_cc) > 1:
    #     try:
    #         subgraph = layer.subgraph(largest_cc)
    #         avg_shortest_path = nx.average_shortest_path_length(subgraph)
    #     except:
    #         pass
    
    # Clustering coefficient
    avg_clustering = nx.average_clustering(layer)
    
    # Seed gene analysis
    seeds_in_layer = set()
    seed_pct_of_layer = 0
    seed_pct_of_total = 0
    seed_genes_connected = False
    n_seed_components = 0
    
    if seed_genes:
        seeds_in_layer = seed_genes.intersection(layer.nodes())
        if seeds_in_layer:
            seed_pct_of_layer = len(seeds_in_layer) / n_nodes * 100
            seed_pct_of_total = len(seeds_in_layer) / len(seed_genes) * 100
            
            if len(seeds_in_layer) > 1:
                seed_subgraph = layer.subgraph(seeds_in_layer)
                n_seed_components = nx.number_connected_components(seed_subgraph)
                seed_genes_connected = (n_seed_components == 1)
            elif len(seeds_in_layer) == 1:
                seed_genes_connected = True
                n_seed_components = 1
    
    return {
        'cell_type': cell_type,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'avg_degree': avg_degree,
        'n_components': n_components,
        'largest_cc_size': largest_cc_size,
        'largest_cc_pct': largest_cc_pct,
        #'avg_shortest_path': avg_shortest_path,
        'n_seed_genes': len(seeds_in_layer),
        'seed_pct_of_layer': seed_pct_of_layer,
        'seed_pct_of_total': seed_pct_of_total,
        'seed_genes_connected': seed_genes_connected,
        'n_seed_components': n_seed_components,
        'avg_clustering': avg_clustering
    }

def analyze_multiplex(layers, seed_genes=None, name="Multiplex"):
    """Analyze multiplex network (sequential, memory-efficient)"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {name}")
    print(f"{'='*80}")
    
    # Sequential layer analysis (avoids serialization overhead)
    print(f"Processing {len(layers)} layers...")
    layer_stats = []
    for i, (ct, layer) in enumerate(sorted(layers.items())):
        stat = compute_layer_stats(ct, layer, seed_genes)
        if stat:
            layer_stats.append(stat)
            print(f"  [{i+1}/{len(layers)}] {ct}: {stat['n_nodes']} nodes, density={stat['density']:.4f}, clustering={stat['avg_clustering']:.4f}")
    
    # Compute multiplex-level stats (vectorized)
    all_genes = set()
    all_edges = set()
    for ct, layer in layers.items():
        all_genes.update(layer.nodes())
        all_edges.update((min(u, v), max(u, v)) for u, v in layer.edges())
    
    # Vectorized stats
    stats_array = np.array([
        [s['n_nodes'], s['n_edges'], s['density'], s['avg_degree'], 
         s['n_components'], s['largest_cc_pct'], 
         #s['avg_shortest_path'],
         s['avg_clustering']]
        for s in layer_stats
    ])
    
    multiplex_stats = {
        'n_total_genes': len(all_genes),
        'n_total_edges': len(all_edges),
        'avg_genes_per_layer': float(np.mean(stats_array[:, 0])),
        'avg_edges_per_layer': float(np.mean(stats_array[:, 1])),
        'avg_density': float(np.mean(stats_array[:, 2])),
        'avg_degree': float(np.mean(stats_array[:, 3])),
        'avg_components': float(np.mean(stats_array[:, 4])),
        'avg_largest_cc_pct': float(np.mean(stats_array[:, 5])),
        'avg_clustering': float(np.mean(stats_array[:, 6]))
    }
    
    # # Handle avg_shortest_path (skip zeros)
    # valid_paths = stats_array[stats_array[:, 6] > 0, 6]
    # multiplex_stats['avg_shortest_path'] = float(np.mean(valid_paths)) if len(valid_paths) > 0 else 0
    
    print(f"\nMultiplex stats: {multiplex_stats['n_total_genes']} genes, "
          f"{multiplex_stats['n_total_edges']} edges, "
          f"avg_density={multiplex_stats['avg_density']:.4f}")
    
    # Seed gene stats
    seed_gene_stats = {}
    if seed_genes:
        seeds_in_multiplex = seed_genes.intersection(all_genes)
        seeds_pct = len(seeds_in_multiplex) / len(seed_genes) * 100
        
        # Seeds in all layers
        seeds_in_all = set(layers[list(layers.keys())[0]].nodes()).intersection(seed_genes)
        for cell_type in list(layers.keys())[1:]:
            seeds_in_all = seeds_in_all.intersection(set(layers[cell_type].nodes()).intersection(seed_genes))
        
        # Seed spread
        seed_layer_counts = defaultdict(int)
        for seed in seeds_in_multiplex:
            for cell_type, layer in layers.items():
                if seed in layer.nodes():
                    seed_layer_counts[seed] += 1
        
        if seed_layer_counts:
            counts = list(seed_layer_counts.values())
            seed_gene_stats = {
                'total_seed_genes': len(seed_genes),
                'seed_genes_in_multiplex': len(seeds_in_multiplex),
                'seed_gene_recovery_pct': seeds_pct,
                'seed_genes_in_all_layers': len(seeds_in_all),
                'avg_layers_per_seed': float(np.mean(counts)),
                'min_layers_per_seed': int(np.min(counts)),
                'max_layers_per_seed': int(np.max(counts))
            }
    
    return {
        'name': name,
        'n_layers': len(layers),
        'layer_stats': layer_stats,
        'multiplex_stats': multiplex_stats,
        'seed_gene_stats': seed_gene_stats
    }

def create_comparison_dataframe(analyses):
    """Create comparison dataframe"""
    rows = []
    for analysis in analyses:
        for stat in analysis['layer_stats']:
            rows.append({
                'Multiplex': analysis['name'],
                'Cell Type': stat['cell_type'],
                'Nodes': stat['n_nodes'],
                'Edges': stat['n_edges'],
                'Density': stat['density'],
                'Avg Degree': stat['avg_degree'],
                'Components': stat['n_components'],
                'Largest CC %': stat['largest_cc_pct'],
                'Avg Clustering': stat['avg_clustering'],
                'Seed Genes': stat['n_seed_genes'],
                'Seed % of Layer': stat['seed_pct_of_layer'],
                'Seed Connected': stat['seed_genes_connected']
            })
    return pd.DataFrame(rows)

def create_multiplex_summary(analyses):
    """Create summary dataframe"""
    rows = []
    for analysis in analyses:
        row = {
            'Multiplex': analysis['name'],
            'N Layers': analysis['n_layers'],
            'Total Genes': analysis['multiplex_stats']['n_total_genes'],
            'Total Edges': analysis['multiplex_stats']['n_total_edges'],
            'Avg Density': analysis['multiplex_stats']['avg_density'],
            'Avg Degree': analysis['multiplex_stats']['avg_degree'],
            'Avg Clustering': analysis['multiplex_stats']['avg_clustering']
        }
        if analysis['seed_gene_stats']:
            row['Seed Recovery %'] = analysis['seed_gene_stats']['seed_gene_recovery_pct']
        rows.append(row)
    return pd.DataFrame(rows)

def plot_all_comparisons(comparison_df, summary_df, output_dir):
    """Create all comparison plots efficiently"""
    multiplexes = comparison_df['Multiplex'].unique()
    
    # Figure 1: Comparison metrics (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        (axes[0, 0], 'Density', 'steelblue'),
        (axes[0, 1], 'Avg Degree', 'coral'),
        (axes[0, 2], 'Avg Clustering', 'mediumseagreen'),
        (axes[1, 0], 'Components', 'mediumpurple'),
        (axes[1, 1], 'Largest CC %', 'lightsalmon'),
        (axes[1, 2], 'Seed % of Layer', 'gold')
    ]
    
    for ax, metric, color in metrics:
        for multiplex in multiplexes:
            data = comparison_df[comparison_df['Multiplex'] == multiplex][metric]
            ax.hist(data, bins=12, alpha=0.5, label=multiplex, color=color, edgecolor='black')
        ax.set_title(f'{metric}', fontweight='bold', fontsize=11)
        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: comparison_metrics.png")
    
    # Figure 2: Multiplex summary (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    mux_names = summary_df['Multiplex'].values
    
    metrics_summary = [
        (axes[0, 0], 'Total Genes', 'steelblue'),
        (axes[0, 1], 'Total Edges', 'coral'),
        (axes[0, 2], 'Avg Density', 'mediumseagreen'),
        (axes[1, 0], 'Avg Degree', 'mediumpurple'),
        (axes[1, 1], 'Avg Clustering', 'gold'),
    ]
    
    for ax, col, color in metrics_summary:
        ax.bar(mux_names, summary_df[col], color=color, alpha=0.7, edgecolor='black')
        ax.set_title(col, fontweight='bold')
        ax.set_ylabel(col)
        for i, v in enumerate(summary_df[col]):
            ax.text(i, v * 1.02, f'{v:.2f}' if v < 100 else f'{int(v)}', 
                   ha='center', fontweight='bold', fontsize=9)
    
    # Seed recovery if available
    if 'Seed Recovery %' in summary_df.columns:
        axes[1, 2].bar(mux_names, summary_df['Seed Recovery %'], color='lightsalmon', alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Seed Recovery %', fontweight='bold')
        axes[1, 2].set_ylabel('Percentage')
        for i, v in enumerate(summary_df['Seed Recovery %']):
            axes[1, 2].text(i, v * 1.02, f'{v:.1f}%', ha='center', fontweight='bold')
    else:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiplex_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: multiplex_summary.png")

def plot_jaccard_heatmaps(analyses_layers, output_dir):
    """Plot Jaccard similarity heatmaps for each multiplex"""
    for name, layers in analyses_layers.items():
        cell_types = sorted(layers.keys())
        n = len(cell_types)
        
        if n > 1:
            matrix = np.zeros((n, n))
            for i, ct1 in enumerate(cell_types):
                for j, ct2 in enumerate(cell_types):
                    s1 = set(layers[ct1].nodes())
                    s2 = set(layers[ct2].nodes())
                    union = s1 | s2
                    matrix[i, j] = len(s1 & s2) / len(union) if union else 0
            
            fig, ax = plt.subplots(figsize=(max(8, n//2), max(6, n//2)))
            sns.heatmap(matrix, xticklabels=cell_types, yticklabels=cell_types,
                       cmap='YlOrRd', ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Jaccard'})
            ax.set_title(f'{name} — Jaccard Similarity', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'jaccard_{name.replace(" ", "_")}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: jaccard_{name.replace(' ', '_')}.png")

def plot_gene_ubiquity(analyses_layers, seed_genes, output_dir):
    """Plot gene ubiquity for each multiplex"""
    fig, axes = plt.subplots(1, len(analyses_layers), figsize=(16, 4))
    if len(analyses_layers) == 1:
        axes = [axes]
    
    for ax, (name, layers) in zip(axes, analyses_layers.items()):
        gene_counts = defaultdict(int)
        for layer in layers.values():
            for gene in layer.nodes():
                gene_counts[gene] += 1
        
        counts = list(gene_counts.values())
        n_layers = len(layers)
        
        ax.hist(counts, bins=range(1, n_layers + 2), alpha=0.7, color='steelblue', edgecolor='black', align='left')
        ax.set_xlabel('Layers')
        ax.set_ylabel('Genes')
        ax.set_title(f'{name}', fontweight='bold')
        ax.set_xticks(range(1, n_layers + 1))
        
        if seed_genes:
            seed_counts = [gene_counts[g] for g in seed_genes if g in gene_counts]
            if seed_counts:
                ax.hist(seed_counts, bins=range(1, n_layers + 2), alpha=0.5, color='red', 
                       edgecolor='black', align='left', label='Seed genes')
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_ubiquity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: gene_ubiquity.png")

def analyze_hub_genes(layers, seed_genes, name, top_n=15):
    """Find top hub genes"""
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
            'layers': degree_count[gene],
            'is_seed': gene in seed_genes
        }
        for gene in degree_sum
    ]).sort_values('total_degree', ascending=False).head(top_n)
    
    return hub_df

def plot_hub_genes(analyses_layers, seed_genes, output_dir):
    """Plot hub genes for all multiplexes"""
    fig, axes = plt.subplots(1, len(analyses_layers), figsize=(16, 5))
    if len(analyses_layers) == 1:
        axes = [axes]
    
    for ax, (name, layers) in zip(axes, analyses_layers.items()):
        hub_df = analyze_hub_genes(layers, seed_genes, name, top_n=12)
        colors = ['crimson' if s else 'steelblue' for s in hub_df['is_seed']]
        ax.barh(hub_df['gene'], hub_df['total_degree'], color=colors, edgecolor='black', alpha=0.8)
        ax.invert_yaxis()
        ax.set_title(f'{name}', fontweight='bold')
        ax.set_xlabel('Total Degree')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hub_genes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: hub_genes.png")

def plot_gene_overlap(analyses_layers, output_dir):
    """Plot gene set overlaps"""
    gene_sets = {name: set().union(*[G.nodes() for G in layers.values()])
                 for name, layers in analyses_layers.items()}
    
    names = list(gene_sets.keys())
    sets = list(gene_sets.values())
    
    overlap_data = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            key = f"{names[i][:15]} ∩ {names[j][:15]}"
            overlap_data[key] = len(sets[i] & sets[j])
    
    if len(sets) == 3:
        all_three = sets[0] & sets[1] & sets[2]
        overlap_data['All three'] = len(all_three)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(overlap_data.keys()), list(overlap_data.values()), 
           color='mediumpurple', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Shared Genes')
    ax.set_title('Gene Set Overlap', fontweight='bold')
    plt.xticks(rotation=20, ha='right', fontsize=9)
    for i, v in enumerate(overlap_data.values()):
        ax.text(i, v + 50, str(v), ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_overlap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: gene_overlap.png")

def main():
    parser = argparse.ArgumentParser(description='HPC-optimized multiplex network comparison')
    parser.add_argument('--tabula-layers', type=str, required=True)
    parser.add_argument('--pain-layers', type=str, required=True)
    parser.add_argument('--pain-subset-layers', type=str, required=True)
    parser.add_argument('--seed-genes', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='multiplex_comparison')
  
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
  
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    seed_genes = load_seed_genes(args.seed_genes)
    print(f"Loaded {len(seed_genes)} seed genes")
    
    print("\nLoading Tabula Sapiens multiplex...")
    tabula_layers = load_multiplex_layers(args.tabula_layers)
    print(f"Total: {len(tabula_layers)} layers")
    
    print("\nLoading 2025 hDRG Atlas...")
    pain_layers = load_multiplex_layers(args.pain_layers)
    print(f"Total: {len(pain_layers)} layers")
    
    print("\nLoading Pain subset of Tabula multiplex...")
    pain_subset_layers = load_multiplex_layers(args.pain_subset_layers)
    print(f"Total: {len(pain_subset_layers)} layers")
    
    analyses_layers = {
        "Tabula Sapiens": tabula_layers,
        "2025 hDRG Atlas": pain_layers,
        "Pain Subset (Tabula)": pain_subset_layers
    }
    
    print("\n" + "="*80)
    print("ANALYZING NETWORKS")
    print("="*80)
    
    # Sequential analysis (avoids serialization overhead)
    analyses = []
    for name, layers in analyses_layers.items():
        analysis = analyze_multiplex(layers, seed_genes, name)
        analyses.append(analysis)
    
    print("\n" + "="*80)
    print("CREATING DATAFRAMES")
    print("="*80)
    
    comparison_df = create_comparison_dataframe(analyses)
    summary_df = create_multiplex_summary(analyses)
    
    comparison_df.to_csv(os.path.join(args.output_dir, 'layer_comparison.csv'), index=False)
    print("Saved: layer_comparison.csv")
    
    summary_df.to_csv(os.path.join(args.output_dir, 'multiplex_summary.csv'), index=False)
    print("Saved: multiplex_summary.csv")
    
    # Save JSON
    report = {analysis['name']: analysis for analysis in analyses}
    with open(os.path.join(args.output_dir, 'detailed_analysis.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print("Saved: detailed_analysis.json")
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    plot_all_comparisons(comparison_df, summary_df, args.output_dir)
    plot_jaccard_heatmaps(analyses_layers, args.output_dir)
    plot_gene_ubiquity(analyses_layers, seed_genes, args.output_dir)
    plot_hub_genes(analyses_layers, seed_genes, args.output_dir)
    plot_gene_overlap(analyses_layers, args.output_dir)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Output: {args.output_dir}")
    print("Files: layer_comparison.csv, multiplex_summary.csv, detailed_analysis.json")
    print("Plots: comparison_metrics.png, multiplex_summary.png, jaccard_*.png, gene_ubiquity.png, hub_genes.png, gene_overlap.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())