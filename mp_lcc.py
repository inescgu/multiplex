#!/usr/bin/env python3
"""
Multiplex LCC Analysis

Analyze largest connected component distribution across multiplex layers
and compare to random expectations.
"""

import networkx as nx
import pandas as pd
import numpy as np
import pickle
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def load_seed_genes():
    """Load seed genes from CSV file"""
    try:
        seed_genes = pd.read_csv('../og_gene_set/seed_genes.csv')
        return set(seed_genes['gene'].tolist())
    except Exception as e:
        print(f"Error loading seed genes: {str(e)}")
        return set()

def load_multiplex_layers(multiplex_dir='multiplex_interactome'):
    """Load all multiplex layers"""
    layers_dir = os.path.join(multiplex_dir, 'layers')
    
    layers = {}
    for layer_file in os.listdir(layers_dir):
        if layer_file.endswith('_layer.pkl'):
            layer_name = layer_file.replace('_layer.pkl', '')
            layer_path = os.path.join(layers_dir, layer_file)
            
            with open(layer_path, 'rb') as f:
                layer_graph = pickle.load(f)
            
            layers[layer_name] = layer_graph
            print(f"Loaded {layer_name}: {layer_graph.number_of_nodes()} nodes")
    
    return layers

def largest_connected_component(G):
    """Find largest connected component"""
    if G.number_of_nodes() == 0:
        return G, 0
    
    connected_components = list(nx.connected_components(G))
    if not connected_components:
        return G.subgraph([]), 0
    
    largest_component = max(connected_components, key=len)
    largest_component_subgraph = G.subgraph(largest_component).copy()
    
    return largest_component_subgraph, len(largest_component)

def calculate_lcc_significance(G, seed_genes, n_iterations=1000):
    """Calculate LCC significance for seed genes in a layer"""
    seed_gene_ids = [g for g in seed_genes if g in G.nodes()]
    n_seed_genes = len(seed_gene_ids)
    
    if n_seed_genes == 0:
        return {'p_value': 1.0, 'z_score': 0.0, 'observed_size': 0,
                'random_mean': 0, 'random_std': 0, 'random_sizes': []}
    
    # Observed LCC size
    seed_subgraph = G.subgraph(seed_gene_ids).copy()
    _, observed_size = largest_connected_component(seed_subgraph)
    
    # Random samples
    all_nodes = list(G.nodes())
    random_sizes = []
    
    for _ in tqdm(range(n_iterations), desc="LCC significance"):
        if len(all_nodes) >= n_seed_genes:
            random_nodes = random.sample(all_nodes, n_seed_genes)
        else:
            random_nodes = all_nodes.copy()
        
        random_subgraph = G.subgraph(random_nodes).copy()
        _, random_size = largest_connected_component(random_subgraph)
        random_sizes.append(random_size)
    
    # Calculate statistics
    random_mean = np.mean(random_sizes)
    random_std = np.std(random_sizes)
    
    if random_std > 0:
        z_score = (observed_size - random_mean) / random_std
    else:
        z_score = 0
    
    p_value = sum(1 for size in random_sizes if size >= observed_size) / n_iterations
    
    return {
        'p_value': p_value,
        'z_score': z_score,
        'observed_size': observed_size,
        'random_mean': random_mean,
        'random_std': random_std,
        'random_sizes': random_sizes
    }

def analyze_multiplex_lcc(layers, seed_genes, output_dir='output/multiplex/lcc'):
    """Analyze LCC across all multiplex layers"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for layer_name, layer_graph in layers.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {layer_name}")
        print(f"{'='*60}")
        
        # Calculate LCC significance
        lcc_sig = calculate_lcc_significance(layer_graph, seed_genes, n_iterations=1000)
        
        # Get component statistics
        components = list(nx.connected_components(layer_graph))
        component_sizes = [len(c) for c in components]
        
        results[layer_name] = {
            'lcc_significance': lcc_sig,
            'n_components': len(components),
            'component_sizes': component_sizes,
            'n_nodes': layer_graph.number_of_nodes(),
            'n_edges': layer_graph.number_of_edges(),
            'density': nx.density(layer_graph),
            'seed_genes_in_layer': len([g for g in seed_genes if g in layer_graph.nodes()])
        }
        
        print(f"LCC observed: {lcc_sig['observed_size']}")
        print(f"LCC random mean: {lcc_sig['random_mean']:.2f}")
        print(f"LCC p-value: {lcc_sig['p_value']:.6f}")
        print(f"LCC z-score: {lcc_sig['z_score']:.2f}")
    
    # Save results
    with open(os.path.join(output_dir, 'multiplex_lcc_results.pkl'), 'wb') as f:
        pickle.dump(results, f, protocol=4)
    
    # Create summary DataFrame
    summary_rows = []
    for layer_name, data in results.items():
        summary_rows.append({
            'Layer': layer_name,
            'Tissue': layer_name.split('_')[0],
            'N_Nodes': data['n_nodes'],
            'N_Edges': data['n_edges'],
            'Density': data['density'],
            'Seed_Genes': data['seed_genes_in_layer'],
            'LCC_Observed': data['lcc_significance']['observed_size'],
            'LCC_Random_Mean': data['lcc_significance']['random_mean'],
            'LCC_Random_Std': data['lcc_significance']['random_std'],
            'LCC_Z_Score': data['lcc_significance']['z_score'],
            'LCC_P_Value': data['lcc_significance']['p_value'],
            'Significant': data['lcc_significance']['p_value'] < 0.05,
            'N_Components': data['n_components']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'multiplex_lcc_summary.csv'), index=False)
    
    # Create visualizations
    plot_multiplex_lcc(results, output_dir)
    
    return results

def plot_multiplex_lcc(results, output_dir):
    """Create visualizations for multiplex LCC analysis - grouped by tissue, color-coded"""
    
    # Filter for significant results only
    significant_results = {k: v for k, v in results.items() 
                          if v['lcc_significance']['p_value'] < 0.05}
    
    if not significant_results:
        print("No significant LCC results to plot (p < 0.05)")
        return
    
    print(f"\nPlotting {len(significant_results)} significant layers")
    
    # Group by tissue type
    tissue_groups = {}
    for layer_name in significant_results.keys():
        tissue = layer_name.split('_')[0]
        if tissue not in tissue_groups:
            tissue_groups[tissue] = []
        tissue_groups[tissue].append(layer_name)
    
    # Define color palette for tissues
    tissues = sorted(tissue_groups.keys())
    colors = sns.color_palette("husl", len(tissues))
    tissue_colors = {tissue: colors[i] for i, tissue in enumerate(tissues)}
    
    # Plot 1: LCC metrics by tissue (overview)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    layer_names = sorted(significant_results.keys())
    observed_sizes = [significant_results[l]['lcc_significance']['observed_size'] for l in layer_names]
    z_scores = [significant_results[l]['lcc_significance']['z_score'] for l in layer_names]
    p_values = [significant_results[l]['lcc_significance']['p_value'] for l in layer_names]
    densities = [significant_results[l]['density'] for l in layer_names]
    
    # Color bars by tissue
    bar_colors = [tissue_colors[l.split('_')[0]] for l in layer_names]
    display_names = [l.replace('_', '\n') for l in layer_names]
    
    # Plot LCC sizes
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(layer_names)), observed_sizes, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('LCC Size', fontsize=12, fontweight='bold')
    ax1.set_title('Observed LCC Size (Significant Layers)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, observed_sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(int(val)), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot Z-scores
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(layer_names)), z_scores, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1.96, color='red', linestyle='--', linewidth=2, label='p=0.05 threshold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Z-Score', fontsize=12, fontweight='bold')
    ax2.set_title('LCC Z-Score (Significant Layers)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot network properties by tissue
    ax3 = axes[1, 0]
    tissue_list = sorted(tissue_groups.keys())
    avg_nodes = [np.mean([significant_results[l]['n_nodes'] for l in tissue_groups[t]]) for t in tissue_list]
    avg_edges = [np.mean([significant_results[l]['n_edges'] for l in tissue_groups[t]]) for t in tissue_list]
    
    x = np.arange(len(tissue_list))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, avg_nodes, width, label='Avg Nodes', 
                    color=[tissue_colors[t] for t in tissue_list], alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, avg_edges, width, label='Avg Edges', 
                    color=[tissue_colors[t] for t in tissue_list], alpha=0.4, edgecolor='black', linewidth=1.5)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(tissue_list, fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Network Size by Tissue', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot p-values on log scale
    ax4 = axes[1, 1]
    bars = ax4.bar(range(len(layer_names)), -np.log10(p_values), color=bar_colors, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05 (-log10=1.30)')
    ax4.set_xticks(range(len(layer_names)))
    ax4.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('-log10(p-value)', fontsize=12, fontweight='bold')
    ax4.set_title('LCC Significance Strength', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiplex_lcc_overview_significant.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create legend figure for tissue colors
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    for i, tissue in enumerate(tissues):
        ax.add_patch(plt.Rectangle((0.1, 0.85 - i*0.15), 0.05, 0.08, 
                                   facecolor=tissue_colors[tissue], edgecolor='black', linewidth=2))
        ax.text(0.2, 0.89 - i*0.15, f'{tissue}', fontsize=14, fontweight='bold', va='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tissue_color_legend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot LCC distributions for each significant layer
    print("\nGenerating individual LCC distribution plots...")
    for layer_name, data in significant_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tissue = layer_name.split('_')[0]
        random_sizes = data['lcc_significance']['random_sizes']
        observed = data['lcc_significance']['observed_size']
        p_value = data['lcc_significance']['p_value']
        z_score = data['lcc_significance']['z_score']
        
        ax.hist(random_sizes, bins=30, alpha=0.7, color=tissue_colors[tissue], label='Random')
        ax.axvline(x=observed, color='#C73188', linestyle='--', linewidth=2.5,
                  label=f'Observed ({int(observed)})')
        
        # Add statistics text
        stats_text = f'p = {p_value:.4f}\nz = {z_score:.2f}\nTissue: {tissue}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5),
               fontsize=11, fontweight='bold')
        
        ax.set_xlabel('LCC Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'LCC Distribution: {layer_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        safe_name = layer_name.replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'lcc_dist_{safe_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print("="*80)
    print("MULTIPLEX LCC ANALYSIS")
    print("="*80)
    
    # Load seed genes
    seed_genes = load_seed_genes()
    print(f"\nLoaded {len(seed_genes)} seed genes")
    
    # Load multiplex layers
    print("\nLoading multiplex layers...")
    layers = load_multiplex_layers('multiplex_interactome')
    print(f"Loaded {len(layers)} layers")
    
    # Analyze LCC
    results = analyze_multiplex_lcc(layers, seed_genes)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Results saved to: output/multiplex/lcc/")

if __name__ == "__main__":
    main()