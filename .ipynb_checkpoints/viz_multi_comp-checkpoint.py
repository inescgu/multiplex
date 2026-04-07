#!/usr/bin/env python3
"""
Advanced Comparative Visualization for Multiplex Networks

Generates segmented visualizations comparing:
1. Tabula Sapiens multiplex (full)
2. 2025 hDRG Atlas multiplex
3. Pain subset of Tabula Sapiens multiplex

Visualizations:
- Layer size comparisons (nodes, edges, absolute & percentage)
- Seed gene presence across layers
- Gene set Venn diagrams (seeds and non-seeds)
- Jaccard similarities heatmaps
- Largest connected component comparisons
- Top 10% ubiquitous genes analysis
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
from matplotlib.patches import Rectangle
from matplotlib_venn import venn2, venn3
import warnings
warnings.filterwarnings('ignore')

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

def filter_layers_by_seed_genes(layers, seed_genes):
    """Filter out layers that have no seed genes"""
    filtered = {}
    for cell_type, G in layers.items():
        seeds_in_layer = seed_genes.intersection(G.nodes())
        if len(seeds_in_layer) > 0:
            filtered[cell_type] = G
    return filtered

def get_layer_stats(layers, seed_genes):
    """Get comprehensive layer statistics"""
    stats = {}
    
    for cell_type, G in layers.items():
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # LCC
        largest_cc = max(nx.connected_components(G), default=set())
        lcc_size = len(largest_cc)
        lcc_pct = lcc_size / n_nodes * 100 if n_nodes > 0 else 0
        
        # Seed genes
        seeds_in_layer = seed_genes.intersection(G.nodes())
        n_seeds = len(seeds_in_layer)
        
        stats[cell_type] = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'lcc_size': lcc_size,
            'lcc_pct': lcc_pct,
            'n_seeds': n_seeds,
            'seed_genes': seeds_in_layer
        }
    
    return stats

def plot_layer_sizes_comparison(analyses_layers, output_dir):
    """Three side-by-side bar charts comparing layer sizes with adaptive sizing"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    multiplexes = list(analyses_layers.keys())
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for ax_idx, (multiplex_name, layers) in enumerate(analyses_layers.items()):
        cell_types = sorted(layers.keys())
        n_nodes = [layers[ct].number_of_nodes() for ct in cell_types]
        n_edges = [layers[ct].number_of_edges() for ct in cell_types]
        
        x = np.arange(len(cell_types))
        width = 0.35
        
        ax = axes[ax_idx]
        bars1 = ax.bar(x - width/2, n_nodes, width, label='Nodes', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, n_edges, width, label='Edges', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Cell Type', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(f'{multiplex_name}\nLayer Sizes (Absolute)', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        
        # Adaptive font size based on number of layers
        fontsize = max(6, min(9, 150 / len(cell_types)))
        ax.set_xticklabels(cell_types, rotation=90, ha='right', fontsize=fontsize)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_layer_sizes_absolute.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: 01_layer_sizes_absolute.png")
    plt.close()

def plot_layer_sizes_percentage(analyses_layers, output_dir):
    """Three side-by-side bar charts comparing layer sizes as percentage of total"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    for ax_idx, (multiplex_name, layers) in enumerate(analyses_layers.items()):
        cell_types = sorted(layers.keys())
        
        # Calculate totals
        total_nodes = sum(layers[ct].number_of_nodes() for ct in cell_types)
        total_edges = sum(layers[ct].number_of_edges() for ct in cell_types)
        
        node_pcts = [layers[ct].number_of_nodes() / total_nodes * 100 for ct in cell_types]
        edge_pcts = [layers[ct].number_of_edges() / total_edges * 100 for ct in cell_types]
        
        x = np.arange(len(cell_types))
        width = 0.35
        
        ax = axes[ax_idx]
        bars1 = ax.bar(x - width/2, node_pcts, width, label='Nodes %', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, edge_pcts, width, label='Edges %', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Cell Type', fontweight='bold')
        ax.set_ylabel('Percentage of Total (%)', fontweight='bold')
        ax.set_title(f'{multiplex_name}\nLayer Sizes (% of Total)', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        
        # Adaptive font size based on number of layers
        fontsize = max(6, min(9, 150 / len(cell_types)))
        ax.set_xticklabels(cell_types, rotation=90, ha='right', fontsize=fontsize)
        ax.legend()
        ax.set_ylim(0, max(max(node_pcts), max(edge_pcts)) * 1.1)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_layer_sizes_percentage.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: 02_layer_sizes_percentage.png")
    plt.close()

def plot_seed_presence_comparison(analyses_layers, seed_genes, output_dir):
    """Three side-by-side bar charts comparing seed gene presence"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    for ax_idx, (multiplex_name, layers) in enumerate(analyses_layers.items()):
        cell_types = sorted(layers.keys())
        
        # Calculate seed stats
        seed_counts = []
        seed_pct_layer = []
        seed_pct_multiplex = []
        
        total_seeds_in_multiplex = len(seed_genes.intersection(
            set().union(*[set(G.nodes()) for G in layers.values()])
        ))
        
        for ct in cell_types:
            seeds_in_layer = seed_genes.intersection(layers[ct].nodes())
            n_seeds = len(seeds_in_layer)
            seed_counts.append(n_seeds)
            
            layer_size = layers[ct].number_of_nodes()
            pct_layer = n_seeds / layer_size * 100 if layer_size > 0 else 0
            seed_pct_layer.append(pct_layer)
            
            pct_multiplex = n_seeds / total_seeds_in_multiplex * 100 if total_seeds_in_multiplex > 0 else 0
            seed_pct_multiplex.append(pct_multiplex)
        
        x = np.arange(len(cell_types))
        width = 0.25
        
        ax = axes[ax_idx]
        bars1 = ax.bar(x - width, seed_counts, width, label='Absolute Count', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, seed_pct_layer, width, label='% of Layer', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, seed_pct_multiplex, width, label='% of Multiplex', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Cell Type', fontweight='bold')
        ax.set_ylabel('Count / Percentage', fontweight='bold')
        ax.set_title(f'{multiplex_name}\nSeed Gene Presence', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        
        # Adaptive font size based on number of layers
        fontsize = max(6, min(9, 150 / len(cell_types)))
        ax.set_xticklabels(cell_types, rotation=90, ha='right', fontsize=fontsize)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_seed_presence_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: 03_seed_presence_comparison.png")
    plt.close()

def plot_lcc_comparison(analyses_layers, output_dir):
    """Three side-by-side bar charts comparing LCC sizes"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    for ax_idx, (multiplex_name, layers) in enumerate(analyses_layers.items()):
        cell_types = sorted(layers.keys())
        
        lcc_abs = []
        lcc_pct = []
        
        for ct in cell_types:
            G = layers[ct]
            largest_cc = max(nx.connected_components(G), default=set())
            lcc_size = len(largest_cc)
            lcc_abs.append(lcc_size)
            
            n_nodes = G.number_of_nodes()
            pct = lcc_size / n_nodes * 100 if n_nodes > 0 else 0
            lcc_pct.append(pct)
        
        x = np.arange(len(cell_types))
        width = 0.35
        
        ax = axes[ax_idx]
        bars1 = ax.bar(x - width/2, lcc_abs, width, label='Absolute Size', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, lcc_pct, width, label='% of Layer', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Cell Type', fontweight='bold')
        ax.set_ylabel('Count / Percentage', fontweight='bold')
        ax.set_title(f'{multiplex_name}\nLargest Connected Component', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        
        # Adaptive font size based on number of layers
        fontsize = max(6, min(9, 150 / len(cell_types)))
        ax.set_xticklabels(cell_types, rotation=90, ha='right', fontsize=fontsize)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_lcc_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: 04_lcc_comparison.png")
    plt.close()

def plot_gene_venn_nonseed(analyses_layers, seed_genes, output_dir):
    """Venn diagram of non-seed genes across three multiplexes"""
    gene_sets = {}
    
    for name, layers in analyses_layers.items():
        all_genes = set().union(*[set(G.nodes()) for G in layers.values()])
        non_seed_genes = all_genes - seed_genes
        gene_sets[name] = non_seed_genes
    
    names = list(gene_sets.keys())
    sets = list(gene_sets.values())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(sets) == 3:
        venn3(sets, set_labels=names, ax=ax)
    else:
        venn2(sets[:2], set_labels=names[:2], ax=ax)
    
    ax.set_title('Non-Seed Gene Set Overlap Across Multiplexes', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_venn_nonseed_genes.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: 05_venn_nonseed_genes.png")
    plt.close()

def plot_gene_venn_seed(analyses_layers, seed_genes, output_dir):
    """Venn diagram of seed genes across three multiplexes"""
    gene_sets = {}
    
    for name, layers in analyses_layers.items():
        all_genes = set().union(*[set(G.nodes()) for G in layers.values()])
        seed_genes_in_multiplex = all_genes.intersection(seed_genes)
        gene_sets[name] = seed_genes_in_multiplex
    
    names = list(gene_sets.keys())
    sets = list(gene_sets.values())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(sets) == 3:
        venn3(sets, set_labels=names, ax=ax)
    else:
        venn2(sets[:2], set_labels=names[:2], ax=ax)
    
    ax.set_title('Seed Gene Set Overlap Across Multiplexes', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_venn_seed_genes.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: 06_venn_seed_genes.png")
    plt.close()

def plot_jaccard_similarities(analyses_layers, output_dir):
    """Jaccard similarity heatmaps for each multiplex"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax_idx, (multiplex_name, layers) in enumerate(analyses_layers.items()):
        cell_types = sorted(layers.keys())
        n = len(cell_types)
        matrix = np.zeros((n, n))
        
        for i, ct1 in enumerate(cell_types):
            for j, ct2 in enumerate(cell_types):
                s1 = set(layers[ct1].nodes())
                s2 = set(layers[ct2].nodes())
                union = s1 | s2
                matrix[i, j] = len(s1 & s2) / len(union) if union else 0
        
        ax = axes[ax_idx]
        sns.heatmap(matrix, xticklabels=cell_types, yticklabels=cell_types,
                    cmap='YlOrRd', ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Jaccard Similarity'})
        ax.set_title(f'{multiplex_name}\nJaccard Similarity', fontweight='bold', fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_jaccard_similarities.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: 07_jaccard_similarities.png")
    plt.close()

def get_ubiquitous_genes(layers, percentile=90):
    """Get top percentile most ubiquitous genes"""
    gene_counts = defaultdict(int)
    
    for G in layers.values():
        for gene in G.nodes():
            gene_counts[gene] += 1
    
    if not gene_counts:
        return set()
    
    threshold = np.percentile(list(gene_counts.values()), percentile)
    ubiquitous = {gene for gene, count in gene_counts.items() if count >= threshold}
    
    return ubiquitous

def plot_ubiquitous_genes_venn(analyses_layers, output_dir):
    """Venn diagram of top 10% most ubiquitous genes"""
    gene_sets = {}
    
    for name, layers in analyses_layers.items():
        ubiquitous = get_ubiquitous_genes(layers, percentile=90)
        gene_sets[name] = ubiquitous
    
    names = list(gene_sets.keys())
    sets = list(gene_sets.values())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(sets) == 3:
        venn3(sets, set_labels=names, ax=ax)
    else:
        venn2(sets[:2], set_labels=names[:2], ax=ax)
    
    ax.set_title('Top 10% Most Ubiquitous Genes Overlap', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_venn_ubiquitous_genes.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: 08_venn_ubiquitous_genes.png")
    plt.close()

def analyze_ubiquitous_genes(analyses_layers, seed_genes, output_dir):
    """Analyze and report top 10% ubiquitous genes"""
    results = {}
    
    for multiplex_name, layers in analyses_layers.items():
        gene_counts = defaultdict(int)
        
        for G in layers.values():
            for gene in G.nodes():
                gene_counts[gene] += 1
        
        # Get top 10%
        threshold = np.percentile(list(gene_counts.values()), 90)
        ubiquitous_genes = {gene: count for gene, count in gene_counts.items() if count >= threshold}
        
        # Sort by count
        sorted_genes = sorted(ubiquitous_genes.items(), key=lambda x: x[1], reverse=True)
        
        # Create dataframe
        df = pd.DataFrame([
            {
                'Gene': gene,
                'Layers Present': count,
                'Is Seed': gene in seed_genes,
                'Total Layers': len(layers)
            }
            for gene, count in sorted_genes
        ])
        
        results[multiplex_name] = df
        
        # Save to CSV
        csv_file = os.path.join(output_dir, f'ubiquitous_genes_{multiplex_name.replace(" ", "_")}.csv')
        df.to_csv(csv_file, index=False)
        print(f"Saved: ubiquitous_genes_{multiplex_name.replace(' ', '_')}.csv")
        
        # Print summary
        print(f"\n{multiplex_name} - Top 10% Ubiquitous Genes (n={len(df)}):")
        print(df.head(20).to_string(index=False))
    
    return results

def filter_tabula_layers(layers, seed_genes, min_seeds=3):
    """Filter Tabula Sapiens layers: remove those with <min_seeds seed genes and outliers"""
    # First pass: remove low seed count layers
    filtered = {}
    for cell_type, G in layers.items():
        seeds_in_layer = len(seed_genes.intersection(G.nodes()))
        if seeds_in_layer >= min_seeds:
            filtered[cell_type] = G
    
    if not filtered:
        return layers  # Fallback if filtering removes everything
    
    # Second pass: identify and remove outliers (top 5% by node/edge count)
    node_counts = [G.number_of_nodes() for G in filtered.values()]
    edge_counts = [G.number_of_edges() for G in filtered.values()]
    
    node_95th = np.percentile(node_counts, 99)
    edge_95th = np.percentile(edge_counts, 99)
    
    final_filtered = {}
    for cell_type, G in filtered.items():
        if G.number_of_nodes() <= node_95th and G.number_of_edges() <= edge_95th:
            final_filtered[cell_type] = G
    
    return final_filtered if final_filtered else filtered

def plot_ubiquitous_genes_distribution(analyses_layers, output_dir):
    """Bar chart showing distribution of ubiquitous genes"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax_idx, (multiplex_name, layers) in enumerate(analyses_layers.items()):
        gene_counts = defaultdict(int)
        
        for G in layers.values():
            for gene in G.nodes():
                gene_counts[gene] += 1
        
        threshold = np.percentile(list(gene_counts.values()), 90)
        ubiquitous_genes = {gene: count for gene, count in gene_counts.items() if count >= threshold}
        
        # Get top 15
        sorted_genes = sorted(ubiquitous_genes.items(), key=lambda x: x[1], reverse=True)[:15]
        genes = [g[0] for g in sorted_genes]
        counts = [g[1] for g in sorted_genes]
        
        ax = axes[ax_idx]
        bars = ax.barh(genes, counts, alpha=0.8, edgecolor='black', color='mediumpurple')
        ax.invert_yaxis()
        ax.set_xlabel('Number of Layers', fontweight='bold')
        ax.set_title(f'{multiplex_name}\nTop 15 Ubiquitous Genes', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (gene, count) in enumerate(sorted_genes):
            ax.text(count + 0.05, i, str(int(count)), va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_ubiquitous_genes_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: 09_ubiquitous_genes_distribution.png")
    plt.close()

def create_summary_report(analyses_layers, seed_genes, output_dir):
    """Create a comprehensive text summary report"""
    report = []
    report.append("="*80)
    report.append("MULTIPLEX COMPARISON SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    
    for multiplex_name, layers in analyses_layers.items():
        report.append(f"\n{multiplex_name}")
        report.append("-" * len(multiplex_name))
        
        all_genes = set().union(*[set(G.nodes()) for G in layers.values()])
        all_edges = sum(G.number_of_edges() for G in layers.values())
        
        report.append(f"Number of layers: {len(layers)}")
        report.append(f"Total unique genes: {len(all_genes)}")
        report.append(f"Total edges: {all_edges}")
        
        seed_in_multiplex = all_genes.intersection(seed_genes)
        report.append(f"Seed genes in multiplex: {len(seed_in_multiplex)} ({len(seed_in_multiplex)/len(seed_genes)*100:.1f}% of total seeds)")
        
        # Ubiquitous genes
        ubiquitous = get_ubiquitous_genes(layers, percentile=90)
        report.append(f"Top 10% ubiquitous genes: {len(ubiquitous)}")
        
        ubiquitous_seeds = ubiquitous.intersection(seed_genes)
        report.append(f"  - Seed genes in ubiquitous set: {len(ubiquitous_seeds)}")
        
        report.append("")
    
    # Cross-multiplex comparison
    report.append("\nCross-Multiplex Comparison")
    report.append("-" * 25)
    
    all_gene_sets = {name: set().union(*[set(G.nodes()) for G in layers.values()])
                     for name, layers in analyses_layers.items()}
    
    names = list(all_gene_sets.keys())
    sets = list(all_gene_sets.values())
    
    if len(sets) == 3:
        all_three = sets[0] & sets[1] & sets[2]
        report.append(f"Genes in all three multiplexes: {len(all_three)}")
        
        pairwise = [
            (names[0], names[1], len(sets[0] & sets[1])),
            (names[0], names[2], len(sets[0] & sets[2])),
            (names[1], names[2], len(sets[1] & sets[2]))
        ]
        
        for n1, n2, count in pairwise:
            report.append(f"Genes in {n1} AND {n2}: {count}")
    
    report_text = "\n".join(report)
    
    report_file = os.path.join(output_dir, 'comparison_summary.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"\nSaved: comparison_summary.txt")
    print("\n" + report_text)

def main():
    parser = argparse.ArgumentParser(
        description='Advanced comparative visualization of multiplex networks'
    )
    parser.add_argument('--tabula-layers', type=str, required=True,
                      help='Directory with Tabula Sapiens multiplex layers')
    parser.add_argument('--pain-layers', type=str, required=True,
                      help='Directory with 2025 hDRG Atlas multiplex layers')
    parser.add_argument('--pain-subset-layers', type=str, required=True,
                      help='Directory with Pain subset of Tabula multiplex layers')
    parser.add_argument('--seed-genes', type=str, required=True,
                      help='Path to seed genes CSV file')
    parser.add_argument('--output-dir', type=str, default='multiplex_comparison_advanced',
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
  
    print("\nLoading 2025 hDRG Atlas multiplex...")
    pain_layers = load_multiplex_layers(args.pain_layers)
    print(f"Loaded {len(pain_layers)} layers")
  
    print("\nLoading Pain subset of Tabula multiplex...")
    pain_subset_layers = load_multiplex_layers(args.pain_subset_layers)
    print(f"Loaded {len(pain_subset_layers)} layers")
  
    # Filter layers to keep only those with seed genes
    print("\n" + "="*80)
    print("FILTERING LAYERS (keeping only those with seed genes)")
    print("="*80)
    
    # Tabula: apply strict filtering (remove low-priority and outliers)
    tabula_layers_filtered = filter_tabula_layers(tabula_layers, seed_genes, min_seeds=5)
    print(f"Tabula Sapiens: {len(tabula_layers)} → {len(tabula_layers_filtered)} layers (removed low-priority and outliers)")
    
    # Other multiplexes: standard filtering
    pain_layers_filtered = filter_layers_by_seed_genes(pain_layers, seed_genes)
    print(f"2025 hDRG Atlas: {len(pain_layers)} → {len(pain_layers_filtered)} layers (removed {len(pain_layers) - len(pain_layers_filtered)} without seed genes)")
    
    pain_subset_layers_filtered = filter_layers_by_seed_genes(pain_subset_layers, seed_genes)
    print(f"Pain Subset (Tabula): {len(pain_subset_layers)} → {len(pain_subset_layers_filtered)} layers (removed {len(pain_subset_layers) - len(pain_subset_layers_filtered)} without seed genes)")
    
    analyses_layers = {
        "Tabula Sapiens": tabula_layers_filtered,
        "2025 hDRG Atlas": pain_layers_filtered,
        "Pain Subset (Tabula)": pain_subset_layers_filtered
    }
  
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
  
    print("\nGenerating layer size comparisons...")
    plot_layer_sizes_comparison(analyses_layers, args.output_dir)
    plot_layer_sizes_percentage(analyses_layers, args.output_dir)
  
    print("Generating seed gene presence comparisons...")
    plot_seed_presence_comparison(analyses_layers, seed_genes, args.output_dir)
  
    print("Generating LCC comparisons...")
    plot_lcc_comparison(analyses_layers, args.output_dir)
  
    print("Generating Venn diagrams...")
    plot_gene_venn_nonseed(analyses_layers, seed_genes, args.output_dir)
    plot_gene_venn_seed(analyses_layers, seed_genes, args.output_dir)
  
    print("Generating Jaccard similarity heatmaps...")
    plot_jaccard_similarities(analyses_layers, args.output_dir)
  
    print("Generating ubiquitous genes analysis...")
    plot_ubiquitous_genes_venn(analyses_layers, args.output_dir)
    plot_ubiquitous_genes_distribution(analyses_layers, args.output_dir)
    ubiquitous_results = analyze_ubiquitous_genes(analyses_layers, seed_genes, args.output_dir)
  
    print("Generating summary report...")
    create_summary_report(analyses_layers, seed_genes, args.output_dir)
  
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"  01_layer_sizes_absolute.png")
    print(f"  02_layer_sizes_percentage.png")
    print(f"  03_seed_presence_comparison.png")
    print(f"  04_lcc_comparison.png")
    print(f"  05_venn_nonseed_genes.png")
    print(f"  06_venn_seed_genes.png")
    print(f"  07_jaccard_similarities.png")
    print(f"  08_venn_ubiquitous_genes.png")
    print(f"  09_ubiquitous_genes_distribution.png")
    print(f"  ubiquitous_genes_*.csv (3 files)")
    print(f"  comparison_summary.txt")
  
    return 0

if __name__ == "__main__":
    sys.exit(main())