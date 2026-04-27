#!/usr/bin/env python3
"""
Generate Venn diagrams for multiplex network comparison
- Unique nodes per multiplex source
- Unique edges per multiplex source
"""

import os
import sys
import argparse
import networkx as nx
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib_venn import venn3, venn3_circles
import warnings
warnings.filterwarnings('ignore')

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

def get_all_nodes(layers):
    """Get all unique nodes from all layers in a multiplex"""
    all_nodes = set()
    for layer in layers.values():
        all_nodes.update(layer.nodes())
    return all_nodes

def get_all_edges(layers):
    """Get all unique edges from all layers in a multiplex (normalized)"""
    all_edges = set()
    for layer in layers.values():
        for u, v in layer.edges():
            # Normalize edge representation (smaller node first)
            edge = (min(u, v), max(u, v))
            all_edges.add(edge)
    return all_edges

def add_arrow_labels(ax, venn, region_id, count, set_sizes, threshold_pct=10):
    """Add arrow labels for small regions that are hard to read (relative to set size)"""
    label = venn.get_label_by_id(region_id)
    if label is None:
        return
    
    label.set_text(f"{count}")
    
    # Determine which set(s) this region belongs to and calculate relative size
    # Region IDs: 100=set0 only, 010=set1 only, 001=set2 only, 110=set0&set1, 101=set0&set2, 011=set1&set2, 111=all
    region_to_sets = {
        '100': [0],
        '010': [1],
        '001': [2],
        '110': [0, 1],
        '101': [0, 2],
        '011': [1, 2],
        '111': [0, 1, 2]
    }
    
    set_indices = region_to_sets.get(region_id, [])
    
    # Calculate relative size: count / average size of relevant sets
    if set_indices:
        relevant_sizes = [set_sizes[i] for i in set_indices]
        avg_set_size = sum(relevant_sizes) / len(relevant_sizes)
        relative_pct = (count / avg_set_size * 100) if avg_set_size > 0 else 0
    else:
        relative_pct = 100
    
    # If relative size is small, move label outside with arrow
    if relative_pct < threshold_pct:
        label.set_visible(False)
        
        # Get label position
        x, y = label.get_position()
        
        # Define offset positions for each region (adjusted for readability)
        offsets = {
            '100': (-0.30, 0.0),   # Left circle - far left
            '010': (0.30, 0.0),    # Right circle - far right
            '001': (0.0, -0.35),   # Bottom circle - far bottom
            '110': (0.0, 0.25),    # Top overlap - above
            '101': (-0.25, -0.25), # Bottom-left overlap - outside
            '011': (0.25, -0.25),  # Bottom-right overlap - outside
            '111': (0.0, 0.35)     # Center - above all
        }
        
        offset = offsets.get(region_id, (0.1, 0.1))
        new_x = x + offset[0]
        new_y = y + offset[1]
        
        # Add text at offset position
        ax.text(new_x, new_y, f"{count}", fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=1.0, edgecolor='black'),
                ha='center', va='center', zorder=100)
        
        # Draw arrow from original position to new position
        ax.annotate('', xy=(new_x, new_y), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.6),
                   zorder=50)

def plot_venn_nodes(analyses_layers, output_dir):
    """Create Venn diagram for node overlap"""
    print("\nGenerating node overlap Venn diagram...")
    
    node_sets = {}
    for name, layers in analyses_layers.items():
        node_sets[name] = get_all_nodes(layers)
    
    names = list(node_sets.keys())
    sets = list(node_sets.values())
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if len(sets) == 3:
        venn = venn3(sets, set_labels=names, ax=ax)
        venn3_circles(sets, ax=ax, linewidth=1.5)
        
        # Calculate set sizes for relative threshold calculation
        set_sizes = [len(s) for s in sets]
        
        # Calculate all regions
        regions = {
            '100': len(sets[0] - sets[1] - sets[2]),
            '010': len(sets[1] - sets[0] - sets[2]),
            '001': len(sets[2] - sets[0] - sets[1]),
            '110': len((sets[0] & sets[1]) - sets[2]),
            '101': len((sets[0] & sets[2]) - sets[1]),
            '011': len((sets[1] & sets[2]) - sets[0]),
            '111': len(sets[0] & sets[1] & sets[2])
        }
        
        # Add labels with arrows for small regions (relative to set size)
        for region_id, count in regions.items():
            add_arrow_labels(ax, venn, region_id, count, set_sizes, threshold_pct=10)
        
        ax.set_title('Node Overlap Across Multiplexes', fontweight='bold', fontsize=14)
    
    elif len(sets) == 2:
        from matplotlib_venn import venn2, venn2_circles
        venn = venn2(sets, set_labels=names, ax=ax)
        venn2_circles(sets, ax=ax, linewidth=1.5)
        ax.set_title('Node Overlap Across Multiplexes', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'venn_nodes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: venn_nodes.png")
    
    # Print summary
    print("\nNode Overlap Summary:")
    for name, nodes in node_sets.items():
        print(f"  {name}: {len(nodes)} nodes")
    
    if len(sets) == 3:
        print(f"  All three: {len(sets[0] & sets[1] & sets[2])} nodes")
        print(f"  {names[0]} ∩ {names[1]}: {len(sets[0] & sets[1])} nodes")
        print(f"  {names[0]} ∩ {names[2]}: {len(sets[0] & sets[2])} nodes")
        print(f"  {names[1]} ∩ {names[2]}: {len(sets[1] & sets[2])} nodes")

def plot_venn_edges(analyses_layers, output_dir):
    """Create Venn diagram for edge overlap"""
    print("\nGenerating edge overlap Venn diagram...")
    
    edge_sets = {}
    for name, layers in analyses_layers.items():
        edge_sets[name] = get_all_edges(layers)
    
    names = list(edge_sets.keys())
    sets = list(edge_sets.values())
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if len(sets) == 3:
        venn = venn3(sets, set_labels=names, ax=ax)
        venn3_circles(sets, ax=ax, linewidth=1.5)
        
        # Calculate set sizes for relative threshold calculation
        set_sizes = [len(s) for s in sets]
        
        # Calculate all regions
        regions = {
            '100': len(sets[0] - sets[1] - sets[2]),
            '010': len(sets[1] - sets[0] - sets[2]),
            '001': len(sets[2] - sets[0] - sets[1]),
            '110': len((sets[0] & sets[1]) - sets[2]),
            '101': len((sets[0] & sets[2]) - sets[1]),
            '011': len((sets[1] & sets[2]) - sets[0]),
            '111': len(sets[0] & sets[1] & sets[2])
        }
        
        # Add labels with arrows for small regions (relative to set size)
        for region_id, count in regions.items():
            add_arrow_labels(ax, venn, region_id, count, set_sizes, threshold_pct=10)
        
        ax.set_title('Edge Overlap Across Multiplexes', fontweight='bold', fontsize=14)
    
    elif len(sets) == 2:
        from matplotlib_venn import venn2, venn2_circles
        venn = venn2(sets, set_labels=names, ax=ax)
        venn2_circles(sets, ax=ax, linewidth=1.5)
        ax.set_title('Edge Overlap Across Multiplexes', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'venn_edges.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: venn_edges.png")
    
    # Print summary
    print("\nEdge Overlap Summary:")
    for name, edges in edge_sets.items():
        print(f"  {name}: {len(edges)} edges")
    
    if len(sets) == 3:
        print(f"  All three: {len(sets[0] & sets[1] & sets[2])} edges")
        print(f"  {names[0]} ∩ {names[1]}: {len(sets[0] & sets[1])} edges")
        print(f"  {names[0]} ∩ {names[2]}: {len(sets[0] & sets[2])} edges")
        print(f"  {names[1]} ∩ {names[2]}: {len(sets[1] & sets[2])} edges")

def main():
    parser = argparse.ArgumentParser(description='Generate Venn diagrams for multiplex network comparison')
    parser.add_argument('--tabula-layers', type=str, required=True)
    parser.add_argument('--pain-layers', type=str, required=True)
    parser.add_argument('--pain-subset-layers', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='multiplex_comparison')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
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
    print("GENERATING VENN DIAGRAMS")
    print("="*80)
    
    plot_venn_nodes(analyses_layers, args.output_dir)
    plot_venn_edges(analyses_layers, args.output_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Output: {args.output_dir}")
    print("Files: venn_nodes.png, venn_edges.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())