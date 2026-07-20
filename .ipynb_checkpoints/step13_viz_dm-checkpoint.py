#!/usr/bin/env python3
"""
Multiplex-aware disease module visualization.
Shows: (1) full multiplex, (2) seeds, (3) seeds+neighbors, (4) final module
Iteratively developed version: Spectral analysis optional.
"""
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────
# LAYER-AWARE UTILITIES
# ─────────────────────────────────────────────────────────────────────

LAYER_COLORS = {
    'ppi': '#E74C3C',
    'pathway': '#2ECC71',
    'ligand_receptor': '#3498DB',
}
DEFAULT_LAYER_COLOR = '#95A5A6'

def get_layers(G):
    """Extract unique edge types from multiplex."""
    layers = set()
    for u, v, d in G.edges(data=True):
        et = d.get('edge_type', 'ppi')
        layers.add(et)
    return sorted(layers)

def get_nodes_in_layer(G, layer):
    """Get nodes that participate in a layer."""
    nodes = set()
    for u, v, d in G.edges(data=True):
        if d.get('edge_type') == layer:
            nodes.add(u)
            nodes.add(v)
    return nodes

def get_edges_in_layer(G, layer):
    """Get edges in a specific layer."""
    return [(u, v) for u, v, d in G.edges(data=True) 
            if d.get('edge_type') == layer]

def get_dominant_layer(G, node):
    """Most connected layer for a node."""
    layer_degree = {}
    for layer in get_layers(G):
        edges = get_edges_in_layer(G, layer)
        degree = sum(1 for u, v in edges if u == node or v == node)
        layer_degree[layer] = degree
    return max(layer_degree, key=layer_degree.get) if layer_degree else None

# ─────────────────────────────────────────────────────────────────────
# LOAD DATA (CORRECTED PATHS & FLEXIBLE SPECTRAL)
# ─────────────────────────────────────────────────────────────────────

def load_data(graph_pkl, dm_csv, seed_csv, spectral_pkl=None, out_dir=None):
    """
    Load multiplex graph, disease module, seeds.
    Spectral data is OPTIONAL (can pass None or nonexistent path).
    """
    print("Loading data...")
    
    # Full multiplex
    print(f"  Loading multiplex from: {graph_pkl}")
    with open(graph_pkl, 'rb') as f:
        G_full = pickle.load(f)
    
    # Disease module genes
    print(f"  Loading disease module from: {dm_csv}")
    dm_df = pd.read_csv(dm_csv)
    disease_module = set(dm_df['gene'].str.upper())
    seed_genes_in_net = set(dm_df[dm_df['is_seed']]['gene'].str.upper())
    neighbor_genes = disease_module - seed_genes_in_net
    
    # All other genes
    other_genes = set(G_full.nodes()) - disease_module
    
    # Spectral data (optional)
    fiedler_set = set()
    hub_set = set()
    spectral_note = "not loaded (optional)"
    
    if spectral_pkl and Path(spectral_pkl).exists():
        try:
            print(f"  Loading spectral data from: {spectral_pkl}")
            with open(spectral_pkl, 'rb') as f:
                spectral_data = pickle.load(f)
            fiedler_set = set(spectral_data.get('fiedler_nodes', []))
            hub_set = set(spectral_data.get('hub_nodes', []))
            spectral_note = f"loaded: {len(fiedler_set)} fiedler, {len(hub_set)} hubs"
        except Exception as e:
            print(f"  ⚠ Could not load spectral data: {e}")
            spectral_note = f"failed to load ({str(e)[:50]})"
    
    print(f"  Full multiplex: {G_full.number_of_nodes():,} nodes, {G_full.number_of_edges():,} edges")
    print(f"  Disease module: {len(disease_module):,} genes ({len(seed_genes_in_net):,} seeds)")
    print(f"  Spectral analysis: {spectral_note}")
    print(f"  Layers: {get_layers(G_full)}\n")
    
    return G_full, disease_module, seed_genes_in_net, neighbor_genes, other_genes, fiedler_set, hub_set

# ─────────────────────────────────────────────────────────────────────
# LAYOUT (UMAP WITH CACHING)
# ─────────────────────────────────────────────────────────────────────

def compute_layout(G, layers, cache_path=None, use_spring=False):
    """
    Compute layout with caching.
    - use_spring=False: tries UMAP (slower but prettier)
    - use_spring=True: uses spring layout (faster, for quick iterations)
    """
    if cache_path and cache_path.exists():
        print(f"Loading cached layout from {cache_path}...")
        return pickle.load(open(cache_path, 'rb'))
    
    if use_spring or G.number_of_nodes() > 5000:
        print(f"Graph has {G.number_of_nodes():,} nodes → using spring layout (faster)")
        print("  Computing spring layout (Fruchterman-Reingold)...")
        pos = nx.spring_layout(G, k=0.5, iterations=20, seed=42)
        # Scale to [0, 100]
        for node in pos:
            pos[node] = tuple(x * 100 for x in pos[node])
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(pos, open(cache_path, 'wb'))
            print(f"  Cached layout to {cache_path}")
        return pos
    
    print("Computing UMAP layout (this may take a few minutes)...")
    try:
        import umap
    except ImportError:
        print("  ⚠ umap not installed, falling back to spring layout")
        return nx.spring_layout(G, k=0.5, iterations=20, seed=42)
    
    nodes = list(G.nodes())
    
    # Simple features: degree per layer
    features = []
    for node in nodes:
        row = []
        for layer in layers:
            degree = sum(1 for u, v in get_edges_in_layer(G, layer) 
                        if u == node or v == node)
            row.append(degree)
        features.append(row)
    
    features = np.array(features, dtype=float)
    
    # Normalize
    for col in range(features.shape[1]):
        lo, hi = features[:, col].min(), features[:, col].max()
        if hi > lo:
            features[:, col] = (features[:, col] - lo) / (hi - lo)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(features)
    
    # Scale to [0, 100]
    for dim in range(2):
        lo, hi = embedding[:, dim].min(), embedding[:, dim].max()
        embedding[:, dim] = (embedding[:, dim] - lo) / (hi - lo) * 100
    
    pos = {node: embedding[i] for i, node in enumerate(nodes)}
    
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(pos, open(cache_path, 'wb'))
        print(f"  Cached layout to {cache_path}")
    
    return pos

# ─────────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────────

def draw_panel(ax, G, pos, layers, panel_type, disease_module, 
               seed_genes, neighbor_genes, other_genes, title, subtitle):
    """Draw one visualization panel."""
    
    ax.set_facecolor('white')
    
    # EDGES: color by layer
    for layer in layers:
        layer_color = LAYER_COLORS.get(layer, DEFAULT_LAYER_COLOR)
        layer_edges = get_edges_in_layer(G, layer)
        
        xs, ys = [], []
        for u, v in layer_edges:
            # Filter based on panel type
            if panel_type == 'full':
                pass  # Show all
            elif panel_type == 'seeds':
                if not (u in seed_genes or v in seed_genes):
                    continue
            elif panel_type == 'seeds_neighbors':
                if not ((u in seed_genes or v in seed_genes) or 
                       (u in neighbor_genes or v in neighbor_genes)):
                    continue
            else:  # module_only
                if not (u in disease_module or v in disease_module):
                    continue
            
            if u not in pos or v not in pos:
                continue
            
            xu, yu = pos[u]
            xv, yv = pos[v]
            xs += [xu, xv, None]
            ys += [yu, yv, None]
        
        if xs:
            ax.plot(xs, ys, color=layer_color, alpha=0.3, linewidth=0.5, zorder=1)
    
    # NODES: by role
    node_colors = {}
    node_sizes = {}
    
    for node in G.nodes():
        if node in seed_genes:
            node_colors[node] = '#1B88A7'
            node_sizes[node] = 80
        elif node in neighbor_genes:
            node_colors[node] = '#BEE3EF'
            node_sizes[node] = 40
        else:
            node_colors[node] = '#e8e8e8'
            node_sizes[node] = 20
    
    # Draw nodes
    for node in G.nodes():
        if node not in pos:
            continue
        ax.scatter(*pos[node], c=node_colors[node], s=node_sizes[node],
                  zorder=4, edgecolors='white', linewidths=0.5)
    
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.text(0.5, -0.05, subtitle, transform=ax.transAxes, 
           fontsize=14, ha='center', color='#666')
    ax.axis('off')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1B88A7',
              markersize=10, label='Seeds'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#BEE3EF',
              markersize=10, label='Neighbors'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e8e8e8',
              markersize=10, label='Other'),
    ]
    for layer in layers:
        legend_elements.append(
            Line2D([0], [0], color=LAYER_COLORS.get(layer, DEFAULT_LAYER_COLOR),
                  lw=3, label=layer)
        )
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualize disease module across 4 panels with optional spectral highlights"
    )
    ap.add_argument("--graph",      required=True, help="integrated multiplex .pkl")
    ap.add_argument("--dm-csv",     required=True, help="disease_module.csv")
    ap.add_argument("--seed-csv",   required=True, help="seed_genes.csv")
    ap.add_argument("--spectral",   default=None,  help="spectral_analysis.pkl (optional)")
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--use-umap",   action="store_true", 
                   help="Use UMAP layout (slow but prettier). Default: spring layout (fast)")
    args = ap.parse_args()
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Load
    G, disease_module, seeds, neighbors, others, fiedler, hubs = \
        load_data(args.graph, args.dm_csv, args.seed_csv, args.spectral, out)
    
    layers = get_layers(G)
    use_spring = not args.use_umap
    print(f"Layout method: {'spring (fast)' if use_spring else 'UMAP (slower, prettier)'}\n")
    pos = compute_layout(G, layers, out / 'layout.pkl', use_spring=use_spring)
    
    # Draw 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.patch.set_facecolor('white')
    
    panels = [
        (axes[0, 0], 'full', 'Step 1: Full Multiplex', 
         f'{G.number_of_nodes():,} proteins, {len(layers)} layers'),
        (axes[0, 1], 'seeds', 'Step 2: Seeds Only',
         f'{len(seeds):,} seed genes'),
        (axes[1, 0], 'seeds_neighbors', 'Step 3: Seeds + Neighbors',
         f'{len(seeds):,} seeds + {len(neighbors):,} neighbors'),
        (axes[1, 1], 'module_only', 'Step 4: Disease Module',
         f'{len(disease_module):,} total genes'),
    ]
    
    for ax, ptype, title, subtitle in panels:
        draw_panel(ax, G, pos, layers, ptype, disease_module, 
                  seeds, neighbors, others, title, subtitle)
    
    plt.tight_layout()
    plt.savefig(out / 'disease_module_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {out / 'disease_module_visualization.png'}")
    plt.close()

if __name__ == "__main__":
    main()