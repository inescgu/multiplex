#!/usr/bin/env python3
"""
Prepare Multiplex RWR: Set up directory structure and configuration for MultiXrank

This script prepares the necessary file structure and configuration for running 
Random Walk with Restart on a multiplex interactome using the MultiXrank package.
"""

import os
import sys
import argparse
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import time
import yaml
from pathlib import Path


def load_multiplex_layers(multiplex_dir, min_edges=10):
    """
    Load all layer networks from multiplex directory
    
    Parameters:
    - multiplex_dir: directory containing multiplex layers
    - min_edges: minimum number of edges required to include layer (default: 10)
    
    Returns:
    - Dictionary of valid layers
    """
    print(f"Loading multiplex layers from {multiplex_dir}...")
    print(f"Filtering layers with at least {min_edges} edges...")
    
    layers_dir = os.path.join(multiplex_dir, 'layers')
    
    if not os.path.exists(layers_dir):
        raise FileNotFoundError(f"Layers directory not found: {layers_dir}")
    
    layers = {}
    filtered_layers = {}
    layer_files = sorted([f for f in os.listdir(layers_dir) if f.endswith('_layer.pkl')])
    
    for layer_file in layer_files:
        layer_name = layer_file.replace('_layer.pkl', '')
        layer_path = os.path.join(layers_dir, layer_file)
        
        with open(layer_path, 'rb') as f:
            layer_graph = pickle.load(f)
        
        n_nodes = layer_graph.number_of_nodes()
        n_edges = layer_graph.number_of_edges()
        
        layers[layer_name] = layer_graph
        
        # Filter by minimum edges
        if n_edges >= min_edges:
            filtered_layers[layer_name] = layer_graph
            print(f"  ✓ {layer_name}: {n_nodes} nodes, {n_edges} edges")
        else:
            print(f"  ✗ {layer_name}: {n_nodes} nodes, {n_edges} edges (FILTERED)")
    
    print(f"\nLayer filtering summary:")
    print(f"  Total layers: {len(layers)}")
    print(f"  Valid layers (>={min_edges} edges): {len(filtered_layers)}")
    print(f"  Filtered out: {len(layers) - len(filtered_layers)}")
    
    if len(filtered_layers) == 0:
        raise ValueError(f"No layers have at least {min_edges} edges! Try lowering --min-edges threshold.")
    
    return filtered_layers


def export_layers_to_tsv(layers, output_dir):
    """
    Export network layers to TSV format required by MultiXrank
    
    MultiXrank expects edge lists as two-column (unweighted) or three-column (weighted) 
    TSV files without headers.
    
    Parameters:
    - layers: dict of layer_name -> NetworkX graph
    - output_dir: directory to save TSV files
    
    Returns:
    - Dictionary mapping layer names to TSV file paths
    """
    print("\nExporting layers to TSV format for MultiXrank...")
    
    multiplex_dir = os.path.join(output_dir, 'multiplex', '1')
    os.makedirs(multiplex_dir, exist_ok=True)
    
    layer_files = {}
    
    for layer_name, graph in layers.items():
        tsv_filename = f"{layer_name}.tsv"
        tsv_path = os.path.join(multiplex_dir, tsv_filename)
        
        # Write edge list as TSV (source, target, weight)
        with open(tsv_path, 'w') as f:
            for u, v, data in graph.edges(data=True):
                weight = data.get('weight', 1.0)
                f.write(f"{u}\t{v}\t{weight}\n")
        
        # Store relative path for config
        layer_files[layer_name] = f"multiplex/1/{tsv_filename}"
        print(f"  Exported {layer_name}: {graph.number_of_edges()} edges -> {tsv_path}")
    
    return layer_files


def create_multixrank_config(layer_files, output_dir, restart_prob=0.7, 
                              delta=None, tau=None, eta=None):
    """
    Create MultiXrank configuration YAML file
    
    Parameters:
    - layer_files: dict mapping layer names to relative TSV paths
    - output_dir: directory to save config
    - restart_prob: restart probability (r parameter, default 0.7)
    - delta: inter-layer jump probability within multiplex (default: auto)
    - tau: restart probability distribution across layers (default: uniform)
    - eta: restart probability distribution across multiplexes (default: [1])
    
    Returns:
    - Path to config file
    """
    print("\nCreating MultiXrank configuration file...")
    
    n_layers = len(layer_files)
    
    # Build multiplex configuration
    config = {
        'multiplex': {
            1: {
                'layers': list(layer_files.values())
            }
        },
        'seed': 'seeds.txt',
        'r': restart_prob
    }
    
    # DON'T add delta at the top level - it causes a bug in MultiXrank
    # Delta should be set per-multiplex, not globally
    # MultiXrank will use default delta=0.5 for multiplex with >1 layer
    
    # Add delta to the multiplex if user specified it
    if delta is not None:
        # Delta goes inside the multiplex config, not at top level
        config['multiplex'][1]['delta'] = delta
        print(f"  Setting delta={delta} for multiplex 1")
    else:
        print(f"  Using default delta (0.5 for {n_layers} layers)")
    
    # Set tau (restart distribution across layers within multiplex)
    # Tau goes inside the multiplex config
    if tau is None:
        tau = [1.0 / n_layers] * n_layers  # Uniform across layers (1D list)
    
    config['multiplex'][1]['tau'] = tau
    
    # Set eta (restart distribution across multiplexes)
    # Since we have only ONE multiplex, eta MUST be [1.0]
    eta_value = [1.0]  # ALWAYS [1.0] for single multiplex
    config['eta'] = eta_value
    
    # Write config file
    config_path = os.path.join(output_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  Config file created: {config_path}")
    print(f"  Restart probability (r): {restart_prob}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Tau (per-layer restart): {tau[:3]}... (showing first 3)")
    print(f"  Tau sum: {sum(tau):.4f} (should be ~1.0)")
    print(f"  Eta: {eta_value} (MUST be [1.0] for single multiplex)")
    
    # Read back and verify
    print("\n  Verifying written config...")
    with open(config_path, 'r') as f:
        written_config = yaml.safe_load(f)
        print(f"    Written eta: {written_config.get('eta', 'NOT FOUND')}")
        print(f"    Written delta: {written_config.get('multiplex', {}).get('1', {}).get('delta', 'default (0.5)')}")
        print(f"    Written tau length: {len(written_config.get('multiplex', {}).get('1', {}).get('tau', []))}")
    
    return config_path

def save_layer_info(layers, output_dir, original_n_layers=None):
    """Save layer metadata for downstream analysis"""
    print("\nSaving layer information...")
    
    all_genes = sorted(list(set().union(*[set(g.nodes()) for g in layers.values()])))
    layer_names = sorted(layers.keys())
    
    layer_info = {
        'layer_names': layer_names,
        'n_genes': len(all_genes),
        'n_layers': len(layers),
        'all_genes': all_genes,
        'gene_to_idx': {gene: idx for idx, gene in enumerate(all_genes)},
    }
    
    # Add filtering info if provided
    if original_n_layers is not None:
        layer_info['original_n_layers'] = original_n_layers
        layer_info['n_filtered_layers'] = original_n_layers - len(layers)
    
    # Add layer-specific gene lists and stats
    layer_info['layer_genes'] = {}
    layer_info['layer_stats'] = {}
    
    for name, graph in layers.items():
        layer_info['layer_genes'][name] = list(graph.nodes())
        layer_info['layer_stats'][name] = {
            'n_nodes': graph.number_of_nodes(),
            'n_edges': graph.number_of_edges(),
            'density': nx.density(graph) if graph.number_of_nodes() > 0 else 0
        }
    
    layer_info_file = os.path.join(output_dir, 'multiplex_layer_info.pkl')
    with open(layer_info_file, 'wb') as f:
        pickle.dump(layer_info, f, protocol=4)
    
    print(f"  Saved layer info: {layer_info_file}")
    
    return layer_info


def load_and_prepare_seed_genes(seed_file, layer_info, output_dir):
    """
    Load seed genes and capitalize them to match network genes
    
    Parameters:
    - seed_file: path to seed genes CSV file
    - layer_info: dictionary with layer information (contains all_genes in network)
    - output_dir: directory to save processed seeds.txt
    
    Returns:
    - List of seed genes that are in the network (capitalized)
    """
    import pandas as pd
    
    print("\nPreparing seed genes for MultiXrank...")
    
    # Load seed genes
    try:
        seed_df = pd.read_csv(seed_file)
        
        # Find gene column
        gene_col = None
        for col in ['gene', 'Gene', 'gene_id', 'Gene_ID', 'symbol', 'Symbol']:
            if col in seed_df.columns:
                gene_col = col
                break
        
        if gene_col is None:
            gene_col = seed_df.columns[0]
        
        # Convert to string safely (handles int, float, NaN, etc.)
        seed_genes_raw = []
        for item in seed_df[gene_col].dropna():
            # Convert to string and skip if it's NaN or empty
            item_str = str(item).strip()
            if item_str and item_str.lower() != 'nan':
                seed_genes_raw.append(item_str)
        
        print(f"  Loaded {len(seed_genes_raw)} seed genes from {seed_file}")
        
    except Exception as e:
        print(f"  ERROR: Could not load seed genes: {str(e)}")
        raise
    
    # Capitalize to match network genes
    seed_genes_capitalized = [str(g).upper() for g in seed_genes_raw]
    
    # Get set of genes in network (already capitalized from build step)
    network_genes = set(layer_info['all_genes'])
    
    # Filter to seeds that are in network
    seed_genes_in_network = [g for g in seed_genes_capitalized if g in network_genes]
    seeds_not_in_network = set(seed_genes_capitalized) - set(seed_genes_in_network)
    
    print(f"  Seed genes in network: {len(seed_genes_in_network)} / {len(seed_genes_capitalized)}")
    
    if seeds_not_in_network:
        print(f"  WARNING: {len(seeds_not_in_network)} seed genes not found in network:")
        not_found_list = sorted(list(seeds_not_in_network))[:20]
        print(f"    {', '.join(not_found_list)}{'...' if len(seeds_not_in_network) > 20 else ''}")
    
    if not seed_genes_in_network:
        raise ValueError("No seed genes found in network! Check gene ID formats match.")
    
    # Write seeds.txt file for MultiXrank
    seeds_path = os.path.join(output_dir, 'seeds.txt')
    with open(seeds_path, 'w') as f:
        for gene in seed_genes_in_network:
            f.write(f"{gene}\n")
    
    print(f"  Created seeds.txt with {len(seed_genes_in_network)} genes: {seeds_path}")
    
    return seed_genes_in_network, seeds_not_in_network


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MultiXrank configuration for multiplex interactome'
    )
    parser.add_argument('--multiplex-dir', type=str, required=True,
                      help='Directory containing multiplex interactome')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for MultiXrank files')
    parser.add_argument('--restart', type=float, default=0.7,
                      help='Restart probability r (default: 0.7)')
    parser.add_argument('--delta', type=float, default=None,
                      help='Inter-layer jump probability (default: 1/n_layers)')
    parser.add_argument('--min-edges', type=int, default=10,
                      help='Minimum number of edges required to include layer (default: 10)')
    parser.add_argument('--seed-file', type=str, default='../og_gene_set/seed_genes.csv',
                      help='Path to seed genes CSV file')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MULTIXRANK PREPARATION")
    print("="*80)
    print(f"Multiplex directory: {args.multiplex_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Restart probability: {args.restart}")
    print(f"Minimum edges per layer: {args.min_edges}")
    print(f"Seed genes file: {args.seed_file}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load and filter multiplex layers
        layers = load_multiplex_layers(args.multiplex_dir, min_edges=args.min_edges)
        
        # Export layers to TSV format for MultiXrank
        layer_files = export_layers_to_tsv(layers, args.output_dir)
        
        # Create MultiXrank config
        delta = [args.delta] if args.delta is not None else None
        config_path = create_multixrank_config(
            layer_files, 
            args.output_dir,
            restart_prob=args.restart,
            delta=delta
        )
        
        # Save layer info for downstream use
        layer_info = save_layer_info(layers, args.output_dir)
        
        # Load and prepare seed genes with capitalization
        seed_genes_in_network, seeds_not_in_network = load_and_prepare_seed_genes(
            args.seed_file, layer_info, args.output_dir
        )
        
        # Save metadata
        # Load seeds again for count (handle non-strings)
        try:
            seed_df_for_count = pd.read_csv(args.seed_file)
            seed_col_for_count = None
            for col in ['gene', 'Gene', 'gene_id', 'Gene_ID', 'symbol', 'Symbol']:
                if col in seed_df_for_count.columns:
                    seed_col_for_count = col
                    break
            if seed_col_for_count is None:
                seed_col_for_count = seed_df_for_count.columns[0]
            
            # Count seeds safely (convert to string first, handle non-strings)
            seeds_for_count = []
            for item in seed_df_for_count[seed_col_for_count].dropna():
                item_str = str(item).strip()
                if item_str and item_str.lower() != 'nan':
                    seeds_for_count.append(item_str)
        except Exception as e:
            print(f"  Warning: Could not count seeds for metadata: {e}")
            seeds_for_count = len(seed_genes_in_network) + len(seeds_not_in_network)
        
        metadata = {
            'restart_probability': args.restart,
            'n_layers': layer_info['n_layers'],
            'n_genes': layer_info['n_genes'],
            'layer_names': layer_info['layer_names'],
            'min_edges_filter': args.min_edges,
            'method': 'MultiXrank',
            'config_file': config_path,
            'seed_genes_loaded': seeds_for_count if isinstance(seeds_for_count, int) else len(seeds_for_count),
            'seed_genes_in_network': len(seed_genes_in_network),
            'seed_genes_missing': len(seeds_not_in_network)
        }
        
        metadata_file = os.path.join(args.output_dir, 'multiplex_rwr_metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f, protocol=4)
        print(f"\n  Saved metadata: {metadata_file}")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("MULTIXRANK PREPARATION COMPLETE")
        print("="*80)
        print(f"Valid layers: {metadata['n_layers']}")
        print(f"Total unique genes: {metadata['n_genes']}")
        print(f"Seed genes loaded: {metadata['seed_genes_loaded']}")
        print(f"Seed genes in network: {metadata['seed_genes_in_network']}")
        print(f"Seed genes missing: {metadata['seed_genes_missing']}")
        print(f"Minimum edges threshold: {args.min_edges}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Output directory: {args.output_dir}")
        print("\nNext steps:")
        print(f"  1. seeds.txt has been created automatically with {len(seed_genes_in_network)} genes")
        print("  2. Run: python run_multiplex_rwr.py --multixrank-dir <output_dir> --output-dir <results_dir>")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}")
        return 1


if __name__ == "__main__":
    sys.exit(main())