#!/usr/bin/env python3
"""
Analyze Subset Multiplex Interactome from Tabula Sapiens

This script:
1. Loads a full multiplex interactome
2. Subsets it to only include specified cell types
3. Analyzes the resulting subset multiplex network
4. Generates comprehensive reports on the subset
"""

import os
import sys
import argparse
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
import time
import json

def load_multiplex_layers(layers_dir):
    """
    Load all layer networks from directory
    
    Parameters:
    - layers_dir: directory containing layer pickle files
    
    Returns:
    - Dictionary mapping cell_type -> NetworkX graph
    """
    print(f"Loading multiplex layers from {layers_dir}...")
    
    layers = {}
    layer_files = sorted(Path(layers_dir).glob('*_layer.pkl'))
    
    print(f"Found {len(layer_files)} layer files")
    
    for layer_file in layer_files:
        cell_type = layer_file.stem.replace('_layer', '')
        
        try:
            with open(layer_file, 'rb') as f:
                G = pickle.load(f)
            
            if G.number_of_nodes() > 0:
                layers[cell_type] = G
                print(f"  {cell_type}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            else:
                print(f"  {cell_type}: EMPTY (skipped)")
        except Exception as e:
            print(f"  ERROR loading {cell_type}: {str(e)}")
    
    return layers

def load_celltype_list(celltype_file):
    """
    Load list of cell types to keep
    
    Parameters:
    - celltype_file: path to file with cell types (one per line, or CSV)
    
    Returns:
    - Set of cell type names
    """
    print(f"Loading cell type list from {celltype_file}...")
    
    cell_types = set()
    
    try:
        # Try reading as CSV first
        if celltype_file.endswith('.csv'):
            df = pd.read_csv(celltype_file)
            # Try to find cell_type column
            for col in ['cell_type', 'Cell_Type', 'celltype', 'CellType', 'type']:
                if col in df.columns:
                    cell_types = set(df[col].astype(str).str.strip())
                    break
            
            # If no specific column found, use first column
            if not cell_types:
                cell_types = set(df.iloc[:, 0].astype(str).str.strip())
        else:
            # Read as plain text file (one per line)
            with open(celltype_file, 'r') as f:
                for line in f:
                    ct = line.strip()
                    if ct and not ct.startswith('#'):
                        cell_types.add(ct)
        
        print(f"Loaded {len(cell_types)} cell types:")
        for ct in sorted(cell_types):
            print(f"  - {ct}")
        
        return cell_types
    
    except Exception as e:
        print(f"ERROR loading cell type list: {str(e)}")
        return set()

def load_seed_genes(seed_genes_file):
    """Load seed genes from CSV file"""
    if not os.path.exists(seed_genes_file):
        print(f"Seed genes file not found: {seed_genes_file}")
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
        
        print(f"Loaded {len(seed_genes)} seed genes from {seed_genes_file}")
        return seed_genes
    
    except Exception as e:
        print(f"Error loading seed genes: {str(e)}")
        return set()

def subset_multiplex(layers, celltype_list):
    """
    Subset multiplex to only include specified cell types
    
    Parameters:
    - layers: dictionary of all layers
    - celltype_list: set of cell types to keep
    
    Returns:
    - Dictionary of subset layers
    """
    print(f"\nSubsetting multiplex to {len(celltype_list)} cell types...")
    
    subset_layers = {}
    missing_celltypes = []
    
    for cell_type in sorted(celltype_list):
        if cell_type in layers:
            subset_layers[cell_type] = layers[cell_type].copy()
            print(f"  ✓ {cell_type}: {subset_layers[cell_type].number_of_nodes()} nodes, {subset_layers[cell_type].number_of_edges()} edges")
        else:
            missing_celltypes.append(cell_type)
            print(f"  ✗ {cell_type}: NOT FOUND in multiplex")
    
    if missing_celltypes:
        print(f"\n⚠️  WARNING: {len(missing_celltypes)} cell types not found in multiplex:")
        for ct in missing_celltypes:
            print(f"     - {ct}")
        print(f"\nAvailable cell types in multiplex:")
        for ct in sorted(layers.keys()):
            print(f"     - {ct}")
    
    return subset_layers

def analyze_subset_multiplex(subset_layers, seed_genes=None, output_dir='subset_analysis'):
    """
    Analyze the subset multiplex network
    
    Parameters:
    - subset_layers: dictionary of subset layers
    - seed_genes: optional set of seed genes
    - output_dir: output directory for results
    
    Returns:
    - Dictionary with analysis results
    """
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("ANALYZING SUBSET MULTIPLEX")
    print("="*80)
    
    # Initialize report
    report = {
        'summary': {},
        'per_celltype': {},
        'inter_layer_analysis': {},
        'seed_gene_analysis': {'per_celltype': {}, 'summary': {}}
    }
    
    # ============================================================================
    # SUMMARY STATISTICS
    # ============================================================================
    print("\nCalculating summary statistics...")
    
    all_genes = set()
    all_edges = 0
    layer_stats = []
    
    for cell_type, layer in subset_layers.items():
        all_genes.update(layer.nodes())
        all_edges += layer.number_of_edges()
        
        n_nodes = layer.number_of_nodes()
        n_edges = layer.number_of_edges()
        density = nx.density(layer) if n_nodes > 0 else 0
        avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
        
        # Seed genes in this layer
        seeds_in_layer = seed_genes.intersection(layer.nodes()) if seed_genes else set()
        
        # Check seed gene connectivity
        seed_genes_connected = False
        n_seed_components = 0
        if len(seeds_in_layer) > 1:
            seed_subgraph = layer.subgraph(seeds_in_layer)
            n_seed_components = nx.number_connected_components(seed_subgraph)
            seed_genes_connected = (n_seed_components == 1)
        elif len(seeds_in_layer) == 1:
            seed_genes_connected = True
            n_seed_components = 1
        
        stat = {
            'cell_type': cell_type,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'avg_degree': avg_degree,
            'n_seed_genes': len(seeds_in_layer),
            'seed_gene_pct': len(seeds_in_layer) / len(seed_genes) * 100 if seed_genes else 0,
            'seed_genes_connected': seed_genes_connected,
            'n_seed_components': n_seed_components
        }
        
        layer_stats.append(stat)
        
        report['per_celltype'][cell_type] = stat
        
        print(f"{cell_type}:")
        print(f"  Nodes: {n_nodes} | Edges: {n_edges} | Density: {density:.4f} | Avg degree: {avg_degree:.2f}")
        if seed_genes:
            print(f"  Seed genes: {len(seeds_in_layer)} / {len(seed_genes)} ({stat['seed_gene_pct']:.1f}%)")
    
    report['summary']['n_cell_types'] = len(subset_layers)
    report['summary']['n_total_genes'] = len(all_genes)
    report['summary']['n_total_edges'] = all_edges
    report['summary']['avg_genes_per_layer'] = np.mean([s['n_nodes'] for s in layer_stats])
    report['summary']['avg_edges_per_layer'] = np.mean([s['n_edges'] for s in layer_stats])
    report['summary']['avg_density'] = np.mean([s['density'] for s in layer_stats])
    report['summary']['avg_degree'] = np.mean([s['avg_degree'] for s in layer_stats])
    
    # ============================================================================
    # INTER-LAYER ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("INTER-LAYER ANALYSIS")
    print("="*80)
    
    # Genes shared between layers
    print("\nGene overlap between layers:")
    
    layer_names = sorted(subset_layers.keys())
    overlap_matrix = {}
    
    for i, ct1 in enumerate(layer_names):
        genes1 = set(subset_layers[ct1].nodes())
        for ct2 in layer_names[i+1:]:
            genes2 = set(subset_layers[ct2].nodes())
            overlap = genes1.intersection(genes2)
            overlap_pct = len(overlap) / min(len(genes1), len(genes2)) * 100 if min(len(genes1), len(genes2)) > 0 else 0
            
            overlap_matrix[(ct1, ct2)] = {
                'n_overlap': len(overlap),
                'overlap_pct': overlap_pct
            }
            
            print(f"  {ct1} <-> {ct2}: {len(overlap)} genes ({overlap_pct:.1f}% of smaller layer)")
    
    report['inter_layer_analysis']['pairwise_overlaps'] = overlap_matrix
    
    # Genes unique to each layer
    print("\nGenes unique to each layer:")
    
    for cell_type in sorted(subset_layers.keys()):
        genes_in_layer = set(subset_layers[cell_type].nodes())
        genes_in_other = set()
        
        for other_ct, other_layer in subset_layers.items():
            if other_ct != cell_type:
                genes_in_other.update(other_layer.nodes())
        
        unique_genes = genes_in_layer - genes_in_other
        unique_pct = len(unique_genes) / len(genes_in_layer) * 100 if len(genes_in_layer) > 0 else 0
        
        print(f"  {cell_type}: {len(unique_genes)} unique genes ({unique_pct:.1f}%)")
        
        report['inter_layer_analysis'][f'{cell_type}_unique'] = {
            'n_unique': len(unique_genes),
            'unique_pct': unique_pct
        }
    
    # Genes in all layers
    genes_in_all = set(subset_layers[layer_names[0]].nodes())
    for cell_type in layer_names[1:]:
        genes_in_all = genes_in_all.intersection(set(subset_layers[cell_type].nodes()))
    
    print(f"\nGenes in ALL layers: {len(genes_in_all)}")
    report['inter_layer_analysis']['genes_in_all_layers'] = len(genes_in_all)
    
    # ============================================================================
    # SEED GENE ANALYSIS
    # ============================================================================
    if seed_genes:
        print("\n" + "="*80)
        print("SEED GENE ANALYSIS")
        print("="*80)
        
        seeds_in_multiplex = seed_genes.intersection(all_genes)
        seeds_pct = len(seeds_in_multiplex) / len(seed_genes) * 100
        
        print(f"\nSeed genes in subset multiplex: {len(seeds_in_multiplex)} / {len(seed_genes)} ({seeds_pct:.1f}%)")
        
        # Seed genes in all layers
        seeds_in_all = set(subset_layers[layer_names[0]].nodes()).intersection(seed_genes)
        for cell_type in layer_names[1:]:
            seeds_in_all = seeds_in_all.intersection(set(subset_layers[cell_type].nodes()).intersection(seed_genes))
        
        print(f"Seed genes in ALL layers: {len(seeds_in_all)}")
        
        # Seed genes unique to each layer
        print(f"\nSeed genes unique to each layer:")
        for cell_type in sorted(subset_layers.keys()):
            seeds_in_layer = seed_genes.intersection(set(subset_layers[cell_type].nodes()))
            seeds_in_other = set()
            
            for other_ct, other_layer in subset_layers.items():
                if other_ct != cell_type:
                    seeds_in_other.update(seed_genes.intersection(other_layer.nodes()))
            
            unique_seeds = seeds_in_layer - seeds_in_other
            print(f"  {cell_type}: {len(unique_seeds)} unique seed genes")
        
        report['seed_gene_analysis']['summary']['total_seed_genes'] = len(seed_genes)
        report['seed_gene_analysis']['summary']['seed_genes_in_multiplex'] = len(seeds_in_multiplex)
        report['seed_gene_analysis']['summary']['seed_gene_recovery_pct'] = seeds_pct
        report['seed_gene_analysis']['summary']['seed_genes_in_all_layers'] = len(seeds_in_all)
    
    # ============================================================================
    # SAVE OUTPUTS
    # ============================================================================
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Save layer statistics
    stats_df = pd.DataFrame(layer_stats)
    stats_file = os.path.join(output_dir, 'subset_layer_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved layer statistics to {stats_file}")
    
    # Save subset layers
    layers_dir = os.path.join(output_dir, 'layers')
    os.makedirs(layers_dir, exist_ok=True)
    
    for cell_type, layer in subset_layers.items():
        layer_file = os.path.join(layers_dir, f"{cell_type}_layer.pkl")
        with open(layer_file, 'wb') as f:
            pickle.dump(layer, f, protocol=4)
    
    print(f"Saved {len(subset_layers)} subset layer networks to {layers_dir}")
    
    # ============================================================================
    # WRITE COMPREHENSIVE REPORT
    # ============================================================================
    report_file = os.path.join(output_dir, 'subset_analysis_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SUBSET MULTIPLEX ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of cell types: {report['summary']['n_cell_types']}\n")
        f.write(f"Total unique genes: {report['summary']['n_total_genes']}\n")
        f.write(f"Total edges: {report['summary']['n_total_edges']}\n")
        f.write(f"Average genes per layer: {report['summary']['avg_genes_per_layer']:.1f}\n")
        f.write(f"Average edges per layer: {report['summary']['avg_edges_per_layer']:.1f}\n")
        f.write(f"Average density: {report['summary']['avg_density']:.4f}\n")
        f.write(f"Average degree: {report['summary']['avg_degree']:.2f}\n\n")
        
        # Per cell type
        f.write("PER CELL TYPE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Cell Type':<40} {'Nodes':<12} {'Edges':<12} {'Density':<12} {'Avg Degree':<12}\n")
        f.write("-"*80 + "\n")
        
        for cell_type in sorted(report['per_celltype'].keys()):
            ct_report = report['per_celltype'][cell_type]
            f.write(f"{cell_type:<40} ")
            f.write(f"{ct_report['n_nodes']:<12} ")
            f.write(f"{ct_report['n_edges']:<12} ")
            f.write(f"{ct_report['density']:.4f}{'':<7} ")
            f.write(f"{ct_report['avg_degree']:.2f}\n")
        
        # Inter-layer analysis
        f.write("\n" + "="*80 + "\n")
        f.write("INTER-LAYER ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("PAIRWISE GENE OVERLAPS\n")
        f.write("-"*80 + "\n")
        
        for (ct1, ct2), overlap_data in sorted(report['inter_layer_analysis']['pairwise_overlaps'].items()):
            f.write(f"{ct1} <-> {ct2}: {overlap_data['n_overlap']} genes ({overlap_data['overlap_pct']:.1f}%)\n")
        
        f.write(f"\nGenes in ALL layers: {report['inter_layer_analysis']['genes_in_all_layers']}\n\n")
        
        f.write("UNIQUE GENES PER LAYER\n")
        f.write("-"*80 + "\n")
        
        for cell_type in sorted(report['per_celltype'].keys()):
            if f'{cell_type}_unique' in report['inter_layer_analysis']:
                unique_data = report['inter_layer_analysis'][f'{cell_type}_unique']
                f.write(f"{cell_type}: {unique_data['n_unique']} unique genes ({unique_data['unique_pct']:.1f}%)\n")
        
        # Seed gene analysis
        if seed_genes:
            f.write("\n" + "="*80 + "\n")
            f.write("SEED GENE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            sg_summary = report['seed_gene_analysis']['summary']
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total seed genes: {sg_summary['total_seed_genes']}\n")
            f.write(f"Seed genes in multiplex: {sg_summary['seed_genes_in_multiplex']}\n")
            f.write(f"Seed gene recovery: {sg_summary['seed_gene_recovery_pct']:.1f}%\n")
            f.write(f"Seed genes in ALL layers: {sg_summary['seed_genes_in_all_layers']}\n\n")
            
            f.write("PER CELL TYPE SEED GENE STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Cell Type':<40} {'Seed Genes':<15} {'% of Total':<15} {'Connected':<12}\n")
            f.write("-"*80 + "\n")
            
            for cell_type in sorted(report['per_celltype'].keys()):
                ct_report = report['per_celltype'][cell_type]
                f.write(f"{cell_type:<40} ")
                f.write(f"{ct_report['n_seed_genes']:<15} ")
                f.write(f"{ct_report['seed_gene_pct']:.1f}%{'':<10} ")
                f.write(f"{str(ct_report['seed_genes_connected']):<12}\n")
    
    print(f"Saved analysis report to {report_file}")
    
    # Save JSON version
    json_report_file = os.path.join(output_dir, 'subset_analysis_report.json')
    with open(json_report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved JSON report to {json_report_file}")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("SUBSET MULTIPLEX ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total cell types: {len(subset_layers)}")
    print(f"Total unique genes: {len(all_genes)}")
    print(f"Total edges: {all_edges}")
    if seed_genes:
        print(f"Seed gene recovery: {len(seeds_in_multiplex)} / {len(seed_genes)} ({seeds_pct:.1f}%)")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    return report

def main():
    parser = argparse.ArgumentParser(
        description='Analyze subset of multiplex interactome by cell type'
    )
    parser.add_argument('--layers-dir', type=str, required=True,
                      help='Directory containing multiplex layer pickle files')
    parser.add_argument('--celltype-list', type=str, required=True,
                      help='File with cell types to keep (one per line or CSV)')
    parser.add_argument('--output-dir', type=str, default='subset_analysis',
                      help='Output directory for analysis results')
    parser.add_argument('--seed-genes-file', type=str, default=None,
                      help='Path to seed genes CSV file (optional)')
    
    args = parser.parse_args()
    
    # Load full multiplex
    layers = load_multiplex_layers(args.layers_dir)
    
    if not layers:
        print("ERROR: No layers loaded!")
        return 1
    
    # Load cell type list
    celltype_list = load_celltype_list(args.celltype_list)
    
    if not celltype_list:
        print("ERROR: No cell types loaded!")
        return 1
    
    # Load seed genes if provided
    seed_genes = set()
    if args.seed_genes_file:
        print()
        seed_genes = load_seed_genes(args.seed_genes_file)
    
    # Subset multiplex
    subset_layers = subset_multiplex(layers, celltype_list)
    
    if not subset_layers:
        print("ERROR: No layers in subset!")
        return 1
    
    # Analyze subset
    report = analyze_subset_multiplex(subset_layers, seed_genes=seed_genes, output_dir=args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())