#!/usr/bin/env python3
"""
Build Multiplex Interactome from Cell-Type-Specific PPI Data (Tabula Sapiens)

This script creates a multiplex network where:
- Each layer represents a cell type
- Nodes are genes with PPI interactions in that cell type
- Edges are PPI interactions
- Inter-layer connections link the same gene across cell types
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

def load_celltype_ppi(ppi_file):
    """
    Load cell-type-specific PPI from TSV file
    
    Format:
    # Graph Type: undirected
    1	A	B
    2	FAF2	SLC25A4
    ...
    
    Parameters:
    - ppi_file: path to TSV file
    
    Returns:
    - NetworkX graph
    """
    G = nx.Graph()
    
    try:
        # Read TSV, skip first 2 rows (header and "Graph Type" line)
        df = pd.read_csv(ppi_file, sep='\t', header=None, skiprows=2)
        
        # Columns: [index, gene1, gene2]
        if len(df.columns) >= 2:
            gene1_col = df.iloc[:, 0]
            gene2_col = df.iloc[:, 1]
        else:
            raise ValueError(f"Expected at least 2 columns, got {len(df.columns)}")
        
        # Add edges
        for g1, g2 in zip(gene1_col, gene2_col):
            g1 = str(g1).strip().upper()
            g2 = str(g2).strip().upper()
            if g1 and g2 and g1 != g2:  # Skip empty or self-loops
                G.add_edge(g1, g2)
        
        return G
    
    except Exception as e:
        print(f"  Error loading {ppi_file}: {str(e)}")
        return nx.Graph()

def load_all_celltype_ppis(ppi_dir):
    """
    Load all cell-type PPI files from directory
    
    Parameters:
    - ppi_dir: directory containing .tsv files
    
    Returns:
    - Dictionary mapping cell_type -> NetworkX graph
    """
    print(f"Loading cell-type PPI files from {ppi_dir}...")
    
    celltype_ppis = {}
    ppi_files = sorted(Path(ppi_dir).glob('*.tsv'))
    
    print(f"Found {len(ppi_files)} PPI files")
    
    for ppi_file in ppi_files:
        cell_type = ppi_file.stem  # filename without .tsv
        G = load_celltype_ppi(str(ppi_file))
        
        if G.number_of_nodes() > 0:
            celltype_ppis[cell_type] = G
            print(f"  {cell_type}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        else:
            print(f"  {cell_type}: EMPTY (skipped)")
    
    return celltype_ppis

def load_seed_genes(seed_genes_file):
    """
    Load seed genes from CSV file
    
    Parameters:
    - seed_genes_file: path to CSV file
    
    Returns:
    - Set of gene names (uppercase)
    """
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

def build_multiplex_interactome(ppi_dir, output_dir, seed_genes_file=None):
    """
    Build complete multiplex interactome from cell-type PPI files
    
    Parameters:
    - ppi_dir: directory containing cell-type PPI TSV files
    - output_dir: directory to save outputs
    - seed_genes_file: optional path to seed genes CSV
    
    Returns:
    - Dictionary of layer graphs and metadata
    """
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all cell-type PPIs
    print("\n" + "="*80)
    print("LOADING CELL-TYPE PPI NETWORKS")
    print("="*80)
    celltype_ppis = load_all_celltype_ppis(ppi_dir)
    
    if not celltype_ppis:
        print("ERROR: No valid PPI files loaded!")
        return None, None
    
    # Load seed genes if provided
    seed_genes = set()
    if seed_genes_file:
        print("\n" + "="*80)
        print("LOADING SEED GENES")
        print("="*80)
        seed_genes = load_seed_genes(seed_genes_file)
    
    # ============================================================================
    # ANALYSIS: Collect statistics before filtering
    # ============================================================================
    print("\n" + "="*80)
    print("PRE-FILTERING STATISTICS")
    print("="*80)
    
    all_genes_before = set()
    all_edges_before = 0
    ppi_genes_per_celltype_before = {}
    
    for cell_type, G in celltype_ppis.items():
        all_genes_before.update(G.nodes())
        all_edges_before += G.number_of_edges()
        ppi_genes_per_celltype_before[cell_type] = len(G.nodes())
    
    print(f"Total unique genes across all cell types: {len(all_genes_before)}")
    print(f"Total edges across all cell types: {all_edges_before}")
    print(f"Average genes per cell type: {np.mean(list(ppi_genes_per_celltype_before.values())):.1f}")
    print(f"Average edges per cell type: {all_edges_before / len(celltype_ppis):.1f}")
    
    # ============================================================================
    # SEED GENE ANALYSIS (BEFORE FILTERING)
    # ============================================================================
    if seed_genes:
        print("\n" + "="*80)
        print("SEED GENE ANALYSIS - BEFORE FILTERING")
        print("="*80)
        
        seeds_in_ppi = seed_genes.intersection(all_genes_before)
        seeds_pct = len(seeds_in_ppi) / len(seed_genes) * 100 if seed_genes else 0
        
        print(f"Total seed genes: {len(seed_genes)}")
        print(f"Seed genes in PPI networks: {len(seeds_in_ppi)} ({seeds_pct:.1f}%)")
        
        # Per cell type
        print(f"\nSeed genes per cell type (before filtering):")
        seed_per_celltype_before = {}
        for cell_type, G in celltype_ppis.items():
            seeds_in_ct = seed_genes.intersection(G.nodes())
            seed_per_celltype_before[cell_type] = len(seeds_in_ct)
            pct = len(seeds_in_ct) / len(seed_genes) * 100 if seed_genes else 0
            print(f"  {cell_type}: {len(seeds_in_ct)} / {len(seed_genes)} ({pct:.1f}%)")
        
        # Show examples
        print(f"\nExample seed genes found in PPI:")
        for gene in list(seeds_in_ppi)[:10]:
            # Count how many cell types have this gene
            ct_count = sum(1 for G in celltype_ppis.values() if gene in G.nodes())
            print(f"  ✓ {gene} (in {ct_count}/{len(celltype_ppis)} cell types)")
        
        if len(seeds_in_ppi) > 10:
            print(f"  ... and {len(seeds_in_ppi) - 10} more")
        
        # Show missing seeds
        missing_seeds = seed_genes - seeds_in_ppi
        if missing_seeds:
            print(f"\nExample seed genes NOT found in PPI:")
            for gene in list(missing_seeds)[:10]:
                print(f"  ✗ {gene}")
            if len(missing_seeds) > 10:
                print(f"  ... and {len(missing_seeds) - 10} more")
    
    # ============================================================================
    # BUILD MULTIPLEX LAYERS
    # ============================================================================
    print("\n" + "="*80)
    print("BUILDING MULTIPLEX LAYERS")
    print("="*80)
    
    layers = {}
    layer_stats = []
    
    for cell_type, G in celltype_ppis.items():
        # Create layer (copy of the cell-type PPI)
        layer = G.copy()
        layers[cell_type] = layer
        
        # Calculate statistics
        n_nodes = layer.number_of_nodes()
        n_edges = layer.number_of_edges()
        density = nx.density(layer) if n_nodes > 0 else 0
        avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
        
        # Seed gene statistics
        seeds_in_layer = seed_genes.intersection(layer.nodes()) if seed_genes else set()

        # Check if seed genes form a connected component
        seed_genes_connected = False
        n_seed_components = 0
        if seeds_in_layer and len(seeds_in_layer) > 1:
            # Create subgraph of seed genes
            seed_subgraph = layer.subgraph(seeds_in_layer)
            n_seed_components = nx.number_connected_components(seed_subgraph)
            seed_genes_connected = (n_seed_components == 1)
        elif len(seeds_in_layer) == 1:
            seed_genes_connected = True
            n_seed_components = 1

        # Percentage of seed genes relative to total genes in layer
        seed_gene_pct_of_layer = len(seeds_in_layer) / n_nodes * 100 if n_nodes > 0 else 0

        # Node/edge survival (100% since we're not filtering yet)
        node_survival = 100.0
        edge_survival = 100.0

        stat = {
            'cell_type': cell_type,
            'n_nodes_before': n_nodes,
            'n_nodes_after': n_nodes,
            'n_edges_before': n_edges,
            'n_edges_after': n_edges,
            'node_survival_pct': node_survival,
            'edge_survival_pct': edge_survival,
            'density': density,
            'avg_degree': avg_degree,
            'n_seed_genes': len(seeds_in_layer),
            'seed_gene_pct': len(seeds_in_layer) / len(seed_genes) * 100 if seed_genes else 0,
            'seed_gene_pct_of_layer': seed_gene_pct_of_layer,
            'seed_genes_connected': seed_genes_connected,
            'n_seed_components': n_seed_components
        }
        
        layer_stats.append(stat)
        
        print(f"{cell_type}:")
        print(f"  Nodes: {n_nodes} | Edges: {n_edges} | Density: {density:.4f} | Avg degree: {avg_degree:.2f}")
        if seed_genes:
            print(f"  Seed genes: {len(seeds_in_layer)} / {len(seed_genes)} ({stat['seed_gene_pct']:.1f}%)")
            print(f"  Seed genes % of layer: {seed_gene_pct_of_layer:.1f}%")
            print(f"  Seed genes connected: {seed_genes_connected} ({n_seed_components} component(s))")
    
    # ============================================================================
    # MULTIPLEX-LEVEL STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print("MULTIPLEX-LEVEL STATISTICS")
    print("="*80)
    
    all_genes_after = set()
    all_edges_after = 0
    
    for layer in layers.values():
        all_genes_after.update(layer.nodes())
        all_edges_after += layer.number_of_edges()
    
    print(f"Total unique genes in multiplex: {len(all_genes_after)}")
    print(f"Total edges in multiplex: {all_edges_after}")
    print(f"Number of layers: {len(layers)}")
    
    # Seed genes in final multiplex
    if seed_genes:
        seeds_in_multiplex = seed_genes.intersection(all_genes_after)
        seeds_pct_final = len(seeds_in_multiplex) / len(seed_genes) * 100
        
        print(f"\nSeed genes in final multiplex: {len(seeds_in_multiplex)} / {len(seed_genes)} ({seeds_pct_final:.1f}%)")
        
        # Gene loss
        seeds_lost = len(seed_genes) - len(seeds_in_multiplex)
        print(f"Seed genes lost: {seeds_lost}")
    
    # ============================================================================
    # SAVE OUTPUTS
    # ============================================================================
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Save individual layers
    layers_dir = os.path.join(output_dir, 'layers')
    os.makedirs(layers_dir, exist_ok=True)
    
    for cell_type, layer in layers.items():
        layer_file = os.path.join(layers_dir, f"{cell_type}_layer.pkl")
        with open(layer_file, 'wb') as f:
            pickle.dump(layer, f, protocol=4)
    
    print(f"Saved {len(layers)} layer networks to {layers_dir}")
    
    # Save layer statistics
    stats_df = pd.DataFrame(layer_stats)
    stats_file = os.path.join(output_dir, 'layer_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved layer statistics to {stats_file}")
    
    # ============================================================================
    # GENERATE CONVERSION REPORT
    # ============================================================================
    print("\n" + "="*80)
    print("GENERATING CONVERSION REPORT")
    print("="*80)
    
    report = generate_conversion_report(
        celltype_ppis=celltype_ppis,
        layers=layers,
        seed_genes=seed_genes,
        output_dir=output_dir
    )
    
    # Save metadata
    metadata = {
        'n_cell_types': len(layers),
        'n_total_genes': len(all_genes_after),
        'n_total_edges': all_edges_after,
        'n_seed_genes': len(seed_genes),
        'n_seed_genes_in_multiplex': len(seeds_in_multiplex) if seed_genes else 0,
        'seed_gene_recovery_pct': seeds_pct_final if seed_genes else 0,
        'layer_stats': layer_stats,
        'report': report
    }
    
    metadata_file = os.path.join(output_dir, 'multiplex_metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f, protocol=4)
    
    # Save report as JSON
    report_json_file = os.path.join(output_dir, 'conversion_report.json')
    with open(report_json_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved conversion report to {report_json_file}")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("MULTIPLEX INTERACTOME CONSTRUCTION COMPLETE")
    print("="*80)
    print(f"Total cell types: {len(layers)}")
    print(f"Total unique genes: {len(all_genes_after)}")
    print(f"Total edges: {all_edges_after}")
    if seed_genes:
        print(f"Seed gene recovery: {len(seeds_in_multiplex)} / {len(seed_genes)} ({seeds_pct_final:.1f}%)")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    return layers, metadata

def generate_conversion_report(celltype_ppis, layers, seed_genes, output_dir):
    """
    Generate detailed conversion report with PPI and seed gene statistics
    
    Returns:
    - Dictionary with report data
    """
    report = {
        'summary': {},
        'per_celltype': {},
        'seed_gene_analysis': {}
    }
    
    # ============================================================================
    # SUMMARY STATISTICS
    # ============================================================================
    all_genes_before = set()
    all_edges_before = 0
    
    for G in celltype_ppis.values():
        all_genes_before.update(G.nodes())
        all_edges_before += G.number_of_edges()
    
    all_genes_after = set()
    all_edges_after = 0
    
    for layer in layers.values():
        all_genes_after.update(layer.nodes())
        all_edges_after += layer.number_of_edges()
    
    report['summary'] = {
        'n_cell_types': len(celltype_ppis),
        'genes_before': len(all_genes_before),
        'genes_after': len(all_genes_after),
        'genes_lost': len(all_genes_before) - len(all_genes_after),
        'gene_retention_pct': len(all_genes_after) / len(all_genes_before) * 100 if all_genes_before else 0,
        'edges_before': all_edges_before,
        'edges_after': all_edges_after,
        'edges_lost': all_edges_before - all_edges_after,
        'edge_retention_pct': all_edges_after / all_edges_before * 100 if all_edges_before else 0
    }
    
    # ============================================================================
    # PER CELL TYPE STATISTICS
    # ============================================================================
    for cell_type in sorted(celltype_ppis.keys()):
        G_before = celltype_ppis[cell_type]
        G_after = layers[cell_type]

        n_nodes_before = G_before.number_of_nodes()
        n_nodes_after = G_after.number_of_nodes()
        n_edges_before = G_before.number_of_edges()
        n_edges_after = G_after.number_of_edges()

        node_survival = n_nodes_after / n_nodes_before * 100 if n_nodes_before > 0 else 0
        edge_survival = n_edges_after / n_edges_before * 100 if n_edges_before > 0 else 0

        # Calculate seed gene statistics for this cell type
        seeds_in_layer = seed_genes.intersection(G_after.nodes()) if seed_genes else set()
        seed_gene_pct_of_layer = len(seeds_in_layer) / n_nodes_after * 100 if n_nodes_after > 0 else 0

        # Check if seed genes form a connected component
        seed_genes_connected = False
        n_seed_components = 0
        if seeds_in_layer and len(seeds_in_layer) > 1:
            seed_subgraph = G_after.subgraph(seeds_in_layer)
            n_seed_components = nx.number_connected_components(seed_subgraph)
            seed_genes_connected = (n_seed_components == 1)
        elif len(seeds_in_layer) == 1:
            seed_genes_connected = True
            n_seed_components = 1

        report['per_celltype'][cell_type] = {
            'ppi_genes_before': n_nodes_before,
            'ppi_genes_after': n_nodes_after,
            'ppi_genes_lost': n_nodes_before - n_nodes_after,
            'ppi_node_survival_pct': node_survival,
            'ppi_edges_before': n_edges_before,
            'ppi_edges_after': n_edges_after,
            'ppi_edges_lost': n_edges_before - n_edges_after,
            'ppi_edge_survival_pct': edge_survival,
            'seed_gene_pct_of_layer': seed_gene_pct_of_layer,
            'seed_genes_connected': seed_genes_connected,
            'n_seed_components': n_seed_components
        }
    
    # ============================================================================
    # SEED GENE ANALYSIS
    # ============================================================================
    if seed_genes:
        # Overall seed gene statistics
        all_genes_before = set()
        for G in celltype_ppis.values():
            all_genes_before.update(G.nodes())
        
        all_genes_after = set()
        for layer in layers.values():
            all_genes_after.update(layer.nodes())
        
        seeds_in_before = seed_genes.intersection(all_genes_before)
        seeds_in_after = seed_genes.intersection(all_genes_after)
        
        report['seed_gene_analysis']['summary'] = {
            'total_seed_genes': len(seed_genes),
            'seed_genes_in_ppi_before': len(seeds_in_before),
            'seed_genes_in_multiplex_after': len(seeds_in_after),
            'seed_genes_lost': len(seeds_in_before) - len(seeds_in_after),
            'seed_gene_recovery_pct': len(seeds_in_after) / len(seed_genes) * 100 if seed_genes else 0
        }
        
        # Per cell type seed gene statistics
        report['seed_gene_analysis']['per_celltype'] = {}
        
        for cell_type in sorted(celltype_ppis.keys()):
            G_before = celltype_ppis[cell_type]
            G_after = layers[cell_type]
            
            seeds_before = seed_genes.intersection(G_before.nodes())
            seeds_after = seed_genes.intersection(G_after.nodes())
            
            report['seed_gene_analysis']['per_celltype'][cell_type] = {
                'seed_genes_in_ppi': len(seeds_before),
                'seed_genes_in_layer': len(seeds_after),
                'seed_genes_lost': len(seeds_before) - len(seeds_after),
                'seed_gene_pct_of_total': len(seeds_after) / len(seed_genes) * 100 if seed_genes else 0
            }
    
    # ============================================================================
    # WRITE TEXT REPORT
    # ============================================================================
    report_file = os.path.join(output_dir, 'conversion_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTIPLEX INTERACTOME CONVERSION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of cell types: {report['summary']['n_cell_types']}\n")
        f.write(f"Total genes (before): {report['summary']['genes_before']}\n")
        f.write(f"Total genes (after): {report['summary']['genes_after']}\n")
        f.write(f"Genes lost: {report['summary']['genes_lost']}\n")
        f.write(f"Gene retention: {report['summary']['gene_retention_pct']:.1f}%\n\n")
        
        f.write(f"Total edges (before): {report['summary']['edges_before']}\n")
        f.write(f"Total edges (after): {report['summary']['edges_after']}\n")
        f.write(f"Edges lost: {report['summary']['edges_lost']}\n")
        f.write(f"Edge retention: {report['summary']['edge_retention_pct']:.1f}%\n\n")
        
        # Per cell type
        f.write("PER CELL TYPE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Cell Type':<50} {'Nodes':<15} {'Edges':<15} {'Node %':<10} {'Edge %':<10}\n")
        f.write("-"*80 + "\n")
        
        for cell_type in sorted(report['per_celltype'].keys()):
            ct_report = report['per_celltype'][cell_type]
            f.write(f"{cell_type:<50} ")
            f.write(f"{ct_report['ppi_genes_after']:<15} ")
            f.write(f"{ct_report['ppi_edges_after']:<15} ")
            f.write(f"{ct_report['ppi_node_survival_pct']:.1f}%{'':<6} ")
            f.write(f"{ct_report['ppi_edge_survival_pct']:.1f}%\n")
        
        f.write(f"{'Cell Type':<50} {'Seed %':<10} {'Connected':<12} {'Components':<12}\n")
        f.write("-"*84 + "\n")

        for cell_type in sorted(report['per_celltype'].keys()):
            ct_report = report['per_celltype'][cell_type]
            f.write(f"{cell_type:<50} ")
            f.write(f"{ct_report.get('seed_gene_pct_of_layer', 0):.1f}%{'':<6} ")
            f.write(f"{str(ct_report.get('seed_genes_connected', False)):<12} ")
            f.write(f"{ct_report.get('n_seed_components', 0)}\n")
            
        # Seed gene analysis
        if seed_genes:
            f.write("\n" + "="*80 + "\n")
            f.write("SEED GENE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            sg_summary = report['seed_gene_analysis']['summary']
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total seed genes: {sg_summary['total_seed_genes']}\n")
            f.write(f"Seed genes in PPI (before): {sg_summary['seed_genes_in_ppi_before']}\n")
            f.write(f"Seed genes in multiplex (after): {sg_summary['seed_genes_in_multiplex_after']}\n")
            f.write(f"Seed genes lost: {sg_summary['seed_genes_lost']}\n")
            f.write(f"Seed gene recovery: {sg_summary['seed_gene_recovery_pct']:.1f}%\n\n")
            
            f.write("PER CELL TYPE SEED GENE STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Cell Type':<50} {'In Layer':<15} {'% of Total':<15}\n")
            f.write("-"*80 + "\n")
            
            for cell_type in sorted(report['seed_gene_analysis']['per_celltype'].keys()):
                sg_ct = report['seed_gene_analysis']['per_celltype'][cell_type]
                f.write(f"{cell_type:<50} ")
                f.write(f"{sg_ct['seed_genes_in_layer']:<15} ")
                f.write(f"{sg_ct['seed_gene_pct_of_total']:.1f}%\n")
    
    print(f"Saved text report to {report_file}")
    
    return report

def main():
    parser = argparse.ArgumentParser(
        description='Build multiplex interactome from cell-type-specific PPI files'
    )
    parser.add_argument('--ppi-dir', type=str, required=True,
                      help='Directory containing cell-type PPI TSV files')
    parser.add_argument('--output-dir', type=str, default='multiplex_interactome',
                      help='Output directory for multiplex network')
    parser.add_argument('--seed-genes', type=str, default=None,
                      help='Path to seed genes CSV file (optional)')
    
    args = parser.parse_args()
    
    # Build multiplex interactome
    layers, metadata = build_multiplex_interactome(
        ppi_dir=args.ppi_dir,
        output_dir=args.output_dir,
        seed_genes_file=args.seed_genes
    )
    
    if layers is None:
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())