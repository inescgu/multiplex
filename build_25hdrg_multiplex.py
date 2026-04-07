#!/usr/bin/env python3
"""
Build Multiplex Interactome from Cell-Type Marker Genes from the 2025 Hrdg atlas

This script creates a multiplex network where:
- Each layer represents a cell type
- Nodes are marker genes for that cell type
- Edges are PPI interactions between marker genes
- Inter-layer connections link the same gene across cell types
- Seed genes are overlaid and tracked across layers
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

def load_celltype_markers(markers_file, p_val_threshold=0.05, log2fc_threshold=0.0):
    """
    Load cell-type marker genes from CSV file
    
    Parameters:
    - markers_file: path to CSV file with columns: p_val, avg_log2FC, pct.1, pct.2, p_val_adj, gene, cell_type
    - p_val_threshold: p-value threshold for significance
    - log2fc_threshold: log2 fold change threshold
    
    Returns:
    - Dictionary mapping cell_type -> list of marker genes
    - DataFrame with all markers
    """
    print(f"Loading cell-type markers from {markers_file}...")
    
    markers_df = pd.read_csv(markers_file)
    
    print(f"  Loaded {len(markers_df)} total marker entries")
    print(f"  Columns: {list(markers_df.columns)}")
    
    # Identify gene and cell_type columns
    gene_col = None
    celltype_col = None
    
    for col in ['gene', 'Gene', 'gene_id', 'Gene_ID', 'symbol', 'Symbol']:
        if col in markers_df.columns:
            gene_col = col
            break
    
    for col in ['cell_type', 'Cell_Type', 'celltype', 'CellType', 'cluster']:
        if col in markers_df.columns:
            celltype_col = col
            break
    
    if gene_col is None or celltype_col is None:
        print(f"  ERROR: Could not identify gene or cell_type columns")
        print(f"  Available columns: {list(markers_df.columns)}")
        return None, None
    
    print(f"  Using gene column: '{gene_col}'")
    print(f"  Using cell_type column: '{celltype_col}'")
    
#     # Filter by p-value if available
#     if 'p_val_adj' in markers_df.columns:
#         print(f"  Filtering by adjusted p-value < {p_val_threshold}...")
#         markers_df = markers_df[markers_df['p_val_adj'] < p_val_threshold]
#         print(f"  After p-value filtering: {len(markers_df)} markers")
#     elif 'p_val' in markers_df.columns:
#         print(f"  Filtering by p-value < {p_val_threshold}...")
#         markers_df = markers_df[markers_df['p_val'] < p_val_threshold]
#         print(f"  After p-value filtering: {len(markers_df)} markers")
    
#     # Filter by log2FC if available
#     if 'avg_log2FC' in markers_df.columns:
#         print(f"  Filtering by log2FC > {log2fc_threshold}...")
#         markers_df = markers_df[markers_df['avg_log2FC'] > log2fc_threshold]
#         print(f"  After log2FC filtering: {len(markers_df)} markers")
    
    # Convert gene names to uppercase
    markers_df[gene_col] = markers_df[gene_col].astype(str).str.upper()
    
    # Group by cell type
    celltype_markers = {}
    for cell_type in markers_df[celltype_col].unique():
        genes = markers_df[markers_df[celltype_col] == cell_type][gene_col].tolist()
        genes = [g.strip() for g in genes if g.strip()]  # Remove empty strings
        celltype_markers[cell_type] = genes
        print(f"  {cell_type}: {len(genes)} marker genes")
    
    return celltype_markers, markers_df

def load_base_ppi(ppi_file):
    """Load base PPI network"""
    print(f"Loading base PPI network from {ppi_file}...")
    
    with open(ppi_file, 'rb') as f:
        G = pickle.load(f)
    
    print(f"  Loaded PPI with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Convert node names to uppercase
    G = nx.relabel_nodes(G, {node: str(node).upper() for node in G.nodes()})
    print(f"  Converted node names to uppercase")
    
    return G

def load_seed_genes(seed_genes_file):
    """
    Load seed genes from CSV file
    
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

def build_layer_network(cell_type, marker_genes, base_ppi, seed_genes=None):
    """
    Build a single layer of the multiplex network for one cell type
    
    Parameters:
    - cell_type: name of the cell type
    - marker_genes: list of marker gene names
    - base_ppi: base PPI network
    - seed_genes: optional set of seed genes to track
    
    Returns:
    - NetworkX graph for this layer
    """
    print(f"\nBuilding layer for cell type: {cell_type}")
    
    marker_genes_set = set(marker_genes)
    ppi_genes_set = set(base_ppi.nodes())
    overlap = marker_genes_set.intersection(ppi_genes_set)
    
    print(f"  Marker genes: {len(marker_genes_set)}")
    print(f"  PPI genes: {len(ppi_genes_set)}")
    print(f"  Overlap: {len(overlap)} ({len(overlap)/len(marker_genes_set)*100:.1f}%)")
    
    if len(overlap) < 1:
        print(f"  ❌ ERROR: No marker genes found in PPI network!")
        empty_graph = nx.Graph()
        empty_graph.graph['error'] = 'no_genes_in_ppi'
        return empty_graph
    
    # Create subgraph with marker genes that are in PPI
    layer_graph = base_ppi.subgraph(overlap).copy()
    
    print(f"  Nodes in layer: {layer_graph.number_of_nodes()}")
    print(f"  Edges in layer: {layer_graph.number_of_edges()}")
    
    # Add node attributes
    for gene in layer_graph.nodes():
        layer_graph.nodes[gene]['cell_type'] = cell_type
        layer_graph.nodes[gene]['is_marker'] = True
        
        # Check if this is a seed gene
        if seed_genes and gene in seed_genes:
            layer_graph.nodes[gene]['is_seed'] = True
        else:
            layer_graph.nodes[gene]['is_seed'] = False
    
    # Track seed genes in this layer
    if seed_genes:
        seeds_in_layer = seed_genes.intersection(set(layer_graph.nodes()))
        print(f"  Seed genes in layer: {len(seeds_in_layer)} / {len(seed_genes)} ({len(seeds_in_layer)/len(seed_genes)*100:.1f}%)")
        
        # Check connectivity of seed genes
        if len(seeds_in_layer) > 1:
            seed_subgraph = layer_graph.subgraph(seeds_in_layer)
            n_components = nx.number_connected_components(seed_subgraph)
            print(f"  Seed genes form {n_components} connected component(s)")
        elif len(seeds_in_layer) == 1:
            print(f"  Single seed gene in layer")
    
    return layer_graph

def build_multiplex_from_markers(markers_file, base_ppi_file, output_dir,
                                 p_val_threshold=0.05, log2fc_threshold=0.0,
                                 seed_genes_file=None):
    """
    Build multiplex interactome from cell-type marker genes
    
    Parameters:
    - markers_file: path to cell-type markers CSV
    - base_ppi_file: path to base PPI network pickle
    - output_dir: output directory
    - p_val_threshold: p-value threshold for markers
    - log2fc_threshold: log2FC threshold for markers
    - seed_genes_file: optional path to seed genes CSV
    
    Returns:
    - Dictionary of layers and metadata
    """
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize report
    report = {
        'summary': {},
        'per_celltype': {},
        'seed_gene_analysis': {'per_celltype': {}, 'summary': {}}
    }
    
    # ============================================================================
    # LOAD DATA
    # ============================================================================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load markers
    celltype_markers, markers_df = load_celltype_markers(
        markers_file,
        p_val_threshold=p_val_threshold,
        log2fc_threshold=log2fc_threshold
    )
    
    if celltype_markers is None:
        print("ERROR: Failed to load markers")
        return None, None
    
    # Load PPI
    base_ppi = load_base_ppi(base_ppi_file)
    
    # Load seed genes if provided
    seed_genes = set()
    if seed_genes_file:
        print()
        seed_genes = load_seed_genes(seed_genes_file)
    
    # ============================================================================
    # VERIFICATION AND ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("GENE ID VERIFICATION")
    print("="*80)
    
    all_marker_genes = set()
    for genes in celltype_markers.values():
        all_marker_genes.update(genes)
    
    ppi_genes = set(base_ppi.nodes())
    
    print(f"\n1. Gene ID Format Check:")
    print(f"   Total marker genes: {len(all_marker_genes)}")
    print(f"   PPI genes: {len(ppi_genes)}")
    
    marker_examples = list(all_marker_genes)[:5]
    ppi_examples = list(ppi_genes)[:5]
    print(f"\n   Marker gene examples:")
    for i, gene in enumerate(marker_examples, 1):
        print(f"      {i}. {gene}")
    print(f"\n   PPI gene examples:")
    for i, gene in enumerate(ppi_examples, 1):
        print(f"      {i}. {gene}")
    
    overlap = all_marker_genes.intersection(ppi_genes)
    overlap_pct = len(overlap) / len(all_marker_genes) * 100 if all_marker_genes else 0
    
    print(f"\n2. Gene Overlap:")
    print(f"   Genes in both: {len(overlap)} ({overlap_pct:.1f}%)")
    
    if overlap_pct < 50:
        print(f"   ⚠️  WARNING: Low overlap! Likely gene ID format mismatch.")
    elif overlap_pct < 80:
        print(f"   ⚠️  NOTICE: Moderate overlap. Some ID conversion issues possible.")
    else:
        print(f"   ✓ Good overlap! Gene IDs appear compatible.")
    
    # Seed gene verification
    if seed_genes:
        print(f"\n3. Seed Gene Verification:")
        seeds_in_markers = seed_genes.intersection(all_marker_genes)
        seeds_in_ppi = seed_genes.intersection(ppi_genes)
        seeds_in_both = seed_genes.intersection(overlap)
        
        print(f"   Total seed genes: {len(seed_genes)}")
        print(f"   Seeds in marker genes: {len(seeds_in_markers)} ({len(seeds_in_markers)/len(seed_genes)*100:.1f}%)")
        print(f"   Seeds in PPI: {len(seeds_in_ppi)} ({len(seeds_in_ppi)/len(seed_genes)*100:.1f}%)")
        print(f"   Seeds in BOTH (markers AND PPI): {len(seeds_in_both)} ({len(seeds_in_both)/len(seed_genes)*100:.1f}%)")
        
        report['seed_gene_analysis']['summary']['total_seed_genes'] = len(seed_genes)
        report['seed_gene_analysis']['summary']['seed_genes_in_markers'] = len(seeds_in_markers)
        report['seed_gene_analysis']['summary']['seed_genes_in_ppi'] = len(seeds_in_ppi)
        report['seed_gene_analysis']['summary']['seed_genes_in_both_before'] = len(seeds_in_both)
    
    print("="*80 + "\n")
    
    # ============================================================================
    # BUILD MULTIPLEX LAYERS
    # ============================================================================
    print("\n" + "="*80)
    print("BUILDING MULTIPLEX LAYERS")
    print("="*80)
    
    layers = {}
    layer_stats = []
    
    for cell_type in sorted(celltype_markers.keys()):
        marker_genes = celltype_markers[cell_type]
        
        layer_graph = build_layer_network(
            cell_type, marker_genes, base_ppi, seed_genes=seed_genes
        )
        
        layers[cell_type] = layer_graph
        
        # Calculate layer statistics
        if layer_graph.number_of_nodes() > 0:
            density = nx.density(layer_graph)
            avg_degree = 2 * layer_graph.number_of_edges() / layer_graph.number_of_nodes()
        else:
            density = 0
            avg_degree = 0
        
        # Marker genes in PPI
        marker_genes_set = set(marker_genes)
        markers_in_ppi = marker_genes_set.intersection(ppi_genes)
        
        # Node retention (nodes in layer / markers in PPI)
        node_retention_pct = (layer_graph.number_of_nodes() / len(markers_in_ppi) * 100) if len(markers_in_ppi) > 0 else 0
        
        # Edge retention (edges in layer / potential edges in subgraph)
        potential_edges = len(base_ppi.subgraph(markers_in_ppi).edges()) if len(markers_in_ppi) > 0 else 0
        edge_retention_pct = (layer_graph.number_of_edges() / potential_edges * 100) if potential_edges > 0 else 0
        
        # Seed genes in this layer
        seeds_in_layer = seed_genes.intersection(set(layer_graph.nodes())) if seed_genes else set()
        
        # Check seed gene connectivity
        seed_genes_connected = False
        n_seed_components = 0
        if len(seeds_in_layer) > 1:
            seed_subgraph = layer_graph.subgraph(seeds_in_layer)
            n_seed_components = nx.number_connected_components(seed_subgraph)
            seed_genes_connected = (n_seed_components == 1)
        elif len(seeds_in_layer) == 1:
            seed_genes_connected = True
            n_seed_components = 1
        
        stat = {
            'cell_type': cell_type,
            'n_marker_genes': len(marker_genes_set),
            'n_markers_in_ppi': len(markers_in_ppi),
            'n_nodes_in_layer': layer_graph.number_of_nodes(),
            'n_edges_in_layer': layer_graph.number_of_edges(),
            'node_retention_pct': node_retention_pct,
            'edge_retention_pct': edge_retention_pct,
            'density': density,
            'avg_degree': avg_degree,
            'n_seed_genes': len(seeds_in_layer),
            'seed_gene_pct': len(seeds_in_layer) / len(seed_genes) * 100 if seed_genes else 0,
            'seed_genes_connected': seed_genes_connected,
            'n_seed_components': n_seed_components
        }
        
        layer_stats.append(stat)
        
        # Store per-celltype report
        report['per_celltype'][cell_type] = {
            'marker_genes': len(marker_genes_set),
            'markers_in_ppi': len(markers_in_ppi),
            'nodes_in_layer': layer_graph.number_of_nodes(),
            'edges_in_layer': layer_graph.number_of_edges(),
            'node_retention_pct': node_retention_pct,
            'edge_retention_pct': edge_retention_pct,
            'density': density,
            'avg_degree': avg_degree
        }
        
        # Track seed genes in this layer
        if seed_genes:
            report['seed_gene_analysis']['per_celltype'][cell_type] = {
                'seed_genes_in_layer': len(seeds_in_layer),
                'seed_gene_pct_of_total': (len(seeds_in_layer) / len(seed_genes) * 100) if seed_genes else 0,
                'seed_genes_connected': seed_genes_connected,
                'n_seed_components': n_seed_components
            }
    
    # ============================================================================
    # MULTIPLEX-LEVEL STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print("MULTIPLEX-LEVEL STATISTICS")
    print("="*80)
    
    all_layer_nodes = set().union(*[set(g.nodes()) for g in layers.values()])
    all_layer_edges = sum(g.number_of_edges() for g in layers.values())
    
    print(f"Total unique genes in multiplex: {len(all_layer_nodes)}")
    print(f"Total edges in multiplex: {all_layer_edges}")
    print(f"Number of layers: {len(layers)}")
    
    report['summary']['n_cell_types'] = len(layers)
    report['summary']['marker_genes_total'] = len(all_marker_genes)
    report['summary']['marker_genes_in_ppi'] = len(overlap)
    report['summary']['genes_in_multiplex'] = len(all_layer_nodes)
    report['summary']['genes_lost'] = len(overlap) - len(all_layer_nodes)
    report['summary']['gene_retention_pct'] = (len(all_layer_nodes) / len(overlap) * 100) if overlap else 0
    report['summary']['edges_in_multiplex'] = all_layer_edges
    
    # Seed gene summary
    if seed_genes:
        seeds_in_multiplex = seed_genes.intersection(all_layer_nodes)
        report['seed_gene_analysis']['summary']['seed_genes_in_multiplex'] = len(seeds_in_multiplex)
        report['seed_gene_analysis']['summary']['seed_genes_lost'] = len(seed_genes) - len(seeds_in_multiplex)
        report['seed_gene_analysis']['summary']['seed_gene_recovery_pct'] = (len(seeds_in_multiplex) / len(seed_genes) * 100) if seed_genes else 0
        
        print(f"\nSeed genes in multiplex: {len(seeds_in_multiplex)} / {len(seed_genes)} ({report['seed_gene_analysis']['summary']['seed_gene_recovery_pct']:.1f}%)")
    
    # ============================================================================
    # SAVE OUTPUTS
    # ============================================================================
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Save individual layers
    layers_dir = os.path.join(output_dir, 'layers')
    os.makedirs(layers_dir, exist_ok=True)
    
    for cell_type, layer_graph in layers.items():
        # Replace problematic characters in filename
        safe_cell_type = cell_type.replace('/', '_')
        layer_file = os.path.join(layers_dir, f"{safe_cell_type}_layer.pkl")
        with open(layer_file, 'wb') as f:
            pickle.dump(layer_graph, f, protocol=4)
    
    print(f"Saved {len(layers)} layer networks to {layers_dir}")
    
    # Save layer statistics
    stats_df = pd.DataFrame(layer_stats)
    stats_file = os.path.join(output_dir, 'layer_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved layer statistics to {stats_file}")
    
    # Save markers dataframe with layer assignments
    if markers_df is not None:
        markers_output_file = os.path.join(output_dir, 'markers_with_layers.csv')
        markers_df.to_csv(markers_output_file, index=False)
        print(f"Saved markers with layer assignments to {markers_output_file}")
    
    # ============================================================================
    # WRITE COMPREHENSIVE REPORT
    # ============================================================================
    report_file = os.path.join(output_dir, 'multiplex_conversion_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTIPLEX INTERACTOME FROM CELL-TYPE MARKERS - CONVERSION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of cell types: {report['summary']['n_cell_types']}\n")
        f.write(f"P-value threshold: {p_val_threshold}\n")
        f.write(f"Log2FC threshold: {log2fc_threshold}\n\n")
        
        f.write("MARKER GENE STATISTICS\n")
        f.write(f"  Total marker genes: {report['summary']['marker_genes_total']:,}\n")
        f.write(f"  Marker genes in PPI: {report['summary']['marker_genes_in_ppi']:,}\n")
        f.write(f"  Genes in multiplex: {report['summary']['genes_in_multiplex']:,}\n")
        f.write(f"  Genes lost: {report['summary']['genes_lost']:,}\n")
        f.write(f"  Gene retention: {report['summary']['gene_retention_pct']:.1f}%\n\n")
        
        f.write("EDGE STATISTICS\n")
        f.write(f"  Total edges in multiplex: {report['summary']['edges_in_multiplex']:,}\n\n")
        
        # Per cell type
        f.write("PER CELL TYPE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Cell Type':<40} {'Markers':<12} {'In PPI':<12} {'In Layer':<12} {'Node %':<10}\n")
        f.write("-"*80 + "\n")
        
        for cell_type in sorted(report['per_celltype'].keys()):
            ct_report = report['per_celltype'][cell_type]
            f.write(f"{cell_type:<40} ")
            f.write(f"{ct_report['marker_genes']:<12} ")
            f.write(f"{ct_report['markers_in_ppi']:<12} ")
            f.write(f"{ct_report['nodes_in_layer']:<12} ")
            f.write(f"{ct_report['node_retention_pct']:.1f}%\n")
        
        # Seed gene analysis
        if seed_genes:
            f.write("\n" + "="*80 + "\n")
            f.write("SEED GENE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            sg_summary = report['seed_gene_analysis']['summary']
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total seed genes: {sg_summary['total_seed_genes']}\n")
            f.write(f"Seed genes in marker genes: {sg_summary['seed_genes_in_markers']}\n")
            f.write(f"Seed genes in PPI: {sg_summary['seed_genes_in_ppi']}\n")
            f.write(f"Seed genes in both (before multiplex): {sg_summary['seed_genes_in_both_before']}\n")
            f.write(f"Seed genes in multiplex (after): {sg_summary['seed_genes_in_multiplex']}\n")
            f.write(f"Seed genes lost: {sg_summary['seed_genes_lost']}\n")
            f.write(f"Seed gene recovery: {sg_summary['seed_gene_recovery_pct']:.1f}%\n\n")
            
            f.write("PER CELL TYPE SEED GENE STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Cell Type':<40} {'In Layer':<15} {'% of Total':<15} {'Connected':<12}\n")
            f.write("-"*80 + "\n")
            
            for cell_type in sorted(report['seed_gene_analysis']['per_celltype'].keys()):
                sg_ct = report['seed_gene_analysis']['per_celltype'][cell_type]
                f.write(f"{cell_type:<40} ")
                f.write(f"{sg_ct['seed_genes_in_layer']:<15} ")
                f.write(f"{sg_ct['seed_gene_pct_of_total']:.1f}%{'':<10} ")
                f.write(f"{str(sg_ct['seed_genes_connected']):<12}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*80 + "\n\n")
        
        f.write("Gene Retention:\n")
        f.write(f"  Current: {report['summary']['gene_retention_pct']:.1f}%\n")
        if report['summary']['gene_retention_pct'] >= 80:
            f.write("  Status: ✓ EXCELLENT - Most genes retained\n")
        elif report['summary']['gene_retention_pct'] >= 50:
            f.write("  Status: ✓ GOOD - Majority of genes retained\n")
        elif report['summary']['gene_retention_pct'] >= 20:
            f.write("  Status: ⚠️  MODERATE - Significant gene loss\n")
        else:
            f.write("  Status: ❌ POOR - Severe gene loss\n")
        
        if seed_genes:
            f.write("\nSeed Gene Recovery:\n")
            f.write(f"  Current: {sg_summary['seed_gene_recovery_pct']:.1f}%\n")
            if sg_summary['seed_gene_recovery_pct'] >= 80:
                f.write("  Status: ✓ EXCELLENT - Most seed genes recovered\n")
            elif sg_summary['seed_gene_recovery_pct'] >= 50:
                f.write("  Status: ✓ GOOD - Majority of seed genes recovered\n")
            elif sg_summary['seed_gene_recovery_pct'] >= 20:
                f.write("  Status: ⚠️  MODERATE - Significant seed gene loss\n")
            else:
                f.write("  Status: ❌ POOR - Severe seed gene loss\n")
        
        f.write("\nWhere genes are lost:\n")
        f.write("  1. Marker filtering: Genes not meeting p-value/log2FC thresholds\n")
        f.write("  2. PPI filtering: Marker genes not in PPI network\n")
        f.write("  3. Connectivity: Genes with no edges in the subgraph\n")
    
    print(f"Saved comprehensive report to {report_file}")
    
    # Save JSON version
    json_report_file = os.path.join(output_dir, 'multiplex_conversion_report.json')
    with open(json_report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved JSON report to {json_report_file}")
    
    # Save metadata
    metadata = {
        'n_layers': len(layers),
        'n_total_genes': len(all_layer_nodes),
        'n_total_edges': all_layer_edges,
        'p_val_threshold': p_val_threshold,
        'log2fc_threshold': log2fc_threshold,
        'layer_stats': layer_stats,
        'report': report
    }
    
    metadata_file = os.path.join(output_dir, 'multiplex_metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f, protocol=4)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("MULTIPLEX INTERACTOME CONSTRUCTION COMPLETE")
    print("="*80)
    print(f"Total layers: {len(layers)}")
    print(f"Total unique genes: {len(all_layer_nodes)}")
    print(f"Total edges: {all_layer_edges}")
    print(f"Gene retention: {report['summary']['gene_retention_pct']:.1f}%")
    if seed_genes:
        print(f"Seed gene recovery: {sg_summary['seed_gene_recovery_pct']:.1f}%")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    return layers, metadata

def main():
    parser = argparse.ArgumentParser(
        description='Build multiplex interactome from cell-type marker genes'
    )
    parser.add_argument('--markers-file', type=str, required=True,
                      help='Path to cell-type markers CSV file')
    parser.add_argument('--base-ppi', type=str, required=True,
                      help='Path to base PPI network (pickle file)')
    parser.add_argument('--output-dir', type=str, default='multiplex_markers',
                      help='Output directory for multiplex network')
    parser.add_argument('--p-val-threshold', type=float, default=0.05,
                      help='P-value threshold for marker significance')
    parser.add_argument('--log2fc-threshold', type=float, default=0.0,
                      help='Log2 fold change threshold for markers')
    parser.add_argument('--seed-genes-file', type=str, default=None,
                      help='Path to seed genes CSV file (optional)')
    
    args = parser.parse_args()
    
    # Build multiplex interactome from markers
    layers, metadata = build_multiplex_from_markers(
        markers_file=args.markers_file,
        base_ppi_file=args.base_ppi,
        output_dir=args.output_dir,
        p_val_threshold=args.p_val_threshold,
        log2fc_threshold=args.log2fc_threshold,
        seed_genes_file=args.seed_genes_file
    )
    
    if layers is None:
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())