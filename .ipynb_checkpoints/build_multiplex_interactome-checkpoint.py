#!/usr/bin/env python3
"""
Build Multiplex Interactome from Cell-Type-Specific Expression Data

This script creates a multiplex network where:
- Each layer represents a cell type
- Nodes are genes expressed in that cell type
- Edges are PPI interactions weighted by co-expression
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
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
import time
import json

def load_celltype_expression(base_prefix, expression_threshold=0.000001, 
                            detection_rate_threshold=0.000001, expr_dir='converted_expression'):
    """Load cell type average expression data with detection rate filtering"""
    expr_file = os.path.join(expr_dir, f'{base_prefix}_celltype_avg_expression.csv')
    detection_file = os.path.join(expr_dir, f'{base_prefix}_celltype_detection_rate.csv')
    
    if not os.path.exists(expr_file):
        raise FileNotFoundError(f"Expression file not found: {expr_file}")
    
    print(f"Loading expression data from {expr_file}...")
    expr_df = pd.read_csv(expr_file, index_col=0)
    
    print(f"  Loaded {expr_df.shape[0]} genes × {expr_df.shape[1]} cell types")
    print(f"  Expression range: {expr_df.min().min():.3f} - {expr_df.max().max():.3f}")
    
    if os.path.exists(detection_file):
        print(f"  Loading detection rates from {detection_file}...")
        detection_df = pd.read_csv(detection_file, index_col=0)
        
        common_genes = expr_df.index.intersection(detection_df.index)
        
        if len(common_genes) < len(expr_df):
            print(f"  Note: {len(expr_df) - len(common_genes)} genes missing from detection file")
        
        expr_df = expr_df.loc[common_genes]
        detection_df = detection_df.loc[common_genes]
        
        print(f"  Filtering by detection rate (>{detection_rate_threshold*100}% of cells)...")
        detection_mask = detection_df.max(axis=1) >= detection_rate_threshold
        expr_df = expr_df.loc[detection_mask.index[detection_mask]]
        print(f"  After detection filtering: {expr_df.shape[0]} genes")
    
    print(f"  Filtering genes with max expression >= {expression_threshold}...")
    expr_df = expr_df[expr_df.max(axis=1) >= expression_threshold]
    print(f"  Final retained: {expr_df.shape[0]} genes")
    
    if not expr_df.index.is_unique:
        print(f"  ⚠️  Found duplicate genes - averaging within each cell type...")
        n_before = len(expr_df)
        expr_df = expr_df.groupby(level=0).mean()
        print(f"     Before: {n_before} rows, After: {len(expr_df)} unique genes")
    
    expr_df.index = expr_df.index.str.upper()
    
    return expr_df

def load_base_ppi(ppi_file):
    """Load base PPI network"""
    print(f"Loading base PPI network from {ppi_file}...")
    
    with open(ppi_file, 'rb') as f:
        G = pickle.load(f)
    
    print(f"  Loaded PPI with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    G = nx.relabel_nodes(G, {node: str(node).upper() for node in G.nodes()})
    print(f"  Converted node names to uppercase")
    
    return G

def calculate_coexpression_weights(expr_df, method='pearson'):
    """Calculate co-expression weights between genes"""
    print(f"Calculating co-expression weights using {method} correlation...")
    
    genes = expr_df.index.tolist()
    expr_matrix = expr_df.values
    
    if method == 'pearson':
        corr_matrix = np.corrcoef(expr_matrix)
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr_matrix, _ = spearmanr(expr_matrix, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    corr_matrix = np.abs(corr_matrix)
    corr_matrix = np.clip(corr_matrix, 0, 1)
    
    coexpr_weights = {}
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            if corr_matrix[i, j] > 0.1:
                coexpr_weights[(genes[i], genes[j])] = corr_matrix[i, j]
    
    print(f"  Calculated {len(coexpr_weights)} gene pairs with correlation > 0.1")
    
    return coexpr_weights

def build_layer_network(cell_type, expr_df, base_ppi, coexpr_weights, 
                       expression_threshold=0.5):
    """Build a single layer of the multiplex network for one cell type"""
    print(f"\nBuilding layer for cell type: {cell_type}")
    
    expressed_genes = expr_df[expr_df[cell_type] >= expression_threshold].index.tolist()
    print(f"  {len(expressed_genes)} genes expressed in {cell_type} (threshold: {expression_threshold})")
    
    ppi_genes_set = set(base_ppi.nodes())
    expressed_genes_set = set(expressed_genes)
    overlap = expressed_genes_set.intersection(ppi_genes_set)
    
    print(f"  Checking gene ID overlap:")
    print(f"    Expressed genes: {len(expressed_genes)}")
    print(f"    PPI genes: {len(ppi_genes_set)}")
    print(f"    Overlap: {len(overlap)} ({len(overlap)/len(expressed_genes)*100:.1f}%)")
    
    if len(overlap) < 10:
        print(f"  ⚠️  WARNING: Very low overlap! Possible gene ID mismatch.")
    
    layer_genes = [g for g in expressed_genes if g in base_ppi.nodes()]
    
    if len(layer_genes) == 0:
        print(f"  ❌ ERROR: No genes found in PPI network!")
        empty_graph = nx.Graph()
        empty_graph.graph['error'] = 'gene_id_mismatch'
        return empty_graph
    
    layer_graph = base_ppi.subgraph(layer_genes).copy()
    
    print(f"  {layer_graph.number_of_nodes()} genes have PPI edges")
    print(f"  {layer_graph.number_of_edges()} PPI edges in layer")
    
    if layer_graph.number_of_edges() == 0:
        print(f"  ⚠️  WARNING: Layer has 0 edges!")
    
    edges_with_weights = 0
    for u, v in layer_graph.edges():
        ppi_weight = layer_graph[u][v].get('weight', 1.0)
        coexpr_weight = coexpr_weights.get((u, v), coexpr_weights.get((v, u), 0))
        
        if coexpr_weight > 0:
            combined_weight = np.sqrt(ppi_weight * coexpr_weight)
            edges_with_weights += 1
        else:
            combined_weight = ppi_weight * 0.5
        
        layer_graph[u][v]['weight'] = combined_weight
        layer_graph[u][v]['ppi_weight'] = ppi_weight
        layer_graph[u][v]['coexpr_weight'] = coexpr_weight
    
    print(f"  {edges_with_weights} edges with co-expression support")
    
    for gene in layer_graph.nodes():
        layer_graph.nodes[gene]['expression'] = expr_df.loc[gene, cell_type]
        layer_graph.nodes[gene]['cell_type'] = cell_type
    
    return layer_graph

def build_multiplex_interactome_from_human(human_expr_data, base_ppi_file, output_dir,
                                           expression_threshold=0.5, coexpr_method='pearson',
                                           seed_genes_file='../og_gene_set/seed_genes.csv'):
    """
    Build multiplex interactome from pre-converted human orthologs expression data
    WITH COMPREHENSIVE LOSS TRACKING
    """
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize report dictionary
    report = {
        'summary': {},
        'per_celltype': {},
        'seed_gene_analysis': {'per_celltype': {}, 'summary': {}},
        'expression_filtering': {}
    }
    
    # Load human PPI
    base_ppi = load_base_ppi(base_ppi_file)
    
    # Record initial PPI stats
    ppi_genes_before = set(base_ppi.nodes())
    ppi_edges_before = base_ppi.number_of_edges()
    report['summary']['ppi_genes_before'] = len(ppi_genes_before)
    report['summary']['ppi_edges_before'] = ppi_edges_before
    
    # Use provided human expression data
    combined_expr = human_expr_data.copy()
    combined_expr = combined_expr.fillna(0)
    combined_expr.index = combined_expr.index.astype(str).str.upper()
    
    # Track expression filtering
    expr_genes_before = set(combined_expr.index)
    report['expression_filtering']['genes_in_expression_file'] = len(expr_genes_before)
    report['expression_filtering']['genes_in_ppi'] = len(ppi_genes_before)
    report['expression_filtering']['genes_in_both'] = len(expr_genes_before.intersection(ppi_genes_before))
    
    print(f"\nHuman expression matrix:")
    print(f"  {combined_expr.shape[0]} total genes")
    print(f"  {combined_expr.shape[1]} total cell types")
    
    # ============================================================================
    # VERIFICATION SECTION
    # ============================================================================
    print("\n" + "="*80)
    print("GENE ID VERIFICATION")
    print("="*80)
    
    expr_genes = set(combined_expr.index)
    ppi_genes = set(base_ppi.nodes())
    
    print(f"\n1. Gene ID Format Check:")
    print(f"   Expression genes: {len(expr_genes)}")
    print(f"   PPI genes: {len(ppi_genes)}")
    
    expr_examples = list(expr_genes)[:5]
    ppi_examples = list(ppi_genes)[:5]
    print(f"\n   Expression gene examples:")
    for i, gene in enumerate(expr_examples, 1):
        print(f"      {i}. {gene}")
    print(f"\n   PPI gene examples:")
    for i, gene in enumerate(ppi_examples, 1):
        print(f"      {i}. {gene}")
    
    overlap = expr_genes.intersection(ppi_genes)
    overlap_pct = len(overlap) / len(expr_genes) * 100 if expr_genes else 0
    
    print(f"\n2. Gene Overlap:")
    print(f"   Genes in both: {len(overlap)} ({overlap_pct:.1f}%)")
    
    if overlap_pct < 50:
        print(f"   ⚠️  WARNING: Low overlap! Likely gene ID format mismatch.")
    elif overlap_pct < 80:
        print(f"   ⚠️  NOTICE: Moderate overlap. Some ID conversion issues possible.")
    else:
        print(f"   ✓ Good overlap! Gene IDs appear compatible.")
    
    # Load seed genes if available
    seed_genes = set()
    if os.path.exists(seed_genes_file):
        print(f"\n3. Seed Gene Verification:")
        try:
            seed_df = pd.read_csv(seed_genes_file)
            gene_col = None
            for col in ['gene', 'Gene', 'gene_id', 'Gene_ID', 'symbol', 'Symbol']:
                if col in seed_df.columns:
                    gene_col = col
                    break
            if gene_col is None:
                gene_col = seed_df.columns[0]
            
            seed_genes = set(seed_df[gene_col].dropna().astype(str))
            seed_genes = set([g.upper() for g in seed_genes])
            print(f"   Total seed genes: {len(seed_genes)}")
            
            seeds_in_expr = seed_genes.intersection(expr_genes)
            seeds_in_expr_pct = len(seeds_in_expr) / len(seed_genes) * 100
            print(f"   Seeds in expression data: {len(seeds_in_expr)} ({seeds_in_expr_pct:.1f}%)")
            
            seeds_in_ppi = seed_genes.intersection(ppi_genes)
            seeds_in_ppi_pct = len(seeds_in_ppi) / len(seed_genes) * 100
            print(f"   Seeds in PPI network: {len(seeds_in_ppi)} ({seeds_in_ppi_pct:.1f}%)")
            
            seeds_in_both = seed_genes.intersection(overlap)
            seeds_in_both_pct = len(seeds_in_both) / len(seed_genes) * 100
            print(f"   Seeds in BOTH (expression AND PPI): {len(seeds_in_both)} ({seeds_in_both_pct:.1f}%)")
            
            # Store for later analysis
            report['seed_gene_analysis']['summary']['total_seed_genes'] = len(seed_genes)
            report['seed_gene_analysis']['summary']['seed_genes_in_expr'] = len(seeds_in_expr)
            report['seed_gene_analysis']['summary']['seed_genes_in_ppi'] = len(seeds_in_ppi)
            report['seed_gene_analysis']['summary']['seed_genes_in_both_before'] = len(seeds_in_both)
            
            if seeds_in_both_pct >= 80:
                print(f"\n   ✓✓ EXCELLENT: {seeds_in_both_pct:.1f}% of seed genes will be in multiplex network")
            elif seeds_in_both_pct >= 50:
                print(f"\n   ✓ GOOD: {seeds_in_both_pct:.1f}% of seed genes will be in multiplex network")
            elif seeds_in_both_pct >= 20:
                print(f"\n   ⚠️  POOR: Only {seeds_in_both_pct:.1f}% of seed genes will be in multiplex network")
            else:
                print(f"\n   ❌ CRITICAL: Only {seeds_in_both_pct:.1f}% of seed genes will be in multiplex network")
        except Exception as e:
            print(f"   Could not load seed genes: {str(e)}")
    
    print("="*80 + "\n")
    
    # Calculate co-expression weights
    print("\n" + "="*80)
    print("Calculating co-expression weights...")
    print("="*80)
    coexpr_weights = calculate_coexpression_weights(combined_expr, method=coexpr_method)
    
    # Build layer for each cell type
    print("\n" + "="*80)
    print("Building multiplex layers...")
    print("="*80)
    
    layers = {}
    layer_stats = []
    
    for cell_type in combined_expr.columns:
        layer_graph = build_layer_network(
            cell_type, combined_expr, base_ppi, coexpr_weights,
            expression_threshold=expression_threshold
        )
        
        layers[cell_type] = layer_graph
        
        # Calculate layer statistics
        if layer_graph.number_of_nodes() > 0:
            density = nx.density(layer_graph)
            avg_degree = 2 * layer_graph.number_of_edges() / layer_graph.number_of_nodes()
        else:
            density = 0
            avg_degree = 0
        
        # Get expressed genes for this cell type
        expressed_genes_ct = set(combined_expr[combined_expr[cell_type] >= expression_threshold].index)
        genes_in_ppi_ct = expressed_genes_ct.intersection(ppi_genes)
        
        # Calculate retention percentages
        node_retention_pct = (layer_graph.number_of_nodes() / len(genes_in_ppi_ct) * 100) if len(genes_in_ppi_ct) > 0 else 0
        
        # For edges, we need to count potential edges in the subgraph
        potential_edges = len(base_ppi.subgraph(genes_in_ppi_ct).edges()) if len(genes_in_ppi_ct) > 0 else 0
        edge_retention_pct = (layer_graph.number_of_edges() / potential_edges * 100) if potential_edges > 0 else 0
        
        layer_stats.append({
            'cell_type': cell_type,
            'n_expressed_genes': len(expressed_genes_ct),
            'n_expressed_in_ppi': len(genes_in_ppi_ct),
            'n_nodes_in_layer': layer_graph.number_of_nodes(),
            'n_edges_in_layer': layer_graph.number_of_edges(),
            'node_retention_pct': node_retention_pct,
            'edge_retention_pct': edge_retention_pct,
            'density': density,
            'avg_degree': avg_degree
        })
        
        # Store per-celltype report
        report['per_celltype'][cell_type] = {
            'expressed_genes': len(expressed_genes_ct),
            'expressed_genes_in_ppi': len(genes_in_ppi_ct),
            'nodes_in_layer': layer_graph.number_of_nodes(),
            'edges_in_layer': layer_graph.number_of_edges(),
            'node_retention_pct': node_retention_pct,
            'edge_retention_pct': edge_retention_pct,
            'density': density,
            'avg_degree': avg_degree
        }
        
        # Track seed genes in this layer
        if seed_genes:
            seeds_in_layer = seed_genes.intersection(set(layer_graph.nodes()))
            report['seed_gene_analysis']['per_celltype'][cell_type] = {
                'seed_genes_in_layer': len(seeds_in_layer),
                'seed_gene_pct_of_total': (len(seeds_in_layer) / len(seed_genes) * 100) if seed_genes else 0
            }
    
    # Calculate overall summary statistics
    all_layer_nodes = set().union(*[set(g.nodes()) for g in layers.values()])
    all_layer_edges = sum(g.number_of_edges() for g in layers.values())
    
    report['summary']['n_cell_types'] = len(layers)
    report['summary']['genes_in_multiplex'] = len(all_layer_nodes)
    report['summary']['genes_lost'] = len(ppi_genes_before) - len(all_layer_nodes)
    report['summary']['gene_retention_pct'] = (len(all_layer_nodes) / len(ppi_genes_before) * 100) if ppi_genes_before else 0
    report['summary']['edges_in_multiplex'] = all_layer_edges
    report['summary']['edges_lost'] = ppi_edges_before - all_layer_edges
    report['summary']['edge_retention_pct'] = (all_layer_edges / ppi_edges_before * 100) if ppi_edges_before else 0
    
    # Update seed gene summary
    if seed_genes:
        seeds_in_multiplex = seed_genes.intersection(all_layer_nodes)
        report['seed_gene_analysis']['summary']['seed_genes_in_multiplex'] = len(seeds_in_multiplex)
        report['seed_gene_analysis']['summary']['seed_genes_lost'] = len(seed_genes) - len(seeds_in_multiplex)
        report['seed_gene_analysis']['summary']['seed_gene_recovery_pct'] = (len(seeds_in_multiplex) / len(seed_genes) * 100) if seed_genes else 0
    
    # Save individual layers
    print("\n" + "="*80)
    print("Saving layer networks...")
    print("="*80)
    
    layers_dir = os.path.join(output_dir, 'layers')
    os.makedirs(layers_dir, exist_ok=True)
    
    for cell_type, layer_graph in layers.items():
        layer_file = os.path.join(layers_dir, f"{cell_type}_layer.pkl")
        with open(layer_file, 'wb') as f:
            pickle.dump(layer_graph, f, protocol=4)
        print(f"  Saved {cell_type}")
    
    # Save layer statistics
    stats_df = pd.DataFrame(layer_stats)
    stats_file = os.path.join(output_dir, 'layer_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"\nLayer statistics saved to {stats_file}")
    
    # ============================================================================
    # WRITE COMPREHENSIVE REPORT
    # ============================================================================
    report_file = os.path.join(output_dir, 'multiplex_conversion_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTIPLEX INTERACTOME CONSTRUCTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of cell types: {report['summary']['n_cell_types']}\n")
        f.write(f"Expression threshold: {expression_threshold}\n")
        f.write(f"Co-expression method: {coexpr_method}\n\n")
        
        f.write("GENE STATISTICS\n")
        f.write(f"  PPI genes (before): {report['summary']['ppi_genes_before']:,}\n")
        f.write(f"  Genes in multiplex (after): {report['summary']['genes_in_multiplex']:,}\n")
        f.write(f"  Genes lost: {report['summary']['genes_lost']:,}\n")
        f.write(f"  Gene retention: {report['summary']['gene_retention_pct']:.1f}%\n\n")
        
        f.write("EDGE STATISTICS\n")
        f.write(f"  PPI edges (before): {report['summary']['ppi_edges_before']:,}\n")
        f.write(f"  Edges in multiplex (after): {report['summary']['edges_in_multiplex']:,}\n")
        f.write(f"  Edges lost: {report['summary']['edges_lost']:,}\n")
        f.write(f"  Edge retention: {report['summary']['edge_retention_pct']:.1f}%\n\n")
        
        # Expression filtering details
        f.write("EXPRESSION FILTERING DETAILS\n")
        f.write("-"*80 + "\n")
        f.write(f"Genes in expression file: {report['expression_filtering']['genes_in_expression_file']:,}\n")
        f.write(f"Genes in PPI network: {report['expression_filtering']['genes_in_ppi']:,}\n")
        f.write(f"Genes in both (expression AND PPI): {report['expression_filtering']['genes_in_both']:,}\n")
        f.write(f"Overlap percentage: {(report['expression_filtering']['genes_in_both'] / report['expression_filtering']['genes_in_expression_file'] * 100):.1f}%\n\n")
        
        # Per cell type
        f.write("PER CELL TYPE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Cell Type':<40} {'Expressed':<12} {'In PPI':<12} {'In Layer':<12} {'Node %':<10} {'Edge %':<10}\n")
        f.write("-"*80 + "\n")
        
        for cell_type in sorted(report['per_celltype'].keys()):
            ct_report = report['per_celltype'][cell_type]
            f.write(f"{cell_type:<40} ")
            f.write(f"{ct_report['expressed_genes']:<12} ")
            f.write(f"{ct_report['expressed_genes_in_ppi']:<12} ")
            f.write(f"{ct_report['nodes_in_layer']:<12} ")
            f.write(f"{ct_report['node_retention_pct']:.1f}%{'':<6} ")
            f.write(f"{ct_report['edge_retention_pct']:.1f}%\n")
        
        # Seed gene analysis
        if seed_genes:
            f.write("\n" + "="*80 + "\n")
            f.write("SEED GENE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            sg_summary = report['seed_gene_analysis']['summary']
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total seed genes: {sg_summary['total_seed_genes']}\n")
            f.write(f"Seed genes in expression file: {sg_summary['seed_genes_in_expr']}\n")
            f.write(f"Seed genes in PPI network: {sg_summary['seed_genes_in_ppi']}\n")
            f.write(f"Seed genes in both (before multiplex): {sg_summary['seed_genes_in_both_before']}\n")
            f.write(f"Seed genes in multiplex (after): {sg_summary['seed_genes_in_multiplex']}\n")
            f.write(f"Seed genes lost: {sg_summary['seed_genes_lost']}\n")
            f.write(f"Seed gene recovery: {sg_summary['seed_gene_recovery_pct']:.1f}%\n\n")
            
            f.write("PER CELL TYPE SEED GENE STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Cell Type':<40} {'In Layer':<15} {'% of Total':<15}\n")
            f.write("-"*80 + "\n")
            
            for cell_type in sorted(report['seed_gene_analysis']['per_celltype'].keys()):
                sg_ct = report['seed_gene_analysis']['per_celltype'][cell_type]
                f.write(f"{cell_type:<40} ")
                f.write(f"{sg_ct['seed_genes_in_layer']:<15} ")
                f.write(f"{sg_ct['seed_gene_pct_of_total']:.1f}%\n")
        
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
        
        f.write("\nEdge Retention:\n")
        f.write(f"  Current: {report['summary']['edge_retention_pct']:.1f}%\n")
        if report['summary']['edge_retention_pct'] >= 80:
            f.write("  Status: ✓ EXCELLENT - Most edges retained\n")
        elif report['summary']['edge_retention_pct'] >= 50:
            f.write("  Status: ✓ GOOD - Majority of edges retained\n")
        elif report['summary']['edge_retention_pct'] >= 20:
            f.write("  Status: ⚠️  MODERATE - Significant edge loss\n")
        else:
            f.write("  Status: ❌ POOR - Severe edge loss\n")
        
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
        f.write("  1. Expression filtering: Genes not expressed above threshold\n")
        f.write("  2. PPI filtering: Expressed genes not in PPI network\n")
        f.write("  3. Connectivity: Genes with no edges in the subgraph\n")
        f.write("  4. Co-expression: Edges without co-expression support (weighted down)\n")
    
    print(f"\nSaved comprehensive report to {report_file}")
    
    # Save JSON version for programmatic access
    json_report_file = os.path.join(output_dir, 'multiplex_conversion_report.json')
    with open(json_report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved JSON report to {json_report_file}")
    
    # Save metadata
    metadata = {
        'n_layers': len(layers),
        'n_total_genes': len(all_layer_nodes),
        'expression_threshold': expression_threshold,
        'coexpr_method': coexpr_method,
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
    print(f"Edge retention: {report['summary']['edge_retention_pct']:.1f}%")
    if seed_genes:
        print(f"Seed gene recovery: {sg_summary['seed_gene_recovery_pct']:.1f}%")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    return layers, metadata

def main():
    parser = argparse.ArgumentParser(
        description='Build multiplex interactome from cell-type expression data'
    )
    parser.add_argument('--base-ppi', type=str, required=True,
                      help='Path to base PPI network (pickle file)')
    parser.add_argument('--human-expr-file', type=str, required=True,
                      help='Path to converted human orthologs expression file (CSV)')
    parser.add_argument('--output-dir', type=str, default='multiplex_interactome',
                      help='Output directory for multiplex network')
    parser.add_argument('--expression-threshold', type=float, default=0.000001,
                      help='Minimum expression threshold for gene inclusion')
    parser.add_argument('--coexpr-method', type=str, default='pearson',
                      choices=['pearson', 'spearman'],
                      help='Correlation method for co-expression')
    parser.add_argument('--seed-genes-file', type=str, default='../og_gene_set/seed_genes.csv',
                      help='Path to seed genes CSV file for verification')
    
    args = parser.parse_args()
    
    # Load human orthologs expression data directly
    print("Loading human orthologs expression data...")
    human_expr = pd.read_csv(args.human_expr_file, index_col=0)
    print(f"  Loaded {human_expr.shape[0]} human genes × {human_expr.shape[1]} cell types")
    
    # Build multiplex interactome from human data
    layers, metadata = build_multiplex_interactome_from_human(
        human_expr_data=human_expr,
        base_ppi_file=args.base_ppi,
        output_dir=args.output_dir,
        expression_threshold=args.expression_threshold,
        coexpr_method=args.coexpr_method,
        seed_genes_file=args.seed_genes_file
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())