#!/usr/bin/env python3
"""
Run Random Walk with Restart on Multiplex Interactome using MultiXrank

This script runs RWR starting from seed genes across all layers of the multiplex network
using the MultiXrank package, then processes results to identify disease modules.

Key differences from single-network RWR:
- MultiXrank finds global rankings across all layers simultaneously
- No need to manually extract connected components (MultiXrank handles this)
- Results are naturally aggregated across layers by the RWR algorithm
- Disease module = top N genes by RWR score (no need for connectivity filtering)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import time
from collections import defaultdict

try:
    import multixrank
except ImportError:
    print("ERROR: multixrank package not found!")
    print("Install with: pip install multixrank --break-system-packages")
    sys.exit(1)


def load_seed_genes_from_file(seed_file='seeds.txt'):
    """
    Load seed genes from seeds.txt file
    
    Parameters:
    - seed_file: path to seeds.txt (one gene per line)
    
    Returns:
    - List of seed gene IDs
    """
    seed_genes = []
    try:
        with open(seed_file, 'r') as f:
            for line in f:
                gene = line.strip()
                if gene:  # Skip empty lines
                    seed_genes.append(gene)
        print(f"  Loaded {len(seed_genes)} seed genes from {seed_file}")
        return seed_genes
    except Exception as e:
        print(f"ERROR loading seed genes from {seed_file}: {str(e)}")
        raise


def filter_seeds_to_network(seed_genes, layer_info):
    """
    Filter seed genes to only those that appear in at least one layer
    
    Parameters:
    - seed_genes: list of all seed genes
    - layer_info: dictionary with layer information (contains all_genes)
    
    Returns:
    - Tuple of (seeds_in_network, seeds_missing)
    """
    network_genes = set(layer_info['all_genes'])
    
    seeds_in_network = [g for g in seed_genes if g in network_genes]
    seeds_missing = set(seed_genes) - set(seeds_in_network)
    
    print(f"  Seeds in network: {len(seeds_in_network)} / {len(seed_genes)}")
    
    if seeds_missing:
        print(f"  WARNING: {len(seeds_missing)} seed genes not found in network:")
        missing_list = sorted(list(seeds_missing))[:20]
        print(f"    {', '.join(missing_list)}{'...' if len(seeds_missing) > 20 else ''}")
    
    return seeds_in_network, seeds_missing


def filter_seeds_to_tsv(multixrank_dir):
    """
    Filter seed genes to only those that actually appear in TSV network files.
    Some genes in layer_info might be isolated nodes with no edges.
    
    Parameters:
    - multixrank_dir: directory with TSV files
    
    Returns:
    - Set of genes that are in TSV files
    """
    print("\nFiltering seeds to genes in TSV files...")
    
    multiplex_subdir = os.path.join(multixrank_dir, 'multiplex', '1')
    if not os.path.exists(multiplex_subdir):
        print(f"  Warning: TSV directory not found: {multiplex_subdir}")
        print(f"  Skipping TSV-based filtering")
        return None
    
    # Collect all genes from TSV files
    genes_in_tsv = set()
    tsv_files = [f for f in os.listdir(multiplex_subdir) if f.endswith('.tsv')]
    
    for tsv_file in tsv_files:
        path = os.path.join(multiplex_subdir, tsv_file)
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        genes_in_tsv.add(parts[0])
                        genes_in_tsv.add(parts[1])
        except Exception as e:
            print(f"  Warning: Could not read {tsv_file}: {e}")
    
    print(f"  Genes in TSV files: {len(genes_in_tsv)}")
    return genes_in_tsv


def load_layer_info(multixrank_dir):
    """Load layer information from preparation step"""
    layer_info_file = os.path.join(multixrank_dir, 'multiplex_layer_info.pkl')
    
    if not os.path.exists(layer_info_file):
        raise FileNotFoundError(f"Layer info not found: {layer_info_file}")
    
    with open(layer_info_file, 'rb') as f:
        layer_info = pickle.load(f)
    
    return layer_info


def run_multixrank_rwr(multixrank_dir, restart_prob=0.7):
    """
    Run RWR on multiplex network using MultiXrank package
    
    Parameters:
    - multixrank_dir: directory with MultiXrank config and network files
    - restart_prob: restart probability (must match config)
    
    Returns:
    - DataFrame with RWR results (multiplex, node, layer, score)
    """
    print("\nRunning MultiXrank Random Walk with Restart...")
    
    # Get absolute path first - this is critical!
    abs_multixrank_dir = os.path.abspath(multixrank_dir)
    print(f"  Working directory: {abs_multixrank_dir}")
    
    # Check for config file using absolute path
    config_path = os.path.join(abs_multixrank_dir, 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"  Using config: {config_path}")
    
    # Change to working directory for MultiXrank
    original_dir = os.getcwd()
    os.chdir(abs_multixrank_dir)
    
    try:
        print("  Initializing MultiXrank object...")
        # Note: wdir parameter is required in newer MultiXrank versions
        # Pass just 'config.yml' since we're now in that directory
        multixrank_obj = multixrank.Multixrank(config='config.yml', wdir=abs_multixrank_dir)
        
        # Run random walk with restart
        print("  Executing random walk...")
        start_time = time.time()
        
        try:
            # MultiXrank.random_walk_rank() runs the RWR and returns results
            ranking_df = multixrank_obj.random_walk_rank()
        except Exception as e:
            error_msg = str(e)
            # Check if it's just a warning about seeds not found
            if "seed nodes were not found in the network" in error_msg.lower():
                print(f"\n  Note: Some seed genes not found in network layers")
                print(f"  This can happen if genes aren't expressed in certain contexts")
                print(f"  MultiXrank will continue with available seeds...")
                # Try to continue anyway - some versions throw warning as exception
                try:
                    ranking_df = multixrank_obj.random_walk_rank()
                except:
                    # If it really fails, re-raise
                    raise
            else:
                raise
        
        rwr_time = time.time() - start_time
        print(f"  RWR computation: {rwr_time:.2f} seconds")
        print(f"  Results shape: {ranking_df.shape}")
        print(f"  Columns: {list(ranking_df.columns)}")
        
        # Show sample results
        print(f"\n  Sample results (first 10 rows):")
        print(ranking_df.head(10))
        
        return ranking_df, multixrank_obj
        
    except Exception as e:
        error_msg = str(e)
        print(f"\nERROR during MultiXrank execution: {error_msg}")
        
        # Check if it's about missing seeds - if so, be more informative
        if "seed nodes were not found" in error_msg.lower():
            print("\n" + "="*80)
            print("SEED NODE MISMATCH")
            print("="*80)
            print("Some seed genes were not found in the network TSV files.")
            print("\nThis can happen because:")
            print("  1. Genes aren't expressed in ANY layer (filtered during build)")
            print("  2. Gene ID format mismatch between layer_info and TSV files")
            print("  3. Genes are in layer_info but edges weren't saved to TSV")
            print("\nPossible solutions:")
            print("  1. Check that build_multiplex_interactome.py ran successfully")
            print("  2. Verify TSV files were created: ls multixrank_setup/multiplex/1/*.tsv")
            print("  3. Lower expression threshold in build step")
            print("  4. Check seed gene capitalization matches network genes")
            print("="*80)
        else:
            print("\nNote: MultiXrank may fail if:")
            print("  1. Seed nodes are not in ANY layer of the network")
            print("  2. The config.yml file is malformed")
            print("  3. Network files are missing or corrupted")
        raise
    finally:
        os.chdir(original_dir)


def process_multixrank_results(ranking_df, layer_info, seed_genes):
    """
    Process MultiXrank results into aggregated gene scores
    
    MultiXrank returns one row per (node, layer) combination with the RWR score
    for that node in that specific layer.
    
    Parameters:
    - ranking_df: DataFrame from MultiXrank 
    - layer_info: dictionary with layer information
    - seed_genes: list of seed genes
    
    Returns:
    - aggregated_scores: dict of gene -> score info
    - layer_scores: dict of layer -> {gene: score}
    """
    print("\nProcessing MultiXrank results...")
    
    # MultiXrank results have columns: typically 'multiplex', 'node', 'layer', 'score'
    # If the column names are different, try to identify them
    print(f"  Result columns: {list(ranking_df.columns)}")
    
    # Map column names (they might be different)
    node_col = None
    score_col = None
    layer_col = None
    
    for col in ranking_df.columns:
        col_lower = col.lower()
        if col_lower in ['node', 'gene', 'protein']:
            node_col = col
        elif col_lower in ['score', 'rank', 'value', 'probability']:
            score_col = col
        elif col_lower in ['layer', 'sublayer']:
            layer_col = col
    
    if not node_col or not score_col:
        raise ValueError(f"Could not identify node and score columns in: {list(ranking_df.columns)}")
    
    print(f"  Using columns: node='{node_col}', score='{score_col}', layer='{layer_col}'")
    
    # Group by node to aggregate across layers
    gene_scores = defaultdict(lambda: {'scores': [], 'layers': []})
    layer_scores = defaultdict(dict)
    
    for _, row in ranking_df.iterrows():
        node = str(row[node_col])
        score = float(row[score_col])
        layer = str(row[layer_col]) if layer_col else 'unknown'
        
        gene_scores[node]['scores'].append(score)
        gene_scores[node]['layers'].append(layer)
        layer_scores[layer][node] = score
    
    # Compute aggregated scores across layers
    aggregated_scores = {}
    for gene, data in gene_scores.items():
        scores = np.array(data['scores'])
        aggregated_scores[gene] = {
            'max_score': float(np.max(scores)),
            'mean_score': float(np.mean(scores)),
            'median_score': float(np.median(scores)),
            'sum_score': float(np.sum(scores)),
            'gmean_score': float(np.exp(np.mean(np.log(scores + 1e-10)))),  # Geometric mean
            'n_layers': len(scores),
            'layer_scores': dict(zip(data['layers'], [float(s) for s in scores]))
        }
    
    print(f"  Processed scores for {len(aggregated_scores)} genes")
    print(f"  Number of layers with results: {len(layer_scores)}")
    
    return aggregated_scores, dict(layer_scores)


def extract_disease_module(aggregated_scores, seed_genes, k=200, aggregation='gmean'):
    """
    Extract disease module from RWR scores
    
    For multiplex RWR, the disease module is simply the top N genes by aggregated score.
    There's no need for connectivity filtering because:
    1. RWR naturally produces a ranked list
    2. The multiplex structure preserves connectivity across layers
    3. Top genes are already biologically relevant
    
    Parameters:
    - aggregated_scores: dict of gene -> score info
    - seed_genes: list of seed genes
    - k: number of top genes to include (in addition to seeds)
    - aggregation: 'max', 'mean', 'median', 'gmean', or 'sum' for score aggregation
    
    Returns:
    - DataFrame with disease module info
    """
    print(f"\nExtracting disease module (k={k}, aggregation={aggregation})...")
    
    # Get score key
    score_key = f'{aggregation}_score'
    
    # Filter to genes that have scores
    scored_genes = {g: s for g, s in aggregated_scores.items() if score_key in s}
    
    # Rank genes by aggregated score
    ranked_genes = sorted(
        scored_genes.items(),
        key=lambda x: x[1][score_key],
        reverse=True
    )
    
    # Convert seed_genes to set of strings for comparison
    seed_genes_set = set(str(g).upper() for g in seed_genes)
    
    # Create module: start with seeds, then add top k non-seed genes
    module_genes = []
    added_non_seeds = 0
    
    for gene, scores in ranked_genes:
        if str(gene).upper() in seed_genes_set:
            # Always include seed genes
            module_genes.append((gene, scores, True))
        else:
            # Add non-seed genes up to limit
            if added_non_seeds < k:
                module_genes.append((gene, scores, False))
                added_non_seeds += 1
    
    print(f"  Disease module size: {len(module_genes)} genes")
    print(f"  Seed genes: {sum(1 for _, _, is_seed in module_genes if is_seed)}")
    print(f"  Added genes: {sum(1 for _, _, is_seed in module_genes if not is_seed)}")
    
    # Create DataFrame
    module_df = pd.DataFrame([
        {
            'Gene': gene,
            f'{aggregation.upper()}_Score': scores[score_key],
            'Max_Score': scores['max_score'],
            'Mean_Score': scores['mean_score'],
            'N_Layers': scores['n_layers'],
            'Is_Seed': is_seed,
            'Rank': idx + 1
        }
        for idx, (gene, scores, is_seed) in enumerate(module_genes)
    ])
    
    return module_df


def save_multiplex_rwr_results(aggregated_scores, layer_scores, disease_module_df,
                                ranking_df, seed_genes, layer_info, output_dir,
                                multixrank_obj=None):
    """Save multiplex RWR results to output directory"""
    print("\nSaving results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert seed_genes to set of strings (uppercase)
    seed_genes_set = set(str(g).upper() for g in seed_genes)
    
    # 1. Save raw MultiXrank results
    raw_results_file = os.path.join(output_dir, 'multixrank_raw_results.csv')
    ranking_df.to_csv(raw_results_file, index=False)
    print(f"  ✓ Saved raw MultiXrank results: {raw_results_file}")
    
    # 2. Save aggregated gene scores
    scores_rows = []
    for gene, scores in aggregated_scores.items():
        row = {
            'Gene': gene,
            'Max_Score': scores['max_score'],
            'Mean_Score': scores['mean_score'],
            'Median_Score': scores['median_score'],
            'Sum_Score': scores['sum_score'],
            'GMean_Score': scores['gmean_score'],
            'N_Layers': scores['n_layers'],
            'Is_Seed': str(gene).upper() in seed_genes_set
        }
        # Add layer-specific scores
        for layer, score in scores['layer_scores'].items():
            row[f'Score_{layer}'] = score
        scores_rows.append(row)
    
    scores_df = pd.DataFrame(scores_rows)
    scores_df = scores_df.sort_values('GMean_Score', ascending=False)
    
    scores_file = os.path.join(output_dir, 'multiplex_gene_scores.csv')
    scores_df.to_csv(scores_file, index=False)
    print(f"  ✓ Saved all gene scores: {scores_file}")
    
    # 3. Save disease module
    module_file = os.path.join(output_dir, 'multiplex_disease_module.csv')
    disease_module_df.to_csv(module_file, index=False)
    print(f"  ✓ Saved disease module: {module_file}")
    
    # 4. Save layer-specific results
    layer_files_created = 0
    for layer_name, layer_genes in layer_scores.items():
        layer_df = pd.DataFrame([
            {'Gene': gene, 'Score': score, 'Is_Seed': str(gene).upper() in seed_genes_set}
            for gene, score in layer_genes.items()
        ])
        layer_df = layer_df.sort_values('Score', ascending=False)
        
        # Sanitize layer name for filename
        safe_layer_name = str(layer_name).replace('/', '_').replace('\\', '_').replace(' ', '_')
        layer_file = os.path.join(output_dir, f'layer_{safe_layer_name}_scores.csv')
        layer_df.to_csv(layer_file, index=False)
        layer_files_created += 1
    
    print(f"  ✓ Saved {layer_files_created} layer-specific score files")
    
    # 5. Save summary statistics
    summary = {
        'method': 'MultiXrank',
        'n_layers': len(layer_scores),
        'layer_names': list(layer_scores.keys()),
        'n_seed_genes_provided': len(seed_genes),
        'n_seed_genes_in_network': sum(1 for g in seed_genes if str(g).upper() in aggregated_scores),
        'n_genes_scored': len(aggregated_scores),
        'disease_module_size': len(disease_module_df),
        'n_seed_genes_in_module': sum(disease_module_df['Is_Seed']),
        'n_added_genes': len(disease_module_df) - sum(disease_module_df['Is_Seed'])
    }
    
    summary_file = os.path.join(output_dir, 'multiplex_rwr_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTIPLEX RWR ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        for key, value in summary.items():
            if isinstance(value, list):
                f.write(f"{key}: ({len(value)} items)\n")
                if len(value) <= 10:
                    for item in value:
                        f.write(f"  - {item}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"  ✓ Saved summary: {summary_file}")
    
    # 6. Save summary as pickle
    summary_pkl_file = os.path.join(output_dir, 'multiplex_rwr_summary.pkl')
    with open(summary_pkl_file, 'wb') as f:
        pickle.dump(summary, f, protocol=4)
    
    print(f"\n  Results saved to: {output_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run RWR on multiplex interactome using MultiXrank'
    )
    parser.add_argument('--multixrank-dir', type=str, required=True,
                      help='Directory containing MultiXrank config and network files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for results')
    parser.add_argument('--restart', type=float, default=0.7,
                      help='Restart probability (must match config, default: 0.7)')
    parser.add_argument('--k', type=int, default=200,
                      help='Number of top non-seed genes to include in disease module (default: 200)')
    parser.add_argument('--aggregation', type=str, default='gmean',
                      choices=['max', 'mean', 'median', 'gmean', 'sum'],
                      help='Score aggregation method across layers (default: gmean)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MULTIPLEX RWR EXECUTION - MultiXrank")
    print("="*80)
    print(f"MultiXrank directory: {args.multixrank_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Restart probability: {args.restart}")
    print(f"k (top genes): {args.k}")
    print(f"Aggregation: {args.aggregation}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        # Load layer info
        print("Loading configuration...")
        layer_info = load_layer_info(args.multixrank_dir)
        print(f"  Layers: {layer_info['n_layers']}")
        print(f"  Genes: {layer_info['n_genes']}")
        
        # Load seed genes from seeds.txt
        print("\nLoading seed genes...")
        seeds_file = os.path.join(args.multixrank_dir, 'seeds.txt')
        seed_genes = load_seed_genes_from_file(seeds_file)
        
        if not seed_genes:
            raise ValueError("No seed genes loaded from seeds.txt!")
        
        print(f"  Total seed genes: {len(seed_genes)}")
        
        # Filter seeds to only those in the network
        print("\nFiltering seeds to network genes...")
        seed_genes_in_network, seeds_missing = filter_seeds_to_network(seed_genes, layer_info)
        
        if not seed_genes_in_network:
            raise ValueError("No seed genes found in network! Gene ID mismatch?")
        
        # Additional filtering: remove seeds that aren't in TSV files
        # (some might be isolated nodes)
        genes_in_tsv = filter_seeds_to_tsv(args.multixrank_dir)
        if genes_in_tsv:
            seed_genes_before_tsv_filter = len(seed_genes_in_network)
            seed_genes_in_network = [g for g in seed_genes_in_network if g in genes_in_tsv]
            seeds_removed = seed_genes_before_tsv_filter - len(seed_genes_in_network)
            if seeds_removed > 0:
                print(f"  Removed {seeds_removed} seeds that aren't in TSV files (isolated nodes)")
            print(f"  Seeds after TSV filtering: {len(seed_genes_in_network)}")
        
        # UPDATE seeds.txt to only include seeds that pass both filters
        # This prevents MultiXrank from complaining about missing seeds
        print(f"\nUpdating seeds.txt to only include valid genes...")
        with open(seeds_file, 'w') as f:
            for gene in seed_genes_in_network:
                f.write(f"{gene}\n")
        print(f"  Updated seeds.txt with {len(seed_genes_in_network)} genes")
        
        # Run MultiXrank RWR
        ranking_df, multixrank_obj = run_multixrank_rwr(
            args.multixrank_dir,
            args.restart
        )
        
        # Process results
        aggregated_scores, layer_scores = process_multixrank_results(
            ranking_df, layer_info, seed_genes_in_network
        )
        
        # Extract disease module
        disease_module_df = extract_disease_module(
            aggregated_scores, seed_genes_in_network, args.k, args.aggregation
        )
        
        # Save results
        summary = save_multiplex_rwr_results(
            aggregated_scores, layer_scores, disease_module_df,
            ranking_df, seed_genes_in_network, layer_info, args.output_dir,
            multixrank_obj
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("MULTIPLEX RWR COMPLETE")
        print("="*80)
        print(f"Method: MultiXrank")
        print(f"Layers: {summary['n_layers']}")
        print(f"Genes scored: {summary['n_genes_scored']}")
        print(f"Seed genes: {summary['n_seed_genes_provided']}")
        print(f"  In network: {summary['n_seed_genes_in_network']}")
        print(f"Disease module size: {summary['disease_module_size']}")
        print(f"  Seed genes: {summary['n_seed_genes_in_module']}")
        print(f"  Added genes: {summary['n_added_genes']}")
        print(f"Execution time: {elapsed_time:.2f} seconds")
        print(f"Results: {args.output_dir}")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}")
        return 1


if __name__ == "__main__":
    sys.exit(main())