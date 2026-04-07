#!/usr/bin/env python3
"""
Compare top ranked genes across multiple multiplex RWR results

This script compares the top ranked genes from three different multiplex RWR analyses:
- rwr_ss_results (Tabula Sapiens Pain Subset)
- rwr_ts_results (Tabula Sapiens)
- rwr_2025_results (2025 hDRG Atlas)

It identifies:
1. Genes consistently ranked high across all multiplexes
2. Genes unique to specific multiplexes
3. Ranking consistency and agreement
4. Excludes seed genes from analysis
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def load_seed_genes(multixrank_dir):
    """Load seed genes from seeds.txt file"""
    seeds_file = os.path.join(multixrank_dir, 'seeds.txt')
    seed_genes = set()
    
    try:
        with open(seeds_file, 'r') as f:
            for line in f:
                gene = line.strip().upper()
                if gene:
                    seed_genes.add(gene)
        print(f"  Loaded {len(seed_genes)} seed genes from {multixrank_dir}")
        return seed_genes
    except FileNotFoundError:
        print(f"  Warning: seeds.txt not found in {multixrank_dir}")
        return set()


def load_multiplex_results(results_dir, name):
    """
    Load multiplex RWR results from output directory
    
    Parameters:
    - results_dir: directory containing multiplex_gene_scores.csv
    - name: name of this multiplex (for labeling)
    
    Returns:
    - DataFrame with gene scores
    - Set of seed genes
    """
    print(f"\nLoading {name}...")
    
    scores_file = os.path.join(results_dir, 'multiplex_gene_scores.csv')
    
    if not os.path.exists(scores_file):
        raise FileNotFoundError(f"Gene scores file not found: {scores_file}")
    
    # Load scores
    df = pd.read_csv(scores_file)
    print(f"  Loaded {len(df)} genes")
    print(f"  Columns: {list(df.columns)}")
    
    # Normalize gene names to uppercase for comparison
    df['Gene'] = df['Gene'].str.upper()
    
    # Load seed genes
    # Try to find the multixrank_dir by going up from results_dir
    # Typically: multixrank_setup/rwr_results -> multixrank_setup
    parent_dir = os.path.dirname(results_dir)
    seed_genes = load_seed_genes(parent_dir)
    
    return df, seed_genes


def filter_non_seed_genes(df, seed_genes, name):
    """
    Filter out seed genes from results
    
    Parameters:
    - df: DataFrame with gene scores
    - seed_genes: set of seed gene names
    - name: name of multiplex (for reporting)
    
    Returns:
    - Filtered DataFrame
    """
    n_before = len(df)
    df_filtered = df[~df['Gene'].isin(seed_genes)].copy()
    n_after = len(df_filtered)
    n_removed = n_before - n_after
    
    print(f"  Filtered {name}: removed {n_removed} seed genes, {n_after} genes remaining")
    
    return df_filtered


def get_top_genes(df, k=200, score_col='GMean_Score'):
    """
    Get top k genes by score
    
    Parameters:
    - df: DataFrame with gene scores
    - k: number of top genes
    - score_col: column to use for ranking
    
    Returns:
    - Set of top gene names
    - DataFrame of top genes with ranks
    """
    if score_col not in df.columns:
        # Try alternative column names
        possible_cols = [col for col in df.columns if 'score' in col.lower()]
        if possible_cols:
            score_col = possible_cols[0]
            print(f"    Using column: {score_col}")
        else:
            raise ValueError(f"Could not find score column. Available: {list(df.columns)}")
    
    df_sorted = df.sort_values(score_col, ascending=False).head(k).copy()
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    top_genes = set(df_sorted['Gene'].values)
    
    return top_genes, df_sorted


def compare_rankings(results_dict, k=200, score_col='GMean_Score'):
    """
    Compare rankings across multiplexes
    
    Parameters:
    - results_dict: dict of {name: (df, seed_genes)}
    - k: number of top genes to compare
    - score_col: column to use for ranking
    
    Returns:
    - Dictionary with comparison results
    """
    print(f"\n{'='*80}")
    print(f"COMPARING TOP {k} GENES ACROSS MULTIPLEXES")
    print(f"{'='*80}")
    
    # Get top genes for each multiplex
    top_genes_dict = {}
    top_dfs_dict = {}
    
    for name, (df, seed_genes) in results_dict.items():
        print(f"\n{name}:")
        df_filtered = filter_non_seed_genes(df, seed_genes, name)
        top_genes, top_df = get_top_genes(df_filtered, k, score_col)
        top_genes_dict[name] = top_genes
        top_dfs_dict[name] = top_df
        print(f"  Top {k} genes identified")
    
    # Get all unique genes across multiplexes
    all_genes = set()
    for genes in top_genes_dict.values():
        all_genes.update(genes)
    
    print(f"\nTotal unique genes in top {k} across all multiplexes: {len(all_genes)}")
    
    # Find genes in different categories
    names = list(top_genes_dict.keys())
    
    # Genes in all multiplexes
    genes_in_all = set.intersection(*[top_genes_dict[name] for name in names])
    
    # Genes in exactly 2 multiplexes
    genes_in_two = set()
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            genes_in_two.update(
                top_genes_dict[name1] & top_genes_dict[name2] - genes_in_all
            )
    
    # Genes in exactly 1 multiplex
    genes_in_one = all_genes - genes_in_all - genes_in_two
    
    print(f"\nGene distribution:")
    print(f"  In all 3 multiplexes: {len(genes_in_all)}")
    print(f"  In exactly 2 multiplexes: {len(genes_in_two)}")
    print(f"  In exactly 1 multiplex: {len(genes_in_one)}")
    
    # Create comparison results
    comparison_results = {
        'top_genes_dict': top_genes_dict,
        'top_dfs_dict': top_dfs_dict,
        'genes_in_all': genes_in_all,
        'genes_in_two': genes_in_two,
        'genes_in_one': genes_in_one,
        'all_genes': all_genes,
        'k': k,
        'score_col': score_col
    }
    
    return comparison_results


def create_comparison_dataframe(comparison_results, results_dict):
    """
    Create a comprehensive comparison DataFrame
    
    Parameters:
    - comparison_results: results from compare_rankings()
    - results_dict: dict of {name: (df, seed_genes)}
    
    Returns:
    - DataFrame with comparison info for all genes
    """
    print(f"\nCreating comparison DataFrame...")
    
    names = list(results_dict.keys())
    all_genes = comparison_results['all_genes']
    top_dfs_dict = comparison_results['top_dfs_dict']
    score_col = comparison_results['score_col']
    
    rows = []
    
    for gene in sorted(all_genes):
        row = {'Gene': gene}
        
        # Track which multiplexes have this gene in top k
        in_top_k = []
        ranks = []
        scores = []
        
        for name in names:
            top_df = top_dfs_dict[name]
            gene_data = top_df[top_df['Gene'] == gene]
            
            if len(gene_data) > 0:
                in_top_k.append(name)
                rank = gene_data['Rank'].values[0]
                score = gene_data[score_col].values[0]
                ranks.append(rank)
                scores.append(score)
                row[f'{name}_Rank'] = rank
                row[f'{name}_Score'] = score
            else:
                row[f'{name}_Rank'] = np.nan
                row[f'{name}_Score'] = np.nan
        
        # Summary columns
        row['N_Multiplexes'] = len(in_top_k)
        row['In_Multiplexes'] = ', '.join(in_top_k)
        row['Mean_Rank'] = np.nanmean(ranks) if ranks else np.nan
        row['Mean_Score'] = np.nanmean(scores) if scores else np.nan
        row['Rank_Std'] = np.nanstd(ranks) if len(ranks) > 1 else np.nan
        
        rows.append(row)
    
    comparison_df = pd.DataFrame(rows)
    
    # Sort by number of multiplexes (descending) then by mean rank
    comparison_df = comparison_df.sort_values(
        ['N_Multiplexes', 'Mean_Rank'],
        ascending=[False, True]
    )
    
    return comparison_df


def create_comparison_plots(comparison_results, comparison_df, output_dir):
    """Create visualization plots for comparison results"""
    print(f"\nCreating visualization plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # ===== PLOT 1: Gene Distribution Bar Chart =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['All 3\nMultiplexes', 'Exactly 2\nMultiplexes', 'Exactly 1\nMultiplex']
    counts = [
        len(comparison_results['genes_in_all']),
        len(comparison_results['genes_in_two']),
        len(comparison_results['genes_in_one'])
    ]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Genes', fontsize=12, fontweight='bold')
    ax.set_title('Gene Distribution Across Multiplexes\n(Excluding Seed Genes)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'gene_distribution_barplot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved gene distribution plot: {plot_file}")
    
    # ===== PLOT 2: Ranking Heatmap (Top genes in all multiplexes) =====
    genes_all_df = comparison_df[comparison_df['N_Multiplexes'] == 3].head(30).copy()
    
    if len(genes_all_df) > 0:
        names = list(comparison_results['top_genes_dict'].keys())
        
        # Create heatmap data
        heatmap_data = []
        for _, row in genes_all_df.iterrows():
            gene_ranks = []
            for name in names:
                rank_col = f'{name}_Rank'
                if rank_col in row and pd.notna(row[rank_col]):
                    gene_ranks.append(row[rank_col])
                else:
                    gene_ranks.append(np.nan)
            heatmap_data.append(gene_ranks)
        
        heatmap_df = pd.DataFrame(
            heatmap_data,
            columns=names,
            index=genes_all_df['Gene'].values
        )
        
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Create heatmap with inverted colormap (lower rank = darker = better)
        sns.heatmap(heatmap_df, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                    cbar_kws={'label': 'Rank (lower is better)'}, ax=ax,
                    linewidths=0.5, linecolor='gray')
        
        ax.set_title('Top 30 Genes in All Multiplexes - Ranking Heatmap\n(Lower rank = Better)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Multiplex', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'ranking_heatmap_all_multiplexes.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved ranking heatmap: {plot_file}")
    
    # ===== PLOT 3: Score Heatmap (Top genes in all multiplexes) =====
    if len(genes_all_df) > 0:
        # Create score heatmap data
        score_data = []
        for _, row in genes_all_df.iterrows():
            gene_scores = []
            for name in names:
                score_col = f'{name}_Score'
                if score_col in row and pd.notna(row[score_col]):
                    gene_scores.append(row[score_col])
                else:
                    gene_scores.append(np.nan)
            score_data.append(gene_scores)
        
        score_df = pd.DataFrame(
            score_data,
            columns=names,
            index=genes_all_df['Gene'].values
        )
        
        fig, ax = plt.subplots(figsize=(10, 12))
        
        sns.heatmap(score_df, annot=True, fmt='.4f', cmap='YlOrRd', 
                    cbar_kws={'label': 'RWR Score'}, ax=ax,
                    linewidths=0.5, linecolor='gray')
        
        ax.set_title('Top 30 Genes in All Multiplexes - Score Heatmap\n(Higher score = Better)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Multiplex', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'score_heatmap_all_multiplexes.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved score heatmap: {plot_file}")
    
    # ===== PLOT 4: Rank Consistency Scatter Plot =====
    genes_all_df = comparison_df[comparison_df['N_Multiplexes'] == 3].copy()
    
    if len(genes_all_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(genes_all_df['Mean_Rank'], genes_all_df['Rank_Std'],
                            s=100, alpha=0.6, c=genes_all_df['Mean_Score'],
                            cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Mean Score', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Mean Rank', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rank Standard Deviation', fontsize=12, fontweight='bold')
        ax.set_title('Rank Consistency Across Multiplexes\n(Genes in All 3 Multiplexes)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Annotate top 10 most consistent genes
        top_consistent = genes_all_df.nsmallest(10, 'Rank_Std')
        for _, row in top_consistent.iterrows():
            ax.annotate(row['Gene'], 
                       (row['Mean_Rank'], row['Rank_Std']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'rank_consistency_scatter.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved rank consistency plot: {plot_file}")
    
    # ===== PLOT 5: Venn-like Distribution (Pie Chart) =====
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sizes = [
        len(comparison_results['genes_in_all']),
        len(comparison_results['genes_in_two']),
        len(comparison_results['genes_in_one'])
    ]
    labels = [
        f'All 3 Multiplexes\n({sizes[0]} genes)',
        f'Exactly 2 Multiplexes\n({sizes[1]} genes)',
        f'Exactly 1 Multiplex\n({sizes[2]} genes)'
    ]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0.02, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        explode=explode, startangle=90, textprops={'fontsize': 11})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax.set_title('Gene Distribution Across Multiplexes\n(Excluding Seed Genes)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'gene_distribution_pie.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved gene distribution pie chart: {plot_file}")
    
    # ===== PLOT 6: Top Genes Comparison (Top 15 in all multiplexes) =====
    genes_all_df = comparison_df[comparison_df['N_Multiplexes'] == 3].head(15).copy()
    
    if len(genes_all_df) > 0:
        names = list(comparison_results['top_genes_dict'].keys())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(genes_all_df))
        width = 0.25
        
        for i, name in enumerate(names):
            score_col = f'{name}_Score'
            scores = genes_all_df[score_col].values
            ax.bar(x + i*width, scores, width, label=name, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Gene', fontsize=12, fontweight='bold')
        ax.set_ylabel('RWR Score', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Genes in All Multiplexes - Score Comparison', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(genes_all_df['Gene'].values, rotation=45, ha='right')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'top_genes_score_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved top genes comparison plot: {plot_file}")
    
    # ===== PLOT 7: Multiplex-specific genes (Top 10 from each) =====
    names = list(comparison_results['top_genes_dict'].keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for idx, name in enumerate(names):
        # Get genes unique to this multiplex
        unique_genes = comparison_results['top_genes_dict'][name] - comparison_results['genes_in_all'] - comparison_results['genes_in_two']
        
        if len(unique_genes) > 0:
            unique_df = comparison_df[comparison_df['Gene'].isin(unique_genes)].head(10)
            
            score_col = f'{name}_Score'
            unique_df_sorted = unique_df.sort_values(score_col, ascending=True)
            
            axes[idx].barh(unique_df_sorted['Gene'], unique_df_sorted[score_col], 
                          color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[idx].set_xlabel('RWR Score', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'Top 10 Unique Genes\n{name}', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
        else:
            axes[idx].text(0.5, 0.5, 'No unique genes', ha='center', va='center',
                          transform=axes[idx].transAxes, fontsize=12)
            axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    
    plt.suptitle('Multiplex-Specific Genes (Top 10 from Each)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'multiplex_specific_genes.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved multiplex-specific genes plot: {plot_file}")


def save_comparison_results(comparison_results, comparison_df, output_dir):
    """Save comparison results to files"""
    print(f"\nSaving results to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization plots
    create_comparison_plots(comparison_results, comparison_df, output_dir)
    
    # 1. Save comprehensive comparison DataFrame
    comparison_file = os.path.join(output_dir, 'gene_comparison_all.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"  ✓ Saved comprehensive comparison: {comparison_file}")
    
    # 2. Save genes in all multiplexes
    genes_all_file = os.path.join(output_dir, 'genes_in_all_multiplexes.csv')
    genes_all_df = comparison_df[comparison_df['N_Multiplexes'] == 3].copy()
    genes_all_df.to_csv(genes_all_file, index=False)
    print(f"  ✓ Saved genes in all 3 multiplexes ({len(genes_all_df)} genes): {genes_all_file}")
    
    # 3. Save genes in exactly 2 multiplexes
    genes_two_file = os.path.join(output_dir, 'genes_in_two_multiplexes.csv')
    genes_two_df = comparison_df[comparison_df['N_Multiplexes'] == 2].copy()
    genes_two_df.to_csv(genes_two_file, index=False)
    print(f"  ✓ Saved genes in exactly 2 multiplexes ({len(genes_two_df)} genes): {genes_two_file}")
    
    # 4. Save genes in exactly 1 multiplex
    genes_one_file = os.path.join(output_dir, 'genes_in_one_multiplex.csv')
    genes_one_df = comparison_df[comparison_df['N_Multiplexes'] == 1].copy()
    genes_one_df.to_csv(genes_one_file, index=False)
    print(f"  ✓ Saved genes in exactly 1 multiplex ({len(genes_one_df)} genes): {genes_one_file}")
    
    # 5. Save summary statistics
    summary_file = os.path.join(output_dir, 'comparison_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTIPLEX RANKING COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Top K genes compared: {comparison_results['k']}\n")
        f.write(f"Score column used: {comparison_results['score_col']}\n\n")
        
        f.write("GENE DISTRIBUTION:\n")
        f.write(f"  In all 3 multiplexes: {len(comparison_results['genes_in_all'])}\n")
        f.write(f"  In exactly 2 multiplexes: {len(comparison_results['genes_in_two'])}\n")
        f.write(f"  In exactly 1 multiplex: {len(comparison_results['genes_in_one'])}\n")
        f.write(f"  Total unique genes: {len(comparison_results['all_genes'])}\n\n")
        
        f.write("TOP GENES IN ALL MULTIPLEXES:\n")
        genes_all_sorted = sorted(
            comparison_results['genes_in_all'],
            key=lambda g: comparison_df[comparison_df['Gene'] == g]['Mean_Rank'].values[0]
        )
        for i, gene in enumerate(genes_all_sorted[:20], 1):
            mean_rank = comparison_df[comparison_df['Gene'] == gene]['Mean_Rank'].values[0]
            f.write(f"  {i:2d}. {gene:15s} (mean rank: {mean_rank:.1f})\n")
        if len(genes_all_sorted) > 20:
            f.write(f"  ... and {len(genes_all_sorted) - 20} more\n")
    
    print(f"  ✓ Saved summary: {summary_file}")
    
    # 6. Save as pickle for further analysis
    pickle_file = os.path.join(output_dir, 'comparison_results.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump({
            'comparison_results': comparison_results,
            'comparison_df': comparison_df
        }, f, protocol=4)
    print(f"  ✓ Saved pickle: {pickle_file}")
    
    print(f"\nAll results saved to: {output_dir}")


def print_detailed_comparison(comparison_results, comparison_df):
    """Print detailed comparison information"""
    print(f"\n{'='*80}")
    print("DETAILED COMPARISON")
    print(f"{'='*80}")
    
    # Top genes in all multiplexes
    print(f"\nTOP 20 GENES IN ALL 3 MULTIPLEXES:")
    genes_all = comparison_df[comparison_df['N_Multiplexes'] == 3].head(20)
    for idx, row in genes_all.iterrows():
        print(f"  {row['Gene']:15s} | Mean Rank: {row['Mean_Rank']:6.1f} | "
              f"Std: {row['Rank_Std']:5.1f} | Mean Score: {row['Mean_Score']:.6f}")
    
    # Genes unique to each multiplex
    names = list(comparison_results['top_genes_dict'].keys())
    
    for name in names:
        unique_genes = comparison_results['top_genes_dict'][name] - comparison_results['genes_in_all'] - comparison_results['genes_in_two']
        print(f"\nTOP 10 GENES UNIQUE TO {name}:")
        unique_df = comparison_df[comparison_df['Gene'].isin(unique_genes)].head(10)
        for idx, row in unique_df.iterrows():
            print(f"  {row['Gene']:15s} | Rank: {row[f'{name}_Rank']:.0f} | "
                  f"Score: {row[f'{name}_Score']:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare top ranked genes across multiple multiplex RWR results'
    )
    parser.add_argument('--ss-dir', type=str, required=True,
                      help='Output directory for Tabula Sapiens Pain Subset (rwr_ss_results)')
    parser.add_argument('--ts-dir', type=str, required=True,
                      help='Output directory for Tabula Sapiens (rwr_ts_results)')
    parser.add_argument('--atlas-dir', type=str, required=True,
                      help='Output directory for 2025 hDRG Atlas (rwr_2025_results)')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for comparison results')
    parser.add_argument('--k', type=int, default=200,
                      help='Number of top genes to compare (default: 200)')
    parser.add_argument('--score-col', type=str, default='GMean_Score',
                      help='Column to use for ranking (default: GMean_Score)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MULTIPLEX RANKING COMPARISON")
    print("="*80)
    print(f"Tabula Sapiens Pain Subset: {args.ss_dir}")
    print(f"Tabula Sapiens: {args.ts_dir}")
    print(f"2025 hDRG Atlas: {args.atlas_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Top K genes: {args.k}")
    print(f"Score column: {args.score_col}")
    print("="*80)
    
    try:
        # Load results from all three multiplexes
        results_dict = {
            'Tabula_Sapiens_Pain_Subset': load_multiplex_results(args.ss_dir, 'Tabula Sapiens Pain Subset'),
            'Tabula_Sapiens': load_multiplex_results(args.ts_dir, 'Tabula Sapiens'),
            'hDRG_Atlas_2025': load_multiplex_results(args.atlas_dir, '2025 hDRG Atlas')
        }
        
        # Compare rankings
        comparison_results = compare_rankings(results_dict, args.k, args.score_col)
        
        # Create comprehensive comparison DataFrame
        comparison_df = create_comparison_dataframe(comparison_results, results_dict)
        
        # Print detailed comparison
        print_detailed_comparison(comparison_results, comparison_df)
        
        # Save results
        save_comparison_results(comparison_results, comparison_df, args.output_dir)
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())