#!/usr/bin/env python3
"""
Compare Single PPI vs Multiplex Interactome Results

This script provides direct comparison between your original single PPI analysis
and the new multiplex analysis to highlight cell-type-specific findings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_single_ppi_results(output_dir='../og_gene_set/output/combined/rwr'):
    """Load results from single PPI RWR"""
    print("Loading single PPI results...")
    
    # Disease module
    module_file = os.path.join(output_dir, 'disease_module.csv')
    if os.path.exists(module_file):
        module_df = pd.read_csv(module_file)
        print(f"  Single PPI disease module: {len(module_df)} genes")
    else:
        print(f"  WARNING: {module_file} not found")
        module_df = None
    
    # Gene rankings
    rankings_file = os.path.join(output_dir, 'all_gene_rankings.csv')
    if os.path.exists(rankings_file):
        rankings_df = pd.read_csv(rankings_file)
        print(f"  Single PPI rankings: {len(rankings_df)} genes")
    else:
        print(f"  WARNING: {rankings_file} not found")
        rankings_df = None
    
    return {'module': module_df, 'rankings': rankings_df}

def load_multiplex_results(output_dir='output'):
    """Load results from multiplex RWR"""
    print("\nLoading multiplex results...")
    
    # Disease module
    module_file = os.path.join(output_dir, 'multiplex_disease_module.csv')
    if os.path.exists(module_file):
        module_df = pd.read_csv(module_file)
        print(f"  Multiplex disease module: {len(module_df)} genes")
    else:
        print(f"  WARNING: {module_file} not found")
        module_df = None
    
    # Gene scores
    scores_file = os.path.join(output_dir, 'multiplex_gene_scores.csv')
    if os.path.exists(scores_file):
        scores_df = pd.read_csv(scores_file)
        print(f"  Multiplex gene scores: {len(scores_df)} genes")
    else:
        print(f"  WARNING: {scores_file} not found")
        scores_df = None
    
    return {'module': module_df, 'scores': scores_df}

def compare_disease_modules(single_results, multiplex_results, output_dir='output/comparisons'):
    """Compare disease modules between methods"""
    print("\n" + "="*60)
    print("DISEASE MODULE COMPARISON")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    single_module = set(single_results['module']['Gene'])
    multiplex_module = set(multiplex_results['module']['Gene'])
    
    # Overlap analysis
    overlap = single_module.intersection(multiplex_module)
    single_only = single_module - multiplex_module
    multiplex_only = multiplex_module - single_module
    
    print(f"\nOverlap Analysis:")
    print(f"  Single PPI module:     {len(single_module)} genes")
    print(f"  Multiplex module:      {len(multiplex_module)} genes")
    print(f"  Shared genes:          {len(overlap)} ({len(overlap)/len(single_module)*100:.1f}%)")
    print(f"  Single-only genes:     {len(single_only)}")
    print(f"  Multiplex-only genes:  {len(multiplex_only)}")
    
    # Create Venn diagram data
    venn_data = pd.DataFrame({
        'Category': ['Shared', 'Single PPI Only', 'Multiplex Only'],
        'Count': [len(overlap), len(single_only), len(multiplex_only)]
    })
    venn_data.to_csv(os.path.join(output_dir, 'module_overlap.csv'), index=False)
    
    # Save gene lists
    pd.DataFrame({'Gene': list(overlap)}).to_csv(
        os.path.join(output_dir, 'shared_genes.csv'), index=False)
    pd.DataFrame({'Gene': list(single_only)}).to_csv(
        os.path.join(output_dir, 'single_ppi_only_genes.csv'), index=False)
    pd.DataFrame({'Gene': list(multiplex_only)}).to_csv(
        os.path.join(output_dir, 'multiplex_only_genes.csv'), index=False)
    
    # Visualize overlap
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(venn_data['Category'], venn_data['Count'], color=['#1B88A7', '#96C03A', '#E95E30'])
    ax.set_ylabel('Number of Genes')
    ax.set_title('Disease Module Comparison: Single PPI vs. Multiplex')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'module_overlap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return overlap, single_only, multiplex_only

def compare_gene_rankings(single_results, multiplex_results, output_dir='output/comparisons'):
    """Compare gene rankings between methods"""
    print("\n" + "="*60)
    print("GENE RANKING COMPARISON")
    print("="*60)
    
    # Merge rankings
    single_ranks = single_results['rankings'][['Gene', 'RWR_Score']].copy()
    single_ranks.columns = ['Gene', 'Single_PPI_Score']
    
    multiplex_ranks = multiplex_results['scores'][['Gene', 'Max_Score', 'Mean_Score']].copy()
    multiplex_ranks.columns = ['Gene', 'Multiplex_Max_Score', 'Multiplex_Mean_Score']
    
    merged = single_ranks.merge(multiplex_ranks, on='Gene', how='outer')
    merged = merged.fillna(0)
    
    # Calculate rank correlation
    from scipy.stats import spearmanr, pearsonr
    
    mask = (merged['Single_PPI_Score'] > 0) & (merged['Multiplex_Max_Score'] > 0)
    
    if mask.sum() > 0:
        spearman_corr, spearman_p = spearmanr(
            merged.loc[mask, 'Single_PPI_Score'],
            merged.loc[mask, 'Multiplex_Max_Score']
        )
        pearson_corr, pearson_p = pearsonr(
            merged.loc[mask, 'Single_PPI_Score'],
            merged.loc[mask, 'Multiplex_Max_Score']
        )
        
        print(f"\nScore Correlation:")
        print(f"  Spearman ρ: {spearman_corr:.3f} (p={spearman_p:.2e})")
        print(f"  Pearson r:  {pearson_corr:.3f} (p={pearson_p:.2e})")
    
    # Save merged rankings
    merged.to_csv(os.path.join(output_dir, 'merged_rankings.csv'), index=False)
    
    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Single vs Multiplex Max
    ax = axes[0]
    ax.scatter(merged['Single_PPI_Score'], merged['Multiplex_Max_Score'], 
              alpha=0.3, s=10, color='#1B88A7')
    ax.set_xlabel('Single PPI RWR Score')
    ax.set_ylabel('Multiplex Max RWR Score')
    ax.set_title(f'Score Comparison\nSpearman ρ = {spearman_corr:.3f}')
    ax.plot([0, merged['Single_PPI_Score'].max()], [0, merged['Single_PPI_Score'].max()], 
           'r--', alpha=0.5, label='y=x')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Single vs Multiplex Mean
    ax = axes[1]
    ax.scatter(merged['Single_PPI_Score'], merged['Multiplex_Mean_Score'],
              alpha=0.3, s=10, color='#E95E30')
    ax.set_xlabel('Single PPI RWR Score')
    ax.set_ylabel('Multiplex Mean RWR Score')
    ax.set_title('Score Comparison (Mean Across Layers)')
    ax.plot([0, merged['Single_PPI_Score'].max()], [0, merged['Single_PPI_Score'].max()],
           'r--', alpha=0.5, label='y=x')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return merged

def identify_celltype_specific_genes(multiplex_results, threshold=2.0, output_dir='output/comparisons'):
    """Identify genes with strong cell-type-specific activity"""
    print("\n" + "="*60)
    print("CELL-TYPE-SPECIFIC GENE ANALYSIS")
    print("="*60)
    
    scores_df = multiplex_results['scores']
    
    # Get layer-specific score columns
    layer_cols = [c for c in scores_df.columns if c.startswith('Score_')]
    
    if not layer_cols:
        print("  WARNING: No layer-specific scores found")
        return None
    
    # For each gene, find if it's highly enriched in specific layer
    celltype_specific_genes = []
    
    for _, row in scores_df.iterrows():
        gene = row['Gene']
        layer_scores = row[layer_cols]
        
        if layer_scores.max() == 0:
            continue
        
        max_score = layer_scores.max()
        mean_score = layer_scores.mean()
        
        # Gene is cell-type-specific if max >> mean
        if max_score > threshold * mean_score:
            max_layer = layer_scores.idxmax().replace('Score_', '')
            
            celltype_specific_genes.append({
                'Gene': gene,
                'Primary_Layer': max_layer,
                'Max_Score': max_score,
                'Mean_Score': mean_score,
                'Specificity_Ratio': max_score / mean_score if mean_score > 0 else np.inf,
                'Is_Seed': row['Is_Seed'] if 'Is_Seed' in row else False
            })
    
    if not celltype_specific_genes:
        print("  No cell-type-specific genes found with current threshold")
        return None
    
    specific_df = pd.DataFrame(celltype_specific_genes)
    specific_df = specific_df.sort_values('Specificity_Ratio', ascending=False)
    
    print(f"\nFound {len(specific_df)} cell-type-specific genes:")
    
    # Count by primary layer
    layer_counts = specific_df['Primary_Layer'].value_counts()
    for layer, count in layer_counts.items():
        print(f"  {layer}: {count} genes")
    
    # Save results
    specific_df.to_csv(os.path.join(output_dir, 'celltype_specific_genes.csv'), index=False)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Count by layer
    ax = axes[0]
    layer_counts.plot(kind='bar', ax=ax, color='#1B88A7')
    ax.set_ylabel('Number of Specific Genes')
    ax.set_title('Cell-Type-Specific Gene Distribution')
    ax.set_xlabel('Primary Cell Type')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Specificity ratio distribution
    ax = axes[1]
    ax.hist(specific_df['Specificity_Ratio'], bins=30, color='#E95E30', alpha=0.7)
    ax.set_xlabel('Specificity Ratio (Max/Mean)')
    ax.set_ylabel('Number of Genes')
    ax.set_title('Cell-Type Specificity Distribution')
    ax.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'celltype_specificity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print top specific genes
    print("\nTop 10 Most Cell-Type-Specific Genes:")
    print(specific_df.head(10)[['Gene', 'Primary_Layer', 'Specificity_Ratio', 'Is_Seed']].to_string(index=False))
    
    return specific_df

def generate_summary_report(single_results, multiplex_results, overlap, single_only, 
                           multiplex_only, specific_genes, output_dir='output/comparisons'):
    """Generate comprehensive comparison report"""
    
    report_file = os.path.join(output_dir, 'comparison_summary.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SINGLE PPI VS MULTIPLEX INTERACTOME - COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Module sizes
        f.write("DISEASE MODULE SIZES\n")
        f.write("-"*40 + "\n")
        f.write(f"Single PPI:  {len(single_results['module'])} genes\n")
        f.write(f"Multiplex:   {len(multiplex_results['module'])} genes\n\n")
        
        # Overlap
        f.write("MODULE OVERLAP\n")
        f.write("-"*40 + "\n")
        f.write(f"Shared genes:          {len(overlap)}\n")
        f.write(f"Single PPI only:       {len(single_only)}\n")
        f.write(f"Multiplex only:        {len(multiplex_only)}\n")
        f.write(f"Overlap percentage:    {len(overlap)/len(single_results['module'])*100:.1f}%\n\n")
        
        # Cell-type specificity
        if specific_genes is not None:
            f.write("CELL-TYPE-SPECIFIC GENES\n")
            f.write("-"*40 + "\n")
            f.write(f"Total specific genes:  {len(specific_genes)}\n\n")
            
            f.write("Distribution by cell type:\n")
            layer_counts = specific_genes['Primary_Layer'].value_counts()
            for layer, count in layer_counts.items():
                f.write(f"  {layer}: {count}\n")
            
            f.write("\n")
        
        # Key findings
        f.write("KEY FINDINGS\n")
        f.write("-"*40 + "\n")
        f.write("1. Multiplex analysis reveals cell-type-specific gene activity\n")
        f.write(f"2. {len(overlap)} genes are consistently identified by both methods\n")
        f.write(f"3. {len(multiplex_only)} new genes discovered through multiplex approach\n")
        
        if specific_genes is not None:
            f.write(f"4. {len(specific_genes)} genes show strong cell-type specificity\n")
        
        f.write("\n")
        
        f.write("INTERPRETATION\n")
        f.write("-"*40 + "\n")
        f.write("- Shared genes represent core pain mechanisms active across cell types\n")
        f.write("- Cell-type-specific genes suggest specialized roles in different tissues\n")
        f.write("- Consider biological context when prioritizing genes for validation\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nSummary report saved to: {report_file}")

def main():
    print("="*80)
    print("SINGLE PPI VS MULTIPLEX INTERACTOME COMPARISON")
    print("="*80)
    
    # Load results
    single_results = load_single_ppi_results()
    multiplex_results = load_multiplex_results()
    
    # Check if data loaded
    if single_results['module'] is None or multiplex_results['module'] is None:
        print("\nERROR: Could not load required result files")
        print("Make sure both analyses have completed successfully")
        return 1
    
    # Create output directory
    output_dir = 'output/comparisons'
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare disease modules
    overlap, single_only, multiplex_only = compare_disease_modules(
        single_results, multiplex_results, output_dir
    )
    
    # Compare gene rankings
    merged_rankings = compare_gene_rankings(
        single_results, multiplex_results, output_dir
    )
    
    # Identify cell-type-specific genes
    specific_genes = identify_celltype_specific_genes(
        multiplex_results, threshold=2.0, output_dir=output_dir
    )
    
    # Generate summary report
    generate_summary_report(
        single_results, multiplex_results, overlap, single_only,
        multiplex_only, specific_genes, output_dir
    )
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - module_overlap.csv / .png")
    print("  - merged_rankings.csv")
    print("  - score_correlation.png")
    print("  - celltype_specific_genes.csv")
    print("  - celltype_specificity.png")
    print("  - comparison_summary.txt")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())