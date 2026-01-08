#!/usr/bin/env python3
"""
SIMPLE RWR Visualization - No complex matplotlib issues
Just creates clean, simple plots that work
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def load_results(rwr_dir='rwr_results'):
    """Load RWR results"""
    
    print("Loading results...")
    
    disease_module_file = os.path.join(rwr_dir, 'multiplex_disease_module.csv')
    
    if not os.path.exists(disease_module_file):
        print(f"ERROR: {disease_module_file} not found!")
        return None
    
    disease_module = pd.read_csv(disease_module_file)
    
    # Load layer results
    layer_files = sorted([f for f in os.listdir(rwr_dir) if f.startswith('layer_') and f.endswith('_scores.csv')])
    layers = {}
    
    for layer_file in layer_files:
        layer_name = layer_file.replace('layer_', '').replace('_scores.csv', '')
        path = os.path.join(rwr_dir, layer_file)
        try:
            layers[layer_name] = pd.read_csv(path)
        except Exception as e:
            print(f"  Warning: Could not load {layer_file}: {e}")
    
    print(f"  Loaded disease module: {len(disease_module)} genes")
    print(f"  Loaded {len(layers)} layer files")
    
    return {'disease_module': disease_module, 'layers': layers}


def parse_layer_name(layer_name):
    """Parse layer name into tissue and cell type"""
    parts = layer_name.split('_')
    
    if parts[0] in ['DRG', 'TG']:
        tissue = parts[0]
        celltype = '_'.join(parts[1:])
        return tissue, celltype
    
    return layer_name, 'unknown'


def plot_1_tissue_overview(results, output_dir='plots'):
    """Simple tissue overview"""
    print("\nCreating tissue overview...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect tissue data
    tissue_data = {}
    for layer_name, layer_df in results['layers'].items():
        tissue, _ = parse_layer_name(layer_name)
        if tissue not in tissue_data:
            tissue_data[tissue] = {'scores': [], 'n_genes': 0}
        
        tissue_data[tissue]['scores'].extend(layer_df['Score'].tolist())
        tissue_data[tissue]['n_genes'] += len(layer_df)
    
    tissues = sorted(tissue_data.keys())
    avg_scores = [np.mean(tissue_data[t]['scores']) for t in tissues]
    max_scores = [np.max(tissue_data[t]['scores']) for t in tissues]
    
    # Create simple bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.bar(tissues, avg_scores, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Average Score')
    ax1.set_title('Average RWR Score by Tissue')
    ax1.set_ylim(0, max(avg_scores) * 1.2)
    
    ax2.bar(tissues, max_scores, color='coral', alpha=0.7)
    ax2.set_ylabel('Maximum Score')
    ax2.set_title('Maximum RWR Score by Tissue')
    ax2.set_ylim(0, max(max_scores) * 1.2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '01_tissue_overview.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: 01_tissue_overview.png")
    plt.close()


def plot_2_cell_types_by_tissue(results, output_dir='plots'):
    """Plot each tissue separately - FIX for size issue"""
    print("Creating cell type plots...")
    
    tissues = {}
    for layer_name in results['layers'].keys():
        tissue, celltype = parse_layer_name(layer_name)
        if tissue not in tissues:
            tissues[tissue] = []
        tissues[tissue].append((celltype, layer_name))
    
    # One figure per tissue
    for tissue in sorted(tissues.keys()):
        celltypes = sorted(tissues[tissue], key=lambda x: results['layers'][x[1]]['Score'].mean(), reverse=True)
        
        n_cells = len(celltypes)
        # CRITICAL: Limit height to prevent overflow
        height = min(12, 0.4 * n_cells + 1)
        
        fig, ax = plt.subplots(figsize=(10, height))
        
        cell_names = [ct[0] for ct in celltypes]
        scores = [results['layers'][ct[1]]['Score'].mean() for ct in celltypes]
        
        ax.barh(cell_names, scores, color='steelblue', alpha=0.7)
        ax.set_xlabel('Average RWR Score')
        ax.set_title(f'{tissue} - Cell Types by Score')
        
        # Add value labels
        for i, score in enumerate(scores):
            ax.text(score + max(scores)*0.01, i, f'{score:.5f}', va='center', fontsize=8)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'02_celltypes_{tissue}.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: 02_celltypes_{tissue}.png")
        plt.close()


def plot_3_top_genes_per_tissue(results, output_dir='plots', top_n=10):
    """Top genes per tissue"""
    print(f"Creating top {top_n} genes plots...")
    
    tissues = {}
    for layer_name, layer_df in results['layers'].items():
        tissue, _ = parse_layer_name(layer_name)
        if tissue not in tissues:
            tissues[tissue] = pd.concat([layer_df], ignore_index=True)
        else:
            tissues[tissue] = pd.concat([tissues[tissue], layer_df], ignore_index=True)
    
    for tissue in sorted(tissues.keys()):
        tissue_df = tissues[tissue]
        top_genes = tissue_df.nlargest(top_n, 'Score')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['gold' if s else 'steelblue' for s in top_genes['Is_Seed']]
        ax.barh(top_genes['Gene'], top_genes['Score'], color=colors, alpha=0.7)
        ax.set_xlabel('RWR Score')
        ax.set_title(f'{tissue} - Top {top_n} Genes')
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'03_top_genes_{tissue}.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: 03_top_genes_{tissue}.png")
        plt.close()


def plot_4_distributions(results, output_dir='plots'):
    """Score distributions"""
    print("Creating score distributions...")
    
    disease_module = results['disease_module']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Histogram of GMEAN scores
    axes[0, 0].hist(disease_module['GMEAN_Score'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('GMEAN Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of GMEAN Scores')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of N_Layers
    axes[0, 1].hist(disease_module['N_Layers'], bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('N_Layers (Cell Types)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Gene Consistency Across Cell Types')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter: Score vs N_Layers
    seed_genes = disease_module[disease_module['Is_Seed'] == True]
    discovered = disease_module[disease_module['Is_Seed'] == False]
    
    axes[1, 0].scatter(seed_genes['N_Layers'], seed_genes['GMEAN_Score'], 
                      color='gold', s=50, alpha=0.6, label='Seed genes')
    axes[1, 0].scatter(discovered['N_Layers'], discovered['GMEAN_Score'], 
                      color='steelblue', s=30, alpha=0.4, label='Discovered')
    axes[1, 0].set_xlabel('N_Layers')
    axes[1, 0].set_ylabel('GMEAN Score')
    axes[1, 0].set_title('Score vs Consistency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Pie chart: Consistency categories
    consistency_cat = pd.cut(disease_module['N_Layers'], 
                            bins=[0, 10, 20, 30, 40, 51],
                            labels=['1-10', '11-20', '21-30', '31-40', '41-51'])
    counts = consistency_cat.value_counts().sort_index()
    
    axes[1, 1].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Gene Distribution by Consistency')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '04_distributions.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: 04_distributions.png")
    plt.close()


def create_summary_tables(results, output_dir='plots'):
    """Create summary CSV tables"""
    print("Creating summary tables...")
    
    # Tissue summary
    tissue_summary = []
    tissues = {}
    
    for layer_name, layer_df in results['layers'].items():
        tissue, celltype = parse_layer_name(layer_name)
        
        if tissue not in tissues:
            tissues[tissue] = {
                'avg_score': [],
                'max_score': [],
                'n_genes': 0
            }
        
        tissues[tissue]['avg_score'].append(layer_df['Score'].mean())
        tissues[tissue]['max_score'].append(layer_df['Score'].max())
        tissues[tissue]['n_genes'] += len(layer_df)
    
    for tissue in sorted(tissues.keys()):
        tissue_summary.append({
            'Tissue': tissue,
            'Avg_Score': np.mean(tissues[tissue]['avg_score']),
            'Max_Score': np.mean(tissues[tissue]['max_score']),
            'N_Genes': tissues[tissue]['n_genes']
        })
    
    tissue_df = pd.DataFrame(tissue_summary)
    tissue_df.to_csv(os.path.join(output_dir, 'tissue_summary.csv'), index=False)
    print(f"  Saved: tissue_summary.csv")
    
    # Cell type summary
    celltype_summary = []
    for layer_name, layer_df in results['layers'].items():
        tissue, celltype = parse_layer_name(layer_name)
        celltype_summary.append({
            'Tissue': tissue,
            'CellType': celltype,
            'Avg_Score': layer_df['Score'].mean(),
            'Max_Score': layer_df['Score'].max(),
            'N_Genes': len(layer_df),
            'N_Seeds': (layer_df['Is_Seed'] == True).sum()
        })
    
    celltype_df = pd.DataFrame(celltype_summary).sort_values('Avg_Score', ascending=False)
    celltype_df.to_csv(os.path.join(output_dir, 'celltype_summary.csv'), index=False)
    print(f"  Saved: celltype_summary.csv")
    
    return tissue_df, celltype_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple RWR visualization')
    parser.add_argument('--rwr-dir', type=str, default='rwr_results', help='RWR results directory')
    parser.add_argument('--output-dir', type=str, default='plots', help='Output directory')
    parser.add_argument('--top-n', type=int, default=10, help='Top N genes to show')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SIMPLE RWR VISUALIZATION")
    print("="*80 + "\n")
    
    results = load_results(args.rwr_dir)
    if results is None:
        return 1
    
    # Create all plots
    plot_1_tissue_overview(results, args.output_dir)
    plot_2_cell_types_by_tissue(results, args.output_dir)
    plot_3_top_genes_per_tissue(results, args.output_dir, args.top_n)
    plot_4_distributions(results, args.output_dir)
    
    # Create tables
    tissue_df, celltype_df = create_summary_tables(results, args.output_dir)
    
    print("\n" + "="*80)
    print("TISSUE SUMMARY")
    print("="*80)
    print(tissue_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("TOP 10 CELL TYPES")
    print("="*80)
    celltype_df_display = pd.read_csv(os.path.join(args.output_dir, 'celltype_summary.csv'))
    print(celltype_df_display.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nPlots saved to: {args.output_dir}/")
    print("\nFiles created:")
    print("  01_tissue_overview.png")
    print("  02_celltypes_DRG.png")
    print("  02_celltypes_TG.png")
    print("  03_top_genes_DRG.png")
    print("  03_top_genes_TG.png")
    print("  04_distributions.png")
    print("  tissue_summary.csv")
    print("  celltype_summary.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())