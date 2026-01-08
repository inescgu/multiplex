#!/usr/bin/env python3
"""
Convert mouse gene symbols to human gene symbols
"""

import os
import pandas as pd
import json
import time
import mygene

def query_mouse_symbols_to_human(mouse_symbols, batch_size=100, delay=0.2):
    """Convert mouse gene symbols to human gene symbols"""
    
    orthologs = {}
    failed_symbols = []
    
    overall_start = time.time()
    symbols_list = list(mouse_symbols)
    total_symbols = len(symbols_list)
    processed = 0
    batch_num = 0
    
    print(f"\nConverting {total_symbols} mouse gene symbols to human symbols...")
    print(f"Using batch size: {batch_size} symbols per request")
    
    mg = mygene.MyGeneInfo()
    
    i = 0
    while i < len(symbols_list):
        batch_num += 1
        batch_start = time.time()
        batch = symbols_list[i:i+batch_size]
        
        try:
            # Query batch by SYMBOL
            gene_infos = mg.querymany(batch, scopes='symbol', species='mouse', 
                                      fields='homologene,pantherdb,symbol,entrezgene', 
                                      returnall=True)
            
            if 'out' in gene_infos:
                for hit in gene_infos['out']:
                    # Skip if not found
                    if hit.get('notfound'):
                        failed_symbols.append(hit.get('query', ''))
                        continue
                    
                    mouse_symbol = str(hit.get('query', '')).upper()
                    human_orthologs = []
                    
                    # Method 1: HomoloGene - extract human taxid (9606)
                    if 'homologene' in hit and 'genes' in hit['homologene']:
                        for gene_pair in hit['homologene']['genes']:
                            if len(gene_pair) >= 2:
                                taxid, human_entrez = gene_pair[0], gene_pair[1]
                                if taxid == 9606:  # Human
                                    try:
                                        human_info = mg.query(f"entrezgene:{human_entrez}", 
                                                             species='human', 
                                                             fields='symbol')
                                        if human_info and 'hits' in human_info and len(human_info['hits']) > 0:
                                            symbol = human_info['hits'][0].get('symbol')
                                            if symbol:
                                                human_orthologs.append({
                                                    'source': 'HomoloGene',
                                                    'symbol': symbol.upper(),
                                                    'gene_id': human_entrez
                                                })
                                    except:
                                        pass
                    
                    # Method 2: PantherDB - extract HGNC info
                    if 'pantherdb' in hit and 'ortholog' in hit['pantherdb']:
                        for ortho in hit['pantherdb']['ortholog']:
                            if ortho.get('taxid') == 9606 and 'HGNC' in ortho:
                                human_orthologs.append({
                                    'source': 'PantherDB',
                                    'symbol': ortho.get('HGNC', '').upper(),
                                })
                    
                    if human_orthologs:
                        # Prefer PantherDB over HomoloGene
                        human_gene = None
                        for ortho in human_orthologs:
                            if ortho.get('source') == 'PantherDB':
                                human_gene = ortho.get('symbol')
                                break
                        if not human_gene:
                            human_gene = human_orthologs[0].get('symbol')
                        
                        if human_gene:
                            orthologs[mouse_symbol] = human_gene
                    else:
                        failed_symbols.append(mouse_symbol)
            
            processed += len(batch)
            batch_time = time.time() - batch_start
            elapsed = time.time() - overall_start
            print(f"  Batch {batch_num}: {processed}/{total_symbols} symbols | Time: {elapsed:.1f}s")
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"  ⚠️  Error on batch {batch_num}: {str(e)}")
            failed_symbols.extend(batch)
            processed += len(batch)
        
        i += batch_size
    
    return orthologs, failed_symbols

def convert_expression_file(expr_file, orthologs, output_file):
    """Convert expression file from mouse symbols to human symbols"""
    print(f"\nLoading {expr_file}...")
    expr_df = pd.read_csv(expr_file, index_col=0)
    print(f"  Loaded {expr_df.shape[0]} genes × {expr_df.shape[1]} columns")
    
    # Convert index to uppercase for matching
    expr_df.index = expr_df.index.astype(str).str.upper()
    expr_df['human_symbol'] = expr_df.index.map(lambda x: orthologs.get(x, None))
    
    # Keep only converted genes
    expr_converted = expr_df[expr_df['human_symbol'].notna()].copy()
    expr_converted.index = expr_converted['human_symbol']
    expr_converted = expr_converted.drop('human_symbol', axis=1)
    
    # Handle duplicates (multiple mouse genes → same human gene)
    if not expr_converted.index.is_unique:
        n_before = len(expr_converted)
        expr_converted = expr_converted.groupby(level=0).mean()
        n_after = len(expr_converted)
        print(f"  Merged {n_before - n_after} duplicate human genes")
    
    # Save
    expr_converted.to_csv(output_file)
    
    n_converted = len(expr_converted)
    n_original = expr_df.shape[0]
    conversion_rate = n_converted / n_original * 100
    
    print(f"  Original genes: {n_original}")
    print(f"  Converted: {n_converted} ({conversion_rate:.1f}%)")
    print(f"  ✓ Saved to {output_file}")
    
    return n_original, n_converted

def main():
    expr_dir = '../R/processed'
    output_dir = 'converted_expression'
    tissues = ['drgn', 'drgnn', 'tgn', 'tgnn']
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("MOUSE GENE SYMBOLS → HUMAN GENE SYMBOLS CONVERSION")
    print("="*80)
    
    # Step 1: Collect all unique mouse symbols
    print("\nStep 1: Collecting all unique mouse gene symbols...")
    all_mouse_symbols = set()
    
    for tissue in tissues:
        expr_file = os.path.join(expr_dir, f'{tissue}_celltype_avg_expression.csv')
        if not os.path.exists(expr_file):
            print(f"  ⚠️  File not found: {expr_file}")
            continue
        
        expr_df = pd.read_csv(expr_file, index_col=0)
        all_mouse_symbols.update(set(expr_df.index.astype(str).str.upper()))
    
    print(f"  Total unique mouse symbols: {len(all_mouse_symbols)}")
    print(f"  Examples: {list(all_mouse_symbols)[:5]}")
    
    # Step 2: Query orthologs
    print("\nStep 2: Querying mygene.info for human orthologs...")
    orthologs, failed_symbols = query_mouse_symbols_to_human(list(all_mouse_symbols))
    
    print(f"\n✓ Found orthologs for {len(orthologs)} symbols")
    print(f"✗ Failed for {len(failed_symbols)} symbols")
    
    if len(orthologs) > 0:
        print(f"\nExample conversions:")
        for i, (mouse, human) in enumerate(list(orthologs.items())[:5]):
            print(f"  {mouse} → {human}")
    
    # Save orthologs mapping
    orthologs_file = os.path.join(output_dir, 'mouse_to_human_orthologs.json')
    with open(orthologs_file, 'w') as f:
        json.dump(orthologs, f)
    print(f"\n✓ Saved mapping to {orthologs_file}")
    
    # Step 3: Convert all expression files
    print("\nStep 3: Converting expression files...")
    all_stats = []
    all_expr_dfs = []
    
    for tissue in tissues:
        expr_file = os.path.join(expr_dir, f'{tissue}_celltype_avg_expression.csv')
        if not os.path.exists(expr_file):
            continue
        
        output_file = os.path.join(output_dir, f'{tissue}_celltype_avg_expression_human.csv')
        n_orig, n_conv = convert_expression_file(expr_file, orthologs, output_file)
        
        all_stats.append({
            'tissue': tissue,
            'original_genes': n_orig,
            'converted_genes': n_conv,
            'conversion_rate': n_conv / n_orig * 100
        })
        print(n_conv.head())
        # Load for combining
        expr_converted = pd.read_csv(output_file, index_col=0)
        all_expr_dfs.append(expr_converted)
    
    # Step 4: Combine and deduplicate
    print("\nStep 4: Combining and deduplicating across tissues...")
    if all_expr_dfs:
        expr_combined = pd.concat(all_expr_dfs, axis=1)
        n_before = len(expr_combined)
        expr_combined = expr_combined.groupby(level=0).mean()
        n_after = len(expr_combined)
        
        print(f"  Before deduplication: {n_before} genes")
        print(f"  After deduplication: {n_after} genes")
        
        combined_output = os.path.join(output_dir, 'all_tissues_combined_human.csv')
        expr_combined.to_csv(combined_output)
        print(f"  ✓ Saved to {combined_output}")
    
    # Step 5: Report
    print("\nStep 5: Summary Report")
    print("="*80)
    stats_df = pd.DataFrame(all_stats)
    print(stats_df.to_string(index=False))
    
    total_orig = stats_df['original_genes'].sum()
    total_conv = stats_df['converted_genes'].sum()
    print(f"\nOverall: {total_conv}/{total_orig} genes converted ({total_conv/total_orig*100:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    main()