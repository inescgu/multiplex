#!/usr/bin/env python3
"""
Convert UniProt IDs to gene symbols in a CSV file
Fast command-line script to avoid Jupyter slowness

Usage:
    python uniprot_to_symbols.py kegg_signaling_fast_pathways_with_names.csv
    
Or with custom options:
    python uniprot_to_symbols.py kegg_signaling_fast_pathways_with_names.csv \
        --uniprot-column gene_name_uniprot \
        --output my_output.csv \
        --symbol-column gene_symbol \
        --rate-limit 0.5
"""

import requests
import time
from typing import Dict, List
import pandas as pd
import argparse
import sys


def convert_uniprot_to_gene_names(uniprot_ids: List[str],
                                  rate_limit_delay: float = 0.5,
                                  verbose: bool = True) -> Dict[str, str]:
    """
    Convert UniProt IDs to gene names using UniProt REST API
    
    Args:
        uniprot_ids: List of UniProt IDs (e.g., ['P38398', 'P04637'])
        rate_limit_delay: Delay between API calls in seconds
        verbose: Print progress information
    
    Returns:
        Dictionary mapping {uniprot_id: gene_name}
    """
    
    if not uniprot_ids:
        if verbose:
            print("No UniProt IDs provided")
        return {}
    
    # Remove duplicates
    unique_ids = list(dict.fromkeys(uniprot_ids))
    
    if verbose:
        print(f"Fetching gene names for {len(unique_ids)} UniProt IDs")
        print(f"(Using UniProt REST API)\n")
    
    gene_mapping = {}
    failed_ids = []
    processed = 0
    last_request_time = 0
    
    # Process one by one
    for idx, uniprot_id in enumerate(unique_ids):
        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < rate_limit_delay:
            time.sleep(rate_limit_delay - elapsed)
        
        last_request_time = time.time()
        
        try:
            # Query UniProt API
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                gene_name = None
                
                # Extract gene symbol from genes array
                if 'genes' in data and len(data['genes']) > 0:
                    for gene_entry in data['genes']:
                        if 'geneName' in gene_entry:
                            gene_name_obj = gene_entry['geneName']
                            if isinstance(gene_name_obj, dict) and 'value' in gene_name_obj:
                                gene_name = gene_name_obj['value']
                                break
                
                if gene_name:
                    gene_mapping[uniprot_id] = gene_name
                    processed += 1
                else:
                    failed_ids.append(uniprot_id)
            else:
                failed_ids.append(uniprot_id)
            
            # Progress update
            if verbose and (idx + 1) % 100 == 0 or (idx + 1) == len(unique_ids):
                pct = 100 * (idx + 1) // len(unique_ids)
                print(f"  {idx + 1}/{len(unique_ids)} ({pct}%) - Got {len(gene_mapping)} gene names")
        
        except requests.exceptions.Timeout:
            if verbose and idx == 0:
                print(f"  Note: Some requests may timeout (network dependent)")
            failed_ids.append(uniprot_id)
        
        except requests.exceptions.RequestException as e:
            if verbose and idx < 5:
                print(f"  Error fetching {uniprot_id}: {str(e)}")
            failed_ids.append(uniprot_id)
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"UniProt to Gene Name Conversion Results")
        print(f"{'='*60}")
        print(f"Total input IDs: {len(unique_ids)}")
        print(f"Successful conversions: {len(gene_mapping)}")
        print(f"Failed conversions: {len(failed_ids)}")
        if len(unique_ids) > 0:
            print(f"Success rate: {100*len(gene_mapping)//len(unique_ids)}%")
    
    return gene_mapping


def convert_uniprot_to_symbols_in_csv(csv_file: str,
                                     uniprot_column: str = 'gene_name_uniprot',
                                     output_file: str = None,
                                     symbol_column_name: str = 'gene_symbol',
                                     rate_limit_delay: float = 0.5,
                                     verbose: bool = True) -> pd.DataFrame:
    """
    Load a CSV with UniProt IDs and convert them to gene symbols
    
    Args:
        csv_file: Path to input CSV file
        uniprot_column: Name of column containing UniProt IDs
        output_file: Path to save output CSV
        symbol_column_name: Name for the new gene symbol column
        rate_limit_delay: Delay between API calls in seconds
        verbose: Print progress information
    
    Returns:
        DataFrame with gene symbols added
    """
    
    if verbose:
        print("="*60)
        print("UniProt ID → Gene Symbol Conversion")
        print("="*60)
        print()
        print(f"Loading {csv_file}...")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    if uniprot_column not in df.columns:
        raise ValueError(f"Column '{uniprot_column}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # Get unique UniProt IDs
    uniprot_ids = df[uniprot_column].unique().tolist()
    
    if verbose:
        print(f"Found {len(df)} rows with {len(uniprot_ids)} unique UniProt IDs\n")
    
    # Convert to gene names
    if verbose:
        print(f"Converting UniProt IDs to gene symbols...")
        start_time = time.time()
    
    gene_names = convert_uniprot_to_gene_names(
        uniprot_ids,
        rate_limit_delay=rate_limit_delay,
        verbose=verbose
    )
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\nConversion time: {elapsed:.2f} seconds")
    
    # Apply mapping
    df[symbol_column_name] = df[uniprot_column].map(gene_names)
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("Results Summary")
        print(f"{'='*60}")
        print(f"Total rows: {len(df)}")
        print(f"Genes with symbols: {df[symbol_column_name].notna().sum()}")
        print(f"Genes without symbols: {df[symbol_column_name].isna().sum()}")
        print(f"\nExample rows:")
        example_cols = [uniprot_column, symbol_column_name]
        if 'pathway_name' in df.columns:
            example_cols.append('pathway_name')
        print(df[example_cols].drop_duplicates(uniprot_column).head(10).to_string(index=False))
    
    # Save if requested
    if output_file:
        df.to_csv(output_file, index=False)
        if verbose:
            print(f"\nSaved to: {output_file}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert UniProt IDs to gene symbols in a CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - uses default column names
  python uniprot_to_symbols.py kegg_signaling_fast_pathways_with_names.csv
  
  # Specify output file
  python uniprot_to_symbols.py kegg_signaling_fast_pathways_with_names.csv \\
    --output my_output.csv
  
  # Custom column names
  python uniprot_to_symbols.py my_file.csv \\
    --uniprot-column my_uniprot_col \\
    --symbol-column my_symbol_col
  
  # Slower rate limit (if getting API errors)
  python uniprot_to_symbols.py kegg_signaling_fast_pathways_with_names.csv \\
    --rate-limit 1.0
        """
    )
    parser.add_argument('input_csv', type=str,
                       help='Input CSV file with UniProt IDs')
    parser.add_argument('--uniprot-column', type=str, default='gene_name_uniprot',
                       help='Column name containing UniProt IDs (default: gene_name_uniprot)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output CSV file (default: adds _symbols suffix to input)')
    parser.add_argument('--symbol-column', type=str, default='gene_symbol',
                       help='Name for new gene symbol column (default: gene_symbol)')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                       help='Delay between API calls in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        input_stem = args.input_csv.rsplit('.', 1)[0]
        args.output = f"{input_stem}_symbols.csv"
    
    print("="*80)
    print("UniProt to Gene Symbol Converter")
    print("="*80)
    print()
    
    try:
        # Run conversion
        df = convert_uniprot_to_symbols_in_csv(
            csv_file=args.input_csv,
            uniprot_column=args.uniprot_column,
            output_file=args.output,
            symbol_column_name=args.symbol_column,
            rate_limit_delay=args.rate_limit,
            verbose=True
        )
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"\nOutput file: {args.output}")
        sys.exit(0)
        
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {args.input_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
