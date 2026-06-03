#!/usr/bin/env python3
"""
Convert mouse LR pairs to human orthologs using MyGene.info API
"""

import pandas as pd
import requests
import time
import logging
import argparse
import sys
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ortholog_conversion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_human_ortholog(mouse_gene: str, 
                       session: requests.Session,
                       max_retries: int = 3,
                       delay: float = 0.1) -> Optional[str]:
    """
    Query MyGene.info to get human ortholog for a mouse gene symbol.
    
    Args:
        mouse_gene: Mouse gene symbol
        session: requests Session object
        max_retries: Number of retry attempts
        delay: Delay between requests in seconds
        
    Returns:
        Human ortholog gene symbol or None if not found
    """
    url = "https://mygene.info/v3/query"
    params = {
        "q": f"symbol:{mouse_gene}",
        "species": "mouse",
        "fields": "symbol,name,homologene",
        "size": 1
    }
    
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("hits"):
                logger.warning(f"No hits found for mouse gene: {mouse_gene}")
                return None
            
            hit = data["hits"][0]
            
            # Extract human ortholog from homologene
            if "homologene" in hit:
                homologene = hit["homologene"]
                genes = homologene.get("genes", [])
                
                # Human taxid is 9606
                for taxid, gene_id in genes:
                    if taxid == 9606:
                        # Query for human gene symbol using gene ID
                        human_symbol = get_human_gene_symbol(
                            gene_id, session, delay
                        )
                        return human_symbol
            
            logger.warning(f"No human ortholog found for: {mouse_gene}")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Attempt {attempt + 1}/{max_retries} failed for "
                f"{mouse_gene}: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1) * 2)  # exponential backoff
            else:
                return None


def get_human_gene_symbol(gene_id: int,
                          session: requests.Session,
                          delay: float = 0.1) -> Optional[str]:
    """
    Get human gene symbol from Entrez gene ID.
    
    Args:
        gene_id: Entrez gene ID
        session: requests Session object
        delay: Delay between requests
        
    Returns:
        Gene symbol string or None
    """
    url = f"https://mygene.info/v3/gene/{gene_id}"
    params = {"fields": "symbol,name"}
    
    try:
        time.sleep(delay)
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("symbol")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get symbol for gene ID {gene_id}: {e}")
        return None


def batch_convert_orthologs(genes: list,
                            session: requests.Session,
                            batch_size: int = 100,
                            delay: float = 0.1) -> dict:
    """
    Convert a list of mouse genes to human orthologs in batches.
    Uses POST endpoint for efficiency.
    
    Args:
        genes: List of mouse gene symbols
        session: requests Session object
        batch_size: Number of genes per batch
        delay: Delay between batches
        
    Returns:
        Dictionary mapping mouse gene -> human ortholog
    """
    unique_genes = list(set(genes))
    results = {}
    
    logger.info(f"Converting {len(unique_genes)} unique genes to human orthologs")
    
    # Process in batches
    for i in range(0, len(unique_genes), batch_size):
        batch = unique_genes[i:i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/"
            f"{(len(unique_genes) + batch_size - 1)//batch_size} "
            f"({len(batch)} genes)"
        )
        
        url = "https://mygene.info/v3/query"
        payload = {
            "q": batch,
            "scopes": "symbol",
            "species": "mouse",
            "fields": "symbol,homologene",
            "size": 1
        }
        
        try:
            time.sleep(delay)
            response = session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for item in data:
                mouse_gene = item.get("query")
                
                if item.get("notfound") or "homologene" not in item:
                    logger.warning(
                        f"No homologene data found for: {mouse_gene}"
                    )
                    results[mouse_gene] = None
                    continue
                
                homologene = item["homologene"]
                genes_list = homologene.get("genes", [])
                
                human_gene_id = None
                for taxid, gene_id in genes_list:
                    if taxid == 9606:
                        human_gene_id = gene_id
                        break
                
                if human_gene_id:
                    human_symbol = get_human_gene_symbol(
                        human_gene_id, session, delay
                    )
                    results[mouse_gene] = human_symbol
                    logger.info(f"  {mouse_gene} -> {human_symbol}")
                else:
                    logger.warning(
                        f"No human ortholog found for: {mouse_gene}"
                    )
                    results[mouse_gene] = None
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch request failed: {e}")
            # Fall back to individual queries for this batch
            for gene in batch:
                results[gene] = get_human_ortholog(gene, session, delay=delay)
    
    return results


def process_lr_pairs(input_file: str,
                     output_file: str,
                     unmapped_file: str,
                     delay: float = 0.1,
                     keep_unmapped: bool = False) -> pd.DataFrame:
    """
    Process LR pairs CSV file and convert mouse genes to human orthologs.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        unmapped_file: Path to file for unmapped genes
        delay: Delay between API requests
        keep_unmapped: Whether to keep rows with unmapped genes
        
    Returns:
        DataFrame with human orthologs
    """
    # Read input file
    logger.info(f"Reading input file: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception:
        df = pd.read_csv(input_file, sep='\t')
    
    logger.info(f"Loaded {len(df)} LR pairs")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Clean column names (remove BOM if present)
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()
    
    # Identify gene columns
    ligand_col = 'ligand_gname'
    receptor_col = 'receptor_gname'
    
    if ligand_col not in df.columns or receptor_col not in df.columns:
        raise ValueError(
            f"Expected columns '{ligand_col}' and '{receptor_col}'. "
            f"Found: {list(df.columns)}"
        )
    
    # Collect all unique genes
    all_genes = list(
        set(df[ligand_col].tolist() + df[receptor_col].tolist())
    )
    logger.info(f"Found {len(all_genes)} unique genes to convert")
    
    # Convert to human orthologs
    with requests.Session() as session:
        session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        ortholog_map = batch_convert_orthologs(
            all_genes, session, delay=delay
        )
    
    # Add ortholog columns
    df['human_ligand'] = df[ligand_col].map(ortholog_map)
    df['human_receptor'] = df[receptor_col].map(ortholog_map)
    
    # Summary statistics
    total = len(df)
    ligand_mapped = df['human_ligand'].notna().sum()
    receptor_mapped = df['human_receptor'].notna().sum()
    both_mapped = (
        df['human_ligand'].notna() & df['human_receptor'].notna()
    ).sum()
    
    logger.info("\n=== Conversion Summary ===")
    logger.info(f"Total LR pairs: {total}")
    logger.info(f"Ligands mapped: {ligand_mapped}/{total} "
                f"({100*ligand_mapped/total:.1f}%)")
    logger.info(f"Receptors mapped: {receptor_mapped}/{total} "
                f"({100*receptor_mapped/total:.1f}%)")
    logger.info(f"Both mapped: {both_mapped}/{total} "
                f"({100*both_mapped/total:.1f}%)")
    
    # Save unmapped genes
    unmapped_mask = (
        df['human_ligand'].isna() | df['human_receptor'].isna()
    )
    unmapped_df = df[unmapped_mask].copy()
    
    if len(unmapped_df) > 0:
        unmapped_df.to_csv(unmapped_file, index=False)
        logger.info(
            f"Saved {len(unmapped_df)} unmapped pairs to: {unmapped_file}"
        )
    
    # Filter or keep unmapped rows
    if not keep_unmapped:
        df_filtered = df[~unmapped_mask].copy()
        logger.info(
            f"Keeping {len(df_filtered)} fully mapped pairs"
        )
    else:
        df_filtered = df.copy()
    
    # Reorder columns for clarity
    original_cols = list(df_filtered.columns)
    human_cols = ['human_ligand', 'human_receptor']
    other_cols = [c for c in original_cols if c not in human_cols]
    df_filtered = df_filtered[
        [other_cols[0], 'human_ligand'] + 
        other_cols[1:3] + 
        ['human_receptor'] + 
        other_cols[3:]
    ]
    
    # Save output
    df_filtered.to_csv(output_file, index=False)
    logger.info(f"Saved converted pairs to: {output_file}")
    
    # Print ortholog mapping summary
    logger.info("\n=== Gene Mapping Table ===")
    for mouse_gene, human_gene in sorted(ortholog_map.items()):
        status = human_gene if human_gene else "NOT FOUND"
        logger.info(f"  {mouse_gene:20s} -> {status}")
    
    return df_filtered


def main():
    parser = argparse.ArgumentParser(
        description='Convert mouse LR pairs to human orthologs'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file with mouse LR pairs'
    )
    parser.add_argument(
        '--output', '-o',
        default='human_lr_pairs.csv',
        help='Output CSV file (default: human_lr_pairs.csv)'
    )
    parser.add_argument(
        '--unmapped',
        default='unmapped_lr_pairs.csv',
        help='Output file for unmapped pairs (default: unmapped_lr_pairs.csv)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.1,
        help='Delay between API requests in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--keep-unmapped',
        action='store_true',
        help='Keep rows where ortholog mapping failed'
    )
    
    args = parser.parse_args()
    
    result = process_lr_pairs(
        input_file=args.input,
        output_file=args.output,
        unmapped_file=args.unmapped,
        delay=args.delay,
        keep_unmapped=args.keep_unmapped
    )
    
    logger.info(f"\nDone! Output saved to {args.output}")
    return result


if __name__ == "__main__":
    main()