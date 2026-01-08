#!/usr/bin/env python3
"""
Debug script to test mygene.info queries
"""

import pandas as pd
import mygene

# Load one expression file
expr_file = '../R/processed/drgn_celltype_avg_expression.csv'
expr_df = pd.read_csv(expr_file, index_col=0)

print("Expression file loaded")
print(f"Index type: {expr_df.index.dtype}")
print(f"First 10 indices: {list(expr_df.index[:10])}")
print(f"Index as strings: {[str(x) for x in expr_df.index[:10]]}")

# Get first 10 unique IDs
test_ids = [str(x) for x in expr_df.index[:10]]
print(f"\nTest IDs: {test_ids}")

# Try querying mygene
mg = mygene.MyGeneInfo()

print("\n" + "="*80)
print("Testing mygene.info query with scopes='entrezgene'")
print("="*80)

try:
    result = mg.querymany(test_ids, scopes='entrezgene', species='mouse', 
                         fields='homologene,pantherdb,symbol,entrezgene', 
                         returnall=True)
    
    print(f"\nQuery result keys: {result.keys()}")
    print(f"Number of results in 'out': {len(result.get('out', []))}")
    
    if 'out' in result:
        print("\nFirst 3 results:")
        for i, hit in enumerate(result['out'][:3]):
            print(f"\n  Result {i}:")
            print(f"    Query: {hit.get('query')}")
            print(f"    NotFound: {hit.get('notfound', False)}")
            print(f"    Symbol: {hit.get('symbol')}")
            print(f"    Has homologene: {'homologene' in hit}")
            print(f"    Has pantherdb: {'pantherdb' in hit}")
            
            if 'homologene' in hit:
                print(f"    HomoloGene genes: {hit['homologene'].get('genes', [])[:3]}")
            
            if 'pantherdb' in hit:
                print(f"    PantherDB orthologs: {hit['pantherdb'].get('ortholog', [])[:1]}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Also try with symbol scope
print("\n" + "="*80)
print("Testing mygene.info query with scopes='symbol'")
print("="*80)

# Get some actual symbols from the data
test_symbols = ['Atf3', 'Calca', 'Trpv1']
print(f"Test symbols: {test_symbols}")

try:
    result = mg.querymany(test_symbols, scopes='symbol', species='mouse', 
                         fields='homologene,pantherdb,symbol,entrezgene', 
                         returnall=True)
    
    print(f"\nQuery result keys: {result.keys()}")
    print(f"Number of results in 'out': {len(result.get('out', []))}")
    
    if 'out' in result:
        print("\nFirst 3 results:")
        for i, hit in enumerate(result['out'][:3]):
            print(f"\n  Result {i}:")
            print(f"    Query: {hit.get('query')}")
            print(f"    NotFound: {hit.get('notfound', False)}")
            print(f"    Symbol: {hit.get('symbol')}")
            print(f"    Has homologene: {'homologene' in hit}")
            print(f"    Has pantherdb: {'pantherdb' in hit}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
