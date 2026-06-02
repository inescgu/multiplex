#!/usr/bin/env python3
"""
diagnose_multixrank.py
Quick diagnostic to inspect raw MultiXrank output structure.
"""
import json
import pickle
from pathlib import Path
import pandas as pd
import multixrank

wdir = Path("multixrank_input_consolidated")
config_path = wdir / "config.yml"

print("=" * 60)
print("MultiXrank diagnostic")
print("=" * 60)

# Check seeds are in layers
seeds = [s.strip().upper() for s in
         (wdir / "seeds" / "seeds.txt").read_text().splitlines()
         if s.strip()]
print(f"\nSeeds ({len(seeds)}): {seeds[:10]}")

# Check seed presence in a few layers
print("\nChecking seed coverage in layers:")
layer_dir = wdir / "multiplex" / "gene_multiplex"
seed_set = set(seeds)
for tsv in sorted(layer_dir.glob("*.tsv"))[:5]:
    df = pd.read_csv(tsv, sep="\t", header=None,
                     names=["g1","g2"], dtype=str)
    df["g1"] = df["g1"].str.strip().str.upper()
    df["g2"] = df["g2"].str.strip().str.upper()
    genes = set(df["g1"]) | set(df["g2"])
    found = seed_set & genes
    print(f"  {tsv.name}: {len(found)}/{len(seeds)} seeds present "
          f"| example genes: {list(genes)[:3]}")

# Check ppi_layer specifically (global backbone)
ppi = layer_dir / "ppi_layer.tsv"
if ppi.exists():
    df = pd.read_csv(ppi, sep="\t", header=None,
                     names=["g1","g2"], dtype=str)
    df["g1"] = df["g1"].str.strip().str.upper()
    df["g2"] = df["g2"].str.strip().str.upper()
    genes = set(df["g1"]) | set(df["g2"])
    found = seed_set & genes
    print(f"\n  ppi_layer.tsv: {len(found)}/{len(seeds)} seeds | "
          f"{len(df):,} edges | {len(genes):,} unique genes")
    print(f"  Seeds found: {found}")
    print(f"  Seeds missing: {seed_set - found}")

# Run MultiXrank on tiny test
print("\n" + "=" * 60)
print("Running MultiXrank...")
mxr = multixrank.Multixrank(config=str(config_path), wdir=str(wdir))
result = mxr.random_walk_rank()

print(f"\nResult type: {type(result)}")

if isinstance(result, list):
    print(f"Result is LIST of length {len(result)}")
    for i, item in enumerate(result):
        print(f"\n  Item [{i}]: type={type(item)}")
        if isinstance(item, pd.DataFrame):
            print(f"    shape   : {item.shape}")
            print(f"    columns : {list(item.columns)}")
            print(f"    dtypes  : {dict(item.dtypes)}")
            print(f"    head(3) :\n{item.head(3).to_string()}")
            # Check for non-zero
            num_cols = item.select_dtypes(include='number').columns
            if len(num_cols) > 0:
                nonzero = (item[num_cols] > 0).any(axis=1).sum()
                print(f"    non-zero rows: {nonzero:,}")
                print(f"    score stats:\n"
                      f"{item[num_cols].describe().to_string()}")

elif isinstance(result, pd.DataFrame):
    print(f"Result is DataFrame")
    print(f"  shape   : {result.shape}")
    print(f"  columns : {list(result.columns)}")
    print(f"  head(5) :\n{result.head(5).to_string()}")
    num_cols = result.select_dtypes(include='number').columns
    if len(num_cols) > 0:
        nonzero = (result[num_cols] > 0).any(axis=1).sum()
        print(f"  non-zero rows: {nonzero:,}")
        print(f"  stats:\n{result[num_cols].describe().to_string()}")

elif isinstance(result, dict):
    print(f"Result is DICT with keys: {list(result.keys())}")
    for k, v in result.items():
        print(f"\n  key='{k}': type={type(v)}")
        if isinstance(v, dict):
            print(f"    sub-keys: {list(v.keys())}")
            for k2, v2 in v.items():
                print(f"    sub-key='{k2}': type={type(v2)}")
                if isinstance(v2, pd.DataFrame):
                    print(f"      shape={v2.shape}, cols={list(v2.columns)}")
                    print(f"      head(3):\n{v2.head(3).to_string()}")
        elif isinstance(v, pd.DataFrame):
            print(f"    shape={v.shape}, cols={list(v.columns)}")
            print(f"    head(3):\n{v.head(3).to_string()}")

# Also check what write_ranking produces
print("\n" + "=" * 60)
print("Checking write_ranking output...")
raw_out = Path("diag_raw_output")
raw_out.mkdir(exist_ok=True)
mxr.write_ranking(result, path=str(raw_out))

for f in sorted(raw_out.rglob("*")):
    if f.is_file():
        print(f"\n  File: {f}")
        try:
            df = pd.read_csv(f, sep="\t")
            print(f"    shape: {df.shape}")
            print(f"    cols : {list(df.columns)}")
            print(f"    head :\n{df.head(5).to_string()}")
            num = df.select_dtypes(include='number')
            if not num.empty:
                nonzero = (num > 0).any(axis=1).sum()
                print(f"    non-zero rows: {nonzero:,}")
        except Exception as e:
            print(f"    Error reading: {e}")

print("\nDone.")