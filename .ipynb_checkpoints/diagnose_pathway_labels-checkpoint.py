#!/usr/bin/env python3
"""
diagnose_pathway_labels.py
===========================
Run this standalone to figure out why pathway edges are falling back
to the generic "pathway" label instead of getting a named pathway.
"""
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

# ── Point these at your actual files ────────────────────────────────────────
REACTOME_CSV     = Path("./pathway_distance/reactome_interactions_with_pathways.csv")
UNIPROT_SYM_CSV  = Path("../og_gene_set/ppi_chloe/d_uniprot_to_symbol_20251120.csv")
PATHWAY_TSV      = Path("./multixrank_input_consolidated/multiplex/gene_multiplex/pathway_layer.tsv")   # one example layer is enough
N_SAMPLE         = 50                           # how many examples to print
# ────────────────────────────────────────────────────────────────────────────


def load_uniprot_symbol_map(csv_path: Path) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    uid_col = next(
        (c for c in df.columns if c in ("uniprotid", "uniprot_id", "uniprot")), None
    )
    sym_col = next(
        (c for c in df.columns if c in ("symbol", "gene_symbol", "genesymbol")), None
    )
    if not uid_col or not sym_col:
        raise ValueError(f"Cannot find id/symbol columns. Found: {list(df.columns)}")
    mapping = {}
    for _, row in df.iterrows():
        uid = str(row[uid_col]).strip().upper()
        sym = str(row[sym_col]).strip().upper()
        if uid and sym and uid != "NAN" and sym != "NAN":
            mapping[uid] = sym
    print(f"[uniprot_map] Loaded {len(mapping):,} entries")
    return mapping


def load_reactome_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"[reactome]   Shape: {df.shape}")
    print(f"[reactome]   Columns: {list(df.columns)}")
    return df


def load_pathway_tsv(tsv_path: Path) -> list[tuple[str, str]]:
    df = pd.read_csv(
        tsv_path, sep="\t", header=None,
        names=["g1", "g2"], dtype=str,
        on_bad_lines="skip",
    )
    edges = []
    for _, row in df.iterrows():
        a = str(row["g1"]).strip().upper()
        b = str(row["g2"]).strip().upper()
        if a and b and a != "NAN" and b != "NAN" and a != b:
            edges.append((a, b))
    print(f"[pathway_tsv] Loaded {len(edges):,} edges from {tsv_path.name}")
    return edges


# ============================================================================
print("\n" + "="*70)
print("STEP 1 — Load files")
print("="*70)
uniprot_map  = load_uniprot_symbol_map(UNIPROT_SYM_CSV)
reactome_df  = load_reactome_raw(REACTOME_CSV)
tsv_edges    = load_pathway_tsv(PATHWAY_TSV)

# ============================================================================
print("\n" + "="*70)
print("STEP 2 — Inspect the reactome CSV columns we rely on")
print("="*70)

for col in ("uniprot1", "uniprot2", "common_pathway", "pathway_sources"):
    if col in reactome_df.columns:
        n_null  = reactome_df[col].isna().sum()
        n_empty = (reactome_df[col].astype(str).str.strip() == "").sum()
        print(f"  {col:20s}: {len(reactome_df):,} rows | "
              f"null={n_null:,} | empty_str={n_empty:,}")
        if col == "common_pathway":
            n_has_pw = reactome_df[col].notna().sum()
            print(f"    → rows WITH a pathway name : {n_has_pw:,} "
                  f"({100*n_has_pw//len(reactome_df)}%)")
            # Sample some pathway names
            sample_pw = (
                reactome_df[col].dropna()
                .str.split("|").explode()
                .str.strip()
                .value_counts()
                .head(10)
            )
            print(f"    → Top 10 pathway names:\n{sample_pw.to_string()}")
    else:
        print(f"  ⚠  Column '{col}' NOT FOUND in reactome CSV")

# ============================================================================
print("\n" + "="*70)
print("STEP 3 — What format are uniprot1/uniprot2 in the reactome CSV?")
print("="*70)

u1_sample = reactome_df["uniprot1"].dropna().head(10).tolist()
u2_sample = reactome_df["uniprot2"].dropna().head(10).tolist()
print(f"  uniprot1 samples : {u1_sample}")
print(f"  uniprot2 samples : {u2_sample}")

# Check if they have prefixes like "uniprotkb:"
has_prefix = reactome_df["uniprot1"].astype(str).str.startswith("uniprotkb:").sum()
print(f"\n  Rows where uniprot1 starts with 'uniprotkb:' : {has_prefix:,}")

# Check case
upper_match = sum(
    1 for v in reactome_df["uniprot1"].dropna()
    if str(v) == str(v).upper()
)
print(f"  Rows where uniprot1 is already upper-case    : {upper_match:,} "
      f"/ {reactome_df['uniprot1'].notna().sum():,}")

# ============================================================================
print("\n" + "="*70)
print("STEP 4 — What format are the gene symbols in the pathway TSV?")
print("="*70)

tsv_syms = set()
for a, b in tsv_edges[:1000]:   # sample first 1000
    tsv_syms.add(a)
    tsv_syms.add(b)

print(f"  Unique symbols in TSV (sample of 1000 edges): {len(tsv_syms):,}")
print(f"  Sample TSV symbols: {sorted(tsv_syms)[:15]}")

# Are they already upper?
n_upper = sum(1 for s in tsv_syms if s == s.upper())
print(f"  Already upper-case: {n_upper:,} / {len(tsv_syms):,}")

# ============================================================================
print("\n" + "="*70)
print("STEP 5 — Can the uniprot_map convert reactome UniProt IDs to symbols?")
print("="*70)

# Strip any prefix from reactome UIDs first
reactome_u1 = (
    reactome_df["uniprot1"].dropna()
    .astype(str)
    .str.replace("uniprotkb:", "", regex=False)
    .str.strip()
    .str.upper()
)
reactome_u2 = (
    reactome_df["uniprot2"].dropna()
    .astype(str)
    .str.replace("uniprotkb:", "", regex=False)
    .str.strip()
    .str.upper()
)
all_reactome_uids = set(reactome_u1) | set(reactome_u2)
print(f"  Unique UniProt IDs in reactome CSV : {len(all_reactome_uids):,}")

mappable = all_reactome_uids & set(uniprot_map.keys())
unmappable = all_reactome_uids - set(uniprot_map.keys())
print(f"  Mappable via uniprot_map           : {len(mappable):,} "
      f"({100*len(mappable)//max(len(all_reactome_uids),1)}%)")
print(f"  NOT mappable                       : {len(unmappable):,}")
print(f"  Sample unmappable UIDs             : {sorted(unmappable)[:15]}")

# ============================================================================
print("\n" + "="*70)
print("STEP 6 — Build symbol-pair map and test against TSV edges")
print("="*70)

# Build the map exactly as the main script does
pw_map: dict[tuple[str,str], list[str]] = defaultdict(list)
skipped_no_sym = skipped_no_pw = built = 0

for _, row in reactome_df.iterrows():
    u1 = str(row["uniprot1"]).strip().replace("uniprotkb:", "").upper()
    u2 = str(row["uniprot2"]).strip().replace("uniprotkb:", "").upper()
    sym1 = uniprot_map.get(u1)
    sym2 = uniprot_map.get(u2)
    if not sym1 or not sym2:
        skipped_no_sym += 1
        continue
    pw_raw = row.get("common_pathway", "")
    if pd.isna(pw_raw) or str(pw_raw).strip() in ("", "nan", "NAN"):
        skipped_no_pw += 1
        continue
    pathways = [p.strip() for p in str(pw_raw).split("|") if p.strip()]
    key = (min(sym1, sym2), max(sym1, sym2))
    pw_map[key].extend(pathways)
    built += 1

print(f"  Reactome rows processed          : {len(reactome_df):,}")
print(f"  Rows skipped (no symbol)         : {skipped_no_sym:,}")
print(f"  Rows skipped (no pathway)        : {skipped_no_pw:,}")
print(f"  Symbol-pair keys built           : {len(pw_map):,}")

# Now test TSV edges against the map
hit = miss = 0
miss_examples = []
hit_examples  = []

for sym_a, sym_b in tsv_edges:
    key = (min(sym_a, sym_b), max(sym_a, sym_b))
    if pw_map.get(key):
        hit += 1
        if len(hit_examples) < N_SAMPLE:
            hit_examples.append((sym_a, sym_b, pw_map[key][0]))
    else:
        miss += 1
        if len(miss_examples) < N_SAMPLE:
            miss_examples.append((sym_a, sym_b))

total = hit + miss
print(f"\n  TSV edges matched to a pathway   : {hit:,} / {total:,} "
      f"({100*hit//max(total,1)}%)")
print(f"  TSV edges with NO match (→ 'pathway' fallback): {miss:,} "
      f"({100*miss//max(total,1)}%)")

print(f"\n  Sample HITS (sym_a, sym_b, best_label):")
for h in hit_examples:
    print(f"    {h[0]:12s} {h[1]:12s}  →  {h[2][:60]}")

print(f"\n  Sample MISSES (sym_a, sym_b) — these become 'pathway':")
for m in miss_examples:
    print(f"    {m[0]:12s} {m[1]:12s}")

# ============================================================================
print("\n" + "="*70)
print("STEP 7 — Why are the misses missing? Drill into a few")
print("="*70)

# For each missed pair, check whether either symbol appears
# in the reactome map at all (wrong partner?) vs not at all (not in reactome)
reactome_syms_all: set[str] = set()
for (s1, s2) in pw_map.keys():
    reactome_syms_all.add(s1)
    reactome_syms_all.add(s2)

n_neither = n_one = n_both_wrong_pair = 0
for sym_a, sym_b in tsv_edges:
    in_a = sym_a in reactome_syms_all
    in_b = sym_b in reactome_syms_all
    key  = (min(sym_a, sym_b), max(sym_a, sym_b))
    if pw_map.get(key):
        continue    # it's a hit, skip
    if not in_a and not in_b:
        n_neither += 1
    elif in_a and in_b:
        n_both_wrong_pair += 1   # both symbols known but THIS pair not in map
    else:
        n_one += 1

print(f"  Of the {miss:,} misses:")
print(f"    Neither symbol in reactome map  : {n_neither:,}  "
      f"(genes not in reactome at all)")
print(f"    Only one symbol in reactome map : {n_one:,}  "
      f"(one gene known, partner not)")
print(f"    Both symbols known, pair absent : {n_both_wrong_pair:,}  "
      f"(interaction not in reactome CSV)")

# ============================================================================
print("\n" + "="*70)
print("STEP 8 — Check if TSV uses mixed case or prefixes")
print("="*70)

raw_df = pd.read_csv(
    PATHWAY_TSV, sep="\t", header=None,
    names=["g1","g2"], dtype=str, on_bad_lines="skip"
)
print(f"  Raw TSV sample (first 10 rows):")
print(raw_df.head(10).to_string(index=False))

has_colon = raw_df["g1"].astype(str).str.contains(":").sum()
print(f"\n  Rows where g1 contains ':' (e.g. 'uniprotkb:') : {has_colon:,}")

mixed_case = sum(
    1 for v in raw_df["g1"].dropna()
    if str(v) != str(v).upper() and str(v) != str(v).lower()
)
print(f"  Rows where g1 is mixed-case                    : {mixed_case:,}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE — read the percentages above to find the culprit")
print("="*70)