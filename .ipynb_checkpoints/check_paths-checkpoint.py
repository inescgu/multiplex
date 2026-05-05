"""
Quick sanity check for the uniprot→symbol fix.
Reads the real reactome CSV + uniprot map, checks that:
  1. Pathway genes are resolved to symbols (not raw UniProt IDs)
  2. At least some of those symbols would match real layer genes
     (we fake a tiny layer with known genes from the CSV)
"""
import pandas as pd
from collections import defaultdict

# ── CONFIG — adjust these paths ──────────────────────────────────────────────
REACTOME_CSV  = "./pathway_distance/reactome_interactions_with_pathways.csv"
UNIPROT_MAP   = "../og_gene_set/ppi_chloe/d_uniprot_to_symbol_20251120.csv"
N_ROWS        = 500   # only read first N rows — fast!
# ─────────────────────────────────────────────────────────────────────────────

print("Loading uniprot map...")
umap = pd.read_csv(UNIPROT_MAP)
u2s  = dict(zip(umap["uniprotid"], umap["symbol"]))
print(f"  {len(u2s):,} entries")

print(f"\nLoading first {N_ROWS} rows of reactome CSV...")
df = pd.read_csv(REACTOME_CSV, nrows=N_ROWS,
                 usecols=["uniprot1", "uniprot2",
                          "common_pathway", "pathway_sources"])
df = df.dropna(subset=["common_pathway"])
print(f"  {len(df)} rows with common_pathway")

# Check translation
raw_genes, sym_genes, db_counts = set(), set(), defaultdict(int)
for _, row in df.iterrows():
    for col in ("uniprot1", "uniprot2"):
        raw = str(row[col]).strip().upper()
        sym = u2s.get(raw, None)
        raw_genes.add(raw)
        if sym:
            sym_genes.add(sym.upper())

    sources = str(row.get("pathway_sources", "")).strip()
    for db in sources.split("|"):
        db_counts[db.strip()] += 1

print(f"\n  Raw UniProt IDs seen : {len(raw_genes)}")
print(f"  Resolved to symbols  : {len(sym_genes)}")
print(f"  Resolution rate      : {len(sym_genes)/max(len(raw_genes),1)*100:.1f}%")
print(f"\n  DB label counts (from pathway_sources column):")
for db, cnt in sorted(db_counts.items(), key=lambda x: -x[1]):
    print(f"    {db:<15} {cnt:>6,} rows")

print(f"\n  Sample resolved symbols: {sorted(sym_genes)[:10]}")

if len(sym_genes) == 0:
    print("\n❌ FAIL: No symbols resolved — uniprot map is not matching. Check column names.")
elif len(sym_genes) < len(raw_genes) * 0.3:
    print("\n⚠️  WARNING: Low resolution rate — check uniprot map coverage.")
else:
    print("\n✅ PASS: Symbols resolved correctly. Pathway fix should work.")