"""
Read-only diagnostic: pinpoints exactly why seeds aren't matching graph
nodes. Checks for (a) case-duplicate cell-type folders, (b) any seed
strings that DO match a node case-insensitively but not case-sensitively.
"""
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx

wdir = Path(sys.argv[1])

# 1. Check for case-duplicate multiplex folders
mul_root = wdir / "multiplex"
folders = [p.name for p in mul_root.iterdir() if p.is_dir()]
by_lower = defaultdict(list)
for f in folders:
    by_lower[f.lower()].append(f)
dupes = {k: v for k, v in by_lower.items() if len(v) > 1}
print(f"Total multiplex folders: {len(folders)}")
if dupes:
    print(f"\n*** FOUND {len(dupes)} case-duplicate folder name(s): ***")
    for k, v in dupes.items():
        print(f"  {v}")
else:
    print("No case-duplicate folders found.")

# 2. Load seeds + reference graph, check case-insensitive matches
with open(wdir / "gene_reference_graph.pkl", "rb") as f:
    ref = pickle.load(f)
node_set = set(ref.nodes)
node_lower_map = defaultdict(list)
for n in node_set:
    node_lower_map[n.lower()].append(n)

seeds_path = wdir / "seeds" / "seeds.txt"
seeds = [s.strip() for s in seeds_path.read_text().splitlines() if s.strip()]

exact_match = 0
case_insensitive_only = 0
no_match_at_all = 0
sample_case_mismatches = []
sample_no_match = []

for s in seeds:
    if s in node_set:
        exact_match += 1
    elif s.lower() in node_lower_map:
        case_insensitive_only += 1
        if len(sample_case_mismatches) < 10:
            sample_case_mismatches.append((s, node_lower_map[s.lower()]))
    else:
        no_match_at_all += 1
        if len(sample_no_match) < 10:
            sample_no_match.append(s)

print(f"\nSeed matching against {len(node_set):,} graph nodes:")
print(f"  Exact match           : {exact_match}/{len(seeds)}")
print(f"  Case-insensitive only : {case_insensitive_only}/{len(seeds)}")
print(f"  No match at all       : {no_match_at_all}/{len(seeds)}")

if sample_case_mismatches:
    print("\nSample case mismatches (seed -> actual graph node(s)):")
    for s, matches in sample_case_mismatches:
        print(f"  {s}  ->  {matches}")

if sample_no_match:
    print("\nSample seeds with no match at all (exact or case-insensitive):")
    for s in sample_no_match:
        print(f"  {s}")