#!/usr/bin/env python3
"""
Step 3 (replaces old step4's config-writing, step6's seed filtering, and
step7's reference-graph rebuild — all three collapse into one pass now
that everything needed is already sitting in celltype_gene_sets.json and
bipartite_manifest.json): write config.yml, filter seeds, save reference
graphs.

config.yml structure produced
------------------------------
multiplex:
  <cell_type_1>:
    layers:
      - multiplex/<cell_type_1>/ppi_layer.tsv
  <cell_type_2>:
    layers:
      - multiplex/<cell_type_2>/ppi_layer.tsv
  ...
bipartite:
  bipartite/<ct_a>__<ct_b>.tsv:
    source: <ct_a>
    target: <ct_b>
  ...
seed: seeds/seeds.txt
self_loops: 0
r: 0.7

Deliberately NOT written: delta, tau, eta, lamb (the actual YAML key for
"lambda" — confirmed from MultiXrank's own ConfigParser.py, the docs' use
of the Greek letter is just notation). Leaving these out is not a
simplification-for-convenience; it's the numerically correct default:
MultiXrank computes homogeneous delta/tau/eta/lamb automatically to keep
exploration balanced across every layer, multiplex, and bipartite jump,
which is what "connected only via explicit ligand-receptor edges, weighted
by a new lambda parameter" means in practice unless you have a specific
reason to bias the walk toward/away from particular cell types. If you
DO want to bias particular multiplex-to-multiplex jumps later, pass
--lamb-json with a JSON nested dict {ct_i: {ct_j: value, ...}, ...} and
this script will convert it (row-normalised, diagonal filled in per
MultiXrank's own check_lamb constraint) into the `lamb` matrix for you.
"""
import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd
import yaml


def discover_multiplexes(wdir: Path) -> list[str]:
    mul_root = wdir / "multiplex"
    cell_types = sorted(p.name for p in mul_root.iterdir() if p.is_dir())
    if not cell_types:
        raise SystemExit(f"ERROR: no cell-type subfolders found under {mul_root} — run step 1 first.")
    return cell_types


def build_lamb_matrix(lamb_json_path: str, cell_types: list[str]) -> list[list[float]]:
    """Converts a user-supplied {ct_i: {ct_j: value}} dict into MultiXrank's
    required lamb matrix, respecting its check_lamb constraint: column k's
    sum must equal 1, and lamb[k,k] must equal 1 - sum(off-diagonal column
    entries) (i.e. diagonal = probability of staying in that multiplex)."""
    with open(lamb_json_path) as f:
        user_lamb = json.load(f)
    n = len(cell_types)
    idx = {ct: i for i, ct in enumerate(cell_types)}
    M = [[0.0] * n for _ in range(n)]
    for ct_i, row in user_lamb.items():
        for ct_j, val in row.items():
            M[idx[ct_i]][idx[ct_j]] = float(val)
    # fill diagonal so each column sums to 1 (stay probability = 1 - leave probability)
    for j in range(n):
        col_off_diag = sum(M[i][j] for i in range(n) if i != j)
        M[j][j] = round(1.0 - col_off_diag, 6)
    return M


def main():
    ap = argparse.ArgumentParser(description="Build config.yml + filtered seeds + reference graphs")
    ap.add_argument("--wdir", required=True)
    ap.add_argument("--seed-csv", required=True, help="CSV with a 'gene' column")
    ap.add_argument("--r", type=float, default=0.7)
    ap.add_argument("--self-loops", type=int, default=0, choices=[0, 1])
    ap.add_argument("--lamb-json", default=None,
                     help="Optional {cell_type: {cell_type: value}} JSON to override MultiXrank's auto-computed lambda")
    args = ap.parse_args()

    wdir = Path(args.wdir)

    cell_types = discover_multiplexes(wdir)
    print(f"Found {len(cell_types)} cell-type multiplex(es): {cell_types}\n")

    manifest_path = wdir / "bipartite_manifest.json"
    manifest = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        print("⚠ No bipartite_manifest.json found (did you run step 2?) — "
              "writing a config with NO bipartite connections. Cell types will be fully isolated.\n")

    # ------------------------------------------------------------------
    # config.yml
    # ------------------------------------------------------------------
    config = {
        "multiplex": {
            ct: {"layers": [f"multiplex/{ct}/ppi_layer.tsv"]}
            for ct in cell_types
        },
        "bipartite": {
            m["file"]: {"source": m["source"], "target": m["target"]}
            for m in manifest
        },
        "seed": "seeds/seeds.txt",
        "self_loops": args.self_loops,
        "r": args.r,
    }

    if args.lamb_json:
        config["lamb"] = build_lamb_matrix(args.lamb_json, cell_types)
        print(f"Using user-supplied lamb matrix from {args.lamb_json}\n")
    else:
        print("No --lamb-json given: MultiXrank will auto-compute a homogeneous lambda matrix.\n")

    with open(wdir / "config.yml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved config.yml -> {wdir / 'config.yml'}")
    print(f"  Multiplexes : {len(cell_types)}")
    print(f"  Bipartites  : {len(manifest)}")
    print(f"  r           : {args.r}")
    print(f"  self_loops  : {args.self_loops}\n")

    # ------------------------------------------------------------------
    # Seeds — filter against the union gene universe
    # ------------------------------------------------------------------
    with open(wdir / "celltype_gene_sets.json") as f:
        celltype_gene_sets = json.load(f)
    gene_universe = set()
    for genes in celltype_gene_sets.values():
        gene_universe.update(genes)

    seeds_dir = wdir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.seed_csv)
    if "gene" not in df.columns:
        raise ValueError(f"Expected 'gene' column in {args.seed_csv}, found {list(df.columns)}")
    seeds = [str(g).strip().upper() for g in df["gene"].dropna()]
    seeds_in = sorted(set(s for s in seeds if s in gene_universe))
    seeds_out = sorted(set(s for s in seeds if s not in gene_universe))

    with open(seeds_dir / "seeds_original.txt", "w") as f:
        for s in seeds:
            f.write(s + "\n")
    with open(seeds_dir / "seeds.txt", "w") as f:
        for s in seeds_in:
            f.write(s + "\n")

    print(f"Seeds: {len(seeds)} provided, {len(seeds_in)} present in network "
          f"({100*len(seeds_in)/max(len(seeds),1):.1f}%), {len(seeds_out)} missing")
    if seeds_out:
        print(f"  Missing (first 10): {seeds_out[:10]}")
    print(f"  Saved -> {seeds_dir / 'seeds.txt'}\n")

    if not seeds_in:
        raise SystemExit("ERROR: none of your seed genes are present in any cell-type layer. "
                          "Check gene-symbol formatting before running MultiXrank.")

    # ------------------------------------------------------------------
    # Reference graphs (optional convenience output for downstream analysis
    # — NOT required by MultiXrank itself)
    # ------------------------------------------------------------------
    print("Building combined reference graphs for downstream analysis ...")
    ref = nx.Graph()
    ppi_only = nx.Graph()

    for ct in cell_types:
        layer_path = wdir / "multiplex" / ct / "ppi_layer.tsv"
        edges_df = pd.read_csv(layer_path, sep="\t", header=None, names=["g1", "g2"], dtype=str)
        for a, b in zip(edges_df["g1"], edges_df["g2"]):
            if a == b:
                continue
            if not ref.has_edge(a, b):
                ref.add_edge(a, b, cell_types=set(), inter_cell_types=set(), edge_type=None)
            ref[a][b]["cell_types"].add(ct)
            ref[a][b]["edge_type"] = "both" if ref[a][b]["edge_type"] == "inter" else "ppi"
            ppi_only.add_edge(a, b)

    for m in manifest:
        edges_df = pd.read_csv(wdir / m["file"], sep="\t", header=None, names=["g1", "g2"], dtype=str)
        pair_label = (m["source"], m["target"])
        for a, b in zip(edges_df["g1"], edges_df["g2"]):
            if a == b:
                continue
            if not ref.has_edge(a, b):
                ref.add_edge(a, b, cell_types=set(), inter_cell_types=set(), edge_type=None)
            ref[a][b]["inter_cell_types"].add(pair_label)
            ref[a][b]["edge_type"] = "both" if ref[a][b]["edge_type"] == "ppi" else "inter"

    with open(wdir / "gene_reference_graph.pkl", "wb") as f:
        pickle.dump(ref, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(wdir / "gene_ppi_reference_graph.pkl", "wb") as f:
        pickle.dump(ppi_only, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Reference graph : {ref.number_of_nodes():,} nodes, {ref.number_of_edges():,} edges")
    print(f"  PPI-only graph  : {ppi_only.number_of_nodes():,} nodes, {ppi_only.number_of_edges():,} edges")

    print("\n" + "=" * 60)
    print("Step 3 complete — ready to run MultiXrank")
    print("=" * 60)


if __name__ == "__main__":
    main()