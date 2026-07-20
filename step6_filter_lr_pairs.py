#!/usr/bin/env python3
"""
Step 6: Integrate Special Layers (Comprehensive PPI, LR, Reactome)
and Auto-Filter Seed Genes.
"""
import argparse
import pickle
import json
import shutil
import pandas as pd
import yaml
from pathlib import Path
import networkx as nx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppi-pkl", required=True)
    parser.add_argument("--integrated-multiplex", required=True)
    parser.add_argument("--reactome-csv", required=True)
    parser.add_argument("--wdir", required=True)
    parser.add_argument("--validate-against-ppi", action="store_true",
                        help="Only keep ligand-receptor and Reactome edges whose gene pair "
                             "is ALSO an edge in the comprehensive PPI network (--ppi-pkl) — "
                             "your highest-confidence source of real protein-protein interaction.")
    args = parser.parse_args()

    wdir = Path(args.wdir)
    mul_dir = wdir / "multiplex" / "gene_multiplex"
    mul_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 6: Extracting & Merging Special Layers")
    print("=" * 60 + "\n")

    # ---------------------------------------------------------
    # 1. Process Comprehensive PPI
    # ---------------------------------------------------------
    print(f"Loading PPI network from: {args.ppi_pkl}")
    with open(args.ppi_pkl, "rb") as f:
        G_ppi = pickle.load(f)
    
    ppi_edges = set()
    for u, v in G_ppi.edges():
        gu = str(u).strip().upper() if u else None
        gv = str(v).strip().upper() if v else None
        if gu and gv and gu != gv:
            ppi_edges.add(tuple(sorted([gu, gv])))

    ppi_path = mul_dir / "ppi_layer.tsv"
    pd.DataFrame(sorted(ppi_edges), columns=["g1", "g2"]).to_csv(ppi_path, sep="\t", index=False, header=False)
    print(f"✓ Wrote Comprehensive PPI Layer ({len(ppi_edges):,} edges)")

    # Establish the unified gene universe to filter downstream pathways
    gene_universe = set()
    for a, b in ppi_edges:
        gene_universe.add(a)
        gene_universe.add(b)

    # ---------------------------------------------------------
    # 2. Extract LR Layers
    # ---------------------------------------------------------
    print(f"\nLoading integrated multiplex for LR: {args.integrated_multiplex}")
    with open(args.integrated_multiplex, "rb") as f:
        G_int = pickle.load(f)

    lr_edges = set()
    lr_edges_dropped_no_ppi = 0
    for u, v, d in G_int.edges(data=True):
        if d.get("edge_type") == "ligand_receptor":
            du = G_int.nodes[u]
            dv = G_int.nodes[v]
            gu = str(du.get("gene", "")).strip().upper()
            gv = str(dv.get("gene", "")).strip().upper()
            if gu and gv and gu != gv:
                pair = tuple(sorted([gu, gv]))
                if args.validate_against_ppi and pair not in ppi_edges:
                    lr_edges_dropped_no_ppi += 1
                    continue
                lr_edges.add(pair)

    lr_path = mul_dir / "ligand_receptor_layer.tsv"
    pd.DataFrame(sorted(lr_edges), columns=["g1", "g2"]).to_csv(lr_path, sep="\t", index=False, header=False)
    print(f"✓ Wrote Ligand-Receptor Layer ({len(lr_edges):,} edges)")
    if args.validate_against_ppi:
        print(f"  (dropped {lr_edges_dropped_no_ppi:,} LR edges not corroborated by PPI network)")

    # ---------------------------------------------------------
    # 3. Process Reactome Layer
    # ---------------------------------------------------------
    print(f"\nLoading Reactome pairs: {args.reactome_csv}")
    df_react = pd.read_csv(args.reactome_csv)
    react_edges = set()
    react_edges_dropped_no_ppi = 0
    for _, row in df_react.iterrows():
        g1 = str(row.get("symbol1", "")).strip().upper()
        g2 = str(row.get("symbol2", "")).strip().upper()
        if g1 and g2 and g1 != g2 and g1 != "NAN" and g2 != "NAN":
            if g1 in gene_universe and g2 in gene_universe:
                pair = tuple(sorted([g1, g2]))
                if args.validate_against_ppi and pair not in ppi_edges:
                    react_edges_dropped_no_ppi += 1
                    continue
                react_edges.add(pair)

    react_path = mul_dir / "reactome_layer.tsv"
    pd.DataFrame(sorted(react_edges), columns=["g1", "g2"]).to_csv(react_path, sep="\t", index=False, header=False)
    print(f"✓ Wrote Reactome Layer ({len(react_edges):,} edges)")
    if args.validate_against_ppi:
        print(f"  (dropped {react_edges_dropped_no_ppi:,} Reactome edges not corroborated by PPI network)")

    # ---------------------------------------------------------
    # 4. Update configuration
    # ---------------------------------------------------------
    config_path = wdir / "config.yml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    layers = cfg["multiplex"]["gene_multiplex"]["layers"]
    added_layers = [
        "multiplex/gene_multiplex/ppi_layer.tsv",
        "multiplex/gene_multiplex/ligand_receptor_layer.tsv",
        "multiplex/gene_multiplex/reactome_layer.tsv"
    ]
    for layer in added_layers:
        if layer not in layers:
            layers.append(layer)

    n_layers = len(layers)
    tau_value = round(1.0 / n_layers, 6)
    cfg["multiplex"]["gene_multiplex"]["layers"] = sorted(layers)
    cfg["multiplex"]["gene_multiplex"]["graph_type"] = ["00"] * n_layers
    cfg["multiplex"]["gene_multiplex"]["tau"] = [tau_value] * n_layers

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\n✓ Updated config.yml. Total layers: {n_layers} (New Tau: {tau_value})")

    # Update layer info
    layer_info_path = wdir / "layer_info.json"
    with open(layer_info_path) as f:
        layer_info = json.load(f)

    layer_info["ppi_layer"] = {"type": "special", "cell_types": [], "edge_type": "ppi"}
    layer_info["ligand_receptor_layer"] = {"type": "special", "cell_types": [], "edge_type": "ligand_receptor"}
    layer_info["reactome_layer"] = {"type": "special", "cell_types": [], "edge_type": "reactome"}

    with open(layer_info_path, "w") as f:
        json.dump(layer_info, f, indent=2)

    # ---------------------------------------------------------
    # 5. Automatically Filter Seeds by Universe
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("Executing Seed-Gene Universe Scan...")
    print("=" * 40)

    seeds_file = wdir / "seeds" / "seeds.txt"
    if not seeds_file.exists():
        print(f"⚠ Seed file not found at: {seeds_file}")
        return

    # Read active seeds
    seeds = set()
    with open(seeds_file) as f:
        for line in f:
            if line.strip():
                seeds.add(line.strip().upper())

    # Compile the final active gene universe across ALL layers
    active_universe = set()
    for layer_rel in layers:
        layer_path = wdir / layer_rel
        if layer_path.exists():
            df = pd.read_csv(layer_path, sep="\t", header=None, dtype=str)
            active_universe.update(df[0].dropna().str.strip().str.upper())
            active_universe.update(df[1].dropna().str.strip().str.upper())

    valid_seeds = seeds & active_universe
    missing_seeds = seeds - active_universe

    print(f"Total Seeds Loaded:        {len(seeds)}")
    print(f"Seeds Present in Network:  {len(valid_seeds)} ({100*len(valid_seeds)/max(len(seeds),1):.1f}%)")
    print(f"Seeds Missing (REMOVED):   {len(missing_seeds)}")

    # Backup original seeds
    shutil.copy(seeds_file, wdir / "seeds" / "seeds_original.txt")

    # Save cleaned seeds file
    with open(seeds_file, "w") as f:
        for seed in sorted(valid_seeds):
            f.write(f"{seed}\n")
    print(f"✓ Overwrote active clean seeds file: {seeds_file}")

    if missing_seeds:
        print(f"  First 10 excluded seeds: {sorted(list(missing_seeds))[:10]}...")

    print("\n" + "=" * 60)
    print("Special Layers & Seed Filtering completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()